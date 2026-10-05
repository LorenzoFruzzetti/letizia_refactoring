"""ROI connectivity of three PV sessions, each compared with the same animal's day 1.

Target sessions: 260716 PV3, 260716 PV4, 260805 PV5. The baseline of each is the
session with day_index 1 for that animal in `pixel_data/experimental_design.csv`
(260520 PV3, 260520 PV4, 260608 PV5).

Input is the same as test_nbs_pv_group.py: every recording's `R_roi` (22x22 Pearson r
of box-mean traces, roi_pixel_connectivity.py), Fisher-z transformed. All four
connectivity variants are analysed side by side, because 9.33 showed that most of the
raw-dF/F change is global signal / drift:

    pixels_dff_full                 mean-baseline dF/F (the original NBS input)
    pixels_dff_full_gsr             + global signal regression
    pixels_median_dff_20s_full      20 s running-median baseline (drift removed)
    pixels_median_dff_20s_full_gsr  both

Statistics are WITHIN ONE ANIMAL with the 5 recordings of a session as replicates.
The recordings of a session are not independent samples of the animal-day, so the
tests describe how far the change exceeds recording-to-recording variability; they are
not population inference. The exact 5-vs-5 permutation test has 252 splits, smallest
two-sided p = 2/252 = 0.008.

Outputs (outputs/pv_session_vs_day1/):
    metrics_per_recording.csv   summary connectivity per recording (edge-set mean z)
    metric_tests.csv            day 1 vs target per animal, metric and variant
    edge_changes.csv            every edge: mean z day 1, target, difference, Welch t
    edge_consensus.csv          per edge: mean difference over animals, sign agreement
    pattern_similarity.csv      edge-pattern correlation within / between sessions
    nbs_component_changes.csv   change of the PV_slope NBS components in these sessions
    matrices_<variant>.png      per animal: day 1, target, difference, Welch t
    summary_metrics.png         per-recording summary metrics, all variants
    consensus_matrices.png      mean difference over the three animals, agreement marked
    pattern_similarity.png      within- vs between-session edge-pattern correlation
    seed_maps_<variant>.png     target - day 1 pixel seed maps for selected seeds

Run: conda run --no-capture-output -n letizia python test_pv_session_vs_day1.py
"""

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbs import fisher_z

# ---------------------------------------------------------------- parameters
repo_root = Path(__file__).resolve().parent
design_path = repo_root / "pixel_data" / "experimental_design.csv"
connectivity_root = repo_root / "outputs" / "roi_pixel_connectivity"
connectivity_file = "roi_pixel_connectivity.npz"   # per recording: R_roi, r_maps
nbs_root = repo_root / "outputs"                    # nbs_pv_group[_<variant>]/components.csv
output_dir = repo_root / "outputs" / "pv_session_vs_day1"

target_sessions = ["260716_PV3", "260716_PV4", "260805_PV5"]  # <day>_<animal>
baseline_day_index = 1                                        # compared against
variants = [
    "pixels_dff_full",
    "pixels_dff_full_gsr",
    "pixels_median_dff_20s_full",
    "pixels_median_dff_20s_full_gsr",
]
edge_t_threshold = 2.5      # |Welch t| counted as a per-animal edge change
seed_map_variants = ["pixels_dff_full", "pixels_dff_full_gsr"]
seed_map_seeds = ["M2L_alta", "M1L_alta", "HLL", "RSL_alta", "V1L"]
r_clip = 0.999999           # pixel r_maps are clipped before arctanh
matrix_z_limit = 1.5        # colour limit of the mean-z matrices
difference_z_limit = 0.6    # colour limit of the difference matrices
seed_difference_limit = 0.6

# Region family of each ROI, from the label prefix (side letter removed).
network_of_prefix = {"M2": "motor", "M1": "motor", "BFD": "somatosensory",
                     "Tr": "somatosensory", "FL": "somatosensory", "HL": "somatosensory",
                     "RS": "posterior", "V1a": "posterior", "V1": "posterior"}

output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loading
# Pair each target session with the same animal's day-1 session.
design = pd.read_csv(design_path, dtype={"day": str})
design["session"] = design["day"] + "_" + design["animal"]
session_pairs = []                                     # (animal, baseline, target)
for target in target_sessions:
    animal = target.split("_")[1]
    baseline = design.loc[(design["animal"] == animal)
                          & (design["day_index"] == baseline_day_index), "session"].unique()
    if len(baseline) != 1:
        raise ValueError(f"{animal}: expected one day_index {baseline_day_index} session, got {baseline}")
    session_pairs.append((animal, baseline[0], target))
target_day_index = {t: int(design.loc[design["session"] == t, "day_index"].iloc[0])
                    for t in target_sessions}
for animal, baseline, target in session_pairs:
    print(f"{animal}: day 1 = {baseline}, target = {target} (day_index {target_day_index[target]})")

all_sessions = [s for _, b, t in session_pairs for s in (b, t)]
recordings_of = {s: design.loc[design["session"] == s, "recording"].sort_values().tolist()
                 for s in all_sessions}

# session_z[variant][session]: (n_recordings, n_rois, n_rois) Fisher z per recording.
# seed_z[variant][session]:   (n_recordings, n_rois, n_rows, n_cols) pixel seed maps.
session_z, seed_z = {}, {}
roi_labels = None
for variant in variants:
    session_z[variant], seed_z[variant] = {}, {}
    for session in all_sessions:
        matrices, maps = [], []
        for recording in recordings_of[session]:
            data = np.load(connectivity_root / variant / session / recording / connectivity_file)
            labels = list(data["roi_labels"])
            if roi_labels is None:
                roi_labels = labels
                bregma_crop = data["bregma_crop_zero_based"]
            if labels != roi_labels:
                raise ValueError(f"ROI labels differ in {variant}/{session}/{recording}")
            matrices.append(fisher_z(data["R_roi"]))
            if variant in seed_map_variants:
                maps.append(np.arctanh(np.clip(data["r_maps"].astype(np.float64), -r_clip, r_clip)))
        session_z[variant][session] = np.stack(matrices)
        if maps:
            seed_z[variant][session] = np.stack(maps)
n_rois = len(roi_labels)
print(f"loaded {len(all_sessions)} sessions x {len(variants)} variants, {n_rois} ROIs")

# ---------------------------------------------------------------- edge sets
# Side and family of each ROI. The first 11 labels are left, the last 11 their right
# twins in the same order; check that rather than assume it.
half = n_rois // 2
for i_roi in range(half):
    left, right = roi_labels[i_roi], roi_labels[i_roi + half]
    if left.replace("L", "R", 1) != right and left[::-1].replace("L", "R", 1)[::-1] != right:
        raise ValueError(f"{left} and {right} are not homotopic twins")
side = np.array(["L"] * half + ["R"] * half)
prefix = [label.split("_")[0][:-1] for label in roi_labels]   # "M2L_alta" -> "M2"
network = np.array([network_of_prefix[p] for p in prefix])

upper = np.triu(np.ones((n_rois, n_rois), dtype=bool), k=1)
row_side, col_side = side[:, None], side[None, :]
homotopic = np.zeros((n_rois, n_rois), dtype=bool)
for i_roi in range(half):
    homotopic[i_roi, i_roi + half] = True
edge_sets = {
    "all": upper,
    "intra_left": upper & (row_side == "L") & (col_side == "L"),
    "intra_right": upper & (row_side == "R") & (col_side == "R"),
    "homotopic": homotopic,
    "heterotopic": upper & (row_side != col_side) & ~homotopic,
}
network_names = ["motor", "somatosensory", "posterior"]
for i_net, first in enumerate(network_names):
    for second in network_names[i_net:]:
        pair = ((network[:, None] == first) & (network[None, :] == second)) | \
               ((network[:, None] == second) & (network[None, :] == first))
        edge_sets[f"{first}-{second}"] = upper & pair
edge_rows, edge_cols = np.nonzero(upper)                  # 231 unique edges
edge_names = [f"{roi_labels[r]}-{roi_labels[c]}" for r, c in zip(edge_rows, edge_cols)]


def permutation_p(first: np.ndarray, second: np.ndarray) -> float:
    """Exact two-sided p of the difference of means over every relabelling."""
    pooled = np.concatenate([first, second])
    observed = abs(second.mean() - first.mean())
    differences = []
    for chosen in itertools.combinations(range(len(pooled)), len(second)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(chosen)] = True
        differences.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
    return float(np.mean(np.array(differences) >= observed - 1e-12))


def welch_t(first: np.ndarray, second: np.ndarray, axis: int = 0) -> np.ndarray:
    """Welch t of second - first along `axis`."""
    standard_error = np.sqrt(first.var(axis=axis, ddof=1) / first.shape[axis]
                             + second.var(axis=axis, ddof=1) / second.shape[axis])
    return (second.mean(axis=axis) - first.mean(axis=axis)) / standard_error


# ---------------------------------------------------------------- summary metrics
# Mean z over each edge set, one value per recording.
metric_rows = []
for variant in variants:
    for animal, baseline, target in session_pairs:
        for role, session in (("day1", baseline), ("target", target)):
            for i_rec, recording in enumerate(recordings_of[session]):
                z = session_z[variant][session][i_rec]
                row = {"variant": variant, "animal": animal, "role": role,
                       "session": session, "recording": recording}
                row.update({name: z[mask].mean() for name, mask in edge_sets.items()})
                metric_rows.append(row)
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(output_dir / "metrics_per_recording.csv", index=False)

# Day 1 vs target for every metric: difference of means, Welch t, exact permutation p.
test_rows = []
for (variant, animal), group in metrics.groupby(["variant", "animal"], sort=False):
    for name in edge_sets:
        first = group.loc[group["role"] == "day1", name].to_numpy()
        second = group.loc[group["role"] == "target", name].to_numpy()
        test_rows.append({"variant": variant, "animal": animal, "metric": name,
                          "z_day1": first.mean(), "z_target": second.mean(),
                          "difference": second.mean() - first.mean(),
                          "welch_t": float(welch_t(first, second)),
                          "permutation_p": permutation_p(first, second)})
metric_tests = pd.DataFrame(test_rows)
metric_tests.to_csv(output_dir / "metric_tests.csv", index=False)

# ---------------------------------------------------------------- per-edge change
# For every edge and animal: session means, difference, Welch t over recordings.
edge_rows_out = []
difference_matrix, t_matrix = {}, {}                     # [variant][animal]: (n_rois, n_rois)
for variant in variants:
    difference_matrix[variant], t_matrix[variant] = {}, {}
    for animal, baseline, target in session_pairs:
        first = session_z[variant][baseline]               # (n_recordings, n_rois, n_rois)
        second = session_z[variant][target]
        difference = second.mean(axis=0) - first.mean(axis=0)
        t_values = welch_t(first, second)
        np.fill_diagonal(t_values, 0.0)
        difference_matrix[variant][animal], t_matrix[variant][animal] = difference, t_values
        for name, r, c in zip(edge_names, edge_rows, edge_cols):
            edge_rows_out.append({"variant": variant, "animal": animal, "edge": name,
                                  "z_day1": first[:, r, c].mean(), "z_target": second[:, r, c].mean(),
                                  "difference": difference[r, c], "welch_t": t_values[r, c]})
edge_changes = pd.DataFrame(edge_rows_out)
edge_changes.to_csv(output_dir / "edge_changes.csv", index=False)

# Across the three animals: mean difference, and how many animals change the edge in
# the same direction with |t| above the threshold.
animals = [animal for animal, _, _ in session_pairs]
consensus_rows = []
consensus_mask = {}                                      # [variant]: edges all animals agree on
for variant in variants:
    differences = np.stack([difference_matrix[variant][a] for a in animals])   # (n_animals, n_rois, n_rois)
    t_values = np.stack([t_matrix[variant][a] for a in animals])
    n_up = (t_values > edge_t_threshold).sum(axis=0)
    n_down = (t_values < -edge_t_threshold).sum(axis=0)
    consensus_mask[variant] = (n_up == len(animals)) | (n_down == len(animals))
    for name, r, c in zip(edge_names, edge_rows, edge_cols):
        consensus_rows.append({"variant": variant, "edge": name,
                               "mean_difference": differences[:, r, c].mean(),
                               "n_animals_up": int(n_up[r, c]), "n_animals_down": int(n_down[r, c]),
                               "same_sign_all": bool(np.all(differences[:, r, c] > 0)
                                                     or np.all(differences[:, r, c] < 0))})
edge_consensus = pd.DataFrame(consensus_rows)
edge_consensus.to_csv(output_dir / "edge_consensus.csv", index=False)

# ---------------------------------------------------------------- pattern similarity
# Pearson r between the 231-edge z vectors of two recordings. It ignores a uniform
# shift of every edge, so it measures whether the PATTERN changed, not the level.
similarity_rows = []
for variant in variants:
    for animal, baseline, target in session_pairs:
        first = session_z[variant][baseline][:, edge_rows, edge_cols]   # (n_recordings, n_edges)
        second = session_z[variant][target][:, edge_rows, edge_cols]
        pairs = {
            "within_day1": [np.corrcoef(first[a], first[b])[0, 1]
                            for a, b in itertools.combinations(range(len(first)), 2)],
            "within_target": [np.corrcoef(second[a], second[b])[0, 1]
                              for a, b in itertools.combinations(range(len(second)), 2)],
            "between": [np.corrcoef(x, y)[0, 1] for x in first for y in second],
        }
        for comparison, values in pairs.items():
            similarity_rows.append({"variant": variant, "animal": animal, "comparison": comparison,
                                    "mean_r": np.mean(values), "sd_r": np.std(values, ddof=1),
                                    "n_pairs": len(values)})
pattern_similarity = pd.DataFrame(similarity_rows)
pattern_similarity.to_csv(output_dir / "pattern_similarity.csv", index=False)

# ---------------------------------------------------------------- NBS components
# How the significant PV_slope components move between day 1 and these sessions.
# Circular by design: these sessions are part of the data that produced them.
label_index = {label: i for i, label in enumerate(roi_labels)}
nbs_rows = []
for variant in variants:
    suffix = "" if variant == "pixels_dff_full" else f"_{variant}"
    components = pd.read_csv(nbs_root / f"nbs_pv_group{suffix}" / "components.csv")
    selected = components[(components["analysis"] == "PV_slope") & components["significant"]]
    for _, component in selected.iterrows():
        mask = np.zeros((n_rois, n_rois), dtype=bool)
        for edge in component["edges"].split():
            first_label, second_label = edge.split("-")
            mask[label_index[first_label], label_index[second_label]] = True
        mask = mask | mask.T
        mask &= upper
        for animal, baseline, target in session_pairs:
            first = session_z[variant][baseline][:, mask].mean(axis=1)   # (n_recordings,)
            second = session_z[variant][target][:, mask].mean(axis=1)
            nbs_rows.append({"variant": variant, "direction": component["direction"],
                             "threshold": component["primary_threshold"], "rank": component["rank"],
                             "n_edges": int(mask.sum()), "nbs_p": component["p_value"],
                             "animal": animal, "z_day1": first.mean(), "z_target": second.mean(),
                             "difference": second.mean() - first.mean(),
                             "welch_t": float(welch_t(first, second)),
                             "permutation_p": permutation_p(first, second)})
nbs_component_changes = pd.DataFrame(nbs_rows)
nbs_component_changes.to_csv(output_dir / "nbs_component_changes.csv", index=False)

# ---------------------------------------------------------------- console summary
pd.set_option("display.width", 200)
print("\nmean z over ALL edges (target - day 1):")
print(metric_tests[metric_tests["metric"] == "all"]
      .pivot(index="animal", columns="variant", values="difference").round(3))
print("\npermutation p (ALL edges):")
print(metric_tests[metric_tests["metric"] == "all"]
      .pivot(index="animal", columns="variant", values="permutation_p").round(3))
print("\nedges with |t| > threshold in all three animals, same direction:")
for variant in variants:
    agreed = edge_consensus[(edge_consensus["variant"] == variant)
                            & ((edge_consensus["n_animals_up"] == 3) | (edge_consensus["n_animals_down"] == 3))]
    print(f"  {variant}: {len(agreed)} -> " + ", ".join(
        f"{e} ({'+' if d > 0 else '-'}{abs(d):.2f})" for e, d in zip(agreed["edge"], agreed["mean_difference"])))
print("\npattern similarity (mean r):")
print(pattern_similarity.pivot_table(index=["variant", "animal"], columns="comparison",
                                     values="mean_r").round(3))

# ---------------------------------------------------------------- plotting: matrices
# One figure per variant: rows = animals; day 1, target, difference, Welch t.
for variant in variants:
    fig, axes = plt.subplots(len(session_pairs), 4, figsize=(17, 4.2 * len(session_pairs)),
                             constrained_layout=True, squeeze=False)
    for i_animal, (animal, baseline, target) in enumerate(session_pairs):
        panels = [
            (session_z[variant][baseline].mean(axis=0), f"{baseline} (day 1)", "viridis",
             0, matrix_z_limit, "mean z"),
            (session_z[variant][target].mean(axis=0), f"{target} (day {target_day_index[target]})",
             "viridis", 0, matrix_z_limit, "mean z"),
            (difference_matrix[variant][animal], "target - day 1", "RdBu_r",
             -difference_z_limit, difference_z_limit, "difference in z"),
            (t_matrix[variant][animal], f"Welch t (|t| > {edge_t_threshold:g} dotted)", "RdBu_r",
             -8, 8, "t"),
        ]
        for i_col, (matrix, title, cmap, low, high, unit) in enumerate(panels):
            ax = axes[i_animal, i_col]
            shown = matrix.copy()
            np.fill_diagonal(shown, np.nan)
            image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)
            if i_col == 3:
                marked_r, marked_c = np.nonzero(np.abs(matrix) > edge_t_threshold)
                ax.plot(marked_c, marked_r, "k.", markersize=2)
            ax.set_title(f"{animal}: {title}", fontsize=9)
            ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(n_rois), roi_labels, fontsize=5)
            ax.axhline(half - 0.5, color="w", linewidth=0.6)
            ax.axvline(half - 0.5, color="w", linewidth=0.6)
            fig.colorbar(image, ax=ax, shrink=0.75, label=unit)
    fig.suptitle(f"ROI connectivity, target session vs day 1 ({variant}, 5 recordings each)")
    fig.savefig(output_dir / f"matrices_{variant}.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- plotting: summary metrics
# Rows = variants, columns = edge sets; each dot is a recording, bar = session mean.
summary_metrics = ["all", "intra_left", "intra_right", "homotopic", "heterotopic",
                   "motor-posterior", "somatosensory-posterior"]
fig, axes = plt.subplots(len(variants), len(summary_metrics),
                         figsize=(2.6 * len(summary_metrics), 2.6 * len(variants)),
                         constrained_layout=True, squeeze=False)
role_colours = {"day1": "#7a8aa0", "target": "#d0622f"}
for i_var, variant in enumerate(variants):
    for i_met, name in enumerate(summary_metrics):
        ax = axes[i_var, i_met]
        for i_animal, animal in enumerate(animals):
            for i_role, role in enumerate(["day1", "target"]):
                values = metrics.loc[(metrics["variant"] == variant) & (metrics["animal"] == animal)
                                     & (metrics["role"] == role), name].to_numpy()
                x_position = 3 * i_animal + i_role
                ax.bar(x_position, values.mean(), color=role_colours[role], alpha=0.35, width=0.8)
                ax.plot(np.full(len(values), x_position), values, "o",
                        color=role_colours[role], markersize=3)
            p_value = metric_tests.loc[(metric_tests["variant"] == variant) & (metric_tests["animal"] == animal)
                                       & (metric_tests["metric"] == name), "permutation_p"].iloc[0]
            ax.text(3 * i_animal + 0.5, 1.0, f"p={p_value:.3f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6)
        ax.set_xticks([3 * i + 0.5 for i in range(len(animals))], animals, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        if i_var == 0:
            ax.set_title(name, fontsize=8)
        if i_met == 0:
            ax.set_ylabel(f"{variant}\nmean z", fontsize=7)
fig.suptitle("Connectivity per recording: day 1 (grey) vs target session (orange); "
             "p = exact 5-vs-5 permutation")
fig.savefig(output_dir / "summary_metrics.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- plotting: consensus
# Mean difference over the three animals; dots = all three |t| > threshold, same sign.
fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 4.6),
                         constrained_layout=True, squeeze=False)
for i_var, variant in enumerate(variants):
    ax = axes[0, i_var]
    mean_difference = np.mean([difference_matrix[variant][a] for a in animals], axis=0)
    np.fill_diagonal(mean_difference, np.nan)
    image = ax.imshow(mean_difference, cmap="RdBu_r", vmin=-difference_z_limit, vmax=difference_z_limit)
    marked_r, marked_c = np.nonzero(consensus_mask[variant])
    ax.plot(marked_c, marked_r, "k.", markersize=3)
    ax.set_title(f"{variant}\n{int(consensus_mask[variant][upper].sum())} edges agreed", fontsize=8)
    ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n_rois), roi_labels, fontsize=5)
    ax.axhline(half - 0.5, color="k", linewidth=0.5)
    ax.axvline(half - 0.5, color="k", linewidth=0.5)
    fig.colorbar(image, ax=ax, shrink=0.75, label="mean difference in z")
fig.suptitle(f"Target - day 1, mean over {', '.join(animals)}; "
             f"dot = |Welch t| > {edge_t_threshold:g} in all three, same direction")
fig.savefig(output_dir / "consensus_matrices.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- plotting: similarity
fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 3.4),
                         constrained_layout=True, squeeze=False, sharey=True)
comparison_colours = {"within_day1": "#7a8aa0", "within_target": "#d0622f", "between": "#4a4a4a"}
for i_var, variant in enumerate(variants):
    ax = axes[0, i_var]
    subset = pattern_similarity[pattern_similarity["variant"] == variant]
    for i_animal, animal in enumerate(animals):
        for i_cmp, comparison in enumerate(comparison_colours):
            row = subset[(subset["animal"] == animal) & (subset["comparison"] == comparison)].iloc[0]
            ax.bar(4 * i_animal + i_cmp, row["mean_r"], yerr=row["sd_r"],
                   color=comparison_colours[comparison], label=comparison if i_animal == 0 else None)
    ax.set_xticks([4 * i + 1 for i in range(len(animals))], animals, fontsize=7)
    ax.set_title(variant, fontsize=8)
    ax.set_ylim(0, 1)
axes[0, 0].set_ylabel("edge-pattern r between recordings")
axes[0, 0].legend(fontsize=6, loc="lower left")
fig.suptitle("Is the connectivity PATTERN different from day 1? (between < within = yes)")
fig.savefig(output_dir / "pattern_similarity.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- plotting: seed maps
# Target - day 1 pixel seed maps (mean z over recordings). The crop is Bregma-relative
# (CLAUDE.md 9.15), so the same pixel is the same anatomy in every session.
for variant in seed_map_variants:
    fig, axes = plt.subplots(len(session_pairs), len(seed_map_seeds),
                             figsize=(2.9 * len(seed_map_seeds), 2.7 * len(session_pairs)),
                             constrained_layout=True, squeeze=False)
    for i_animal, (animal, baseline, target) in enumerate(session_pairs):
        difference_maps = seed_z[variant][target].mean(axis=0) - seed_z[variant][baseline].mean(axis=0)
        for i_seed, seed in enumerate(seed_map_seeds):
            ax = axes[i_animal, i_seed]
            image = ax.imshow(difference_maps[label_index[seed]], cmap="RdBu_r",
                              vmin=-seed_difference_limit, vmax=seed_difference_limit)
            ax.plot(bregma_crop[1], bregma_crop[0], "k+", markersize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{animal}  seed {seed}", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.6, label="target - day 1, z")
    fig.suptitle(f"Seed maps, target - day 1 ({variant}; + = Bregma)")
    fig.savefig(output_dir / f"seed_maps_{variant}.png", dpi=200)
    plt.close(fig)

print(f"\nwrote {output_dir}")
