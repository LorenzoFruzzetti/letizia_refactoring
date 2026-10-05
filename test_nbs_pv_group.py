"""NBS on the 22x22 ROI connectivity of the PV group: change over days, and PV vs R/T.

Same data and session averaging as test_nbs_change_over_time.py: every recording's
`R_roi` (roi_pixel_connectivity.py) is Fisher-z transformed and averaged over the
t# of its session (one animal on one day). Factors come from
`pixel_data/experimental_design.csv`.

All six PV-group animals (PV3-PV8) are PV-CRE, so the mouse line cannot be compared
inside the group. The line is held FIXED instead: only PV-CRE animals are loaded, and
PV is compared with the PV-CRE animals of groups R (R6, R7) and T (T9-T14). Any
difference is therefore a group difference, not a line difference.

Analyses, each run in both directions and reported at three primary thresholds:

    PV_slope    PV sessions      z_edge ~ animal + day_index      contrast: day_index
    PV_vs_R     one obs/animal   z ~ PV + R                       contrast: PV - R
    PV_vs_T     one obs/animal   z ~ PV + T                       contrast: PV - T
    PV_vs_RT    one obs/animal   z ~ PV + (R and T together)      contrast: PV - RT

PV_slope: the animal dummies make the slope WITHIN-animal, sessions are permuted only
inside an animal (Freedman-Lane, Monte Carlo). PV3-PV5 have day_index 1-4, PV6-PV8
only 1-2. "increase" = connectivity rises with day_index.

PV_vs_*: the animal is the unit. Each animal contributes the mean of its sessions with
day_index in `between_day_indices` (default 1-2, the only days every PV animal has),
so every animal is compared over the same days. Group labels are permuted across
animals and enumerated exactly. PV_vs_R has 6 vs 2 animals = 28 relabellings, so its
smallest attainable p is 1/28 = 0.036. "increase" = higher in PV.

The recordings in `exclude_recordings` are dropped before averaging (260828_PV7_t2:
one-frame reflectance glitch plus a persistent GCaMP step, CLAUDE.md 9.23/9.27).

Outputs (outputs/nbs_pv_group/):
    components.csv            every component: analysis, direction, threshold, size, p
    sessions.csv              one row per session
    animal_means.csv          the per-animal means used by the PV_vs_* tests
    edge_t_<analysis>.csv     22x22 observed t map (positive = rises with day / higher in PV)
    nbs_<analysis>.png        significant components, rows = direction, cols = threshold
    edge_t_heatmaps.png       observed t of every edge per analysis, significant edges boxed
    null_distributions.png    permutation null of the max component size vs the observed
    group_mean_matrices.png   PV, R and T mean z (days in between_day_indices) and PV - R, PV - T
    pv_by_day.png             per PV animal: mean z vs day_index (all edges, and each
                              significant PV_slope component)

Run: conda run --no-capture-output -n letizia python test_nbs_pv_group.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbs import fisher_z, nbs, rethreshold
from wfci.significance import circular_layout, mask_by_adjacency, network_figure

# ---------------------------------------------------------------- parameters
repo_root = Path(__file__).resolve().parent
design_path = repo_root / "pixel_data" / "experimental_design.csv"
# Which roi_pixel_connectivity.py output folder to read: "pixels_dff_full" (mean baseline),
# "pixels_dff_full_gsr" (+ global signal regression), "pixels_median_dff_20s_full"
# (running-median baseline, no drift), "pixels_median_dff_20s_full_gsr" (both).
# The NBS_CONNECTIVITY_VARIANT environment variable overrides it for batch reruns.
connectivity_variant = os.environ.get("NBS_CONNECTIVITY_VARIANT", "pixels_dff_full")
connectivity_root = repo_root / "outputs" / "roi_pixel_connectivity" / connectivity_variant
# The original variant keeps its original output folder; any other gets a suffix.
variant_suffix = "" if connectivity_variant == "pixels_dff_full" else f"_{connectivity_variant}"
connectivity_file = "roi_pixel_connectivity.npz"   # per recording, holds R_roi (22, 22)
mouse_line = "PV-CRE"                  # line held fixed; every PV animal is PV-CRE
focus_group = "PV"
comparison_groups = ["R", "T"]         # same-line groups PV is compared with
exclude_recordings = ["260828_PV7_t2"]  # glitch frame + GCaMP step (CLAUDE.md 9.23/9.27)
between_day_indices = [1, 2]           # sessions averaged per animal for PV_vs_*
primary_thresholds = [2.5, 3.1, 3.5]   # t cut-offs, reported together
n_permutations = 5000                  # upper bound; exact enumeration when fewer relabellings exist
alpha = 0.05
size_measure = "extent"                # component size = number of edges
seed = 0
heatmap_t_limit = 4.0                  # shared colour scale of the t heatmaps
difference_z_limit = 0.4               # colour scale of the group difference maps
output_dir = repo_root / "outputs" / f"nbs_pv_group{variant_suffix}"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loading
design = pd.read_csv(design_path)
design = design[
    (design["mouse_line"] == mouse_line)
    & design["group"].isin([focus_group, *comparison_groups])
    & ~design["recording_id"].isin(exclude_recordings)
].reset_index(drop=True)

# Fisher z of every recording's ROI x ROI matrix.
recording_z = []
roi_labels = None
for row in design.itertuples():
    data = np.load(connectivity_root / row.pixel_dir / connectivity_file)
    if roi_labels is None:
        roi_labels = [str(label) for label in data["roi_labels"]]
    if [str(label) for label in data["roi_labels"]] != roi_labels:
        raise ValueError(f"{row.recording_id}: ROI labels differ from the first recording")
    recording_z.append(fisher_z(data["R_roi"]))
recording_z = np.stack(recording_z, axis=-1)                 # (n_rois, n_rois, n_recordings)
n_rois = len(roi_labels)

# ---------------------------------------------------------------- sessions
# One observation per animal-day: the Fisher-z mean over that session's t#.
design["session"] = design["day"].astype(str) + "_" + design["animal"]
sessions = (
    design.groupby("session", sort=False)
    .agg(day=("day", "first"), animal=("animal", "first"), group=("group", "first"),
         day_index=("day_index", "first"), n_recordings=("recording", "size"))
    .reset_index()
)
session_z = np.stack(
    [recording_z[:, :, design.index[design["session"] == name]].mean(axis=-1)
     for name in sessions["session"]],
    axis=-1,
)                                                            # (n_rois, n_rois, n_sessions)

print(f"{len(design)} {mouse_line} recordings -> {len(sessions)} sessions, "
      f"{sessions['animal'].nunique()} animals (excluded: {', '.join(exclude_recordings)})")
print(pd.crosstab([sessions["group"], sessions["animal"]], sessions["day_index"]))

# ---------------------------------------------------------------- per-animal means
# Each animal's mean z over its sessions in between_day_indices: the observation of
# the PV_vs_* tests, so every animal counts once and over the same days.
in_window = sessions["day_index"].isin(between_day_indices)
animal_rows = []
animal_z = []
for (group, animal), animal_sessions in sessions[in_window].groupby(["group", "animal"]):
    animal_rows.append(dict(group=group, animal=animal, n_sessions=len(animal_sessions),
                            day_indices=" ".join(str(d) for d in sorted(animal_sessions["day_index"]))))
    animal_z.append(session_z[:, :, animal_sessions.index].mean(axis=-1))
animal_table = pd.DataFrame(animal_rows)
animal_z = np.stack(animal_z, axis=-1)                       # (n_rois, n_rois, n_animals)
animal_table.to_csv(output_dir / "animal_means.csv", index=False)

# ---------------------------------------------------------------- analyses
# Every analysis names its observations (matrices), its exchange blocks (None = free
# permutation), what a positive t means, and the animals it uses.
analyses = {}

# PV_slope: animal dummies (no intercept) + day_index; slope is within-animal.
pv_rows = (sessions["group"] == focus_group).to_numpy()
pv_sessions = sessions[pv_rows]
pv_animals = sorted(pv_sessions["animal"].unique())
animal_dummies = (pv_sessions["animal"].to_numpy()[:, None] == np.array(pv_animals)[None, :]).astype(float)
analyses[f"{focus_group}_slope"] = dict(
    matrices=session_z[:, :, pv_rows],
    design=np.column_stack([animal_dummies, pv_sessions["day_index"].to_numpy(float)]),
    contrast=np.r_[np.zeros(len(pv_animals)), 1.0],
    exchange_blocks=pv_sessions["animal"].to_numpy(), animals=pv_sessions["animal"].to_numpy(),
    positive="rises with day_index", description=f"{focus_group} change over day_index",
)

# PV vs each same-line group, and vs both together: cell-means design, contrast
# PV - other. The nuisance is only the grand mean, so the null is enumerated exactly.
window_text = f"day_index {'/'.join(str(d) for d in between_day_indices)}"
for other_name, other_groups in [*[(g, [g]) for g in comparison_groups],
                                 ("".join(comparison_groups), comparison_groups)]:
    selected = animal_table["group"].isin([focus_group, *other_groups]).to_numpy()
    is_focus = (animal_table.loc[selected, "group"] == focus_group).to_numpy().astype(float)
    analyses[f"{focus_group}_vs_{other_name}"] = dict(
        matrices=animal_z[:, :, selected],
        design=np.column_stack([is_focus, 1 - is_focus]), contrast=np.array([1.0, -1.0]),
        exchange_blocks=None, animals=animal_table.loc[selected, "animal"].to_numpy(),
        positive=f"higher in {focus_group}",
        description=f"{focus_group} vs {'+'.join(other_groups)} ({mouse_line}, {window_text})",
    )

# ---------------------------------------------------------------- NBS
# Each analysis x direction: one permutation run, the other thresholds reuse it.
component_rows = []
results = {}                           # (analysis, direction) -> [NBSResult per threshold]
for name, analysis in analyses.items():
    animals = analysis["animals"]
    for direction, sign in [("increase", 1.0), ("decrease", -1.0)]:
        first = nbs(analysis["matrices"], analysis["design"], sign * analysis["contrast"],
                    primary_threshold=primary_thresholds[0], n_permutations=n_permutations,
                    alpha=alpha, size_measure=size_measure,
                    exchange_blocks=analysis["exchange_blocks"],
                    seed=seed, store_null_stats=True)
        per_threshold = [first] + [rethreshold(first, t) for t in primary_thresholds[1:]]
        results[(name, direction)] = per_threshold

        if direction == "increase":
            pd.DataFrame(first.stat_matrix, index=roi_labels, columns=roi_labels).to_csv(
                output_dir / f"edge_t_{name}.csv"
            )
        for result in per_threshold:
            for rank, component in enumerate(result.components, start=1):
                component_rows.append(dict(
                    analysis=name, direction=direction, primary_threshold=result.primary_threshold,
                    rank=rank, size=component.size, p_value=component.p_value,
                    significant=component.p_value < alpha,
                    n_observations=analysis["matrices"].shape[-1], n_animals=len(np.unique(animals)),
                    exact=result.exact, n_null=result.null_max_size.size,
                    nodes=" ".join(roi_labels[i] for i in component.nodes),
                    edges=" ".join(f"{roi_labels[i]}-{roi_labels[j]}" for i, j in component.edges),
                ))

components = pd.DataFrame(component_rows)
components.to_csv(output_dir / "components.csv", index=False)

# Console summary: the largest component of every run. With an exact null of N
# relabellings the smallest attainable p is 1/N.
print(f"\nsize = {size_measure}; increase = rises with day_index (slope) or higher in {focus_group}")
for (name, direction), per_threshold in results.items():
    for result in per_threshold:
        best = result.components[0] if result.components else None
        text = (f"largest {best.size:g} edges, p = {best.p_value:.4f}" if best else "no edge above threshold")
        flag = "  *" if result.significant else ""
        null_text = f"exact, {result.null_max_size.size} relabellings" if result.exact else \
            f"{result.null_max_size.size} random permutations"
        print(f"  {name:12s} {direction:8s} t > {result.primary_threshold:g}: {text}{flag}  ({null_text})")
for name in analyses:
    first = results[(name, "increase")][0]
    if first.exact:
        print(f"  NOTE {name}: {first.null_max_size.size} relabellings, smallest attainable "
              f"p = {1 / first.null_max_size.size:.3f}")

# ---------------------------------------------------------------- plots
# One figure per analysis: rows = direction, columns = threshold, significant edges only.
node_positions = circular_layout(n_rois)
for name in analyses:
    fig, axes = plt.subplots(2, len(primary_thresholds), figsize=(6 * len(primary_thresholds), 12),
                             constrained_layout=True, squeeze=False)
    for i_row, direction in enumerate(["increase", "decrease"]):
        for i_col, result in enumerate(results[(name, direction)]):
            significant_adjacency = np.zeros((n_rois, n_rois), dtype=bool)
            for component in result.significant:
                significant_adjacency |= component.adjacency
            masked_t = mask_by_adjacency(result.stat_matrix, significant_adjacency)
            ax = axes[i_row, i_col]
            network_figure(masked_t, node_positions=node_positions, labels=roi_labels, ax=ax)
            best_p = min((c.p_value for c in result.components), default=np.nan)
            ax.set_title(f"{name} {direction} ({analyses[name]['positive'] if direction == 'increase' else 'opposite'}), "
                         f"t > {result.primary_threshold:g}\n"
                         f"{len(result.significant)} significant (min p = {best_p:.3f})")
    first = results[(name, "increase")][0]
    null_text = "within-animal permutations" if analyses[name]["exchange_blocks"] is not None else \
        f"group-label permutations across animals ({'exact' if first.exact else 'random'}, " \
        f"n = {first.null_max_size.size})"
    fig.suptitle(f"NBS {analyses[name]['description']}, {len(np.unique(analyses[name]['animals']))} animals, "
                 f"{null_text}")
    fig.savefig(output_dir / f"nbs_{name}.png", dpi=200)
    plt.close(fig)

# Observed t of every edge, one heatmap per analysis. Edges of any significant
# component at the lowest threshold are boxed: black = increase, white = decrease.
analysis_names = list(analyses)
heatmap_columns = 2
heatmap_rows = int(np.ceil(len(analysis_names) / heatmap_columns))
fig, axes = plt.subplots(heatmap_rows, heatmap_columns, figsize=(6.5 * heatmap_columns, 6 * heatmap_rows),
                         constrained_layout=True, squeeze=False)
for ax in axes.flat[len(analysis_names):]:
    ax.axis("off")
for ax, name in zip(axes.flat, analysis_names):
    t_matrix = results[(name, "increase")][0].stat_matrix          # (n_rois, n_rois)
    image = ax.imshow(t_matrix, cmap="RdBu_r", vmin=-heatmap_t_limit, vmax=heatmap_t_limit)
    for direction, edge_color in [("increase", "black"), ("decrease", "white")]:
        for component in results[(name, direction)][0].significant:
            for i_node, j_node in component.edges:
                for row_index, col_index in [(i_node, j_node), (j_node, i_node)]:
                    ax.add_patch(plt.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1,
                                               fill=False, edgecolor=edge_color, linewidth=1.5))
    ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rois), roi_labels, fontsize=7)
    ax.set_title(f"{name}: t, {analyses[name]['description']}\n"
                 f"(boxed = significant at t > {primary_thresholds[0]:g})", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.7, extend="both", label=f"t (positive = {analyses[name]['positive']})")
fig.savefig(output_dir / "edge_t_heatmaps.png", dpi=200)
plt.close(fig)

# Permutation null of the maximum component size, observed largest component marked.
fig, axes = plt.subplots(2 * len(analysis_names), len(primary_thresholds),
                         figsize=(4.5 * len(primary_thresholds), 2.6 * 2 * len(analysis_names)),
                         constrained_layout=True, squeeze=False)
for i_row, (name, direction) in enumerate(results):
    for i_col, result in enumerate(results[(name, direction)]):
        ax = axes[i_row, i_col]
        null_sizes = result.null_max_size                           # (n_null,)
        bins = np.arange(null_sizes.max() + 2) - 0.5
        ax.hist(null_sizes, bins=bins, color="0.6")
        critical_size = np.quantile(null_sizes, 1 - alpha)
        ax.axvline(critical_size, color="black", linestyle="--", label=f"{1 - alpha:.0%} null")
        if result.components:
            best = result.components[0]
            ax.axvline(best.size, color="red", label=f"observed {best.size:g} (p={best.p_value:.3f})")
        ax.set_yscale("log")
        null_text = "exact" if result.exact else "random"
        ax.set_title(f"{name} {direction}, t > {result.primary_threshold:g} "
                     f"({null_text}, n = {null_sizes.size})", fontsize=9)
        ax.set_xlabel("max component size (edges)", fontsize=8)
        ax.legend(fontsize=7)
fig.savefig(output_dir / "null_distributions.png", dpi=150)
plt.close(fig)

# Descriptive group matrices over the between-group window: PV, R, T mean z (each
# animal weighted once), then PV - R and PV - T.
all_groups = [focus_group, *comparison_groups]
group_means = {}
group_animals = {}
for group in all_groups:
    selected = (animal_table["group"] == group).to_numpy()
    group_means[group] = animal_z[:, :, selected].mean(axis=-1)            # (n_rois, n_rois)
    group_animals[group] = animal_table.loc[selected, "animal"].tolist()
off_diagonal = ~np.eye(n_rois, dtype=bool)
z_low, z_high = np.percentile(np.concatenate([group_means[g][off_diagonal] for g in all_groups]), [2, 98])
panels = [(group_means[g], f"group {g} ({', '.join(group_animals[g])})", "viridis", z_low, z_high,
           "mean Fisher z") for g in all_groups]
panels += [(group_means[focus_group] - group_means[g], f"{focus_group} - {g}", "RdBu_r",
            -difference_z_limit, difference_z_limit, "difference in Fisher z") for g in comparison_groups]
fig, axes = plt.subplots(2, len(all_groups), figsize=(6.3 * len(all_groups), 12),
                         constrained_layout=True, squeeze=False)
for ax in axes.flat[len(panels):]:
    ax.axis("off")
for ax, (matrix, title, cmap, low, high, label) in zip(axes.flat, panels):
    shown = matrix.copy()
    np.fill_diagonal(shown, np.nan)
    image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)
    ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rois), roi_labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.7, extend="both", label=label)
fig.suptitle(f"{mouse_line} mean connectivity, {window_text} (one mean per animal)")
fig.savefig(output_dir / "group_mean_matrices.png", dpi=200)
plt.close(fig)

# PV animals over days: mean z over all off-diagonal edges, then over each significant
# PV_slope component (lowest threshold).
slope_name = f"{focus_group}_slope"
off_diagonal_rows, off_diagonal_cols = np.triu_indices(n_rois, k=1)
edge_sets = [("all edges", off_diagonal_rows, off_diagonal_cols)]
for direction in ["increase", "decrease"]:
    for component in results[(slope_name, direction)][0].significant:
        edge_sets.append((f"{slope_name} {direction} component ({component.size:g} edges, "
                          f"t > {primary_thresholds[0]:g}, p = {component.p_value:.3f})",
                          component.edges[:, 0], component.edges[:, 1]))
fig, axes = plt.subplots(len(edge_sets), 1, figsize=(6, 4.2 * len(edge_sets)),
                         constrained_layout=True, squeeze=False)
for ax, (title, edge_rows, edge_cols) in zip(axes[:, 0], edge_sets):
    edge_mean_z = session_z[edge_rows, edge_cols, :].mean(axis=0)          # (n_sessions,)
    for animal in pv_animals:
        animal_sessions = sessions[sessions["animal"] == animal].sort_values("day_index")
        ax.plot(animal_sessions["day_index"], edge_mean_z[animal_sessions.index], "o-", label=animal)
    ax.set_xlabel("day_index")
    ax.set_ylabel("mean Fisher z (session)")
    ax.set_title(f"group {focus_group}: {title}", fontsize=9)
    ax.legend(fontsize=7)
fig.savefig(output_dir / "pv_by_day.png", dpi=200)
plt.close(fig)

sessions.to_csv(output_dir / "sessions.csv", index=False)
print(f"\noutputs: {output_dir}")
