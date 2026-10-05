"""PV3, PV4, PV5: ROI connectivity on EVERY later session, each compared with day 1.

Extends test_pv_session_vs_day1.py / test_pv_last_vs_day1_pooled.py, which only looked
at the last session, to the whole time course: day_index 2, 3, 4 and 5 of each animal
are compared with the same animal's day_index 1 session, so one can see whether the
day-5 change (e.g. PV5's left-hemisphere block) builds up gradually or appears late.

Figures show Pearson r by default (`display_scale`): z is averaged, then r = tanh(mean z).
The Welch t / permutation tests are always on Fisher z.

Input is the same: every recording's `R_roi` (22x22 Pearson r of box-mean traces,
roi_pixel_connectivity.py), Fisher-z transformed, in all four connectivity variants.

Statistics are WITHIN ONE ANIMAL, the 5 recordings of a session as replicates (exact
5-vs-5 permutation, smallest two-sided p = 2/252 = 0.008). They describe how far a
change exceeds recording-to-recording variability, not population inference. Nothing
is corrected for the 4 days x 4 variants x 3 animals x metrics compared.

Outputs (outputs/pv_days_vs_day1/):
    metrics_per_recording.csv     edge-set mean z per recording, every session
    metric_tests.csv              each day vs day 1, per animal / metric / variant
    edge_changes.csv              every edge, day, animal, variant: r day 1 / day, differences, Welch t
    pattern_similarity.csv        session x session edge-pattern r per animal and variant
    difference_similarity.csv     r between each day's change and the day-5 change
    difference_by_day_<variant>.png  rows PV3/PV4/PV5/pooled, columns day 2..5 - day 1
    metrics_by_day.png            edge-set mean z over days, per animal
    pattern_similarity_by_day.png session x session edge-pattern r matrices

Run: conda run --no-capture-output -n letizia python test_pv_days_vs_day1.py
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
connectivity_file = "roi_pixel_connectivity.npz"   # per recording: R_roi, roi_labels
output_dir = repo_root / "outputs" / "pv_days_vs_day1"

animals = ["PV3", "PV4", "PV5"]
baseline_day_index = 1
variants = [
    "pixels_dff_full",
    "pixels_dff_full_gsr",
    "pixels_median_dff_20s_full",
    "pixels_median_dff_20s_full_gsr",
]
edge_t_threshold = 2.5      # per-animal |Welch t| that counts as an edge change
# Values shown in the figures. "r": Pearson correlation; "z": Fisher z. Averaging is always
# done in z (over recordings, then animals) and only the result is converted back to r,
# r = tanh(mean z), so the two scales show the same comparison. Tests stay in z.
display_scale = "r"
difference_limit = {"r": 0.3, "z": 0.6}[display_scale]   # colour limit of the difference matrices
# Edges of the GSR-variant PV_slope decrease core (9.34/9.35): left visual - right limbs.
core_edges = [("V1aL", "FLR"), ("V1aL", "HLR"), ("V1L", "FLR"), ("V1L", "HLR")]
animal_colors = {"PV3": "tab:blue", "PV4": "tab:orange", "PV5": "tab:green"}

output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loading
# Every session of each animal, ordered by day_index.
design = pd.read_csv(design_path, dtype={"day": str})
design["session"] = design["day"] + "_" + design["animal"]
sessions = {}                                          # animal -> {day_index: session}
for animal in animals:
    rows = design[design["animal"] == animal]
    by_day = rows.groupby("day_index")["session"].unique()
    if any(len(s) != 1 for s in by_day):
        raise ValueError(f"{animal}: more than one session on a day_index: {by_day.to_dict()}")
    sessions[animal] = {int(day): s[0] for day, s in by_day.items()}
    if baseline_day_index not in sessions[animal]:
        raise ValueError(f"{animal}: no day_index {baseline_day_index} session")
    print(f"{animal}: " + ", ".join(f"day {d} = {s}" for d, s in sessions[animal].items()))
day_indices = sorted({d for a in animals for d in sessions[a]})
later_days = [d for d in day_indices if d != baseline_day_index]
last_day = max(day_indices)

recordings_of = {s: design.loc[design["session"] == s, "recording"].sort_values().tolist()
                 for a in animals for s in sessions[a].values()}

# session_z[variant][session]: (n_recordings, n_rois, n_rois) Fisher z per recording.
session_z = {}
roi_labels = None
for variant in variants:
    session_z[variant] = {}
    for session, recordings in recordings_of.items():
        matrices = []
        for recording in recordings:
            data = np.load(connectivity_root / variant / session / recording / connectivity_file)
            labels = list(data["roi_labels"])
            if roi_labels is None:
                roi_labels = labels
            if labels != roi_labels:
                raise ValueError(f"ROI labels differ in {variant}/{session}/{recording}")
            matrices.append(fisher_z(data["R_roi"]))
        session_z[variant][session] = np.stack(matrices)
n_rois = len(roi_labels)
print(f"loaded {len(recordings_of)} sessions x {len(variants)} variants, {n_rois} ROIs")

# ---------------------------------------------------------------- edge sets
# First 11 labels are the left ROIs, the last 11 their right twins in the same order
# (checked in test_pv_session_vs_day1.py; checked again here).
half = n_rois // 2
for i_roi in range(half):
    left, right = roi_labels[i_roi], roi_labels[i_roi + half]
    if left.replace("L", "R", 1) != right and left[::-1].replace("L", "R", 1)[::-1] != right:
        raise ValueError(f"{left} and {right} are not homotopic twins")
side = np.array(["L"] * half + ["R"] * half)
upper = np.triu(np.ones((n_rois, n_rois), dtype=bool), k=1)
row_side, col_side = side[:, None], side[None, :]
homotopic = np.zeros((n_rois, n_rois), dtype=bool)
for i_roi in range(half):
    homotopic[i_roi, i_roi + half] = True
core = np.zeros((n_rois, n_rois), dtype=bool)
for first, second in core_edges:
    i_first, i_second = roi_labels.index(first), roi_labels.index(second)
    core[min(i_first, i_second), max(i_first, i_second)] = True
edge_sets = {
    "all": upper,
    "intra_left": upper & (row_side == "L") & (col_side == "L"),
    "intra_right": upper & (row_side == "R") & (col_side == "R"),
    "homotopic": homotopic,
    "heterotopic": upper & (row_side != col_side) & ~homotopic,
    "V1L-limbR core": core,
}
edge_rows, edge_cols = np.nonzero(upper)                  # 231 unique edges
edge_names = [f"{roi_labels[r]}-{roi_labels[c]}" for r, c in zip(edge_rows, edge_cols)]


def to_display(z_values):
    """Fisher z -> the display scale (Pearson r = tanh(z), or z unchanged)."""
    return np.tanh(z_values) if display_scale == "r" else z_values


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
# Mean z over each edge set, one value per recording, every session.
metric_rows = []
for variant in variants:
    for animal in animals:
        for day, session in sessions[animal].items():
            for i_rec, recording in enumerate(recordings_of[session]):
                z = session_z[variant][session][i_rec]
                row = {"variant": variant, "animal": animal, "day_index": day,
                       "session": session, "recording": recording}
                row.update({name: z[mask].mean() for name, mask in edge_sets.items()})
                metric_rows.append(row)
metrics = pd.DataFrame(metric_rows)
metrics.to_csv(output_dir / "metrics_per_recording.csv", index=False)

# Each later day vs day 1: difference of means, Welch t, exact permutation p.
test_rows = []
for (variant, animal), group in metrics.groupby(["variant", "animal"], sort=False):
    first_rows = group[group["day_index"] == baseline_day_index]
    for day in later_days:
        second_rows = group[group["day_index"] == day]
        for name in edge_sets:
            first, second = first_rows[name].to_numpy(), second_rows[name].to_numpy()
            test_rows.append({"variant": variant, "animal": animal, "day_index": day,
                              "metric": name, "z_day1": first.mean(), "z_day": second.mean(),
                              "difference": second.mean() - first.mean(),
                              "welch_t": float(welch_t(first, second)),
                              "permutation_p": permutation_p(first, second)})
metric_tests = pd.DataFrame(test_rows)
metric_tests.to_csv(output_dir / "metric_tests.csv", index=False)

# ---------------------------------------------------------------- per-edge change
# difference[variant][animal][day], t_values[...]: (n_rois, n_rois), day - day 1, in z.
# display_difference[...]: the same change on the display scale, to_display(mean z day)
# - to_display(mean z day 1); with display_scale "r" a change in Pearson r.
difference, t_values, display_difference = {}, {}, {}
edge_rows_out = []
for variant in variants:
    difference[variant], t_values[variant], display_difference[variant] = {}, {}, {}
    for animal in animals:
        difference[variant][animal], t_values[variant][animal] = {}, {}
        display_difference[variant][animal] = {}
        first = session_z[variant][sessions[animal][baseline_day_index]]   # (n_recordings, n_rois, n_rois)
        for day in later_days:
            second = session_z[variant][sessions[animal][day]]
            change = second.mean(axis=0) - first.mean(axis=0)
            t_matrix = welch_t(first, second)
            np.fill_diagonal(t_matrix, 0.0)
            difference[variant][animal][day], t_values[variant][animal][day] = change, t_matrix
            r_day1, r_day = np.tanh(first.mean(axis=0)), np.tanh(second.mean(axis=0))
            display_difference[variant][animal][day] = (to_display(second.mean(axis=0))
                                                        - to_display(first.mean(axis=0)))
            for name, r, c in zip(edge_names, edge_rows, edge_cols):
                edge_rows_out.append({"variant": variant, "animal": animal, "day_index": day,
                                      "edge": name, "r_day1": r_day1[r, c], "r_day": r_day[r, c],
                                      "r_difference": r_day[r, c] - r_day1[r, c],
                                      "difference": change[r, c],
                                      "welch_t": t_matrix[r, c]})
edge_changes = pd.DataFrame(edge_rows_out)
edge_changes.to_csv(output_dir / "edge_changes.csv", index=False)

# Does the last-day change build up? r over the 231 edges between each day's change and
# the last day's change (1.0 on the last day by definition), plus the number of edges
# past the threshold on that day.
difference_similarity_rows = []
for variant in variants:
    for animal in animals:
        last_change = difference[variant][animal][last_day][edge_rows, edge_cols]
        for day in later_days:
            change = difference[variant][animal][day][edge_rows, edge_cols]
            t_edges = t_values[variant][animal][day][edge_rows, edge_cols]
            difference_similarity_rows.append({
                "variant": variant, "animal": animal, "day_index": day,
                "r_with_last_day_change": np.corrcoef(change, last_change)[0, 1],
                "mean_abs_change": np.abs(change).mean(),
                "n_edges_up": int((t_edges > edge_t_threshold).sum()),
                "n_edges_down": int((t_edges < -edge_t_threshold).sum())})
difference_similarity = pd.DataFrame(difference_similarity_rows)
difference_similarity.to_csv(output_dir / "difference_similarity.csv", index=False)

# ---------------------------------------------------------------- pattern similarity
# Mean Pearson r between the 231-edge z vectors of recordings of two sessions. The
# diagonal is within-session (distinct recording pairs). Insensitive to a uniform shift.
similarity = {}                                           # [variant][animal]: (n_days, n_days)
similarity_rows = []
for variant in variants:
    similarity[variant] = {}
    for animal in animals:
        vectors = {d: session_z[variant][s][:, edge_rows, edge_cols]          # (n_recordings, n_edges)
                   for d, s in sessions[animal].items()}
        matrix = np.zeros((len(day_indices), len(day_indices)))
        for i_a, day_a in enumerate(day_indices):
            for i_b, day_b in enumerate(day_indices):
                if day_a == day_b:
                    values = [np.corrcoef(vectors[day_a][x], vectors[day_a][y])[0, 1]
                              for x, y in itertools.combinations(range(len(vectors[day_a])), 2)]
                else:
                    values = [np.corrcoef(x, y)[0, 1] for x in vectors[day_a] for y in vectors[day_b]]
                matrix[i_a, i_b] = np.mean(values)
                similarity_rows.append({"variant": variant, "animal": animal, "day_a": day_a,
                                        "day_b": day_b, "mean_r": matrix[i_a, i_b]})
        similarity[variant][animal] = matrix
pd.DataFrame(similarity_rows).to_csv(output_dir / "pattern_similarity.csv", index=False)

# ---------------------------------------------------------------- console summary
print("\nmean z over all edges / intra_left / homotopic, per day (session mean):")
print(metrics.pivot_table(index=["variant", "animal"], columns="day_index",
                          values="all").round(3).to_string())
for name in ["intra_left", "homotopic", "V1L-limbR core"]:
    print(f"\n{name}:")
    print(metrics.pivot_table(index=["variant", "animal"], columns="day_index",
                              values=name).round(3).to_string())
print(f"\nr of each day's change with the day-{last_day} change, and edges |t| > {edge_t_threshold:g} (up/down):")
summary = difference_similarity.assign(
    cell=lambda d: d["r_with_last_day_change"].round(2).astype(str) + " ("
    + d["n_edges_up"].astype(str) + "/" + d["n_edges_down"].astype(str) + ")")
print(summary.pivot_table(index=["variant", "animal"], columns="day_index",
                          values="cell", aggfunc="first").to_string())

display_label = "Pearson r = tanh(mean z)" if display_scale == "r" else "Fisher z"

# ---------------------------------------------------------------- plotting: difference matrices
# One figure per variant. Rows = animals + pooled; columns = each later day - day 1.
# Dots: per animal |Welch t| > threshold; pooled, all animals past it in the same direction.
row_names = animals + ["pooled"]
for variant in variants:
    fig, axes = plt.subplots(len(row_names), len(later_days),
                             figsize=(4.3 * len(later_days), 4.2 * len(row_names)),
                             constrained_layout=True, squeeze=False)
    for i_col, day in enumerate(later_days):
        changes = np.stack([display_difference[variant][a][day] for a in animals])   # (n_animals, n_rois, n_rois)
        t_stack = np.stack([t_values[variant][a][day] for a in animals])
        agreement = (t_stack > edge_t_threshold).sum(axis=0) - (t_stack < -edge_t_threshold).sum(axis=0)
        for i_row, name in enumerate(row_names):
            ax = axes[i_row, i_col]
            if name == "pooled":
                shown = changes.mean(axis=0)
                marked = np.abs(agreement) == len(animals)
                title = f"pooled, day {day} - day {baseline_day_index}"
            else:
                shown = display_difference[variant][name][day]
                marked = np.abs(t_values[variant][name][day]) > edge_t_threshold
                title = f"{name}: {sessions[name][day]} - {sessions[name][baseline_day_index]}"
            shown = shown.copy()
            np.fill_diagonal(shown, np.nan)
            image = ax.imshow(shown, cmap="RdBu_r", vmin=-difference_limit, vmax=difference_limit)
            marked_r, marked_c = np.nonzero(marked)
            ax.plot(marked_c, marked_r, "k.", markersize=2)
            ax.set_title(f"day {day}\n{title}", fontsize=9)
            ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(n_rois), roi_labels, fontsize=5)
            ax.axhline(half - 0.5, color="k", linewidth=0.6)
            ax.axvline(half - 0.5, color="k", linewidth=0.6)
            fig.colorbar(image, ax=ax, shrink=0.75, label=f"difference in {display_scale}")
    fig.suptitle(f"{variant}: ROI connectivity change vs day {baseline_day_index} ({display_label}; dots: "
                 f"per animal |Welch t| > {edge_t_threshold:g} over 5 vs 5 recordings, "
                 f"pooled = all {len(animals)} agree)")
    fig.savefig(output_dir / f"difference_by_day_{variant}.png", dpi=200)
    plt.close(fig)
    print(f"saved {output_dir / f'difference_by_day_{variant}.png'}")

# ---------------------------------------------------------------- plotting: metrics over days
# Rows = variants; columns = edge sets. Points = recordings, line = session mean.
metric_names = list(edge_sets)
fig, axes = plt.subplots(len(variants), len(metric_names),
                         figsize=(3.2 * len(metric_names), 2.8 * len(variants)),
                         constrained_layout=True, squeeze=False, sharex=True)
jitter = {a: (i - 1) * 0.12 for i, a in enumerate(animals)}   # horizontal offset per animal
for i_var, variant in enumerate(variants):
    subset = metrics[metrics["variant"] == variant]
    for i_col, name in enumerate(metric_names):
        ax = axes[i_var, i_col]
        for animal in animals:
            rows = subset[subset["animal"] == animal]
            ax.plot(rows["day_index"] + jitter[animal], to_display(rows[name]), ".",
                    color=animal_colors[animal], alpha=0.4, markersize=4)
            session_mean = rows.groupby("day_index")[name].mean()
            ax.plot(session_mean.index + jitter[animal], to_display(session_mean.to_numpy()), "-o",
                    color=animal_colors[animal], markersize=3, label=animal)
        ax.set_title(f"{variant}\n{name}", fontsize=8)
        ax.set_xticks(day_indices)
        if i_var == len(variants) - 1:
            ax.set_xlabel("day_index")
        if i_col == 0:
            ax.set_ylabel(f"mean {display_scale}")
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Edge-set mean connectivity over sessions ({display_label}; "
             f"dots = recordings, line = session mean)")
fig.savefig(output_dir / "metrics_by_day.png", dpi=200)
plt.close(fig)
print(f"saved {output_dir / 'metrics_by_day.png'}")

# ---------------------------------------------------------------- plotting: pattern similarity
# Rows = variants; columns = animals. Cell = mean edge-pattern r between recordings of
# two sessions; the diagonal is the within-session reference.
all_values = np.concatenate([similarity[v][a].ravel() for v in variants for a in animals])
fig, axes = plt.subplots(len(variants), len(animals),
                         figsize=(3.8 * len(animals), 3.4 * len(variants)),
                         constrained_layout=True, squeeze=False)
for i_var, variant in enumerate(variants):
    for i_col, animal in enumerate(animals):
        ax = axes[i_var, i_col]
        matrix = similarity[variant][animal]
        image = ax.imshow(matrix, cmap="viridis", vmin=all_values.min(), vmax=1.0)
        for i_a in range(len(day_indices)):
            for i_b in range(len(day_indices)):
                ax.text(i_b, i_a, f"{matrix[i_a, i_b]:.2f}", ha="center", va="center",
                        fontsize=6, color="w" if matrix[i_a, i_b] < 0.9 else "k")
        ax.set_xticks(range(len(day_indices)), day_indices)
        ax.set_yticks(range(len(day_indices)), day_indices)
        ax.set_title(f"{variant}\n{animal}", fontsize=8)
        ax.set_xlabel("day_index")
        ax.set_ylabel("day_index")
        fig.colorbar(image, ax=ax, shrink=0.8, label="edge-pattern r")
fig.suptitle("Edge-pattern similarity between sessions (diagonal = within session)")
fig.savefig(output_dir / "pattern_similarity_by_day.png", dpi=200)
plt.close(fig)
print(f"saved {output_dir / 'pattern_similarity_by_day.png'}")
