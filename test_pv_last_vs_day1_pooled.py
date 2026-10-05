"""PV3 + PV4 + PV5 pooled: ROI connectivity on day 1 vs on each animal's last day.

Same input as test_pv_session_vs_day1.py (every recording's `R_roi`, Fisher z), but
the three animals are combined instead of shown one per row:

    day 1     each animal's day_index 1 session (260520 PV3, 260520 PV4, 260608 PV5)
    last day  each animal's highest day_index session (260805 for all three, day 5)

Each session is first averaged over its 5 recordings, then the three animals are
averaged with equal weight, so no animal counts more because it has more recordings.

With only three animals there is no meaningful group test per edge, so the fourth
column shows agreement instead: for every edge, the number of animals whose own
Welch t (5 vs 5 recordings) exceeds +threshold minus the number below -threshold.
+3 / -3 = all three animals change that edge in the same direction.

Rows are the four connectivity variants; the `_gsr` rows have global signal
regression (roi_pixel_connectivity.py --gsr), the others do not.

Outputs: outputs/pv_session_vs_day1/pooled_last_vs_day1.png (day 1, last day, difference, agreement)
         outputs/pv_session_vs_day1/difference_last_minus_day1.png (difference per animal + pooled)

Run: conda run --no-capture-output -n letizia python test_pv_last_vs_day1_pooled.py
"""

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
output_dir = repo_root / "outputs" / "pv_session_vs_day1"

animals = ["PV3", "PV4", "PV5"]
baseline_day_index = 1
variants = [
    "pixels_dff_full",
    "pixels_dff_full_gsr",
    "pixels_median_dff_20s_full",
    "pixels_median_dff_20s_full_gsr",
]
edge_t_threshold = 2.5      # per-animal |Welch t| that counts as a change
matrix_z_limit = 1.5        # colour limit of the mean-z matrices (as in matrices_<variant>.png)
difference_z_limit = 0.6    # colour limit of the difference matrices

output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loading
# Day-1 and last-day session of each animal from the experimental design.
design = pd.read_csv(design_path, dtype={"day": str})
design["session"] = design["day"] + "_" + design["animal"]
sessions = {}                                          # animal -> (day1 session, last session, last day_index)
for animal in animals:
    rows = design[design["animal"] == animal]
    first = rows.loc[rows["day_index"] == baseline_day_index, "session"].unique()
    last_index = int(rows["day_index"].max())
    last = rows.loc[rows["day_index"] == last_index, "session"].unique()
    if len(first) != 1 or len(last) != 1:
        raise ValueError(f"{animal}: expected one day-1 and one last session, got {first} / {last}")
    sessions[animal] = (first[0], last[0], last_index)
    print(f"{animal}: day 1 = {first[0]}, last = {last[0]} (day {last_index})")

recordings_of = {s: design.loc[design["session"] == s, "recording"].sort_values().tolist()
                 for pair in sessions.values() for s in pair[:2]}

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
half = n_rois // 2                                     # first 11 left, last 11 right

# ---------------------------------------------------------------- pooled matrices
# Per animal: session mean over recordings; then the equal-weight mean over animals.
pooled = {}                                            # [variant] -> dict of (n_rois, n_rois)
for variant in variants:
    day1_means = np.stack([session_z[variant][sessions[a][0]].mean(axis=0) for a in animals])   # (n_animals, n_rois, n_rois)
    last_means = np.stack([session_z[variant][sessions[a][1]].mean(axis=0) for a in animals])

    # Agreement: per animal Welch t over its 5 vs 5 recordings, then signed count.
    # The per-animal t matrices are kept for the difference figure below.
    agreement = np.zeros((n_rois, n_rois))
    animal_t = {}                                      # animal -> (n_rois, n_rois) Welch t
    for animal in animals:
        first = session_z[variant][sessions[animal][0]]
        second = session_z[variant][sessions[animal][1]]
        standard_error = np.sqrt(first.var(axis=0, ddof=1) / len(first)
                                 + second.var(axis=0, ddof=1) / len(second))
        t_values = (second.mean(axis=0) - first.mean(axis=0)) / standard_error
        np.fill_diagonal(t_values, 0.0)
        agreement += (t_values > edge_t_threshold).astype(float) - (t_values < -edge_t_threshold)
        animal_t[animal] = t_values

    pooled[variant] = {"day1": day1_means.mean(axis=0), "last": last_means.mean(axis=0),
                       "difference": (last_means - day1_means).mean(axis=0), "agreement": agreement,
                       "animal_difference": dict(zip(animals, last_means - day1_means)),
                       "animal_t": animal_t}
    upper = np.triu_indices(n_rois, k=1)
    print(f"{variant}: mean z day 1 {pooled[variant]['day1'][upper].mean():.3f}, "
          f"last {pooled[variant]['last'][upper].mean():.3f}; edges all 3 up "
          f"{int((agreement[upper] == 3).sum())}, all 3 down {int((agreement[upper] == -3).sum())}")

# ---------------------------------------------------------------- plotting
# Rows = variants; columns = pooled day 1, pooled last day, difference, agreement.
last_days = sorted({sessions[a][2] for a in animals})
last_label = f"day {last_days[0]}" if len(last_days) == 1 else "last day"
fig, axes = plt.subplots(len(variants), 4, figsize=(17, 4.2 * len(variants)),
                         constrained_layout=True, squeeze=False)
for i_var, variant in enumerate(variants):
    panels = [
        (pooled[variant]["day1"], "day 1, PV3+PV4+PV5", "viridis", 0, matrix_z_limit, "mean z"),
        (pooled[variant]["last"], f"{last_label}, PV3+PV4+PV5", "viridis", 0, matrix_z_limit, "mean z"),
        (pooled[variant]["difference"], f"{last_label} - day 1", "RdBu_r",
         -difference_z_limit, difference_z_limit, "difference in z"),
        (pooled[variant]["agreement"], f"animals with |t| > {edge_t_threshold:g} (up - down)",
         "RdBu_r", -3, 3, "n animals"),
    ]
    for i_col, (matrix, title, cmap, low, high, unit) in enumerate(panels):
        ax = axes[i_var, i_col]
        shown = matrix.copy()
        np.fill_diagonal(shown, np.nan)
        image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)
        if i_col == 3:
            # Dots: all three animals change the edge in the same direction.
            marked_r, marked_c = np.nonzero(np.abs(matrix) == len(animals))
            ax.plot(marked_c, marked_r, "k.", markersize=3)
        ax.set_title(f"{variant}\n{title}", fontsize=9)
        ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(n_rois), roi_labels, fontsize=5)
        ax.axhline(half - 0.5, color="w" if i_col < 2 else "k", linewidth=0.6)
        ax.axvline(half - 0.5, color="w" if i_col < 2 else "k", linewidth=0.6)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.75, label=unit)
        if i_col == 3:
            colorbar.set_ticks(range(-3, 4))
fig.suptitle(f"ROI connectivity pooled over {', '.join(animals)}: {last_label} vs day 1 "
             f"(each animal = mean of its 5 recordings, animals weighted equally; "
             f"dots in column 4 = all {len(animals)} agree)")
fig.savefig(output_dir / "pooled_last_vs_day1.png", dpi=200)
plt.close(fig)
print(f"saved {output_dir / 'pooled_last_vs_day1.png'}")

# ---------------------------------------------------------------- plotting: differences only
# Rows = variants; columns = each animal's last day - day 1, then the pooled mean.
# Dots: per animal, |Welch t| > threshold; pooled, all animals agree in direction.
column_names = animals + ["pooled"]
fig, axes = plt.subplots(len(variants), len(column_names),
                         figsize=(4.3 * len(column_names), 4.2 * len(variants)),
                         constrained_layout=True, squeeze=False)
for i_var, variant in enumerate(variants):
    for i_col, name in enumerate(column_names):
        ax = axes[i_var, i_col]
        if name == "pooled":
            difference = pooled[variant]["difference"]
            marked = np.abs(pooled[variant]["agreement"]) == len(animals)
            title = f"pooled {'+'.join(animals)}"
        else:
            difference = pooled[variant]["animal_difference"][name]
            marked = np.abs(pooled[variant]["animal_t"][name]) > edge_t_threshold
            title = f"{name}: {sessions[name][1]} - {sessions[name][0]}"
        shown = difference.copy()
        np.fill_diagonal(shown, np.nan)
        image = ax.imshow(shown, cmap="RdBu_r", vmin=-difference_z_limit, vmax=difference_z_limit)
        marked_r, marked_c = np.nonzero(marked)
        ax.plot(marked_c, marked_r, "k.", markersize=2)
        ax.set_title(f"{variant}\n{title}", fontsize=9)
        ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(n_rois), roi_labels, fontsize=5)
        ax.axhline(half - 0.5, color="k", linewidth=0.6)
        ax.axvline(half - 0.5, color="k", linewidth=0.6)
        fig.colorbar(image, ax=ax, shrink=0.75, label="difference in z")
fig.suptitle(f"ROI connectivity change, {last_label} - day 1 (Fisher z; dots: per animal "
             f"|Welch t| > {edge_t_threshold:g} over 5 vs 5 recordings, pooled = all "
             f"{len(animals)} animals agree)")
fig.savefig(output_dir / "difference_last_minus_day1.png", dpi=200)
plt.close(fig)
print(f"saved {output_dir / 'difference_last_minus_day1.png'}")
