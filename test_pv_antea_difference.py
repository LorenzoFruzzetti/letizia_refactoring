"""PV3, PV4, PV5: Antea's difference-matrix figure, each later day minus day 1.

Replicates Antea_scripts/(5)_matrici_e_figure.txt, with "group A - group B" replaced by
"day N - baseline (day 1)" of the same animals:

    mean_R (per animal-day) = mean(R, 3)      plain mean of the recordings' Pearson r,
                                              script (4) line 89; no Fisher z
    group mean              = mean over animals of their mean_R (equal weight)
    DIFF                    = mean_day - mean_baseline
    figure                  = imagesc(DIFF): full 22x22 matrix including the (zero)
                              diagonal, MATLAB's parula colormap (parula_colormap.py),
                              Allen-style labels from script (5); colour scale
                              fixed at -1..1 (imagesc would autoscale per panel)

Input: every recording's `R_roi` (Pearson r) from roi_pixel_connectivity.py. Antea's
chain is mean-baseline dF/F + global signal regression, i.e. `pixels_dff_full_gsr`;
the other variants are drawn too for comparison.

Outputs (outputs/pv_antea_difference/):
    difference_<variant>.png   rows PV3/PV4/PV5/mean of the three, columns day 2..5 - day 1
    matrices_<variant>.png     same rows, columns day 1..5: the mean_R matrices themselves
    difference_matrices.csv    every panel's DIFF in long form (variant, animal, day, row, col)

Run: conda run --no-capture-output -n letizia python test_pv_antea_difference.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parula_colormap import parula

# ---------------------------------------------------------------- parameters
repo_root = Path(__file__).resolve().parent
design_path = repo_root / "pixel_data" / "experimental_design.csv"
connectivity_root = repo_root / "outputs" / "roi_pixel_connectivity"
connectivity_file = "roi_pixel_connectivity.npz"   # per recording: R_roi, roi_labels
output_dir = repo_root / "outputs" / "pv_antea_difference"

animals = ["PV3", "PV4", "PV5"]
baseline_day_index = 1
variants = [
    "pixels_dff_full_gsr",          # Antea's chain: mean-baseline dF/F + GSR
    "pixels_dff_full",
    "pixels_median_dff_20s_full",
    "pixels_median_dff_20s_full_gsr",
]
colormap = parula               # MATLAB's default, what Antea's imagesc used
color_limits = (-1.0, 1.0)      # fixed colour scale of every panel (None = per-panel
                                # autoscale, as imagesc does)
# Our ROI label -> the label Antea's script (5) uses for it.
antea_label_of = {
    "M2L_alta": "MOs-a_L", "M2L_bassa": "MOs-p_L", "M1L_alta": "MOp-a_L", "M1L_bassa": "MOp-p_L",
    "BFDL": "SSp-bfd_L", "TrL": "SSp-tr_L", "FLL": "SSp-fL_L", "HLL": "SSp-hl_L",
    "RSL_alta": "RSP_L", "V1aL": "VISa_L", "V1L": "VISp_L",
    "M2R_alta": "MOs-a_R", "M2R_bassa": "MOs-p_R", "M1R_alta": "MOp-a_R", "M1R_bassa": "MOp-p_R",
    "BFDR": "SSp-bfd_R", "TrR": "SSp-tr_R", "FLR": "SSp-fL_R", "HLR": "SSp-hl_R",
    "RSR_alta": "RSP_R", "V1aR": "VISa_R", "V1R": "VISp_R",
}

output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loading
# Every session of each animal by day_index.
design = pd.read_csv(design_path, dtype={"day": str})
design["session"] = design["day"] + "_" + design["animal"]
sessions = {}                                          # animal -> {day_index: session}
for animal in animals:
    by_day = design[design["animal"] == animal].groupby("day_index")["session"].unique()
    if any(len(s) != 1 for s in by_day):
        raise ValueError(f"{animal}: more than one session on a day_index: {by_day.to_dict()}")
    sessions[animal] = {int(day): s[0] for day, s in by_day.items()}
    if baseline_day_index not in sessions[animal]:
        raise ValueError(f"{animal}: no day_index {baseline_day_index} session")
later_days = sorted({d for a in animals for d in sessions[a]} - {baseline_day_index})

# mean_r[variant][animal][day]: (n_rois, n_rois) plain mean of R over the session's
# recordings -- Antea's mean_R = mean(R, 3).
mean_r = {}
roi_labels = None
for variant in variants:
    mean_r[variant] = {}
    for animal in animals:
        mean_r[variant][animal] = {}
        for day, session in sessions[animal].items():
            recordings = design.loc[design["session"] == session, "recording"].sort_values()
            matrices = []
            for recording in recordings:
                data = np.load(connectivity_root / variant / session / recording / connectivity_file)
                labels = list(data["roi_labels"])
                if roi_labels is None:
                    roi_labels = labels
                if labels != roi_labels:
                    raise ValueError(f"ROI labels differ in {variant}/{session}/{recording}")
                matrices.append(data["R_roi"])
            mean_r[variant][animal][day] = np.mean(matrices, axis=0)
n_rois = len(roi_labels)
tick_labels = [antea_label_of[label] for label in roi_labels]

# ---------------------------------------------------------------- differences
# DIFF = mean_day - mean_baseline per animal; the "mean" row averages the three animals'
# mean_R first, as Antea's group means do, then subtracts.
panel_names = animals + ["mean"]
difference = {}                                        # [variant][name][day]: (n_rois, n_rois)
long_rows = []
for variant in variants:
    difference[variant] = {name: {} for name in panel_names}
    for day in later_days:
        for animal in animals:
            difference[variant][animal][day] = (mean_r[variant][animal][day]
                                                - mean_r[variant][animal][baseline_day_index])
        group_day = np.mean([mean_r[variant][a][day] for a in animals], axis=0)
        group_baseline = np.mean([mean_r[variant][a][baseline_day_index] for a in animals], axis=0)
        difference[variant]["mean"][day] = group_day - group_baseline
        for name in panel_names:
            for i_row in range(n_rois):
                for i_col in range(n_rois):
                    long_rows.append({"variant": variant, "animal": name, "day_index": day,
                                      "row": roi_labels[i_row], "col": roi_labels[i_col],
                                      "diff_r": difference[variant][name][day][i_row, i_col]})
pd.DataFrame(long_rows).to_csv(output_dir / "difference_matrices.csv", index=False)

# ---------------------------------------------------------------- plotting
# One figure per variant; rows = animals + mean, columns = day N - day 1.
for variant in variants:
    fig, axes = plt.subplots(len(panel_names), len(later_days),
                             figsize=(4.6 * len(later_days), 4.4 * len(panel_names)),
                             constrained_layout=True, squeeze=False)
    for i_row, name in enumerate(panel_names):
        for i_col, day in enumerate(later_days):
            ax = axes[i_row, i_col]
            matrix = difference[variant][name][day]
            low, high = color_limits if color_limits is not None else (matrix.min(), matrix.max())
            image = ax.imshow(matrix, cmap=colormap, vmin=low, vmax=high, interpolation="nearest")
            if name == "mean":
                title = f"mean {'+'.join(animals)}: day {day} - day {baseline_day_index}"
            else:
                title = f"{name}: {sessions[name][day]} - {sessions[name][baseline_day_index]}"
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(n_rois), tick_labels, fontsize=5)
            fig.colorbar(image, ax=ax, shrink=0.75, label="difference in Pearson r")
    scale_note = (f"colour scale {color_limits[0]:g} to {color_limits[1]:g}" if color_limits is not None
                  else "colour scale per panel (imagesc)")
    fig.suptitle(f"{variant}: DIFF = mean R day N - mean R day {baseline_day_index} "
                 f"(plain mean of Pearson r, Antea script 5; {scale_note})")
    fig.savefig(output_dir / f"difference_{variant}.png", dpi=200)
    plt.close(fig)
    print(f"saved {output_dir / f'difference_{variant}.png'}")

# ---------------------------------------------------------------- plotting: matrices per day
# Same layout, but each panel is the mean_R of that day itself (day 1 included); the
# "mean" row is the equal-weight mean of the three animals' mean_R.
all_days = [baseline_day_index] + later_days
for variant in variants:
    fig, axes = plt.subplots(len(panel_names), len(all_days),
                             figsize=(4.6 * len(all_days), 4.4 * len(panel_names)),
                             constrained_layout=True, squeeze=False)
    for i_row, name in enumerate(panel_names):
        for i_col, day in enumerate(all_days):
            ax = axes[i_row, i_col]
            if name == "mean":
                matrix = np.mean([mean_r[variant][a][day] for a in animals], axis=0)
                title = f"mean {'+'.join(animals)}: day {day}"
            else:
                matrix = mean_r[variant][name][day]
                title = f"{name}: {sessions[name][day]} (day {day})"
            low, high = color_limits if color_limits is not None else (matrix.min(), matrix.max())
            image = ax.imshow(matrix, cmap=colormap, vmin=low, vmax=high, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(n_rois), tick_labels, fontsize=5)
            fig.colorbar(image, ax=ax, shrink=0.75, label="Pearson r")
    scale_note = (f"colour scale {color_limits[0]:g} to {color_limits[1]:g}" if color_limits is not None
                  else "colour scale per panel (imagesc)")
    fig.suptitle(f"{variant}: mean R per day (plain mean of Pearson r over 5 recordings, "
                 f"Antea script 5; {scale_note})")
    fig.savefig(output_dir / f"matrices_{variant}.png", dpi=200)
    plt.close(fig)
    print(f"saved {output_dir / f'matrices_{variant}.png'}")
