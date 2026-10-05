"""Tails of the peak-height distribution, per recording, over every cached pixel dump.

For each recording in `pixel_data/experimental_design.csv` the cached 60 s median
dF/F volume (`pixels_median_dff_60s_full.npy`, see cache_median_dff.py) is averaged
inside each of the 22 atlas ROI boxes, plus three summary traces (the mean of all
ROIs and of each hemisphere). Every local maximum is then found with
`scipy.signal.find_peaks` at a DELIBERATELY LOW prominence, so the peak list is
close to "every wiggle" and its height distribution is the trace's own, not one
already cut by a detector. The top `tail_percent` of those peaks (per trace, per
recording) is the tail analysed here.

Every tail metric is given twice:
    *_pct   percent dF/F, the value as recorded;
    *_z     robust z of the trace itself, (height - median) / (1.4826 * MAD).
Noise level differs 2-3x between recordings (e.g. 0.55 % vs 1.85 % robust SD on
M2L_alta), so a raw tail can be heavy only because the recording is noisy; the z
version asks whether the tail is heavy RELATIVE TO that recording's own noise.

The 60 s baseline already removes slow drift, so no further detrending is done.
The first `skip_start_s` seconds are ignored: they are an onset artifact (see the
parameter's comment). Frame numbers in the outputs are still the full recording's.

Input:  pixel_data/experimental_design.csv
        pixel_data/<day>_<animal>/<t#>/pixels_median_dff_60s_full.npy  (time, y, x)
        pixel_data/<day>_<animal>/<t#>/pixels_meta_full.npz            crop + ROI set
        roi_sets/rebuilt/<day>_<animal>_<t#>.yaml                       the ROI boxes
Output: outputs/peak_tails_60s/
          roi_traces.npz                  every trace, (n_recordings, n_frames, n_traces)
          tail_summary.csv                one row per recording x trace
          tail_peaks.csv                  every peak in a tail, one row each
          tail_by_animal_day.csv          summary traces averaged over the t# of a day
          tail_by_group_roi.csv           per group x trace: medians over recordings
          tail_ranking_cortex_mean.csv    recordings ordered by Cortex_mean tail (z)
          peak_height_ccdf_by_group.png   survival function of peak height, per group
          tail_by_group.png               per-recording tail metrics by group
          tail_by_day.png                 tail per animal across recording days
          tail_by_roi_heatmap.png         tail per ROI x group
          top_tail_recordings.png         the recordings with the heaviest tails

Run it (the env's own interpreter dies on the first BLAS call, CLAUDE.md 9.17):
    conda run --no-capture-output -n letizia python -u analyze_peak_tails.py
"""

import os

# MUST run before numpy/scipy/matplotlib are imported, i.e. before MKL loads
# (CLAUDE.md 9.11/9.17: the threaded layer's delay-loaded DLL crashes natively).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks

from epileptic_by_area_animal_day_pixels import crop_slices, resolve_boxes, roi_categories

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
pixel_root = repo_root / "pixel_data"
design_path = pixel_root / "experimental_design.csv"
baseline_window_s = 60  # selects pixels_median_dff_<N>s_full.npy
output_dir = repo_root / "outputs" / f"peak_tails_{baseline_window_s}s"

sampling_rate_hz = 10.0
percent_scale = 100.0    # the cache is a ratio; x100 gives percent dF/F
# Deliberately low: 0.1 % dF/F is ~0.1-0.2 robust SD of an ROI trace, so almost
# every local maximum survives and the tail is not pre-selected by the detector.
peak_prominence_pct = 0.1
min_peak_distance_s = None  # None: no refractory period between peaks
tail_percent = 5.0          # the top 5 % of each trace's peaks form its tail
# Peaks in the first seconds of a recording are an onset artifact: with nothing
# skipped, the first 10 s held 2.8x the uniform share of tail peaks (the next
# 10 s 1.3x, flat after that) and 18 % of recordings had their largest
# Cortex_mean peak there (uniform: 3 %). The frames are dropped before find_peaks.
skip_start_s = 20.0

# 260828_PV7/t2 frame 1809 is a field-wide reflectance glitch with a persistent
# GCaMP step after it (CLAUDE.md 9.23/9.27): a fake whole-cortex peak. Excluded.
exclude_recordings = ["260828_PV7/t2"]

# Summary traces derived from the ROIs (the mean of ROI means, not of pixels).
overall_name = "Cortex_mean"
hemisphere_names = {"L": "CortexL_mean", "R": "CortexR_mean"}

# Fixed categorical order for the groups (dataviz palette slots 1-3).
group_order = ["PV", "R", "T"]
group_colors = {"PV": "#2a78d6", "R": "#eb6834", "T": "#1baf7a"}
n_top_recordings = 25  # recordings shown in the ranking figure
figure_dpi = 200

# Index-space values derived from the physical ones above.
min_peak_distance_frames = (None if min_peak_distance_s is None
                            else max(1, round(min_peak_distance_s * sampling_rate_hz)))
tail_quantile = 100.0 - tail_percent
skip_start_frames = round(skip_start_s * sampling_rate_hz)  # first frame analysed

output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loading: the design table and the recordings to analyse
# ---------------------------------------------------------------------------
design = pd.read_csv(design_path, dtype={"day": str})
if design["pixel_dir"].duplicated().any():
    raise ValueError(f"Duplicate pixel_dir in {design_path}")
cache_name = f"pixels_median_dff_{baseline_window_s}s_full.npy"
is_excluded = design["pixel_dir"].isin(exclude_recordings)
has_cache = design["pixel_dir"].map(lambda pixel_dir: (pixel_root / pixel_dir / cache_name).is_file())
if (~has_cache & ~is_excluded).any():
    missing = design.loc[~has_cache & ~is_excluded, "pixel_dir"].tolist()
    raise FileNotFoundError(f"No {cache_name} for {len(missing)} recordings: {missing[:5]} ...")
recordings = design[~is_excluded].reset_index(drop=True)
n_recordings = len(recordings)
print(f"{n_recordings} recordings to analyse ({int(is_excluded.sum())} excluded: {exclude_recordings})",
      flush=True)

# ---------------------------------------------------------------------------
# Computation: ROI traces from the cached volumes
# ---------------------------------------------------------------------------
# For each recording, the mean of the cached dF/F inside every atlas box, in
# percent. The box list must be the same in every recording so the traces stack.
roi_names = None
trace_blocks = []  # one (n_frames, n_traces) array per recording
start_time = time.time()
for i_recording, recording in recordings.iterrows():
    folder = pixel_root / recording["pixel_dir"]
    with np.load(folder / "pixels_meta_full.npz", allow_pickle=False) as meta_file:
        meta = {key: meta_file[key].tolist() for key in meta_file.files if key not in ("mean_f", "mean_r")}
    cache_meta = json.loads((folder / f"pixels_median_dff_{baseline_window_s}s_meta.json").read_text())
    if cache_meta["window_s"] != baseline_window_s or cache_meta["sampling_rate_hz"] != sampling_rate_hz:
        raise ValueError(f"Cache marker disagrees with the parameters: {folder}")
    atlas_path = Path(meta["roi_set"])
    atlas_path = atlas_path if atlas_path.is_absolute() else repo_root / atlas_path
    atlas = yaml.safe_load(atlas_path.read_text(encoding="utf-8"))
    for key in ("bregma_row", "bregma_col"):
        if atlas[key] != meta[key]:
            raise ValueError(f"Atlas {key} differs from the dump's metadata: {atlas_path}")
    boxes = resolve_boxes(atlas, all_rois=True)  # the atlas as drawn, no custom boxes
    if roi_names is None:
        roi_names = list(boxes)
    elif list(boxes) != roi_names:
        raise ValueError(f"ROI list differs from the first recording's: {atlas_path}")

    volume = np.load(folder / cache_name)  # (n_frames, n_rows, n_cols) float16
    if list(volume.shape) != list(meta["shape"]):
        raise ValueError(f"Cache shape {volume.shape} differs from the dump's {meta['shape']}: {folder}")
    # roi_traces: (n_frames, n_rois); float64 before reducing a float16 volume (9.16).
    roi_traces = np.empty((volume.shape[0], len(roi_names)))
    for i_roi, name in enumerate(roi_names):
        row_slice, col_slice = crop_slices(boxes[name], int(meta["y_1"]), int(meta["x_2"]), meta["region"])
        roi_traces[:, i_roi] = volume[:, row_slice, col_slice].astype(np.float64).mean(axis=(1, 2))
    trace_blocks.append(roi_traces * percent_scale)
    if (i_recording + 1) % 50 == 0 or i_recording + 1 == n_recordings:
        print(f"  [{i_recording + 1}/{n_recordings}] traces read, {time.time() - start_time:.0f} s", flush=True)

# Area and hemisphere of each ROI, then the three summary traces appended.
categories = roi_categories(roi_names)
left_index = [i_roi for i_roi, name in enumerate(roi_names) if categories[name][1] == "L"]
right_index = [i_roi for i_roi, name in enumerate(roi_names) if categories[name][1] == "R"]
trace_names = roi_names + [overall_name, hemisphere_names["L"], hemisphere_names["R"]]
n_traces = len(trace_names)
n_frames = trace_blocks[0].shape[0]
if any(block.shape[0] != n_frames for block in trace_blocks):
    raise ValueError("Recordings differ in frame count; cannot stack the traces")
# traces: (n_recordings, n_frames, n_traces), percent dF/F
roi_stack = np.stack(trace_blocks)  # (n_recordings, n_frames, n_rois)
traces = np.concatenate([
    roi_stack,
    roi_stack.mean(axis=2, keepdims=True),
    roi_stack[:, :, left_index].mean(axis=2, keepdims=True),
    roi_stack[:, :, right_index].mean(axis=2, keepdims=True),
], axis=2)
recorded_min = (n_frames - skip_start_frames) / sampling_rate_hz / 60.0  # the analysed span
np.savez_compressed(output_dir / "roi_traces.npz", traces=traces.astype(np.float32),
                    trace_names=np.array(trace_names), recording_id=recordings["recording_id"].to_numpy(),
                    unit="percent dF/F, 60 s running-median baseline", sampling_rate_hz=sampling_rate_hz)

# ---------------------------------------------------------------------------
# Computation: peaks and their tails
# ---------------------------------------------------------------------------
# Every peak of every trace, then the top `tail_percent` of them by height.
# summary_rows: one per recording x trace; tail_rows: one per tail peak;
# all_heights: every peak height, kept for the group survival functions.
design_columns = ["recording_id", "day", "animal", "group", "mouse_line", "day_index",
                  "animal_day", "recording", "pixel_dir"]
summary_rows = []
tail_rows = []
all_heights = []  # (trace_name, group, height_pct, height_z) arrays
for i_recording, recording in recordings.iterrows():
    for i_trace, trace_name in enumerate(trace_names):
        trace = traces[i_recording, skip_start_frames:, i_trace]  # (n_frames - skip_start_frames,)
        trace_median = np.median(trace)
        noise_sd = 1.4826 * np.median(np.abs(trace - trace_median))  # robust SD of the trace
        peak_positions, peak_properties = find_peaks(trace, prominence=peak_prominence_pct,
                                                     distance=min_peak_distance_frames)
        heights = trace[peak_positions]                    # (n_peaks,) percent dF/F
        peak_frames = peak_positions + skip_start_frames   # frame numbers of the full recording
        heights_z = (heights - trace_median) / noise_sd    # (n_peaks,) robust z
        prominences = peak_properties["prominences"]       # (n_peaks,) percent dF/F
        tail_threshold = np.percentile(heights, tail_quantile)
        in_tail = heights >= tail_threshold
        all_heights.append((trace_name, recording["group"], heights, heights_z))

        summary_rows.append({
            **{column: recording[column] for column in design_columns},
            "trace": trace_name,
            "area": categories[trace_name][0] if trace_name in categories else "summary",
            "hemisphere": categories[trace_name][1] if trace_name in categories else
                          {overall_name: "both", hemisphere_names["L"]: "L", hemisphere_names["R"]: "R"}[trace_name],
            "n_peaks": len(peak_frames),
            "peaks_per_min": len(peak_frames) / recorded_min,
            "n_tail_peaks": int(in_tail.sum()),
            "trace_median_pct": trace_median,
            "noise_sd_pct": noise_sd,
            "height_median_pct": np.median(heights),
            "tail_threshold_pct": tail_threshold,
            "tail_mean_pct": heights[in_tail].mean(),
            "max_height_pct": heights.max(),
            "height_median_z": np.median(heights_z),
            "tail_threshold_z": (tail_threshold - trace_median) / noise_sd,
            "tail_mean_z": heights_z[in_tail].mean(),
            "max_height_z": heights_z.max(),
            "tail_prominence_median_pct": np.median(prominences[in_tail]),
            "frame_of_max": int(peak_frames[np.argmax(heights)]),
        })
        tail_rows.append(pd.DataFrame({
            "recording_id": recording["recording_id"],
            "group": recording["group"],
            "animal": recording["animal"],
            "day_index": recording["day_index"],
            "trace": trace_name,
            "frame": peak_frames[in_tail],
            "time_s": peak_frames[in_tail] / sampling_rate_hz,
            "height_pct": heights[in_tail],
            "height_z": heights_z[in_tail],
            "prominence_pct": prominences[in_tail],
            "tail_threshold_pct": tail_threshold,
        }))
summary = pd.DataFrame(summary_rows)
tail_peaks = pd.concat(tail_rows, ignore_index=True)
if (summary["n_peaks"] < 100).any():
    raise ValueError("A trace has fewer than 100 peaks; a 5 % tail of it is not a distribution")

# Aggregates: summary traces averaged over the t# of a day (one value per animal
# and day), and per group x trace medians over recordings.
metric_columns = ["n_peaks", "noise_sd_pct", "tail_threshold_pct", "tail_mean_pct", "max_height_pct",
                  "tail_threshold_z", "tail_mean_z", "max_height_z"]
summary_traces = summary[summary["area"] == "summary"]
by_animal_day = summary_traces.groupby(["trace", "group", "animal", "day", "day_index"], as_index=False).agg(
    n_recordings=("recording_id", "size"), **{column: (column, "mean") for column in metric_columns})
by_group_trace = summary.groupby(["group", "trace", "area", "hemisphere"], as_index=False).agg(
    n_recordings=("recording_id", "size"), n_animals=("animal", "nunique"),
    **{column: (column, "median") for column in metric_columns})
ranking = (summary[summary["trace"] == overall_name]
           .sort_values("tail_mean_z", ascending=False).reset_index(drop=True))

summary.to_csv(output_dir / "tail_summary.csv", index=False)
tail_peaks.to_csv(output_dir / "tail_peaks.csv", index=False)
by_animal_day.to_csv(output_dir / "tail_by_animal_day.csv", index=False)
by_group_trace.to_csv(output_dir / "tail_by_group_roi.csv", index=False)
ranking.to_csv(output_dir / "tail_ranking_cortex_mean.csv", index=False)
(output_dir / "config.json").write_text(json.dumps(dict(
    baseline_window_s=baseline_window_s, peak_prominence_pct=peak_prominence_pct,
    min_peak_distance_s=min_peak_distance_s, tail_percent=tail_percent, skip_start_s=skip_start_s,
    exclude_recordings=exclude_recordings, n_recordings=n_recordings, n_frames=n_frames,
    sampling_rate_hz=sampling_rate_hz, design=str(design_path)), indent=2), encoding="utf-8")

cortex = summary[summary["trace"] == overall_name]
print(f"{len(summary)} traces, {int(summary['n_peaks'].sum())} peaks, {len(tail_peaks)} in tails "
      f"(prominence >= {peak_prominence_pct:g} % dF/F, top {tail_percent:g} %)")
print(f"{overall_name} per group (median over recordings):")
print(cortex.groupby("group")[["n_peaks", "noise_sd_pct", "tail_threshold_pct", "tail_mean_pct",
                               "tail_threshold_z", "tail_mean_z", "max_height_z"]].median().round(3).to_string())
print(f"wrote {output_dir}")

# ---------------------------------------------------------------------------
# Plot: survival function of peak height per group (the tail itself)
# ---------------------------------------------------------------------------
# P(peak height >= x) over every peak of every recording of a group, for the
# whole-cortex trace; log y so the top 5 % and beyond are readable.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False, constrained_layout=True)
for i_col, (unit_index, unit_label) in enumerate([(2, "peak height (% dF/F)"),
                                                  (3, "peak height (robust z of the trace)")]):
    ax = axes[0, i_col]
    for group in group_order:
        group_heights = np.concatenate([entry[unit_index] for entry in all_heights
                                        if entry[0] == overall_name and entry[1] == group])
        sorted_heights = np.sort(group_heights)
        survival = 1.0 - np.arange(len(sorted_heights)) / len(sorted_heights)  # P(height >= x)
        n_group = int((cortex["group"] == group).sum())
        ax.plot(sorted_heights, survival, color=group_colors[group], linewidth=2,
                label=f"{group} ({n_group} recordings)")
    ax.axhline(tail_percent / 100.0, color="#52514e", linewidth=1, linestyle="--",
               label=f"top {tail_percent:g} %")
    ax.set_yscale("log")
    ax.set_xlabel(unit_label)
    ax.set_ylabel("fraction of peaks >= x")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"{overall_name} peak-height survival by group (find_peaks prominence "
             f">= {peak_prominence_pct:g} % dF/F, {baseline_window_s} s median dF/F)")
fig.savefig(output_dir / "peak_height_ccdf_by_group.png", dpi=figure_dpi)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: per-recording tail metrics by group
# ---------------------------------------------------------------------------
# One dot per recording, jittered; the black bar is the group median. Raw and
# noise-normalised versions side by side, plus the noise itself for reference.
panel_metrics = [("noise_sd_pct", "trace robust SD (% dF/F)"),
                 ("tail_threshold_pct", f"top-{tail_percent:g} % threshold (% dF/F)"),
                 ("tail_mean_pct", f"top-{tail_percent:g} % mean (% dF/F)"),
                 ("tail_threshold_z", f"top-{tail_percent:g} % threshold (z)"),
                 ("tail_mean_z", f"top-{tail_percent:g} % mean (z)"),
                 ("max_height_z", "largest peak (z)")]
jitter_generator = np.random.default_rng(0)  # fixed seed: the figure is reproducible
fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False, constrained_layout=True)
for i_panel, (column, label) in enumerate(panel_metrics):
    ax = axes[i_panel // 3, i_panel % 3]
    for i_group, group in enumerate(group_order):
        values = cortex.loc[cortex["group"] == group, column].to_numpy()
        jitter = jitter_generator.uniform(-0.25, 0.25, len(values))
        ax.scatter(i_group + jitter, values, s=10, color=group_colors[group], alpha=0.5, linewidths=0)
        ax.hlines(np.median(values), i_group - 0.35, i_group + 0.35, color="#0b0b0b", linewidth=2)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels([f"{group}\n(n={int((cortex['group'] == group).sum())})" for group in group_order])
    ax.set_ylabel(label)
    ax.grid(alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"{overall_name}: peak tails per recording by group (dot = recording, bar = median; "
             f"prominence >= {peak_prominence_pct:g} %, top {tail_percent:g} %)")
fig.savefig(output_dir / "tail_by_group.png", dpi=figure_dpi)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: tail across recording days, one line per animal
# ---------------------------------------------------------------------------
cortex_days = by_animal_day[by_animal_day["trace"] == overall_name]
fig, axes = plt.subplots(2, len(group_order), figsize=(15, 7.5), sharey="row", squeeze=False,
                         constrained_layout=True)
for i_row, (column, label) in enumerate([("tail_mean_pct", f"top-{tail_percent:g} % mean (% dF/F)"),
                                         ("tail_mean_z", f"top-{tail_percent:g} % mean (z)")]):
    for i_col, group in enumerate(group_order):
        ax = axes[i_row, i_col]
        group_days = cortex_days[cortex_days["group"] == group]
        for animal, animal_days in group_days.groupby("animal"):
            animal_days = animal_days.sort_values("day_index")
            ax.plot(animal_days["day_index"], animal_days[column], marker="o", markersize=4,
                    color=group_colors[group], alpha=0.6, linewidth=1.2)
            last = animal_days.iloc[-1]
            ax.annotate(animal, (last["day_index"], last[column]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color="#52514e", va="center")
        ax.set_title(f"{group} ({group_days['animal'].nunique()} animals)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # days are whole
        ax.set_xlabel("recording day index")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"{overall_name}: peak tail per animal across days (mean over the t# of each day)")
fig.savefig(output_dir / "tail_by_day.png", dpi=figure_dpi)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: tail per ROI x group
# ---------------------------------------------------------------------------
# Median over recordings of each ROI's top-tail mean in z; ROIs ordered by area
# with the two hemispheres adjacent, summary traces last.
roi_order = sorted(roi_names, key=lambda name: categories[name]) + trace_names[len(roi_names):]
heatmap = (by_group_trace.pivot(index="trace", columns="group", values="tail_mean_z")
           .loc[roi_order, group_order])  # (n_traces, n_groups)
fig, axes = plt.subplots(1, 1, figsize=(5.5, 0.3 * n_traces + 1.8), squeeze=False, constrained_layout=True)
ax = axes[0, 0]
image = ax.imshow(heatmap.to_numpy(), cmap="Blues", aspect="auto")
for i_row in range(heatmap.shape[0]):
    for i_col in range(heatmap.shape[1]):
        value = heatmap.iat[i_row, i_col]
        # White only on the darkest cells, where black ink would lose contrast.
        text_color = "white" if image.norm(value) > 0.6 else "#0b0b0b"
        ax.text(i_col, i_row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)
ax.set_xticks(range(len(group_order)))
ax.set_xticklabels(group_order)
ax.set_yticks(range(len(roi_order)))
ax.set_yticklabels(roi_order, fontsize=7)
fig.colorbar(image, ax=ax, label=f"median top-{tail_percent:g} % mean (z)")
ax.set_title(f"Peak tail by ROI and group\n(median over recordings, top {tail_percent:g} %)", fontsize=10)
fig.savefig(output_dir / "tail_by_roi_heatmap.png", dpi=figure_dpi)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: the recordings with the heaviest tails
# ---------------------------------------------------------------------------
top = ranking.head(n_top_recordings).iloc[::-1]  # reversed so the heaviest is on top
fig, axes = plt.subplots(1, 1, figsize=(8, 0.28 * len(top) + 1.5), squeeze=False, constrained_layout=True)
ax = axes[0, 0]
ax.barh(range(len(top)), top["tail_mean_z"], color=[group_colors[group] for group in top["group"]],
        height=0.7)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{row.recording_id} ({row.group})" for row in top.itertuples()], fontsize=7)
ax.set_xlabel(f"{overall_name} top-{tail_percent:g} % mean (robust z)")
handles = [plt.Rectangle((0, 0), 1, 1, color=group_colors[group]) for group in group_order]
ax.legend(handles, group_order, frameon=False, fontsize=8, loc="lower right")
ax.grid(alpha=0.2, axis="x")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"{n_top_recordings} recordings with the heaviest {overall_name} peak tails "
             f"(of {n_recordings})", fontsize=10)
fig.savefig(output_dir / "top_tail_recordings.png", dpi=figure_dpi)
plt.close(fig)
print("figures written")
