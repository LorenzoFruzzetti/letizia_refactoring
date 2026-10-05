"""Tails of the peak-height distribution with the GCaMP signal expressed as a
fraction of the reflectance (emo) signal, instead of as dF/F.

The counterpart of analyze_peak_tails.py. Same cohort, same detector, same top
`tail_percent` definition; only the signal differs, and `signal_mode` picks how
much of a baseline it gets:

    analyze_peak_tails.py    (F/Fbar)/(R/Rbar) - 1   per-pixel 60 s median, then the
                                                     detector high-passes again (9.22)
    this script, "raw"        F / R                  no baseline at all
    this script, "detrended"  F / R                  ONE high-pass, on the ROI mean

Per saved pixel the raw GCaMP counts are divided by the raw reflectance counts
and the ratio is averaged inside each of the 22 atlas ROI boxes (plus the three
summary traces). `signal_mode` then says what is done with that ratio trace:

    "raw"        100 * (F/R)(t) / median_t(F/R) - 100
                 A rescale by one constant. The ratio LEVEL is arbitrary -- it
                 depends on dye, illumination and camera gain and runs 0.40-1.71
                 across recordings, so raw heights are not comparable and the
                 detector would not have the same sensitivity everywhere -- and
                 dividing it out is all this does. No slow trend is removed.

    "detrended"  100 * (F/R)(t) / running_median_{detrend_window_s}(F/R)(t) - 100
                 The same ratio high-passed ONCE, on the ROI-mean trace. The
                 middle option: the reflectance normalisation of "raw" with the
                 trend taken out, but not the two stages the median dF/F path
                 applies (per pixel, then again in the detector).

    *_pct   percent of the trace's own baseline (its median for "raw", its running
            median for "detrended");
    *_z     robust z of the trace itself, (height - median) / (1.4826 * MAD).

WHY "detrended" EXISTS. In "raw" mode the slow trend is not a caveat, it is the
dominant term: the ratio decays (bleaching) in 96.3 % of recordings, a median
fall of 2.54 % of the median ratio from the first to the last 10 s of the
analysed span, and its 60 s baseline span is a median 2.03 noise SDs against 0.54
for the same traces taken as median dF/F. Heights are measured from the
WHOLE-recording median, so the early part of a recording sits above it and the
late part below, and the top 5 % is largely a selection of early frames: 4.1x
the uniform share of Cortex_mean tail peaks falls in the first tenth of the
analysed span, decaying monotonically to 0.3x in the last. `drift_ptp_pct` is
the span of each trace's own 60 s running median in either mode -- the trend in
"raw", what the high-pass left behind in "detrended". See tail_time_course.png.

WHAT "detrended" ANSWERS. It lands on the median dF/F run: per recording,
Spearman 0.99 on the noise, the tail mean in % and in z and the largest peak,
median ratios 0.96-0.99, 21 of the 25 heaviest recordings shared, and the same
eight recordings above z 10. So the second high-pass in the dF/F path (9.22)
changes the tails hardly at all, and neither does correcting per pixel rather
than on the ROI mean; what separates "raw" from both is the trend alone.

Input:  pixel_data/experimental_design.csv
        pixel_data/<day>_<animal>/<t#>/pixels_f_gcamp_full.npy  (time, y, x) float32
        pixel_data/<day>_<animal>/<t#>/pixels_f_emo_full.npy    (time, y, x) float32
        pixel_data/<day>_<animal>/<t#>/pixels_meta_full.npz     crop + ROI set
        roi_sets/rebuilt/<day>_<animal>_<t#>.yaml                the ROI boxes
Output: outputs/peak_tails_emo_ratio/            ("raw")
        outputs/peak_tails_emo_ratio_detrended<N>s/  ("detrended"), so the two modes
        never overwrite each other's tables (CLAUDE.md 9.22)
          tail_summary.csv                one row per recording x trace
          tail_peaks.csv                  every peak in a tail, one row each
          tail_by_animal_day.csv          summary traces averaged over the t# of a day
          tail_by_group_roi.csv           per group x trace: medians over recordings
          tail_ranking_cortex_mean.csv    recordings ordered by Cortex_mean tail (z)
          tail_vs_<run>.csv               this mode against another run, per recording
          peak_height_ccdf_by_group.png   survival function of peak height, per group
          tail_by_group.png               per-recording tail metrics by group
          tail_by_day.png                 tail per animal across recording days
          tail_by_roi_heatmap.png         tail per ROI x group
          top_tail_recordings.png         the recordings with the heaviest tails
          tail_time_course.png            the trend, and when the tail peaks fall
          tail_vs_<run>.png               this mode against another run

Reading the two raw F volumes is 158 MB per recording and the disk saturates at
~36 MB/s here (8 parallel readers are no faster than 1), so the first pass takes
about 40 min. The ratio traces DO NOT DEPEND ON THE MODE, so both modes share one
cache, `trace_cache_path`, and only the first run of either pays that cost; with
`reuse_cached_traces` on, a run is about a minute. The cache is float32, so a
reused run is not bit-identical to the first pass: it moved 6 peaks of
11,253,628 (5e-7), none of them in a tail metric's third digit.

Run it (the env's own interpreter dies on the first BLAS call, CLAUDE.md 9.17):
    conda run --no-capture-output -n letizia python -u analyze_peak_tails_emo.py
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
# "raw": the ratio rescaled by one constant, no trend removed.
# "detrended": the same ratio divided by its own running median, one high-pass.
signal_mode = "detrended"
detrend_window_s = 60.0  # the running-median window of "detrended" mode
output_dir = repo_root / "outputs" / ("peak_tails_emo_ratio" if signal_mode == "raw"
                                      else f"peak_tails_emo_ratio_detrended{detrend_window_s:g}s")

gcamp_name = "pixels_f_gcamp_full.npy"
emo_name = "pixels_f_emo_full.npy"
# Reading 158 MB per recording takes ~40 min for the cohort; the ratio traces are
# the same in both modes, so the cache is shared and lives outside `output_dir`.
trace_cache_path = repo_root / "outputs" / "peak_tails_emo_ratio" / "roi_traces.npz"
reuse_cached_traces = True
# Runs this one is compared against, per recording: label -> output directory.
# An entry that is this run itself, or has no tail_summary.csv yet, is skipped.
reference_runs = {
    "median_dff": repo_root / "outputs" / "peak_tails_60s",
    "emo_raw": repo_root / "outputs" / "peak_tails_emo_ratio",
    "emo_detrended": repo_root / "outputs" / f"peak_tails_emo_ratio_detrended{detrend_window_s:g}s",
}

sampling_rate_hz = 10.0
# Deliberately low: 0.1 % of the median ratio is ~0.1 robust SD of an ROI trace,
# so almost every local maximum survives and the tail is not pre-selected by the
# detector. Same number as analyze_peak_tails.py, and it yields the same ~730-870
# peaks per trace, so the two signals are compared at equal detector sensitivity.
peak_prominence_pct = 0.1
min_peak_distance_s = None  # None: no refractory period between peaks
tail_percent = 5.0          # the top 5 % of each trace's peaks form its tail
# Peaks in the first seconds of a recording are an onset artifact (measured on
# the dF/F run: the first 10 s held 2.8x the uniform share of tail peaks and 18 %
# of recordings had their largest Cortex_mean peak there, uniform 3 %).
skip_start_s = 20.0
# The window of the drift REPORT (not of the detrend): the peak-to-peak span of
# the trace's own centred running median, in whichever mode is running.
drift_window_s = 60.0

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
if signal_mode not in ("raw", "detrended"):
    raise ValueError(f"signal_mode must be 'raw' or 'detrended', not {signal_mode!r}")
min_peak_distance_frames = (None if min_peak_distance_s is None
                            else max(1, round(min_peak_distance_s * sampling_rate_hz)))
tail_quantile = 100.0 - tail_percent
skip_start_frames = round(skip_start_s * sampling_rate_hz)  # first frame analysed
drift_window_frames = round(drift_window_s * sampling_rate_hz) + 1    # odd: centred
detrend_window_frames = round(detrend_window_s * sampling_rate_hz) + 1

# How the signal is named on the figures, so a panel cannot be read as the wrong mode.
signal_unit = ("% of the median F/R ratio" if signal_mode == "raw"
               else f"% of the {detrend_window_s:g} s running median F/R")
signal_label = ("F/R ratio, no baseline" if signal_mode == "raw"
                else f"F/R ratio, {detrend_window_s:g} s high-pass")

output_dir.mkdir(parents=True, exist_ok=True)
trace_cache_path.parent.mkdir(parents=True, exist_ok=True)
print(f"signal_mode = {signal_mode} ({signal_label})", flush=True)

# ---------------------------------------------------------------------------
# Loading: the design table and the recordings to analyse
# ---------------------------------------------------------------------------
design = pd.read_csv(design_path, dtype={"day": str})
if design["pixel_dir"].duplicated().any():
    raise ValueError(f"Duplicate pixel_dir in {design_path}")
is_excluded = design["pixel_dir"].isin(exclude_recordings)
has_dumps = design["pixel_dir"].map(
    lambda pixel_dir: (pixel_root / pixel_dir / gcamp_name).is_file()
    and (pixel_root / pixel_dir / emo_name).is_file())
if (~has_dumps & ~is_excluded).any():
    missing = design.loc[~has_dumps & ~is_excluded, "pixel_dir"].tolist()
    raise FileNotFoundError(f"No raw F dumps for {len(missing)} recordings: {missing[:5]} ...")
recordings = design[~is_excluded].reset_index(drop=True)
n_recordings = len(recordings)
print(f"{n_recordings} recordings to analyse ({int(is_excluded.sum())} excluded: {exclude_recordings})",
      flush=True)

# ---------------------------------------------------------------------------
# Computation: ROI traces of the raw F/R ratio
# ---------------------------------------------------------------------------
# For each recording, the mean over each atlas box of the per-pixel GCaMP /
# reflectance ratio. The box list must be the same in every recording so the
# traces stack. Cached at `trace_cache_path`, which both modes share, because
# reading the raw volumes dominates the runtime and the ratio is mode-independent.
cached = None
if reuse_cached_traces and trace_cache_path.is_file():
    with np.load(trace_cache_path, allow_pickle=False) as cache_file:
        cached_ids = [str(value) for value in cache_file["recording_id"]]
        if cached_ids == recordings["recording_id"].tolist():
            cached = {"traces": cache_file["traces"].astype(np.float64),
                      "trace_names": [str(value) for value in cache_file["trace_names"]]}
            print(f"reusing the cached ratio traces in {trace_cache_path}", flush=True)

if cached is None:
    roi_names = None
    trace_blocks = []  # one (n_frames, n_rois) array of ratios per recording
    start_time = time.time()
    for i_recording, recording in recordings.iterrows():
        folder = pixel_root / recording["pixel_dir"]
        with np.load(folder / "pixels_meta_full.npz", allow_pickle=False) as meta_file:
            meta = {key: meta_file[key].tolist() for key in meta_file.files
                    if key not in ("mean_f", "mean_r")}
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

        gcamp = np.load(folder / gcamp_name)  # (n_frames, n_rows, n_cols) float32 raw counts
        emo = np.load(folder / emo_name)      # (n_frames, n_rows, n_cols) float32 raw counts
        if gcamp.shape != emo.shape or list(gcamp.shape) != list(meta["shape"]):
            raise ValueError(f"Dump shapes {gcamp.shape}/{emo.shape} differ from {meta['shape']}: {folder}")
        if not (emo > 0).all():
            raise ValueError(f"Reflectance reaches 0 and cannot be divided by: {folder}")
        # roi_traces: (n_frames, n_rois); float64 before dividing and reducing (9.16).
        roi_traces = np.empty((gcamp.shape[0], len(roi_names)))
        for i_roi, name in enumerate(roi_names):
            row_slice, col_slice = crop_slices(boxes[name], int(meta["y_1"]), int(meta["x_2"]), meta["region"])
            box_ratio = (gcamp[:, row_slice, col_slice].astype(np.float64)
                         / emo[:, row_slice, col_slice].astype(np.float64))
            roi_traces[:, i_roi] = box_ratio.mean(axis=(1, 2))  # per-pixel ratio, then space
        trace_blocks.append(roi_traces)
        if (i_recording + 1) % 25 == 0 or i_recording + 1 == n_recordings:
            elapsed = time.time() - start_time
            print(f"  [{i_recording + 1}/{n_recordings}] traces read, {elapsed / 60:.1f} min "
                  f"(~{elapsed / (i_recording + 1) * (n_recordings - i_recording - 1) / 60:.0f} min left)",
                  flush=True)

    # Area and hemisphere of each ROI, then the three summary traces appended.
    categories = roi_categories(roi_names)
    trace_names = roi_names + [overall_name, hemisphere_names["L"], hemisphere_names["R"]]
    n_frames = trace_blocks[0].shape[0]
    if any(block.shape[0] != n_frames for block in trace_blocks):
        raise ValueError("Recordings differ in frame count; cannot stack the traces")
    left_index = [i_roi for i_roi, name in enumerate(roi_names) if categories[name][1] == "L"]
    right_index = [i_roi for i_roi, name in enumerate(roi_names) if categories[name][1] == "R"]
    roi_stack = np.stack(trace_blocks)  # (n_recordings, n_frames, n_rois)
    # traces: (n_recordings, n_frames, n_traces), the raw F/R ratio
    traces = np.concatenate([
        roi_stack,
        roi_stack.mean(axis=2, keepdims=True),
        roi_stack[:, :, left_index].mean(axis=2, keepdims=True),
        roi_stack[:, :, right_index].mean(axis=2, keepdims=True),
    ], axis=2)
    np.savez_compressed(trace_cache_path, traces=traces.astype(np.float32),
                        trace_names=np.array(trace_names),
                        # .astype(str): a pandas object array would need pickle to read back
                        recording_id=recordings["recording_id"].to_numpy().astype(str),
                        unit="F_gcamp / F_emo, raw ratio, no baseline",
                        sampling_rate_hz=sampling_rate_hz)
else:
    traces = cached["traces"]
    trace_names = cached["trace_names"]
    roi_names = [name for name in trace_names if name not in (overall_name, *hemisphere_names.values())]
    n_frames = traces.shape[1]

categories = roi_categories(roi_names)
n_traces = len(trace_names)
recorded_min = (n_frames - skip_start_frames) / sampling_rate_hz / 60.0  # the analysed span

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
start_time = time.time()
for i_recording, recording in recordings.iterrows():
    for i_trace, trace_name in enumerate(trace_names):
        ratio_full = traces[i_recording, :, i_trace]                    # (n_frames,) F/R
        ratio_median = np.median(ratio_full[skip_start_frames:])
        # The signal, in percent of its own baseline. The baseline is computed on
        # the FULL trace so the first analysed frame gets a whole window, and only
        # then are the onset frames dropped.
        if signal_mode == "raw":
            signal_full = 100.0 * ratio_full / ratio_median - 100.0     # one constant, no trend removed
        else:
            ratio_baseline = (pd.Series(ratio_full)
                              .rolling(detrend_window_frames, center=True, min_periods=1)
                              .median().to_numpy())                     # (n_frames,) the single high-pass
            signal_full = 100.0 * ratio_full / ratio_baseline - 100.0
        trace = signal_full[skip_start_frames:]                         # (n_analysed,)
        trace_median = np.median(trace)
        noise_sd = 1.4826 * np.median(np.abs(trace - trace_median))     # robust SD of the trace
        # What a `drift_window_s` median baseline would still take out of it.
        drift = pd.Series(trace).rolling(drift_window_frames, center=True, min_periods=1).median().to_numpy()

        peak_positions, peak_properties = find_peaks(trace, prominence=peak_prominence_pct,
                                                     distance=min_peak_distance_frames)
        heights = trace[peak_positions]                    # (n_peaks,) % of the trace's baseline
        peak_frames = peak_positions + skip_start_frames   # frame numbers of the full recording
        heights_z = (heights - trace_median) / noise_sd    # (n_peaks,) robust z
        prominences = peak_properties["prominences"]       # (n_peaks,) % of the trace's baseline
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
            "ratio_median": ratio_median,
            "trace_median_pct": trace_median,
            "noise_sd_pct": noise_sd,
            "drift_ptp_pct": np.ptp(drift),
            "drift_ptp_z": np.ptp(drift) / noise_sd,
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
    if (i_recording + 1) % 100 == 0 or i_recording + 1 == n_recordings:
        print(f"  [{i_recording + 1}/{n_recordings}] peaks found, {time.time() - start_time:.0f} s", flush=True)
summary = pd.DataFrame(summary_rows)
tail_peaks = pd.concat(tail_rows, ignore_index=True)
if (summary["n_peaks"] < 100).any():
    raise ValueError("A trace has fewer than 100 peaks; a 5 % tail of it is not a distribution")

# Aggregates: summary traces averaged over the t# of a day (one value per animal
# and day), and per group x trace medians over recordings.
metric_columns = ["n_peaks", "noise_sd_pct", "drift_ptp_pct", "drift_ptp_z", "tail_threshold_pct",
                  "tail_mean_pct", "max_height_pct", "tail_threshold_z", "tail_mean_z", "max_height_z"]
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
    signal=("F_gcamp / F_emo, percent of the recording's own median, no baseline" if signal_mode == "raw"
            else f"F_gcamp / F_emo, percent of its own {detrend_window_s:g} s running median"),
    signal_mode=signal_mode, detrend_window_s=detrend_window_s,
    peak_prominence_pct=peak_prominence_pct, min_peak_distance_s=min_peak_distance_s,
    tail_percent=tail_percent, skip_start_s=skip_start_s, drift_window_s=drift_window_s,
    exclude_recordings=exclude_recordings, n_recordings=n_recordings, n_frames=n_frames,
    sampling_rate_hz=sampling_rate_hz, design=str(design_path)), indent=2), encoding="utf-8")

cortex = summary[summary["trace"] == overall_name]
print(f"{len(summary)} traces, {int(summary['n_peaks'].sum())} peaks, {len(tail_peaks)} in tails "
      f"(prominence >= {peak_prominence_pct:g} %, {signal_label}, top {tail_percent:g} %)")
print(f"{overall_name} per group (median over recordings):")
print(cortex.groupby("group")[["n_peaks", "noise_sd_pct", "drift_ptp_pct", "drift_ptp_z",
                               "tail_threshold_pct", "tail_mean_pct", "tail_mean_z",
                               "max_height_z"]].median().round(3).to_string())
print(f"wrote {output_dir}")

# ---------------------------------------------------------------------------
# Plot: survival function of peak height per group (the tail itself)
# ---------------------------------------------------------------------------
# P(peak height >= x) over every peak of every recording of a group, for the
# whole-cortex trace; log y so the top 5 % and beyond are readable.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False, constrained_layout=True)
for i_col, (unit_index, unit_label) in enumerate([(2, f"peak height ({signal_unit})"),
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
fig.suptitle(f"{overall_name} peak-height survival by group ({signal_label}; "
             f"find_peaks prominence >= {peak_prominence_pct:g} %)")
fig.savefig(output_dir / "peak_height_ccdf_by_group.png", dpi=figure_dpi)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: per-recording tail metrics by group
# ---------------------------------------------------------------------------
# One dot per recording, jittered; the black bar is the group median. Raw and
# noise-normalised versions, plus the noise and the drift that was NOT removed.
panel_metrics = [("noise_sd_pct", f"trace robust SD ({signal_unit})"),
                 ("drift_ptp_pct", f"{drift_window_s:g} s baseline span ({signal_unit})"),
                 ("tail_threshold_pct", f"top-{tail_percent:g} % threshold (%)"),
                 ("tail_mean_pct", f"top-{tail_percent:g} % mean (%)"),
                 ("drift_ptp_z", f"{drift_window_s:g} s baseline span (z)"),
                 ("tail_threshold_z", f"top-{tail_percent:g} % threshold (z)"),
                 ("tail_mean_z", f"top-{tail_percent:g} % mean (z)"),
                 ("max_height_z", "largest peak (z)")]
jitter_generator = np.random.default_rng(0)  # fixed seed: the figure is reproducible
fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False, constrained_layout=True)
for i_panel, (column, label) in enumerate(panel_metrics):
    ax = axes[i_panel // 4, i_panel % 4]
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
fig.suptitle(f"{overall_name}: peak tails per recording by group, {signal_label} "
             f"(dot = recording, bar = median; prominence >= {peak_prominence_pct:g} %, top {tail_percent:g} %)")
fig.savefig(output_dir / "tail_by_group.png", dpi=figure_dpi)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot: tail across recording days, one line per animal
# ---------------------------------------------------------------------------
cortex_days = by_animal_day[by_animal_day["trace"] == overall_name]
fig, axes = plt.subplots(2, len(group_order), figsize=(15, 7.5), sharey="row", squeeze=False,
                         constrained_layout=True)
for i_row, (column, label) in enumerate([("tail_mean_pct", f"top-{tail_percent:g} % mean ({signal_unit})"),
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
fig.suptitle(f"{overall_name}: peak tail per animal across days, {signal_label} (mean over the t# of each day)")
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
ax.set_title(f"Peak tail by ROI and group, {signal_label}\n(median over recordings, top {tail_percent:g} %)",
             fontsize=10)
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
ax.set_title(f"{n_top_recordings} recordings with the heaviest {overall_name} peak tails, {signal_label} "
             f"(of {n_recordings})", fontsize=10)
fig.savefig(output_dir / "top_tail_recordings.png", dpi=figure_dpi)
plt.close(fig)
print("figures written")

# ---------------------------------------------------------------------------
# Plot: the drift that was left in, and when the tail peaks fall
# ---------------------------------------------------------------------------
# Left: the whole-cortex signal averaged over the recordings of each group -- the
# bleaching decay in "raw" mode, what the high-pass left in "detrended" mode.
# Right: when the tail peaks happen, against the dF/F run's flat distribution.
overall_index = trace_names.index(overall_name)
overall_ratio = traces[:, :, overall_index]  # (n_recordings, n_frames)
if signal_mode == "raw":
    overall_baseline = np.median(overall_ratio[:, skip_start_frames:], axis=1, keepdims=True)
else:
    overall_baseline = (pd.DataFrame(overall_ratio.T)
                        .rolling(detrend_window_frames, center=True, min_periods=1)
                        .median().to_numpy().T)
# overall_signal: (n_recordings, n_analysed), the same signal the peaks were found in
overall_signal = (100.0 * overall_ratio / overall_baseline - 100.0)[:, skip_start_frames:]
analysed_time_s = np.arange(skip_start_frames, n_frames) / sampling_rate_hz
time_bins = np.linspace(skip_start_frames, n_frames, 11)  # ten equal bins of the analysed span
cortex_tail = tail_peaks[tail_peaks["trace"] == overall_name]
tail_histogram, _ = np.histogram(cortex_tail["frame"], time_bins)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), squeeze=False, constrained_layout=True)
ax = axes[0, 0]
for group in group_order:
    in_group = (recordings["group"] == group).to_numpy()
    ax.plot(analysed_time_s, overall_signal[in_group].mean(axis=0), color=group_colors[group],
            linewidth=1.2, label=f"{group} ({int(in_group.sum())} recordings)")
ax.axhline(0.0, color="#52514e", linewidth=1, linestyle="--", label="the trace's baseline")
ax.set_xlabel("time in the recording (s)")
ax.set_ylabel(f"{overall_name} ({signal_unit})")
ax.set_title(f"What the baseline left in ({signal_label}): mean over recordings", fontsize=9)
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
bin_centres = (time_bins[:-1] + time_bins[1:]) / 2 / sampling_rate_hz
bin_width_s = np.diff(time_bins)[0] / sampling_rate_hz
# Neutral grey, not a group colour: this panel pools the groups.
ax.bar(bin_centres, tail_histogram, width=bin_width_s * 0.9, color="#9a9a94",
       label=signal_label)
timing_reference = reference_runs["median_dff"]
if timing_reference != output_dir and (timing_reference / "tail_peaks.csv").is_file():
    reference_peaks = pd.read_csv(timing_reference / "tail_peaks.csv", usecols=["trace", "frame"])
    reference_histogram, _ = np.histogram(
        reference_peaks.loc[reference_peaks["trace"] == overall_name, "frame"], time_bins)
    ax.step(bin_centres, reference_histogram, where="mid", color="#0b0b0b", linewidth=1.5,
            label="60 s median dF/F")
ax.axhline(tail_histogram.sum() / len(tail_histogram), color="#52514e", linewidth=1, linestyle="--",
           label="uniform in time")
ax.set_xlabel("time in the recording (s)")
ax.set_ylabel(f"{overall_name} tail peaks per {bin_width_s:.0f} s bin")
ax.set_title(f"When the top {tail_percent:g} % happens", fontsize=10)
ax.grid(alpha=0.2, axis="y")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.suptitle("Without a baseline the tail selects the start of the recording" if signal_mode == "raw"
             else f"With the {detrend_window_s:g} s high-pass, when the tail peaks fall")
fig.savefig(output_dir / "tail_time_course.png", dpi=figure_dpi)
plt.close(fig)

baseline_start = overall_signal[:, :100].mean(axis=1)   # first 10 s of the analysed span
baseline_end = overall_signal[:, -100:].mean(axis=1)    # last 10 s
print(f"trend: falls over the recording in {(baseline_start > baseline_end).mean():.1%} of recordings, "
      f"median fall {np.median(baseline_start - baseline_end):.2f} %; "
      f"tail peaks in the first tenth of the span: {tail_histogram[0] / tail_histogram.mean():.1f}x uniform")

# ---------------------------------------------------------------------------
# Comparison: the same recordings under the other runs
# ---------------------------------------------------------------------------
# Only the signal differs between the runs, so a per-recording scatter says
# whether the baseline changes the answer or only its units. One figure and one
# table per reference, named after it.
compared_metrics = [("noise_sd_pct", "trace robust SD (%)"),
                    ("tail_mean_pct", f"top-{tail_percent:g} % mean (%)"),
                    ("tail_mean_z", f"top-{tail_percent:g} % mean (z)"),
                    ("max_height_z", "largest peak (z)")]
for reference_label, reference_dir in reference_runs.items():
    if reference_dir == output_dir or not (reference_dir / "tail_summary.csv").is_file():
        continue  # this run itself, or a run that has not been done yet
    reference = pd.read_csv(reference_dir / "tail_summary.csv")
    reference_cortex = reference[reference["trace"] == overall_name]
    comparison = cortex.merge(reference_cortex, on="recording_id", suffixes=("_this", "_other"))
    if len(comparison) != len(cortex):
        raise ValueError(f"{reference_dir.name} covers {len(comparison)} of {len(cortex)} recordings")
    comparison.to_csv(output_dir / f"tail_vs_{reference_label}.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), squeeze=False, constrained_layout=True)
    for i_col, (column, label) in enumerate(compared_metrics):
        ax = axes[0, i_col]
        for group in group_order:
            rows = comparison[comparison["group_this"] == group]
            ax.scatter(rows[f"{column}_other"], rows[f"{column}_this"], s=10, alpha=0.5, linewidths=0,
                       color=group_colors[group], label=f"{group} (n={len(rows)})")
        limits = [min(comparison[f"{column}_other"].min(), comparison[f"{column}_this"].min()),
                  max(comparison[f"{column}_other"].max(), comparison[f"{column}_this"].max())]
        ax.plot(limits, limits, color="#52514e", linewidth=1, linestyle="--", label="equal")
        correlation = comparison[f"{column}_other"].corr(comparison[f"{column}_this"], method="spearman")
        ax.set_title(f"{label}\nSpearman r = {correlation:.2f}", fontsize=9)
        ax.set_xlabel(reference_label)
        ax.set_ylabel(signal_label)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{overall_name} peak tails: {signal_label} against {reference_label}, "
                 f"one dot per recording")
    fig.savefig(output_dir / f"tail_vs_{reference_label}.png", dpi=figure_dpi)
    plt.close(fig)
    print(f"vs {reference_label} ({reference_dir.name}), {overall_name} medians by group:")
    print(comparison.groupby("group_this")[["tail_mean_z_this", "tail_mean_z_other",
                                            "max_height_z_this", "max_height_z_other"]].median().round(3).to_string())
