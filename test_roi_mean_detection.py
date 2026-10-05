"""Epileptiform detection on the ROI means of ONE recording, by area, with the events mapped.

The single-recording counterpart of `epileptic_by_area_animal_day.py`. That script
runs every recording of `outputs/botox_restani_rebuilt` and calibrates one z cut-off
over all their frames; this one takes a single `roi_fluorescence_full.csv` so the
intermediate steps can be looked at, and splits that recording's events by area and
hemisphere the way the cohort tables do.

It imports the cohort script's detection functions rather than copying them, so the
two cannot drift apart. (`test_epileptic_detection.py` is the older single-recording
script and holds its own copies, which have since diverged from the cohort: a
different `rise_window_s`, `rise_prominence_sd` and `rise_lag_s`, no
`rise_on_smoothed` and no `signal_at_peak`. Prefer this file when the question is
"what would the cohort run do to this recording".)

Signal
    The `roi_fluorescence_full.csv` written by run_botox_batch.py: percent dF/F
    averaged inside each ROI box, one column per ROI, one row per frame, trials
    stacked. Any table with `trial`, `frame` and one column per ROI works, so the
    per-recording traces of the pixel detectors can be pointed at too.

    Three summary traces are derived and drawn beside the ROIs - `Cortex_mean`
    (every ROI averaged), `CortexL_mean` and `CortexR_mean` (each hemisphere's).
    They are written to the trace CSV, but events are NOT detected in them, so the
    event tables stay the ones the cohort run would produce.

ONE DELIBERATE DIFFERENCE. The epileptiform cut-off here is the top
`calibration_frame_percent` of THIS recording's frames (or the fixed
`epileptic_z_threshold`, when set). The cohort pools the frames of every recording,
so counts from this script are NOT comparable across recordings - they answer "what
did the detector do on this file", not "how many events does this animal have".

Input:  outputs/botox_restani_rebuilt/<date>_<animal>/<t#>/roi_fluorescence_full.csv
Output: outputs/test_roi_mean_detection/<date>_<animal>/<t#>/
          roi_traces.csv               the loaded ROIs + the three summary traces
          roi_events.csv               every amplitude peak with its flags
          roi_epileptic_events.csv     only the epileptiform ones
          roi_event_counts.csv         per ROI: area, hemisphere, counts, per-minute rate
          roi_event_counts_by_area.csv the same summed over the two hemispheres
          roi_activity_overview.png    overall activity, detrended, per hemisphere
          roi_event_raster.png         when each ROI fired, and how often, by area
          roi_traces_all.png           every ROI overlaid over the whole recording
          roi_traces_detrended.png     each trace vs its running median, detrended
          roi_traces_events.png        the detections on each ROI
          roi_traces_rise.png          rise rate over the smoothed amplitude
          roi_event_zooms.png          the largest events, with every other ROI behind

Run it (the env's own interpreter dies on the first BLAS call, CLAUDE.md 9.17):
    conda run --no-capture-output -n letizia python test_roi_mean_detection.py
"""

import os

# MUST run before numpy/scipy/matplotlib are imported, i.e. before MKL loads.
# MKL's mkl_intel_thread.3.dll delay-loads libiomp5md.dll by bare name; when that
# fails the process dies with native exception 0xc06d007f and no traceback
# (CLAUDE.md 9.11/9.17). The sequential layer never loads that DLL.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epileptic_by_area_animal_day import (detect_events, remove_slow_trend, rise_by_trial,
                                          robust_z_frames, smooth_by_trial)
from epileptic_by_area_animal_day_pixels import roi_categories
from epileptic_diagnostics import plot_detections, plot_detrending, plot_rise, plot_roi_overview

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
csv_path = repo_root / "outputs/botox_restani_rebuilt/260611_PV5/t1/roi_fluorescence_full.csv"
starting_column = 2  # first 2 columns (trial, frame) are metadata, not ROIs
signal_unit = "DF/F (%)"  # run_botox_batch writes percent dF/F; only a label

# Detection - the same values as epileptic_by_area_animal_day.py's module level.
sampling_rate_hz = 10.0      # per-channel rate, after the channels are split
median_window_s = 20.0       # running median subtracted as the slow trend
smooth_window_s = 1.0        # rolling-mean width; the result is the "amplitude"
min_event_distance_s = 0.5   # refractory period between two peaks of one ROI
amplitude_prominence_sd = 3.0  # peak prominence in robust SDs of the ROI's own signal
rise_prominence_sd = 3.0
rise_window_s = 0.5          # interval the rise is measured over
not_linear_rise = True       # True: window max - min; False: end - start
rise_on_smoothed = False     # the cohort default here: rise of the detrended, unsmoothed trace
rise_lead_s = 1.0            # a rise peak this long BEFORE the amplitude peak is concurrent
rise_lag_s = 0.3             # ... and this long after it
epileptic_require_confirmed = True  # epileptiform also needs a concurrent rise

# The epileptiform cut-off in robust z. A number fixes it; None takes the top
# `calibration_frame_percent` of this recording's own frames, which is what makes
# the counts recording-specific (see the docstring). To reproduce a cohort run's
# flags on this recording exactly, set this to that run's `z_threshold`, from
# `outputs/epileptic_by_area_animal_day/detection_config.json` (3.58 for the run
# saved there). Verified on 260611_PV5/t1: the 419 peaks and 220 confirmations
# are already identical to that run's, the cut-off being the only difference -
# and at its own top 1% this quiet recording flags 57 events where the pooled
# cut-off flags 8.
epileptic_z_threshold = None
calibration_frame_percent = 1.0
# An epileptiform peak may also be required to reach this value in the LOADED
# signal (before detrending and smoothing) at the peak frame, in `signal_unit`.
# None is the cohort default for ROI means, where dF/F has no natural floor.
epileptic_min_signal = None

# Column names of the derived summary traces: drawn and saved, never detected in.
OVERALL_NAME = "Cortex_mean"
HEMISPHERE_NAMES = {"L": "CortexL_mean", "R": "CortexR_mean"}

# Figures.
n_event_zooms = 4            # largest events to draw a zoom for
event_zoom_window_s = 10.0   # width of each zoom panel, centred on the peak
figure_dpi = 150
trace_width_scale = 4.0      # stretch of the whole-recording overview and rise figures
overview_width_in = 22.0     # width of the overall-activity figure
overview_row_height_in = 2.4  # height of each of its three rows

# Index-space values derived from the physical ones above.
n_median = 2 * round(median_window_s * sampling_rate_hz / 2) + 1  # odd, so the median is centred
n_smooth = max(1, round(smooth_window_s * sampling_rate_hz))      # frames per rolling mean
edge_frames = n_smooth // 2  # frames at each trial end averaged over a partial window
min_distance_frames = max(1, round(min_event_distance_s * sampling_rate_hz))
n_rise = max(1, round(rise_window_s * sampling_rate_hz))  # frames spanned by one rise value
rise_lead_frames = max(1, round(rise_lead_s * sampling_rate_hz))
rise_lag_frames = max(1, round(rise_lag_s * sampling_rate_hz))
n_zoom_frames = max(1, round(event_zoom_window_s * sampling_rate_hz / 2))  # half-width of a zoom

recording_label = f"{csv_path.parent.parent.name}/{csv_path.parent.name}"
output_dir = repo_root / "outputs" / "test_roi_mean_detection" / csv_path.parent.parent.name / csv_path.parent.name
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# fluorescence: (n_frames, 2 + n_rois), trials stacked one after another
fluorescence = pd.read_csv(csv_path)
roi_names = list(fluorescence.columns[starting_column:])
n_rois = len(roi_names)
n_frames = len(fluorescence)
time_s = fluorescence["frame"].to_numpy(dtype=float) / sampling_rate_hz  # (n_frames,)
recorded_min = n_frames / sampling_rate_hz / 60.0

# Area and hemisphere of every ROI: M2L_alta -> (M2_alta, L), RSL_alta -> (RS_alta, L),
# V1aR -> (V1a, R). The cohort derives the same split, by looking for the L/R whose
# swap names another ROI; the check below is that rule, so a label the two rules
# would disagree on cannot pass silently.
categories = roi_categories(roi_names)
for roi, (area, hemisphere) in categories.items():
    twin = [other for other, (other_area, other_side) in categories.items()
            if other_area == area and other_side != hemisphere]
    if len(twin) != 1:
        raise ValueError(f"ROI {roi!r} ({area}, {hemisphere}) has no single bilateral twin: {twin}")

# ---------------------------------------------------------------------------
# Summary traces
# ---------------------------------------------------------------------------
# Whole-cortex and per-hemisphere activity: the plain mean over the ROIs of that
# side. Every ROI box holds a different number of pixels, so this is the mean of
# the ROI means, not the mean over pixels - the ROIs are what was detected in.
left_rois = [roi for roi, (_, side) in categories.items() if side == "L"]
right_rois = [roi for roi, (_, side) in categories.items() if side == "R"]
traces = fluorescence.copy()
traces[OVERALL_NAME] = fluorescence[roi_names].mean(axis=1)  # (n_frames,)
traces[HEMISPHERE_NAMES["L"]] = fluorescence[left_rois].mean(axis=1)
traces[HEMISPHERE_NAMES["R"]] = fluorescence[right_rois].mean(axis=1)
summary_names = [OVERALL_NAME, *HEMISPHERE_NAMES.values()]
plot_trace_names = roi_names + summary_names  # detection still uses roi_names only

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# Slow trends removed; everything after this works on `detrended`. The summary
# traces go through the same detrend and smoothing so the figures show them
# treated identically, but only `roi_names` reaches the detection, which keeps the
# event tables the ones the cohort script would produce.
# detrended: (n_frames, 2 + n_rois + 3), trend/smoothed: (n_frames, n_rois + 3)
detrended, trend = remove_slow_trend(traces, plot_trace_names, n_median)
smoothed = smooth_by_trial(detrended, plot_trace_names, n_smooth)  # the "amplitude"
# Every frame in robust z within its trial; the cut-off below is a percentile of these.
frame_z = robust_z_frames(detrended, smoothed, roi_names, edge_frames)  # (n_frames, n_rois)

# One row per amplitude peak, `confirmed` when a rise peak falls in its window.
events = detect_events(
    detrended, smoothed, roi_names,
    rise_traces=smoothed if rise_on_smoothed else detrended,
    sampling_rate_hz=sampling_rate_hz,
    min_distance_frames=min_distance_frames,
    amplitude_prominence_sd=amplitude_prominence_sd,
    rise_prominence_sd=rise_prominence_sd,
    n_rise=n_rise,
    rise_lead_frames=rise_lead_frames,
    rise_lag_frames=rise_lag_frames,
    not_linear_rise=not_linear_rise,
)

# The loaded signal at each peak frame, for the optional epileptic_min_signal gate.
# Addressed by (trial, frame) rather than row number, so a multi-trial table works.
loaded = traces.set_index(["trial", "frame"])[roi_names]  # (n_frames, n_rois)
row_position = loaded.index.get_indexer(pd.MultiIndex.from_arrays([events["trial"], events["frame"]]))
if (row_position < 0).any():
    raise ValueError(f"Peak frames missing from the loaded trace: {csv_path}")
roi_position = events["roi"].map({roi: i_roi for i_roi, roi in enumerate(roi_names)}).to_numpy(dtype=int)
events["signal_at_peak"] = loaded.to_numpy(dtype=float)[row_position, roi_position]

# The cut-off, then the flags. A fixed threshold is comparable across recordings;
# the percentile of this recording's own frames is not (see the docstring).
if epileptic_z_threshold is not None:
    z_threshold = float(epileptic_z_threshold)
    cutoff_label = f"z >= {z_threshold:.2f} (fixed)"
else:
    pooled_z = frame_z.to_numpy(dtype=float).ravel()
    pooled_z = pooled_z[np.isfinite(pooled_z)]
    z_threshold = float(np.percentile(pooled_z, 100.0 - calibration_frame_percent))
    cutoff_label = f"z >= {z_threshold:.2f} (top {calibration_frame_percent:g}% of this recording's frames)"

is_epileptic = events["amplitude_z"].to_numpy(dtype=float) >= z_threshold  # NaN compares False
if epileptic_require_confirmed:
    is_epileptic &= events["confirmed"].to_numpy(dtype=bool)
if epileptic_min_signal is not None:
    is_epileptic &= events["signal_at_peak"].to_numpy(dtype=float) >= epileptic_min_signal
events["epileptic"] = is_epileptic
events["epileptic_threshold"] = z_threshold
epileptic_events = events[events["epileptic"]]

# Rise of the same signal the peaks were found in, as a table for the rise figure.
rise_per_s = rise_by_trial(detrended, roi_names, n_rise, sampling_rate_hz, not_linear_rise,
                           rise_traces=smoothed if rise_on_smoothed else detrended)

# ---------------------------------------------------------------------------
# Counts by ROI and by area
# ---------------------------------------------------------------------------
# One row per ROI even when it fired nothing, so a zero means "looked at, found
# nothing" rather than "missing" - the same rule as the cohort's long table.
counted = events.groupby("roi").agg(n_peaks=("frame", "size"), n_confirmed=("confirmed", "sum"),
                                    n_epileptic=("epileptic", "sum"))
roi_counts = pd.DataFrame({
    "roi": roi_names,
    "area": [categories[roi][0] for roi in roi_names],
    "hemisphere": [categories[roi][1] for roi in roi_names],
    "recorded_min": recorded_min,
}).join(counted, on="roi")
count_columns = ["n_peaks", "n_confirmed", "n_epileptic"]
roi_counts[count_columns] = roi_counts[count_columns].fillna(0).astype(np.int64)
roi_counts["epileptic_per_min"] = roi_counts["n_epileptic"] / roi_counts["recorded_min"]

# The same summed over the two hemispheres, so the rate stays per ROI-minute.
area_counts = roi_counts.groupby("area", as_index=False, sort=False).agg(
    n_rois=("roi", "nunique"), roi_min=("recorded_min", "sum"), n_peaks=("n_peaks", "sum"),
    n_confirmed=("n_confirmed", "sum"), n_epileptic=("n_epileptic", "sum"))
area_counts["epileptic_per_roi_min"] = area_counts["n_epileptic"] / area_counts["roi_min"]

# The events drawn as zooms below: the largest epileptiform ones, or the largest
# peaks of any kind when the cut-off flagged nothing.
ranked = (epileptic_events if len(epileptic_events) else events).sort_values(
    "amplitude_z", ascending=False)
zoomed = ranked.head(n_event_zooms)

# ---------------------------------------------------------------------------
# Saving tables
# ---------------------------------------------------------------------------
traces.to_csv(output_dir / "roi_traces.csv", index=False)
events.to_csv(output_dir / "roi_events.csv", index=False)
epileptic_events.to_csv(output_dir / "roi_epileptic_events.csv", index=False)
roi_counts.to_csv(output_dir / "roi_event_counts.csv", index=False)
area_counts.to_csv(output_dir / "roi_event_counts_by_area.csv", index=False)

rise_source = "smoothed amplitude" if rise_on_smoothed else "detrended trace"
print(f"{csv_path}")
print(f"  {n_rois} ROIs in {len(area_counts)} bilateral areas, {n_frames} frames at "
      f"{sampling_rate_hz:g} Hz ({recorded_min:.1f} min)")
print(f"  {median_window_s:g} s median detrend, {smooth_window_s:g} s smoothing, "
      f"rise from the {rise_source} over {rise_window_s:g} s")
for name in summary_names:
    print(f"  {name}: mean {traces[name].mean():.3f}, max {traces[name].max():.3f} {signal_unit}")
print(f"  {len(events)} amplitude peaks, {int(events['confirmed'].sum())} confirmed")
print(f"  {len(epileptic_events)} epileptiform ({cutoff_label})"
      f"{'' if epileptic_min_signal is None else f', min signal {epileptic_min_signal:g}'}")
if len(epileptic_events):
    busiest = roi_counts.sort_values("n_epileptic", ascending=False).iloc[0]
    print(f"  most epileptiform: {busiest['roi']} ({busiest['area']}, {busiest['hemisphere']}) "
          f"with {busiest['n_epileptic']}")
print(f"  wrote {output_dir}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# Overall activity over the whole recording: the ROI-averaged signal as loaded,
# then the same trace after the running median is taken out, then the two
# hemispheres. The epileptiform peaks of any ROI are marked on all three, so a
# whole-cortex rise can be told from a one-sided or single-ROI one.
epileptic_times_s = np.unique(epileptic_events["time_s"].to_numpy())  # (n_event_frames,)
fig, axes = plt.subplots(3, 1, figsize=(overview_width_in, 3 * overview_row_height_in),
                         sharex=True, squeeze=False, constrained_layout=True)
axes[0, 0].plot(time_s, traces[OVERALL_NAME].to_numpy(), color="#7a7f87", linewidth=0.8,
                label=f"mean of {n_rois} ROIs")
axes[0, 0].plot(time_s, trend[OVERALL_NAME].to_numpy(), color="#e08214", linewidth=1.6,
                label=f"{median_window_s:g} s running median")
if epileptic_min_signal is not None:
    # The gate an epileptiform peak must also clear, on this very signal.
    axes[0, 0].axhline(epileptic_min_signal, color="#2f6fd0", linewidth=1.0, linestyle="--",
                       label=f"epileptic_min_signal = {epileptic_min_signal:g}")
axes[0, 0].set_ylabel(f"all ROIs\n{signal_unit}")
axes[0, 0].set_title(f"Overall activity - all {n_rois} ROIs averaged")

axes[1, 0].plot(time_s, detrended[OVERALL_NAME].to_numpy(), color="#c9ccd1", linewidth=0.8,
                label="detrended")
axes[1, 0].plot(time_s, smoothed[OVERALL_NAME].to_numpy(), color="#2f6fd0", linewidth=1.4,
                label=f"smoothed ({smooth_window_s:g} s)")
axes[1, 0].set_ylabel(f"detrended\n{signal_unit}")
axes[1, 0].set_title("The same trace with the running median subtracted")

for side, name in HEMISPHERE_NAMES.items():
    axes[2, 0].plot(time_s, traces[name].to_numpy(), linewidth=0.9,
                    label=f"{name} ({len(left_rois) if side == 'L' else len(right_rois)} ROIs)")
axes[2, 0].set_ylabel(f"per hemisphere\n{signal_unit}")
axes[2, 0].set_title("Each hemisphere on its own, for comparison")
axes[2, 0].set_xlabel("Time (s)")

for ax in axes[:, 0]:
    for event_time_s in epileptic_times_s:
        ax.axvline(event_time_s, color="#c2354a", linewidth=0.8, alpha=0.4, zorder=0)
    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"ROI-mean activity over time - {recording_label} "
             f"(red lines = the {len(epileptic_times_s)} frames with an epileptiform peak "
             f"in any of the {n_rois} ROIs)")
fig.savefig(output_dir / "roi_activity_overview.png", dpi=figure_dpi)
plt.close(fig)
print(f"  wrote {output_dir / 'roi_activity_overview.png'}")

# When each ROI fired and how often, ordered by area with the two hemispheres
# adjacent: a real event crosses many rows at once, a noisy ROI fires on its own.
raster_order = sorted(roi_names, key=lambda roi: (categories[roi][0], categories[roi][1]))
raster_row = {roi: i_row for i_row, roi in enumerate(raster_order)}
fig, axes = plt.subplots(1, 2, figsize=(18, 0.32 * n_rois + 2.2), squeeze=False,
                         constrained_layout=True, width_ratios=[4, 1])
ax = axes[0, 0]
tiers = [(~events["confirmed"], "#d89020", "amplitude peak only", 14),
         (events["confirmed"] & ~events["epileptic"], "#c2354a", "concurrent peak", 16),
         (events["epileptic"], "#c2354a", f"epileptiform ({cutoff_label})", 90)]
for mask, colour, label, size in tiers:
    tier = events[mask]
    marker = "*" if size > 50 else "|"
    ax.scatter(tier["time_s"], tier["roi"].map(raster_row), s=size, marker=marker,
               color=colour, linewidths=1.0, label=label)
ax.set_yticks(range(n_rois))
ax.set_yticklabels(raster_order, fontsize=7)
ax.set_ylim(n_rois - 0.5, -0.5)
ax.set_xlabel("Time (s)")
# Above the axes: inside them the legend would cover the first ROI rows.
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=8)
ax.grid(alpha=0.2, axis="x")
ax = axes[0, 1]
bar_rows = np.arange(n_rois)
counts_by_roi = roi_counts.set_index("roi").loc[raster_order]  # same order as the raster rows
ax.barh(bar_rows, counts_by_roi["n_peaks"], color="#d0d3d8", label="all peaks")
ax.barh(bar_rows, counts_by_roi["n_epileptic"], color="#c2354a", label="epileptiform")
ax.set_yticks(bar_rows)
ax.set_yticklabels([])
ax.set_ylim(n_rois - 0.5, -0.5)
ax.set_xlabel("events")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.grid(alpha=0.2, axis="x")
for ax in axes[0]:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Detections per ROI - {recording_label} ({len(events)} peaks, "
             f"{int(events['confirmed'].sum())} confirmed, {len(epileptic_events)} epileptiform)")
fig.savefig(output_dir / "roi_event_raster.png", dpi=figure_dpi)
plt.close(fig)
print(f"  wrote {output_dir / 'roi_event_raster.png'}")

# The loaded signal itself, before detrending: every ROI over the whole recording
# on one axis, the same look as run_botox_batch's roi_traces_full.png.
plot_roi_overview(
    traces, roi_names, time_s, output_dir / "roi_traces_all.png",
    f"All ROIs - {recording_label} ({n_rois} ROIs, {signal_unit})",
    dpi=figure_dpi, signal_unit=signal_unit, width_scale=trace_width_scale,
)
print(f"  wrote {output_dir / 'roi_traces_all.png'}")

plot_detrending(
    traces, trend, detrended, plot_trace_names, time_s,
    output_dir / "roi_traces_detrended.png",
    f"Slow-trend removal - {recording_label} ({median_window_s:g} s running median; "
    f"the last three rows are the derived summary traces)",
    dpi=figure_dpi,
)
print(f"  wrote {output_dir / 'roi_traces_detrended.png'}")

plot_detections(
    detrended, smoothed, events, roi_names, time_s, cutoff_label,
    output_dir / "roi_traces_events.png",
    f"Epileptiform event detection - {recording_label} ({median_window_s:g} s median detrend, "
    f"{smooth_window_s:g} s smoothing, prominence {amplitude_prominence_sd:g} SD, {cutoff_label})",
    dpi=figure_dpi,
)
print(f"  wrote {output_dir / 'roi_traces_events.png'}")

plot_rise(
    smoothed, rise_per_s, events, roi_names, time_s, n_rise / 2 / sampling_rate_hz,
    output_dir / "roi_traces_rise.png",
    f"Rise rate vs smoothed amplitude - {recording_label} ({rise_window_s:g} s window, "
    f"not_linear_rise={not_linear_rise}, rise_on_smoothed={rise_on_smoothed})",
    dpi=figure_dpi, signal_unit=signal_unit, rise_source=rise_source,
    width_scale=trace_width_scale,
)
print(f"  wrote {output_dir / 'roi_traces_rise.png'}")

# The largest events with every other ROI drawn faintly behind the one that fired,
# so how far an event spread is visible: a seizure-like event lifts most ROIs.
if len(zoomed):
    fig, axes = plt.subplots(1, len(zoomed), figsize=(4.0 * len(zoomed), 4.0),
                             squeeze=False, constrained_layout=True)
    for i_event, (_, event) in enumerate(zoomed.iterrows()):
        trial_rows = np.flatnonzero(detrended["trial"].to_numpy() == event["trial"])
        peak_row = trial_rows[np.searchsorted(detrended["frame"].to_numpy()[trial_rows], event["frame"])]
        first = max(trial_rows[0], peak_row - n_zoom_frames)
        last = min(trial_rows[-1], peak_row + n_zoom_frames) + 1
        ax = axes[0, i_event]
        ax.plot(time_s[first:last], detrended[roi_names].to_numpy()[first:last],
                color="#c9ccd1", linewidth=0.7)
        ax.plot(time_s[first:last], smoothed[event["roi"]].to_numpy()[first:last],
                color="#2f6fd0", linewidth=1.6, label=event["roi"])
        ax.plot(time_s[first:last], smoothed[OVERALL_NAME].to_numpy()[first:last],
                color="#1baf7a", linewidth=1.4, label=OVERALL_NAME)
        ax.axvline(event["time_s"], color="#c2354a", linewidth=1.0)
        # How many OTHER ROIs also flagged an epileptiform peak inside this window.
        in_window = epileptic_events[(epileptic_events["time_s"] >= time_s[first])
                                     & (epileptic_events["time_s"] <= time_s[last - 1])]
        ax.set_title(f"{event['roi']} at {event['time_s']:.1f} s\n"
                     f"z {event['amplitude_z']:.1f}, {in_window['roi'].nunique()} of {n_rois} "
                     f"ROIs epileptiform here")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"detrended {signal_unit}")
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    kind = "epileptiform" if len(epileptic_events) else "amplitude (none flagged epileptiform)"
    fig.suptitle(f"Largest {kind} events - {recording_label} ({event_zoom_window_s:g} s zoom; "
                 f"every other ROI in grey)")
    fig.savefig(output_dir / "roi_event_zooms.png", dpi=figure_dpi)
    plt.close(fig)
    print(f"  wrote {output_dir / 'roi_event_zooms.png'}")
