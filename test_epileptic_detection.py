"""Detect epileptiform events in the per-ROI fluorescence traces of one recording.

Per trial and per ROI:
  1. remove slow trends by subtracting a centred running median (10 s),
  2. smooth the detrended trace with a centred rolling mean (the amplitude),
  3. take the rise rate: the change of the detrended (unsmoothed) trace across
     a short `rise_window_s` window,
  4. find peaks in both signals with scipy.signal.find_peaks,
  5. confirm the amplitude peaks that have a concurrent rise-rate peak - slow
     drift reaches a high value without a sharp rise, so it is not confirmed,
  6. flag as epileptiform the confirmed peaks above a cut-off chosen by
     `epileptic_criterion`.

Input:  a roi_fluorescence_full.csv (columns trial, frame, <one per ROI>).
Output: <csv folder>/epileptic_detection/
          roi_events.csv             every amplitude peak with its flags
          roi_epileptic_events.csv   only the epileptiform ones
          roi_traces_detrended.png   original vs running median, and the detrended trace
          roi_traces_events.png      one panel per ROI with the detections
          roi_traces_rise.png        rise of the detrended trace over the smoothed amplitude
"""

import os

# MUST run before numpy/scipy/matplotlib are imported, i.e. before MKL loads.
# In this conda env MKL's mkl_intel_thread.3.dll delay-loads libiomp5md.dll by
# bare name, the load fails, and the process dies with native exception
# 0xc06d007f and no traceback (CLAUDE.md 9.11/9.17). The sequential threading
# layer never loads that DLL.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.signal import find_peaks

from epileptic_diagnostics import plot_detrending, plot_detections, plot_rise

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
csv_path = (Path(__file__).resolve().parent
            /"outputs/botox_restani_rebuilt/260622_T7/t3/roi_fluorescence_full.csv")
starting_column = 2  # first 2 columns (trial, frame) are metadata, not ROIs

sampling_rate_hz = 10.0
# Slow-trend removal. The running median follows drift and bleaching but not
# events lasting a few seconds, so subtracting it removes the trend and keeps
# the events. It must stay several times longer than the events of interest.
median_window_s = 20.0
smooth_window_s = 1.0        # rolling-mean width
min_event_distance_s = 0.5   # refractory period between two peaks of one ROI

# Prominence thresholds in robust SDs (1.4826 * MAD) of each ROI's own signal,
# so ROIs with different baseline noise get comparable sensitivity.
amplitude_prominence_sd = 3.0
rise_prominence_sd = 4.0

# The rise signal is the change of the DETRENDED trace (not the smoothed one)
# across this window, detrended[i + n] - detrended[i], and its peaks are the
# steepest rises. The window itself averages out frame-to-frame noise. It is
# rounded to whole frames, minimum one (a plain first difference): 0.5 s is
# 5 frames at 10 Hz.
rise_window_s = 0.4
not_linear_rise = True  # True: window max - min; False: end - start

# A rise peak is concurrent when it falls in [peak - lead, peak + lag]. The
# derivative maximum precedes the amplitude maximum, hence lead >> lag.
rise_lead_s = 1.0
rise_lag_s = 0.1

# How the epileptiform cut-off is set:
#   "calibrated_z"     - a fixed threshold in robust SDs, `epileptic_z_threshold`,
#                        the same in every recording. When it is None the cut-off
#                        is the top `epileptic_frame_percent` of this recording's
#                        pooled frames instead, which is NOT comparable across
#                        recordings.
#   "frame_percentile" - the top `epileptic_frame_percent` of this recording's
#                        frames, pooled or per ROI (`epileptic_frame_population`).
#   "peak_percentile"  - the top `epileptic_top_percent` of the peaks themselves,
#                        ranked by `epileptic_rank_metric` within `epileptic_scope`.
epileptic_criterion = "calibrated_z"
epileptic_z_threshold = None     # robust SDs above each ROI's median, or None
epileptic_frame_percent = 1.0    # 1.0 = level exceeded by 1% of frames
epileptic_frame_population = "recording"  # "recording" (pooled) or "roi"
epileptic_top_percent = 10.0     # peak_percentile only
epileptic_rank_metric = "prominence_sd"   # "prominence_sd", "prominence" or "amplitude"
epileptic_scope = "roi"          # "roi", "recording" or "trial"
# True: epileptiform = above the cut-off AND confirmed by a concurrent rise.
# False: every peak above the cut-off, to see what the concurrency test discards.
epileptic_require_confirmed = True

# Index-space values derived from the physical ones above.
# Odd, so the running median is centred exactly on its own frame.
n_median = 2 * round(median_window_s * sampling_rate_hz / 2) + 1  # frames per running median
n_smooth = max(1, round(smooth_window_s * sampling_rate_hz))  # frames per rolling mean
edge_frames = n_smooth // 2  # frames at each trial end averaged over a partial window
min_distance_frames = max(1, round(min_event_distance_s * sampling_rate_hz))
n_rise = max(1, round(rise_window_s * sampling_rate_hz))  # frames spanned by one rise value
rise_lead_frames = rise_lead_s * sampling_rate_hz
rise_lag_frames = rise_lag_s * sampling_rate_hz

output_dir = csv_path.parent / "epileptic_detection"
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Functions, one per step. Every table below has one row per frame, trials
# stacked, and the same row index as the loaded CSV.
# ---------------------------------------------------------------------------
def robust_sd(values: np.ndarray) -> float:
    """SD estimated from the median absolute deviation, so events barely move it."""
    mad = np.median(np.abs(values - np.median(values)))
    return float(1.4826 * mad)


def rise_signal(amplitude: np.ndarray, n_rise: int, not_linear_rise: bool) -> np.ndarray:
    """Endpoint change, or window maximum minus minimum if `not_linear_rise`.

    Element i spans frames i -> i + n_rise, so its effective position is the
    window centre, i + n_rise / 2. Shared by the detection and the rise plot so
    the plotted signal is exactly the one peaks were found in.
    """
    # Both modes span i through i + n_rise (n_rise + 1 samples).
    # The nonlinear range is unsigned, so falling windows can also be positive.
    if not_linear_rise:
        if len(amplitude) <= n_rise:
            return amplitude[:0].copy()
        window_size = n_rise + 1
        window_min = minimum_filter1d(amplitude, size=window_size, axis=0)
        window_max = maximum_filter1d(amplitude, size=window_size, axis=0)
        # Keep only complete windows, aligned with the original endpoint rise.
        start = window_size // 2
        stop = start + len(amplitude) - n_rise
        return window_max[start:stop] - window_min[start:stop]

    return amplitude[n_rise:] - amplitude[:-n_rise]  # (n_trial_frames - n_rise,)


def remove_slow_trend(
    fluorescence: pd.DataFrame, roi_names: list[str], n_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract a centred running median from every ROI, inside each trial.

    Returns `(detrended, trend)`. `detrended` is a copy of `fluorescence` with
    the ROI columns replaced, so it keeps the trial/frame columns and can be
    used wherever the original table was. `trend` holds only the ROI columns.
    Near the trial ends the window is truncated (min_periods=1), so the median
    there is taken over fewer frames rather than padded with invented values.
    """
    # trend: (n_frames, n_rois)
    trend = (
        fluorescence.groupby("trial")[roi_names]
        .rolling(window=n_window, center=True, min_periods=1)
        .median()
        .reset_index(level="trial", drop=True)
        .sort_index()
    )
    detrended = fluorescence.copy()
    detrended[roi_names] = fluorescence[roi_names] - trend
    return detrended, trend


def smooth_by_trial(traces: pd.DataFrame, roi_names: list[str], n_window: int) -> pd.DataFrame:
    """Centred rolling mean of every ROI, computed inside each trial.

    Grouping by trial keeps the window from bleeding across a trial boundary;
    centring keeps peak timing unbiased.
    """
    # smoothed: (n_frames, n_rois)
    return (
        traces.groupby("trial")[roi_names]
        .rolling(window=n_window, center=True, min_periods=1)
        .mean()
        .reset_index(level="trial", drop=True)
        .sort_index()
    )


def robust_z_frames(
    traces: pd.DataFrame, smoothed: pd.DataFrame, roi_names: list[str], edge_frames: int,
) -> pd.DataFrame:
    """Every smoothed frame in robust SDs above its ROI's median within its trial.

    Median and MAD are set by the baseline, so z is comparable across ROIs and a
    gain change divides out. The partial-window frames at each trial end are left
    NaN so they do not vote on where a frame-based cut-off sits.
    """
    n_rois = len(roi_names)
    # frame_z: (n_frames, n_rois)
    frame_z = pd.DataFrame(np.nan, index=smoothed.index, columns=roi_names, dtype=float)
    for trial_index in traces.groupby("trial").groups.values():
        # trial_smoothed: (n_trial_frames, n_rois)
        trial_smoothed = smoothed.loc[trial_index, roi_names].to_numpy(dtype=float)
        roi_median = np.median(trial_smoothed, axis=0)  # (n_rois,)
        roi_scale = np.array([robust_sd(trial_smoothed[:, i_roi]) for i_roi in range(n_rois)])
        # A flat ROI has no scale, hence no z: NaN keeps it out of every cut-off.
        roi_scale = np.where(roi_scale > 0, roi_scale, np.nan)
        trial_z = (trial_smoothed - roi_median) / roi_scale
        if edge_frames:
            trial_z[:edge_frames] = np.nan
            trial_z[-edge_frames:] = np.nan
        frame_z.loc[trial_index, roi_names] = trial_z
    return frame_z


def find_trace_events(
    amplitude: np.ndarray,
    detrended_trace: np.ndarray,
    frames: np.ndarray,
    *,
    sampling_rate_hz: float,
    min_distance_frames: int,
    amplitude_prominence_sd: float,
    rise_prominence_sd: float,
    n_rise: int,
    rise_lead_frames: float,
    rise_lag_frames: float,
) -> list[dict]:
    """Amplitude peaks of ONE smoothed trace, each checked for a concurrent rise.

    `amplitude` is the smoothed trace of one ROI in one trial, (n_trial_frames,),
    and is where amplitude peaks are found. `detrended_trace` is the same ROI and
    trial before smoothing, (n_trial_frames,), and is where the rise is taken.
    `frames` is the frame number of each sample, `n_rise` the rise window in
    frames. Returns one dict per amplitude peak.
    """
    rise = rise_signal(detrended_trace, n_rise, not_linear_rise)  # (n_trial_frames - n_rise,)
    rise_centre_offset = n_rise / 2
    rise_window_duration_s = n_rise / sampling_rate_hz
    amplitude_sd = robust_sd(amplitude)
    amplitude_median = float(np.median(amplitude))
    has_scale = amplitude_sd > 0

    # A flat trace has zero MAD, so no peak can clear the prominence and
    # nothing is detected there - rather than everything.
    amplitude_peaks, amplitude_props = find_peaks(
        amplitude,
        prominence=amplitude_prominence_sd * amplitude_sd,
        distance=min_distance_frames,
    )
    rise_peaks, rise_props = find_peaks(
        rise,
        prominence=rise_prominence_sd * robust_sd(rise),
        distance=min_distance_frames,
    )
    rise_positions = rise_peaks + rise_centre_offset

    rows = []
    for i_peak, peak in enumerate(amplitude_peaks):
        concurrent = np.flatnonzero(
            (rise_positions >= peak - rise_lead_frames)
            & (rise_positions <= peak + rise_lag_frames)
        )
        confirmed = concurrent.size > 0
        if confirmed:
            # The steepest rise in the window marks the event onset.
            steepest = concurrent[np.argmax(rise_props["prominences"][concurrent])]
            rise_frame = float(frames[rise_peaks[steepest]]) + rise_centre_offset
            # Change across the window divided by the window length = DF/F per second.
            rise_per_s = float(rise[rise_peaks[steepest]] / rise_window_duration_s)
        else:
            rise_frame = np.nan
            rise_per_s = np.nan

        peak_frame = float(frames[peak])
        peak_amplitude = float(amplitude[peak])
        prominence = float(amplitude_props["prominences"][i_peak])
        rows.append({
            "frame": int(peak_frame),
            "time_s": peak_frame / sampling_rate_hz,
            "amplitude": peak_amplitude,
            # Same scale as robust_z_frames: a peak is one of those frames.
            "amplitude_z": ((peak_amplitude - amplitude_median) / amplitude_sd
                            if has_scale else np.nan),
            "prominence": prominence,
            "prominence_sd": prominence / amplitude_sd if has_scale else np.nan,
            "rise_time_s": rise_frame / sampling_rate_hz,
            # How far the rise led the peak; compare against rise_lead_s.
            "rise_offset_s": (peak_frame - rise_frame) / sampling_rate_hz,
            "rise_per_s": rise_per_s,
            "confirmed": confirmed,
        })
    return rows


def detect_events(
    traces: pd.DataFrame, smoothed: pd.DataFrame, roi_names: list[str], **peak_options,
) -> pd.DataFrame:
    """Run `find_trace_events` on every ROI of every trial and collect one table.

    `peak_options` are the keyword arguments of `find_trace_events`.
    """
    event_rows = []
    for trial, trial_index in traces.groupby("trial").groups.items():
        frames = traces.loc[trial_index, "frame"].to_numpy()  # (n_trial_frames,)
        # trial_smoothed, trial_detrended: (n_trial_frames, n_rois)
        trial_smoothed = smoothed.loc[trial_index, roi_names].to_numpy(dtype=float)
        trial_detrended = traces.loc[trial_index, roi_names].to_numpy(dtype=float)
        for i_roi, roi in enumerate(roi_names):
            roi_rows = find_trace_events(
                trial_smoothed[:, i_roi], trial_detrended[:, i_roi], frames, **peak_options)
            for row in roi_rows:
                event_rows.append({"trial": trial, "roi": roi, **row})

    # Dtypes are pinned so a recording with no peaks still gives a table with the
    # right columns and a boolean `confirmed` usable as a mask.
    event_dtypes = {
        "trial": "int64", "roi": "object", "frame": "int64", "time_s": "float64",
        "amplitude": "float64", "amplitude_z": "float64", "prominence": "float64",
        "prominence_sd": "float64", "rise_time_s": "float64",
        "rise_offset_s": "float64", "rise_per_s": "float64", "confirmed": "bool",
    }
    return pd.DataFrame(event_rows, columns=list(event_dtypes)).astype(event_dtypes)


def peak_percentile_cut(
    events: pd.DataFrame, *, top_percent: float, rank_metric: str, scope: str,
) -> np.ndarray:
    """Per-peak cut-off: the top `top_percent` of the peaks in the same group.

    Unconfirmed peaks are ranked too: they are real peaks, so they belong in the
    size distribution. A group with nothing finite to rank gets a NaN cut-off.
    """
    if scope == "recording":
        group_labels = pd.Series(0, index=events.index)
    else:
        group_labels = events[scope]
    rank_values = events[rank_metric].to_numpy(dtype=float)  # (n_events,)
    peak_cut = np.full(len(events), np.nan)  # (n_events,)
    for group_rows in events.groupby(group_labels, sort=False).indices.values():
        group_values = rank_values[group_rows]
        # A flat ROI gives NaN; it can neither set the cut-off nor clear it.
        group_values = group_values[np.isfinite(group_values)]
        if group_values.size:
            peak_cut[group_rows] = np.percentile(group_values, 100.0 - top_percent)
    return peak_cut


def frame_z_cut(
    events: pd.DataFrame,
    frame_z: pd.DataFrame,
    roi_names: list[str],
    *,
    criterion: str,
    z_threshold: float | None,
    frame_percent: float,
    frame_population: str,
) -> np.ndarray:
    """Per-peak cut-off in robust z, for "calibrated_z" and "frame_percentile"."""
    frame_quantile = 100.0 - frame_percent
    if criterion == "calibrated_z" and z_threshold is not None:
        return np.full(len(events), float(z_threshold))

    if criterion == "calibrated_z" or frame_population == "recording":
        # One cut-off from all ROIs' frames pooled.
        pooled_z = frame_z.to_numpy(dtype=float).ravel()
        pooled_z = pooled_z[np.isfinite(pooled_z)]
        pooled_cut = float(np.percentile(pooled_z, frame_quantile)) if pooled_z.size else np.nan
        return np.full(len(events), pooled_cut)

    if frame_population == "roi":
        # One cut-off per ROI from that ROI's own frames.
        roi_cut = {}
        for roi in roi_names:
            roi_z = frame_z[roi].to_numpy(dtype=float)
            roi_z = roi_z[np.isfinite(roi_z)]
            if roi_z.size:
                roi_cut[roi] = float(np.percentile(roi_z, frame_quantile))
        return events["roi"].map(roi_cut).to_numpy(dtype=float)

    raise ValueError(f"unknown epileptic_frame_population: {frame_population!r}")


def flag_epileptic(
    events: pd.DataFrame,
    frame_z: pd.DataFrame,
    roi_names: list[str],
    *,
    criterion: str,
    z_threshold: float | None,
    frame_percent: float,
    frame_population: str,
    top_percent: float,
    rank_metric: str,
    scope: str,
    require_confirmed: bool,
) -> pd.DataFrame:
    """Copy of `events` with `epileptic`, `epileptic_threshold`, `epileptic_criterion`.

    A peak is epileptiform when its value is at or above its cut-off (ties kept)
    and, if `require_confirmed`, it has a concurrent rise.
    """
    if criterion == "peak_percentile":
        peak_value = events[rank_metric].to_numpy(dtype=float)
        peak_cut = peak_percentile_cut(
            events, top_percent=top_percent, rank_metric=rank_metric, scope=scope)
    elif criterion in {"calibrated_z", "frame_percentile"}:
        peak_value = events["amplitude_z"].to_numpy(dtype=float)
        peak_cut = frame_z_cut(
            events, frame_z, roi_names, criterion=criterion, z_threshold=z_threshold,
            frame_percent=frame_percent, frame_population=frame_population)
    else:
        raise ValueError(f"unknown epileptic_criterion: {criterion!r}")

    is_epileptic = np.isfinite(peak_value) & np.isfinite(peak_cut) & (peak_value >= peak_cut)
    if require_confirmed:
        is_epileptic &= events["confirmed"].to_numpy(dtype=bool)

    events = events.copy()
    events["epileptic"] = is_epileptic.astype(bool)
    events["epileptic_threshold"] = peak_cut.astype(float)
    events["epileptic_criterion"] = criterion
    return events


def describe_criterion(
    events: pd.DataFrame, *, criterion: str, top_percent: float, rank_metric: str,
) -> str:
    """Short text for the legend and title describing the cut-off that was applied."""
    applied_cuts = events["epileptic_threshold"].dropna()
    if criterion == "peak_percentile":
        return f"top {top_percent:g}% of peaks by {rank_metric}"
    if applied_cuts.empty:
        return "no cut-off"
    if applied_cuts.nunique() == 1:
        return f"z >= {applied_cuts.iloc[0]:.2f}"
    return f"z >= {applied_cuts.min():.2f}..{applied_cuts.max():.2f} per ROI"


def rise_by_trial(
    traces: pd.DataFrame, roi_names: list[str], n_rise: int, sampling_rate_hz: float,
) -> pd.DataFrame:
    """Rise of every ROI's detrended trace in DF/F per second, one row per frame.

    `traces` is the detrended table, the same signal `find_trace_events` takes
    the rise from. Row i holds the change across frames i -> i + n_rise, so it
    belongs at time_s + n_rise / 2 / sampling_rate_hz. The last n_rise rows of
    each trial have no full window and stay NaN.
    """
    window_duration_s = n_rise / sampling_rate_hz
    # rise_per_s: (n_frames, n_rois)
    rise_per_s = pd.DataFrame(np.nan, index=traces.index, columns=roi_names, dtype=float)
    for trial_index in traces.groupby("trial").groups.values():
        # trial_detrended: (n_trial_frames, n_rois)
        trial_detrended = traces.loc[trial_index, roi_names].to_numpy(dtype=float)
        # trial_rise: (n_trial_frames, n_rois), NaN-padded at the end
        trial_rise = np.full_like(trial_detrended, np.nan)
        trial_rise[:-n_rise] = rise_signal(trial_detrended, n_rise, not_linear_rise) / window_duration_s
        rise_per_s.loc[trial_index, roi_names] = trial_rise
    return rise_per_s


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# fluorescence: (n_frames, 2 + n_rois), trials stacked one after another
fluorescence = pd.read_csv(csv_path)
roi_names = list(fluorescence.columns[starting_column:])
n_rois = len(roi_names)
time_s = fluorescence["frame"].to_numpy() / sampling_rate_hz  # (n_frames,)
recording_label = f"{csv_path.parent.parent.name}/{csv_path.parent.name}"

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
# Slow trends removed; everything after this point works on `detrended`.
# detrended: (n_frames, 2 + n_rois), trend: (n_frames, n_rois)
detrended, trend = remove_slow_trend(fluorescence, roi_names, n_median)

# Short-window smoothing of the detrended traces. smoothed: (n_frames, n_rois)
smoothed = smooth_by_trial(detrended, roi_names, n_smooth)

# Every frame in robust z, for the frame-based cut-offs. frame_z: (n_frames, n_rois)
frame_z = robust_z_frames(detrended, smoothed, roi_names, edge_frames)

# One row per amplitude peak, `confirmed` when a rise peak falls in its window.
events = detect_events(
    detrended, smoothed, roi_names,
    sampling_rate_hz=sampling_rate_hz,
    min_distance_frames=min_distance_frames,
    amplitude_prominence_sd=amplitude_prominence_sd,
    rise_prominence_sd=rise_prominence_sd,
    n_rise=n_rise,
    rise_lead_frames=rise_lead_frames,
    rise_lag_frames=rise_lag_frames,
)

# The cut-off each peak was measured against and whether it clears it.
events = flag_epileptic(
    events, frame_z, roi_names,
    criterion=epileptic_criterion,
    z_threshold=epileptic_z_threshold,
    frame_percent=epileptic_frame_percent,
    frame_population=epileptic_frame_population,
    top_percent=epileptic_top_percent,
    rank_metric=epileptic_rank_metric,
    scope=epileptic_scope,
    require_confirmed=epileptic_require_confirmed,
)
epileptic_events = events[events["epileptic"]]

# The rise of every ROI's detrended trace as a table, for plotting.
# rise_per_s: (n_frames, n_rois)
rise_per_s = rise_by_trial(detrended, roi_names, n_rise, sampling_rate_hz)

criterion_label = describe_criterion(
    events, criterion=epileptic_criterion,
    top_percent=epileptic_top_percent, rank_metric=epileptic_rank_metric)

# ---------------------------------------------------------------------------
# Saving tables
# ---------------------------------------------------------------------------
events.to_csv(output_dir / "roi_events.csv", index=False)
epileptic_events.to_csv(output_dir / "roi_epileptic_events.csv", index=False)

print(csv_path)
print(f"  {n_rois} ROIs, {len(fluorescence)} frames at {sampling_rate_hz:g} Hz, "
      f"slow trend removed with a {median_window_s:g} s running median")
print(f"  {len(events)} amplitude peaks, {int(events['confirmed'].sum())} confirmed")
print(f"  {len(epileptic_events)} epileptiform ({epileptic_criterion}: {criterion_label})")
print(f"  wrote {output_dir}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
plot_detrending(
    fluorescence, trend, detrended, roi_names, time_s,
    output_dir / "roi_traces_detrended.png",
    f"Slow-trend removal - {recording_label} ({median_window_s:g} s running median)",
)
# print where the plot is saved 
print(f"  wrote {output_dir / 'roi_traces_detrended.png'}")

plot_detections(
    detrended, smoothed, events, roi_names, time_s, criterion_label,
    output_dir / "roi_traces_events.png",
    f"Epileptiform event detection - {recording_label} "
    f"({median_window_s:g} s median detrend, {smooth_window_s:g} s smoothing, "
    f"prominence {amplitude_prominence_sd:g} SD, {epileptic_criterion}: {criterion_label})",
)
print(f"  wrote {output_dir / 'roi_traces_events.png'}")

plot_rise(
    smoothed, rise_per_s, events, roi_names, time_s,
    n_rise / 2 / sampling_rate_hz,
    output_dir / "roi_traces_rise.png",
    f"Rise rate of the detrended trace vs smoothed amplitude - {recording_label} "
    f"({rise_window_s:g} s rise window = {n_rise} frames, "
    f"{smooth_window_s:g} s smoothing, rise prominence {rise_prominence_sd:g} SD)",
)
print(f"  wrote {output_dir / 'roi_traces_rise.png'}")
