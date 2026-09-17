"""Epileptiform events over every recording, tabulated by area, animal and day.

Runs the detection of `test_epileptic_detection.py` on every
`<input_root>/<date>_<animal>/<t#>/roi_fluorescence_full.csv` and writes tables
that split the flagged events by cortical area, animal and day.

"Day" is the animal's recording day, not the calendar date: `recording_day` 1 is
that animal's first date in the dataset, 2 its second, and so on. Animals were
recorded on different dates, so this is what lines them up across the time
course. The date is kept as a reference column.

The detection functions below are COPIED from test_epileptic_detection.py (that
file is a flat script and runs on import, so it cannot be imported). Keep the
two in sync when the method changes.

One difference is deliberate: the epileptiform cut-off. The single-recording
script, with `epileptic_z_threshold = None`, takes the top 1% of THAT
recording's frames, which moves with each recording's own activity. Here one z
threshold is calibrated over the pooled frames of ALL recordings, so a count
means the same thing in every area, animal and day.

Counts are also given as a rate, `epileptic_per_roi_min` = events divided by
ROI-minutes (recorded minutes x number of ROIs summed into the row). One unit at
every level means a per-area row and a per-animal row can be compared directly,
and a day+animal unit missing a `t#` does not look like a quiet one.
"""

import os

# MUST run before numpy/scipy/matplotlib are imported, i.e. before MKL loads:
# the delay-load fault documented in test_epileptic_detection.py (CLAUDE.md 9.11/9.17).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.signal import find_peaks

from epileptic_diagnostics import plot_rise, plot_roi_overview

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
input_root = repo_root / "outputs" / "botox_restani_rebuilt"
csv_name = "roi_fluorescence_full.csv"  # per-ROI dF/F, one file per t# recording
summary_name = "batch_summary.csv"      # connectivity batch index; carries each animal's group
starting_column = 2  # first 2 columns (trial, frame) are metadata, not ROIs
recording_limit = None  # stop after this many recordings; None runs all of them (smoke tests)
write_recording_diagnostics = True  # save event tables and the rise plot in each recording
diagnostic_dpi = 120  # faster, smaller plots; set 200 for the original resolution
progress_every = 50     # print a progress line every this many recordings

# Detection - same values as test_epileptic_detection.py; see that file for the reasoning.
sampling_rate_hz = 10.0
median_window_s = 20.0       # running median subtracted to remove slow trends
smooth_window_s = 1.0        # rolling-mean width
min_event_distance_s = 0.5   # refractory period between two peaks of one ROI
amplitude_prominence_sd = 3.0
rise_prominence_sd = 3.0
rise_window_s = 0.5          # rise = detrended[i + n] - detrended[i] over this window
not_linear_rise = True  # True: window max - min; False: end - start
# True: take the rise from the smoothed amplitude (the trace the peaks are found
# on) instead of the detrended, unsmoothed trace. Smoothing trades the rise's
# sharpness for less frame-to-frame noise, so rise_prominence_sd may need retuning.
rise_on_smoothed = False
rise_lead_s = 1.0            # a rise peak up to this long BEFORE the amplitude peak is concurrent
rise_lag_s = 0.3             # ... and up to this long after it
# True: epileptiform = above the cut-off AND confirmed by a concurrent rise.
epileptic_require_confirmed = True

# Calibration: the cut-off is the robust z exceeded by this % of ALL frames of ALL recordings.
calibration_frame_percent = 1.0
z_histogram_step = 0.01  # SD resolution of the pooled histogram the cut-off is read from
z_histogram_range = (-10.0, 30.0)  # values outside land in the end bins, so tail fractions stay exact

# Palette: one-hue sequential blue for the heatmap, fixed categorical order for groups.
sequential_ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
group_colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3: validated all-pairs

# Index-space values derived from the physical ones above.
n_median = 2 * round(median_window_s * sampling_rate_hz / 2) + 1  # odd, frames per running median
n_smooth = max(1, round(smooth_window_s * sampling_rate_hz))  # frames per rolling mean
edge_frames = n_smooth // 2  # frames at each trial end averaged over a partial window
min_distance_frames = max(1, round(min_event_distance_s * sampling_rate_hz))
n_rise = max(1, round(rise_window_s * sampling_rate_hz))  # frames spanned by one rise value
rise_lead_frames = max(1, round(rise_lead_s * sampling_rate_hz))
rise_lag_frames = max(1, round(rise_lag_s * sampling_rate_hz))
z_bin_edges = np.arange(z_histogram_range[0], z_histogram_range[1] + z_histogram_step / 2,
                        z_histogram_step)  # (n_bins + 1,)

output_dir = input_root.parent / "epileptic_by_area_animal_day"


# ---------------------------------------------------------------------------
# Functions. The first seven are copied unchanged from test_epileptic_detection.py.
# ---------------------------------------------------------------------------
def robust_sd(values: np.ndarray) -> float:
    """SD estimated from the median absolute deviation, so events barely move it."""
    mad = np.median(np.abs(values - np.median(values)))
    return float(1.4826 * mad)


def rise_signal(amplitude: np.ndarray, n_rise: int, not_linear_rise: bool) -> np.ndarray:
    """Endpoint change, or window maximum minus minimum if `not_linear_rise`.

    Element i spans frames i -> i + n_rise, so its effective position is the
    window centre, i + n_rise / 2.
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
    the ROI columns replaced; `trend` holds only the ROI columns.
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
    """Centred rolling mean of every ROI, computed inside each trial."""
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

    The partial-window frames at each trial end are left NaN so they do not vote
    on where the cut-off sits.
    """
    n_rois = len(roi_names)
    # frame_z: (n_frames, n_rois)
    frame_z = pd.DataFrame(np.nan, index=smoothed.index, columns=roi_names, dtype=float)
    for trial_index in traces.groupby("trial").groups.values():
        # trial_smoothed: (n_trial_frames, n_rois)
        trial_smoothed = smoothed.loc[trial_index, roi_names].to_numpy(dtype=float)
        roi_median = np.median(trial_smoothed, axis=0)  # (n_rois,)
        roi_scale = np.array([robust_sd(trial_smoothed[:, i_roi]) for i_roi in range(n_rois)])
        # A flat ROI has no scale, hence no z: NaN keeps it out of the cut-off.
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
    not_linear_rise: bool = not_linear_rise,
) -> list[dict]:
    """Amplitude peaks of ONE smoothed trace, each checked for a concurrent rise.

    `amplitude` is the smoothed trace of one ROI in one trial, (n_trial_frames,);
    `detrended_trace` is the trace the rise is taken from - the same ROI and trial
    before smoothing, or the smoothed amplitude itself when `rise_on_smoothed`.
    Returns one dict per amplitude peak.
    """
    rise = rise_signal(detrended_trace, n_rise, not_linear_rise)  # (n_trial_frames - n_rise,)
    rise_centre_offset = n_rise / 2
    rise_window_duration_s = n_rise / sampling_rate_hz
    amplitude_sd = robust_sd(amplitude)
    amplitude_median = float(np.median(amplitude))
    has_scale = amplitude_sd > 0

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
            "amplitude_z": ((peak_amplitude - amplitude_median) / amplitude_sd
                            if has_scale else np.nan),
            "prominence": prominence,
            "prominence_sd": prominence / amplitude_sd if has_scale else np.nan,
            "rise_time_s": rise_frame / sampling_rate_hz,
            "rise_offset_s": (peak_frame - rise_frame) / sampling_rate_hz,
            "rise_per_s": rise_per_s,
            "confirmed": confirmed,
        })
    return rows


def detect_events(
    traces: pd.DataFrame, smoothed: pd.DataFrame, roi_names: list[str],
    rise_traces: pd.DataFrame | None = None, **peak_options,
) -> pd.DataFrame:
    """Run `find_trace_events` on every ROI of every trial and collect one table.

    `rise_traces` is the table the rise is taken from, indexed like `traces`;
    None means `traces` itself (the detrended signal).
    """
    rise_traces = traces if rise_traces is None else rise_traces
    event_rows = []
    for trial, trial_index in traces.groupby("trial").groups.items():
        frames = traces.loc[trial_index, "frame"].to_numpy()  # (n_trial_frames,)
        # trial_smoothed, trial_detrended: (n_trial_frames, n_rois)
        trial_smoothed = smoothed.loc[trial_index, roi_names].to_numpy(dtype=float)
        trial_detrended = rise_traces.loc[trial_index, roi_names].to_numpy(dtype=float)
        for i_roi, roi in enumerate(roi_names):
            roi_rows = find_trace_events(
                trial_smoothed[:, i_roi], trial_detrended[:, i_roi], frames, **peak_options)
            for row in roi_rows:
                event_rows.append({"trial": trial, "roi": roi, **row})

    # Dtypes are pinned so a recording with no peaks still gives the right columns.
    event_dtypes = {
        "trial": "int64", "roi": "object", "frame": "int64", "time_s": "float64",
        "amplitude": "float64", "amplitude_z": "float64", "prominence": "float64",
        "prominence_sd": "float64", "rise_time_s": "float64",
        "rise_offset_s": "float64", "rise_per_s": "float64", "confirmed": "bool",
    }
    return pd.DataFrame(event_rows, columns=list(event_dtypes)).astype(event_dtypes)


def rise_by_trial(
    traces: pd.DataFrame, roi_names: list[str], n_rise: int, sampling_rate_hz: float,
    not_linear_rise: bool = not_linear_rise,
    rise_traces: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rise of every ROI's trace in DF/F per second, one row per frame.

    `traces` is the detrended table and supplies the trial grouping. The rise is
    taken from `rise_traces` (None: `traces`), the same signal `detect_events`
    was given. Row i holds the change across frames i -> i + n_rise, so it
    belongs at time_s + n_rise / 2 / sampling_rate_hz. The last n_rise rows of
    each trial have no full window and stay NaN.
    """
    rise_traces = traces if rise_traces is None else rise_traces
    window_duration_s = n_rise / sampling_rate_hz
    # rise_per_s: (n_frames, n_rois)
    rise_per_s = pd.DataFrame(np.nan, index=traces.index, columns=roi_names, dtype=float)
    for trial_index in traces.groupby("trial").groups.values():
        # trial_detrended: (n_trial_frames, n_rois)
        trial_detrended = rise_traces.loc[trial_index, roi_names].to_numpy(dtype=float)
        # trial_rise: (n_trial_frames, n_rois), NaN-padded at the end
        trial_rise = np.full_like(trial_detrended, np.nan)
        trial_rise[:-n_rise] = rise_signal(trial_detrended, n_rise, not_linear_rise) / window_duration_s
        rise_per_s.loc[trial_index, roi_names] = trial_rise
    return rise_per_s


def summarise(long_table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum the long table over `keys`; the rate stays per ROI-minute at every level."""
    table = long_table.assign(unit=long_table["date"] + "_" + long_table["animal"])
    summary = table.groupby(keys, as_index=False, sort=False).agg(
        n_units=("unit", "nunique"),
        n_rois=("roi", "nunique"),
        roi_min=("recorded_min", "sum"),
        n_peaks=("n_peaks", "sum"),
        n_confirmed=("n_confirmed", "sum"),
        n_epileptic=("n_epileptic", "sum"),
    )
    summary["epileptic_per_roi_min"] = summary["n_epileptic"] / summary["roi_min"]
    return summary


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def select_diagnostic_recordings(n_recordings, count, seed=0):
    """Choose recordings without replacement; None means all, zero means none."""
    if count is not None and count < 0:
        raise ValueError("diagnostic_plot_count must be nonnegative or None")
    if count is None:
        return set(range(n_recordings))
    return set(np.random.default_rng(seed).choice(n_recordings, min(count, n_recordings), replace=False).tolist())


def select_diagnostic_animals(recordings, animals_per_group, seed=0):
    """Pick `animals_per_group` animals at random inside each group.

    Returns the sorted animal names; a group with fewer animals keeps all of
    them. None means every animal.
    """
    if animals_per_group is None:
        return sorted(recordings["animal"].unique())
    if animals_per_group < 0:
        raise ValueError("diagnostic_animals_per_group must be nonnegative or None")
    rng = np.random.default_rng(seed)
    chosen = []
    for group in sorted(recordings["group"].unique()):
        group_animals = sorted(recordings.loc[recordings["group"] == group, "animal"].unique())
        n_chosen = min(animals_per_group, len(group_animals))
        chosen.extend(rng.choice(group_animals, n_chosen, replace=False).tolist())
    return sorted(chosen)


def detect_recording(job: dict) -> dict:
    """Detection pass for ONE recording; a module-level function so pool workers can run it.

    Returns the unflagged events, this recording's contribution to the pooled z
    histogram, its frame count and its ROI columns (checked by the caller).
    """
    # fluorescence: (n_frames, 2 + n_rois), trials stacked
    fluorescence = job["trace_loader"](job["csv_path"])
    roi_names = list(fluorescence.columns[starting_column:])
    detrended, _ = remove_slow_trend(fluorescence, roi_names, job["n_median"])
    smoothed = smooth_by_trial(detrended, roi_names, job["n_smooth"])  # (n_frames, n_rois)
    frame_z = robust_z_frames(detrended, smoothed, roi_names, job["edge_frames"])  # (n_frames, n_rois)

    z_bin_edges = job["z_bin_edges"]
    pooled_z = frame_z.to_numpy(dtype=float).ravel()
    pooled_z = pooled_z[np.isfinite(pooled_z)]
    # Clipped into the end bins rather than dropped, so the counts sum to the real frames.
    pooled_z = np.clip(pooled_z, z_bin_edges[0], z_bin_edges[-1] - job["z_histogram_step"] / 2)
    histogram = np.histogram(pooled_z, bins=z_bin_edges)[0]  # (n_bins,)

    events = detect_events(
        detrended, smoothed, roi_names,
        rise_traces=smoothed if job["rise_on_smoothed"] else detrended,
        **job["peak_options"],
    )
    # Value of the loaded trace (not detrended, not smoothed) at each peak frame,
    # for the optional epileptic_min_signal gate.
    loaded = fluorescence.set_index(["trial", "frame"])[roi_names]  # (n_frames, n_rois)
    row_pos = loaded.index.get_indexer(pd.MultiIndex.from_arrays([events["trial"], events["frame"]]))
    if (row_pos < 0).any():
        raise ValueError(f"Peak frames missing from the loaded trace: {job['csv_path']}")
    roi_pos = events["roi"].map({roi: i_roi for i_roi, roi in enumerate(roi_names)}).to_numpy(dtype=int)
    events["signal_at_peak"] = loaded.to_numpy(dtype=float)[row_pos, roi_pos]
    for i_column, (name, value) in enumerate(job["labels"].items()):
        events.insert(i_column, name, value)
    return dict(events=events, histogram=histogram, n_frames=len(fluorescence), roi_names=roi_names)


def write_recording_outputs(job: dict) -> float:
    """Event tables and (if selected) figures of ONE recording; returns seconds taken.

    Module-level so pool workers can run it; everything it needs is in `job`.
    """
    started = time.perf_counter()
    recording_dir = job["recording_dir"]
    recording_dir.mkdir(parents=True, exist_ok=True)
    recording_events = job["events"]
    recording_events.to_csv(recording_dir / "roi_events.csv", index=False)
    recording_events.loc[recording_events["epileptic"]].to_csv(
        recording_dir / "roi_epileptic_events.csv", index=False)
    if not job["plot"]:
        return time.perf_counter() - started
    roi_names = job["roi_names"]
    sampling_rate_hz = job["sampling_rate_hz"]
    n_rise = job["n_rise"]
    fluorescence = job["trace_loader"](job["csv_path"])
    detrended, _ = remove_slow_trend(fluorescence, roi_names, job["n_median"])
    smoothed = smooth_by_trial(detrended, roi_names, job["n_smooth"])
    rise_per_s = rise_by_trial(detrended, roi_names, n_rise, sampling_rate_hz, job["not_linear_rise"],
                               rise_traces=smoothed if job["rise_on_smoothed"] else detrended)
    time_s = fluorescence["frame"].to_numpy(dtype=float) / sampling_rate_hz
    label = job["label"]
    # The loaded signal itself, before detrending: every ROI over the whole
    # recording on one axis, the counterpart of run_botox_batch's roi_traces_full.png.
    plot_roi_overview(fluorescence, roi_names, time_s,
                      recording_dir / "roi_traces_all.png",
                      f"All ROIs - {label} ({len(roi_names)} ROIs, {job['signal_unit']})",
                      dpi=diagnostic_dpi, signal_unit=job["signal_unit"],
                      width_scale=job["width_scale"])
    plot_rise(smoothed, rise_per_s, recording_events, roi_names, time_s,
              n_rise / 2 / sampling_rate_hz, recording_dir / "roi_traces_rise.png",
              f"Rise rate - {label} ({job['rise_window_s']:g} s window, "
              f"not_linear_rise={job['not_linear_rise']}, rise_on_smoothed={job['rise_on_smoothed']})",
              dpi=diagnostic_dpi, signal_unit=job["signal_unit"],
              rise_source="smoothed amplitude" if job["rise_on_smoothed"] else "detrended trace",
              width_scale=job["width_scale"])
    return time.perf_counter() - started


def map_jobs(function, jobs: list[dict], workers: int, ordered: bool):
    """Yield (index, result) for every job, serially or on a process pool.

    `workers == 1` runs in this process (no pool, easiest to debug). Otherwise a
    `multiprocessing.Pool`, whose context exit terminates the workers, so Ctrl+C
    does not leave them running (CLAUDE.md 9.10).
    """
    indexed = list(enumerate(jobs))
    if workers == 1:
        for i_job, job in indexed:
            yield i_job, function(job)
        return
    with Pool(min(workers, len(jobs))) as pool:
        mapper = pool.imap if ordered else pool.imap_unordered
        yield from mapper(_indexed_call, [(function, i_job, job) for i_job, job in indexed], chunksize=1)


def _indexed_call(item):
    function, i_job, job = item
    return i_job, function(job)


def run_analysis(*, input_root=input_root, output_dir=output_dir,
                 recording_limit=recording_limit, trace_loader=pd.read_csv,
                 recording_paths=None, diagnostic_plot_count=None, diagnostic_seed=0,
                 diagnostic_animals_per_group=None, diagnostic_width_scale=1.0,
                 diagnostics_root=None, roi_metadata=None, provenance=None,
                 signal_unit="DF/F",
                 sampling_rate_hz=sampling_rate_hz,
                 median_window_s=median_window_s,
                 smooth_window_s=smooth_window_s,
                 min_event_distance_s=min_event_distance_s,
                 amplitude_prominence_sd=amplitude_prominence_sd,
                 rise_prominence_sd=rise_prominence_sd,
                 rise_window_s=rise_window_s,
                 not_linear_rise=not_linear_rise,
                 rise_on_smoothed=rise_on_smoothed,
                 rise_lead_s=rise_lead_s,
                 rise_lag_s=rise_lag_s,
                 epileptic_require_confirmed=epileptic_require_confirmed,
                 calibration_frame_percent=calibration_frame_percent,
                 z_histogram_step=z_histogram_step,
                 z_histogram_range=z_histogram_range,
                 epileptic_min_signal=None,
                 workers=1,
                 ):
    """Run the shared detector, optionally loading ROI traces from another source.

    Plot sampling only affects per-recording figures; event tables and pooled
    calibration still include every recording. Existing script defaults are retained.
    `epileptic_min_signal` (None: off) additionally requires the LOADED trace -
    before detrending and smoothing - to be at least this value at the peak frame
    (column `signal_at_peak`), e.g. 0.2 = 20% of pixels active.
    `workers` > 1 runs the per-recording detection and figures on a process pool;
    `trace_loader` must then be picklable (a module-level function or class
    instance). Results are identical to the serial run.
    """
    if epileptic_min_signal is not None and not np.isfinite(epileptic_min_signal):
        raise ValueError("epileptic_min_signal must be finite or None")
    if int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    # Derive every frame window from this run's settings, without changing globals.
    positive = dict(sampling_rate_hz=sampling_rate_hz, rise_window_s=rise_window_s,
                    z_histogram_step=z_histogram_step, diagnostic_width_scale=diagnostic_width_scale)
    nonnegative = dict(median_window_s=median_window_s, smooth_window_s=smooth_window_s,
                       min_event_distance_s=min_event_distance_s,
                       amplitude_prominence_sd=amplitude_prominence_sd,
                       rise_prominence_sd=rise_prominence_sd,
                       rise_lead_s=rise_lead_s, rise_lag_s=rise_lag_s)
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in nonnegative.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < calibration_frame_percent < 100:
        raise ValueError("calibration_frame_percent must be between 0 and 100")
    if len(z_histogram_range) != 2 or not np.isfinite(z_histogram_range).all() or z_histogram_range[0] >= z_histogram_range[1]:
        raise ValueError("z_histogram_range must contain two finite increasing bounds")
    n_median = 2 * round(median_window_s * sampling_rate_hz / 2) + 1  # odd, frames per running median
    n_smooth = max(1, round(smooth_window_s * sampling_rate_hz))  # frames per rolling mean
    edge_frames = n_smooth // 2  # frames at each trial end averaged over a partial window
    min_distance_frames = max(1, round(min_event_distance_s * sampling_rate_hz))
    n_rise = max(1, round(rise_window_s * sampling_rate_hz))  # frames spanned by one rise value
    rise_lead_frames = max(1, round(rise_lead_s * sampling_rate_hz))
    rise_lag_frames = max(1, round(rise_lag_s * sampling_rate_hz))
    z_bin_edges = np.arange(z_histogram_range[0], z_histogram_range[1] + z_histogram_step / 2,
                            z_histogram_step)  # (n_bins + 1,)
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    select_diagnostic_recordings(0, diagnostic_plot_count, diagnostic_seed)
    # One row per recording: where its CSV is and which day+animal it belongs to.
    csv_paths = sorted(input_root.glob(f"*/*/{csv_name}")) if recording_paths is None else sorted(recording_paths)
    if recording_limit is not None:
        csv_paths = csv_paths[:recording_limit]
    if not csv_paths:
        raise FileNotFoundError(f"no {csv_name} under {input_root}/<date>_<animal>/<t#>/")

    recordings = pd.DataFrame({
        "csv_path": csv_paths,
        "unit": [path.parent.parent.name for path in csv_paths],
        "recording": [path.parent.name for path in csv_paths],
    })
    # Split on the FIRST underscore only: dates are numeric, animals carry the letters.
    recordings[["date", "animal"]] = recordings["unit"].str.split("_", n=1, expand=True)

    # Group (P/R/T) per date+animal, read from the connectivity batch index (which
    # calls the date "day"). validate="many_to_one" raises if one date+animal were
    # listed under two groups.
    batch_summary = pd.read_csv(input_root / summary_name, dtype={"day": str})
    unit_groups = (batch_summary[["day", "animal", "group"]].drop_duplicates()
                   .rename(columns={"day": "date"}))
    recordings = recordings.merge(unit_groups, on=["date", "animal"], how="left",
                                  validate="many_to_one")
    missing_group = recordings.loc[recordings["group"].isna(), "unit"].unique()
    if missing_group.size:
        raise ValueError(f"no group in {summary_name} for: {sorted(missing_group)}")

    # Recording day: 1 for each animal's earliest date, 2 for the next, ...
    # Dates are YYMMDD, so string order is chronological; "dense" gives every t#
    # of one date the same number.
    recordings["recording_day"] = (recordings.groupby("animal")["date"]
                                   .rank(method="dense").astype(np.int64))

    # Animals in natural order (T4 before T10): letter prefix, then the number.
    animal_names = recordings["animal"].unique()
    animal_rank = {
        name: i_animal for i_animal, name in enumerate(sorted(
            animal_names,
            key=lambda name: (name.rstrip("0123456789"), int(name[len(name.rstrip("0123456789")):]))))
    }
    print(f"{len(recordings)} recordings in {recordings['unit'].nunique()} day+animal units "
          f"under {input_root}")

    # ---------------------------------------------------------------------------
    # Computation: detection on every recording, plus the pooled z histogram
    # ---------------------------------------------------------------------------
    # One pass per recording: detect the peaks (unflagged) and add its frames' robust
    # z to the pooled histogram. The cut-off is only known once every recording has
    # been seen, so flagging happens after the loop.
    event_tables: list[pd.DataFrame] = []
    z_histogram = np.zeros(len(z_bin_edges) - 1, dtype=np.int64)  # (n_bins,)
    n_frames = np.zeros(len(recordings), dtype=np.int64)  # (n_recordings,)
    roi_names: list[str] = []
    started = time.time()
    peak_options = dict(sampling_rate_hz=sampling_rate_hz, min_distance_frames=min_distance_frames,
                        amplitude_prominence_sd=amplitude_prominence_sd,
                        rise_prominence_sd=rise_prominence_sd, n_rise=n_rise,
                        rise_lead_frames=rise_lead_frames, rise_lag_frames=rise_lag_frames,
                        not_linear_rise=not_linear_rise)
    detection_jobs = [dict(csv_path=recording.csv_path, trace_loader=trace_loader,
                           n_median=n_median, n_smooth=n_smooth, edge_frames=edge_frames,
                           z_bin_edges=z_bin_edges, z_histogram_step=z_histogram_step,
                           rise_on_smoothed=rise_on_smoothed, peak_options=peak_options,
                           labels=dict(group=recording.group, animal=recording.animal,
                                       recording_day=recording.recording_day,
                                       date=recording.date, recording=recording.recording))
                      for recording in recordings.itertuples(index=False)]
    print(f"Detection on {workers} worker(s)", flush=True)
    # Ordered, so event_tables[i] belongs to recordings row i.
    for i_done, (i_rec, result) in enumerate(map_jobs(detect_recording, detection_jobs, workers, ordered=True)):
        if not roi_names:
            roi_names = result["roi_names"]
        elif result["roi_names"] != roi_names:
            # ROI columns are the tables' schema; another set or order would file
            # counts under the wrong area.
            raise ValueError(f"{recordings.iloc[i_rec].csv_path} has ROI columns {result['roi_names']}, "
                             f"expected {roi_names}")
        z_histogram += result["histogram"]
        n_frames[i_rec] = result["n_frames"]
        event_tables.append(result["events"])

        if (i_done + 1) % progress_every == 0 or i_done + 1 == len(recordings):
            print(f"  [{i_done + 1}/{len(recordings)}] {int(z_histogram.sum()):,} frames  "
                  f"{time.time() - started:.0f}s", flush=True)

    recordings["n_frames"] = n_frames
    recordings["recorded_min"] = n_frames / sampling_rate_hz / 60.0

    # Dataset-wide cut-off: the left edge of the highest bin whose tail still holds
    # at least `calibration_frame_percent` of all frames, so the achieved tail errs
    # slightly larger rather than smaller.
    if not z_histogram.sum():
        raise ValueError("No finite, non-flat ROI frames available for calibration")
    tail_fraction = z_histogram[::-1].cumsum()[::-1] / z_histogram.sum()  # (n_bins,) frames at or above each left edge
    cutoff_bin = int(np.flatnonzero(tail_fraction >= calibration_frame_percent / 100.0)[-1])
    if cutoff_bin == len(tail_fraction) - 1:
        # The last bin also holds every clipped frame above the range, so a cut-off
        # there is the range's edge, not the requested percentile.
        raise ValueError(f"More than {calibration_frame_percent:g}% of frames have z >= "
                         f"{z_bin_edges[-2]:g}, the top of z_histogram_range; widen z_histogram_range")
    z_threshold = float(z_bin_edges[cutoff_bin])
    achieved_percent = 100.0 * float(tail_fraction[np.searchsorted(z_bin_edges, z_threshold)])
    print(f"  cut-off z >= {z_threshold:.2f} (target top {calibration_frame_percent:g}% of "
          f"{int(z_histogram.sum()):,} frames, achieved {achieved_percent:.3f}%)")

    # all_events: one row per amplitude peak of every recording, flagged against the cut-off.
    all_events = pd.concat(event_tables, ignore_index=True)
    is_epileptic = all_events["amplitude_z"].to_numpy(dtype=float) >= z_threshold  # NaN compares False
    if epileptic_require_confirmed:
        is_epileptic &= all_events["confirmed"].to_numpy(dtype=bool)
    if epileptic_min_signal is not None:
        is_epileptic &= all_events["signal_at_peak"].to_numpy(dtype=float) >= epileptic_min_signal
    all_events["epileptic"] = is_epileptic
    all_events["epileptic_threshold"] = z_threshold

    # The calibrated flags are now final. Reload one recording at a time to bound memory.
    # Figures are drawn for the recordings of the sampled animals (all animals by
    # default), then optionally thinned to `diagnostic_plot_count` of those.
    diagnostic_animals = select_diagnostic_animals(recordings, diagnostic_animals_per_group, diagnostic_seed)
    animal_rows = np.flatnonzero(recordings["animal"].isin(diagnostic_animals).to_numpy())
    sampled_rows = select_diagnostic_recordings(len(animal_rows), diagnostic_plot_count, diagnostic_seed)
    selected_plots = {int(animal_rows[i_row]) for i_row in sampled_rows}
    print(f"Figures for {len(selected_plots)} recordings of {len(diagnostic_animals)} animals: "
          f"{', '.join(diagnostic_animals)}", flush=True)
    if write_recording_diagnostics:
        print(f"Writing per-recording diagnostics on {workers} worker(s)...", flush=True)
        diagnostics_started = time.perf_counter()
        # One job per recording, carrying its own slice of the flagged events.
        output_jobs = []
        event_offset = 0
        for i_rec, recording in enumerate(recordings.itertuples(index=False)):
            event_count = len(event_tables[i_rec])
            output_jobs.append(dict(
                recording_dir=((Path(recording.csv_path).parent if diagnostics_root is None
                                else Path(diagnostics_root) / recording.unit / recording.recording)
                               / "epileptic_detection"),
                events=all_events.iloc[event_offset:event_offset + event_count].copy(),
                plot=i_rec in selected_plots, csv_path=recording.csv_path,
                trace_loader=trace_loader, label=f"{recording.unit}/{recording.recording}",
                roi_names=roi_names, sampling_rate_hz=sampling_rate_hz, n_rise=n_rise,
                n_median=n_median, n_smooth=n_smooth, not_linear_rise=not_linear_rise,
                rise_on_smoothed=rise_on_smoothed, rise_window_s=rise_window_s,
                signal_unit=signal_unit, width_scale=diagnostic_width_scale))
            event_offset += event_count
        for i_done, (i_rec, seconds) in enumerate(map_jobs(write_recording_outputs, output_jobs,
                                                           workers, ordered=False)):
            elapsed = time.perf_counter() - diagnostics_started
            remaining_s = elapsed / (i_done + 1) * (len(recordings) - i_done - 1)
            print(f"  diagnostics [{i_done + 1}/{len(recordings)}] {output_jobs[i_rec]['label']} "
                  f"done in {seconds:.1f}s; estimated remaining {remaining_s / 60:.1f} min", flush=True)

    # Area and hemisphere of each ROI. The side letter is the L or R whose swap names
    # another ROI, so RSL_alta's leading R is not mistaken for the side (the rule
    # roi_editor.mirror_twin uses). Removing it gives the bilateral area:
    # M2L_alta -> M2_alta, V1aR -> V1a.
    roi_area: dict[str, str] = {}
    roi_hemisphere: dict[str, str] = {}
    for roi in roi_names:
        if roi_metadata is not None and roi in roi_metadata:
            roi_area[roi], roi_hemisphere[roi] = roi_metadata[roi]
            continue
        for i_char, char in enumerate(roi):
            if char not in "LR":
                continue
            twin = roi[:i_char] + ("R" if char == "L" else "L") + roi[i_char + 1:]
            if twin in roi_names:
                roi_area[roi] = roi[:i_char] + roi[i_char + 1:]
                roi_hemisphere[roi] = char
                break
        if roi not in roi_area:
            raise ValueError(f"ROI {roi!r} has no bilateral twin in {roi_names}")

    # Long table: one row per group x animal x recording day x ROI, t1..t5 summed.
    # Every ROI of every recording day gets a row even with no events, so a zero
    # means "looked at, found nothing" rather than "missing".
    unit_keys = ["group", "animal", "recording_day", "date"]
    unit_table = recordings.groupby(unit_keys, as_index=False).agg(
        n_recordings=("recording", "size"),
        recorded_min=("recorded_min", "sum"),
    )
    unit_table = unit_table.sort_values(
        ["group", "animal", "recording_day"], ignore_index=True,
        key=lambda column: column.map(animal_rank) if column.name == "animal" else column)
    roi_table = pd.DataFrame({
        "roi": roi_names,
        "area": [roi_area[roi] for roi in roi_names],
        "hemisphere": [roi_hemisphere[roi] for roi in roi_names],
    })
    event_counts = all_events.groupby(["animal", "recording_day", "roi"], as_index=False).agg(
        n_peaks=("frame", "size"),
        n_confirmed=("confirmed", "sum"),
        n_epileptic=("epileptic", "sum"),
    )
    # by_area_animal_day: (n_units * n_rois) rows, ROIs in CSV order within each unit.
    by_area_animal_day = unit_table.merge(roi_table, how="cross").merge(
        event_counts, on=["animal", "recording_day", "roi"], how="left", validate="one_to_one")
    count_columns = ["n_peaks", "n_confirmed", "n_epileptic"]
    by_area_animal_day[count_columns] = by_area_animal_day[count_columns].fillna(0).astype(np.int64)
    by_area_animal_day["epileptic_per_roi_min"] = (
        by_area_animal_day["n_epileptic"] / by_area_animal_day["recorded_min"])

    # Summaries split by area, by animal and by day. Group leads or accompanies every
    # key because P/R/T is the contrast the study is about.
    by_area = summarise(by_area_animal_day, ["group", "area", "hemisphere", "roi"])
    by_animal = summarise(by_area_animal_day, ["group", "animal"])
    by_day = summarise(by_area_animal_day, ["recording_day", "group"]).sort_values(
        ["recording_day", "group"], ignore_index=True)

    # Wide view of the long table for spreadsheets: one row per animal+recording day,
    # one column per ROI, value = epileptiform event count (t1..t5 summed).
    wide_index = [*unit_keys, "n_recordings", "recorded_min"]
    counts_wide = by_area_animal_day.pivot_table(
        index=wide_index, columns="roi", values="n_epileptic", sort=False,
    )[roi_names].astype(np.int64).reset_index()  # pivot_table averages; one value per cell, so exact
    counts_wide.columns.name = None
    counts_wide.insert(len(wide_index), "total", counts_wide[roi_names].sum(axis=1))

    # ---------------------------------------------------------------------------
    # Saving tables
    # ---------------------------------------------------------------------------
    all_events.to_csv(output_dir / "all_peaks.csv", index=False)
    by_area_animal_day.to_csv(output_dir / "epileptic_by_area_animal_day.csv", index=False)
    by_area.to_csv(output_dir / "epileptic_by_area.csv", index=False)
    by_animal.to_csv(output_dir / "epileptic_by_animal.csv", index=False)
    by_day.to_csv(output_dir / "epileptic_by_day.csv", index=False)
    counts_wide.to_csv(output_dir / "epileptic_counts_wide.csv", index=False)

    # Provenance: the exact settings, cut-off included, that produced the tables.
    detection_settings = {
        "median_window_s": median_window_s, "smooth_window_s": smooth_window_s,
        "min_event_distance_s": min_event_distance_s,
        "amplitude_prominence_sd": amplitude_prominence_sd, "rise_prominence_sd": rise_prominence_sd,
        "not_linear_rise": not_linear_rise, "rise_on_smoothed": rise_on_smoothed,
        "rise_window_s": rise_window_s, "rise_lead_s": rise_lead_s, "rise_lag_s": rise_lag_s,
        "epileptic_require_confirmed": epileptic_require_confirmed,
        "epileptic_min_signal": epileptic_min_signal,
        "workers": workers,
        "sampling_rate_hz": sampling_rate_hz,
        "calibration_frame_percent": calibration_frame_percent,
        "z_histogram_step": z_histogram_step, "z_histogram_range": list(z_histogram_range),
        "calibration_achieved_percent": achieved_percent,
        "calibration_n_frames": int(z_histogram.sum()),
        "z_threshold": z_threshold,
        "n_recordings": len(recordings),
    }
    detection_settings.update(provenance or {})
    detection_settings.update(diagnostic_plot_count=diagnostic_plot_count, diagnostic_seed=diagnostic_seed,
                              diagnostic_animals_per_group=diagnostic_animals_per_group,
                              diagnostic_animals=diagnostic_animals if write_recording_diagnostics else [],
                              diagnostic_width_scale=diagnostic_width_scale,
                              diagnostic_recordings=[f"{recordings.iloc[i].unit}/{recordings.iloc[i].recording}"
                                                     for i in sorted(selected_plots)] if write_recording_diagnostics else [])
    (output_dir / "detection_config.json").write_text(
        json.dumps(detection_settings, indent=2), encoding="utf-8")

    print(f"{int(all_events['epileptic'].sum())} epileptiform events of {len(all_events)} peaks "
          f"({int(all_events['confirmed'].sum())} confirmed)")
    print(by_animal.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Plot: rate per ROI for every day+animal
    # ---------------------------------------------------------------------------
    # rate_matrix: (n_units, n_rois), rows in long-table order: group, animal, recording day.
    rate_wide = by_area_animal_day.pivot_table(
        index=["group", "animal", "recording_day"], columns="roi",
        values="epileptic_per_roi_min", sort=False,
    )[roi_names]
    rate_matrix = rate_wide.to_numpy(dtype=float)
    row_labels = [f"{group}  {animal}  day {recording_day}"
                  for group, animal, recording_day in rate_wide.index]
    row_groups = rate_wide.index.get_level_values("group").to_numpy()

    sequential_cmap = LinearSegmentedColormap.from_list("sequential_blue", sequential_ramp)
    fig, axes = plt.subplots(1, 1, figsize=(9, 2 + 0.13 * len(row_labels)),
                             constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    image = ax.imshow(rate_matrix, aspect="auto", cmap=sequential_cmap, vmin=0,
                      interpolation="nearest")
    ax.set_xticks(np.arange(len(roi_names)), labels=roi_names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels, fontsize=6)
    # A thin surface-coloured rule between groups, so each group reads as a block.
    group_boundaries = np.flatnonzero(row_groups[1:] != row_groups[:-1]) + 0.5
    for boundary in group_boundaries:
        ax.axhline(boundary, color="#fcfcfb", linewidth=2)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.3, aspect=20)
    colorbar.set_label("epileptiform events / min")
    ax.set_title(f"Epileptiform events per minute per ROI "
                 f"(z >= {z_threshold:.2f}, top {calibration_frame_percent:g}% of frames, "
                 f"{median_window_s:g} s detrend, t1..t5 summed)", fontsize=9)
    fig.savefig(output_dir / "epileptic_rate_heatmap.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Plot: rate per ROI, one bar per group
    # ---------------------------------------------------------------------------
    # rate_by_group: (n_groups, n_rois), each group's rate per ROI-minute.
    group_names = sorted(by_area["group"].unique())
    rate_by_group = (by_area.pivot_table(index="group", columns="roi",
                                         values="epileptic_per_roi_min")
                     .loc[group_names, roi_names].to_numpy(dtype=float))
    bar_width = 0.8 / len(group_names)
    roi_positions = np.arange(len(roi_names))

    fig, axes = plt.subplots(1, 1, figsize=(14, 4.5), constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    for i_group, group in enumerate(group_names):
        n_group_units = int(by_animal.loc[by_animal["group"] == group, "n_units"].sum())
        ax.bar(roi_positions + (i_group - (len(group_names) - 1) / 2) * bar_width,
               rate_by_group[i_group], width=bar_width * 0.9,
               color=group_colors[i_group], label=f"{group} ({n_group_units} day+animal units)")
    ax.set_xticks(roi_positions, labels=roi_names, rotation=90)
    ax.set_ylabel("epileptiform events / min")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title(f"Epileptiform rate per ROI by group "
                 f"(z >= {z_threshold:.2f}, top {calibration_frame_percent:g}% of frames, "
                 f"{median_window_s:g} s detrend)")
    fig.savefig(output_dir / "epileptic_rate_by_area_group.png", dpi=200)
    plt.close(fig)

    print(f"wrote {output_dir}")


if __name__ == "__main__":
    run_analysis()
