"""Detect epileptiform events in per-ROI widefield fluorescence traces.

Per trial and per ROI:
  1. smooth the DF/F trace with a centred rolling mean (1 s at 10 Hz),
  2. take the first difference of the smoothed trace (the rise rate),
  3. run scipy.signal.find_peaks on both signals,
  4. keep the amplitude peaks that have a concurrent rise-rate peak - slow
     drift reaches a high value without a sharp rise, so it is dropped,
  5. call epileptiform the confirmed peaks that stand far enough above their own
     ROI's baseline, measured in robust SDs (z). The default threshold is a
     single constant calibrated across the whole dataset so that a target
     fraction of ALL frames exceeds it - see `epileptic_criterion`.

Why frames and not peaks. Ranking peaks against peaks makes the denominator
depend on how sensitive detection happened to be: an ROI with one peak flags
it, an ROI with forty flags four. Frames are a fixed denominator - every
recording here is 2980 frames - so "big" stops depending on how much else was
found. Ranking in robust z on top of that makes the number comparable between
ROIs, animals and days, because a multiplicative gain change divides out.

Run straight from the editor: edit RUN_CONFIG below, no CLI flags needed.
"""

import os

# MUST run before numpy/scipy/matplotlib import, i.e. before MKL is loaded.
# In this conda env MKL's mkl_intel_thread.3.dll delay-loads libiomp5md.dll by
# bare name, and Python 3.8+ restricts the DLL search so <env>\Library\bin is
# never looked at. The load fails as native exception 0xc06d007f, killing the
# process with no Python traceback - matplotlib triggers it on every draw via
# np.linalg. The sequential layer skips mkl_intel_thread entirely. This is the
# same fault as CLAUDE.md 9.11/9.12; fixing it in the env would retire this line.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Edit this section to run the analysis without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "csv_path": "outputs/botox_restani_rebuilt/260611_PV5/t1/roi_fluorescence_full.csv",
    "output_dir": None,  # None -> "<csv folder>/epileptic_detection"
    "sampling_rate_hz": 10.0,
    "smooth_window_s": 1.0,       # rolling-mean width
    "min_event_distance_s": 1.0,  # refractory period between peaks of one ROI
    # Prominence thresholds in robust SDs (1.4826 * MAD) of each ROI's own
    # signal, so ROIs with different baseline noise get comparable sensitivity.
    "amplitude_prominence_sd": 3.0,
    "rise_prominence_sd": 3.0,
    # A rise peak counts as concurrent when it falls in [peak - lead, peak + lag].
    # The derivative maximum *precedes* the amplitude maximum, hence lead >> lag.
    "rise_lead_s": 1.0,
    "rise_lag_s": 0.3,
    # Single knob on how strict concurrency is: it scales both bounds above, so
    # 2.0 doubles the window and 0.5 halves it, keeping the lead/lag asymmetry.
    # 1.0 uses the bounds exactly as written. Raising it confirms more peaks and
    # admits more slow drift; the rise_offset_s column shows the observed lead of
    # each match, so a run at a loose tolerance tells you where to set it.
    "concurrency_tolerance": 1.0,
    # ---- Step 5: which peaks count as epileptiform -------------------------
    # How the cut-off is set.
    #   "calibrated_z"     - a FIXED threshold in robust SDs, the same number in
    #                        every recording, taken from `epileptic_z_threshold`.
    #                        run_epileptic_batch calibrates it once over the
    #                        whole dataset so `epileptic_frame_percent` of all
    #                        frames exceed it. Being fixed, counts are comparable
    #                        across ROIs, animals and days, and a recording that
    #                        is half seizure does not raise its own bar.
    #   "frame_percentile" - the cut-off is the top `epileptic_frame_percent`
    #                        percent of THIS recording's frames. Removes session
    #                        gain, but is re-estimated per recording and sits
    #                        inside the pathology when events are frequent.
    #   "peak_percentile"  - the original rule: top `epileptic_top_percent` of
    #                        the peaks themselves. Kept so old runs reproduce;
    #                        note it pins the answer near one event per active
    #                        ROI regardless of how big anything was.
    "epileptic_criterion": "calibrated_z",
    # Target tail size for the two frame-based criteria: 1.0 means "the level
    # exceeded by 1% of frames" = 29.8 of the 2980 frames of a recording. Those
    # frames arrive in runs, so they collapse to far fewer than 30 events.
    "epileptic_frame_percent": 1.0,
    # The calibrated constant, in robust SDs above each ROI's own median. None
    # means "derive it from this recording's own frames", which is what a
    # standalone editor run does; the batch passes the dataset-wide value.
    "epileptic_z_threshold": None,
    # Which frames the percentile is taken over, for "frame_percentile" and for
    # the None fallback above.
    #   "recording" - all ROIs pooled, one cut-off for the recording.
    #   "roi"       - each area against its own frames, so every area
    #                 contributes its own top slice.
    "epileptic_frame_population": "recording",
    # ---- "peak_percentile" only --------------------------------------------
    # The cut-off is the top `epileptic_top_percent` percent of ALL amplitude
    # peaks - concurrency-rejected ones included in the ranking, since they are
    # real peaks and belong in the size distribution even though they cannot be
    # selected. 10.0 keeps the top decile, 100.0 keeps everything.
    "epileptic_top_percent": 10.0,
    # What "largest" means. "prominence_sd" is the peak's prominence in units of
    # its own ROI's robust SD, so a quiet ROI and a noisy one are ranked on the
    # same scale - the default. "prominence" is the same height in raw DF/F
    # units, "amplitude" the absolute value at the peak (which still carries
    # each ROI's baseline offset, so use it only with scope "roi").
    "epileptic_rank_metric": "prominence_sd",
    # Which peaks the percentile is taken over.
    #   "roi"       - per cortical area: each ROI is ranked against its own
    #                 peaks, so every area contributes its own top n percent.
    #   "recording" - all recorded areas pooled into one distribution, so the
    #                 flag means "big for this animal" and quiet areas may
    #                 contribute nothing.
    #   "trial"     - all areas pooled, but one cut-off per trial.
    "epileptic_scope": "roi",
    # ---- applies to every criterion ----------------------------------------
    # Whether the concurrency test of step 4 still gates selection. True is the
    # documented pipeline: big AND with a sharp rise. False reports every peak
    # above the cut-off, which is the way to see how many big peaks the
    # concurrency filter is discarding.
    "epileptic_require_confirmed": True,
    "make_plots": True,
    # Whether main() prints its per-recording report. A batch caller turns this
    # off and prints one line per recording instead.
    "verbose": True,
}

STARTING_COLUMN = 2  # first 2 columns (trial, frame) are metadata, not data

# Column -> dtype of the table detect_events returns. Declared rather than left
# to inference because a quiet recording yields no rows at all: without this a
# zero-peak recording came back as a frame with NO columns, and every consumer
# (plot_traces, the CSV header, any tally) failed on the missing "roi".
EVENT_SCHEMA: dict[str, str] = {
    "trial": "int64",
    "roi": "object",
    "frame": "int64",
    "time_s": "float64",
    "amplitude": "float64",
    # The peak's height above its own ROI's median, in robust SDs. This is the
    # quantity the frame-based criteria threshold, and it is on exactly the same
    # scale as the frame z values, because a peak IS one of those frames.
    "amplitude_z": "float64",
    "prominence": "float64",
    "prominence_sd": "float64",
    "rise_time_s": "float64",
    "rise_offset_s": "float64",
    "rise_per_s": "float64",
    "confirmed": "bool",
}


# Fixed bin grid for the pooled z histogram the dataset calibration runs on.
# 0.01 SD resolution is finer than any threshold difference that matters, and
# the whole grid is 4000 int64 counts, so histograms from 355 recordings can be
# summed in memory. Values outside the range land in the end bins, which keeps
# the tail fraction exact even though the extreme values are not resolved.
Z_HISTOGRAM_BINS = np.arange(-10.0, 30.0 + 1e-9, 0.01)


def robust_sd(x: np.ndarray) -> float:
    """Median-absolute-deviation SD estimate: immune to the events themselves."""
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def robust_z_frames(
    df: pd.DataFrame,
    smoothed: pd.DataFrame,
    roi_names: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Every smoothed frame expressed in robust SDs above its own ROI's median.

    Centre and scale are the median and 1.4826*MAD of that ROI in that trial -
    the same robust pair `detect_events` uses - so they are set by the baseline
    and barely move even if a large part of the recording is event. That is the
    property a high percentile of the recording's own values does NOT have.

    A multiplicative gain change scales the trace, its median and its MAD
    together, so z is gain-invariant: this is what makes a fixed threshold
    comparable across sessions and days.

    The first and last `n_dt // 2` frames of each trial are left NaN. The
    rolling mean runs with min_periods=1, so those frames average fewer samples
    and carry inflated variance; they may still host a detected peak, they just
    do not get a vote on where a frame-based cut-off sits.
    """
    fs = float(cfg["sampling_rate_hz"])
    edge = max(1, round(cfg["smooth_window_s"] * fs)) // 2

    z = pd.DataFrame(np.nan, index=smoothed.index, columns=roi_names, dtype=float)
    for _, trial_index in df.groupby("trial").groups.items():
        block = smoothed.loc[trial_index, roi_names].to_numpy(dtype=float)
        centre = np.median(block, axis=0)
        scale = np.array([robust_sd(block[:, i]) for i in range(block.shape[1])])
        # A flat ROI has no scale, so it has no z either - and it has no peaks
        # to flag. NaN keeps it out of every percentile and comparison.
        scale = np.where(scale > 0, scale, np.nan)
        block = (block - centre) / scale
        if edge:
            block[:edge] = np.nan
            block[-edge:] = np.nan
        z.loc[trial_index, roi_names] = block
    return z


def z_histogram(z: pd.DataFrame) -> np.ndarray:
    """Counts of `z`'s finite values on Z_HISTOGRAM_BINS, for pooled calibration.

    Values are clipped into the end bins rather than dropped, so the counts sum
    to the number of real frames and a tail fraction computed from them is the
    true fraction of frames.
    """
    values = z.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    clipped = np.clip(values, Z_HISTOGRAM_BINS[0], Z_HISTOGRAM_BINS[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=Z_HISTOGRAM_BINS)
    return counts.astype(np.int64)


def z_threshold_from_histogram(counts: np.ndarray, top_percent: float) -> float:
    """The z exceeded by `top_percent` percent of the counted frames.

    Returns a left bin edge, so the fraction of frames at or above it is at
    least `top_percent` - erring towards a slightly larger tail rather than a
    smaller one, and never returning a threshold that nothing reaches.
    """
    total = int(counts.sum())
    if total == 0:
        raise ValueError("no frames to calibrate on")
    # Fraction of frames at or above each bin's left edge.
    tail = counts[::-1].cumsum()[::-1] / total
    reached = np.flatnonzero(tail >= top_percent / 100.0)
    return float(Z_HISTOGRAM_BINS[reached[-1]])


def smooth_by_trial(df: pd.DataFrame, roi_names: list[str], n_dt: int) -> pd.DataFrame:
    """Centred rolling mean of every ROI, computed inside each trial.

    The previous version passed axis=0 to .rolling(), which pandas removed in
    2.x/3.0: rolling already runs down the rows. Grouping by trial keeps the
    window from bleeding across trial boundaries, and center=True keeps peak
    timing unbiased - a trailing window shifts every peak later by ~n_dt/2.
    """
    return (
        df.groupby("trial")[roi_names]
        .rolling(window=n_dt, center=True, min_periods=1)
        .mean()
        .reset_index(level="trial", drop=True)
        .sort_index()
    )


def detect_events(
    df: pd.DataFrame,
    smoothed: pd.DataFrame,
    roi_names: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """One row per amplitude peak, flagged with whether a rise peak is concurrent."""
    fs = float(cfg["sampling_rate_hz"])
    min_distance = max(1, round(cfg["min_event_distance_s"] * fs))
    tolerance = float(cfg["concurrency_tolerance"])
    lead_frames = cfg["rise_lead_s"] * fs * tolerance
    lag_frames = cfg["rise_lag_s"] * fs * tolerance

    rows: list[dict[str, Any]] = []

    for trial, trial_index in df.groupby("trial").groups.items():
        frames = df.loc[trial_index, "frame"].to_numpy()
        smooth_trial = smoothed.loc[trial_index, roi_names].to_numpy(dtype=float)
        # First difference of the smoothed trace. Element i spans frames
        # i -> i+1, so its effective position is i + 0.5.
        rise_trial = np.diff(smooth_trial, axis=0)

        for roi_index, roi in enumerate(roi_names):
            amplitude = smooth_trial[:, roi_index]
            rise = rise_trial[:, roi_index]

            amplitude_sd = robust_sd(amplitude)
            # Baseline of this ROI in this trial, so a peak can be expressed in
            # the same robust z as robust_z_frames gives every frame.
            amplitude_centre = float(np.median(amplitude))
            # A flat ROI has zero MAD, so no prominence can clear the threshold
            # and nothing is detected there - rather than everything.
            amplitude_peaks, amplitude_props = find_peaks(
                amplitude,
                prominence=cfg["amplitude_prominence_sd"] * amplitude_sd,
                distance=min_distance,
            )
            rise_peaks, rise_props = find_peaks(
                rise,
                prominence=cfg["rise_prominence_sd"] * robust_sd(rise),
                distance=min_distance,
            )
            rise_positions = rise_peaks + 0.5

            for k, peak in enumerate(amplitude_peaks):
                concurrent = np.flatnonzero(
                    (rise_positions >= peak - lead_frames)
                    & (rise_positions <= peak + lag_frames)
                )
                confirmed = concurrent.size > 0
                if confirmed:
                    # Steepest rise in the window marks the event onset.
                    best = concurrent[np.argmax(rise_props["prominences"][concurrent])]
                    rise_frame = float(frames[rise_peaks[best]]) + 0.5
                    rise_per_s = float(rise[rise_peaks[best]] * fs)
                else:
                    rise_frame = np.nan
                    rise_per_s = np.nan

                prominence = float(amplitude_props["prominences"][k])
                rows.append(
                    {
                        "trial": trial,
                        "roi": roi,
                        "frame": int(frames[peak]),
                        "time_s": float(frames[peak]) / fs,
                        "amplitude": float(amplitude[peak]),
                        "amplitude_z": ((float(amplitude[peak]) - amplitude_centre)
                                        / amplitude_sd if amplitude_sd > 0 else np.nan),
                        "prominence": prominence,
                        "prominence_sd": prominence / amplitude_sd if amplitude_sd > 0 else np.nan,
                        "rise_time_s": rise_frame / fs,
                        # How far the rise led the peak, in seconds. Positive is
                        # the normal case. Compare against rise_lead_s to see how
                        # much of the tolerance window the matches actually use.
                        "rise_offset_s": (float(frames[peak]) - rise_frame) / fs,
                        "rise_per_s": rise_per_s,
                        "confirmed": confirmed,
                    }
                )

    # astype pins the dtypes for the empty case too, so `confirmed` stays a real
    # boolean and can be used as a mask on a recording that produced no peaks.
    return pd.DataFrame(rows, columns=list(EVENT_SCHEMA)).astype(EVENT_SCHEMA)


def _flag_by_peak_percentile(
    events: pd.DataFrame, cfg: dict[str, Any],
) -> pd.DataFrame:
    """The original rule: rank peaks against other peaks.

    The cut-off is the (100 - top_percent) percentile of EVERY amplitude peak in
    the scope - the ones concurrency rejected included. They are part of the
    population that defines "big" even though they cannot be selected; ranking
    only the survivors would raise the bar on a clean recording and lower it on
    a drifty one, so the same event would pass or fail depending on how much
    drift sat next to it.

    Kept so earlier runs reproduce, but note what it does at scope "roi": most
    ROIs have fewer than 10 peaks, so the 90th percentile of that ROI IS its
    maximum and `>=` keeps exactly one peak. The count then measures how many
    areas were active, not how large anything was. That is what the frame-based
    criteria exist to fix.
    """
    metric = cfg["epileptic_rank_metric"]
    scope = cfg["epileptic_scope"]
    top_percent = float(cfg["epileptic_top_percent"])
    require_confirmed = bool(cfg["epileptic_require_confirmed"])

    if metric not in {"prominence_sd", "prominence", "amplitude"}:
        raise ValueError(f"unknown epileptic_rank_metric: {metric!r}")
    if not 0.0 < top_percent <= 100.0:
        raise ValueError(f"epileptic_top_percent must be in (0, 100]: {top_percent!r}")
    if scope not in {"recording", "roi", "trial"}:
        raise ValueError(f"unknown epileptic_scope: {scope!r}")

    # One label per row saying which population that peak is ranked against;
    # "recording" is the single-group case, every row sharing one label.
    labels = (
        pd.Series(0, index=events.index) if scope == "recording"
        else events[scope]
    )

    for _, index in events.groupby(labels, sort=False).groups.items():
        group = events.loc[index]
        values = group[metric].to_numpy()
        # A flat ROI gives prominence_sd = NaN; such a peak can neither set the
        # cut-off nor clear it, so it is the only thing left out of the ranking.
        ranked = np.isfinite(values)
        if not ranked.any():
            continue
        threshold = float(np.percentile(values[ranked], 100.0 - top_percent))
        # Written on every row of the group, rejected ones included, so the CSV
        # shows the cut-off each peak was measured against.
        events.loc[index, "epileptic_threshold"] = threshold

        selected = ranked & (values >= threshold)
        if require_confirmed:
            selected &= group["confirmed"].to_numpy()
        events.loc[index, "epileptic"] = selected

    return events


def _frame_thresholds(
    z: pd.DataFrame, roi_names: list[str], cfg: dict[str, Any],
) -> dict[str | None, float]:
    """Frame-percentile cut-offs, keyed by ROI name or by None for the pooled one.

    The population is the recording's own frames, so this removes session gain -
    at the cost of being re-estimated from roughly 300 effective samples per ROI
    (a 10-frame rolling mean plus the calcium decay leaves nowhere near 2980
    independent ones), and of rising when events are frequent enough to occupy
    the tail themselves.
    """
    frame_percent = float(cfg["epileptic_frame_percent"])
    population = cfg["epileptic_frame_population"]
    if not 0.0 < frame_percent <= 100.0:
        raise ValueError(
            f"epileptic_frame_percent must be in (0, 100]: {frame_percent!r}")
    if population not in {"recording", "roi"}:
        raise ValueError(f"unknown epileptic_frame_population: {population!r}")

    if population == "recording":
        values = z.to_numpy(dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {}
        return {None: float(np.percentile(values, 100.0 - frame_percent))}

    thresholds: dict[str | None, float] = {}
    for roi in roi_names:
        values = z[roi].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            thresholds[roi] = float(np.percentile(values, 100.0 - frame_percent))
    return thresholds


def flag_epileptic(
    events: pd.DataFrame,
    z: pd.DataFrame,
    roi_names: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Add `epileptic`, `epileptic_threshold` and `epileptic_criterion` columns.

    `z` is the frame table from `robust_z_frames`; it is what the frame-based
    criteria take their cut-off from, and it is on the same scale as each peak's
    `amplitude_z`, because a peak is one of those frames.

    A peak is epileptiform when it is at or above the cut-off and - unless
    `epileptic_require_confirmed` is off - it passed the concurrency test of
    step 4. The comparison is >=, so ties are kept.

    See the `epileptic_criterion` comment in RUN_CONFIG for what each rule does.
    The default, "calibrated_z", compares each peak against one constant in
    robust SDs: identical in every recording, so two recordings' counts mean the
    same thing, and derived from medians and MADs rather than from the tail, so
    a recording full of events does not quietly raise its own bar.
    """
    criterion = cfg["epileptic_criterion"]
    require_confirmed = bool(cfg["epileptic_require_confirmed"])
    if criterion not in {"calibrated_z", "frame_percentile", "peak_percentile"}:
        raise ValueError(f"unknown epileptic_criterion: {criterion!r}")

    events = events.copy()
    events["epileptic"] = pd.Series(False, index=events.index, dtype=bool)
    events["epileptic_threshold"] = pd.Series(np.nan, index=events.index, dtype=float)
    events["epileptic_criterion"] = criterion
    if events.empty:
        return events

    if criterion == "peak_percentile":
        return _flag_by_peak_percentile(events, cfg)

    if criterion == "calibrated_z":
        threshold = cfg["epileptic_z_threshold"]
        if threshold is None:
            # Standalone editor run with no dataset calibration available: fall
            # back to this recording's own frames so the script still runs, but
            # the number is then recording-specific and NOT comparable with a
            # batch that used the calibrated constant.
            pooled = {**cfg, "epileptic_frame_population": "recording"}
            thresholds = _frame_thresholds(z, roi_names, pooled)
        else:
            thresholds = {None: float(threshold)}
    else:
        thresholds = _frame_thresholds(z, roi_names, cfg)

    if not thresholds:
        return events

    # None keys a single cut-off for every row; otherwise it is looked up per ROI.
    if None in thresholds:
        cut = np.full(len(events), thresholds[None], dtype=float)
    else:
        cut = events["roi"].map(thresholds).to_numpy(dtype=float)
    events["epileptic_threshold"] = cut

    values = events["amplitude_z"].to_numpy(dtype=float)
    # A flat ROI has no z at all; it can neither clear a cut-off nor set one.
    selected = np.isfinite(values) & np.isfinite(cut) & (values >= cut)
    if require_confirmed:
        selected &= events["confirmed"].to_numpy()
    events["epileptic"] = selected
    return events


def _criterion_label(cfg: dict[str, Any], events: pd.DataFrame) -> str:
    """Short description of the cut-off, for the figure legend."""
    if cfg["epileptic_criterion"] == "peak_percentile":
        return f"top {cfg['epileptic_top_percent']:g}% of peaks"
    cuts = events["epileptic_threshold"].dropna()
    if cuts.empty:
        return "no cut-off"
    if cuts.nunique() == 1:
        return f"z >= {cuts.iloc[0]:.2f}"
    return f"z >= {cuts.min():.2f}..{cuts.max():.2f} per ROI"


def plot_traces(
    df: pd.DataFrame,
    smoothed: pd.DataFrame,
    events: pd.DataFrame,
    roi_names: list[str],
    cfg: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    """One stacked panel per ROI: raw trace, smoothed trace, detections scattered."""
    fs = float(cfg["sampling_rate_hz"])
    n_rois = len(roi_names)
    time_s = df["frame"].to_numpy() / fs

    fig, axes = plt.subplots(
        n_rois, 1, figsize=(16, 1.6 * n_rois), sharex=True, squeeze=False,
    )
    for roi_index, roi in enumerate(roi_names):
        ax = axes[roi_index, 0]
        ax.plot(time_s, df[roi].to_numpy(), color="#c9ccd1", linewidth=0.8,
                label="raw")
        ax.plot(time_s, smoothed[roi].to_numpy(), color="#2f6fd0", linewidth=1.6,
                label=f"smoothed ({cfg['smooth_window_s']:g} s)")

        roi_events = events[events["roi"] == roi]
        # Three tiers, drawn weakest-claim first so the epileptiform markers sit
        # on top: rejected by concurrency, confirmed but below the percentile,
        # and confirmed and in the top percentile.
        # The three tiers are disjoint: with epileptic_require_confirmed off an
        # unconfirmed peak can be epileptiform, so exclude it from `misses` too
        # rather than drawing two markers on the same point.
        epileptic = roi_events[roi_events["epileptic"]]
        rest = roi_events[~roi_events["epileptic"]]
        misses = rest[~rest["confirmed"]]
        hits = rest[rest["confirmed"]]
        ax.scatter(misses["time_s"], misses["amplitude"], s=45,
                   facecolors="none", edgecolors="#d89020", zorder=3,
                   label="amplitude peak only")
        ax.scatter(hits["time_s"], hits["amplitude"], s=45,
                   facecolors="none", edgecolors="#c2354a", linewidths=1.5,
                   zorder=4, label="concurrent peak")
        ax.scatter(epileptic["time_s"], epileptic["amplitude"], s=110,
                   marker="*", color="#c2354a", zorder=5,
                   label=f"epileptiform ({_criterion_label(cfg, events)})")

        ax.set_ylabel(roi, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(loc="upper right", ncol=5, frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.4 / n_rois, bottom=0.4 / n_rois,
                        hspace=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def frame_z_for_csv(csv_path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Read one fluorescence CSV and return its frames in robust z.

    The calibration pass of run_epileptic_batch needs the frame distribution
    without running detection; sharing this with main() is what guarantees the
    frames that set the threshold are the same frames the peaks are found in.
    """
    fs = float(cfg["sampling_rate_hz"])
    df = pd.read_csv(csv_path)
    roi_names = list(df.columns[STARTING_COLUMN:])
    n_dt = max(1, round(cfg["smooth_window_s"] * fs))
    smoothed = smooth_by_trial(df, roi_names, n_dt)
    return robust_z_frames(df, smoothed, roi_names, cfg)


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    """Where main() writes for this config - so a caller can locate the results
    without re-deriving the rule."""
    if cfg["output_dir"] is not None:
        return Path(cfg["output_dir"])
    return Path(cfg["csv_path"]).parent / "epileptic_detection"


def main(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run the whole detection on one CSV and return the event table.

    `cfg` may be a partial dict: anything it omits falls back to RUN_CONFIG, so
    a caller can override just `csv_path` without restating the thresholds. The
    returned frame is the same one written to roi_events.csv, so a batch driver
    can tally it without re-reading the file it just wrote.
    """
    cfg = {**RUN_CONFIG, **(cfg or {})}
    csv_path = Path(cfg["csv_path"])
    fs = float(cfg["sampling_rate_hz"])

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    roi_names = list(df.columns[STARTING_COLUMN:])
    # At least 1 frame, so a low sampling rate cannot collapse the window to 0.
    n_dt = max(1, round(cfg["smooth_window_s"] * fs))

    smoothed = smooth_by_trial(df, roi_names, n_dt)
    z = robust_z_frames(df, smoothed, roi_names, cfg)
    events = detect_events(df, smoothed, roi_names, cfg)
    events = flag_epileptic(events, z, roi_names, cfg)
    events.to_csv(output_dir / "roi_events.csv", index=False)
    # The epileptiform subset on its own, so downstream work does not have to
    # re-apply the two filters.
    events[events["epileptic"]].to_csv(
        output_dir / "roi_epileptic_events.csv", index=False)

    if cfg["verbose"]:
        print(f"{csv_path}")
        print(f"  {len(roi_names)} ROIs, {len(df)} frames, "
              f"{len(df) / fs:.1f} s at {fs:g} Hz")
        tolerance = float(cfg["concurrency_tolerance"])
        print(f"  {len(events)} amplitude peaks, "
              f"{int(events['confirmed'].sum())} with a concurrent rise peak "
              f"(tolerance {tolerance:g}: "
              f"-{cfg['rise_lead_s'] * tolerance:g} s to "
              f"+{cfg['rise_lag_s'] * tolerance:g} s)")
        matched = events.loc[events["confirmed"], "rise_offset_s"]
        if not matched.empty:
            print(f"  observed rise lead: median {matched.median():.2f} s, "
                  f"max {matched.max():.2f} s")
        criterion = cfg["epileptic_criterion"]
        n_peaks = len(events)
        ranked = (events[cfg["epileptic_rank_metric"]] if criterion == "peak_percentile"
                  else events["amplitude_z"])
        n_big = int((events["epileptic_threshold"].notna()
                     & (ranked >= events["epileptic_threshold"])).sum())
        n_epileptic = int(events["epileptic"].sum())
        share = 100.0 * n_epileptic / n_peaks if n_peaks else float("nan")
        if criterion == "peak_percentile":
            print(f"  top {cfg['epileptic_top_percent']:g}% by "
                  f"{cfg['epileptic_rank_metric']} over all {n_peaks} peaks "
                  f"(scope {cfg['epileptic_scope']}): {n_big} above the cut-off")
        else:
            cuts = events["epileptic_threshold"].dropna()
            span = (f"{cuts.min():.2f}" if cuts.nunique() <= 1
                    else f"{cuts.min():.2f}..{cuts.max():.2f}")
            source = ("dataset calibration" if cfg["epileptic_z_threshold"] is not None
                      else f"this recording's own frames, {cfg['epileptic_frame_population']}")
            # What fraction of THIS recording's frames clears the cut-off. With
            # the calibrated constant that is the diagnostic: the dataset-wide
            # value is the target percent by construction, so a recording well
            # above it is unusually active rather than mis-thresholded.
            values = z.to_numpy(dtype=float).ravel()
            values = values[np.isfinite(values)]
            above = (100.0 * np.mean(values >= cuts.min()) if cuts.size and values.size
                     else float("nan"))
            print(f"  criterion {criterion}: z >= {span} ({source}; "
                  f"target top {cfg['epileptic_frame_percent']:g}% of frames)")
            print(f"  {above:.2f}% of this recording's frames clear it; "
                  f"{n_big} of {n_peaks} peaks do")
        print(f"  {n_epileptic} epileptiform events = {share:.1f}% of all peaks"
              + (f" ({n_big - n_epileptic} of the big ones lacked a concurrent rise)"
                 if cfg["epileptic_require_confirmed"] else " (concurrency not required)"))
        flagged = events[events["epileptic"]]
        if not flagged.empty:
            print(f"  in {flagged['roi'].nunique()} ROIs / "
                  f"{flagged['trial'].nunique()} trials")
        print(f"  wrote {output_dir}")

    if cfg["make_plots"]:
        label = f"{csv_path.parent.parent.name}/{csv_path.parent.name}"
        plot_traces(df, smoothed, events, roi_names, cfg,
                    output_dir / "roi_traces_events.png",
                    f"Epileptiform event detection - {label}")

    return events


if __name__ == "__main__":
    main()
