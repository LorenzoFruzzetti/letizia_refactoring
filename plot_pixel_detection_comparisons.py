"""Copy sampled detection plots and compare them with starting fluorescence.

Run in the letizia environment: python plot_pixel_detection_comparisons.py
Uses the saved sample, settings, fluorescence and event flags; no redetection.

The trace file, its unit and the plot titles all come from the analysis's own
`detection_config.json`, so this works against either `--signal-mode` tree of
`epileptic_by_area_animal_day_pixels.py`.

Each comparison figure is one row per ROI and about 3 in tall per row, so a
22-ROI recording is a ~30 MB image. `--sample-count` therefore samples the
analysis's diagnostic recordings (default 5) instead of drawing all of them;
`--sample-count all` draws every one, which for a 545-recording run means
hours and tens of GB. Recordings named with `--add-recording`, and any kept in
the output folder's `comparison_recordings.json`, are always drawn on top of
the sample.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epileptic_by_area_animal_day import remove_slow_trend, smooth_by_trial, rise_signal
from epileptic_by_area_animal_day_pixels import SIGNAL_UNITS, TRACE_CSV_NAMES
from epileptic_diagnostics import plot_rise

REPO_ROOT = Path(__file__).resolve().parent
# The analysis tree this reads, and the new folder the comparisons go into.
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs/epileptic_by_area_animal_day_pixels_median_dff"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/epileptic_median_dff_analysis/sampled_plot_comparisons"


def sample_count(value):
    """`all` means every diagnostic recording of the analysis; 0 means none."""
    if value is None or str(value).strip().lower() == "all":
        return None
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be >= 0 or 'all'")
    return result


def choose_recordings(available, count, seed):
    """Evenly spaced picks, so a sample spans the cohort instead of clustering.

    The recordings are in date order, so taking every (n/count)th one covers all
    dates, animals and groups; `seed` shifts which member of each stride is taken
    so a second run with the same count can look at different recordings.
    """
    if count is None or count >= len(available):
        return list(available)
    positions = np.linspace(0, len(available), count, endpoint=False)
    offset = np.random.default_rng(seed).integers(0, max(1, len(available) // count))
    return [available[min(len(available) - 1, int(position) + int(offset))] for position in positions]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=sample_count, default=5,
                        help="How many of the analysis's diagnostic recordings to draw; "
                             "'all' draws every one (hours and tens of GB for a full run)")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--add-recording", action="append", default=[],
                        help="Add a recording such as 260611_PV5/t1 to the saved comparison sample")
    args = parser.parse_args()
    root = args.input_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "detection_config.json").read_text(encoding="utf-8"))
    fs = config["sampling_rate_hz"]
    n_median = 2 * round(config["median_window_s"] * fs / 2) + 1
    n_smooth = max(1, round(config["smooth_window_s"] * fs))
    n_rise = max(1, round(config["rise_window_s"] * fs))
    # Which signal the analysis produced: names its trace file, its unit and the titles.
    signal_mode = config.get("signal_mode", "reflectance_ratio")
    signal_unit = SIGNAL_UNITS[signal_mode]
    trace_csv = TRACE_CSV_NAMES[signal_mode]
    signal_description = config.get("signal", "F * mean(R) / R")
    manifest_path = out / "comparison_recordings.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    # Sample the analysis's recordings, then always add the explicitly named ones.
    sampled = choose_recordings(config["diagnostic_recordings"], args.sample_count, args.sample_seed)
    recordings = list(dict.fromkeys([*sampled, *previous, *args.add_recording]))
    print(f"{signal_mode} ({signal_unit}); {len(recordings)} of "
          f"{len(config['diagnostic_recordings'])} recordings -> {out}")
    for recording in recordings:
        folder = root / recording
        stem = recording.replace("/", "_")
        raw = pd.read_csv(folder / trace_csv)
        events = pd.read_csv(folder / "epileptic_detection/roi_events.csv")
        rois = list(raw.columns[2:])
        detrended, trend = remove_slow_trend(raw, rois, n_median)
        smoothed = smooth_by_trial(detrended, rois, n_smooth)
        # The detection's rise source; older configs predate the option.
        rise_on_smoothed = config.get("rise_on_smoothed", False)
        rise_traces = smoothed if rise_on_smoothed else detrended[rois]
        source_plot = folder / "epileptic_detection/roi_traces_rise.png"
        detection_path = out / f"{stem}_original_detection.png"
        if source_plot.exists():
            shutil.copy2(source_plot, detection_path)
        else:
            rises = pd.DataFrame(np.nan, index=raw.index, columns=rois)
            for _, block in detrended.groupby("trial", sort=False):
                values = rise_signal(rise_traces.loc[block.index, rois].to_numpy(), n_rise,
                                     config["not_linear_rise"]) / (n_rise/fs)
                rises.loc[block.index[:len(values)], rois] = values
            plot_rise(smoothed, rises, events, rois, raw["frame"].to_numpy()/fs,
                      n_rise/(2*fs), detection_path,
                      f"Rise rate - {recording} ({config['rise_window_s']:g} s window, "
                      f"not_linear_rise={config['not_linear_rise']})",
                      dpi=120, signal_unit=signal_unit,
                      rise_source="smoothed amplitude" if rise_on_smoothed else "detrended trace")
        fig, axes = plt.subplots(len(rois), 2, figsize=(22, 3 * len(rois)),
                                 sharex=True, squeeze=False, constrained_layout=True)
        for i, roi in enumerate(rois):
            left, right = axes[i]
            for trial, block in raw.groupby("trial", sort=False):
                idx = block.index
                time_s = block["frame"].to_numpy() / fs
                left.plot(time_s, block[roi], color="#256abf", lw=0.8,
                          label=f"Starting signal ({signal_unit})")
                left.plot(time_s, trend.loc[idx, roi], color="#da8c20", lw=1.4,
                          label=f"Running median ({config['median_window_s']:g} s)")
                right.plot(time_s, smoothed.loc[idx, roi], color="#256abf", lw=1,
                           label="Detrended + smoothed")
                rise = rise_signal(rise_traces.loc[idx, roi].to_numpy(), n_rise,
                                   config["not_linear_rise"]) / (n_rise/fs)
                twin = right.twinx()
                twin.plot(time_s[:len(rise)] + n_rise/(2*fs), rise,
                          color="#c2354a", lw=0.7, alpha=0.45)
                twin.set_ylabel(f"Rise ({signal_unit} per s)", color="#c2354a", fontsize=9)
                twin.tick_params(axis="y", colors="#c2354a", labelsize=8)
                roi_events = events[(events.roi == roi) & (events.trial == trial)]
                for mask, marker, size, label in (
                    (roi_events.confirmed & ~roi_events.epileptic, "o", 28, "Other confirmed event"),
                    (roi_events.epileptic, "*", 95, "Epileptiform event"),
                ):
                    selected = roi_events.loc[mask]
                    values = block.set_index("frame")[roi].reindex(selected.frame)
                    if values.isna().any():
                        raise ValueError(f"Event frame missing from {recording}/{roi}")
                    left.scatter(selected.time_s, values, marker=marker, s=size,
                                 facecolors="none" if marker == "o" else "#202020",
                                 edgecolors="#202020", zorder=5, label=label)
                    twin.scatter(selected.rise_time_s, selected.rise_per_s,
                                 marker=marker, s=size,
                                 facecolors="none" if marker == "o" else "#202020",
                                 edgecolors="#202020", zorder=5)
            left.set_ylabel(f"{roi}\n{signal_unit}")
            right.set_ylabel(f"Detrended {signal_unit}")
            for ax in (left, right):
                ax.grid(alpha=0.2)
                ax.spines["top"].set_visible(False)
            if i == 0:
                left.set_title(f"Starting signal: {signal_description}, before detrending",
                               fontsize=9)
                right.set_title("Detection: smoothed amplitude (blue) and rise (red)")
                left.legend(loc="upper right", fontsize=8, ncol=2)
        for ax in axes[-1]:
            ax.set_xlabel("Time since first retained frame (s)")
        fig.suptitle(f"{recording} — starting signal ({signal_unit}) and detection\n"
                     "Markers: amplitude-peak time on the left; associated rise time on the right",
                     fontsize=14)
        path = out / f"{stem}_fluorescence_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)
    manifest_path.write_text(json.dumps(recordings, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        f"Source analysis: {root}\n"
        f"Signal ({signal_mode}, {signal_unit}): {signal_description}\n"
        f"Drawn: {len(recordings)} of {len(config['diagnostic_recordings'])} recordings "
        f"(--sample-count {args.sample_count}, --sample-seed {args.sample_seed}).\n\n"
        "Existing detection plots are copied unchanged; added recordings without a plot\n"
        "have one generated from saved traces and event flags, without redetection.\n"
        "Each comparison has the starting signal named above and its running median on\n"
        "the left, and the detection signals on the right. Rows share the same time axis.\n"
        "Circles = other confirmed events; stars = epileptiform events. Left markers\n"
        "use amplitude-peak times; right markers use the associated rise times.\n"
        "The starting signal has no detrending or smoothing applied; whether it is\n"
        "already a delta-F/F depends on the signal mode named above.\n"
        "Event flags and detection settings come from the saved analysis.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
