"""Zoom the largest detected peaks of each sampled recording, in seconds.

Defaults: three non-overlapping peaks per ROI, with 5 seconds on either side.
Timing defaults to the saved per-channel sampling rate. --sampling-rate-hz can
override the display conversion; saved detection flags are never recalculated.
Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epileptic_by_area_animal_day import remove_slow_trend, smooth_by_trial, rise_signal


# Edit this section to run the script without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "input_root": Path(__file__).resolve().parent / "outputs/epileptic_by_area_animal_day_pixels",
    "sampling_rate_hz": None,  # None: use the saved analysis rate (10 Hz per channel).
    "peaks_per_roi": 3,
    "half_window_s": 5.0,  # Seconds before AND after each selected peak.
    "prefer_cli_args": True,  # False: always use this block, ignoring CLI arguments.
}


def select_peaks(events, frames, half_window_frames, count):
    """Highest detected amplitudes, complete windows only, without overlap."""
    selected = []
    for event in events.sort_values("amplitude", ascending=False).itertuples(index=False):
        if event.frame-half_window_frames < frames.min() or event.frame+half_window_frames > frames.max():
            continue
        if any(abs(event.frame-other.frame) <= 2*half_window_frames for other in selected):
            continue
        selected.append(event)
        if len(selected) == count:
            break
    return selected


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=defaults["input_root"])
    parser.add_argument("--sampling-rate-hz", type=float, default=defaults["sampling_rate_hz"],
                        help="Actual per-channel rate; default is saved analysis rate (10 Hz)")
    parser.add_argument("--peaks-per-roi", type=int, default=defaults["peaks_per_roi"])
    parser.add_argument("--half-window-s", type=float, default=defaults["half_window_s"])
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        input_root=Path(config["input_root"]),
        sampling_rate_hz=config["sampling_rate_hz"],
        peaks_per_roi=int(config["peaks_per_roi"]),
        half_window_s=float(config["half_window_s"]),
    )


def main():
    args = build_runtime_args()
    root = Path(args.input_root)
    comparison = root / "sampled_plot_comparisons"
    out = comparison / "peak_zooms"
    config = json.loads((root / "detection_config.json").read_text())
    recordings = json.loads((comparison / "comparison_recordings.json").read_text())
    fs = config["sampling_rate_hz"] if args.sampling_rate_hz is None else args.sampling_rate_hz
    if not np.isfinite(fs) or fs <= 0 or args.peaks_per_roi < 1 or not np.isfinite(args.half_window_s) or args.half_window_s <= 0:
        raise ValueError("Sampling rate, peak count and window duration must be positive and finite")
    # Reproduce the saved processing in frame space, even if display timing changes.
    detection_fs = config["sampling_rate_hz"]
    n_median = 2*round(config["median_window_s"]*detection_fs/2)+1
    n_smooth = max(1, round(config["smooth_window_s"]*detection_fs))
    n_rise = max(1, round(config["rise_window_s"]*detection_fs))
    out.mkdir(parents=True, exist_ok=True)
    selections = []
    for recording in recordings:
        folder = root / recording
        raw = pd.read_csv(folder / "roi_reflectance_corrected_fluorescence.csv")
        events = pd.read_csv(folder / "epileptic_detection/roi_events.csv")
        if raw.trial.nunique() != 1:
            raise ValueError(f"Expected one trial per recording: {recording}")
        rois = list(raw.columns[2:])
        detrended, trend = remove_slow_trend(raw, rois, n_median)
        smoothed = smooth_by_trial(detrended, rois, n_smooth)
        frames = raw.frame.to_numpy()
        # Compute on the full trace before zooming so window boundaries do not
        # change the rise. Each value belongs at the centre of its rise window.
        # Same rise source as the detection; older configs predate the option.
        rise_traces = smoothed if config.get("rise_on_smoothed", False) else detrended[rois]
        rises = rise_signal(rise_traces[rois].to_numpy(), n_rise,
                            config["not_linear_rise"]) / (n_rise/fs)
        rise_frames = frames[:len(rises)] + n_rise/2
        fig, axes = plt.subplots(2*len(rois), args.peaks_per_roi,
                                 figsize=(7*args.peaks_per_roi, 4.5*len(rois)),
                                 gridspec_kw={"height_ratios": [2, 1]*len(rois)},
                                 squeeze=False, constrained_layout=True)
        for i, roi in enumerate(rois):
            peaks = select_peaks(events[events.roi == roi], frames, fs*args.half_window_s, args.peaks_per_roi)
            for j, ax in enumerate(axes[2*i]):
                rise_ax = axes[2*i+1, j]
                rise_ax.sharex(ax)
                if j >= len(peaks):
                    ax.text(0.5, 0.5, "No further non-overlapping peak\nwith a complete window",
                            ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    rise_ax.set_axis_off()
                    continue
                event = peaks[j]
                relative_s = (frames-event.frame)/fs
                mask = np.abs(relative_s) <= args.half_window_s
                ax.plot(relative_s[mask], raw.loc[mask, roi], color="#256abf", lw=1,
                        label="Starting F corrected for R")
                ax.plot(relative_s[mask], trend.loc[mask, roi], color="#da8c20", lw=1.2,
                        label="Running median")
                peak_value = raw.loc[raw.frame == event.frame, roi].iloc[0]
                ax.scatter([0], [peak_value], marker="*" if event.epileptic else "o",
                           color="#202020", s=75, zorder=5)
                ax.axvline(0, color="#555555", linestyle="--", lw=0.8)
                ax.set_xlim(-args.half_window_s, args.half_window_s)
                ax.set_xticks(np.linspace(-args.half_window_s, args.half_window_s, 11))
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_ylabel("Starting fluorescence units")
                ax.grid(alpha=0.2)
                twin = ax.twinx()
                twin.plot(relative_s[mask], smoothed.loc[mask, roi], color="#c2354a", lw=1.1,
                          alpha=0.8, label="Detrended + smoothed")
                twin.set_ylabel("Detrended fluorescence units", color="#c2354a")
                twin.tick_params(axis="y", colors="#c2354a")
                relative_rise_s = (rise_frames-event.frame)/fs
                rise_mask = np.abs(relative_rise_s) <= args.half_window_s
                rise_ax.plot(relative_rise_s[rise_mask], rises[rise_mask, i],
                             color="#7c3aab", lw=1.2, label="Rise of detrended trace")
                rise_ax.axhline(0, color="#777777", lw=0.6)
                rise_ax.axvline(0, color="#555555", linestyle="--", lw=0.8)
                if event.confirmed and np.isfinite(event.rise_time_s):
                    # Saved event times/rates use detection_fs; convert them if
                    # the display sampling rate was overridden.
                    rise_offset_s = (event.rise_time_s*detection_fs-event.frame)/fs
                    rise_ax.scatter([rise_offset_s], [event.rise_per_s*fs/detection_fs],
                                    marker="*" if event.epileptic else "o",
                                    color="#202020", s=65, zorder=5,
                                    label="Associated rise peak")
                rise_ax.set_ylabel("Rise\n(fluorescence units/s)", color="#7c3aab")
                rise_ax.tick_params(axis="y", colors="#7c3aab")
                rise_ax.set_xlabel("Seconds relative to detected amplitude peak")
                rise_ax.grid(alpha=0.2)
                status = "epileptiform" if event.epileptic else "confirmed" if event.confirmed else "unconfirmed"
                ax.set_title(f"{roi} | peak {j+1}: {status}\n"
                             f"t = {event.frame/fs:.2f} s, frame {event.frame}, z = {event.amplitude_z:.2f}")
                if i == 0 and j == 0:
                    lines, labels = ax.get_legend_handles_labels()
                    extra, extra_labels = twin.get_legend_handles_labels()
                    ax.legend(lines+extra, labels+extra_labels, fontsize=7, loc="upper left")
                    rise_ax.legend(fontsize=7, loc="upper left")
                selections.append(dict(recording=recording, roi=roi, rank=j+1,
                                       frame=event.frame, time_s=event.frame/fs,
                                       window_start_s=event.frame/fs-args.half_window_s,
                                       window_end_s=event.frame/fs+args.half_window_s,
                                       samples=int(mask.sum()), epileptic=event.epileptic))
        fig.suptitle(f"{recording}: largest detected peaks, ±{args.half_window_s:g} seconds\n"
                     f"Time conversion: {fs:g} frames/s per channel; blue = starting fluorescence, "
                     "orange = trend, red = detection signal; purple below = rise", fontsize=14)
        path = out / f"{recording.replace('/', '_')}_peak_zooms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(path)
    pd.DataFrame(selections).to_csv(out / "selected_peaks.csv", index=False)
    (out / "zoom_config.json").write_text(json.dumps(dict(
        sampling_rate_hz=fs, saved_detection_sampling_rate_hz=detection_fs,
        half_window_s=args.half_window_s, peaks_per_roi=args.peaks_per_roi,
        rise_window_frames=n_rise, rise_window_s=n_rise/fs,
        not_linear_rise=config["not_linear_rise"],
        rise_on_smoothed=config.get("rise_on_smoothed", False),
        selection="Largest detected amplitudes, complete non-overlapping windows",
        recordings=recordings, detection_recomputed=False), indent=2))


if __name__ == "__main__":
    main()
