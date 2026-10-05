"""Left vs right hemisphere dF/F traces for every pixel dump, two figures per recording.

The hemisphere counterpart of run_botox_batch.py's `roi_traces_full.png`: instead
of one line per ROI, every saved pixel on each side of the midline is averaged
into ONE trace per hemisphere (no ROI boxes are used). Each figure has three
stacked panels sharing the time axis, all in percent.

`hemisphere_traces.png` -- GCaMP corrected by the reflectance (emo) channel:
1. `pixels_dff_full.npy`             hemodynamic-corrected dF/F0 with the whole-
                                     recording mean baseline, as run_botox_batch
                                     computes it -- already in percent.
2. `pixels_median_dff_20s_full.npy`  (F/Fbar)/(R/Rbar) - 1, 20 s running-median baseline.
3. `pixels_median_dff_60s_full.npy`  same, 60 s running-median baseline.
The two median caches are ratios (cache_median_dff.py), multiplied by 100 here.

`hemisphere_traces_gcamp_only.png` -- the same three baselines WITHOUT the emo
correction, computed here from `pixels_f_gcamp_full.npy`:
1. F/mean_t(F) - 1 per pixel.
2. F/Fbar - 1 per pixel, Fbar = 20 s centred running median (`min_periods=1`).
3. same, 60 s.
Every baseline is per pixel and the hemisphere mean is taken afterwards, the same
order as the corrected figure, so the two figures differ only by the division by
R/Rbar. `running_median` is the one cache_median_dff.py used, so panel 2/3 is the
cached quantity minus the reflectance term. It is also the slow part: there is
no cache for it, so each recording computes two per-pixel running medians
(~14 s per recording serially).

Midline. The crop is Bregma-relative (CLAUDE.md 9.15) and the atlas mirrors a box
by negating its column offset, so the midline is column offset 0, i.e. crop column
`x_2 - 1 - region[2]`. Columns with |offset| <= `midline_half_width_px` are dropped
from both sides, which keeps the split exactly symmetric and keeps the midline
(sagittal sinus) out of both averages. The default 3 matches the atlas, whose
innermost boxes (M2*_bassa) start at |offset| 4.

The whole saved crop is averaged, not just the cortex under the ROI boxes, so
some recordings include non-cortex pixels at the crop's corners.

Outputs, per recording, under `output_root/<date>_<animal>/<t#>/`:
    hemisphere_traces.png / .csv              emo-corrected figure and its six traces (%)
    hemisphere_traces_gcamp_only.png / .csv   GCaMP-only figure and its six traces (%)
Each figure is skipped when its PNG already exists (unless --overwrite), so
adding one figure never redraws the other.

Known artefact: 260828_PV7/t2 has a one-frame reflectance glitch at frame 1809
(CLAUDE.md 9.23); the emo-corrected figure shows a single huge spike there. It
is plotted, not excluded.

Examples (in the letizia environment):
    python plot_hemisphere_traces.py
    python plot_hemisphere_traces.py --recording-limit 2 --workers 1
    python plot_hemisphere_traces.py --figures gcamp_only

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless: figures are only saved, and workers have no display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epileptic_by_area_animal_day_pixels import median_window_frames, running_median

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the plots without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",  # Contains <date>_<animal>/<t#>/pixels_meta_full.npz
    "output_root": REPO_ROOT / "outputs" / "hemisphere_traces",
    "figures": ["emo_corrected", "gcamp_only"],  # Any subset of FIGURE_NAMES.
    "sampling_rate_hz": 10.0,  # Per-channel frame rate of the dumps (x axis, median windows).
    "midline_half_width_px": 3,  # Drop columns with |offset from Bregma| <= this (0: only the midline column).
    "workers": 8,  # Parallel processes; 1 runs serially in this process.
    "recording_limit": None,  # None: all dumps; positive integer: only the first N (smoke test).
    "overwrite": False,  # True: redraw figures that already exist.
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

FIGURE_FILES = {  # figure name -> output file stem
    "emo_corrected": "hemisphere_traces",
    "gcamp_only": "hemisphere_traces_gcamp_only",
}
FIGURE_NAMES = tuple(FIGURE_FILES)

# Emo-corrected panels: (cached volume, panel title, factor to percent, CSV key), top to bottom.
CORRECTED_SIGNALS = (
    ("pixels_dff_full.npy", "dF/F0, mean baseline, emo-corrected", 1.0, "dff_mean"),
    ("pixels_median_dff_20s_full.npy", "dF/F, 20 s running-median baseline, emo-corrected", 100.0, "median_20s"),
    ("pixels_median_dff_60s_full.npy", "dF/F, 60 s running-median baseline, emo-corrected", 100.0, "median_60s"),
)
# GCaMP-only panels: (baseline window in s or None for the whole-recording mean, title, CSV key).
GCAMP_ONLY_BASELINES = (
    (None, "GCaMP dF/F0, mean baseline, no emo correction", "dff_mean"),
    (20.0, "GCaMP dF/F, 20 s running-median baseline, no emo correction", "median_20s"),
    (60.0, "GCaMP dF/F, 60 s running-median baseline, no emo correction", "median_60s"),
)
GCAMP_FILE = "pixels_f_gcamp_full.npy"
HEMISPHERE_COLORS = {"left": "#2a78d6", "right": "#eb6834"}  # Categorical slots 1-2 (blue, orange).


def hemisphere_columns(n_col: int, midline_col: int, half_width: int) -> dict[str, np.ndarray]:
    """Crop column indices of each hemisphere, mirror-symmetric about `midline_col`.

    Only offsets present on BOTH sides are kept, so the two averages cover the same
    lateral extent even if the crop were not centred on Bregma.
    """
    if not 0 <= midline_col < n_col:
        raise ValueError(f"Midline column {midline_col} is outside the {n_col}-column crop")
    max_offset = min(midline_col, n_col - 1 - midline_col)
    offsets = np.arange(half_width + 1, max_offset + 1)
    if offsets.size == 0:
        raise ValueError(f"midline_half_width_px={half_width} leaves no columns (max offset {max_offset})")
    # In image coordinates a negative column offset is the animal's LEFT (the L boxes).
    return {"left": midline_col - offsets, "right": midline_col + offsets}


def corrected_traces(folder: Path, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hemisphere means of the three cached emo-corrected volumes, in percent."""
    traces = {}
    for file_name, _, to_percent, key in CORRECTED_SIGNALS:
        volume = np.load(folder / file_name, mmap_mode="r")  # (n_time, n_row, n_col), float16
        for side, side_columns in columns.items():
            # float64 accumulator: summing float16 overflows (CLAUDE.md 9.16 hazard 3).
            trace = np.asarray(volume[:, :, side_columns]).mean(axis=(1, 2), dtype=np.float64)
            traces[f"{key}_{side}"] = trace * to_percent  # (n_time,)
    return traces


def gcamp_only_traces(folder: Path, columns: dict[str, np.ndarray], sampling_rate_hz: float) -> dict[str, np.ndarray]:
    """Hemisphere means of per-pixel GCaMP F/F0 - 1 (no reflectance term), in percent."""
    volume = np.load(folder / GCAMP_FILE, mmap_mode="r")  # (n_time, n_row, n_col), float32
    traces = {}
    for side, side_columns in columns.items():
        # fluorescence: (n_time, n_row * n_side_columns), float64
        fluorescence = np.array(volume[:, :, side_columns], dtype=np.float64).reshape(volume.shape[0], -1)
        if not np.isfinite(fluorescence).all() or np.any(fluorescence <= 0):
            raise ValueError(f"Non-finite or nonpositive GCaMP fluorescence: {folder}")
        for window_s, _, key in GCAMP_ONLY_BASELINES:
            if window_s is None:
                baseline = fluorescence.mean(axis=0, keepdims=True)  # (1, n_pixels)
            else:
                n_median = median_window_frames(window_s, sampling_rate_hz)
                baseline = running_median(fluorescence, n_median)  # (n_time, n_pixels)
            traces[f"{key}_{side}"] = (fluorescence / baseline - 1.0).mean(axis=1) * 100.0  # (n_time,)
    return traces


def draw_figure(time_s: np.ndarray, traces: dict[str, np.ndarray], panels: list[tuple[str, str]],
                suptitle: str, png_path: Path) -> None:
    """Stacked panels, one per (title, CSV key), left and right hemisphere in each."""
    # Separate y scales because the mean baseline keeps the slow drift that the
    # running-median baselines remove.
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 7.5), sharex=True,
                             constrained_layout=True, squeeze=False)
    for i_panel, (title, key) in enumerate(panels):
        axis = axes[i_panel, 0]
        for side, color in HEMISPHERE_COLORS.items():
            axis.plot(time_s, traces[f"{key}_{side}"], color=color, linewidth=0.7,
                      alpha=0.85, label=f"{side} hemisphere")
        axis.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
        axis.set_title(title, fontsize=10, loc="left")
        axis.set_ylabel("ΔF/F (%)")
        axis.grid(True, color="0.92", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 0].set_xlim(time_s[0], time_s[-1])
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def plot_one(job: dict[str, Any]) -> dict[str, Any]:
    """Compute and draw the requested figures of one recording. Top-level so workers can import it."""
    started = time.perf_counter()
    folder = Path(job["folder"])
    output_folder = Path(job["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)

    with np.load(folder / "pixels_meta_full.npz", allow_pickle=False) as meta:
        x_2, region = int(meta["x_2"]), meta["region"].astype(int)
        group, n_time = str(meta["group"]), int(meta["n_time"])
        n_col = int(meta["shape"][2])
    # Same 0-based conversion as the ROI boxes (epileptic_by_area_animal_day_pixels.crop_slices).
    midline_col = x_2 - 1 - int(region[2])
    columns = hemisphere_columns(n_col, midline_col, job["midline_half_width_px"])
    frame = np.arange(n_time)
    time_s = frame / job["sampling_rate_hz"]
    # Second title line, shared by both figures; the first line names the signal.
    title_detail = (f"hemisphere-averaged, {len(columns['left'])} columns per side, "
                    f"midline ±{job['midline_half_width_px']} px excluded")
    recording = f"{folder.parent.name} {folder.name} (group {group})"

    for figure in job["figures"]:
        if figure == "emo_corrected":
            traces = corrected_traces(folder, columns)
            panels = [(title, key) for _, title, _, key in CORRECTED_SIGNALS]
            suptitle = f"{recording} - GCaMP, emo-corrected\n{title_detail}"
        else:
            traces = gcamp_only_traces(folder, columns, job["sampling_rate_hz"])
            panels = [(title, key) for _, title, key in GCAMP_ONLY_BASELINES]
            suptitle = f"{recording} - GCaMP only, no emo correction\n{title_detail}"
        if any(len(trace) != n_time for trace in traces.values()):
            raise ValueError(f"Volume length disagrees with metadata n_time={n_time}: {folder}")
        stem = FIGURE_FILES[figure]
        table = pd.DataFrame({"frame": frame, "time_s": time_s, **traces})
        table.to_csv(output_folder / f"{stem}.csv", index=False, float_format="%.6g")
        draw_figure(time_s, traces, panels, suptitle, output_folder / f"{stem}.png")
    return dict(folder=str(folder), figures=job["figures"], seconds=time.perf_counter() - started)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--figures", nargs="+", choices=FIGURE_NAMES, default=defaults["figures"])
    parser.add_argument("--sampling-rate-hz", type=float, default=defaults["sampling_rate_hz"])
    parser.add_argument("--midline-half-width-px", type=int, default=defaults["midline_half_width_px"])
    parser.add_argument("--workers", type=int, default=defaults["workers"],
                        help="Parallel processes; 1 runs serially in the main process")
    parser.add_argument("--recording-limit", type=int, default=defaults["recording_limit"])
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=defaults["overwrite"])
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})


def main():
    args = build_runtime_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.midline_half_width_px < 0:
        raise ValueError("midline_half_width_px must be >= 0")
    if args.recording_limit is not None and args.recording_limit <= 0:
        raise ValueError("recording_limit must be positive")
    unknown = set(args.figures) - set(FIGURE_NAMES)
    if not args.figures or unknown:
        raise ValueError(f"figures must be a nonempty subset of {FIGURE_NAMES}, got {args.figures}")
    pixel_root = Path(args.pixel_root)

    folders = sorted(path.parent for path in pixel_root.glob("*/*/pixels_meta_full.npz"))
    if not folders:
        raise FileNotFoundError(f"No full pixel dumps found under {pixel_root}")
    if args.recording_limit is not None:
        folders = folders[:args.recording_limit]
    # Fail up front, not 400 recordings in, if an input volume is missing.
    needed = []
    if "emo_corrected" in args.figures:
        needed += [name for name, _, _, _ in CORRECTED_SIGNALS]
    if "gcamp_only" in args.figures:
        needed.append(GCAMP_FILE)
    missing = [str(folder / name) for folder in folders for name in needed if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input volume(s) missing, e.g. {missing[:3]}; "
                                "build the median caches with cache_median_dff.py --window-s 20 / 60")

    # One job per recording, carrying only the figures it still lacks; the
    # output folder mirrors <unit>/<t#>/.
    jobs = []
    for folder in folders:
        output_folder = Path(args.output_root) / folder.parent.name / folder.name
        figures = [figure for figure in args.figures
                   if args.overwrite or not (output_folder / f"{FIGURE_FILES[figure]}.png").exists()]
        if figures:
            jobs.append(dict(folder=str(folder), output_folder=str(output_folder), figures=figures,
                             sampling_rate_hz=args.sampling_rate_hz,
                             midline_half_width_px=args.midline_half_width_px))
    print(f"{len(folders)} dumps, {len(folders) - len(jobs)} fully plotted, {len(jobs)} to plot "
          f"({', '.join(args.figures)}), {args.workers} worker(s) -> {args.output_root}", flush=True)
    if not jobs:
        return

    started = time.perf_counter()

    def report(i_job, result):
        elapsed = time.perf_counter() - started
        remaining_min = elapsed / i_job * (len(jobs) - i_job) / 60
        print(f"  [{i_job}/{len(jobs)}] {result['seconds']:.1f}s {'+'.join(result['figures'])} {result['folder']}  "
              f"(elapsed {elapsed / 60:.1f} min, ~{remaining_min:.1f} min left)", flush=True)

    if args.workers == 1:
        for i_job, job in enumerate(jobs, start=1):
            report(i_job, plot_one(job))
    else:
        # Pool's context exit calls terminate(), so Ctrl+C or a worker error stops
        # every worker instead of waiting for the queue (the trap in CLAUDE.md 9.10).
        with Pool(processes=min(args.workers, len(jobs))) as pool:
            for i_job, result in enumerate(pool.imap_unordered(plot_one, jobs, chunksize=1), start=1):
                report(i_job, result)
    print(f"done: {len(jobs)} recordings in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
