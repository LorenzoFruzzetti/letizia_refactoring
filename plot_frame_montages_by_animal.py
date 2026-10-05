"""Filmstrip montages for every animal: one figure per animal and peak rank, with and without GSR.

For each animal, every session contributes one panel: its `recording` (t1 by default)
from pixel_data. Figure k of an animal shows, in every panel, that session's k-th
largest field-wide peak (k = 1..n_peaks), with the strip timing of
plot_frame_montage.py. Peaks are ranked on the field mean with its slow trend removed
(a centred running median over `peak_detrend_window_s`), because the mean-baseline
dF/F declines through a recording (CLAUDE.md 9.30) and the raw maximum would always
sit in its first seconds. They are at least `peak_min_separation_s` apart, so the ten
strips of a session never overlap.

Two versions of every figure, from the same frames:
  no_gsr  the volume as saved (`volume_name`, by default the mean-baseline dF/F)
  gsr     the same volume after global signal regression: the mean of the column
          means of the recording's ROI-box union is regressed out of every pixel over
          the whole recording (roi_pixel_connectivity.regress_global_signal,
          gsr_global_mean "column" = Antea's script (2); CLAUDE.md 9.37/9.38)
The peaks are chosen once, on the no_gsr signal, so each pair shows the same frames.
Each figure has its own colour scale (shared by its panels): GSR moves the values to
around zero, so one scale for both versions would wash one of them out.

Outputs: <output_dir>/<version>/<animal>/<animal>_peak<k>.png and .json (frames,
colour limits, settings), and <output_dir>/peaks.csv (every chosen frame).

With figure_per "session" each figure is instead one session's k-th peak alone, so
there are n_peaks figures per day, in <output_dir>/<version>/<animal>/D<i>_<day>/.

Run: conda run --no-capture-output -n letizia python plot_frame_montages_by_animal.py
     ... --animals PV5 T9 --n-peaks 3        (subset / fewer figures)
     ... --animals PV3 PV4 PV5 --figure-per session --output-dir outputs/frame_montages/pv_by_day
"""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plot_frame_montage import binned_frames, draw_montage_figure, ranked_peak_frames, sampling_rate_hz
from roi_pixel_connectivity import crop_slices, read_meta, recording_boxes, regress_global_signal

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the script without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",
    "design_csv": REPO_ROOT / "pixel_data/experimental_design.csv",
    "volume_name": "pixels_dff_full.npy",  # mean-baseline dF/F (Antea's script 1 signal)
    "recording": "t1",  # which recording of each session is drawn
    "animals": None,  # None: every animal in the design; or a list such as ["PV5", "T9"]
    "versions": ["no_gsr", "gsr"],
    # "rank": one figure per animal and peak rank, one panel per session.
    # "session": one figure per session and peak, a single panel (n_peaks figures per day).
    "figure_per": "rank",
    "n_peaks": 10,  # peaks per session = figures per animal ("rank") or per session ("session")
    "peak_detrend_window_s": 20.0,  # running-median window removed from the field mean before ranking
    "peak_min_separation_s": 3.0,  # minimum distance between two chosen peaks of one session
    "peak_exclude_s": 10.0,  # ignore this much at both recording ends (median-filter edges)
    "n_tiles": 20,
    "tile_rows": 2,
    "bin_frames": 1,
    "step_frames": 1,
    "pre_peak_s": 0.5,
    "color_limits": None,  # None: percentiles below, per figure
    "color_percentiles": (1.0, 99.5),
    "value_scale": 1.0,  # pixels_dff_full.npy is already in % dF/F
    "value_label": "dF/F (%)",
    "cmap": "jet",
    "panel_columns": 2,
    "tile_gap_px": 1,
    "show_time_labels": True,
    "dpi": 200,
    "output_dir": REPO_ROOT / "outputs/frame_montages/by_animal",
    "workers": 8,  # one animal per process; the job does no BLAS work (einsum/np.where only)
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume-name", default=defaults["volume_name"])
    parser.add_argument("--recording", default=defaults["recording"])
    parser.add_argument("--animals", nargs="*", default=defaults["animals"])
    parser.add_argument("--versions", nargs="+", choices=("no_gsr", "gsr"), default=defaults["versions"])
    parser.add_argument("--figure-per", choices=("rank", "session"), default=defaults["figure_per"])
    parser.add_argument("--n-peaks", type=int, default=defaults["n_peaks"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--dpi", type=int, default=defaults["dpi"])
    args = parser.parse_args()
    # Everything else stays as in RUN_CONFIG.
    for key, value in defaults.items():
        if key != "prefer_cli_args" and not hasattr(args, key):
            setattr(args, key, value)
    return args


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})


def gsr_volume(recording_dir: Path, volume: np.ndarray) -> np.ndarray:
    """The volume with the ROI-union global signal regressed out of every pixel. (time, y, x) float64."""
    meta = read_meta(recording_dir / "pixels_meta_full.npz")
    _, boxes = recording_boxes(meta)
    n_time, n_rows, n_cols = volume.shape
    union = np.zeros((n_rows, n_cols), dtype=bool)
    for box in boxes.values():
        row_slice, col_slice = crop_slices(box, int(meta["y_1"]), int(meta["x_2"]), meta["region"])
        union[row_slice, col_slice] = True
    pixels = np.asarray(volume, dtype=np.float64).reshape(n_time, -1)
    residuals, _ = regress_global_signal(pixels, union, "column")
    return residuals.reshape(n_time, n_rows, n_cols)


def animal_montages(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Every figure of one animal. Top-level so worker processes can import it."""
    args = argparse.Namespace(**job["args"])
    animal, sessions = job["animal"], job["sessions"]   # sessions: [(day_index, day), ...]
    # Load each session's recording once, choose its peaks, and keep both versions.
    panels = []
    peak_rows = []
    for day_index, day in sessions:
        recording_dir = Path(args.pixel_root) / f"{day}_{animal}" / args.recording
        fs = sampling_rate_hz(recording_dir)
        volume = np.load(recording_dir / args.volume_name).astype(np.float64)   # (time, y, x)
        peaks = ranked_peak_frames(volume, args.n_peaks, args.bin_frames,
                                   int(round(args.peak_exclude_s * fs)),
                                   int(round(args.peak_min_separation_s * fs)),
                                   int(round(args.peak_detrend_window_s * fs)) | 1)   # odd window
        # Kept as float32 (half the memory of float64, ~80 MB per volume): an animal
        # holds up to 6 sessions x 2 versions per worker, and drawing needs no more precision.
        volumes = {"no_gsr": volume.astype(np.float32)}
        if "gsr" in args.versions:
            volumes["gsr"] = gsr_volume(recording_dir, volume).astype(np.float32)
        del volume
        panels.append(dict(label=f"D{day_index}", recording=f"{day}_{animal}/{args.recording}",
                           fs=fs, peaks=peaks, volumes=volumes, n_time=volumes["no_gsr"].shape[0]))
        peak_rows += [dict(animal=animal, day=day, day_index=day_index, recording=args.recording,
                           rank=rank + 1, peak_frame=peak, peak_s=peak / fs) for rank, peak in enumerate(peaks)]

    strip_frames = (args.n_tiles - 1) * args.step_frames + args.bin_frames

    def strip(panel: dict[str, Any], version: str, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
        """The tiles of one session's rank-th peak, and their provenance."""
        peak = panel["peaks"][rank]
        first_frame = peak - args.bin_frames // 2 - int(round(args.pre_peak_s * panel["fs"]))
        first_frame = min(max(first_frame, 0), panel["n_time"] - strip_frames)
        tiles = binned_frames(panel["volumes"][version], first_frame, args.n_tiles,
                              args.bin_frames, args.step_frames)
        info = dict(label=panel["label"], recording=panel["recording"], rank=rank + 1,
                    sampling_rate_hz=panel["fs"], peak_frame=peak, first_frame=first_frame,
                    tile_start_frames=(first_frame + np.arange(args.n_tiles) * args.step_frames).tolist())
        return tiles * args.value_scale, info

    # Every figure to draw: (output folder, file stem, title head, [(panel, rank), ...]).
    figure_specs = []
    for rank in range(args.n_peaks):
        if args.figure_per == "rank":
            members = [(panel, rank) for panel in panels if rank < len(panel["peaks"])]
            figure_specs.append(("", f"{animal}_peak{rank + 1:02d}",
                                 f"{animal}, peak {rank + 1} of each session ({args.recording})", members))
        else:
            for panel, (day_index, day) in zip(panels, sessions):
                if rank < len(panel["peaks"]):
                    figure_specs.append((f"D{day_index}_{day}", f"{animal}_D{day_index}_peak{rank + 1:02d}",
                                         f"{animal} D{day_index} ({day}/{args.recording}), peak {rank + 1}",
                                         [(panel, rank)]))
    # A single-panel figure is drawn one panel wide.
    draw_args = argparse.Namespace(**vars(args))
    if args.figure_per == "session":
        draw_args.panel_columns = 1

    for version in args.versions:
        gsr_note = "GSR (ROI-union column mean)" if version == "gsr" else "no GSR"
        for subfolder, stem, title_head, members in figure_specs:
            if not members:
                continue
            output_dir = Path(args.output_dir) / version / animal / subfolder
            output_dir.mkdir(parents=True, exist_ok=True)
            strips = [strip(panel, version, rank) for panel, rank in members]
            panel_tiles = [tiles for tiles, _ in strips]
            provenance = [info for _, info in strips]
            title = (f"{title_head}, {args.volume_name}, {gsr_note}; "
                     f"{args.n_tiles} tiles x {args.bin_frames / panels[0]['fs']:g} s")
            image_path = output_dir / f"{stem}.png"
            vmin, vmax = draw_montage_figure(panel_tiles, provenance, draw_args, title, image_path)
            settings = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
            image_path.with_suffix(".json").write_text(json.dumps(
                {"settings": settings, "version": version, "color_limits_displayed": [vmin, vmax],
                 "panels": provenance}, indent=2))
    return peak_rows


def main():
    args = build_runtime_args()
    if args.n_peaks < 1 or args.workers < 1:
        raise ValueError("n_peaks and workers must be >= 1")
    design = pd.read_csv(args.design_csv, dtype={"day": str})
    animals = sorted(design["animal"].unique()) if args.animals is None else list(args.animals)
    missing = sorted(set(animals) - set(design["animal"]))
    if missing:
        raise ValueError(f"Animals not in {args.design_csv}: {missing}")

    # One job per animal: its sessions in day_index order, each needing the chosen recording.
    jobs = []
    for animal in animals:
        sessions = (design[design["animal"] == animal][["day_index", "day"]]
                    .drop_duplicates().sort_values("day_index"))
        session_list = [(int(row.day_index), row.day) for row in sessions.itertuples()]
        for _, day in session_list:
            path = Path(args.pixel_root) / f"{day}_{animal}" / args.recording / args.volume_name
            if not path.exists():
                raise FileNotFoundError(path)
        jobs.append(dict(animal=animal, sessions=session_list, args=vars(args)))
    n_units = len(jobs) if args.figure_per == "rank" else sum(len(job["sessions"]) for job in jobs)
    unit_name = "animals" if args.figure_per == "rank" else "sessions"
    n_figures = n_units * args.n_peaks * len(args.versions)
    print(f"{n_units} {unit_name} x {args.n_peaks} peaks x {len(args.versions)} versions = up to {n_figures} "
          f"figures, {args.workers} worker(s) -> {args.output_dir}", flush=True)

    peak_rows = []
    if args.workers == 1:
        for job in jobs:
            peak_rows += animal_montages(job)
            print(f"  {job['animal']} done", flush=True)
    else:
        with Pool(args.workers) as pool:
            for job, rows in zip(jobs, pool.imap(animal_montages, jobs)):
                peak_rows += rows
                print(f"  {job['animal']} done", flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(peak_rows).to_csv(Path(args.output_dir) / "peaks.csv", index=False)
    print(f"Saved {len(peak_rows)} peaks to {Path(args.output_dir) / 'peaks.csv'}")


if __name__ == "__main__":
    main()
