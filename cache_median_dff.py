"""Cache the per-pixel median-baseline dF/F of every pixel dump, once.

For every `pixel_data/<date>_<animal>/<t#>/` written by run_botox_batch.py
--save-data, computes over the WHOLE saved crop

    dF/F(t, p) = (F(t, p) / Fbar(t, p)) / (R(t, p) / Rbar(t, p)) - 1

where Fbar/Rbar are each pixel's centred running median over `window_s`
seconds (`min_periods=1` at the ends). This is exactly the per-pixel quantity
epileptic_by_area_animal_day_pixels.py (`--signal-mode median_dff`) computes
before averaging inside a box: averaging the cached volume over a box reproduces
`median_dff_roi_trace` bit for bit (checked on 260611/PV5/t1). It is a ratio,
not a percentage.

It depends only on F, R and the baseline window, never on detection
parameters, so it is computed once and read many times. The window is part of
the file name, so a cache built with another window is never read by mistake:

    pixels_median_dff_<window>s_full.npy   (n_time, rows, cols), float16 by default
    pixels_median_dff_<window>s_meta.json  provenance; written LAST, it is the done-marker

float16 vs float32, measured on 260611/PV5/t1 against float64: float16 is
39.4 MB per recording (~21.5 GB for 545), worst pixel error 6e-5 = 0.25% of
that pixel's temporal SD, worst ROI-trace error 0.04% of the trace SD, and at
most 2 of 396 pixels change active/inactive in any frame at a 1-SD threshold.
float32 is 78.8 MB (~43 GB) and effectively exact (7e-9). Values span about
-0.07..0.14, far inside float16's +-65504, and the cast is range-checked anyway.

Interruption is safe: the volume is written to a temporary name and renamed,
and the meta JSON is written after it, so a recording without its JSON is
simply recomputed on the next run.

Examples (in the letizia environment):
    python cache_median_dff.py
    python cache_median_dff.py --workers 1 --recording-limit 2
    python cache_median_dff.py --dtype float32 --window-s 30

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from epileptic_by_area_animal_day_pixels import median_window_frames, running_median

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the cache build without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",  # Contains <date>_<animal>/<t#>/pixels_meta_full.npz
    "output_root": None,  # None: write next to each dump; otherwise mirror <unit>/<t#>/ under this folder.
    "window_s": 20.0,  # Running-median baseline, seconds (20 s = 201 frames at 10 Hz).
    "sampling_rate_hz": 10.0,  # Per-channel frame rate of the dumps.
    "dtype": "float16",  # "float16" (half the disk) or "float32" (exact); see the docstring.
    # Parallel processes; 1 runs serially in this process. Measured on this machine
    # (i7-14700, 32 GB): 1 -> 107 min, 4 -> 29, 8 -> 21, 12 -> 17, 20 -> 13 min for
    # 545 recordings; ~1.1 GB RAM per worker. 8 leaves the machine usable meanwhile.
    "workers": 8,
    "recording_limit": None,  # None: all dumps; positive integer: only the first N (smoke test).
    "overwrite": False,  # True: recompute even when a finished cache exists.
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

DTYPES = ("float16", "float32")


def cache_paths(output_folder: Path, window_s: float) -> tuple[Path, Path]:
    """Volume and meta paths; the window is in the name so caches never mix."""
    stem = f"pixels_median_dff_{window_s:g}s"
    return output_folder / f"{stem}_full.npy", output_folder / f"{stem}_meta.json"


def median_dff_volume(fluorescence: np.ndarray, reflectance: np.ndarray, n_median: int) -> np.ndarray:
    """(F/Fbar)/(R/Rbar) - 1 for every pixel of (n_time, rows, cols) volumes, float64.

    The operation order matches `median_dff_roi_trace` exactly, so the two agree
    bit for bit. Each channel's baseline is freed before the next is built, which
    keeps the peak near four float64 volumes (~630 MB for a 2980x76x87 dump).
    """
    if fluorescence.shape != reflectance.shape or fluorescence.ndim != 3:
        raise ValueError("Fluorescence and reflectance must be matching 3-D volumes")
    n_time, n_row, n_col = fluorescence.shape
    # f_ratio, r_ratio: (n_time, n_pixels). Explicit copies: they are divided in
    # place below, and asarray would hand back the caller's own float64 array.
    f_ratio = np.array(fluorescence, dtype=np.float64).reshape(n_time, -1)
    r_ratio = np.array(reflectance, dtype=np.float64).reshape(n_time, -1)
    if not np.isfinite(f_ratio).all() or not np.isfinite(r_ratio).all() or np.any(r_ratio <= 0):
        raise ValueError("Non-finite fluorescence/reflectance or nonpositive reflectance")
    for channel in (f_ratio, r_ratio):
        baseline = running_median(channel, n_median)  # (n_time, n_pixels)
        if np.any(baseline <= 0):
            raise ValueError("Running-median baseline is nonpositive; cannot form F/Fbar")
        np.divide(channel, baseline, out=channel)
        del baseline
    dff = f_ratio / r_ratio - 1.0
    return dff.reshape(n_time, n_row, n_col)


def build_one(job: dict[str, Any]) -> dict[str, Any]:
    """Compute and write one recording's cache. Top-level so worker processes can import it."""
    started = time.perf_counter()
    folder = Path(job["folder"])
    volume_path, meta_path = cache_paths(Path(job["output_folder"]), job["window_s"])
    volume_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(folder / "pixels_meta_full.npz", allow_pickle=False) as data:
        n_time, n_written = int(data["n_time"]), int(data["n_written"])
        axis_order = str(data["axis_order"])
    if axis_order != "time,y,x" or n_written != n_time or n_time <= 0:
        raise ValueError(f"Unsupported or incomplete pixel dump: {folder}")
    fluorescence = np.load(folder / "pixels_f_gcamp_full.npy", mmap_mode="r")
    reflectance = np.load(folder / "pixels_f_emo_full.npy", mmap_mode="r")
    if fluorescence.shape[0] != n_time:
        raise ValueError(f"Volume length disagrees with metadata: {folder}")

    dff = median_dff_volume(fluorescence, reflectance, job["n_median"])
    # Refuse a cast that would turn values into inf (float16 max is 65504).
    dtype_max = float(np.finfo(job["dtype"]).max)
    peak = float(np.abs(dff).max())
    if peak > dtype_max:
        raise ValueError(f"|dF/F| reaches {peak:g}, beyond {job['dtype']} ({dtype_max:g}): {folder}")
    stored = dff.astype(job["dtype"])
    storage_error = float(np.abs(stored.astype(np.float64) - dff).max())

    # Temporary name, then rename: a killed job never leaves a truncated volume under the real name.
    temporary_path = volume_path.with_name(volume_path.stem + ".partial.npy")
    np.save(temporary_path, stored)
    os.replace(temporary_path, volume_path)
    meta = dict(
        formula="(F/Fbar)/(R/Rbar) - 1 per pixel; Fbar/Rbar = centred running median, min_periods=1",
        unit="ratio (x100 for percent)",
        source=str(folder.resolve()),
        inputs=["pixels_f_gcamp_full.npy", "pixels_f_emo_full.npy"],
        window_s=job["window_s"], window_frames=job["n_median"],
        sampling_rate_hz=job["sampling_rate_hz"],
        shape=list(stored.shape), dtype=job["dtype"], axis_order="time,y,x",
        value_min=float(dff.min()), value_max=float(dff.max()),
        max_abs_storage_error=storage_error,
        seconds=round(time.perf_counter() - started, 2),
    )
    # Written last: its presence marks the cache as complete.
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dict(folder=str(folder), seconds=meta["seconds"], error=storage_error)


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"],
                        help="Default: write next to each dump inside --pixel-root")
    parser.add_argument("--window-s", type=float, default=defaults["window_s"])
    parser.add_argument("--sampling-rate-hz", type=float, default=defaults["sampling_rate_hz"])
    parser.add_argument("--dtype", choices=DTYPES, default=defaults["dtype"])
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
    if args.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.recording_limit is not None and args.recording_limit <= 0:
        raise ValueError("recording_limit must be positive")
    pixel_root = Path(args.pixel_root)
    n_median = median_window_frames(args.window_s, args.sampling_rate_hz)

    folders = sorted(path.parent for path in pixel_root.glob("*/*/pixels_meta_full.npz"))
    if not folders:
        raise FileNotFoundError(f"No full pixel dumps found under {pixel_root}")
    if args.recording_limit is not None:
        folders = folders[:args.recording_limit]

    # One job per recording; the output folder mirrors <unit>/<t#>/.
    jobs = []
    for folder in folders:
        output_folder = (folder if args.output_root is None
                         else Path(args.output_root) / folder.parent.name / folder.name)
        _, meta_path = cache_paths(output_folder, args.window_s)
        if meta_path.exists() and not args.overwrite:
            continue
        jobs.append(dict(folder=str(folder), output_folder=str(output_folder),
                         window_s=args.window_s, n_median=n_median,
                         sampling_rate_hz=args.sampling_rate_hz, dtype=args.dtype))
    n_bytes = np.dtype(args.dtype).itemsize
    print(f"{len(folders)} dumps, {len(folders) - len(jobs)} already cached, {len(jobs)} to build "
          f"({args.window_s:g} s = {n_median} frames, {args.dtype}, "
          f"~{len(jobs) * 2980 * 76 * 87 * n_bytes / 1e9:.1f} GB), {args.workers} worker(s)", flush=True)
    if not jobs:
        return

    started = time.perf_counter()
    worst_error = 0.0

    def report(i_job, result):
        nonlocal worst_error
        worst_error = max(worst_error, result["error"])
        elapsed = time.perf_counter() - started
        remaining_min = elapsed / i_job * (len(jobs) - i_job) / 60
        print(f"  [{i_job}/{len(jobs)}] {result['seconds']:.1f}s {result['folder']}  "
              f"(elapsed {elapsed / 60:.1f} min, ~{remaining_min:.1f} min left)", flush=True)

    if args.workers == 1:
        for i_job, job in enumerate(jobs, start=1):
            report(i_job, build_one(job))
    else:
        # Pool's context exit calls terminate(), so Ctrl+C or a worker error stops
        # every worker instead of waiting for the queue (the trap in CLAUDE.md 9.10).
        with Pool(processes=min(args.workers, len(jobs))) as pool:
            for i_job, result in enumerate(pool.imap_unordered(build_one, jobs, chunksize=1), start=1):
                report(i_job, result)
    print(f"done: {len(jobs)} caches in {(time.perf_counter() - started) / 60:.1f} min, "
          f"worst storage error {worst_error:.2e}")


if __name__ == "__main__":
    main()
