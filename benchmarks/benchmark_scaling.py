"""Measure how streaming time and RAM scale with frame count, and extrapolate.

Purpose: before committing to a full run over a large (possibly remote) folder,
answer two questions from a handful of cheap debug-mode runs:
  * how long will the whole folder take?   (time is expected to be O(n) frames)
  * how much RAM will it need?             (streaming is expected to be O(1))

Method: run the streaming pipeline at several ``--debug-max-frames`` limits, each
in an ISOLATED subprocess (the OS peak-working-set is a whole-process high-water
mark, so it must not be polluted by the other runs), fit a straight line
``t = a + b*n`` through the measured times, and extrapolate to the folder's real
frame count. Peak RAM is fitted the same way to test the O(1) claim: a slope of
~0 MB/frame is the streaming path behaving as designed.

Caching (this is what makes or breaks the estimate on a network share): the
pipeline's debug limit always reads the folder's FIRST n frames, so measuring it
directly at increasing limits is self-defeating -- each larger run re-reads files
the smaller runs already pulled into the OS file cache, and only its tail is
actually fetched over the wire. The fitted slope then reflects cached reads and
the extrapolation comes out far too optimistic.

So each measurement here reads a DISJOINT, previously untouched chunk of the
folder (``--offset``), which is what a real full run does: every frame fetched
once, cold. The last chunk is then re-run at the same offset, now fully warm, so
the report can show how much of the time was network I/O vs. compute.

TWO-PASS REGIME (measured on this dataset -- the reason small chunks mislead):
the streaming pipeline reads each channel TWICE (baseline mean, then the
correction). If a chunk is small enough to sit in the OS file cache, pass 2 is
served from RAM for free; once it is not, pass 2 is re-fetched over the wire and
the per-frame cost nearly doubles. On the BOTOX_RESTANI share the cliff sits
between 160 frames/channel (168 MB, ~126 ms/frame, warm rerun 4.5 s) and 200
frames/channel (210 MB, ~180 ms/frame, warm rerun 32 s). A fit over chunks below
the cliff underestimates a full run by ~35%.

Therefore: keep every measured limit ABOVE the cliff, so the fit lives in the
same regime as the real run. The default limits do this. If a run's warm rerun is
a small fraction of its cold time, that chunk was cached -- it is below the cliff
and must not be used for extrapolation.

Run (edit RUN_CONFIG, or pass flags):
    conda run -n letizia python benchmarks/benchmark_scaling.py
    conda run -n letizia python benchmarks/benchmark_scaling.py \
        --folder "\\\\146.48.88.209\\share2\\BOTOX_RESTANI\\260611\\R1\\t1" \
        --limits 40,80,120,160,200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# Make the sibling worker module importable no matter the current directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_worker import peak_working_set_bytes  # noqa: E402

from wfci import (  # noqa: E402
    ROIConfig,
    folder_frame_source,
    interleaved_channel_files,
    run_streaming_resting_state,
    run_streaming_stimulated,
)

# ---------------------------------------------------------------------------
# Edit this section to run without passing CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # The interleaved folder to profile (a single folder of alternating-channel
    # single-page TIFFs). UNC paths must be raw strings.
    "folder": r"\\146.48.88.209\share2\BOTOX_RESTANI\260611\R1\t1",
    # Frame limits (PER CHANNEL) to measure. Each must exceed the 20-frame trim,
    # and -- see the two-pass regime note above -- must be big enough that the
    # chunk does NOT fit the OS file cache, or the pipeline's second pass is free
    # and the extrapolation comes out ~35% low. >=200 frames (210 MB) clears the
    # cliff on this share. An interleaved folder reads 2*n images per n frames.
    "limits": [200, 300, 400],
    # Where the disjoint cold chunks start. Chunks are read back to back from here,
    # so this + sum(limits) must fit within the folder's frame count.
    "cold_start": 0,
    "mode": "resting_state",   # "resting_state" | "stimulated"
    "bregma_row": 121,
    "bregma_col": 134,
    "prefer_cli_args": True,
}

TRIM = 20  # frames dropped from the front of every trial by the pipeline


def _fmt_mb(n_bytes: float) -> str:
    return f"{n_bytes / 1024 ** 2:.1f}"


def _fmt_hms(seconds: float) -> str:
    """Seconds as h/m/s, for extrapolated totals that can run long."""
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(sec)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


# ---------------------------------------------------------------------------
# Worker: one measured run, in its own process.
# ---------------------------------------------------------------------------
def _run_worker(args: argparse.Namespace) -> None:
    """Time one streaming run at a given frame limit; print a METRICS json line."""
    cfg = ROIConfig.from_bregma(args.bregma_row, args.bregma_col)

    t0 = time.perf_counter()
    # The odd/even split + brighter-group decision reads only two images; time it
    # separately since it is a fixed cost that does NOT grow with the limit.
    gcamp_files, emo_files = interleaved_channel_files(args.folder)
    split_s = time.perf_counter() - t0

    # Read a window of n frames starting at --offset. The pipeline's own debug
    # limit is prefix-only; profiling an arbitrary window is what lets each
    # measurement hit files no earlier run has cached (see module docstring).
    lo = min(args.offset, max(0, len(gcamp_files) - args.limit))
    n = min(args.limit, len(gcamp_files) - lo)
    sources = [(
        folder_frame_source(gcamp_files[lo:lo + n]),
        folder_frame_source(emo_files[lo:lo + n]),
    )]

    runner = (
        run_streaming_stimulated
        if args.mode == "stimulated"
        else run_streaming_resting_state
    )
    t0 = time.perf_counter()
    result = runner(sources, cfg)
    compute_s = time.perf_counter() - t0

    # One frame's resident size, so the RAM slope below can be judged against the
    # thing it would scale with if streaming ever started accumulating frames.
    probe = np.asarray(tifffile.imread(str(gcamp_files[lo])), dtype=np.float64)

    metrics = {
        "limit": n,
        "offset": lo,
        "images_read": 2 * n,
        "split_s": split_s,
        "compute_s": compute_s,
        "peak_working_set_bytes": peak_working_set_bytes(),
        "trace_frames": int(result.temp_roi.shape[0]),
        "channel_frames_available": len(gcamp_files),
        "frame_bytes": int(probe.nbytes),
    }
    print("METRICS " + json.dumps(metrics))


def _measure(folder: str, limit: int, offset: int, mode: str,
             bregma_row: int, bregma_col: int) -> dict:
    """Run one worker subprocess and return its metrics dict."""
    proc = subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--folder", folder,
            "--limit", str(limit),
            "--offset", str(offset),
            "--mode", mode,
            "--bregma-row", str(bregma_row),
            "--bregma-col", str(bregma_col),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker failed for limit={limit}:\n{proc.stdout}\n{proc.stderr}"
        )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("METRICS "))
    return json.loads(line[len("METRICS "):])


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares ``y = a + b*x``; returns ``(a, b, r_squared)``."""
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # ss_tot == 0 means y is perfectly flat (e.g. RAM that does not move at all).
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), r2


# A frame is the unit the RAM slope would scale with if the streaming path ever
# started holding the recording: a source that accumulated frames would cost one
# frame (per channel) per frame read. Real streaming holds a fixed handful of
# small images, so its slope sits at essentially 0 MB/frame -- a few hundred
# bytes/frame of per-file bookkeeping, orders of magnitude under this budget.
# 5% of a frame is therefore comfortably above the noise and far below any
# genuine regression.
RAM_SLOPE_BUDGET_FRACTION = 0.05


def _check_ram_slope(ram_slope_bytes_per_frame: float, frame_bytes: int) -> None:
    """Assert peak RAM is flat in frame count -- the constant-memory guarantee.

    The slope was already being fitted and printed; printing alone means a
    regression is only caught if a human reads the number and knows what it
    should be. This turns it into a verdict that fails loudly.
    """
    budget = RAM_SLOPE_BUDGET_FRACTION * frame_bytes
    ok = abs(ram_slope_bytes_per_frame) < budget

    print("\n--- Constant-memory check -----------------------------------------")
    print(f"  frame size          : {_fmt_mb(frame_bytes)} MB")
    print(f"  measured RAM slope  : {ram_slope_bytes_per_frame / 1024 ** 2:+.5f} MB/frame")
    print(f"  budget              : {budget / 1024 ** 2:.5f} MB/frame "
          f"({RAM_SLOPE_BUDGET_FRACTION:.0%} of one frame)")
    print(f"  verdict             : {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(
            f"\nCONSTANT-MEMORY REGRESSION: peak RAM grows "
            f"{ram_slope_bytes_per_frame / 1024 ** 2:+.4f} MB per frame, i.e. it "
            f"scales with the recording length. The streaming path is meant to be "
            f"O(1) in frame count -- something is accumulating frames instead of "
            f"reducing them away. See tests/test_efficiency_invariants.py (I10)."
        )
    print("\n  A ~0 MB/frame slope means the streaming path is genuinely")
    print("  constant-memory and the folder size is irrelevant to RAM.")


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--folder", default=defaults["folder"])
    p.add_argument("--limit", type=int, default=None, help="worker only")
    p.add_argument("--offset", type=int, default=0, help="worker only: first frame index")
    p.add_argument("--cold-start", type=int, default=defaults["cold_start"])
    p.add_argument(
        "--limits",
        default=",".join(str(n) for n in defaults["limits"]),
        help="Comma-separated frame limits (per channel) to measure.",
    )
    p.add_argument("--mode", choices=["resting_state", "stimulated"], default=defaults["mode"])
    p.add_argument("--bregma-row", type=int, default=defaults["bregma_row"])
    p.add_argument("--bregma-col", type=int, default=defaults["bregma_col"])
    return p.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    prefer_cli_args = bool(config.get("prefer_cli_args", True))
    if prefer_cli_args and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        worker=False,
        folder=config["folder"],
        limit=None,
        offset=0,
        cold_start=config["cold_start"],
        limits=",".join(str(n) for n in config["limits"]),
        mode=config["mode"],
        bregma_row=config["bregma_row"],
        bregma_col=config["bregma_col"],
    )


def main() -> None:
    args = build_runtime_args()
    if args.worker:
        _run_worker(args)
        return

    limits = [int(n) for n in args.limits.split(",")]
    bad = [n for n in limits if n <= TRIM]
    if bad:
        raise ValueError(
            f"limits {bad} are at or below the pipeline's {TRIM}-frame trim, so "
            f"they would leave no frames to correlate; use larger values."
        )

    print(f"Folder : {args.folder}")
    print(f"Mode   : {args.mode}  |  streaming (constant memory)")
    print(f"Limits : {limits} frames/channel "
          f"(= {[2 * n for n in limits]} images read)")
    print(f"Chunks : disjoint and cold, starting at frame {args.cold_start}\n")

    rows = []
    offset = args.cold_start
    for n in limits:
        m = _measure(args.folder, n, offset, args.mode, args.bregma_row, args.bregma_col)
        rows.append(m)
        print(f"  {n:>4} frames/ch @ offset {m['offset']:>5}  ->  "
              f"{m['compute_s']:7.2f} s  "
              f"peak RAM {_fmt_mb(m['peak_working_set_bytes']):>7} MB  "
              f"({m['images_read']} images, cold)")
        # Next chunk starts where this one ended, so no file is read twice.
        offset += n

    # Re-run the LAST chunk at its own offset: those exact files are now in the OS
    # cache, so the gap vs. its cold run is the network read contribution.
    warm = _measure(args.folder, limits[-1], rows[-1]["offset"], args.mode,
                    args.bregma_row, args.bregma_col)
    print(f"  {limits[-1]:>4} frames/ch @ offset {rows[-1]['offset']:>5}  ->  "
          f"{warm['compute_s']:7.2f} s  (WARM rerun of the same chunk)")

    n_available = rows[0]["channel_frames_available"]
    x = np.array([r["limit"] for r in rows], dtype=float)
    t = np.array([r["compute_s"] for r in rows], dtype=float)
    ram = np.array([r["peak_working_set_bytes"] for r in rows], dtype=float)

    t_a, t_b, t_r2 = _linear_fit(x, t)
    ram_a, ram_b, ram_r2 = _linear_fit(x, ram)

    print("\n--- Scaling -------------------------------------------------------")
    print(f"  time : {t_a:+.2f} s {t_b:+.4f} s/frame   (R^2 = {t_r2:.4f})")
    print(f"  RAM  : {_fmt_mb(ram_a)} MB {ram_b / 1024 ** 2:+.4f} MB/frame  "
          f"(R^2 = {ram_r2:.4f})")
    print(f"  per-frame time by limit: "
          + ", ".join(f"{r['limit']}:{r['compute_s'] / r['limit'] * 1000:.0f}ms" for r in rows))
    warm_ratio = warm["compute_s"] / t[-1] if t[-1] > 0 else float("nan")
    print(f"  warm rerun at {limits[-1]}: {warm['compute_s']:.2f} s vs "
          f"{t[-1]:.2f} s cold  ({warm_ratio:.0%} of cold -> "
          f"{(1 - warm_ratio):.0%} of the time is network reads)")

    print("\n--- Full-folder estimate ------------------------------------------")
    print(f"  frames available : {n_available} per channel "
          f"({2 * n_available} images)")
    full_t = t_a + t_b * n_available
    print(f"  time  : ~{_fmt_hms(full_t)}  (cold linear fit -- every frame fetched "
          f"once, as in a real full run)")
    if 0 < warm_ratio < 1:
        # Floor: what the same fit predicts if every read were served from cache.
        print(f"  floor : ~{_fmt_hms(t_a + t_b * warm_ratio * n_available)}  "
              f"(if the data were local/cached -- compute only)")
    print(f"  RAM   : ~{_fmt_mb(ram_a + ram_b * n_available)} MB peak "
          f"(streaming: expected flat in frame count)")

    _check_ram_slope(ram_b, rows[0]["frame_bytes"])


if __name__ == "__main__":
    main()
