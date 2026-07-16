"""Benchmark one modality in an isolated process.

Running each layout in its own subprocess is what makes the peak-RAM number
meaningful: the OS ``PeakWorkingSetSize`` is the high-water mark over the whole
process lifetime, so it must not be polluted by the other layouts' allocations.

The worker times the load+compute region over ``--repeats`` runs (min + mean),
records two memory figures, and writes the result arrays plus a JSON metrics
blob so the orchestrator can compare correctness and tabulate performance.

Memory figures reported:
  * ``peak_working_set_bytes`` -- OS process high-water mark (``psapi``); the
    headline "max RAM used", includes interpreter + numpy + data.
  * ``tracemalloc_peak_bytes`` -- Python/numpy-tracked allocation peak of just
    the timed region; isolates the algorithm's own memory.

Run (invoked by ``benchmark_modalities.py``; can also be run standalone):
    python benchmarks/bench_worker.py --layout stack --paths paths.json \
        --out result_stack.npz --repeats 5
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
import tracemalloc
from ctypes import wintypes
from pathlib import Path

import numpy as np

from bench_common import run_layout


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> int:
    """OS peak working set of this process (Windows ``GetProcessMemoryInfo``)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", required=True)
    p.add_argument("--paths", required=True, help="JSON file from prepare_inputs")
    p.add_argument("--out", required=True, help="output .npz for the result arrays")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    paths = json.loads(Path(args.paths).read_text())

    # Time the full load+compute over several repeats for a stable number. The
    # last run's arrays are the ones we keep for the correctness comparison.
    tracemalloc.start()
    times = []
    result = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        result = run_layout(args.layout, paths)
        times.append(time.perf_counter() - t0)
    tm_current, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    temp_roi, R, R_mean, averaged_traces = result

    np.savez(
        args.out,
        temp_roi=temp_roi,
        R=R,
        R_mean=R_mean,
        averaged_traces=averaged_traces,
    )

    metrics = {
        "layout": args.layout,
        "repeats": args.repeats,
        "time_min_s": float(np.min(times)),
        "time_mean_s": float(np.mean(times)),
        "time_all_s": [float(t) for t in times],
        "peak_working_set_bytes": peak_working_set_bytes(),
        "tracemalloc_peak_bytes": int(tm_peak),
        "result_npz": str(args.out),
    }
    # The orchestrator reads this single JSON line from stdout.
    print("METRICS " + json.dumps(metrics))


if __name__ == "__main__":
    main()
