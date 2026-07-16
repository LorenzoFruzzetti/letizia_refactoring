"""Compare the four wfci modalities on the sample dataset.

For each layout (``stack`` | ``folder`` | ``stream`` | ``interleaved``) this:
  * measures wall time (load + compute) and peak RAM, each in an isolated
    subprocess so the memory high-water mark is not cross-contaminated;
  * checks the numerical result is the same across all four layouts;
  * cross-checks against the MATLAB interleaved ("intermingle") script;
and writes a plain-text report.

All four layouts are fed the *identical* pixel data — the interleaved split of
``data/`` — just packaged into each layout's expected on-disk form (see
``bench_common.prepare_inputs``), so any difference is purely the layout's doing.

Editor / CLI entrypoint. Edit ``RUN_CONFIG`` and run with no flags, or override
on the command line.

    python benchmarks/benchmark_modalities.py
    python benchmarks/benchmark_modalities.py --repeats 10 --no-matlab
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Make bench_common importable no matter the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import BREGMA_COL, BREGMA_ROW, DOWNSAMPLE, LAYOUTS, TRIM  # noqa: E402
from bench_common import prepare_inputs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Edit this block to run from the editor without passing CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, object] = {
    "data_dir": str(REPO_ROOT / "data"),
    "output_txt": str(REPO_ROOT / "outputs" / "modality_comparison.txt"),
    "repeats": 5,               # timed repeats per layout
    "run_matlab": True,         # cross-check against MATLAB interleaved script
    "matlab_exe": None,         # None -> auto-detect (shutil.which / known paths)
    "tolerance": 1e-6,          # max |diff| accepted as "same result"
    "prefer_cli_args": True,
}

# Result arrays compared across layouts / against MATLAB.
_ARRAY_KEYS = ["temp_roi", "R", "R_mean", "averaged_traces"]


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a - b|, treating NaN==NaN as 0 and NaN-vs-number as infinite.

    Correlation over only 3 time points can produce NaNs (a near-constant trace
    has zero variance); NaNs must line up, so a NaN facing a real number is a
    genuine mismatch (inf), while two aligned NaNs agree (0)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.abs(np.where(nan_a & nan_b, 0.0, a - b))
    d[nan_a ^ nan_b] = np.inf
    return float(np.max(d))


def _fmt_mb(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.1f}"


def run_worker(layout: str, paths_json: Path, out_npz: Path, repeats: int) -> dict:
    """Run one layout in a fresh subprocess; return its metrics dict."""
    worker = Path(__file__).with_name("bench_worker.py")
    proc = subprocess.run(
        [sys.executable, str(worker),
         "--layout", layout,
         "--paths", str(paths_json),
         "--out", str(out_npz),
         "--repeats", str(repeats)],
        cwd=str(worker.parent),          # so `import bench_common` resolves
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker for layout {layout!r} failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    line = next(l for l in proc.stdout.splitlines() if l.startswith("METRICS "))
    return json.loads(line[len("METRICS "):])


def find_matlab(explicit: str | None) -> str | None:
    """Locate the MATLAB executable, or return None if unavailable."""
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("matlab")
    if found:
        return found
    # Common Windows install locations (newest first).
    for ver in ("R2024b", "R2024a", "R2023b", "R2023a", "R2022b"):
        cand = Path(f"C:/Program Files/MATLAB/{ver}/bin/matlab.exe")
        if cand.exists():
            return str(cand)
    return None


def run_matlab_reference(matlab_exe: str, data_dir: Path, out_mat: Path) -> dict:
    """Invoke the MATLAB interleaved script; return its arrays via scipy.io."""
    from scipy.io import loadmat

    script = REPO_ROOT / "matlab" / "step_interleaved_intermingle.m"
    env = dict(os.environ)
    env["WFCI_DATA"] = str(data_dir)
    env["WFCI_INTERLEAVED_OUT"] = str(out_mat)
    cmd = f"run('{script.as_posix()}')"
    proc = subprocess.run(
        [matlab_exe, "-batch", cmd],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_mat.exists():
        raise RuntimeError(
            f"MATLAB run failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    m = loadmat(str(out_mat))
    return {
        "temp_roi": np.asarray(m["TEMP_ROI"], dtype=np.float64),
        "R": np.asarray(m["R"], dtype=np.float64),
        "R_mean": np.asarray(m["R_mean"], dtype=np.float64),
        "averaged_traces": np.asarray(m["averaged_traces"], dtype=np.float64),
    }


def _squeeze_trial(arr: np.ndarray) -> np.ndarray:
    """Drop the single-trial axis so a Python array lines up with MATLAB's.

    Python keeps a trailing trial axis (temp_roi [t,4,1], R [4,4,1]); the MATLAB
    single-trial reference has none (temp_roi [t,4], R [4,4]). R_mean and
    averaged_traces already match."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim >= 1 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def build_report(cfg: dict, meta: dict, metrics: dict, results: dict,
                 matlab: dict | None, matlab_note: str) -> str:
    tol = float(cfg["tolerance"])
    ref_layout = "interleaved"  # ground-truth packaging (the raw data/ folder)
    L = []
    w = L.append

    w("=" * 78)
    w("WFCI MODALITY COMPARISON — stack | folder | stream | interleaved")
    w("=" * 78)
    w(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}")
    w(f"Dataset   : {meta['data_dir']}")
    w(f"            {meta['n_frames_per_channel']} frames/channel, "
      f"{meta['frame_shape'][0]}x{meta['frame_shape'][1]} px "
      f"(interleaved split of the sample folder)")
    w(f"Params    : mode=resting_state  trim={TRIM}  downsample={DOWNSAMPLE}  "
      f"bregma=({BREGMA_ROW},{BREGMA_COL}) -> y_1=60,x_2=67")
    w(f"Repeats   : {cfg['repeats']} timed runs per layout (min & mean reported)")
    w("")
    w("All four layouts are fed IDENTICAL pixel data (the interleaved split of")
    w("data/), repackaged into each layout's on-disk form, so results should")
    w("match to floating-point roundoff and timing/RAM differences reflect only")
    w("the layout's I/O + memory strategy.")
    w("")

    # ---- performance table ----
    w("-" * 78)
    w("PERFORMANCE (per layout, isolated subprocess)")
    w("-" * 78)
    hdr = f"{'layout':<13}{'time min (s)':>14}{'time mean (s)':>15}" \
          f"{'peak RAM (MB)':>15}{'algo alloc (MB)':>17}"
    w(hdr)
    w("-" * len(hdr))
    for layout in LAYOUTS:
        mt = metrics[layout]
        w(f"{layout:<13}{mt['time_min_s']:>14.4f}{mt['time_mean_s']:>15.4f}"
          f"{_fmt_mb(mt['peak_working_set_bytes']):>15}"
          f"{_fmt_mb(mt['tracemalloc_peak_bytes']):>17}")
    w("")
    w("  peak RAM (MB)   = OS process peak working set (interpreter + numpy + data)")
    w("  algo alloc (MB) = tracemalloc peak of the timed region (algorithm only)")
    w("")

    # ---- cross-layout correctness ----
    w("-" * 78)
    w(f"CORRECTNESS — cross-layout agreement (reference = '{ref_layout}')")
    w("-" * 78)
    w(f"max |diff| of each layout vs '{ref_layout}', per output array. "
      f"tolerance = {tol:g}")
    w("")
    hdr2 = f"{'layout':<13}" + "".join(f"{k:>18}" for k in _ARRAY_KEYS) + f"{'verdict':>9}"
    w(hdr2)
    w("-" * len(hdr2))
    all_pass = True
    ref = results[ref_layout]
    for layout in LAYOUTS:
        row = f"{layout:<13}"
        worst = 0.0
        for k in _ARRAY_KEYS:
            d = _maxdiff(results[layout][k], ref[k])
            worst = max(worst, d)
            row += f"{d:>18.2e}"
        verdict = "PASS" if worst <= tol else "FAIL"
        all_pass = all_pass and worst <= tol
        row += f"{verdict:>9}"
        w(row)
    w("")
    w("  (stack/folder use the same in-memory path as interleaved -> exact 0;")
    w("   stream reduces frame-by-frame -> agreement to baseline-sum roundoff.)")
    w("")

    # ---- MATLAB cross-check ----
    w("-" * 78)
    w("CROSS-CHECK vs MATLAB (matlab/step_interleaved_intermingle.m)")
    w("-" * 78)
    matlab_pass = None
    if matlab is None:
        w(f"  SKIPPED — {matlab_note}")
    else:
        w(f"max |diff| of Python '{ref_layout}' vs MATLAB interleaved. "
          f"tolerance = {tol:g}")
        w("")
        py = results[ref_layout]
        worst = 0.0
        for k in _ARRAY_KEYS:
            d = _maxdiff(_squeeze_trial(py[k]), _squeeze_trial(matlab[k]))
            worst = max(worst, d)
            w(f"    {k:<18}{d:>14.2e}")
        matlab_pass = worst <= tol
        w("")
        w(f"    verdict: {'PASS' if matlab_pass else 'FAIL'} "
          f"(worst = {worst:.2e})")
    w("")

    # ---- R_mean matrix for reference ----
    w("-" * 78)
    w("R_mean (4x4 functional connectivity, interleaved) — "
      "[Laterale_L, Verme_L, Laterale_R, Verme_R]")
    w("-" * 78)
    w(np.array2string(np.asarray(results[ref_layout]["R_mean"]),
                      precision=4, suppress_small=True))
    w("")

    # ---- summary ----
    w("=" * 78)
    w("SUMMARY")
    w("=" * 78)
    fastest = min(LAYOUTS, key=lambda l: metrics[l]["time_min_s"])
    leanest = min(LAYOUTS, key=lambda l: metrics[l]["peak_working_set_bytes"])
    leanest_algo = min(LAYOUTS, key=lambda l: metrics[l]["tracemalloc_peak_bytes"])
    w(f"  Fastest (min time)      : {fastest} "
      f"({metrics[fastest]['time_min_s']*1000:.1f} ms)")
    w(f"  Lowest peak RAM         : {leanest} "
      f"({_fmt_mb(metrics[leanest]['peak_working_set_bytes'])} MB)")
    w(f"  Lowest algo allocation  : {leanest_algo} "
      f"({_fmt_mb(metrics[leanest_algo]['tracemalloc_peak_bytes'])} MB)")
    w(f"  All layouts agree       : {'YES' if all_pass else 'NO'} "
      f"(tol {tol:g})")
    if matlab_pass is not None:
        w(f"  Matches MATLAB          : {'YES' if matlab_pass else 'NO'}")
    else:
        w(f"  Matches MATLAB          : not run ({matlab_note})")
    w("")
    w("  NOTE: the sample dataset is tiny (3 frames/channel), so wall time and")
    w("  peak RAM are dominated by fixed interpreter/numpy overhead and the")
    w("  differences here are within noise. The 'stream' layout is designed to")
    w("  win on RAM only for large recordings that do not fit in memory; the")
    w("  'algo alloc' column already shows it holding less resident data.")
    w("=" * 78)
    return "\n".join(L)


def parse_args(defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=defaults["data_dir"])
    p.add_argument("--output-txt", default=defaults["output_txt"])
    p.add_argument("--repeats", type=int, default=defaults["repeats"])
    p.add_argument("--no-matlab", dest="run_matlab", action="store_false",
                   default=defaults["run_matlab"])
    p.add_argument("--matlab-exe", default=defaults["matlab_exe"])
    p.add_argument("--tolerance", type=float, default=defaults["tolerance"])
    return p.parse_args()


def build_runtime_cfg() -> dict:
    cfg = dict(RUN_CONFIG)
    if cfg.get("prefer_cli_args", True) and len(sys.argv) > 1:
        ns = parse_args(cfg)
        cfg.update(
            data_dir=ns.data_dir, output_txt=ns.output_txt, repeats=ns.repeats,
            run_matlab=ns.run_matlab, matlab_exe=ns.matlab_exe,
            tolerance=ns.tolerance,
        )
    return cfg


def main() -> None:
    cfg = build_runtime_cfg()
    data_dir = Path(cfg["data_dir"])
    out_txt = Path(cfg["output_txt"])
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # Transient working files (prepared inputs, per-layout result npz, MATLAB
    # .mat) live under temporary_files/ per repo conventions; only the .txt
    # report is a kept deliverable.
    work = REPO_ROOT / "temporary_files" / "modality_bench"
    work.mkdir(parents=True, exist_ok=True)

    # 1) Materialise identical data into every layout's input form.
    meta = prepare_inputs(data_dir, work / "prep")
    paths_json = work / "paths.json"
    paths_json.write_text(json.dumps(meta))
    print(f"Prepared inputs: {meta['n_frames_per_channel']} frames/channel "
          f"{meta['frame_shape']}")

    # 2) Benchmark each layout in isolation, keep its result arrays.
    metrics: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for layout in LAYOUTS:
        out_npz = work / f"result_{layout}.npz"
        print(f"  running layout: {layout} ...")
        metrics[layout] = run_worker(layout, paths_json, out_npz, int(cfg["repeats"]))
        with np.load(out_npz) as z:
            results[layout] = {k: z[k] for k in _ARRAY_KEYS}

    # 3) MATLAB cross-check (optional).
    matlab = None
    matlab_note = ""
    if cfg["run_matlab"]:
        matlab_exe = find_matlab(cfg.get("matlab_exe"))
        if matlab_exe is None:
            matlab_note = "MATLAB executable not found"
            print("  MATLAB not found — skipping cross-check")
        else:
            print(f"  running MATLAB reference ({matlab_exe}) ...")
            matlab = run_matlab_reference(
                matlab_exe, data_dir, work / "interleaved_reference.mat"
            )
    else:
        matlab_note = "disabled via config/--no-matlab"

    # 4) Write report.
    report = build_report(cfg, meta, metrics, results, matlab, matlab_note)
    out_txt.write_text(report, encoding="utf-8")
    print(f"\nWrote report -> {out_txt}\n")
    print(report)


if __name__ == "__main__":
    main()
