"""Seed-based connectivity: correlate every ROI with every pixel of the dumps.

run_botox_batch.py reduces each frame to the 22 ROI box means and computes the
22x22 Pearson matrix `R` between them (wfci.roi.functional_connectivity). This
script computes the same correlation, with the same seeds, against EVERY pixel of
the saved crop instead of against the other 21 boxes:

    r_map[roi, y, x] = Pearson( seed_trace[roi](t), pixel(t, y, x) )  over the recording

so each recording yields 22 maps of shape (rows, cols) = (76, 87). A map shows
where on the cortex activity follows that ROI. It is not limited to the other boxes.

Seeds are the nanmean of the dumped pixels inside each box of the recording's
OWN `roi_sets/rebuilt/<day>_<animal>_<t#>.yaml` (the one named in its dump meta),
so they are the traces `R` was computed from, to float16 storage error
(CLAUDE.md 9.19). The ROI x ROI matrix recomputed from them is saved as well
(`R_roi`) so it can be checked against run_botox_batch's `R_mean`.

Averaging. Correlations are averaged as Fisher z (arctanh), then converted back
with tanh: recordings -> one map per animal-day (unit), units -> one map per
group. Every unit therefore weighs the same in its group, however many `t#` it
has. The saved crop is Bregma-relative (CLAUDE.md 9.15), so pixel (y, x) names the
same position relative to Bregma in every animal, and maps can be averaged pixel
by pixel without any registration.

Signal. `volume_file` picks which dumped/cached per-pixel volume is correlated:
`pixels_dff_full.npy` (default) is the mean-baseline dF/F that run_botox_batch's
`R` is computed from. `pixels_median_dff_<N>s_full.npy` (cache_median_dff.py) is
the running-median dF/F the epileptic detectors use. The output folder is named
after the volume so two signals never overwrite each other.

GSR (`gsr`, off by default). Global signal regression as in
`Antea_scripts/(2)_Global_Signal_Regression_SCRIPT.txt` (wfci.gsr): the global
signal g(t) is the mean, per frame, of the pixels inside the union of the
recording's ROI boxes (the cortex, standing in for the MATLAB's brain mask; the
rectangular crop can include off-brain pixels at its edges). Every pixel of the
crop is then fitted as a*g(t) + b by least squares and replaced by its residual,
and seeds and maps are computed from the residuals. Without it every r is
positive and shares the cortex-wide component; with it r is centred near zero
and negative values are expected (a known property of GSR, not an error). The
output folder gets a `_gsr` suffix. `gsr_global_mean` sets how g(t) averages the
union: "column" (default) is the MATLAB script exactly, nanmean(nanmean(data,1),2),
the mean of the column means; "pixel" is the plain pixel mean (folder
`_gsr_pixelmean`), which is what `_gsr` meant before 2026-10-05. They differ
because the union's columns hold different numbers of pixels (CLAUDE.md 9.37).

`260828_PV7/t2` is excluded by default: its frame 1809 is a field-wide
reflectance glitch (dF/F ~10 in every pixel) followed by a persistent GCaMP step
(CLAUDE.md 9.23, 9.27). One such frame is shared by every pixel and would drive
every correlation of that recording towards 1.

Outputs (under `output_root/<volume stem>/`):
    <unit>/<t#>/roi_pixel_connectivity.npz  per recording: r_maps (22, rows, cols),
                                            seed_traces, R_roi, boxes; the resume marker
    <unit>/unit_roi_pixel_connectivity.npz  Fisher-z mean over the unit's recordings
    <unit>/seed_maps.png                    the 22 unit maps (plot_units)
    group_roi_pixel_connectivity.npz        (n_groups, 22, rows, cols) group means
    group_<G>_seed_maps.png                 the 22 maps of each group
    recording_summary.csv                   one row per recording

Examples (in the letizia environment):
    python roi_pixel_connectivity.py
    python roi_pixel_connectivity.py --recording-limit 5
    python roi_pixel_connectivity.py --workers 8
    python roi_pixel_connectivity.py --volume-file pixels_median_dff_20s_full.npy
    python roi_pixel_connectivity.py --no-plot-units

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from epileptic_by_area_animal_day_pixels import crop_slices, recording_roi_set, roi_output_suffix

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the analysis without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",  # Contains <date>_<animal>/<t#>/pixels_meta_full.npz
    "volume_file": "pixels_dff_full.npy",  # Per-pixel signal correlated; see the docstring.
    "output_root": REPO_ROOT / "outputs/roi_pixel_connectivity",  # A subfolder per volume_file.
    # None: each dump's own ROI set. A folder: per-recording sets drawn with
    # pixel_roi_editor.py (every recording must then carry the same labels, in the
    # same order, to be averaged); the output subfolder gets a _roi_<folder> suffix.
    "roi_set_dir": None,
    # Cross-check of each dump's own `group`; None skips the check.
    "groups_csv": REPO_ROOT / "outputs/epileptic_groups/batch_summary.csv",
    "exclude_recordings": ["260828_PV7/t2"],  # "<date>_<animal>/<t#>"; see the docstring.
    "gsr": False,  # True: regress the ROI-union global signal out of every pixel first; see the docstring.
    # How the GSR global signal averages the ROI union: "column" = Antea's script (2)
    # exactly (mean of column means), folder suffix _gsr; "pixel" = plain pixel mean
    # (the behaviour before 2026-10-05), folder suffix _gsr_pixelmean.
    "gsr_global_mean": "column",
    # Parallel processes; 1 runs serially in this process. Measured ~0.5 s per
    # recording, so the whole cohort is ~5 min serially. The job does no BLAS
    # work (see pixel_correlation), so CLAUDE.md 9.11/9.12 do not apply to a pool.
    "workers": 1,
    "recording_limit": None,  # None: all dumps; positive integer: only the first N (smoke test).
    "overwrite": False,  # True: recompute recordings that already have an output .npz.
    "plot_units": True,  # True: one 22-map figure per animal-day as well as per group.
    # Colour scale of the maps (Pearson r). Without GSR every r is positive (the
    # shared global signal), so a sequential 0..1 scale shows the structure.
    "color_range_r": (0.0, 1.0),
    "colormap": "viridis",
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

RESULT_NAME = "roi_pixel_connectivity.npz"
UNIT_RESULT_NAME = "unit_roi_pixel_connectivity.npz"
FISHER_CLIP = 1.0 - 1e-7  # arctanh(+-1) is inf; a pixel can only reach 1 if identical to its seed.


def pixel_correlation(seed_traces: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Pearson r of every seed column with every pixel column, float64.

    seed_traces: (n_time, n_seeds); pixels: (n_time, n_pixels) -> (n_seeds, n_pixels).
    Same explicit centring/norm formula as wfci.roi._corrcoef_matlab, so a pixel
    identical to a seed trace gives exactly the `R` entry. The cross products use
    einsum without `optimize`, which sums in its own loops and never dispatches to
    BLAS -- the native 0xc06d007f failure point on this machine (CLAUDE.md 9.11,
    9.17). A pixel that is constant or has a non-finite value comes out NaN.
    """
    seeds_centred = seed_traces - seed_traces.mean(axis=0, keepdims=True)
    pixels_centred = pixels - pixels.mean(axis=0, keepdims=True)
    cross = np.einsum("ts,tp->sp", seeds_centred, pixels_centred)  # (n_seeds, n_pixels)
    seed_norm = np.sqrt(np.einsum("ts,ts->s", seeds_centred, seeds_centred))
    pixel_norm = np.sqrt(np.einsum("tp,tp->p", pixels_centred, pixels_centred))
    with np.errstate(divide="ignore", invalid="ignore"):
        return cross / (seed_norm[:, None] * pixel_norm[None, :])


GLOBAL_MEANS = ("column", "pixel")  # how the global signal averages the region; see regress_global_signal


def regress_global_signal(pixels: np.ndarray, region_mask: np.ndarray,
                          global_mean: str = "column") -> tuple[np.ndarray, np.ndarray]:
    """Residuals of every pixel after least-squares regression on the global signal.

    pixels: (n_time, n_pixels) float64, the frame flattened row by row;
    region_mask: (n_rows, n_cols) bool, the pixels averaged into the global signal
    (a 1-D (n_pixels,) mask is accepted only with global_mean="pixel").
    global_mean: "column" (default) is Antea's script (2) exactly,
    nanmean(nanmean(data,1),2): the mean down each column of the region, then the
    mean of those column means, so every column counts once however many region
    pixels it holds. "pixel" is the plain mean of all region pixels, which is what
    this function did before 2026-10-05 (the `_gsr_pixelmean` folders); for a union
    of boxes the two move ROI correlations by up to 0.13 (CLAUDE.md 9.37).
    Returns (residuals (n_time, n_pixels), g (n_time,)).
    Same fit as fitlm(g, p) in the MATLAB / wfci.gsr.regress_global, in closed form:
    slope a = cov(g, p) / var(g), residual = p - mean(p) - a * (g - mean(g)).
    einsum instead of a matrix product keeps BLAS out (CLAUDE.md 9.11, 9.17).
    """
    if global_mean not in GLOBAL_MEANS:
        raise ValueError(f"global_mean={global_mean!r} is not one of {GLOBAL_MEANS}")
    if global_mean == "column" and region_mask.ndim != 2:
        raise ValueError("global_mean='column' needs the region mask as (n_rows, n_cols)")
    flat_mask = region_mask.reshape(-1)
    region_pixels = pixels[:, flat_mask]
    if not np.isfinite(region_pixels).all():
        raise ValueError("Non-finite pixel inside the ROI union; the global signal would be undefined")
    if global_mean == "pixel":
        global_trace = region_pixels.mean(axis=1)                      # g: (n_time,)
    else:
        # column_sums: (n_time, n_cols), the region pixels summed down each column.
        n_time = pixels.shape[0]
        n_rows, n_cols = region_mask.shape
        # np.where, not a multiply by the mask: a NaN outside the region times 0 is NaN.
        column_sums = np.where(region_mask[None], pixels.reshape(n_time, n_rows, n_cols), 0.0).sum(axis=1)
        column_counts = region_mask.sum(axis=0)
        has_pixels = column_counts > 0
        global_trace = (column_sums[:, has_pixels] / column_counts[has_pixels]).mean(axis=1)
    global_centred = global_trace - global_trace.mean()
    global_sum_squares = float(np.einsum("t,t->", global_centred, global_centred))
    if global_sum_squares <= 0.0:
        raise ValueError("The global signal is constant, so no slope is identifiable")
    slopes = np.einsum("t,tp->p", global_centred, pixels) / global_sum_squares   # a: (n_pixels,)
    residuals = pixels - pixels.mean(axis=0, keepdims=True) - global_centred[:, None] * slopes[None, :]
    return residuals, global_trace


def fisher_mean(r_maps: np.ndarray) -> np.ndarray:
    """Average correlation maps over axis 0 in Fisher z, returned as r."""
    z_maps = np.arctanh(np.clip(r_maps, -FISHER_CLIP, FISHER_CLIP))
    return np.tanh(z_maps.mean(axis=0))


def read_meta(meta_path: Path) -> dict[str, Any]:
    """The dump's sidecar metadata, without the two large baseline images."""
    with np.load(meta_path, allow_pickle=False) as data:
        meta = {key: data[key].tolist() for key in data.files if key not in ("mean_f", "mean_r")}
    if meta["axis_order"] != "time,y,x":
        raise ValueError(f"Unsupported pixel axes in {meta_path}")
    if meta["n_written"] != meta["n_time"] or meta["n_time"] <= 0:
        raise ValueError(f"Incomplete pixel dump: {meta_path}")
    return meta


def recording_boxes(meta: dict[str, Any],
                    roi_set_dir: Path | None = None) -> tuple[Path, dict[str, dict[str, int]]]:
    """The recording's atlas (or its override), checked against the geometry stored in its dump.

    Only the dump's own set must list the dump's `roi_labels`: an override from
    pixel_roi_editor.py may add, drop or rename boxes.
    """
    atlas_path, atlas = recording_roi_set(meta, roi_set_dir)
    if roi_set_dir is None and list(atlas["boxes"]) != list(meta["roi_labels"]):
        raise ValueError(f"Atlas box order differs from the dump's roi_labels: {atlas_path}")
    return atlas_path, atlas["boxes"]


def analyze_recording(job: dict[str, Any]) -> dict[str, Any]:
    """Seed maps of one recording. Top-level so worker processes can import it."""
    started = time.perf_counter()
    folder = Path(job["folder"])
    output_path = Path(job["output_path"])
    meta = read_meta(folder / "pixels_meta_full.npz")
    atlas_path, boxes = recording_boxes(meta, job.get("roi_set_dir"))
    region = meta["region"]

    # volume: (n_time, rows, cols), float16 on disk -> pixels: (n_time, rows * cols) float64
    volume = np.load(folder / job["volume_file"], mmap_mode="r")
    n_time, n_row, n_col = volume.shape
    if [n_row, n_col] != [region[1] - region[0], region[3] - region[2]] or n_time != meta["n_time"]:
        raise ValueError(f"{job['volume_file']} shape {volume.shape} disagrees with the metadata: {folder}")
    pixels = np.asarray(volume, dtype=np.float64).reshape(n_time, -1)

    # Each ROI's box in crop coordinates, and their union (the GSR region).
    labels = list(boxes)
    box_slices = {}
    roi_union = np.zeros((n_row, n_col), dtype=bool)
    for label in labels:
        row_slice, col_slice = crop_slices(boxes[label], int(meta["y_1"]), int(meta["x_2"]), region)
        box_slices[label] = [row_slice.start, row_slice.stop, col_slice.start, col_slice.stop]
        roi_union[row_slice, col_slice] = True

    # Optional GSR: every pixel replaced by its residual on the ROI-union mean.
    global_trace = None
    if job.get("gsr", False):
        pixels, global_trace = regress_global_signal(pixels, roi_union,
                                                     job.get("gsr_global_mean", "column"))

    # Seed traces: the box mean of each ROI, like wfci.roi.extract_roi_timeseries.
    # seed_traces: (n_time, n_roi)
    seed_traces = np.empty((n_time, len(labels)), dtype=np.float64)
    pixel_volume = pixels.reshape(n_time, n_row, n_col)
    for i_roi, label in enumerate(labels):
        row_start, row_stop, col_start, col_stop = box_slices[label]
        seed_traces[:, i_roi] = np.nanmean(pixel_volume[:, row_start:row_stop, col_start:col_stop], axis=(1, 2))
    if not np.isfinite(seed_traces).all():
        raise ValueError(f"A seed trace has non-finite frames (a whole box is NaN/inf): {folder}")

    # r_maps: (n_roi, rows, cols) -- the result. R_roi: (n_roi, n_roi), the batch's R.
    r_maps = pixel_correlation(seed_traces, pixels).reshape(len(labels), n_row, n_col)
    r_roi = pixel_correlation(seed_traces, seed_traces)
    n_nan_pixels = int(np.isnan(r_maps[0]).sum())

    # Temporary name, then rename: an interrupted job never leaves a truncated marker.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".partial.npz")
    np.savez(temporary_path, r_maps=r_maps.astype(np.float32), seed_traces=seed_traces,
             R_roi=r_roi, roi_labels=np.array(labels),
             box_slices_zero_based_exclusive=np.array([box_slices[label] for label in labels]),
             bregma_crop_zero_based=np.array([meta["y_1"] - 1 - region[0], meta["x_2"] - 1 - region[2]]),
             region=np.array(region), day=meta["day"], animal=meta["animal"], group=meta["group"],
             recording=meta["recording"], roi_set=str(atlas_path), source=str(folder.resolve()),
             volume_file=job["volume_file"], gsr=bool(job.get("gsr", False)),
             gsr_global_mean=job.get("gsr_global_mean", "column") if job.get("gsr", False) else "",
             global_trace=np.array([]) if global_trace is None else global_trace)
    os.replace(temporary_path, output_path)
    return dict(folder=str(folder), seconds=round(time.perf_counter() - started, 2), n_nan_pixels=n_nan_pixels)


def draw_seed_maps(r_maps: np.ndarray, labels: list[str], bregma: np.ndarray,
                   seed_boxes: np.ndarray, title: str, color_range: tuple[float, float], colormap: str,
                   output_path: Path) -> None:
    """One panel per seed, left-hemisphere seeds on the top row and right below.

    seed_boxes: (n_roi, 4) zero-based exclusive [row0, row1, col0, col1] in the crop;
    fractional values (a group mean) are allowed and drawn as-is.
    """
    hemisphere_of = {label: label.partition("_")[0][-1] for label in labels}
    rows_by_side = {side: [label for label in labels if hemisphere_of[label] == side] for side in ("L", "R")}
    n_columns = max(len(members) for members in rows_by_side.values())
    fig, axes = plt.subplots(2, n_columns, figsize=(1.7 * n_columns + 1.2, 4.2),
                             constrained_layout=True, squeeze=False)
    image = None
    for i_row, side in enumerate(("L", "R")):
        for i_col in range(n_columns):
            axis = axes[i_row, i_col]
            axis.set_xticks([])
            axis.set_yticks([])
            if i_col >= len(rows_by_side[side]):
                axis.set_axis_off()
                continue
            label = rows_by_side[side][i_col]
            i_roi = labels.index(label)
            image = axis.imshow(r_maps[i_roi], cmap=colormap, vmin=color_range[0], vmax=color_range[1])
            row0, row1, col0, col1 = seed_boxes[i_roi]
            # imshow pixel centres are integers, so a box's edge sits half a pixel before its first index.
            axis.add_patch(Rectangle((col0 - 0.5, row0 - 0.5), col1 - col0, row1 - row0,
                                     fill=False, edgecolor="white", linewidth=1.0))
            axis.plot(bregma[1], bregma[0], marker="+", color="white", markersize=6)
            axis.set_title(label, fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.8, label="Pearson r (seed vs pixel)")
    fig.suptitle(title, fontsize=10)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def gsr_suffix(gsr: bool, global_mean: str) -> str:
    """Output-folder suffix: none without GSR, _gsr for Antea's column mean, _gsr_pixelmean."""
    if not gsr:
        return ""
    return "_gsr" if global_mean == "column" else "_gsr_pixelmean"


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--volume-file", default=defaults["volume_file"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--groups-csv", type=Path, default=defaults["groups_csv"])
    parser.add_argument("--roi-set-dir", type=Path, default=defaults["roi_set_dir"],
                        help="Folder of per-recording ROI sets (pixel_roi_editor.py) used instead of "
                             "each dump's own; the output subfolder gets a _roi_<folder> suffix")
    parser.add_argument("--exclude-recordings", nargs="*", default=defaults["exclude_recordings"])
    parser.add_argument("--gsr", action=argparse.BooleanOptionalAction, default=defaults["gsr"],
                        help="Regress the ROI-union global signal out of every pixel; output folder gets _gsr")
    parser.add_argument("--gsr-global-mean", choices=GLOBAL_MEANS, default=defaults["gsr_global_mean"],
                        help="column: Antea's nanmean(nanmean(data,1),2) (default, folder _gsr); "
                             "pixel: plain mean of the union pixels (folder _gsr_pixelmean)")
    parser.add_argument("--workers", type=int, default=defaults["workers"],
                        help="Parallel processes; 1 runs serially in the main process")
    parser.add_argument("--recording-limit", type=int, default=defaults["recording_limit"])
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=defaults["overwrite"])
    parser.add_argument("--plot-units", action=argparse.BooleanOptionalAction, default=defaults["plot_units"])
    parser.add_argument("--color-range-r", type=float, nargs=2, default=defaults["color_range_r"])
    parser.add_argument("--colormap", default=defaults["colormap"])
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
    if args.recording_limit is not None and args.recording_limit <= 0:
        raise ValueError("recording_limit must be positive")
    if not -1 <= args.color_range_r[0] < args.color_range_r[1] <= 1:
        raise ValueError("color_range_r must be (low, high) with -1 <= low < high <= 1")
    # After GSR r is centred near zero and often negative: the default 0..1 sequential
    # scale would clip half the map, so switch to a symmetric diverging one.
    if args.gsr and tuple(args.color_range_r) == (0.0, 1.0):
        args.color_range_r, args.colormap = (-1.0, 1.0), "RdBu_r"
    pixel_root = Path(args.pixel_root)
    output_dir = Path(args.output_root) / (Path(args.volume_file).name.removesuffix(".npy")
                                           + (gsr_suffix(args.gsr, args.gsr_global_mean))
                                           + roi_output_suffix(args.roi_set_dir))

    # Recordings to analyse; an exclusion that matches nothing is a typo, so it fails.
    folders = sorted(path.parent for path in pixel_root.glob("*/*/pixels_meta_full.npz"))
    if not folders:
        raise FileNotFoundError(f"No full pixel dumps found under {pixel_root}")
    excluded = {name.replace("\\", "/") for name in args.exclude_recordings}
    found = {f"{folder.parent.name}/{folder.name}" for folder in folders}
    if excluded - found:
        raise ValueError(f"Excluded recordings not found under {pixel_root}: {sorted(excluded - found)}")
    folders = [folder for folder in folders if f"{folder.parent.name}/{folder.name}" not in excluded]
    if args.recording_limit is not None:
        folders = folders[:args.recording_limit]
    missing = [folder for folder in folders if not (folder / args.volume_file).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings have no {args.volume_file}, e.g. {missing[0]}")

    # One job per recording; the output .npz is the resume marker.
    all_outputs = [output_dir / folder.parent.name / folder.name / RESULT_NAME for folder in folders]
    jobs = [dict(folder=str(folder), output_path=str(output_path), volume_file=args.volume_file,
                 roi_set_dir=args.roi_set_dir, gsr=args.gsr, gsr_global_mean=args.gsr_global_mean)
            for folder, output_path in zip(folders, all_outputs)
            if args.overwrite or not output_path.exists()]
    print(f"{len(folders)} recordings ({len(excluded)} excluded), {len(folders) - len(jobs)} already done, "
          f"{len(jobs)} to compute from {args.volume_file}, {args.workers} worker(s)\n"
          f"ROI sets: {args.roi_set_dir or 'each dump own (meta roi_set)'}\nOutput: {output_dir}",
          flush=True)

    started = time.perf_counter()

    def report(i_job, result):
        elapsed = time.perf_counter() - started
        remaining_min = elapsed / i_job * (len(jobs) - i_job) / 60
        nan_note = f", {result['n_nan_pixels']} NaN pixels" if result["n_nan_pixels"] else ""
        print(f"  [{i_job}/{len(jobs)}] {result['seconds']:.1f}s {result['folder']}{nan_note}  "
              f"(elapsed {elapsed / 60:.1f} min, ~{remaining_min:.1f} min left)", flush=True)

    if args.workers == 1 or len(jobs) <= 1:
        for i_job, job in enumerate(jobs, start=1):
            report(i_job, analyze_recording(job))
    elif jobs:
        # Pool's context exit calls terminate(), so Ctrl+C or a worker error stops
        # every worker instead of waiting for the queue (the trap in CLAUDE.md 9.10).
        with Pool(processes=min(args.workers, len(jobs))) as pool:
            for i_job, result in enumerate(pool.imap_unordered(analyze_recording, jobs, chunksize=1), start=1):
                report(i_job, result)

    # Load every recording's result (new and resumed) and index it by unit.
    recordings = []
    for output_path in all_outputs:
        with np.load(output_path, allow_pickle=False) as data:
            recordings.append({key: data[key] for key in data.files})
    labels = recordings[0]["roi_labels"].tolist()
    bregma = recordings[0]["bregma_crop_zero_based"]
    for result in recordings:
        # Maps can only be averaged pixel by pixel if every crop shares one Bregma position and ROI order.
        if result["roi_labels"].tolist() != labels or not np.array_equal(result["bregma_crop_zero_based"], bregma):
            raise ValueError(f"ROI labels or crop Bregma differ in {result['source']}")

    # Optional cross-check of the dumps' own groups against the group table.
    if args.groups_csv is not None:
        group_table = pd.read_csv(args.groups_csv, dtype=str)
        expected_group = {(row.day, row.animal): row.group for row in group_table.itertuples()}
        for result in recordings:
            key = (str(result["day"]), str(result["animal"]))
            if expected_group.get(key) != str(result["group"]):
                raise ValueError(f"Group of {key} is {result['group']} in its dump but "
                                 f"{expected_group.get(key)} in {args.groups_csv}")

    # Per-recording summary table.
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    summary = pd.DataFrame([dict(day=str(r["day"]), animal=str(r["animal"]), group=str(r["group"]),
                                 recording=str(r["recording"]),
                                 mean_offdiag_R_roi=float(r["R_roi"][off_diagonal].mean()),
                                 mean_abs_map_r=float(np.nanmean(np.abs(r["r_maps"]))),
                                 roi_set=str(r["roi_set"]))
                            for r in recordings])
    summary.to_csv(output_dir / "recording_summary.csv", index=False)

    # Unit maps: Fisher-z mean over the unit's recordings.
    # unit_maps[unit]: (n_roi, rows, cols)
    units = sorted({(str(r["day"]), str(r["animal"])) for r in recordings})
    unit_maps, unit_boxes, unit_group = {}, {}, {}
    for day, animal in units:
        members = [r for r in recordings if (str(r["day"]), str(r["animal"])) == (day, animal)]
        unit_maps[(day, animal)] = fisher_mean(np.stack([r["r_maps"].astype(np.float64) for r in members]))
        unit_boxes[(day, animal)] = members[0]["box_slices_zero_based_exclusive"]
        unit_group[(day, animal)] = str(members[0]["group"])
        unit_dir = output_dir / f"{day}_{animal}"
        np.savez(unit_dir / UNIT_RESULT_NAME, r_maps=unit_maps[(day, animal)].astype(np.float32),
                 roi_labels=np.array(labels), recordings=np.array([str(r["recording"]) for r in members]),
                 box_slices_zero_based_exclusive=unit_boxes[(day, animal)],
                 bregma_crop_zero_based=bregma, day=day, animal=animal, group=unit_group[(day, animal)])
        if args.plot_units:
            draw_seed_maps(unit_maps[(day, animal)], labels, bregma, unit_boxes[(day, animal)],
                           f"{day} {animal} (group {unit_group[(day, animal)]}): seed-pixel r, "
                           f"Fisher mean of {len(members)} recordings, {args.volume_file}",
                           tuple(args.color_range_r), args.colormap, unit_dir / "seed_maps.png")

    # Group maps: Fisher-z mean over units, so each animal-day weighs the same.
    # group_maps: (n_groups, n_roi, rows, cols)
    groups = sorted(set(unit_group.values()))
    group_maps = np.stack([fisher_mean(np.stack([unit_maps[unit] for unit in units if unit_group[unit] == group]))
                           for group in groups])
    n_units = [sum(unit_group[unit] == group for unit in units) for group in groups]
    # The seed boxes differ slightly between animals (Lambda scaling); draw their mean outline.
    group_boxes = np.stack([np.mean([unit_boxes[unit] for unit in units if unit_group[unit] == group], axis=0)
                            for group in groups])
    np.savez(output_dir / "group_roi_pixel_connectivity.npz", r_maps=group_maps.astype(np.float32),
             groups=np.array(groups), n_units=np.array(n_units), roi_labels=np.array(labels),
             mean_box_slices_zero_based_exclusive=group_boxes, bregma_crop_zero_based=bregma,
             volume_file=args.volume_file)
    for i_group, group in enumerate(groups):
        draw_seed_maps(group_maps[i_group], labels, bregma, group_boxes[i_group],
                       f"Group {group}: seed-pixel r, Fisher mean of {n_units[i_group]} animal-days, "
                       f"{args.volume_file}", tuple(args.color_range_r), args.colormap, output_dir / f"group_{group}_seed_maps.png")
    print(f"done: {len(recordings)} recordings, {len(units)} units, groups "
          + ", ".join(f"{group} ({n})" for group, n in zip(groups, n_units))
          + f" in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
