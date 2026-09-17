"""Run the cohort event detector on the COUNT of active pixels, not on ROI means.

Signal
    For every recording, a saved per-pixel dF/F volume (n_time, rows, cols) is
    thresholded pixel by pixel. `--dff-source` picks the volume:

    `median_dff` (default)
        `pixels_median_dff_<window>s_full.npy`, built once by cache_median_dff.py:
        (F/Fbar)/(R/Rbar) - 1 against a centred running-median baseline, the same
        per-pixel signal epileptic_by_area_animal_day_pixels.py averages.
    `pipeline_dff`
        `pixels_dff_full.npy`, written by run_botox_batch.py: percent dF/F against
        the whole-recording MEAN, so slow drift stays in each pixel.

    Then

        z(t, p)   = (dff(t, p) - median_t dff(p)) / robust_sd_t dff(p)
        active    = z(t, p) > pixel_z_threshold
        signal(t) = (number of active pixels) / (number of pixels considered)

    The threshold is in SDs of each pixel's OWN trace, so a noisy pixel (vessel,
    edge) is not active more often than a quiet one. `--pixel-scale` picks the SD:

    `mad` (default)
        1.4826 x MAD of the whole trace (robust SD).
    `bottom_percentile`
        Plain SD of the pixel's values at or below its own
        `--pixel-scale-percentile` percentile (default 50, the bottom half). The
        top of the trace, where the events are, then cannot inflate the scale.
        This is the SD of a TRUNCATED distribution, so it is smaller than the
        full SD: for Gaussian noise the bottom 50% gives 0.60 sigma, i.e. a
        threshold of 1.5 here is about 0.9 sigma of the whole trace.

    The centre is the pixel's median in both cases. Pixels with a zero scale are
    left out of both the count and the denominator; how many were dropped is
    written to the geometry JSON.

Pixels considered
    Only pixels inside the union of the recording's atlas boxes (22 for cortex22),
    split by hemisphere into two traces, `CortexL_active` and `CortexR_active`.
    The box boundaries are otherwise ignored: a pixel counts once, whichever box
    it is in. Keeping to the boxes excludes the skull/head the saved crop also
    contains; keeping two traces preserves the left/right tables.

Detection
    Unchanged: `run_analysis` from epileptic_by_area_animal_day.py detrends each
    trace, smooths it (the "amplitude"), takes the rise, and flags peaks against
    one z cut-off pooled over every recording. Both amplitude and rise therefore
    come from the active-pixel fraction. The rise is taken from the SMOOTHED
    fraction by default (`--rise-on-smoothed`); `--no-rise-on-smoothed` takes it
    from the detrended, unsmoothed fraction instead. The output folder is named
    for that choice, so the two runs do not overwrite each other.

    One extra condition for an epileptiform peak: `--epileptic-min-signal`
    (default 0.20) requires at least that fraction of the hemisphere's pixels to
    be active at the peak frame, read from the raw fraction (before detrending and
    smoothing) and saved per peak as `signal_at_peak`. 0 disables it.

NOTE: the per-pixel median/MAD is over the whole recording. With `pipeline_dff`
slow drift in a pixel's dF/F therefore moves how often it is active; the
`median_dff` cache has that drift divided out already. The detector's own 20 s
running median then removes slow changes in the COUNT, not in each pixel.

Examples (in the letizia environment):
    python epileptic_by_active_pixels.py
    python epileptic_by_active_pixels.py --recording-limit 2
    python epileptic_by_active_pixels.py --pixel-z-threshold 2
    python epileptic_by_active_pixels.py --pixel-scale bottom_percentile --pixel-scale-percentile 50
    python epileptic_by_active_pixels.py --exclude-recordings 260828_PV7/t2 260611_PV5/t1
    python epileptic_by_active_pixels.py --no-rise-on-smoothed
    python epileptic_by_active_pixels.py --dff-source pipeline_dff
    python epileptic_by_active_pixels.py --debug-plot-count 0
    python epileptic_by_active_pixels.py --epileptic-min-signal 0.3   # 30% of pixels active
    python epileptic_by_active_pixels.py --workers 1                  # serial, for debugging

Edit RUN_CONFIG below to run directly from the editor without CLI arguments.
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cache_median_dff import cache_paths
from epileptic_by_area_animal_day import run_analysis
from epileptic_by_area_animal_day_pixels import crop_slices, nonnegative, plot_count, roi_categories

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the analysis without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",
    "input_root": REPO_ROOT / "outputs/botox_restani_rebuilt",  # batch_summary.csv
    # None: outputs/epileptic_by_active_pixels_<dff_source>_rise_<smoothed|detrended>
    # (+ _bottom<N>sd with the bottom-percentile scale), so variants do not overwrite each other.
    "output_dir": None,
    "dff_source": "median_dff",  # "median_dff" (cache_median_dff.py) or "pipeline_dff"; see docstring.
    "baseline_window_s": 20.0,  # Which median_dff cache to read (it is named for its window).
    # A pixel is active above this many robust SDs of its own trace. 3 fails: the
    # fraction is then 0 in most frames and has no robust SD (CLAUDE.md 9.23).
    "pixel_z_threshold": 2.0,
    # SD each pixel is thresholded in: "mad" = 1.4826 x MAD of the whole trace;
    # "bottom_percentile" = SD of the values at or below pixel_scale_percentile.
    "pixel_scale": "mad",
    "pixel_scale_percentile": 50.0,  # Used only by "bottom_percentile"; 50 = bottom half.
    # <date>_<animal>/<t#> left out entirely. 260828_PV7/t2 has a field-wide
    # reflectance glitch at frame 1809 that reads as a whole-cortex event (9.23).
    "exclude_recordings": ["260828_PV7/t2"],
    "debug_plot_count": None,  # None: every recording; an integer samples that many; 0 disables.
    "debug_animals_per_group": 5,  # None: every animal; otherwise this many random animals per group.
    "debug_width_scale": 30.0,  # Figure width x this; 30 gives ~1.6 in (190 px) per second.
    "debug_seed": 0,  # Seeds both the animal and the recording sampling.
    "recording_limit": None,  # None: all recordings; positive integer: smoke test.
    # Processes for the per-recording detection and figures; 1 = serial in this
    # process (easiest to debug). 8 leaves the machine usable meanwhile.
    "workers": 8,
    "detection": {
        "sampling_rate_hz": 10.0,  # Per-channel rate, after splitting the channels.
        "median_window_s": 20.0,  # Running median removed as the slow trend.
        "smooth_window_s": 1.0,  # Rolling mean applied before finding peaks.
        "min_event_distance_s": 0.5,  # Minimum spacing between peaks in one trace.
        "amplitude_prominence_sd": 3.0,  # Lower = admit smaller amplitude peaks.
        "rise_prominence_sd": 3.0,  # Lower = admit less prominent rise peaks.
        "rise_window_s": 0.5,  # Interval over which the rise is measured.
        "not_linear_rise": True,  # True: window max-min; False: endpoint change.
        "rise_on_smoothed": True,  # True: rise of the smoothed fraction; False: of the detrended one.
        "rise_lead_s": 1.0,  # Allowed rise time before the amplitude peak.
        "rise_lag_s": 0.3,  # Allowed rise time after the amplitude peak.
        "epileptic_require_confirmed": True,  # Require a concurrent rise peak.
        # An epileptiform peak also needs at least this fraction of the hemisphere's
        # pixels active at the peak frame (raw fraction, before detrending and
        # smoothing; column signal_at_peak). 0.20 = 20%; 0 disables the gate.
        "epileptic_min_signal": 0.20,
        "calibration_frame_percent": 1.0,  # Top % of pooled frame z values; larger = lower cutoff.
        "z_histogram_step": 0.01,  # Resolution of the pooled z cutoff.
        # The active fraction has a small robust SD, so its z values run far higher
        # than an ROI mean's: at 2 SD the 1% cut-off was z 63 on 16 recordings.
        # A cut-off at the top edge raises (CLAUDE.md 9.23).
        "z_histogram_range": (-10.0, 1000.0),
    },
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

AREA_NAME = "Cortex_active"  # Area label of both traces in the cohort tables.
TRACE_NAMES = {"L": "CortexL_active", "R": "CortexR_active"}  # hemisphere -> trace column
SIGNAL_UNIT = "fraction of active pixels"
TRACE_CSV_NAME = "active_pixel_fraction.csv"
DFF_SOURCES = ("median_dff", "pipeline_dff")
PIXEL_SCALES = ("mad", "bottom_percentile")


def default_output_dir(dff_source: str, rise_on_smoothed: bool, pixel_scale: str = "mad",
                       pixel_scale_percentile: float = 50.0) -> Path:
    rise_source = "smoothed" if rise_on_smoothed else "detrended"
    # The MAD run keeps its original name, so existing outputs stay where they are.
    scale = "" if pixel_scale == "mad" else f"_bottom{pixel_scale_percentile:g}sd"
    return REPO_ROOT / f"outputs/epileptic_by_active_pixels_{dff_source}_rise_{rise_source}{scale}"


def dff_volume_path(folder: Path, dff_source: str, baseline_window_s: float) -> Path:
    """The saved dF/F volume of one dump folder; a missing cache is an error, not a fallback."""
    if dff_source == "pipeline_dff":
        return folder / "pixels_dff_full.npy"
    volume_path, meta_path = cache_paths(folder, baseline_window_s)
    if not meta_path.exists():
        raise FileNotFoundError(f"No finished {baseline_window_s:g} s median dF/F cache in {folder}; "
                                f"run cache_median_dff.py --window-s {baseline_window_s:g} first")
    return volume_path


def hemisphere_masks(boxes: dict, y_1: int, x_2: int, region, crop_shape) -> dict[str, np.ndarray]:
    """Union of the atlas boxes of each hemisphere, as boolean masks on the saved crop.

    Returns {"L": (rows, cols) bool, "R": (rows, cols) bool}. The hemispheres come
    from the box names (M2L_alta -> L), not from the Bregma column, so a box is
    counted on the side it was drawn for.
    """
    masks = {side: np.zeros(crop_shape, dtype=bool) for side in TRACE_NAMES}
    for name, (_, side) in roi_categories(boxes).items():
        rs, cs = crop_slices(boxes[name], y_1, x_2, region)
        masks[side][rs, cs] = True
    if (masks["L"] & masks["R"]).any():
        raise ValueError("Left and right atlas boxes overlap; a pixel would count on both sides")
    return masks


def pixel_scales(pixels: np.ndarray, pixel_median: np.ndarray, pixel_scale: str,
                 percentile: float) -> np.ndarray:
    """Per-pixel SD used as the threshold unit; `pixels` is (n_time, n_pixels)."""
    if pixel_scale == "mad":
        return 1.4826 * np.median(np.abs(pixels - pixel_median), axis=0)
    if pixel_scale != "bottom_percentile":
        raise ValueError(f"pixel_scale must be one of {PIXEL_SCALES}")
    if not 0 < percentile <= 100:
        raise ValueError("pixel_scale_percentile must be in (0, 100]")
    # Each pixel's values in ascending order; the SD is over the lowest n_bottom
    # of them, i.e. the frames at or below the pixel's own percentile.
    n_bottom = int(np.ceil(pixels.shape[0] * percentile / 100))
    if n_bottom < 2:
        raise ValueError("pixel_scale_percentile keeps fewer than 2 frames; no SD")
    bottom = np.sort(pixels, axis=0)[:n_bottom]  # (n_bottom, n_pixels)
    return bottom.std(axis=0)


def active_pixel_fraction(dff: np.ndarray, mask: np.ndarray, z_threshold: float,
                          pixel_scale: str = "mad", pixel_scale_percentile: float = 50.0,
                          ) -> tuple[np.ndarray, int]:
    """Fraction of the masked pixels above `z_threshold` SDs of their own trace, per frame.

    `dff` is (n_time, rows, cols), `mask` is (rows, cols). The SD is chosen by
    `pixel_scale` (see `pixel_scales`). Returns the (n_time,) fraction and the
    number of masked pixels dropped for having a zero scale.
    """
    # pixels: (n_time, n_masked_pixels), widened so float16 cannot overflow the MAD.
    pixels = np.asarray(dff[:, mask], dtype=np.float64)
    if not pixels.size:
        raise ValueError("Mask selects no pixels")
    if not np.isfinite(pixels).all():
        raise ValueError("dF/F volume contains non-finite values inside the mask")
    pixel_median = np.median(pixels, axis=0)  # (n_masked_pixels,)
    scale = pixel_scales(pixels, pixel_median, pixel_scale, pixel_scale_percentile)  # (n_masked_pixels,)
    has_scale = scale > 0
    if not has_scale.any():
        raise ValueError("Every masked pixel is flat; no pixel can be thresholded")
    # z: (n_time, n_valid_pixels)
    z = (pixels[:, has_scale] - pixel_median[has_scale]) / scale[has_scale]
    fraction = (z > z_threshold).mean(axis=1)  # (n_time,)
    return fraction, int((~has_scale).sum())


class ActivePixelTraceLoader:
    """`run_analysis` trace loader: pixel dump folder -> (trial, frame, L, R) table.

    Each recording is loaded twice (detection, then figures), so the table is
    written once and re-read from its CSV the second time.
    """

    def __init__(self, output_root, *, pixel_z_threshold=3.0, dff_source="median_dff",
                 baseline_window_s=20.0, pixel_scale="mad", pixel_scale_percentile=50.0):
        if not np.isfinite(pixel_z_threshold):
            raise ValueError("pixel_z_threshold must be finite")
        if dff_source not in DFF_SOURCES:
            raise ValueError(f"dff_source must be one of {DFF_SOURCES}")
        if pixel_scale not in PIXEL_SCALES:
            raise ValueError(f"pixel_scale must be one of {PIXEL_SCALES}")
        if not 0 < pixel_scale_percentile <= 100:
            raise ValueError("pixel_scale_percentile must be in (0, 100]")
        self.pixel_scale = pixel_scale
        self.pixel_scale_percentile = float(pixel_scale_percentile)
        self.output_root = Path(output_root)
        self.pixel_z_threshold = float(pixel_z_threshold)
        self.dff_source = dff_source
        self.baseline_window_s = float(baseline_window_s)
        self.cached = {}
        # Area and hemisphere of each trace, handed to run_analysis for the tables.
        self.categories = {name: (AREA_NAME, side) for side, name in TRACE_NAMES.items()}

    def __call__(self, metadata_path):
        metadata_path = Path(metadata_path)
        if metadata_path in self.cached:
            return pd.read_csv(self.cached[metadata_path])
        folder = metadata_path.parent
        with np.load(metadata_path, allow_pickle=False) as data:
            meta = {key: data[key].tolist() for key in data.files
                    if key not in ("mean_f", "mean_r")}
        if meta["axis_order"] != "time,y,x":
            raise ValueError(f"Unsupported pixel axes in {metadata_path}")
        if meta["n_written"] != meta["n_time"] or meta["n_time"] <= 0:
            raise ValueError(f"Incomplete pixel dump: {metadata_path}")

        # The recording's own atlas: same checks as the ROI-mean pixel script.
        atlas_path = Path(meta["roi_set"])
        if not atlas_path.is_absolute():
            atlas_path = REPO_ROOT / atlas_path
        atlas = yaml.safe_load(atlas_path.read_text(encoding="utf-8"))
        if list(atlas["grid"]) != meta["grid"]:
            raise ValueError(f"Atlas and pixel grid differ: {atlas_path}")
        for key in ("bregma_row", "bregma_col", "downsample"):
            if atlas[key] != meta[key]:
                raise ValueError(f"Atlas {key} differs from saved metadata: {atlas_path}")

        region = meta.get("region", [0, meta["grid"][0], 0, meta["grid"][1]])
        volume_path = dff_volume_path(folder, self.dff_source, self.baseline_window_s)
        dff = np.load(volume_path, mmap_mode="r")  # (n_time, rows, cols)
        expected = (meta["n_time"], region[1] - region[0], region[3] - region[2])
        if dff.shape != expected or list(expected) != meta["shape"]:
            raise ValueError(f"dF/F volume shape disagrees with metadata: {folder}")
        masks = hemisphere_masks(atlas["boxes"], int(meta["y_1"]), int(meta["x_2"]),
                                 region, expected[1:])

        traces = {"trial": np.zeros(expected[0], dtype=int), "frame": np.arange(expected[0])}
        n_flat = {}
        for side, name in TRACE_NAMES.items():
            traces[name], n_flat[name] = active_pixel_fraction(
                dff, masks[side], self.pixel_z_threshold,
                self.pixel_scale, self.pixel_scale_percentile)
        result = pd.DataFrame(traces)

        out = self.output_root / folder.parent.name / folder.name
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / TRACE_CSV_NAME
        result.to_csv(csv_path, index=False)
        geometry = dict(source=str(metadata_path.resolve()), roi_set=str(atlas_path),
                        volume=str(volume_path), dff_source=self.dff_source, region=region,
                        y_1=meta["y_1"], x_2=meta["x_2"],
                        pixel_z_threshold=self.pixel_z_threshold,
                        pixel_scale=self.pixel_scale,
                        pixel_scale_percentile=(self.pixel_scale_percentile
                                                if self.pixel_scale == "bottom_percentile" else None),
                        boxes=atlas["boxes"],
                        n_mask_pixels={TRACE_NAMES[side]: int(mask.sum()) for side, mask in masks.items()},
                        n_flat_pixels_dropped=n_flat,
                        # Crop-relative 0-based (row, col) of every counted pixel, per trace.
                        mask_pixels_zero_based={TRACE_NAMES[side]: np.argwhere(mask).tolist()
                                                for side, mask in masks.items()})
        (out / "active_pixel_geometry.json").write_text(json.dumps(geometry), encoding="utf-8")
        self.cached[metadata_path] = csv_path
        return result


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--input-root", type=Path, default=defaults["input_root"],
                        help="Directory containing batch_summary.csv (group assignments)")
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"],
                        help="Default: outputs/epileptic_by_active_pixels_<dff_source>_rise_<smoothed|detrended>")
    parser.add_argument("--dff-source", choices=DFF_SOURCES, default=defaults["dff_source"],
                        help="median_dff: cache_median_dff.py volume; pipeline_dff: run_botox_batch's dump")
    parser.add_argument("--baseline-window-s", type=float, default=defaults["baseline_window_s"],
                        help="Window of the median_dff cache to read")
    parser.add_argument("--pixel-z-threshold", type=float, default=defaults["pixel_z_threshold"],
                        help="A pixel is active above this many SDs (see --pixel-scale) of its own trace")
    parser.add_argument("--pixel-scale", choices=PIXEL_SCALES, default=defaults["pixel_scale"],
                        help="mad: 1.4826 x MAD; bottom_percentile: SD of the values at or below "
                             "--pixel-scale-percentile")
    parser.add_argument("--pixel-scale-percentile", type=float, default=defaults["pixel_scale_percentile"],
                        help="Percentile for --pixel-scale bottom_percentile (default 50 = bottom half)")
    parser.add_argument("--exclude-recordings", nargs="*", default=defaults["exclude_recordings"],
                        metavar="DATE_ANIMAL/T#", help="Recordings to leave out, e.g. 260828_PV7/t2")
    parser.add_argument("--debug-plot-count", type=plot_count, default=defaults["debug_plot_count"],
                        help="Recordings to draw debug plots for: 'all' (default), 0 to disable, "
                             "or an integer to sample that many")
    parser.add_argument("--debug-animals-per-group", type=plot_count,
                        default=defaults["debug_animals_per_group"],
                        help="Draw figures only for this many random animals per group "
                             "(all of their recordings); 'all' for every animal")
    parser.add_argument("--debug-width-scale", type=float, default=defaults["debug_width_scale"],
                        help="Stretch the per-recording figures horizontally by this factor")
    parser.add_argument("--debug-seed", type=nonnegative, default=defaults["debug_seed"])
    parser.add_argument("--recording-limit", type=int, default=defaults["recording_limit"],
                        help="Analyze only the first N recordings for a smoke test; default: all")
    parser.add_argument("--workers", type=int, default=defaults["workers"],
                        help="Processes for detection and figures; 1 runs serially")
    # CLI overrides use the same names as the editable detection dictionary.
    for name, value in defaults["detection"].items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif name == "z_histogram_range":
            parser.add_argument(flag, type=float, nargs=2, default=value, metavar=("LOW", "HIGH"))
        else:
            parser.add_argument(flag, type=float, default=value)
    args = parser.parse_args()
    args.detection = {name: vars(args).pop(name) for name in defaults["detection"]}
    return args


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        pixel_root=Path(config["pixel_root"]),
        input_root=Path(config["input_root"]),
        output_dir=config["output_dir"],
        dff_source=config["dff_source"],
        baseline_window_s=float(config["baseline_window_s"]),
        pixel_z_threshold=float(config["pixel_z_threshold"]),
        pixel_scale=config["pixel_scale"],
        pixel_scale_percentile=float(config["pixel_scale_percentile"]),
        exclude_recordings=list(config["exclude_recordings"]),
        debug_plot_count=plot_count(config["debug_plot_count"]),
        debug_animals_per_group=plot_count(config["debug_animals_per_group"]),
        debug_width_scale=float(config["debug_width_scale"]),
        debug_seed=int(config["debug_seed"]),
        recording_limit=config["recording_limit"],
        workers=int(config["workers"]),
        detection=dict(config["detection"]),
    )


def main():
    args = build_runtime_args()
    # The output folder follows the rise source unless one was given explicitly.
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.dff_source, args.detection["rise_on_smoothed"],
                                             args.pixel_scale, args.pixel_scale_percentile)
    args.pixel_root, args.input_root, args.output_dir = map(
        Path, (args.pixel_root, args.input_root, args.output_dir))
    if not np.isfinite(args.debug_width_scale) or args.debug_width_scale <= 0:
        raise ValueError("debug_width_scale must be finite and positive")
    if args.recording_limit is not None and args.recording_limit <= 0:
        raise ValueError("recording_limit must be positive")
    paths = sorted(args.pixel_root.glob("*/*/pixels_meta_full.npz"))
    if not paths:
        raise FileNotFoundError(f"No full pixel dumps found under {args.pixel_root}")
    # Drop excluded recordings; an exclusion that matches nothing is a typo, so it fails.
    excluded = {name.replace("\\", "/") for name in args.exclude_recordings}
    found = {f"{path.parent.parent.name}/{path.parent.name}" for path in paths}
    if excluded - found:
        raise ValueError(f"Excluded recordings not found under {args.pixel_root}: {sorted(excluded - found)}")
    paths = [path for path in paths if f"{path.parent.parent.name}/{path.parent.name}" not in excluded]
    print(f"Excluded: {sorted(excluded) or 'none'}")
    loader = ActivePixelTraceLoader(args.output_dir, pixel_z_threshold=args.pixel_z_threshold,
                                    dff_source=args.dff_source,
                                    baseline_window_s=args.baseline_window_s,
                                    pixel_scale=args.pixel_scale,
                                    pixel_scale_percentile=args.pixel_scale_percentile)
    rise_source = "smoothed" if args.detection["rise_on_smoothed"] else "detrended"
    dff_description = (f"median dF/F ({args.baseline_window_s:g} s running-median baseline, cached)"
                       if args.dff_source == "median_dff" else "run_botox_batch dF/F (mean baseline)")
    scale_description = ("1.4826 * MAD over the whole recording, per pixel"
                         if args.pixel_scale == "mad" else
                         f"SD of the values at or below the pixel's own {args.pixel_scale_percentile:g}th "
                         f"percentile, over the whole recording")
    scale_short = ("robust SDs" if args.pixel_scale == "mad"
                   else f"bottom-{args.pixel_scale_percentile:g}% SDs")
    signal_description = (f"fraction of atlas-box pixels per hemisphere with {dff_description} above "
                          f"{args.pixel_z_threshold:g} {scale_short} of the pixel's own trace; "
                          f"rise from the {rise_source} fraction")
    print(f"Signal: {signal_description}")
    print(f"Output: {args.output_dir}")
    run_analysis(input_root=args.input_root, output_dir=args.output_dir,
                 recording_paths=paths, recording_limit=args.recording_limit,
                 trace_loader=loader, diagnostics_root=args.output_dir,
                 diagnostic_plot_count=args.debug_plot_count, diagnostic_seed=args.debug_seed,
                 diagnostic_animals_per_group=args.debug_animals_per_group,
                 diagnostic_width_scale=args.debug_width_scale,
                 roi_metadata=loader.categories, signal_unit=SIGNAL_UNIT, workers=args.workers,
                 provenance=dict(signal=signal_description, signal_mode="active_pixel_fraction",
                                 dff_source=args.dff_source,
                                 baseline_window_s=(args.baseline_window_s
                                                    if args.dff_source == "median_dff" else None),
                                 pixel_z_threshold=args.pixel_z_threshold,
                                 excluded_recordings=sorted(excluded),
                                 pixel_scale=scale_description,
                                 pixel_scale_percentile=(args.pixel_scale_percentile
                                                         if args.pixel_scale == "bottom_percentile" else None),
                                 pixels_counted="union of the recording's atlas boxes, per hemisphere",
                                 pixel_root=str(args.pixel_root.resolve())),
                 **args.detection)


if __name__ == "__main__":
    main()
