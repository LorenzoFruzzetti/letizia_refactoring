"""Run the cohort event detector on pixel dumps, correcting for reflectance.

Two signal modes, chosen with --signal-mode:

`median_dff` (default)
    dF/F(t,p) = (F(t,p) / Fbar(t,p)) / (R(t,p) / Rbar(t,p)) - 1, then the spatial
    ROI mean. This is the correction run_botox_batch.py applies (wfci.correction.
    hemodynamic_correction) with one change: the baselines Fbar and Rbar are a
    CENTRED RUNNING MEDIAN over `baseline_median_window_s` seconds of each pixel's
    own trace, not that pixel's mean over the whole recording. A local baseline
    divides out slow drift in either channel instead of leaving it in the trace.
    The result is a ratio, not a percentage: multiply by 100 for the % that
    run_botox_batch writes. Detection is unaffected by that factor because every
    threshold downstream is in robust SDs of the trace itself.

`reflectance_ratio`
    F_corrected(t,p) = F(t,p) * mean_t(R(t,p)) / R(t,p), then the spatial ROI mean.
    No F-baseline division, no subtraction of 1, no *100; raw fluorescence units.

NOTE: the detector then subtracts its own running median (`--median-window-s`,
also 20 s by default). With `median_dff` the signal is therefore high-passed
twice - once per pixel before averaging, once per ROI after. That is deliberate
here (the two act on different quantities), but set `--median-window-s` larger,
or switch modes, if only one stage is wanted.

ROI boxes: `--all-rois` takes every box of each recording's saved atlas (22 for
cortex22). `--custom-boxes` adds the four hand-specified right motor boxes below,
which REPLACE the atlas boxes of the same name; by default they are used only
when --all-rois is off, because the replacements are not the mirror images of
their left twins and would make a bilateral comparison asymmetric.
--reference-lambda optionally scales the four custom box centres by the
recording's Lambda distance / reference distance, keeping box sizes fixed (as in
roi_editor).

The inputs have already been trimmed/downsampled/cropped by run_botox_batch.py;
they are not trimmed again. Correction happens at the saved pixel resolution.

Examples (in the letizia environment):
    python epileptic_by_area_animal_day_pixels.py
    python epileptic_by_area_animal_day_pixels.py --debug-plot-count 0
    python epileptic_by_area_animal_day_pixels.py --no-all-rois --custom-boxes
    python epileptic_by_area_animal_day_pixels.py --signal-mode reflectance_ratio
    python epileptic_by_area_animal_day_pixels.py --recording-limit 2
    python epileptic_by_area_animal_day_pixels.py --rise-on-smoothed

All recordings are analyzed and calibrated together; the per-recording figures
are drawn only for --debug-animals-per-group random animals per group (every
recording of each), optionally thinned by --debug-plot-count (`all` for every
recording, 0 for none), and stretched --debug-width-scale times wider so a
single second can be read. Outputs, extracted ROI traces and exact
geometry go into a separate output tree.
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

from epileptic_by_area_animal_day import run_analysis

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the analysis without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",
    "input_root": REPO_ROOT / "outputs/botox_restani_rebuilt",  # batch_summary.csv
    # Named for the signal mode: a reflectance_ratio run must not overwrite a
    # median_dff one, the per-recording tables and figures having the same names.
    "output_dir": REPO_ROOT / "outputs/epileptic_by_area_animal_day_pixels_median_dff_wide_smoothed",
    "all_rois": True,  # True: every box of each recording's atlas; False: custom only.
    "custom_boxes": None,  # None: custom boxes only when all_rois is False. True/False force it.
    "signal_mode": "median_dff",  # "median_dff" or "reflectance_ratio"; see the docstring.
    "baseline_median_window_s": 20.0,  # Per-pixel running median used as Fbar/Rbar.
    "reference_lambda": None,  # None: fixed offsets; otherwise scale box centres.
    "debug_plot_count": None,  # None: every recording; an integer samples that many; 0 disables.
    "debug_animals_per_group": 5,  # None: every animal; otherwise this many random animals per group.
    "debug_width_scale": 30.0,  # Figure width x this; 30 gives ~1.6 in (190 px) per second.
    "debug_seed": 0,  # Seeds both the animal and the recording sampling.
    "recording_limit": None,  # None: all recordings; positive integer: smoke test.
    "detection": {
        "sampling_rate_hz": 10.0,  # Per-channel rate, after splitting the channels.
        "median_window_s": 20.0,  # Running median removed as the slow trend.
        "smooth_window_s": 1.0,  # Rolling mean applied before finding peaks.
        "min_event_distance_s": 0.5,  # Minimum spacing between peaks in one ROI.
        "amplitude_prominence_sd": 3.0,  # Lower = admit smaller amplitude peaks.
        "rise_prominence_sd": 3.0,  # Lower = admit less prominent rise peaks.
        "rise_window_s": 0.5,  # Interval over which the rise is measured.
        "not_linear_rise": True,  # True: window max-min; False: endpoint change.
        "rise_on_smoothed": True,  # True: rise of the smoothed amplitude; False: of the detrended trace.
        "rise_lead_s": 1.0,  # Allowed rise time before the amplitude peak.
        "rise_lag_s": 0.3,  # Allowed rise time after the amplitude peak.
        "epileptic_require_confirmed": True,  # Require a concurrent rise peak.
        "calibration_frame_percent": 1.0,  # Top % of pooled frame z values; larger = lower cutoff.
        "z_histogram_step": 0.01,  # Resolution of the pooled z cutoff.
        "z_histogram_range": (-10.0, 30.0),
    },
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

CUSTOM_BOXES = {
    "M2R_alta": dict(row_start=-23, row_end=-18, col_start=8, col_end=13),
    "M2R_bassa": dict(row_start=-8, row_end=-3, col_start=3, col_end=8),
    "M1R_alta": dict(row_start=-14, row_end=-9, col_start=23, col_end=28),
    "M1R_bassa": dict(row_start=-6, row_end=-1, col_start=12, col_end=17),
}


SIGNAL_MODES = ("median_dff", "reflectance_ratio")

# What each mode's trace is in, for plot axes and the provenance JSON.
SIGNAL_UNITS = {"median_dff": "dF/F", "reflectance_ratio": "fluorescence units"}

# One trace file name per mode, so two runs cannot overwrite each other's traces.
TRACE_CSV_NAMES = {"median_dff": "roi_median_dff.csv",
                   "reflectance_ratio": "roi_reflectance_corrected_fluorescence.csv"}


def plot_count(value):
    """`all` (or None from RUN_CONFIG) means every recording; 0 disables plots."""
    if value is None or str(value).strip().lower() == "all":
        return None
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be >= 0 or 'all'")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return result


def median_window_frames(window_s, sampling_rate_hz):
    """Odd frame count spanning `window_s`, the convention the detector also uses."""
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("baseline_median_window_s must be finite and positive")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    return 2 * round(window_s * sampling_rate_hz / 2) + 1


def _as_pixel_pair(fluorescence, reflectance):
    """Validate and widen one ROI's [time,row,col] pixel pair to float64."""
    f = np.asarray(fluorescence, dtype=np.float64)
    r = np.asarray(reflectance, dtype=np.float64)
    if f.shape != r.shape or f.ndim != 3 or not f.size:
        raise ValueError("Fluorescence and reflectance must be matching nonempty 3-D arrays")
    if not np.isfinite(f).all() or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError("ROI contains non-finite fluorescence/reflectance or nonpositive reflectance")
    return f, r


def running_median(traces, n_window):
    """Centred running median down axis 0 of a (n_time, n_pixels) array.

    `min_periods=1` so the partial windows at both ends still give a baseline,
    matching `remove_slow_trend` in epileptic_by_area_animal_day.py.
    """
    return (pd.DataFrame(traces)
            .rolling(window=n_window, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float))


def corrected_roi_trace(fluorescence, reflectance):
    """Correct each saved pixel first, then average; inputs are [time,row,col]."""
    f, r = _as_pixel_pair(fluorescence, reflectance)
    corrected = f * r.mean(axis=0, keepdims=True) / r
    return corrected.mean(axis=(1, 2))


def median_dff_roi_trace(fluorescence, reflectance, n_median):
    """(F/Fbar)/(R/Rbar) - 1 per pixel against running-median baselines, then averaged.

    Same algebra as wfci.correction.hemodynamic_correction, but Fbar and Rbar are
    each pixel's centred `n_median`-frame running median instead of its mean over
    the whole recording, and the result is left as a ratio rather than a percent.
    """
    f, r = _as_pixel_pair(fluorescence, reflectance)
    n_time = f.shape[0]
    # Flatten the box to (n_time, n_pixels): the median is per pixel, independently.
    flat_f = f.reshape(n_time, -1)
    flat_r = r.reshape(n_time, -1)
    baseline_f = running_median(flat_f, n_median)  # (n_time, n_pixels)
    baseline_r = running_median(flat_r, n_median)  # (n_time, n_pixels)
    if np.any(baseline_f <= 0) or np.any(baseline_r <= 0):
        raise ValueError("Running-median baseline is nonpositive; cannot form F/Fbar")
    dff = (flat_f / baseline_f) / (flat_r / baseline_r) - 1.0
    return dff.mean(axis=1)


def crop_slices(box, y_1, x_2, region):
    """Atlas offsets are MATLAB 1-based inclusive; region is 0-based exclusive.

    The exploratory test_new_epilectic_pixels.py maps crop pixels using window_rel.
    Use the stored full-grid origin here, and include the atlas's -1 conversion so
    the boxes match wfci.roi.box_slices_for exactly.
    """
    r0, r1, c0, c1 = map(int, region)
    rs = slice(y_1 + box["row_start"] - 1 - r0, y_1 + box["row_end"] - r0)
    cs = slice(x_2 + box["col_start"] - 1 - c0, x_2 + box["col_end"] - c0)
    if not (0 <= rs.start < rs.stop <= r1-r0 and 0 <= cs.start < cs.stop <= c1-c0):
        raise ValueError(f"ROI {box} falls outside saved crop {region}: {rs}, {cs}")
    return rs, cs


def use_custom_boxes(all_rois, custom_boxes):
    """`custom_boxes=None` means: only when the atlas boxes are not being used.

    The four custom boxes overwrite the atlas boxes of the same name, and they are
    not the mirror images of their left twins, so including them in an all-ROI run
    silently breaks bilateral symmetry for M1R/M2R. Hence the default.
    """
    if custom_boxes is None:
        return not all_rois
    return bool(custom_boxes)


def resolve_boxes(atlas, all_rois=False, reference_lambda=None, custom_boxes=None):
    boxes = dict(atlas["boxes"]) if all_rois else {}
    if use_custom_boxes(all_rois, custom_boxes):
        custom = {name: dict(box) for name, box in CUSTOM_BOXES.items()}
        if reference_lambda is not None:
            if not np.isfinite(reference_lambda) or reference_lambda <= 0:
                raise ValueError("reference_lambda must be finite and positive")
            distance = atlas.get("lambda_row_offset")
            if distance is None or distance <= 0:
                raise ValueError("A positive lambda_row_offset is required for scaling")
            from roi_editor import Box, scale_boxes
            scaled = scale_boxes({k: Box(**v) for k, v in custom.items()},
                                 distance / reference_lambda, scale_size=False)
            custom = {name: vars(box) for name, box in scaled.items()}
        boxes.update(custom)
    if not boxes:
        raise ValueError("No ROI boxes selected: enable --all-rois or --custom-boxes")
    return boxes


def roi_categories(labels):
    """Cortical labels put their hemisphere at the end of the anatomical prefix."""
    categories = {}
    for label in labels:
        prefix, separator, suffix = label.partition("_")
        if prefix[-1:] not in ("L", "R"):
            raise ValueError(f"Cannot determine hemisphere for {label!r}")
        categories[label] = (prefix[:-1] + separator + suffix, prefix[-1])
    return categories


class PixelTraceLoader:
    def __init__(self, output_root, *, all_rois=False, reference_lambda=None,
                 custom_boxes=None, signal_mode="median_dff",
                 baseline_median_window_s=20.0, sampling_rate_hz=10.0):
        if signal_mode not in SIGNAL_MODES:
            raise ValueError(f"signal_mode must be one of {SIGNAL_MODES}")
        self.output_root = Path(output_root)
        self.all_rois = all_rois
        self.reference_lambda = reference_lambda
        self.custom_boxes = custom_boxes
        self.signal_mode = signal_mode
        self.baseline_median_window_s = baseline_median_window_s
        self.sampling_rate_hz = sampling_rate_hz
        # Frames per running-median baseline; only used by the median_dff mode.
        self.n_baseline_median = median_window_frames(baseline_median_window_s, sampling_rate_hz)
        self.cached = {}
        self.categories = {}

    @property
    def signal_unit(self):
        return SIGNAL_UNITS[self.signal_mode]

    @property
    def csv_name(self):
        return TRACE_CSV_NAMES[self.signal_mode]

    def roi_trace(self, fluorescence, reflectance):
        """One ROI's [time,row,col] pixel pair -> one (n_time,) trace."""
        if self.signal_mode == "median_dff":
            return median_dff_roi_trace(fluorescence, reflectance, self.n_baseline_median)
        return corrected_roi_trace(fluorescence, reflectance)

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
        atlas_path = Path(meta["roi_set"])
        if not atlas_path.is_absolute():
            atlas_path = REPO_ROOT / atlas_path
        atlas = yaml.safe_load(atlas_path.read_text(encoding="utf-8"))
        if list(atlas["grid"]) != meta["grid"]:
            raise ValueError(f"Atlas and pixel grid differ: {atlas_path}")
        for key in ("bregma_row", "bregma_col", "downsample"):
            if atlas[key] != meta[key]:
                raise ValueError(f"Atlas {key} differs from saved metadata: {atlas_path}")
        boxes = resolve_boxes(atlas, self.all_rois, self.reference_lambda, self.custom_boxes)
        self.categories.update(roi_categories(boxes))
        region = meta.get("region", [0, meta["grid"][0], 0, meta["grid"][1]])
        f = np.load(folder / "pixels_f_gcamp_full.npy", mmap_mode="r")
        r = np.load(folder / "pixels_f_emo_full.npy", mmap_mode="r")
        expected = (meta["n_time"], region[1]-region[0], region[3]-region[2])
        if f.shape != expected or r.shape != expected or list(expected) != meta["shape"]:
            raise ValueError(f"Pixel volume shape disagrees with metadata: {folder}")
        traces = {"trial": np.zeros(expected[0], dtype=int), "frame": np.arange(expected[0])}
        slices = {}
        # Read only one ROI's pixels at a time, in float64; no full volume copy.
        for name, box in boxes.items():
            rs, cs = crop_slices(box, int(meta["y_1"]), int(meta["x_2"]), region)
            traces[name] = self.roi_trace(f[:, rs, cs], r[:, rs, cs])
            slices[name] = [rs.start, rs.stop, cs.start, cs.stop]
        result = pd.DataFrame(traces)
        out = self.output_root / folder.parent.name / folder.name
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / self.csv_name
        result.to_csv(csv_path, index=False)
        geometry = dict(source=str(metadata_path.resolve()), roi_set=str(atlas_path),
                        coordinate_convention="MATLAB 1-based inclusive atlas offsets",
                        region=region, y_1=meta["y_1"], x_2=meta["x_2"],
                        bregma_crop_zero_based=[meta["y_1"]-1-region[0], meta["x_2"]-1-region[2]],
                        lambda_row_offset=atlas.get("lambda_row_offset"),
                        reference_lambda=self.reference_lambda, boxes=boxes,
                        signal_mode=self.signal_mode, signal_unit=self.signal_unit,
                        baseline_median_window_s=(self.baseline_median_window_s
                                                  if self.signal_mode == "median_dff" else None),
                        baseline_median_frames=(self.n_baseline_median
                                                if self.signal_mode == "median_dff" else None),
                        crop_slices_zero_based_exclusive=slices)
        (out / "roi_geometry.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
        self.cached[metadata_path] = csv_path
        return result


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--input-root", type=Path, default=defaults["input_root"],
                        help="Directory containing batch_summary.csv (group assignments)")
    parser.add_argument("--output-dir", type=Path,
                        default=defaults["output_dir"])
    parser.add_argument("--all-rois", action=argparse.BooleanOptionalAction,
                        default=defaults["all_rois"], help="Use every box of each recording's saved atlas")
    parser.add_argument("--custom-boxes", action=argparse.BooleanOptionalAction,
                        default=defaults["custom_boxes"],
                        help="Include the four hand-specified right motor boxes, which replace the "
                             "atlas boxes of the same name; default: only when --all-rois is off")
    parser.add_argument("--signal-mode", choices=SIGNAL_MODES, default=defaults["signal_mode"],
                        help="median_dff: (F/Fbar)/(R/Rbar)-1 with running-median baselines; "
                             "reflectance_ratio: F * mean_t(R) / R")
    parser.add_argument("--baseline-median-window-s", type=float,
                        default=defaults["baseline_median_window_s"],
                        help="Seconds of the per-pixel running median used as Fbar/Rbar (median_dff only)")
    parser.add_argument("--reference-lambda", type=float, default=defaults["reference_lambda"],
                        help="Scale custom box centres to each recording's Lambda; default: fixed pixel offsets")
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
        output_dir=Path(config["output_dir"]),
        all_rois=bool(config["all_rois"]),
        custom_boxes=config["custom_boxes"],
        signal_mode=config["signal_mode"],
        baseline_median_window_s=float(config["baseline_median_window_s"]),
        reference_lambda=config["reference_lambda"],
        debug_plot_count=plot_count(config["debug_plot_count"]),
        debug_animals_per_group=plot_count(config["debug_animals_per_group"]),
        debug_width_scale=float(config["debug_width_scale"]),
        debug_seed=int(config["debug_seed"]),
        recording_limit=config["recording_limit"],
        detection=dict(config["detection"]),
    )


def main():
    args = build_runtime_args()
    # Normalize paths and validate both editor settings and CLI arguments.
    args.pixel_root, args.input_root, args.output_dir = map(
        Path, (args.pixel_root, args.input_root, args.output_dir))
    if (args.debug_plot_count is not None and args.debug_plot_count < 0) or args.debug_seed < 0:
        raise ValueError("debug_plot_count and debug_seed must be nonnegative")
    if not np.isfinite(args.debug_width_scale) or args.debug_width_scale <= 0:
        raise ValueError("debug_width_scale must be finite and positive")
    if args.signal_mode not in SIGNAL_MODES:
        raise ValueError(f"signal_mode must be one of {SIGNAL_MODES}")
    if args.recording_limit is not None and args.recording_limit <= 0:
        raise ValueError("recording_limit must be positive")
    if args.reference_lambda is not None and (not np.isfinite(args.reference_lambda) or args.reference_lambda <= 0):
        raise ValueError("reference_lambda must be finite and positive")
    paths = sorted(args.pixel_root.glob("*/*/pixels_meta_full.npz"))
    if not paths:
        raise FileNotFoundError(f"No full pixel dumps found under {args.pixel_root}")
    loader = PixelTraceLoader(args.output_dir, all_rois=args.all_rois,
                              reference_lambda=args.reference_lambda,
                              custom_boxes=args.custom_boxes, signal_mode=args.signal_mode,
                              baseline_median_window_s=args.baseline_median_window_s,
                              sampling_rate_hz=args.detection["sampling_rate_hz"])
    if args.signal_mode == "median_dff":
        signal_description = (f"(F/Fbar)/(R/Rbar) - 1 per saved pixel, Fbar/Rbar = centred "
                              f"{args.baseline_median_window_s:g} s running median "
                              f"({loader.n_baseline_median} frames); then ROI mean")
    else:
        signal_description = "F * mean_time(R) / R per saved pixel; then spatial ROI mean"
    print(f"Signal: {signal_description} ({loader.signal_unit})")
    run_analysis(input_root=args.input_root, output_dir=args.output_dir,
                 recording_paths=paths, recording_limit=args.recording_limit,
                 trace_loader=loader, diagnostics_root=args.output_dir,
                 diagnostic_plot_count=args.debug_plot_count, diagnostic_seed=args.debug_seed,
                 diagnostic_animals_per_group=args.debug_animals_per_group,
                 diagnostic_width_scale=args.debug_width_scale,
                 roi_metadata=loader.categories, signal_unit=loader.signal_unit,
                 provenance=dict(signal=signal_description, signal_mode=args.signal_mode,
                                 delta_f_over_f=args.signal_mode == "median_dff",
                                 baseline_median_window_s=(args.baseline_median_window_s
                                                           if args.signal_mode == "median_dff" else None),
                                 baseline_median_frames=(loader.n_baseline_median
                                                         if args.signal_mode == "median_dff" else None),
                                 correction_grid="saved downsampled pixels",
                                 pixel_root=str(args.pixel_root.resolve()), all_rois=args.all_rois,
                                 custom_boxes=(CUSTOM_BOXES if use_custom_boxes(
                                     args.all_rois, args.custom_boxes) else None),
                                 reference_lambda=args.reference_lambda),
                 **args.detection)


if __name__ == "__main__":
    main()
