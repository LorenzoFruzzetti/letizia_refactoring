"""Filmstrip montages of widefield dF/F frames, one labelled panel per condition.

Each panel tiles consecutive (optionally binned) frames of ONE recording into
`tile_rows` rows, like the BL / 3D / 15D / 30D figures in papers. All panels
share one colour scale, so colours are comparable across conditions.

Input: any `(time, y, x)` volume in `pixel_data/<day>_<animal>/<t#>/`, by default
the 20 s running-median dF/F cache (`pixels_median_dff_20s_full.npy`, see
CLAUDE.md 9.22/9.23). The crop is Bregma-relative (9.15), so the same tile
position shows the same anatomy in every animal: rows run anterior (top) to
posterior (bottom).

Where each strip starts is set per panel:
- `"start_s": 120.0`  -> explicit start time in seconds;
- `"start_s": "peak"` -> centred on the frame with the largest field-mean dF/F
  (after temporal binning), with `pre_peak_s` of the strip before it.

Output: `<output_dir>/<output_name>.png` and `<output_name>.json` (which frames
went into every tile, colour limits, all parameters).

Edit RUN_CONFIG below to run from the editor; the CLI overrides it:
    python plot_frame_montage.py --panel BL=260709_T9/t1@peak --panel 3D=260716_T9/t1@120
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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the script without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",
    "volume_name": "pixels_median_dff_20s_full.npy",  # or pixels_dff_full.npy (mean baseline)
    # One entry per condition: label drawn in the corner, recording folder relative
    # to pixel_root, and start time in seconds or "peak".
    "panels": [
        {"label": "BL", "recording": "260709_T9/t1", "start_s": "peak"},
        {"label": "3D", "recording": "260716_T9/t1", "start_s": "peak"},
        {"label": "15D", "recording": "260723_T9/t1", "start_s": "peak"},
        {"label": "30D", "recording": "260807_T9/t1", "start_s": "peak"},
    ],
    "n_tiles": 20,  # frames shown per panel
    "tile_rows": 2,  # rows of tiles inside each panel
    "bin_frames": 1,  # consecutive frames averaged into one tile (1 = raw frames)
    "step_frames": 1,  # frames between the starts of consecutive tiles (>= bin_frames for no overlap)
    "pre_peak_s": 0.5,  # with start_s="peak": strip time shown before the peak
    "peak_exclude_s": 0.0,  # with start_s="peak": ignore this much at both recording ends
    "spatial_sigma_px": 0.0,  # Gaussian blur per tile; 0 = none
    "color_limits": None,  # (vmin, vmax) in volume units; None = shared percentiles below
    "color_percentiles": (1.0, 99.5),  # used when color_limits is None
    "value_scale": 100.0,  # 100 shows the dF/F ratio as %
    "value_label": "dF/F (%)",
    "cmap": "jet",
    "panel_columns": 2,  # panels per figure row
    "tile_gap_px": 1,  # white gap between tiles, in pixels
    "show_time_labels": False,  # small "+0.3 s" in each tile, relative to the strip start/peak
    "output_dir": REPO_ROOT / "outputs/frame_montages",
    "output_name": "T9_t1_peak",
    "dpi": 300,
    "prefer_cli_args": True,  # False: always use this block, ignoring CLI arguments.
}


def parse_panel(text: str) -> dict[str, Any]:
    """`LABEL=DAY_ANIMAL/t#@START` -> panel dict; START is seconds or 'peak'."""
    label, rest = text.split("=", 1)
    recording, start = rest.rsplit("@", 1)
    return {"label": label, "recording": recording,
            "start_s": "peak" if start == "peak" else float(start)}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--volume-name", default=defaults["volume_name"])
    parser.add_argument("--panel", dest="panels", action="append", type=parse_panel,
                        help="LABEL=DAY_ANIMAL/t#@START_S or @peak; repeat per panel (replaces RUN_CONFIG panels)")
    parser.add_argument("--n-tiles", type=int, default=defaults["n_tiles"])
    parser.add_argument("--tile-rows", type=int, default=defaults["tile_rows"])
    parser.add_argument("--bin-frames", type=int, default=defaults["bin_frames"])
    parser.add_argument("--step-frames", type=int, default=defaults["step_frames"])
    parser.add_argument("--pre-peak-s", type=float, default=defaults["pre_peak_s"])
    parser.add_argument("--peak-exclude-s", type=float, default=defaults["peak_exclude_s"])
    parser.add_argument("--spatial-sigma-px", type=float, default=defaults["spatial_sigma_px"])
    parser.add_argument("--color-limits", type=float, nargs=2, default=defaults["color_limits"],
                        metavar=("VMIN", "VMAX"), help="in volume units (before value_scale)")
    parser.add_argument("--color-percentiles", type=float, nargs=2, default=defaults["color_percentiles"])
    parser.add_argument("--value-scale", type=float, default=defaults["value_scale"])
    parser.add_argument("--value-label", default=defaults["value_label"])
    parser.add_argument("--cmap", default=defaults["cmap"])
    parser.add_argument("--panel-columns", type=int, default=defaults["panel_columns"])
    parser.add_argument("--tile-gap-px", type=int, default=defaults["tile_gap_px"])
    parser.add_argument("--show-time-labels", action=argparse.BooleanOptionalAction,
                        default=defaults["show_time_labels"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--output-name", default=defaults["output_name"])
    parser.add_argument("--dpi", type=int, default=defaults["dpi"])
    args = parser.parse_args()
    if args.panels is None:
        args.panels = defaults["panels"]
    return args


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})


def sampling_rate_hz(recording_dir: Path) -> float:
    """Per-channel rate saved next to the median-dF/F caches; 10 Hz otherwise."""
    meta_files = sorted(recording_dir.glob("pixels_median_dff_*_meta.json"))
    if not meta_files:
        return 10.0  # every dump in this project is 10 Hz per channel (CLAUDE.md 9.22)
    return float(json.loads(meta_files[0].read_text())["sampling_rate_hz"])


def binned_frames(volume: np.ndarray, first_frame: int, n_tiles: int, bin_frames: int,
                  step_frames: int) -> np.ndarray:
    """Average `bin_frames` frames per tile. Returns (n_tiles, y, x) float64."""
    tiles = []
    for i_tile in range(n_tiles):
        start = first_frame + i_tile * step_frames
        # float64 before reducing: float16 sums are unsafe (CLAUDE.md 9.16)
        tiles.append(np.asarray(volume[start:start + bin_frames], dtype=np.float64).mean(axis=0))
    return np.stack(tiles)


def field_mean_trace(volume: np.ndarray) -> np.ndarray:
    """Mean over every finite pixel of each frame -> (n_time,) float64."""
    n_time = volume.shape[0]
    field_mean = np.empty(n_time)
    chunk = 500  # frames per read, keeps the float64 copy small
    for start in range(0, n_time, chunk):
        field_mean[start:start + chunk] = np.nanmean(
            np.asarray(volume[start:start + chunk], dtype=np.float64), axis=(1, 2))
    return field_mean


def find_peak_frame(volume: np.ndarray, bin_frames: int, exclude_frames: int) -> int:
    """Frame index of the largest field-mean value, after a bin_frames moving average."""
    n_time = volume.shape[0]
    smoothed = np.convolve(field_mean_trace(volume), np.ones(bin_frames) / bin_frames, mode="same")
    smoothed[:exclude_frames] = -np.inf
    smoothed[n_time - exclude_frames:] = -np.inf
    return int(np.argmax(smoothed))


def ranked_peak_frames(volume: np.ndarray, n_peaks: int, bin_frames: int, exclude_frames: int,
                       min_separation_frames: int, detrend_window_frames: int) -> list[int]:
    """The n_peaks largest field-mean peaks, largest first, at least min_separation_frames apart.

    The field mean is first detrended by its centred running median over
    detrend_window_frames: the mean-baseline dF/F declines through a recording
    (CLAUDE.md 9.30), so without this the largest values would all fall in its
    first seconds. Then a bin_frames moving average, the ends excluded, and greedy
    selection: take the maximum, blank +-min_separation_frames around it, repeat.
    Returns fewer than n_peaks frames only if the recording has no room for more.
    """
    field_mean = field_mean_trace(volume)
    detrended = field_mean - median_filter(field_mean, size=detrend_window_frames, mode="nearest")
    smoothed = np.convolve(detrended, np.ones(bin_frames) / bin_frames, mode="same")
    n_time = smoothed.size
    smoothed[:exclude_frames] = -np.inf
    smoothed[n_time - exclude_frames:] = -np.inf
    peaks = []
    while len(peaks) < n_peaks and np.isfinite(smoothed).any():
        peak = int(np.argmax(smoothed))
        peaks.append(peak)
        smoothed[max(peak - min_separation_frames, 0):peak + min_separation_frames + 1] = -np.inf
    return peaks


def tile_montage(tiles: np.ndarray, tile_rows: int, gap_px: int) -> np.ndarray:
    """(n_tiles, y, x) -> one 2-D image, row-major, NaN gaps (drawn white)."""
    n_tiles, height, width = tiles.shape
    tile_cols = int(np.ceil(n_tiles / tile_rows))
    montage = np.full((tile_rows * height + (tile_rows - 1) * gap_px,
                       tile_cols * width + (tile_cols - 1) * gap_px), np.nan)
    for i_tile in range(n_tiles):
        i_row, i_col = divmod(i_tile, tile_cols)
        row_0 = i_row * (height + gap_px)
        col_0 = i_col * (width + gap_px)
        montage[row_0:row_0 + height, col_0:col_0 + width] = tiles[i_tile]
    return montage


def main():
    args = build_runtime_args()
    if args.n_tiles < 1 or args.tile_rows < 1 or args.bin_frames < 1 or args.step_frames < 1:
        raise ValueError("n_tiles, tile_rows, bin_frames and step_frames must be >= 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load each recording lazily and cut out its strip of tiles.
    panel_tiles = []  # one (n_tiles, y, x) array per panel
    provenance = []
    strip_frames = (args.n_tiles - 1) * args.step_frames + args.bin_frames  # frames one strip spans
    for panel in args.panels:
        recording_dir = Path(args.pixel_root) / panel["recording"]
        volume = np.load(recording_dir / args.volume_name, mmap_mode="r")  # (time, y, x)
        fs = sampling_rate_hz(recording_dir)
        n_time = volume.shape[0]
        peak_frame = None
        if panel["start_s"] == "peak":
            peak_frame = find_peak_frame(volume, args.bin_frames, int(round(args.peak_exclude_s * fs)))
            # Centre the peak's own bin, then step back pre_peak_s.
            first_frame = peak_frame - args.bin_frames // 2 - int(round(args.pre_peak_s * fs))
        else:
            first_frame = int(round(float(panel["start_s"]) * fs))
        # Keep the strip inside the recording rather than failing near its ends.
        first_frame = min(max(first_frame, 0), n_time - strip_frames)
        if first_frame < 0:
            raise ValueError(f"{panel['recording']}: strip needs {strip_frames} frames, recording has {n_time}")
        tiles = binned_frames(volume, first_frame, args.n_tiles, args.bin_frames, args.step_frames)
        if args.spatial_sigma_px > 0:
            tiles = np.stack([gaussian_filter(tile, args.spatial_sigma_px) for tile in tiles])
        panel_tiles.append(tiles * args.value_scale)
        tile_start_frames = first_frame + np.arange(args.n_tiles) * args.step_frames
        provenance.append({**panel, "file": str(recording_dir / args.volume_name), "sampling_rate_hz": fs,
                           "peak_frame": peak_frame, "first_frame": first_frame,
                           "tile_start_frames": tile_start_frames.tolist()})
        print(f"{panel['label']}: {panel['recording']} frames {first_frame}-{first_frame + strip_frames - 1}"
              f" ({first_frame / fs:.1f}-{(first_frame + strip_frames) / fs:.1f} s)")

    tile_duration_s = args.bin_frames / provenance[0]["sampling_rate_hz"]
    title = (f"{args.volume_name}: {args.n_tiles} tiles x {tile_duration_s:g} s, "
             f"step {args.step_frames} frames, blur {args.spatial_sigma_px:g} px")
    image_path = output_dir / f"{args.output_name}.png"
    vmin, vmax = draw_montage_figure(panel_tiles, provenance, args, title, image_path)

    # Record exactly what was drawn, so a figure can be regenerated or audited.
    settings = {key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items() if key != "panels"}
    (output_dir / f"{args.output_name}.json").write_text(json.dumps(
        {"settings": settings, "color_limits_displayed": [float(vmin), float(vmax)], "panels": provenance},
        indent=2))
    print(f"Saved {image_path}")


def draw_montage_figure(panel_tiles: list[np.ndarray], provenance: list[dict[str, Any]],
                        args: argparse.Namespace, title: str, image_path: Path) -> tuple[float, float]:
    """Draw one figure: a filmstrip panel per entry, one shared colour scale. Returns (vmin, vmax).

    panel_tiles: one (n_tiles, y, x) array per panel, already scaled by value_scale.
    provenance: per panel, at least label, recording, first_frame, sampling_rate_hz,
    peak_frame and tile_start_frames. Uses args.color_limits / color_percentiles /
    value_scale / value_label / cmap / panel_columns / tile_rows / tile_gap_px /
    n_tiles / bin_frames / show_time_labels / dpi.
    """
    # One colour scale for every panel.
    if args.color_limits is None:
        all_values = np.concatenate([tiles.ravel() for tiles in panel_tiles])
        vmin, vmax = np.nanpercentile(all_values, args.color_percentiles)
    else:
        vmin, vmax = np.asarray(args.color_limits, dtype=float) * args.value_scale

    # Panels laid out on a grid; each panel is one image of tile_rows x tile_cols tiles.
    n_panels = len(panel_tiles)
    panel_rows = int(np.ceil(n_panels / args.panel_columns))
    tile_height, tile_width = panel_tiles[0].shape[1:]
    tile_cols = int(np.ceil(args.n_tiles / args.tile_rows))
    panel_aspect = (args.tile_rows * tile_height) / (tile_cols * tile_width)
    panel_width_in = 5.0
    fig, axes = plt.subplots(panel_rows, args.panel_columns, squeeze=False, constrained_layout=True,
                             figsize=(panel_width_in * args.panel_columns + 0.8,
                                      panel_width_in * panel_aspect * panel_rows + 0.6))
    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad("white")  # the tile gaps
    image = None
    for i_panel, (tiles, info) in enumerate(zip(panel_tiles, provenance)):
        ax = axes.flat[i_panel]
        montage = tile_montage(tiles, args.tile_rows, args.tile_gap_px)
        image = ax.imshow(montage, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        label = ax.text(0.99, 0.03, info["label"], transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=14, fontweight="bold", color="white")
        label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])
        if args.show_time_labels:
            # Time of each tile relative to the peak (or to the strip start).
            zero_frame = info["peak_frame"] if info["peak_frame"] is not None else info["first_frame"]
            for i_tile, start in enumerate(info["tile_start_frames"]):
                i_row, i_col = divmod(i_tile, tile_cols)
                centre_frame = start + (args.bin_frames - 1) / 2
                ax.text(i_col * (tile_width + args.tile_gap_px) + 1, i_row * (tile_height + args.tile_gap_px) + 1,
                        f"{(centre_frame - zero_frame) / info['sampling_rate_hz']:+.1f}s",
                        ha="left", va="top", fontsize=4, color="white")
        ax.set_title(f"{info['recording']}  ({info['first_frame'] / info['sampling_rate_hz']:.1f} s)",
                     fontsize=7, loc="left")
        ax.set_axis_off()
    for ax in axes.flat[n_panels:]:
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=0.6, fraction=0.015, pad=0.01, label=args.value_label)
    fig.suptitle(title, fontsize=8)
    fig.savefig(image_path, dpi=args.dpi)
    plt.close(fig)
    return float(vmin), float(vmax)


if __name__ == "__main__":
    main()
