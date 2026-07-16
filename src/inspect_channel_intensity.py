"""Inspect two intermingled channels in a TIFF sequence by pixel intensity.

The acquisition interleaves two channels across the image sequence:
    - odd-indexed images  (1, 3, 5, 7, ...) -> channel A
    - even-indexed images (2, 4, 6, ...)     -> channel B

To decide which physical channel is which, we summarise the brightest pixels
of each image (the signal, as opposed to the background). For every TIFF the
script computes the mean of the top 10% brightest pixels and writes one row per
image to a CSV, tagged with its odd/even group.

Run from the editor by editing RUN_CONFIG below, or from the terminal:
    python src/inspect_channel_intensity.py --input-dir data --output-path examples/channel_intensity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

# Edit this section to run the script without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "input_dir": "data",                              # folder holding the TIFF sequence
    "output_path": "examples/channel_intensity.csv",  # CSV to write
    "top_percent": 10.0,                              # summarise the brightest N% of pixels
    "pattern": "*.tif",                               # glob used to collect images (sorted by name)
    "prefer_cli_args": True,                          # CLI flags win over RUN_CONFIG when provided
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=defaults["input_dir"],
                        help="Folder containing the TIFF sequence.")
    parser.add_argument("--output-path", default=defaults["output_path"],
                        help="Destination CSV path.")
    parser.add_argument("--top-percent", type=float, default=defaults["top_percent"],
                        help="Percent of brightest pixels to summarise (default: 10).")
    parser.add_argument("--pattern", default=defaults["pattern"],
                        help="Glob pattern used to collect images (default: *.tif).")
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    prefer_cli_args = bool(config.get("prefer_cli_args", True))

    if prefer_cli_args and len(sys.argv) > 1:
        return parse_args(defaults=config)

    return argparse.Namespace(
        input_dir=config["input_dir"],
        output_path=config["output_path"],
        top_percent=config["top_percent"],
        pattern=config["pattern"],
    )


def summarise_image(image_path: Path, top_percent: float) -> dict[str, Any]:
    """Load one TIFF and summarise its brightest pixels."""
    # tifffile returns the raw pixel array; flatten so multi-page/2D both work.
    pixels = tifffile.imread(image_path).astype(np.float64).ravel()

    # Threshold marking the bottom of the top `top_percent` of intensities.
    percentile_cutoff = 100.0 - top_percent
    threshold = np.percentile(pixels, percentile_cutoff)

    # Mean of the brightest pixels: our proxy for channel signal strength.
    bright_pixels = pixels[pixels >= threshold]
    top_mean = float(bright_pixels.mean())

    return {
        "threshold_p{:g}".format(percentile_cutoff): round(threshold, 3),
        "top{:g}pct_mean".format(top_percent): round(top_mean, 3),
        "top{:g}pct_pixel_count".format(top_percent): int(bright_pixels.size),
        "overall_mean": round(float(pixels.mean()), 3),
        "overall_max": float(pixels.max()),
    }


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by filename so the sequence order matches acquisition order.
    image_paths = sorted(input_dir.glob(args.pattern))
    if not image_paths:
        raise FileNotFoundError(f"No images matching {args.pattern!r} in {input_dir}")

    rows: list[dict[str, Any]] = []
    for seq_index, image_path in enumerate(image_paths, start=1):
        stats = summarise_image(image_path, args.top_percent)
        # Odd/even refers to 1-based position in the sequence (the channel split).
        group = "odd" if seq_index % 2 == 1 else "even"
        row = {"sequence_index": seq_index, "filename": image_path.name, "group": group}
        row.update(stats)
        rows.append(row)
        print(f"[{seq_index:>3}] {image_path.name:<20} group={group:<4} "
              f"top{args.top_percent:g}%_mean={row[f'top{args.top_percent:g}pct_mean']}")

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main(build_runtime_args())
