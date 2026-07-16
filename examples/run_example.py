"""Runnable example on the bundled sample data (data/R11_*.tif).

The sample dataset is 7 single-page 512x512 TIFFs = 7 time frames of ONE
channel. The real pipeline needs two channels (GCaMP + emo) per trial, so this
demo synthesises a second channel deterministically just to exercise the full
resting-state pipeline end to end and produce a functional-connectivity matrix
plus a step-2 ROI overlay image.

Run:
    conda run -n letizia python examples/run_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wfci import ROIConfig, load_frame_folder, run_resting_state
from wfci.visualize import overlay_rois

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "examples" / "output"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # One channel of 7 frames from the sample folder.
    raw = load_frame_folder(DATA, pattern="R11_*.tif")  # [512, 512, 7]
    print(f"Loaded sample stack: {raw.shape} (y, x, time)")

    # Synthesise a two-channel, two-trial dataset (demo only; deterministic).
    trials = [
        (raw + 50.0, 0.5 * raw + 200.0),
        (1.1 * raw + 30.0, 0.4 * raw + 150.0),
    ]

    # Bregma chosen so the ROI boxes fit inside the 128x128 downsampled grid.
    cfg = ROIConfig(y_1=60, x_2=67)

    result = run_resting_state(trials, cfg, trim=0)
    print("\nR_mean (4x4 functional connectivity, order "
          "[Laterale_L, Verme_L, Laterale_R, Verme_R]):")
    print(np.array2string(result.R_mean, precision=4, suppress_small=True))

    np.savez(OUT / "example_results.npz",
             R_mean=result.R_mean, temp_roi=result.temp_roi,
             averaged_traces=result.averaged_traces)
    print(f"\nSaved arrays -> {OUT / 'example_results.npz'}")

    # Step-2 style ROI-placement overlay on a representative frame.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frame = result.dff_stack[:, :, 3, 0]
        overlaid = overlay_rois(frame, cfg)
        fig, ax = plt.subplots()
        im = ax.imshow(overlaid, vmin=0.3, vmax=3.0)
        fig.colorbar(im, ax=ax)
        ax.set_title("ROI placement (step 2)")
        fig.savefig(OUT / "roi_overlay.png", dpi=120, bbox_inches="tight")
        print(f"Saved overlay -> {OUT / 'roi_overlay.png'}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(skipped overlay image: {exc})")


if __name__ == "__main__":
    main()
