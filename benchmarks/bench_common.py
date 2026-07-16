"""Shared helpers for the modality benchmark.

The four on-disk layouts (``stack`` | ``folder`` | ``stream`` | ``interleaved``)
consume *different* input structures, so to compare them fairly they must all be
fed the SAME underlying pixel data. The single source of truth is the
interleaved split of the sample ``data/`` folder
(:func:`wfci.io.load_interleaved_folder`): it yields one ``(gcamp, emo)`` trial.
We then materialise that identical float64 data into the exact input form each
layout expects, so any output difference reflects the *algorithm* of the layout
and nothing else.

Run parameters mirror the MATLAB parity setup (``tests/test_streaming.py`` /
``gen_reference.m``): resting-state, ``trim=0`` (the sample has only 3 frames per
channel, so the default 20-frame trim would empty the stack), ``downsample=0.5``,
Bregma ``(y_1, x_2) = (60, 67)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from wfci import (
    ROIConfig,
    load_frame_folder,
    load_interleaved_folder,
    load_stack,
    run_resting_state,
    run_streaming_resting_state,
)

# The layouts to compare, in report order.
LAYOUTS = ["stack", "folder", "stream", "interleaved"]

# Fixed pipeline parameters (identical across every layout and MATLAB).
TRIM = 0          # keep all sample frames (trim=20 would leave nothing)
DOWNSAMPLE = 0.5  # box downsample factor
BREGMA_ROW = 121  # -> y_1 = floor(121/2) = 60
BREGMA_COL = 134  # -> x_2 = floor(134/2) = 67


def roi_config() -> ROIConfig:
    """The ROI geometry used by every layout (and mirrored in MATLAB)."""
    return ROIConfig.from_bregma(BREGMA_ROW, BREGMA_COL)


def canonical_trial(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """The one ground-truth ``(gcamp, emo)`` trial: the interleaved split of data/.

    Every layout is fed exactly these float64 pixels, just packaged differently.
    """
    return load_interleaved_folder(str(data_dir))


def _write_multipage(path: Path, stack_yxt: np.ndarray) -> None:
    """Write a ``[y, x, time]`` stack as a float64 multi-page TIFF (pages=time).

    float64 (not float32) so a round-trip through disk is bit-exact with the
    in-memory interleaved data -- otherwise ``stack``/``stream`` would differ from
    ``interleaved`` purely from a dtype cast, muddying the parity comparison.

    ``photometric="minisblack"`` forces one greyscale IFD page per time frame.
    Without it, a 3-frame stack is mistaken for a 3-sample RGB image and written
    as a single page, which the streaming reader (page count = frames) then reads
    as a 1-frame recording.
    """
    tifffile.imwrite(
        str(path),
        np.moveaxis(stack_yxt, -1, 0).astype(np.float64),
        photometric="minisblack",
    )


def _write_folder(folder: Path, stack_yxt: np.ndarray) -> None:
    """Write each ``[y, x]`` frame as a zero-padded single-page float64 TIFF."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(stack_yxt.shape[-1]):
        tifffile.imwrite(str(folder / f"frame_{i:05d}.tif"), stack_yxt[:, :, i])


def prepare_inputs(data_dir: str | Path, prep_dir: str | Path) -> dict[str, object]:
    """Materialise the canonical trial into every layout's input form.

    Returns a dict of the concrete paths each layout's worker will consume.
    ``interleaved`` reuses the original ``data/`` folder unchanged.
    """
    data_dir = Path(data_dir)
    prep_dir = Path(prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    gcamp, emo = canonical_trial(data_dir)

    gcamp_tif = prep_dir / "gcamp.tif"
    emo_tif = prep_dir / "emo.tif"
    _write_multipage(gcamp_tif, gcamp)
    _write_multipage(emo_tif, emo)

    gcamp_folder = prep_dir / "gcamp_folder"
    emo_folder = prep_dir / "emo_folder"
    _write_folder(gcamp_folder, gcamp)
    _write_folder(emo_folder, emo)

    return {
        "data_dir": str(data_dir),
        "gcamp_tif": str(gcamp_tif),
        "emo_tif": str(emo_tif),
        "gcamp_folder": str(gcamp_folder),
        "emo_folder": str(emo_folder),
        "n_frames_per_channel": int(gcamp.shape[-1]),
        "frame_shape": [int(gcamp.shape[0]), int(gcamp.shape[1])],
    }


def run_layout(layout: str, paths: dict[str, object]):
    """Run one layout end-to-end (load + compute) and return its result arrays.

    Returns ``(temp_roi, R, R_mean, averaged_traces)``; ``dff_stack`` is not
    compared (the streaming path never produces one).
    """
    cfg = roi_config()

    if layout == "interleaved":
        gcamp, emo = load_interleaved_folder(str(paths["data_dir"]))
        res = run_resting_state([(gcamp, emo)], cfg, trim=TRIM, downsample=DOWNSAMPLE)
    elif layout == "stack":
        gcamp = load_stack(str(paths["gcamp_tif"]))
        emo = load_stack(str(paths["emo_tif"]))
        res = run_resting_state([(gcamp, emo)], cfg, trim=TRIM, downsample=DOWNSAMPLE)
    elif layout == "folder":
        gcamp = load_frame_folder(str(paths["gcamp_folder"]))
        emo = load_frame_folder(str(paths["emo_folder"]))
        res = run_resting_state([(gcamp, emo)], cfg, trim=TRIM, downsample=DOWNSAMPLE)
    elif layout == "stream":
        trials = [(str(paths["gcamp_tif"]), str(paths["emo_tif"]))]
        res = run_streaming_resting_state(trials, cfg, trim=TRIM, downsample=DOWNSAMPLE)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    return res.temp_roi, res.R, res.R_mean, res.averaged_traces
