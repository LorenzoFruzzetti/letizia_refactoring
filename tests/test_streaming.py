"""Equivalence test: streaming path vs in-memory path.

The in-memory path (``build_dff_stack`` + ``extract_roi_timeseries``) is already
validated against MATLAB to machine precision (see ``test_parity.py``). Here we
prove the constant-memory streaming path reproduces it, so streaming inherits
that MATLAB parity. Both paths share the same per-frame arithmetic; they differ
only in the summation order of the baseline mean, so we expect agreement to
floating-point roundoff.

Paired multi-page GCaMP/emo TIFFs are synthesized from the sample frames in
``data/`` (which are single-channel), giving the streaming reader real on-disk
multi-page files to consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from wfci import ROIConfig
from wfci.correction import build_dff_stack
from wfci.roi import extract_roi_timeseries, functional_connectivity
from wfci.streaming import run_streaming, stream_trial_roi

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# The sample frames are 512x512; with two 0.5x downsamples the final grid is
# 128x128, so this Bregma + the default boxes stay in bounds. trim=0 keeps all
# 7 sample frames (trim=20 would leave nothing).
Y_1, X_2 = 60, 67
TRIM = 0
DS = 0.5
ATOL = 1e-6
RTOL = 1e-9


def _sample_raw() -> np.ndarray:
    """Load the 7 sample frames as a [y, x, time] float64 stack."""
    files = sorted(DATA_DIR.glob("R11_*.tif"))
    if not files:
        pytest.skip(f"No sample frames in {DATA_DIR}")
    frames = [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in files]
    return np.stack(frames, axis=-1)


def _write_multipage(path: Path, stack_yxt: np.ndarray) -> None:
    """Write a [y, x, time] stack as a multi-page TIFF (pages = time)."""
    tifffile.imwrite(str(path), np.moveaxis(stack_yxt, -1, 0).astype(np.float32))


@pytest.fixture(scope="module")
def trials_on_disk(tmp_path_factory):
    """Two synthetic (gcamp, emo) trials written as paired multi-page TIFFs.

    Same deterministic construction as the MATLAB reference generator so the two
    channels differ, exercising the hemodynamic division.
    """
    raw = _sample_raw()
    tmp = tmp_path_factory.mktemp("stream_tiffs")
    specs = [
        (raw + 50.0, 0.5 * raw + 200.0),   # trial 1: (gcamp, emo)
        (1.1 * raw + 30.0, 0.4 * raw + 150.0),  # trial 2
    ]
    paths = []
    arrays = []
    for k, (g, e) in enumerate(specs):
        gp = tmp / f"t{k}_gcamp.tif"
        ep = tmp / f"t{k}_emo.tif"
        _write_multipage(gp, g)
        _write_multipage(ep, e)
        paths.append((gp, ep))
        # Reload from disk so the in-memory reference uses identical (float32->
        # float64) pixel values as the streaming reader.
        arrays.append(
            (
                np.moveaxis(np.asarray(tifffile.imread(str(gp)), np.float64), 0, -1),
                np.moveaxis(np.asarray(tifffile.imread(str(ep)), np.float64), 0, -1),
            )
        )
    return paths, arrays


@pytest.mark.parametrize(
    "baseline, window",
    [
        (slice(None), slice(None)),   # resting-state style
        (slice(0, 4), slice(1, 6)),   # stimulated style (short windows)
    ],
    ids=["rs", "stim"],
)
def test_streaming_matches_in_memory(trials_on_disk, baseline, window):
    paths, arrays = trials_on_disk
    cfg = ROIConfig(y_1=Y_1, x_2=X_2)

    # In-memory reference (MATLAB-validated path).
    dff = build_dff_stack(arrays, baseline_slice=baseline, trim=TRIM, downsample=DS)
    temp_roi_ref = extract_roi_timeseries(dff, cfg)
    R_ref, R_mean_ref, avg_ref = functional_connectivity(temp_roi_ref, window=window)

    # Streaming path (constant memory, two passes per file).
    res = run_streaming(
        paths, cfg, baseline_slice=baseline, corr_window=window, trim=TRIM, downsample=DS
    )

    md = np.abs(res.temp_roi - temp_roi_ref).max()
    print(f"  max|diff| TEMP_ROI = {md:.3e}")
    np.testing.assert_allclose(res.temp_roi, temp_roi_ref, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(res.R, R_ref, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(res.R_mean, R_mean_ref, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(res.averaged_traces, avg_ref, atol=ATOL, rtol=RTOL)


def test_single_trial_stream_trace(trials_on_disk):
    """stream_trial_roi alone reproduces the in-memory single-trial trace."""
    paths, arrays = trials_on_disk
    cfg = ROIConfig(y_1=Y_1, x_2=X_2)

    dff = build_dff_stack(arrays[:1], baseline_slice=slice(None), trim=TRIM, downsample=DS)
    ref = extract_roi_timeseries(dff, cfg)[:, :, 0]  # [time, 4]

    got = stream_trial_roi(*paths[0], cfg, baseline_slice=slice(None), trim=TRIM, downsample=DS)
    np.testing.assert_allclose(got, ref, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
