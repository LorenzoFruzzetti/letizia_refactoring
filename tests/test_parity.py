"""Parity test: Python `wfci` package vs MATLAB reference outputs.

Loads ``matlab_reference/reference.mat`` (produced by ``gen_reference.m``),
which contains the exact input trials plus MATLAB's own outputs computed with
native ``imresize('box')``, ``nanmean`` and ``corr``. The Python package is run
on the identical inputs and every output array is asserted close to MATLAB's.

Regenerate the reference with:
    matlab -batch "run('tests/matlab_reference/gen_reference.m')"
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from wfci import ROIConfig
from wfci.config import Box
from wfci.correction import build_dff_stack
from wfci.resize import imresize_box
from wfci.roi import extract_roi_timeseries, functional_connectivity

REF_PATH = Path(__file__).parent / "matlab_reference" / "reference.mat"

# Tight tolerances: identical float64 algorithms should agree to ~1e-9.
ATOL = 1e-7
RTOL = 1e-9

ROI_NAMES = ["Laterale_L", "Verme_L", "Laterale_R", "Verme_R"]


def _mat_to_slice(lo, hi) -> slice:
    """MATLAB 1-based inclusive [lo:hi] -> Python slice; NaN hi means 'full'."""
    start = int(lo) - 1
    stop = None if (isinstance(hi, float) and math.isnan(hi)) else int(hi)
    return slice(start, stop)


@pytest.fixture(scope="module")
def ref():
    if not REF_PATH.exists():
        pytest.skip(f"Reference not found: {REF_PATH}. Run gen_reference.m first.")
    return loadmat(REF_PATH, squeeze_me=True, struct_as_record=False)


def _build_cfg(params) -> ROIConfig:
    y_1 = int(params.y_1)
    x_2 = int(params.x_2)
    rows = np.atleast_2d(params.box_rows)
    cols = np.atleast_2d(params.box_cols)
    boxes = {
        name: Box(int(rows[i, 0]), int(rows[i, 1]), int(cols[i, 0]), int(cols[i, 1]))
        for i, name in enumerate(ROI_NAMES)
    }
    return ROIConfig(y_1=y_1, x_2=x_2, boxes=boxes)


def _trials(ref):
    g = np.asarray(ref["trials_gcamp"], dtype=np.float64)  # [512,512,7,ntrial]
    e = np.asarray(ref["trials_emo"], dtype=np.float64)
    return [(g[:, :, :, i], e[:, :, :, i]) for i in range(g.shape[3])]


def _report(name, py, ml):
    diff = np.abs(np.asarray(py) - np.asarray(ml))
    print(f"  {name:18s} shape={tuple(np.shape(py))}  max|diff|={diff.max():.3e}")
    return diff.max()


# ---------------------------------------------------------------------------
# 1. imresize('box') incl. odd dimensions (general contribution algorithm)
# ---------------------------------------------------------------------------
def test_imresize_box(ref):
    cases = np.atleast_1d(ref["resize_cases"])
    ds = float(ref["params"].ds)
    for c, case in enumerate(cases):
        py = imresize_box(np.asarray(case.__dict__["in"], dtype=np.float64), ds)
        ml = np.asarray(case.out, dtype=np.float64)
        assert py.shape == ml.shape, f"case {c}: shape {py.shape} vs {ml.shape}"
        _report(f"resize_case[{c}]", py, ml)
        np.testing.assert_allclose(py, ml, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# 2. Full pipeline parity for both modes (resting-state + stimulated windows)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["rs", "stim"])
def test_pipeline_parity(ref, mode):
    params = ref["params"]
    cfg = _build_cfg(params)
    trials = _trials(ref)
    res = getattr(ref["results"], mode)

    baseline = _mat_to_slice(res.base_lo, res.base_hi)
    window = _mat_to_slice(res.corr_lo, res.corr_hi)

    dff = build_dff_stack(trials, baseline_slice=baseline, trim=int(params.trim),
                          downsample=float(params.ds))
    temp_roi = extract_roi_timeseries(dff, cfg)
    R, R_mean, averaged = functional_connectivity(temp_roi, window=window)

    print(f"\n[mode={mode}]")
    d1 = _report("dff_stack", dff, res.dff_stack)
    d2 = _report("TEMP_ROI", temp_roi, res.TEMP_ROI)
    d3 = _report("R", R, res.R)
    d4 = _report("R_mean", R_mean, res.R_mean)
    d5 = _report("averaged_traces", averaged, res.averaged_traces)

    np.testing.assert_allclose(dff, res.dff_stack, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(temp_roi, res.TEMP_ROI, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(R, res.R, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(R_mean, res.R_mean, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(averaged, res.averaged_traces, atol=ATOL, rtol=RTOL)
    assert max(d1, d2, d3, d4, d5) < ATOL * 100


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
