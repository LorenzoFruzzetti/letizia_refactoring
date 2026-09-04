from __future__ import annotations

import numpy as np
import pytest

from wfci.roi import _corrcoef_matlab


def test_corrcoef_matlab_matches_numpy_without_blas_dispatch():
    rng = np.random.default_rng(20260812)
    values = rng.normal(size=(2980, 22))

    expected = np.corrcoef(values, rowvar=False)
    actual = _corrcoef_matlab(values)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert actual.dtype == np.float64


def test_corrcoef_matlab_preserves_nan_and_constant_semantics():
    values = np.array([
        [1.0, 4.0, 7.0],
        [2.0, 4.0, np.nan],
        [3.0, 4.0, 9.0],
    ])

    with np.errstate(divide='ignore', invalid='ignore'):
        expected = np.corrcoef(values, rowvar=False)
    actual = _corrcoef_matlab(values)

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_corrcoef_matlab_requires_time_by_region_matrix():
    with pytest.raises(ValueError, match=r'x must be \[time, regions\]'):
        _corrcoef_matlab(np.ones(10))
