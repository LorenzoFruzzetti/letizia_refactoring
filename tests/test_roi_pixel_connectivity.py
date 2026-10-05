"""Checks for the seed-pixel connectivity maps."""

import numpy as np
import pytest

from roi_pixel_connectivity import fisher_mean, pixel_correlation, regress_global_signal


def test_pixel_correlation_matches_corrcoef():
    rng = np.random.default_rng(0)
    seeds = rng.standard_normal((200, 3))
    pixels = rng.standard_normal((200, 10)) + seeds[:, [0]]
    expected = np.corrcoef(seeds.T, pixels.T)[:3, 3:]
    np.testing.assert_allclose(pixel_correlation(seeds, pixels), expected, atol=1e-12)


def test_seed_against_itself_is_the_roi_matrix():
    # A pixel equal to a seed trace must give exactly that seed's R entry.
    rng = np.random.default_rng(1)
    seeds = rng.standard_normal((150, 4))
    roi_matrix = pixel_correlation(seeds, seeds)
    np.testing.assert_allclose(np.diag(roi_matrix), 1.0, atol=1e-12)
    np.testing.assert_allclose(roi_matrix, np.corrcoef(seeds.T), atol=1e-12)


def test_constant_pixel_is_nan_not_an_error():
    rng = np.random.default_rng(2)
    seeds = rng.standard_normal((50, 2))
    pixels = np.column_stack([rng.standard_normal(50), np.ones(50)])
    result = pixel_correlation(seeds, pixels)
    assert np.isfinite(result[:, 0]).all() and np.isnan(result[:, 1]).all()


def test_fisher_mean():
    maps = np.array([[0.2, 0.9], [0.6, 0.9]])
    np.testing.assert_allclose(fisher_mean(maps), np.tanh(np.arctanh(maps).mean(axis=0)))
    # Identical maps average to themselves; r = 1 is clipped, not inf.
    assert np.isfinite(fisher_mean(np.ones((2, 3)))).all()


def test_gsr_matches_per_pixel_least_squares():
    # Same residuals as fitting p = a*g + b pixel by pixel (the MATLAB's fitlm loop),
    # with g the mean of the masked pixels only.
    rng = np.random.default_rng(3)
    shared = rng.standard_normal(300)
    pixels = rng.standard_normal((300, 12)) + rng.uniform(0.5, 2.0, 12) * shared[:, None] + 5.0
    region_mask = np.zeros(12, dtype=bool)
    region_mask[:8] = True
    residuals, global_trace = regress_global_signal(pixels, region_mask, "pixel")
    np.testing.assert_allclose(global_trace, pixels[:, :8].mean(axis=1), atol=1e-12)
    design = np.column_stack([global_trace, np.ones(300)])
    coefficients, *_ = np.linalg.lstsq(design, pixels, rcond=None)
    np.testing.assert_allclose(residuals, pixels - design @ coefficients, atol=1e-10)
    # Every residual is uncorrelated with the global signal.
    np.testing.assert_allclose(pixel_correlation(global_trace[:, None], residuals), 0.0, atol=1e-10)


def test_gsr_column_mean_is_antea_nested_nanmean():
    # Antea's script (2): g = nanmean(nanmean(data,1),2), the mean of the column means.
    # On an irregular region (columns holding 1, 2 or 3 pixels) it is not the pixel mean.
    rng = np.random.default_rng(5)
    n_time, n_rows, n_cols = 200, 3, 4
    pixels = rng.standard_normal((n_time, n_rows * n_cols)) + 3.0
    region = np.array([[1, 1, 0, 1],
                       [0, 1, 0, 1],
                       [0, 1, 1, 1]], dtype=bool)
    residuals, global_trace = regress_global_signal(pixels, region)
    masked = np.where(region[None], pixels.reshape(n_time, n_rows, n_cols), np.nan)
    nested = np.nanmean(np.nanmean(masked, axis=1), axis=1)
    np.testing.assert_allclose(global_trace, nested, atol=1e-12)
    assert np.abs(global_trace - pixels[:, region.reshape(-1)].mean(axis=1)).max() > 1e-3
    np.testing.assert_allclose(pixel_correlation(global_trace[:, None], residuals), 0.0, atol=1e-10)


def test_gsr_column_mean_refuses_a_flat_mask():
    with pytest.raises(ValueError, match="n_rows, n_cols"):
        regress_global_signal(np.ones((10, 4)), np.ones(4, dtype=bool))
