"""The brain-mask stage: pixels outside the mask become NaN and stay ignored."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from wfci import ROIConfig
from wfci.config import Box
from wfci.mask import apply_mask, load_mask, resize_mask, valid_from_mask
from wfci.roi import extract_roi_timeseries

SHAPE = (8, 8)
N_TIME = 5
N_TRIAL = 2


def _stack(shape=SHAPE, n_time=N_TIME, n_trial=N_TRIAL) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(1.0, 2.0, shape + (n_time, n_trial))


def _mask(shape=SHAPE) -> np.ndarray:
    """A mask with the left half brain (1) and the right half background (0)."""
    m = np.zeros(shape, dtype=np.float64)
    m[:, : shape[1] // 2] = 1.0
    return m


def test_masked_pixels_are_nan_across_all_time_and_trials():
    """MATLAB's t_TEMP_resized(i,j,:,:) = NaN -- the whole time-series goes."""
    stack = _stack()
    mask = _mask()

    out = apply_mask(stack, mask)

    assert np.isnan(out[:, 4:, :, :]).all(), "background pixels must be fully NaN"
    assert np.isfinite(out[:, :4, :, :]).all(), "brain pixels must be untouched"
    np.testing.assert_array_equal(out[:, :4, :, :], stack[:, :4, :, :])


def test_apply_mask_does_not_modify_its_input():
    stack = _stack()
    before = stack.copy()

    apply_mask(stack, _mask())

    np.testing.assert_array_equal(stack, before)


def test_the_invalid_set_is_static_over_time():
    """Every pixel is valid for all frames or none -- never some.

    This is the precondition the streaming GSR path relies on (MERGING_PLAN.md
    §6.1): under mask-only NaNs the invalid set does not move, so a per-frame
    reduction and a whole-time-series one agree. The mask stage is what
    guarantees it, so it is asserted here at the source.
    """
    out = apply_mask(_stack(), _mask())

    nan_per_pixel = np.isnan(out).any(axis=(2, 3))
    all_nan_per_pixel = np.isnan(out).all(axis=(2, 3))

    np.testing.assert_array_equal(nan_per_pixel, all_nan_per_pixel)


def test_roi_nanmean_ignores_masked_pixels():
    """A box straddling the mask edge averages only its valid pixels."""
    stack = _stack()
    mask = _mask()
    masked = apply_mask(stack, mask)

    # Bregma at (1,1) puts this box at rows 2:4, cols 2:6 -- cols 2,3 are brain
    # and cols 4,5 are background, so the box straddles the boundary.
    cfg = ROIConfig(y_1=1, x_2=1, boxes={"straddle": Box(2, 3, 2, 5)})

    got = extract_roi_timeseries(masked, cfg)

    # The expected value is the plain mean of the VALID half only.
    expected = stack[2:4, 2:4, :, :].mean(axis=(0, 1))
    assert np.isfinite(got).all(), "a partially-masked box must not go NaN"
    np.testing.assert_allclose(got[:, 0, :], expected)


def test_a_fully_masked_box_is_nan_not_an_error():
    """A box entirely outside the brain yields NaN, loudly but without crashing."""
    masked = apply_mask(_stack(), _mask())
    cfg = ROIConfig(y_1=1, x_2=1, boxes={"outside": Box(2, 3, 5, 6)})  # cols 6:8

    with pytest.warns(RuntimeWarning, match="[Mm]ean of empty slice"):
        got = extract_roi_timeseries(masked, cfg)

    assert np.isnan(got).all()


# ---------------------------------------------------------------------------
# The box-resize / threshold behaviour, which is where the MATLAB is subtle
# ---------------------------------------------------------------------------
def test_resize_keeps_partially_covered_boundary_pixels():
    """A box-resized binary mask is fractional, and MATLAB's `== 0` keeps 0.5.

    This is the behaviour that would silently change under a nearest-neighbour
    resize or an eager binarisation: the brain would shrink by a pixel all round
    and every edge ROI would shift.
    """
    # One brain column out of a 2x2 block -> the block averages to 0.5.
    mask = np.zeros((4, 4), dtype=np.float64)
    mask[:, 0] = 1.0

    resized = resize_mask(mask, 0.5)

    assert resized.shape == (2, 2)
    np.testing.assert_allclose(resized[:, 0], 0.5)
    np.testing.assert_allclose(resized[:, 1], 0.0)

    # threshold=0.0 (the MATLAB default) keeps the half-covered pixel...
    np.testing.assert_array_equal(valid_from_mask(resized), [[True, False]] * 2)
    # ...and a stricter threshold drops it, which is the logical-mask behaviour.
    np.testing.assert_array_equal(
        valid_from_mask(resized, threshold=0.5), [[False, False]] * 2
    )


def test_resize_mask_lands_on_the_same_grid_as_the_data():
    """Mask and data must be downsampled by the identical filter."""
    from wfci.resize import imresize_box

    mask = np.random.default_rng(3).uniform(0.0, 1.0, (16, 16))

    np.testing.assert_array_equal(resize_mask(mask, 0.5), imresize_box(mask, 0.5))


# ---------------------------------------------------------------------------
# Failure modes worth failing loudly on
# ---------------------------------------------------------------------------
def test_shape_mismatch_is_rejected_with_a_useful_message():
    """A mask at the wrong resolution must not broadcast into silent nonsense."""
    stack = _stack()
    half_res_mask = _mask((16, 16))

    with pytest.raises(ValueError, match="does not match the data's spatial shape"):
        apply_mask(stack, half_res_mask)


def test_an_all_zero_mask_is_rejected():
    """Almost certainly an inverted mask; NaN-ing the entire brain is not a
    result the caller wants returned silently."""
    with pytest.raises(ValueError, match="excludes every pixel"):
        apply_mask(_stack(), np.zeros(SHAPE))


def test_apply_mask_accepts_a_single_trial_stack():
    """[y, x, time] works as well as [y, x, time, trial] (the streaming shape)."""
    stack = _stack()[:, :, :, 0]

    out = apply_mask(stack, _mask())

    assert out.shape == stack.shape
    assert np.isnan(out[:, 4:, :]).all()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_load_mask_round_trips_a_tiff(tmp_path):
    """Explicit path in, [y, x] float64 out -- no uiopen, no file picker."""
    path = tmp_path / "mask.tif"
    tifffile.imwrite(str(path), (_mask() * 255).astype(np.uint8))

    loaded = load_mask(path)

    assert loaded.shape == SHAPE
    assert loaded.dtype == np.float64
    # Scale is irrelevant: 0/255 masks the same pixels as 0/1.
    np.testing.assert_array_equal(valid_from_mask(loaded), _mask() > 0)


def test_load_mask_rejects_a_multi_frame_tiff(tmp_path):
    """A mask is one image; a stack here means the wrong file was passed."""
    path = tmp_path / "not_a_mask.tif"
    tifffile.imwrite(str(path), np.zeros((5, 8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected a single 2-D image"):
        load_mask(path)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
