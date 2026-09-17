"""Checks for the median dF/F cache and the active-pixel signal."""

import json

import numpy as np
import pytest

from cache_median_dff import build_one, cache_paths, median_dff_volume
from epileptic_by_active_pixels import active_pixel_fraction, hemisphere_masks, pixel_scales
from epileptic_by_area_animal_day_pixels import median_dff_roi_trace


def _volumes(n_time=60, n_row=5, n_col=6, seed=0):
    rng = np.random.default_rng(seed)
    f = 1000 + 50 * rng.standard_normal((n_time, n_row, n_col))
    r = 2000 + 80 * rng.standard_normal((n_time, n_row, n_col))
    return f, r


def test_cached_volume_matches_roi_trace_exactly():
    # Averaging the cached per-pixel volume over a box must reproduce the
    # ROI-mean script bit for bit, or the two signals would silently differ.
    f, r = _volumes()
    f_before, r_before = f.copy(), r.copy()
    volume = median_dff_volume(f, r, 11)
    # The in-place divisions must not reach the caller's float64 arrays.
    np.testing.assert_array_equal(f, f_before)
    np.testing.assert_array_equal(r, r_before)
    box = (slice(1, 4), slice(2, 5))
    expected = median_dff_roi_trace(f[:, box[0], box[1]], r[:, box[0], box[1]], 11)
    np.testing.assert_array_equal(volume[:, box[0], box[1]].mean(axis=(1, 2)), expected)


def test_cached_volume_rejects_nonpositive_reflectance():
    f, r = _volumes()
    r[3, 0, 0] = 0
    with pytest.raises(ValueError, match="nonpositive"):
        median_dff_volume(f, r, 11)


def test_build_one_writes_volume_then_marker(tmp_path):
    f, r = _volumes()
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "pixels_f_gcamp_full.npy", f.astype(np.float32))
    np.save(source / "pixels_f_emo_full.npy", r.astype(np.float32))
    np.savez(source / "pixels_meta_full.npz", n_time=len(f), n_written=len(f), axis_order="time,y,x")
    out = tmp_path / "out"
    build_one(dict(folder=str(source), output_folder=str(out), window_s=1.0, n_median=11,
                   sampling_rate_hz=10.0, dtype="float16"))
    volume_path, meta_path = cache_paths(out, 1.0)
    assert volume_path.name == "pixels_median_dff_1s_full.npy"
    stored = np.load(volume_path)
    assert stored.dtype == np.float16 and stored.shape == f.shape
    reference = median_dff_volume(f.astype(np.float32), r.astype(np.float32), 11)
    meta = json.loads(meta_path.read_text())
    np.testing.assert_allclose(stored, reference, atol=meta["max_abs_storage_error"])
    assert meta["window_frames"] == 11 and not list(out.glob("*.partial.npy"))


def test_active_fraction_uses_each_pixels_own_scale():
    # Pixel 0 is quiet, pixel 1 is ten times noisier. Frame 10 lifts BOTH by the
    # same absolute amount: only the quiet pixel is active, because the threshold
    # is in each pixel's own robust SDs.
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((200, 1, 2)) * np.array([1.0, 10.0])
    noise[10] += 8.0
    fraction, n_flat = active_pixel_fraction(noise, np.ones((1, 2), dtype=bool), 5.0)
    assert fraction[10] == 0.5 and n_flat == 0


def test_bottom_percentile_scale_is_sd_of_the_lowest_values():
    # 10 frames per pixel; the bottom 50% are the 5 smallest, whatever their order.
    values = np.array([9.0, 0.0, 7.0, 1.0, 8.0, 2.0, 6.0, 3.0, 50.0, 4.0])
    pixels = np.stack([values, 2 * values], axis=1)  # (n_time, 2)
    scale = pixel_scales(pixels, np.median(pixels, axis=0), "bottom_percentile", 50.0)
    np.testing.assert_allclose(scale, [np.std([0, 1, 2, 3, 4]), 2 * np.std([0, 1, 2, 3, 4])])
    # The large value at the top stays out of the bottom half; at 100% it inflates the SD.
    assert pixel_scales(pixels, np.median(pixels, axis=0), "bottom_percentile", 100.0)[0] > 10
    with pytest.raises(ValueError, match="percentile"):
        pixel_scales(pixels, np.median(pixels, axis=0), "bottom_percentile", 0.0)
    with pytest.raises(ValueError, match="fewer than 2"):
        pixel_scales(pixels, np.median(pixels, axis=0), "bottom_percentile", 10.0)


def test_active_fraction_with_bottom_percentile_scale():
    # Gaussian noise: the bottom-half SD is ~0.60 sigma, so a threshold of 1.2
    # bottom-half SDs sits near 0.72 sigma and ~24% of frames exceed it.
    noise = np.random.default_rng(3).standard_normal((20000, 1, 1))
    fraction, n_flat = active_pixel_fraction(noise, np.ones((1, 1), dtype=bool), 1.2,
                                             "bottom_percentile", 50.0)
    assert n_flat == 0 and abs(fraction.mean() - 0.236) < 0.01


def test_flat_pixels_are_dropped_from_the_denominator():
    volume = np.zeros((50, 1, 2))
    volume[:, 0, 1] = np.random.default_rng(2).standard_normal(50)
    volume[7, 0, 1] = 100.0
    fraction, n_flat = active_pixel_fraction(volume, np.ones((1, 2), dtype=bool), 3.0)
    assert n_flat == 1 and fraction[7] == 1.0
    with pytest.raises(ValueError, match="flat"):
        active_pixel_fraction(np.zeros((50, 1, 2)), np.ones((1, 2), dtype=bool), 3.0)


def test_hemisphere_masks_follow_box_names():
    boxes = {"M1L_alta": dict(row_start=1, row_end=2, col_start=-3, col_end=-2),
             "M1R_alta": dict(row_start=1, row_end=2, col_start=2, col_end=3),
             "RSL_alta": dict(row_start=4, row_end=4, col_start=-1, col_end=-1)}
    masks = hemisphere_masks(boxes, y_1=3, x_2=5, region=(0, 10, 0, 10), crop_shape=(10, 10))
    assert masks["L"].sum() == 4 + 1 and masks["R"].sum() == 4
    assert masks["L"][6, 3] and not masks["R"][6, 3]  # RSL is left despite its leading R
    overlapping = dict(boxes, M1R_alta=boxes["M1L_alta"])
    with pytest.raises(ValueError, match="overlap"):
        hemisphere_masks(overlapping, y_1=3, x_2=5, region=(0, 10, 0, 10), crop_shape=(10, 10))
