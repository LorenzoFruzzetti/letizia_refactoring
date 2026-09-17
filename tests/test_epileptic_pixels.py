"""Checks for fluorescence correction, atlas placement, and diagnostic sampling."""

import numpy as np
import pytest

import pandas as pd

from epileptic_by_area_animal_day import select_diagnostic_animals, select_diagnostic_recordings
from epileptic_by_area_animal_day_pixels import (
    CUSTOM_BOXES, corrected_roi_trace, crop_slices, median_dff_roi_trace,
    median_window_frames, plot_count, resolve_boxes, roi_categories, use_custom_boxes,
)
from wfci.config import Box, ROIConfig
from wfci.roi import box_slices_for


def test_reflectance_artifact_removed_without_f_normalization():
    # Two pixels with different brightness and reflectance modulation. The
    # correction must restore their common transient in fluorescence units.
    signal = np.array([10., 10., 30., 10.])[:, None, None]
    brightness = np.array([1., 3.])[None, None, :]
    modulation = np.array([[0.5, 1.5], [1.5, 0.5], [0.5, 1.5], [1.5, 0.5]])[:, None, :]
    f = signal * brightness * modulation
    r = modulation * np.array([100., 200.])[None, None, :]
    np.testing.assert_allclose(corrected_roi_trace(f, r), [20., 20., 60., 20.])
    np.testing.assert_allclose(corrected_roi_trace(2*f, r), [40., 40., 120., 40.])


def test_crop_mapping_matches_pipeline():
    region = (25, 101, 23, 110)
    cfg = ROIConfig(y_1=54, x_2=67, boxes={k: Box(**v) for k, v in CUSTOM_BOXES.items()})
    for name, rs, cs in box_slices_for(cfg, (128, 128)):
        actual_rs, actual_cs = crop_slices(CUSTOM_BOXES[name], 54, 67, region)
        assert (actual_rs.start+25, actual_rs.stop+25) == (rs.start, rs.stop)
        assert (actual_cs.start+23, actual_cs.stop+23) == (cs.start, cs.stop)
        assert actual_rs.stop-actual_rs.start == actual_cs.stop-actual_cs.start == 6
    with pytest.raises(ValueError, match="outside"):
        crop_slices(CUSTOM_BOXES["M2R_alta"], 54, 67, (50, 60, 60, 70))


def test_invalid_reflectance_rejected():
    with pytest.raises(ValueError, match="nonpositive"):
        corrected_roi_trace(np.ones((4, 2, 2)), np.zeros((4, 2, 2)))


def test_sampling_zero_cap_and_reproducibility():
    assert select_diagnostic_recordings(100, 0) == set()
    assert select_diagnostic_recordings(3, 5) == {0, 1, 2}
    selected = select_diagnostic_recordings(100, 5, seed=12)
    assert len(selected) == 5
    assert selected == select_diagnostic_recordings(100, 5, seed=12)
    assert selected != select_diagnostic_recordings(100, 5, seed=13)
    with pytest.raises(ValueError):
        select_diagnostic_recordings(10, -1)


def test_roi_selection_and_reference_scaling():
    atlas = {"boxes": {"V1L": dict(CUSTOM_BOXES["M1R_alta"])}, "lambda_row_offset": 45}
    assert resolve_boxes(atlas) == CUSTOM_BOXES
    # all_rois alone is the atlas as drawn: the custom boxes would replace the
    # atlas's own M1R/M2R with non-mirrored ones, so they have to be asked for.
    assert set(resolve_boxes(atlas, all_rois=True)) == {"V1L"}
    assert set(resolve_boxes(atlas, all_rois=True, custom_boxes=True)) == {*CUSTOM_BOXES, "V1L"}
    assert resolve_boxes(atlas, reference_lambda=45) == CUSTOM_BOXES
    scaled = resolve_boxes(atlas, reference_lambda=90)
    assert scaled["M1R_alta"]["col_start"] < CUSTOM_BOXES["M1R_alta"]["col_start"]
    assert scaled["M1R_alta"]["col_end"] - scaled["M1R_alta"]["col_start"] == 5
    assert roi_categories(CUSTOM_BOXES)["M2R_alta"] == ("M2_alta", "R")


def test_custom_box_default_follows_all_rois():
    assert use_custom_boxes(all_rois=False, custom_boxes=None) is True
    assert use_custom_boxes(all_rois=True, custom_boxes=None) is False
    assert use_custom_boxes(all_rois=True, custom_boxes=True) is True
    assert use_custom_boxes(all_rois=False, custom_boxes=False) is False
    with pytest.raises(ValueError, match="No ROI boxes"):
        resolve_boxes({"boxes": {}}, all_rois=False, custom_boxes=False)


def test_median_baseline_window_is_odd_and_spans_the_requested_seconds():
    assert median_window_frames(20.0, 10.0) == 201
    assert median_window_frames(1.0, 10.0) == 11
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            median_window_frames(bad, 10.0)
        with pytest.raises(ValueError):
            median_window_frames(20.0, bad)


def _drifting_channels(n_time=61):
    """GCaMP and reflectance sharing a slow ramp and a fast reflectance artifact."""
    frames = np.arange(n_time, dtype=float)
    drift = 1.0 + 0.01 * frames  # a slow ramp the running median tracks
    modulation = np.where(frames % 2 == 0, 0.9, 1.1)  # the reflectance artifact
    shared = (drift * modulation)[:, None, None] * np.ones((1, 2, 3))
    return 1000.0 * shared, 500.0 * shared


def test_median_dff_cancels_a_purely_hemodynamic_signal_exactly():
    # When F is a constant multiple of R there is no neural signal at all: the
    # two ratios are identical frame by frame, so the trace must be exactly 0,
    # drift and reflectance artifact included.
    f, r = _drifting_channels()
    trace = median_dff_roi_trace(f, r, n_median=21)
    assert trace.shape == (61,)
    assert np.max(np.abs(trace)) < 1e-14


def test_median_dff_keeps_a_transient_and_is_flat_outside_its_window():
    n_median = 21
    f, r = _drifting_channels()
    f = f.copy()
    f[30] *= 1.5  # one frame, 50% above the local baseline

    trace = median_dff_roi_trace(f, r, n_median)
    assert np.argmax(trace) == 30
    assert trace[30] > 0.4
    # The spike only enters the running median of the windows that contain it,
    # so every frame further than half a window away is still exactly cancelled.
    half = n_median // 2
    untouched = np.r_[trace[:30 - half], trace[30 + half + 1:]]
    assert np.max(np.abs(untouched)) < 1e-14


def test_median_dff_is_invariant_to_channel_gain():
    # Both baselines scale with their channel, so a gain on either camera drops
    # out. This is what separates it from the reflectance_ratio mode, whose
    # output is in fluorescence units and does scale with F.
    f, r = _drifting_channels()
    f = f.copy()
    f[30] *= 1.5
    reference = median_dff_roi_trace(f, r, n_median=21)
    np.testing.assert_allclose(median_dff_roi_trace(7.5 * f, r, 21), reference, atol=1e-14)
    np.testing.assert_allclose(median_dff_roi_trace(f, 0.3 * r, 21), reference, atol=1e-14)


def test_median_dff_rejects_bad_pixels():
    with pytest.raises(ValueError, match="nonpositive"):
        median_dff_roi_trace(np.ones((4, 2, 2)), np.zeros((4, 2, 2)), n_median=3)
    with pytest.raises(ValueError, match="matching nonempty"):
        median_dff_roi_trace(np.ones((4, 2, 2)), np.ones((4, 2, 3)), n_median=3)


def test_plot_count_accepts_all_and_integers():
    assert plot_count("all") is None
    assert plot_count(None) is None
    assert plot_count("0") == 0
    assert plot_count(5) == 5
    with pytest.raises(Exception):
        plot_count("-1")


def test_animal_sampling_is_per_group_capped_and_seeded():
    # Three groups of 6, 2 and 8 animals, several recordings each.
    rows = [(group, f"{group}{i_animal}") for group, n_animals in (("P", 6), ("R", 2), ("T", 8))
            for i_animal in range(n_animals) for _ in range(3)]
    recordings = pd.DataFrame(rows, columns=["group", "animal"])
    chosen = select_diagnostic_animals(recordings, 5, seed=0)
    per_group = pd.Series([name[0] for name in chosen]).value_counts().to_dict()
    assert per_group == {"P": 5, "R": 2, "T": 5}  # R has only 2, so it keeps both
    assert chosen == select_diagnostic_animals(recordings, 5, seed=0)
    assert chosen != select_diagnostic_animals(recordings, 5, seed=1)
    assert select_diagnostic_animals(recordings, None) == sorted(recordings["animal"].unique())
    assert select_diagnostic_animals(recordings, 0) == []
    with pytest.raises(ValueError):
        select_diagnostic_animals(recordings, -1)
