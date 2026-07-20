"""ROI geometry that does not fit the data must fail, not fabricate.

ROI boxes are pixel offsets from Bregma. Whether they land on the brain depends
on the Bregma, the field of view and the downsampling -- none of which a box
knows. When they do not fit, NumPy does not complain; it does something worse.

**The bug this file pins.** A negative index counts from the far end of the axis.
So a left-hemisphere ROI whose box runs off the left edge silently averages the
RIGHT side of the image and returns a perfectly ordinary number. Nothing is NaN,
nothing warns, and the resulting correlation matrix looks publishable. MATLAB
raises on a negative index -- the Python port was strictly *more permissive than
its source*, which is a hazard, not a feature.

**The case bounds-checking cannot catch**, and why atlases declare a grid: an
atlas drawn for a different FOV whose boxes all still fit. Every offset is then
wrong by a scale factor while every box lands on real pixels. Only a declared
grid detects that.
"""

from __future__ import annotations

import numpy as np
import pytest

from wfci import CORTEX_22, ROIConfig
from wfci.atlases import Atlas
from wfci.config import Box
from wfci.roi import box_slices_for, extract_roi_timeseries
from wfci.visualize import overlay_rois

GRID = 128


def _stack(grid=GRID, n_time=3, n_trial=1) -> np.ndarray:
    return np.zeros((grid, grid, n_time, n_trial))


# ---------------------------------------------------------------------------
# The wraparound bug
# ---------------------------------------------------------------------------
def test_a_box_running_off_the_left_edge_raises_instead_of_reading_the_right_edge():
    """The headline bug: BFDL is a LEFT ROI; unchecked it read the RIGHT side.

    With Bregma at x_2=20 and BFDL sitting 42 px to its left, the columns resolve
    to -23:-17, which NumPy happily reads as columns 105:111 -- the opposite
    hemisphere. This asserts we raise rather than return that number.
    """
    cfg = ROIConfig(y_1=63, x_2=20, boxes=dict(CORTEX_22))

    with pytest.raises(ValueError, match="fall outside"):
        box_slices_for(cfg, (GRID, GRID))


def test_the_wraparound_value_is_never_returned():
    """End-to-end proof, at the level a user would actually hit it."""
    cfg = ROIConfig(y_1=63, x_2=20, boxes=dict(CORTEX_22))
    stack = _stack()
    # Paint a marker exactly where the negative slice would have read from.
    stack[:, 105:111, :, :] = 999.0

    with pytest.raises(ValueError, match="fall outside"):
        extract_roi_timeseries(stack, cfg)


def test_the_error_names_the_offending_rois_and_their_coordinates():
    """An error that just says "no" makes the user go hunting. Name the box."""
    cfg = ROIConfig(y_1=63, x_2=20, boxes=dict(CORTEX_22))

    with pytest.raises(ValueError) as excinfo:
        box_slices_for(cfg, (GRID, GRID))

    msg = str(excinfo.value)
    assert "BFDL" in msg, "must name the ROI that does not fit"
    assert "-23:-17" in msg, "must show the resolved coordinates"
    assert "y_1=63" in msg and "x_2=20" in msg, "must show the Bregma in use"
    # And it must point at the two things that are actually wrong.
    assert "bregma" in msg.lower()
    assert "field of view" in msg.lower()


def test_a_box_running_off_the_bottom_raises():
    """The other direction: past the end truncates or empties, both silent."""
    cfg = ROIConfig(y_1=120, x_2=63, boxes={"low": Box(41, 46, -2, 3)})  # rows 160:166

    with pytest.raises(ValueError, match="fall outside"):
        box_slices_for(cfg, (GRID, GRID))


def test_every_offending_box_is_reported_not_just_the_first():
    """Fix one, hit the next is a miserable loop. Report them all at once."""
    cfg = ROIConfig(y_1=63, x_2=20, boxes=dict(CORTEX_22))

    with pytest.raises(ValueError) as excinfo:
        box_slices_for(cfg, (GRID, GRID))

    msg = str(excinfo.value)
    # Every left-hemisphere ROI reaching past x_2=20 should be listed.
    for name in ("BFDL", "M1L_alta", "FLL", "V1L"):
        assert name in msg, f"{name} also does not fit but was not reported"


# ---------------------------------------------------------------------------
# The correct setup keeps working (P1)
# ---------------------------------------------------------------------------
def test_the_real_geometry_passes_untouched():
    """CORTEX_22 at its designed Bregma on its designed grid must be fine."""
    cfg = ROIConfig(y_1=63, x_2=63, boxes=dict(CORTEX_22))

    resolved = box_slices_for(cfg, (GRID, GRID), expected_grid=CORTEX_22.grid)

    assert len(resolved) == 22
    assert [name for name, _, _ in resolved] == list(CORTEX_22)
    for name, rs, cs in resolved:
        assert 0 <= rs.start < rs.stop <= GRID, name
        assert 0 <= cs.start < cs.stop <= GRID, name


def test_the_cerebellar_default_still_passes():
    """The MATLAB-validated path must not have acquired a new failure mode."""
    cfg = ROIConfig.from_bregma(121, 134)

    resolved = box_slices_for(cfg, (GRID, GRID), expected_grid=(GRID, GRID))

    assert len(resolved) == 4


# ---------------------------------------------------------------------------
# The declared grid: catching a scaled atlas whose boxes all fit
# ---------------------------------------------------------------------------
def test_an_atlas_on_the_wrong_sized_fov_is_refused_even_when_every_box_fits():
    """The failure bounds-checking cannot see.

    On a 256x256 grid with Bregma at the centre, every CORTEX_22 box lands well
    inside the image -- so bounds-checking is silent. But the offsets were drawn
    for a 128x128 FOV, so every ROI is off by a factor of two and reads the wrong
    anatomy. Only the declared grid catches this.
    """
    cfg = ROIConfig(y_1=128, x_2=128, boxes=dict(CORTEX_22))

    # Bounds alone: no complaint.
    box_slices_for(cfg, (256, 256))

    # With the atlas's declared grid: refused.
    with pytest.raises(ValueError, match="drawn for a 128x128 frame"):
        box_slices_for(cfg, (256, 256), expected_grid=CORTEX_22.grid)


def test_the_grid_error_offers_the_escape_hatch():
    """If the offsets really are right, the user needs a way to say so."""
    cfg = ROIConfig(y_1=128, x_2=128, boxes=dict(CORTEX_22))

    with pytest.raises(ValueError) as excinfo:
        box_slices_for(cfg, (256, 256), expected_grid=CORTEX_22.grid)

    assert "grid=None" in str(excinfo.value)


def test_an_undeclared_grid_skips_the_check():
    """grid=None means "unknown", and unknown must not mean "assume wrong".

    An ad-hoc atlas has no declared FOV; refusing to run it would make the check
    a tax on anyone who did not opt in.
    """
    custom = Atlas(name="ad_hoc", boxes={"a": Box(1, 2, 1, 2)}, grid=None)
    cfg = ROIConfig(y_1=10, x_2=10, boxes=dict(custom))

    assert len(box_slices_for(cfg, (64, 64), expected_grid=custom.grid)) == 1


# ---------------------------------------------------------------------------
# Every entry point is guarded, not just the one
# ---------------------------------------------------------------------------
def test_the_streaming_path_is_guarded_too(tmp_path):
    """Streaming resolves its boxes separately, so it needs its own check.

    It must also fail BEFORE the second pass re-reads the file: an error after a
    full extra read of a multi-GB recording is a poor way to learn the Bregma was
    wrong.
    """
    import tifffile

    from wfci.io import tiff_frame_source
    from wfci.streaming import stream_trial_roi

    rng = np.random.default_rng(0)
    gcamp = rng.uniform(100.0, 200.0, (512, 512, 4))
    emo = rng.uniform(600.0, 700.0, (512, 512, 4))
    gp, ep = tmp_path / "g.tif", tmp_path / "e.tif"
    # photometric explicit: a 4-page stack otherwise looks like RGBA to tifffile.
    tifffile.imwrite(str(gp), np.moveaxis(gcamp, -1, 0), photometric="minisblack")
    tifffile.imwrite(str(ep), np.moveaxis(emo, -1, 0), photometric="minisblack")

    cfg = ROIConfig(y_1=63, x_2=20, boxes=dict(CORTEX_22))  # Bregma too far left

    with pytest.raises(ValueError, match="fall outside"):
        stream_trial_roi(
            tiff_frame_source(gp), tiff_frame_source(ep), cfg, trim=0, downsample=0.5
        )


def test_the_overlay_is_guarded_too():
    """The overlay's whole job is to be trusted as a check on ROI placement.

    One that silently painted the wrong pixels would be worse than none -- it is
    exactly how the cerebellar MATLAB pair drifted without anyone noticing.
    """
    frame = np.zeros((GRID, GRID))
    cfg = ROIConfig(y_1=63, x_2=20, boxes=dict(CORTEX_22))

    with pytest.raises(ValueError, match="fall outside"):
        overlay_rois(frame, cfg)


def test_the_overlay_marks_exactly_the_pixels_the_pipeline_averages():
    """One atlas, drawn and averaged -- the drift the MATLAB suffered is unreachable.

    step2_area_location_RS.m draws Laterale_L three columns from where
    step3_ROI_functional_connectivity.m averages it. Here both read cfg.boxes, so
    the picture cannot disagree with the numbers. This asserts that structurally.
    """
    cfg = ROIConfig.from_bregma(121, 134)
    frame = np.zeros((GRID, GRID))

    painted = overlay_rois(frame, cfg, fill=1.0)

    # Build the same mask from the extraction path's own slices.
    expected = np.zeros((GRID, GRID), dtype=bool)
    for _name, rs, cs in box_slices_for(cfg, (GRID, GRID)):
        expected[rs, cs] = True

    np.testing.assert_array_equal(painted == 1.0, expected)


def test_run_profile_refuses_a_wrong_fov(tmp_path):
    """The check reaches the normal entry point, using the profile's own atlas."""
    from wfci import CORTICAL_GSR
    from wfci.pipeline import run_profile

    rng = np.random.default_rng(1)
    # 256x256 raw -> 64x64 final: the wrong FOV for CORTEX_22's 128x128 offsets.
    trials = [(
        rng.uniform(100.0, 200.0, (256, 256, 4)),
        rng.uniform(600.0, 700.0, (256, 256, 4)),
    )]
    mask = np.ones((128, 128))

    with pytest.raises(ValueError, match="drawn for a 128x128 frame|fall outside"):
        run_profile(trials, ROIConfig(y_1=63, x_2=63), CORTICAL_GSR, mask=mask)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
