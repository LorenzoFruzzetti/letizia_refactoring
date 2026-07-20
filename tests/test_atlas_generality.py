"""The pipeline is sized by the atlas, not by the number 4.

MERGING_PLAN.md §2.1 claims `wfci` is already anatomy-agnostic -- that "generalise
to 22 ROIs" is a config task, not a refactor, because every output is sized from
``len(cfg.boxes)`` and every ``4`` in the code is in a comment. That claim is load
bearing for the whole merge, so it gets a test rather than a paragraph.

Both the in-memory and the streaming path are exercised: they compute ROI means
in different places (whole-stack ``nanmean`` vs per-frame reduction), so each
could independently have a hard-coded width.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from wfci import CEREBELLUM_4, CORTEX_22, ROIConfig
from wfci.config import Box
from wfci.correction import build_dff_stack
from wfci.roi import extract_roi_timeseries, functional_connectivity
from wfci.streaming import run_streaming

# A 128x128 final grid is what the cortical scripts work on (FOV128x128), and
# CORTEX_22's boxes reach +-42 columns / +46 rows from Bregma, so the raw frames
# must be 512x512 for the two 0.5x downsamples to land there.
RAW_SHAPE = (512, 512)
N_FRAMES = 6
N_TRIALS = 2
TRIM = 0
DS = 0.5
# Bregma at the centre of the 128x128 grid, as in the MATLAB (floor(126/2) = 63).
Y_1, X_2 = 63, 63


def _trial(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gcamp = rng.uniform(100.0, 200.0, RAW_SHAPE + (N_FRAMES,))
    emo = rng.uniform(600.0, 700.0, RAW_SHAPE + (N_FRAMES,))
    return gcamp, emo


@pytest.fixture(scope="module")
def trials() -> list[tuple[np.ndarray, np.ndarray]]:
    return [_trial(i) for i in range(N_TRIALS)]


@pytest.fixture(scope="module")
def trials_on_disk(trials, tmp_path_factory) -> list[tuple[Path, Path]]:
    tmp = tmp_path_factory.mktemp("atlas_generality")
    paths = []
    for i, (gcamp, emo) in enumerate(trials):
        gp, ep = tmp / f"t{i}_g.tif", tmp / f"t{i}_e.tif"
        tifffile.imwrite(str(gp), np.moveaxis(gcamp, -1, 0))
        tifffile.imwrite(str(ep), np.moveaxis(emo, -1, 0))
        paths.append((gp, ep))
    return paths


@pytest.mark.parametrize(
    "atlas, n",
    [(CEREBELLUM_4, 4), (CORTEX_22, 22)],
    ids=["cerebellum_4", "cortex_22"],
)
def test_in_memory_path_is_sized_by_the_atlas(trials, atlas, n):
    """[time, n_rois, trial] and an n_rois x n_rois R, for any n."""
    cfg = ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(atlas))
    assert cfg.n_rois == n
    assert cfg.labels == list(atlas)

    dff = build_dff_stack(trials, trim=TRIM, downsample=DS)
    temp_roi = extract_roi_timeseries(dff, cfg)
    R, R_mean, averaged = functional_connectivity(temp_roi)

    assert temp_roi.shape == (N_FRAMES, n, N_TRIALS)
    assert R.shape == (n, n, N_TRIALS)
    assert R_mean.shape == (n, n)
    assert averaged.shape == (N_FRAMES, n)
    assert np.isfinite(temp_roi).all(), "some ROI box fell outside the frame"


@pytest.mark.parametrize(
    "atlas, n",
    [(CEREBELLUM_4, 4), (CORTEX_22, 22)],
    ids=["cerebellum_4", "cortex_22"],
)
def test_streaming_path_is_sized_by_the_atlas(trials_on_disk, atlas, n):
    """The constant-memory path generalises identically -- it reduces per frame."""
    cfg = ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(atlas))

    res = run_streaming(trials_on_disk, cfg, trim=TRIM, downsample=DS)

    assert res.temp_roi.shape == (N_FRAMES, n, N_TRIALS)
    assert res.R.shape == (n, n, N_TRIALS)
    assert res.R_mean.shape == (n, n)
    assert res.averaged_traces.shape == (N_FRAMES, n)
    assert np.isfinite(res.temp_roi).all()


def test_streaming_matches_in_memory_for_22_rois(trials, trials_on_disk):
    """Generality must not cost equivalence: both paths agree at 22 ROIs too.

    The existing streaming test only ever compares 4 ROIs; a width-dependent bug
    in the per-frame reduction would hide there.
    """
    cfg = ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(CORTEX_22))

    dff = build_dff_stack(trials, trim=TRIM, downsample=DS)
    ref = extract_roi_timeseries(dff, cfg)
    got = run_streaming(trials_on_disk, cfg, trim=TRIM, downsample=DS).temp_roi

    print(f"  max|diff| TEMP_ROI (22 ROIs) = {np.abs(got - ref).max():.3e}")
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-9)


def test_column_order_follows_the_atlas_dict_order(trials):
    """Reordering the atlas permutes the columns -- and nothing else.

    This is what makes the dict order the matrix's meaning: the pipeline reads
    the order off ``cfg.boxes`` rather than imposing one, so a study's own layout
    keeps its own order.
    """
    forward = dict(CEREBELLUM_4)
    reversed_atlas = dict(reversed(list(CEREBELLUM_4.items())))

    dff = build_dff_stack(trials, trim=TRIM, downsample=DS)
    a = extract_roi_timeseries(dff, ROIConfig(y_1=Y_1, x_2=X_2, boxes=forward))
    b = extract_roi_timeseries(dff, ROIConfig(y_1=Y_1, x_2=X_2, boxes=reversed_atlas))

    np.testing.assert_array_equal(b, a[:, ::-1, :])


def test_a_study_can_bring_its_own_atlas(trials):
    """No preset required: an arbitrary dict of boxes is a first-class atlas.

    The library provides mechanism; which regions exist is the study's business
    (MERGING_PLAN.md P2). If this ever needed a library change, the boundary has
    leaked.
    """
    custom = {
        "left": Box(0, 5, -20, -15),
        "right": Box(0, 5, 15, 20),
        "midline": Box(10, 15, -2, 3),
    }
    cfg = ROIConfig(y_1=Y_1, x_2=X_2, boxes=custom)

    dff = build_dff_stack(trials, trim=TRIM, downsample=DS)
    temp_roi = extract_roi_timeseries(dff, cfg)
    _, R_mean, _ = functional_connectivity(temp_roi)

    assert temp_roi.shape == (N_FRAMES, 3, N_TRIALS)
    assert R_mean.shape == (3, 3)
    assert cfg.labels == ["left", "right", "midline"]


def test_default_config_is_still_the_cerebellar_atlas():
    """P1: today's default behaviour must not move. An ROIConfig with no boxes
    argument is exactly what it was before the atlas module existed."""
    cfg = ROIConfig(y_1=Y_1, x_2=X_2)

    assert cfg.boxes == CEREBELLUM_4
    assert cfg.labels == ["Laterale_L", "Verme_L", "Laterale_R", "Verme_R"]


def test_default_boxes_are_not_shared_between_configs():
    """Each config owns its dict -- the default_factory must not hand out one
    shared object that a caller could mutate into every other config."""
    a = ROIConfig(y_1=Y_1, x_2=X_2)
    b = ROIConfig(y_1=Y_1, x_2=X_2)

    a.boxes["Verme_L"] = Box(0, 1, 0, 1)

    assert b.boxes["Verme_L"] == CEREBELLUM_4["Verme_L"]
    assert CEREBELLUM_4["Verme_L"] == Box(22, 27, -12, -7), "the preset itself was mutated"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
