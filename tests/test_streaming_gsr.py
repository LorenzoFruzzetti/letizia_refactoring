"""Streaming GSR must equal in-memory GSR -- without a third pass.

MERGING_PLAN.md §2.2 is the load-bearing claim of the whole merge: that global
signal regression, which looks fundamentally anti-streaming (it regresses each
pixel's WHOLE time-series against the global signal), fits the existing two-pass
constant-memory structure via sufficient statistics.

The claim has two halves and each is worth exactly nothing without the other:

  * **Same numbers.** The streaming path reconstructs the ROI traces from
    accumulated statistics rather than from a regressed stack. If the algebra is
    wrong it is wrong *quietly* -- plausible traces, wrong values.
  * **Same cost.** If it accidentally buys correctness with a third pass or a
    retained stack, it has given away the only reason to stream. So the decode
    count is asserted here too, not just in test_efficiency_invariants.py.

The in-memory path is the reference throughout (MERGING_PLAN.md P1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wfci import CORTICAL_GSR, GSRConfig, ROIConfig  # noqa: E402
from wfci.config import Box  # noqa: E402
from wfci.io import tiff_frame_source  # noqa: E402
from wfci.mask import resize_mask  # noqa: E402
from wfci.pipeline import run_pipeline, run_profile  # noqa: E402
from wfci.streaming import run_streaming, run_streaming_profile, stream_trial_roi  # noqa: E402

from tests.test_efficiency_invariants import CountingFrameSource, _count_page_decodes  # noqa: E402

# Small frames: 32 -> 16 -> 8 after the two 0.5x downsamples.
SHAPE = (32, 32)
N_FRAMES = 24
N_TRIALS = 2
TRIM = 0
DS = 0.5
ATOL = 1e-11

SMALL_ATLAS = {"a": Box(1, 2, 1, 2), "b": Box(3, 4, -3, -2), "c": Box(1, 2, -3, -2)}
Y_1, X_2 = 4, 4


def _channels(seed: int, shape=SHAPE, n=N_FRAMES):
    """A (gcamp, emo) pair with a genuine shared global fluctuation.

    GSR must have something real to remove, otherwise the test passes on data
    where every method agrees. A brain-wide breathing-like signal is added to
    gcamp so the global component is large and the regression bites.
    """
    rng = np.random.default_rng(seed)
    common = 1.0 + 0.2 * np.sin(np.linspace(0, 7.0, n))       # global fluctuation
    gain = rng.uniform(0.5, 1.5, shape)
    gcamp = (
        rng.uniform(100.0, 140.0, shape + (n,))
        + 40.0 * gain[:, :, None] * common[None, None, :]
    )
    emo = rng.uniform(600.0, 700.0, shape + (n,))
    return gcamp, emo


def _mask_for(shape) -> np.ndarray:
    """A mask at the ONCE-downsampled resolution, with a background border."""
    half = (shape[0] // 2, shape[1] // 2)
    m = np.zeros(half, dtype=np.float64)
    m[2:-2, 2:-2] = 1.0
    return m


@pytest.fixture(scope="module")
def trials():
    return [_channels(i) for i in range(N_TRIALS)]


@pytest.fixture(scope="module")
def trials_on_disk(trials, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stream_gsr")
    paths = []
    for i, (gcamp, emo) in enumerate(trials):
        gp, ep = tmp / f"t{i}_g.tif", tmp / f"t{i}_e.tif"
        tifffile.imwrite(str(gp), np.moveaxis(gcamp, -1, 0))
        tifffile.imwrite(str(ep), np.moveaxis(emo, -1, 0))
        paths.append((gp, ep))
    return paths


def _cfg() -> ROIConfig:
    return ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(SMALL_ATLAS))


# ---------------------------------------------------------------------------
# The acceptance criterion: same numbers as the in-memory path
# ---------------------------------------------------------------------------
def test_streaming_gsr_matches_in_memory_gsr(trials, trials_on_disk):
    """The sufficient-statistics identity reproduces the regressed ROI traces."""
    cfg = _cfg()
    mask = resize_mask(_mask_for(SHAPE), 0.5)  # onto the final 8x8 grid
    gsr = GSRConfig()

    ref = run_pipeline(trials, cfg, trim=TRIM, downsample=DS, mask=mask, gsr=gsr)
    got = run_streaming(trials_on_disk, cfg, trim=TRIM, downsample=DS, mask=mask, gsr=gsr)

    diff = np.abs(got.temp_roi - ref.temp_roi).max()
    print(f"  max|streaming - in-memory| TEMP_ROI (GSR) = {diff:.3e}")
    assert diff < ATOL, f"streaming GSR diverges from the in-memory reference: {diff:.3e}"
    np.testing.assert_allclose(got.R, ref.R, atol=1e-9)
    np.testing.assert_allclose(got.R_mean, ref.R_mean, atol=1e-9)
    np.testing.assert_allclose(got.averaged_traces, ref.averaged_traces, atol=ATOL)


def test_streaming_gsr_actually_changed_the_traces(trials, trials_on_disk):
    """Guard against a vacuous pass: GSR must move the numbers.

    If GSR were a no-op on this data (or silently skipped), the equivalence test
    above would still pass while proving nothing at all.
    """
    cfg = _cfg()
    mask = resize_mask(_mask_for(SHAPE), 0.5)

    without = run_streaming(trials_on_disk, cfg, trim=TRIM, downsample=DS, mask=mask)
    with_gsr = run_streaming(
        trials_on_disk, cfg, trim=TRIM, downsample=DS, mask=mask, gsr=GSRConfig()
    )

    delta = np.abs(with_gsr.temp_roi - without.temp_roi).max()
    print(f"  max|with GSR - without GSR| = {delta:.3e}")
    assert delta > 1e-3, "GSR made no difference; the test data has no global signal"


def test_streaming_gsr_residuals_are_orthogonal_to_the_global_signal(trials_on_disk):
    """An independent property check: the regressed traces carry no global signal.

    Doesn't consult the in-memory path at all, so a shared misconception between
    the two implementations would still be caught here.
    """
    cfg = _cfg()
    mask = resize_mask(_mask_for(SHAPE), 0.5)

    # Reconstruct the global signal the same way the pipeline does, from the
    # unregressed traces' own source: use the whole-brain ROI as a stand-in.
    whole_brain = ROIConfig(y_1=Y_1, x_2=X_2, boxes={"all": Box(-3, 4, -3, 4)})
    raw = run_streaming(trials_on_disk, whole_brain, trim=TRIM, downsample=DS, mask=mask)
    g = raw.temp_roi[:, 0, 0]

    out = run_streaming(
        trials_on_disk, cfg, trim=TRIM, downsample=DS, mask=mask, gsr=GSRConfig()
    )

    # Each ROI trace is a mean of residuals, so it too must be ~orthogonal to g.
    g_centred = g - g.mean()
    for j in range(out.temp_roi.shape[1]):
        trace = out.temp_roi[:, j, 0]
        r = abs(np.corrcoef(trace, g_centred)[0, 1])
        assert r < 0.2, f"ROI {j} still correlates with the global signal (r={r:.3f})"


# ---------------------------------------------------------------------------
# The other half: it must still be two passes and constant memory
# ---------------------------------------------------------------------------
def test_streaming_gsr_still_makes_exactly_two_passes(trials_on_disk, monkeypatch):
    """GSR must not cost a third read of the file.

    This is the assertion that stops "make it correct" from quietly becoming
    "make it correct by loading the stack".
    """
    cfg = _cfg()
    mask = resize_mask(_mask_for(SHAPE), 0.5)
    gp, ep = trials_on_disk[0]
    gcamp = CountingFrameSource(tiff_frame_source(gp))
    emo = CountingFrameSource(tiff_frame_source(ep))
    decodes = _count_page_decodes(monkeypatch)

    stream_trial_roi(
        gcamp, emo, cfg, trim=TRIM, downsample=DS, mask=mask, gsr=GSRConfig()
    )

    assert gcamp.opens == 2, f"gcamp: {gcamp.opens} passes, expected 2"
    assert emo.opens == 2, f"emo: {emo.opens} passes, expected 2"
    assert decodes["n"] == 4 * N_FRAMES, (
        f"GSR streaming decoded {decodes['n']} pages; expected 2 channels * 2 "
        f"passes * {N_FRAMES} frames = {4 * N_FRAMES}"
    )


def test_streaming_gsr_peak_memory_does_not_grow_with_frame_count(tmp_path):
    """GSR adds two [y, x] images and two [time] vectors -- not a stack."""
    import tracemalloc

    cfg = _cfg()
    mask = resize_mask(_mask_for((64, 64)), 0.5)
    frame_bytes = 64 * 64 * 8
    small, large = 20, 80

    peaks = {}
    for n in (small, large):
        gcamp, emo = _channels(0, shape=(64, 64), n=n)
        gp, ep = tmp_path / f"g{n}.tif", tmp_path / f"e{n}.tif"
        tifffile.imwrite(str(gp), np.moveaxis(gcamp, -1, 0))
        tifffile.imwrite(str(ep), np.moveaxis(emo, -1, 0))

        tracemalloc.start()
        stream_trial_roi(
            tiff_frame_source(gp), tiff_frame_source(ep), cfg,
            trim=TRIM, downsample=DS, mask=mask, gsr=GSRConfig(),
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks[n] = peak

    growth = peaks[large] - peaks[small]
    budget = (2 * (large - small) * frame_bytes) // 4
    print(f"  GSR peak alloc: {small}f={peaks[small]}B  {large}f={peaks[large]}B  "
          f"growth={growth}B (budget {budget}B)")
    assert growth < budget, (
        f"GSR streaming peak grew {growth} B over {large - small} extra frames: it "
        f"is accumulating the recording, not sufficient statistics"
    )


# ---------------------------------------------------------------------------
# The static-NaN precondition (MERGING_PLAN.md §6.1)
# ---------------------------------------------------------------------------
# The zeroed emo pixel below is *meant* to divide by zero -- that is how the
# non-finite frame is created. numpy's warning is expected, not a symptom.
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_a_partially_nan_pixel_is_refused_rather_than_silently_wrong(tmp_path):
    """A pixel non-finite in SOME frames breaks the identity, so streaming refuses.

    Streaming would include such a pixel wherever it happened to be valid, while
    the in-memory path drops it from every frame. Returning a number here would
    mean streaming and in-memory disagree on real data, silently. P1 says the
    in-memory result is the right one -- so this raises and says so.
    """
    n = 12
    gcamp, emo = _channels(0, n=n)
    # Zeroing emo makes that pixel's Delta F/F non-finite for that frame only --
    # exactly the mid-recording NaN the plan warns about. It must cover a whole
    # 2x2 block: the first 0.5x downsample happens BEFORE the correction, so a
    # lone zero would just be averaged away by its three neighbours.
    emo[0:2, 0:2, 5] = 0.0
    gp, ep = tmp_path / "g.tif", tmp_path / "e.tif"
    tifffile.imwrite(str(gp), np.moveaxis(gcamp, -1, 0))
    tifffile.imwrite(str(ep), np.moveaxis(emo, -1, 0))

    with pytest.raises(ValueError) as excinfo:
        stream_trial_roi(
            tiff_frame_source(gp), tiff_frame_source(ep), _cfg(),
            trim=TRIM, downsample=DS, gsr=GSRConfig(),
        )

    message = str(excinfo.value)
    assert "NaN in some frames but not all" in message, "must say what is wrong"
    # An error that only says "no" wastes the reader's time: name the way out.
    assert "no-streaming" in message, "must say how to get a correct result"


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_the_in_memory_path_handles_what_streaming_refuses(trials):
    """The escape hatch has to actually work, or the error message is a lie.

    Streaming tells the caller to re-run with --no-streaming. That advice is only
    honest if the in-memory path really does produce a correct answer for a
    partially-non-finite pixel: it drops the pixel entirely (MATLAB's rule) and
    keeps every other one.
    """
    gcamp, emo = (a.copy() for a in trials[0])
    emo[0:2, 0:2, 5] = 0.0  # -> that pixel's Delta F/F is inf at frame 5 only

    res = run_pipeline([(gcamp, emo)], _cfg(), trim=TRIM, downsample=DS, gsr=GSRConfig())

    assert np.isfinite(res.temp_roi).all(), (
        "the in-memory path leaked a non-finite pixel into the ROI traces"
    )
    # The offending pixel is gone from every frame, not just frame 5.
    assert np.isnan(res.dff_stack[0, 0, :, 0]).all()


def test_mask_only_nans_are_static_and_therefore_fine(trials_on_disk):
    """The precondition holds for the case that actually matters: a brain mask.

    Mask NaNs are per-pixel and constant in time, so the identity is exact and
    the check must not fire.
    """
    mask = resize_mask(_mask_for(SHAPE), 0.5)

    out = run_streaming(
        trials_on_disk, _cfg(), trim=TRIM, downsample=DS, mask=mask, gsr=GSRConfig()
    )

    assert np.isfinite(out.temp_roi).all()


def test_streaming_gsr_without_a_mask_works(trials, trials_on_disk):
    """No mask means no NaN at all -- the precondition is trivially satisfied."""
    cfg = _cfg()

    ref = run_pipeline(trials, cfg, trim=TRIM, downsample=DS, gsr=GSRConfig())
    got = run_streaming(trials_on_disk, cfg, trim=TRIM, downsample=DS, gsr=GSRConfig())

    np.testing.assert_allclose(got.temp_roi, ref.temp_roi, atol=ATOL)


# ---------------------------------------------------------------------------
# The real cortical profile, end to end, both memory strategies
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cortical_data(tmp_path_factory):
    """512x512 raw frames -> 128x128 final, which is what CORTEX_22 needs."""
    tmp = tmp_path_factory.mktemp("cortical")
    arrays, paths = [], []
    for i in range(2):
        gcamp, emo = _channels(10 + i, shape=(512, 512), n=8)
        gp, ep = tmp / f"t{i}_g.tif", tmp / f"t{i}_e.tif"
        tifffile.imwrite(str(gp), np.moveaxis(gcamp, -1, 0))
        tifffile.imwrite(str(ep), np.moveaxis(emo, -1, 0))
        arrays.append((gcamp, emo))
        paths.append((gp, ep))
    # The cortical mask is drawn on the once-downsampled FOV -> 256x256.
    mask = np.zeros((256, 256), dtype=np.float64)
    mask[20:-20, 20:-20] = 1.0
    return arrays, paths, mask


def test_cortical_gsr_profile_runs_both_ways_and_agrees(cortical_data):
    """The whole point of the merge: 22 ROIs + mask + GSR, streaming == in-memory."""
    arrays, paths, mask = cortical_data
    cfg = ROIConfig(y_1=63, x_2=63)

    ref = run_profile(arrays, cfg, CORTICAL_GSR, mask=mask)
    got = run_streaming_profile(paths, cfg, CORTICAL_GSR, mask=mask)

    assert ref.temp_roi.shape == (8, 22, 2), "cortical profile must yield 22 ROIs"
    assert ref.R_mean.shape == (22, 22)

    diff = np.abs(got.temp_roi - ref.temp_roi).max()
    print(f"  max|streaming - in-memory| cortical TEMP_ROI = {diff:.3e}")
    assert diff < ATOL
    np.testing.assert_allclose(got.R_mean, ref.R_mean, atol=1e-9)


def test_cortical_profile_requires_a_mask(cortical_data):
    """use_mask=True is a requirement, not a hint: no silent whole-FOV run."""
    arrays, _, _ = cortical_data

    with pytest.raises(ValueError, match="requires a brain mask"):
        run_profile(arrays, ROIConfig(y_1=63, x_2=63), CORTICAL_GSR)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
