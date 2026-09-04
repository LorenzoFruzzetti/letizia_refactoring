"""Tests for the per-pixel dumps (``--save-data`` / :class:`wfci.dump.PixelDump`).

The dump is an *observer*: it must record what the pipeline computed without
changing it, and without giving up the properties that make the streaming path
worth having. So the tests split into three groups:

  * **inertness** -- attaching a dump leaves ``temp_roi`` byte-identical, and
    leaves the pass count (I10) and the one-frame memory profile (I2) alone;
  * **fidelity** -- the ROI means recomputed from the dumped dF/F reproduce
    ``temp_roi``, and the dumped dF/F is reproducible from the dumped raw F plus
    the dumped baselines. Together these prove the files hold the real
    intermediate rather than a plausible-looking lookalike;
  * **refusal** -- the cases that must raise instead of silently writing wrong
    data: a box outside the crop window, a value that overflows the target dtype,
    and GSR (whose post-regression pixels do not exist in the streaming path).
"""

from __future__ import annotations

import os
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from wfci import ROIConfig  # noqa: E402
from wfci.config import Box  # noqa: E402
from wfci.dump import VOLUME_NAMES, PixelDump  # noqa: E402
from wfci.gsr import GSRConfig  # noqa: E402
from wfci.io import FrameSource, tiff_frame_source  # noqa: E402
from wfci.resize import imresize_box  # noqa: E402
from wfci.streaming import stream_trial_roi  # noqa: E402

# 32 -> 16 -> 8 after the two 0.5x downsamples, matching the invariants suite.
FRAME_SHAPE = (32, 32)
N_FRAMES = 12
TRIM = 2
DS = 0.5
Y_1, X_2 = 2, 2
SMALL_BOXES = {"a": Box(1, 2, 1, 2), "b": Box(3, 4, 3, 4)}


def _cfg() -> ROIConfig:
    return ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(SMALL_BOXES))


def _frames(n: int = N_FRAMES, offset: float = 0.0) -> np.ndarray:
    """Deterministic, non-constant [y, x, time] stack (no zeros: it is a divisor)."""
    rng = np.random.default_rng(0)
    return rng.uniform(100.0, 200.0, FRAME_SHAPE + (n,)) + offset


class CountingFrameSource(FrameSource):
    """Counts ``open()`` calls and frames decoded, as in test_efficiency_invariants."""

    def __init__(self, inner: FrameSource):
        super().__init__(inner.count, inner.open)
        self.opens = 0
        self.decodes = 0

    def open(self):
        self.opens += 1
        return self._counting(self._open())

    def _counting(self, frames):
        for frame in frames:
            self.decodes += 1
            yield frame


@pytest.fixture(scope="module")
def trial(tmp_path_factory):
    """One (gcamp, emo) trial as two multi-page TIFFs, plus the source stacks."""
    tmp = tmp_path_factory.mktemp("dump")
    gcamp, emo = _frames(), _frames(offset=500.0)
    paths = []
    for name, stack in (("gcamp", gcamp), ("emo", emo)):
        p = tmp / f"{name}.tif"
        tifffile.imwrite(str(p), np.moveaxis(stack, -1, 0).astype(np.float64))
        paths.append(p)
    return tuple(paths)


def _dump(tmp_path, **kw) -> PixelDump:
    kw.setdefault("n_time", N_FRAMES - TRIM)
    kw.setdefault("downsample", DS)
    return PixelDump(tmp_path, "test", **kw)


def _load(dump: PixelDump) -> dict[str, np.ndarray]:
    return {name: np.load(dump.path(name)) for name in VOLUME_NAMES}


# ---------------------------------------------------------------------------
# inertness
# ---------------------------------------------------------------------------
def test_dump_does_not_change_the_roi_traces(trial, tmp_path):
    """The dumped run and the plain run must agree bit for bit.

    Not "to within a tolerance": the dump only reads the loop's locals, so any
    difference at all would mean it perturbed the computation.
    """
    g, e = trial
    plain = stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                             trim=TRIM, downsample=DS)
    dumped = stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                              trim=TRIM, downsample=DS, pixel_dump=_dump(tmp_path))

    assert np.array_equal(plain, dumped), (
        "attaching a PixelDump changed temp_roi; the dump must observe, not participate"
    )


def test_dump_keeps_the_two_pass_invariant(trial, tmp_path):
    """I10: dumping must not cost an extra pass over the data.

    Writing from inside pass 2 is the whole design. A dump that re-read the file
    to get its pixels would double the I/O of every full run.
    """
    g, e = trial
    gcamp = CountingFrameSource(tiff_frame_source(g))
    emo = CountingFrameSource(tiff_frame_source(e))

    stream_trial_roi(gcamp, emo, _cfg(), trim=TRIM, downsample=DS,
                     pixel_dump=_dump(tmp_path))

    for name, src in (("gcamp", gcamp), ("emo", emo)):
        assert src.opens == 2, f"{name}: {src.opens} passes with a dump, expected 2"
        assert src.decodes == 2 * N_FRAMES, (
            f"{name}: {src.decodes} decodes, expected 2 passes x {N_FRAMES}"
        )


def _write_trial(tmp_path, n_frames):
    """Two multi-page TIFFs of ``n_frames`` frames each, for the scaling test."""
    paths = []
    for name, offset in (("g", 0.0), ("e", 500.0)):
        p = tmp_path / f"{name}{n_frames}.tif"
        stack = _frames(n_frames, offset)
        tifffile.imwrite(str(p), np.moveaxis(stack, -1, 0).astype(np.float32))
        paths.append(p)
    return paths


def test_dump_writes_through_to_disk_immediately(trial, tmp_path):
    """The volume must be full-size on disk from the first frame, not at close().

    This is the exact form of the constant-memory claim: ``open_memmap``
    preallocates, so the file reaches its final size before any pixel is written
    and each frame lands in it directly. An implementation that accumulated the
    stack and saved at the end would leave a small or absent file here -- and
    would make memory O(frames), which is what streaming exists to avoid.
    """
    g, e = trial

    class CheckingDump(PixelDump):
        checked = False

        def frame(self, idx, *a):
            super().frame(idx, *a)
            if idx == 0:
                expected = int(np.prod(self.shape)) * self.dtypes["dff"].itemsize
                on_disk = os.path.getsize(self.path("dff"))
                # The .npy header adds a small constant; the point is that the
                # PIXELS are already allocated after a single frame.
                assert on_disk >= expected, (
                    f"after 1 frame the file holds {on_disk} bytes, but the full "
                    f"{self.shape} volume needs {expected}; it is not preallocated"
                )
                CheckingDump.checked = True

    stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                     trim=TRIM, downsample=DS,
                     pixel_dump=CheckingDump(tmp_path, "test", n_time=N_FRAMES - TRIM,
                                             downsample=DS))
    assert CheckingDump.checked, "the frame hook never ran"


def test_dump_peak_memory_does_not_grow_with_recording_length(tmp_path):
    """I2: an 8x longer recording must not cost 8x the RAM.

    Absolute byte thresholds are meaningless at test scale (interpreter overhead
    dwarfs a toy volume), so the assertion is on the SHAPE of the curve: peak
    memory is compared between a short and a long run. Buffering would make the
    peak scale with length; streaming keeps it flat.
    """
    peaks = {}
    for n in (50, 400):
        g, e = _write_trial(tmp_path, n)
        dump = PixelDump(tmp_path / f"out{n}", "test", n_time=n - TRIM, downsample=DS)
        tracemalloc.start()
        stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                         trim=TRIM, downsample=DS, pixel_dump=dump)
        _, peaks[n] = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    assert peaks[400] < 2 * peaks[50], (
        f"peak RAM went from {peaks[50]} to {peaks[400]} bytes for an 8x longer "
        f"recording; the dump appears to scale with length instead of staying flat"
    )


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------
def test_dumped_dff_reproduces_temp_roi(trial, tmp_path):
    """Averaging the dumped dF/F over the boxes gives back the ROI traces.

    This is the test that says the dump is the actual intermediate: it re-derives
    the pipeline's own output from the files alone, using the crop origin from the
    sidecar to place the boxes.
    """
    g, e = trial
    region = (Y_1 + 0, Y_1 + 5, X_2 + 0, X_2 + 5)  # contains both boxes
    dump = _dump(tmp_path, region=region)
    temp_roi = stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                                trim=TRIM, downsample=DS, pixel_dump=dump)

    dff = np.load(dump.path("dff"))          # [time, y, x], cropped
    r0, _, c0, _ = np.load(dump.meta_path)["region"]
    for j, box in enumerate(SMALL_BOXES.values()):
        # Box offsets are MATLAB 1-based inclusive from Bregma; shift by the crop.
        rs = slice(Y_1 + box.row_start - 1 - r0, Y_1 + box.row_end - r0)
        cs = slice(X_2 + box.col_start - 1 - c0, X_2 + box.col_end - c0)
        recomputed = dff[:, rs, cs].astype(np.float64).mean(axis=(1, 2))
        # float16 storage is the only source of disagreement here.
        assert np.allclose(recomputed, temp_roi[:, j], atol=1e-2), (
            f"box {j} recomputed from the dump does not match temp_roi"
        )


def test_dumped_raw_f_is_the_pipelines_own_fluorescence(trial, tmp_path):
    """The raw-F volumes must be the exact per-channel fluorescence, not a rescale.

    Compared against an independent computation from the source stack -- trim,
    then the same two 0.5x box downsamples the pipeline applies -- so this pins
    the F volumes to a definite quantity rather than merely "something F-shaped".
    Exact, because float32 represents these twice-block-averaged values exactly.
    """
    g, e = trial
    dump = _dump(tmp_path)
    stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                     trim=TRIM, downsample=DS, pixel_dump=dump)

    vols = _load(dump)
    for name, raw in (("f_gcamp", _frames()), ("f_emo", _frames(offset=500.0))):
        expected = imresize_box(imresize_box(raw[:, :, TRIM:], DS), DS)
        assert np.array_equal(vols[name], np.moveaxis(expected, -1, 0).astype(np.float32)), (
            f"{name} is not the pipeline's twice-downsampled fluorescence"
        )


def test_baselines_are_the_windows_the_correction_used(trial, tmp_path):
    """The sidecar baselines must be MIf/MIr themselves.

    They are what makes the per-channel normalisation recomputable: F/MIf is the
    MATLAB ``If2``. Stored at half resolution, uncropped, in float64 -- the
    resolution the correction consumed them at.
    """
    g, e = trial
    dump = _dump(tmp_path)
    stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                     trim=TRIM, downsample=DS, pixel_dump=dump)

    meta = np.load(dump.meta_path)
    for key, raw in (("mean_f", _frames()), ("mean_r", _frames(offset=500.0))):
        expected = imresize_box(raw[:, :, TRIM:], DS).mean(axis=2)
        assert np.allclose(meta[key], expected), f"{key} is not the baseline image"
        assert meta[key].shape == (16, 16), "baselines are stored at half resolution"


def test_dff_is_not_reconstructible_from_the_downsampled_f(trial, tmp_path):
    """Document the one thing the dumps deliberately do NOT support.

    The pipeline corrects at half resolution and downsamples the RESULT; anything
    built from the dumped (already downsampled) F corrects afterwards. A ratio
    does not commute with a box mean, so the two differ -- slightly on smooth real
    frames, visibly on noise like this fixture. That is exactly why ``dff`` is
    dumped as its own volume instead of being left to be derived, and this test
    exists so nobody "optimises away" that volume on the assumption it is
    redundant.
    """
    g, e = trial
    dump = _dump(tmp_path, dff_dtype=np.float64, f_dtype=np.float64)
    stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                     trim=TRIM, downsample=DS, pixel_dump=dump)

    vols = _load(dump)
    meta = np.load(dump.meta_path)
    naive = ((vols["f_gcamp"] / imresize_box(meta["mean_f"], DS))
             / (vols["f_emo"] / imresize_box(meta["mean_r"], DS)) - 1.0) * 100.0

    assert not np.allclose(naive, vols["dff"], rtol=1e-3, atol=1e-3), (
        "the naive reconstruction now matches dff exactly -- if the pipeline's "
        "resize/correct order changed, update this test and the dump docstring"
    )
    # Close enough to be recognisably the same signal, far enough to matter.
    assert np.corrcoef(naive.ravel(), vols["dff"].ravel())[0, 1] > 0.9


def test_volumes_have_the_declared_shape_and_dtypes(trial, tmp_path):
    """[time, y, x], cropped, in the requested dtypes -- the documented contract."""
    g, e = trial
    region = (1, 7, 2, 8)
    dump = _dump(tmp_path, region=region)
    stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                     trim=TRIM, downsample=DS, pixel_dump=dump)

    vols = _load(dump)
    for name in VOLUME_NAMES:
        assert vols[name].shape == (N_FRAMES - TRIM, 6, 6), f"{name} shape"
        assert np.isfinite(vols[name]).all(), f"{name} has non-finite values"
    assert vols["dff"].dtype == np.float16
    assert vols["f_gcamp"].dtype == np.float32
    assert vols["f_emo"].dtype == np.float32
    assert str(np.load(dump.meta_path)["axis_order"]) == "time,y,x"


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------
def test_region_outside_the_grid_raises(trial, tmp_path):
    """A window that does not fit is an error, never a silent clip."""
    g, e = trial
    dump = _dump(tmp_path, region=(0, 99, 0, 4))
    with pytest.raises(ValueError, match="does not fit"):
        stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                         trim=TRIM, downsample=DS, pixel_dump=dump)


def test_overflowing_dtype_raises_instead_of_writing_inf(tmp_path):
    """The float16 ceiling must be a loud failure, not an inf in the file.

    This is the guard that makes the measured 3.6% headroom on raw F safe to rely
    on: if a brighter recording ever exceeded it, the run stops.
    """
    dump = PixelDump(tmp_path, "test", n_time=1, downsample=DS, f_dtype=np.float16)
    big = np.full((4, 4), 1e6)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        dump.frame(0, big, big, np.zeros((2, 2)))


def test_gsr_with_a_dump_raises(trial, tmp_path):
    """Post-GSR pixels do not exist in the streaming path, so refuse rather than
    dump the pre-GSR ones under a name that implies otherwise."""
    g, e = trial
    with pytest.raises(ValueError, match="not supported together with GSR"):
        stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                         trim=TRIM, downsample=DS, gsr=GSRConfig(),
                         pixel_dump=_dump(tmp_path))


def test_no_dump_writes_no_files(trial, tmp_path):
    """The default path must be exactly what it was before this feature."""
    g, e = trial
    stream_trial_roi(tiff_frame_source(g), tiff_frame_source(e), _cfg(),
                     trim=TRIM, downsample=DS)
    assert list(tmp_path.iterdir()) == []
