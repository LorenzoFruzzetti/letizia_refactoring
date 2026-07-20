"""Regression guard for the import-efficiency invariants I1-I11.

These invariants are the *reason the streaming path exists*, and every one of
them is invisible to a correctness test: a refactor that turned
``FrameSource.open()`` into a cached list, or made ``tiff_frame_count`` decode
pages, would leave every other test in this suite green -- just slower and
fatter. Nothing would fail until someone pointed the pipeline at a recording
too large for RAM, which is exactly the case the design is for.

So this file asserts the *mechanism*, not the numbers:

  I1  tiff_frame_count decodes ZERO pixels (IFD headers only)
  I2  frames are decoded lazily, one at a time
  I3  FrameSource.open() returns a FRESH iterator every call (two passes need it)
  I4  storage layout x memory strategy stay orthogonal (any source streams)
  I5  interleaved_channel_files decodes exactly TWO images to split a folder
  I6  the same split feeds both the full-load and the streaming path
  I7  a bare path is still coerced to a FrameSource (back-compat)
  I8  --debug-max-frames k reads 2k frames off disk, not 2N
  I9  a debug full-load run routes through the lazy sources too
  I10 a streaming run makes exactly TWO passes: 2N decodes, 2 opens per channel
  I11 per-frame resize == whole-stack resize (why streaming can match exactly)

The counting is done by wrapping a real FrameSource (I2/I3/I8/I10) or by
counting real ``tifffile`` calls (I1/I5), so these test the shipped code paths
rather than a mock of them.
"""

from __future__ import annotations

import itertools
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
# run_pipeline.py is a top-level script, not part of the installed package, so
# the repo root has to be importable to test its debug-mode helpers (I8/I9).
sys.path.insert(0, str(REPO_ROOT))

from wfci import ROIConfig  # noqa: E402
from wfci.config import Box  # noqa: E402
from wfci.io import (  # noqa: E402
    FrameSource,
    folder_frame_source,
    frame_folder_source,
    interleaved_channel_files,
    tiff_frame_count,
    tiff_frame_source,
)
from wfci.resize import imresize_box  # noqa: E402
from wfci.streaming import _as_frame_source, run_streaming, stream_trial_roi  # noqa: E402

from run_pipeline import _build_streaming_sources, _limited_source, _load_trials  # noqa: E402

# Small synthetic frames keep the test fast; the invariants are about call
# counts, not pixel values. 32 -> 16 -> 8 after the two 0.5x downsamples.
FRAME_SHAPE = (32, 32)
N_FRAMES = 10
TRIM = 0
DS = 0.5

# A Bregma + boxes that land inside the final 8x8 grid.
Y_1, X_2 = 2, 2
SMALL_BOXES = {"a": Box(1, 2, 1, 2), "b": Box(3, 4, 3, 4)}


def _cfg() -> ROIConfig:
    return ROIConfig(y_1=Y_1, x_2=X_2, boxes=dict(SMALL_BOXES))


def _frames(n: int = N_FRAMES, offset: float = 0.0) -> np.ndarray:
    """Deterministic, non-constant [y, x, time] stack (no zeros: it is a divisor)."""
    rng = np.random.default_rng(0)
    return rng.uniform(100.0, 200.0, FRAME_SHAPE + (n,)) + offset


class CountingFrameSource(FrameSource):
    """A FrameSource that counts ``open()`` calls and frames actually decoded.

    Subclasses FrameSource rather than mimicking it so that ``_as_frame_source``'s
    isinstance check passes and the streaming path treats it as the real thing.
    """

    def __init__(self, inner: FrameSource):
        # ``inner.open`` (not inner._open) so an already-wrapped source still counts.
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
def multipage_trial(tmp_path_factory):
    """One (gcamp, emo) trial as two multi-page TIFFs on disk."""
    tmp = tmp_path_factory.mktemp("invariants")
    gcamp = _frames()
    emo = _frames(offset=500.0)
    paths = []
    for name, stack in (("gcamp", gcamp), ("emo", emo)):
        p = tmp / f"{name}.tif"
        tifffile.imwrite(str(p), np.moveaxis(stack, -1, 0).astype(np.float32))
        paths.append(p)
    return tuple(paths)


@pytest.fixture(scope="module")
def interleaved_folder(tmp_path_factory):
    """A folder of single-page TIFFs with the two channels alternating.

    Odd-positioned images are the brighter (GCaMP) group, so the split's
    intensity decision has a definite right answer.
    """
    tmp = tmp_path_factory.mktemp("interleaved")
    bright = _frames(offset=1000.0)
    dim = _frames()
    for i in range(N_FRAMES):
        # 1-based positions: odd = bright/gcamp, even = dim/emo.
        tifffile.imwrite(str(tmp / f"img_{2 * i:05d}.tif"), bright[:, :, i].astype(np.float32))
        tifffile.imwrite(str(tmp / f"img_{2 * i + 1:05d}.tif"), dim[:, :, i].astype(np.float32))
    return tmp


# ---------------------------------------------------------------------------
# I1 -- frame count without decoding pixels
# ---------------------------------------------------------------------------
def test_i1_frame_count_decodes_nothing(multipage_trial, monkeypatch):
    """tiff_frame_count parses IFD headers only -- zero pages decoded.

    This is what lets the baseline window be sized before a single frame is read,
    which stays cheap on a multi-GB file.
    """
    gcamp_path, _ = multipage_trial
    decodes = _count_page_decodes(monkeypatch)

    n = tiff_frame_count(gcamp_path)

    assert n == N_FRAMES
    assert decodes["n"] == 0, (
        f"tiff_frame_count decoded {decodes['n']} pages; it must read IFD headers only"
    )


def _count_page_decodes(monkeypatch) -> dict:
    """Patch tifffile's page readers to count every real pixel decode."""
    counter = {"n": 0}
    for cls_name in ("TiffPage", "TiffFrame"):
        cls = getattr(tifffile, cls_name, None)
        if cls is None or not hasattr(cls, "asarray"):
            continue
        original = cls.asarray

        def counting(self, *a, _orig=original, **kw):
            counter["n"] += 1
            return _orig(self, *a, **kw)

        monkeypatch.setattr(cls, "asarray", counting)
    return counter


# ---------------------------------------------------------------------------
# I2 / I3 -- lazy decode, and a fresh iterator per open()
# ---------------------------------------------------------------------------
def test_i2_frames_decode_lazily(multipage_trial, monkeypatch):
    """Building a source and taking 3 frames decodes 3 pages, not all N.

    Counted at the ``tifffile`` page level rather than at the FrameSource wrapper:
    a wrapper only sees frames *handed to it*, so a source that eagerly decoded
    everything and replayed it would look identical from up there. Only the real
    decode count distinguishes lazy from eager.
    """
    gcamp_path, _ = multipage_trial
    decodes = _count_page_decodes(monkeypatch)

    src = tiff_frame_source(gcamp_path)
    assert decodes["n"] == 0, "constructing a FrameSource must decode nothing"

    taken = list(itertools.islice(src.open(), 3))

    assert len(taken) == 3
    assert decodes["n"] == 3, (
        f"consuming 3 frames decoded {decodes['n']} pages; expected exactly 3 "
        f"(a full read would be {N_FRAMES})"
    )


def test_i3_open_returns_a_fresh_independent_iterator(multipage_trial):
    """open() must restart the sequence -- the two-pass math depends on it.

    Scope note: this pins re-iterability only. It does NOT catch a source that
    caches frames in a list, because replaying a list satisfies re-iterability
    perfectly -- it is the *memory* half of the contract that caching breaks, and
    that is pinned by test_i2/test_i10 (page-level decode counts) and the
    peak-memory test at the bottom of this file. Those three are what make
    caching fail; this one would stay green.
    """
    gcamp_path, _ = multipage_trial
    src = tiff_frame_source(gcamp_path)

    first = list(src.open())
    second = list(src.open())

    assert len(first) == len(second) == N_FRAMES
    np.testing.assert_array_equal(first[0], second[0])

    # Interleaving two live iterators proves they hold independent position.
    a, b = src.open(), src.open()
    next(a)
    np.testing.assert_array_equal(next(a), next(itertools.islice(b, 1, None)))


# ---------------------------------------------------------------------------
# I5 / I6 -- the interleaved split
# ---------------------------------------------------------------------------
def test_i5_interleaved_split_reads_only_two_images(interleaved_folder, monkeypatch):
    """Splitting a folder of 2N images decodes exactly 2 of them.

    The odd/even split comes from the sorted file list (no decode); only the
    brightness decision reads pixels, and only the first image of each group.
    This is what lets a folder too large for RAM still be split and streamed.
    """
    decodes = {"n": 0}
    original = tifffile.imread

    def counting_imread(*a, **kw):
        decodes["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(tifffile, "imread", counting_imread)

    gcamp_files, emo_files = interleaved_channel_files(interleaved_folder)

    assert decodes["n"] == 2, (
        f"the split decoded {decodes['n']} images; only the first image of each "
        f"group may be read (the brightness decision)"
    )
    assert len(gcamp_files) == len(emo_files) == N_FRAMES


def test_i6_same_split_feeds_full_load_and_streaming(interleaved_folder):
    """The file-list split is one code path shared by both memory strategies."""
    gcamp_files, emo_files = interleaved_channel_files(interleaved_folder)

    streamed = folder_frame_source(gcamp_files)
    assert streamed.count == len(gcamp_files)

    # Same list, stacked -> what the full-load path builds from the same split.
    stacked = np.stack(list(streamed.open()), axis=-1)
    assert stacked.shape == FRAME_SHAPE + (N_FRAMES,)


# ---------------------------------------------------------------------------
# I7 -- path coercion / back-compat
# ---------------------------------------------------------------------------
def test_i7_bare_path_is_coerced_to_a_frame_source(multipage_trial):
    """The pre-FrameSource call style (bare TIFF paths) still works."""
    gcamp_path, _ = multipage_trial

    from_path = _as_frame_source(gcamp_path)
    from_str = _as_frame_source(str(gcamp_path))
    already = tiff_frame_source(gcamp_path)

    assert from_path.count == from_str.count == N_FRAMES
    assert _as_frame_source(already) is already, "a FrameSource must pass through"


# ---------------------------------------------------------------------------
# I10 -- two passes, no more, no fewer
# ---------------------------------------------------------------------------
def test_i10_streaming_makes_exactly_two_passes(multipage_trial, monkeypatch):
    """A streaming run over N frames decodes exactly 2N per channel, opening each twice.

    Both halves matter and neither implies the other:
      * ``opens`` is counted at the FrameSource wrapper -- it proves the pipeline
        asks for exactly two passes (a third would re-read the whole file);
      * ``decodes`` is counted at the ``tifffile`` page level -- it proves those
        two passes really hit the disk twice. A source that decoded all N frames
        once and replayed them from a list would still hand 2N frames to the
        wrapper, so only the page-level count catches it -- and that regression
        makes memory O(frames), which is the whole thing streaming exists to avoid.
    """
    gcamp_path, emo_path = multipage_trial
    gcamp = CountingFrameSource(tiff_frame_source(gcamp_path))
    emo = CountingFrameSource(tiff_frame_source(emo_path))
    decodes = _count_page_decodes(monkeypatch)

    stream_trial_roi(gcamp, emo, _cfg(), trim=TRIM, downsample=DS)

    for name, src in (("gcamp", gcamp), ("emo", emo)):
        assert src.opens == 2, f"{name}: {src.opens} passes, expected exactly 2"

    # Two channels x two passes x N frames, decoded straight off the pages.
    assert decodes["n"] == 4 * N_FRAMES, (
        f"the run decoded {decodes['n']} pages; expected 2 channels * 2 passes * "
        f"{N_FRAMES} frames = {4 * N_FRAMES}. Fewer means frames were cached and "
        f"replayed (memory is no longer constant); more means an extra pass."
    )


# ---------------------------------------------------------------------------
# I8 / I9 -- the debug limit stays lazy
# ---------------------------------------------------------------------------
def test_i8_debug_limit_decodes_only_the_limit(multipage_trial):
    """--debug-max-frames k reads 2k frames off disk, never 2N.

    _limited_source wraps the source in an islice over a fresh open(), so it
    stays lazy AND re-iterable -- a truncated eager load would read everything.
    """
    limit = 4
    gcamp_path, emo_path = multipage_trial
    gcamp = CountingFrameSource(tiff_frame_source(gcamp_path))
    emo = CountingFrameSource(tiff_frame_source(emo_path))

    stream_trial_roi(
        _limited_source(gcamp, limit), _limited_source(emo, limit),
        _cfg(), trim=TRIM, downsample=DS,
    )

    for name, src in (("gcamp", gcamp), ("emo", emo)):
        assert src.opens == 2, f"{name}: {src.opens} passes, expected exactly 2"
        assert src.decodes == 2 * limit, (
            f"{name}: decoded {src.decodes} frames for a {limit}-frame debug run; "
            f"expected 2*{limit} (a full read would be {2 * N_FRAMES})"
        )


def test_i8_limited_source_is_still_re_iterable(multipage_trial):
    """The debug wrapper must not spend its iterator on the first pass."""
    gcamp_path, _ = multipage_trial
    limited = _limited_source(tiff_frame_source(gcamp_path), 3)

    assert limited.count == 3
    assert len(list(limited.open())) == 3
    assert len(list(limited.open())) == 3, "second pass came back empty"


def test_i9_debug_full_load_routes_through_lazy_sources(interleaved_folder, monkeypatch):
    """An in-memory debug run reads only the frames it keeps.

    _load_trials with a limit goes through _build_streaming_sources, so a debug
    full-load run over a huge folder does not read the whole folder either.
    """
    limit = 3
    decodes = _count_page_decodes(monkeypatch)
    original = tifffile.imread

    def counting_imread(*a, **kw):
        decodes["n"] += 1
        return original(*a, **kw)

    monkeypatch.setattr(tifffile, "imread", counting_imread)

    trials = _load_trials([str(interleaved_folder)], "interleaved_folder", limit)

    gcamp, emo = trials[0]
    assert gcamp.shape == FRAME_SHAPE + (limit,)
    assert emo.shape == FRAME_SHAPE + (limit,)
    # 2 images for the split's brightness decision + `limit` frames per channel.
    assert decodes["n"] == 2 + 2 * limit, (
        f"a {limit}-frame debug full-load decoded {decodes['n']} images; expected "
        f"2 (split) + 2*{limit} (kept frames), not the folder's {2 * N_FRAMES}"
    )


# ---------------------------------------------------------------------------
# I4 -- storage x memory axes stay orthogonal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["stack_file", "frame_folder", "interleaved_folder"])
def test_i4_every_source_layout_streams(source, multipage_trial, interleaved_folder,
                                        tmp_path_factory):
    """Every storage layout runs through the same streaming math.

    The two axes (how it is stored x how much is resident) must stay independent:
    collapsing them back into a matrix of special cases is the regression this
    guards against.
    """
    if source == "stack_file":
        trials = [tuple(str(p) for p in multipage_trial)]
    elif source == "frame_folder":
        tmp = tmp_path_factory.mktemp("frame_folders")
        paths = []
        for name, offset in (("gcamp", 0.0), ("emo", 500.0)):
            d = tmp / name
            d.mkdir()
            stack = _frames(offset=offset)
            for i in range(N_FRAMES):
                tifffile.imwrite(str(d / f"f_{i:05d}.tif"), stack[:, :, i].astype(np.float32))
            paths.append(str(d))
        trials = [tuple(paths)]
    else:
        trials = [str(interleaved_folder)]

    sources = _build_streaming_sources(trials, source)
    result = run_streaming(sources, _cfg(), trim=TRIM, downsample=DS)

    assert result.temp_roi.shape == (N_FRAMES, len(SMALL_BOXES), 1)
    assert np.isfinite(result.R_mean).all()


# ---------------------------------------------------------------------------
# I11 -- per-frame resize == whole-stack resize
# ---------------------------------------------------------------------------
def test_i11_per_frame_resize_equals_stack_resize():
    """imresize_box touches only the spatial axes, so slicing it is free.

    This identity is *why* streaming can match the in-memory path exactly rather
    than approximately; if it ever stopped holding, the streaming numbers would
    drift from the MATLAB-validated ones.
    """
    stack = _frames(n=6)

    whole = imresize_box(stack, DS)
    per_frame = np.stack(
        [imresize_box(stack[:, :, t], DS) for t in range(stack.shape[2])], axis=-1
    )

    np.testing.assert_array_equal(per_frame, whole)


# ---------------------------------------------------------------------------
# The constant-memory guarantee itself: peak allocation flat in frame count
# ---------------------------------------------------------------------------
def test_streaming_peak_memory_does_not_grow_with_frame_count(tmp_path):
    """Extra frames must not add frame-sized memory: the O(1) guarantee itself.

    The decode-count tests above prove the *passes* are right; this proves the
    consequence they exist for. tracemalloc (which numpy's allocator reports into)
    measures this process's own allocations, so it is stable enough to assert on
    in-process, unlike the OS working set.

    The assertion is on absolute growth, not on a ratio, because peak allocation
    is NOT perfectly flat even when the design is working: tifffile holds one page
    object per frame, so a longer recording costs a few hundred bytes per frame no
    matter what. That overhead is real but tiny, and a ratio test drowns in it --
    at 32x32 frames it alone moves the ratio ~1.4x and the test flaps.

    What actually matters is whether extra frames cost *frame-sized* memory. So:
    read 60 more frames, and require the peak to grow by far less than those 60
    frames would occupy. A source that accumulates frames blows through this by an
    order of magnitude; per-page bookkeeping does not come close.
    """
    shape = (64, 64)  # big enough that a frame dwarfs tifffile's per-page overhead
    frame_bytes = shape[0] * shape[1] * 8  # float64, as decoded
    small, large = 20, 80

    peaks = {}
    for n in (small, large):
        rng = np.random.default_rng(1)
        gcamp = rng.uniform(100.0, 200.0, shape + (n,))
        emo = rng.uniform(600.0, 700.0, shape + (n,))
        gcamp_p, emo_p = tmp_path / f"g_{n}.tif", tmp_path / f"e_{n}.tif"
        tifffile.imwrite(str(gcamp_p), np.moveaxis(gcamp, -1, 0).astype(np.float32))
        tifffile.imwrite(str(emo_p), np.moveaxis(emo, -1, 0).astype(np.float32))

        tracemalloc.start()
        stream_trial_roi(
            tiff_frame_source(gcamp_p), tiff_frame_source(emo_p),
            _cfg(), trim=TRIM, downsample=DS,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks[n] = peak

    extra_frames = large - small
    growth = peaks[large] - peaks[small]
    # Two channels' worth of the added frames -- what a caching source would cost.
    would_be_resident = 2 * extra_frames * frame_bytes
    budget = would_be_resident // 4

    print(f"  peak alloc: {small} frames={peaks[small]}B  {large} frames={peaks[large]}B")
    print(f"  growth={growth}B for {extra_frames} more frames/channel; "
          f"holding them would cost {would_be_resident}B (budget {budget}B)")
    assert growth < budget, (
        f"peak memory grew {growth} B when the recording grew by {extra_frames} "
        f"frames/channel ({would_be_resident} B if held resident). Streaming is "
        f"accumulating frames instead of reducing them away."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
