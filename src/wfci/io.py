"""TIFF loading for wide-field imaging stacks.

The original MATLAB reader consumed *multi-page* TIFFs (one page per time
frame) with ``imread(NAME, ii)`` in a loop. It also warned that files larger
than ~4 GB are read incorrectly; ``tifffile`` used here has no such limit.

Two acquisition layouts are supported:
  * a single multi-page TIFF file  -> :func:`load_stack`
  * a folder of single-page TIFFs   -> :func:`load_frame_folder`
    (the sample dataset in ``data/`` is of this second kind: R11_00001..7.tif,
    each one 512x512 frame, forming one channel's time series when sorted.)

Every loader returns a float64 array shaped ``[y, x, time]`` to match the
MATLAB convention (rows, cols, frames).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import tifffile


def tiff_frame_count(path: str | Path) -> int:
    """Number of pages (time frames) in a multi-page TIFF, without reading them.

    Only the IFD headers are parsed, so this is cheap even for multi-GB files.
    Used by the streaming path to size the baseline window before any frame is
    loaded into memory.
    """
    with tifffile.TiffFile(str(path)) as tif:
        return len(tif.pages)


def iter_tiff_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield one ``[y, x]`` float64 frame at a time from a multi-page TIFF.

    This is the constant-memory alternative to :func:`load_stack`: pages are
    decoded lazily as the generator is consumed, so a recording of any length
    (including the >4 GB files the MATLAB reader mishandled) never has more than
    one frame resident at a time.
    """
    with tifffile.TiffFile(str(path)) as tif:
        for page in tif.pages:
            yield np.asarray(page.asarray(), dtype=np.float64)


@dataclass
class FrameSource:
    """A *re-iterable* sequence of ``[y, x]`` float64 frames plus its length.

    This is the storage-format-agnostic input the streaming pipeline consumes.
    Streaming makes TWO passes over each channel (baseline mean, then the
    per-frame correction), so it must be able to restart the sequence: ``open()``
    returns a *fresh* iterator each call. ``count`` is the number of frames,
    known without decoding any pixels, so the baseline window can be sized up
    front while memory stays constant.

    Decoupling the frame source from the on-disk layout is what lets any storage
    format (single multi-page TIFF, folder of single-page TIFFs, or one channel's
    half of an interleaved folder) feed the same constant-memory math -- i.e. the
    "how the channels are stored" axis is now independent of the "stream vs load
    it all" axis.
    """

    count: int
    _open: Callable[[], Iterator[np.ndarray]]

    def open(self) -> Iterator[np.ndarray]:
        """Return a fresh ``[y, x]`` frame iterator for one pass over the data."""
        return self._open()


def tiff_frame_source(path: str | Path) -> FrameSource:
    """A :class:`FrameSource` backed by one multi-page TIFF (one page per frame).

    ``count`` comes from the IFD headers only (:func:`tiff_frame_count`) and each
    pass lazily decodes pages (:func:`iter_tiff_frames`), so nothing beyond a
    single frame is ever resident.
    """
    return FrameSource(tiff_frame_count(path), lambda: iter_tiff_frames(path))


def iter_folder_frames(files: list[str | Path]) -> Iterator[np.ndarray]:
    """Yield ``[y, x]`` float64 frames from an ordered list of single-page TIFFs.

    The folder equivalent of :func:`iter_tiff_frames`: one file is decoded per
    step, so a channel stored as thousands of single-page TIFFs streams in
    constant memory just like a multi-page file.
    """
    for f in files:
        yield np.asarray(tifffile.imread(str(f)), dtype=np.float64)


def folder_frame_source(files: list[str | Path]) -> FrameSource:
    """A :class:`FrameSource` backed by an explicit, ordered list of TIFF files.

    Used for one channel of an interleaved folder (the odd- or even-positioned
    images already split out by :func:`interleaved_channel_files`), where the
    frame list is not a whole directory. ``count`` is just ``len(files)``.
    """
    files = list(files)
    return FrameSource(len(files), lambda: iter_folder_frames(files))


def frame_folder_source(folder: str | Path, pattern: str = "*.tif") -> FrameSource:
    """A :class:`FrameSource` backed by a folder of single-page TIFFs (one channel).

    Streaming counterpart of :func:`load_frame_folder`: same lexicographic sort
    (R11_00001, R11_00002, ...), but frames are decoded one at a time instead of
    stacked into RAM.
    """
    files = sorted(Path(folder).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    return folder_frame_source(files)


def load_stack(path: str | Path) -> np.ndarray:
    """Load a multi-page TIFF as a ``[y, x, time]`` float64 array."""
    arr = tifffile.imread(str(path))  # -> [time, y, x] or [y, x] if single page
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # tifffile yields [time, y, x]; MATLAB convention is [y, x, time].
    return np.moveaxis(arr, 0, -1)


def load_frame_folder(folder: str | Path, pattern: str = "*.tif") -> np.ndarray:
    """Assemble single-page TIFFs in a folder into one ``[y, x, time]`` stack.

    Files are sorted lexicographically by name (R11_00001, R11_00002, ...),
    which is the natural acquisition order for zero-padded frame indices.
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")
    frames = [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in files]
    # Stack along a new time axis -> [y, x, time].
    return np.stack(frames, axis=-1)


def _top_percent_mean(image: np.ndarray, top_percent: float = 10.0) -> float:
    """Mean of the brightest ``top_percent``% of pixels in one image.

    Same signal proxy used by ``src/inspect_channel_intensity.py``: taking only
    the top pixels compares the two channels on their illuminated part rather
    than on background, which is what separates the two interleaved groups
    cleanly. On this rig the reflectance (emo) channel is the *brighter* of the
    two and GCaMP the dimmer, so the group with the LOWER value here is gcamp.
    """
    pixels = np.asarray(image, dtype=np.float64).ravel()
    threshold = np.percentile(pixels, 100.0 - top_percent)
    return float(pixels[pixels >= threshold].mean())


def interleaved_channel_files(
    folder: str | Path,
    pattern: str = "*.tif",
    top_percent: float = 10.0,
    channel_order: str = "auto",
) -> tuple[list[Path], list[Path]]:
    """Split ONE interleaved folder into ordered ``(gcamp_files, emo_files)`` lists.

    This is the *split-and-identify* half of the interleaved layout, done without
    decoding a single full frame's worth of the recording: sorted by name, the
    odd-positioned images (1st, 3rd, 5th, ...) are one channel and the
    even-positioned images (2nd, 4th, ...) the other -- exactly the odd/even
    split that ``src/inspect_channel_intensity.py`` inspects.

    ``channel_order`` decides which group is which:

    ``"auto"`` (default)
        Decide by intensity: the DIMMER group is GCaMP, the brighter one is emo
        (the reflectance channel, which on this rig comes back stronger than the
        GCaMP fluorescence). We compare the first image of each group (i.e. the
        first two images in the folder) by the mean of their top
        ``top_percent``% pixels; the dimmer one's group becomes gcamp. Only those
        two images are read, so this stays cheap even for a folder too large to
        load.
    ``"gcamp_first"`` / ``"emo_first"``
        Assign by position instead, reading nothing at all. This is what the
        MATLAB scripts do implicitly (the cerebellar ones take GCaMP first, the
        cortical ones emo first) -- and getting it wrong silently swaps the
        channels, inverting the hemodynamic correction. Use these only when you
        know the layout and the brightness heuristic misfires (e.g. an unusually
        bright GCaMP recording, where the two channels are close enough that the
        dimmer-is-gcamp rule can pick the wrong group).

    The two lists are truncated to equal length (odd total -> groups differ by
    one) so the channels stay in lockstep for the per-frame hemodynamic
    correction. :func:`load_interleaved_folder` stacks these into memory; the
    streaming path feeds each list to :func:`folder_frame_source` instead, so the
    same split powers both the load-it-all and constant-memory variants.
    """
    valid_orders = ("auto", "gcamp_first", "emo_first")
    if channel_order not in valid_orders:
        raise ValueError(
            f"channel_order={channel_order!r} is not one of {valid_orders}."
        )

    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 images matching {pattern!r} in {folder} to split "
            f"two interleaved channels; found {len(files)}."
        )

    # 1-based position: odd group = 1st, 3rd, ... -> files[0::2];
    #                   even group = 2nd, 4th, ... -> files[1::2].
    odd_files = files[0::2]
    even_files = files[1::2]

    if channel_order == "gcamp_first":
        gcamp_files, emo_files = odd_files, even_files
    elif channel_order == "emo_first":
        gcamp_files, emo_files = even_files, odd_files
    else:
        # Decide which group is the dimmer (GCaMP) channel from the first image
        # of each group -- "the first two images" the user inspects.
        odd_intensity = _top_percent_mean(tifffile.imread(str(odd_files[0])), top_percent)
        even_intensity = _top_percent_mean(tifffile.imread(str(even_files[0])), top_percent)
        if odd_intensity <= even_intensity:
            gcamp_files, emo_files = odd_files, even_files
        else:
            gcamp_files, emo_files = even_files, odd_files

    # Keep the two channels the same length (odd total -> groups differ by one).
    n = min(len(gcamp_files), len(emo_files))
    return gcamp_files[:n], emo_files[:n]


def load_interleaved_folder(
    folder: str | Path,
    pattern: str = "*.tif",
    top_percent: float = 10.0,
    channel_order: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Split ONE folder of interleaved single-page TIFFs into ``(gcamp, emo)``.

    Full-load variant: :func:`interleaved_channel_files` decides the odd/even
    split and channel identity, then each group is stacked into a ``[y, x, time]``
    float64 array ready to be used as one trial by the in-memory pipeline. For a
    recording too large for RAM, feed the same split to the streaming path via
    :func:`interleaved_channel_files` + :func:`folder_frame_source` instead.
    """
    gcamp_files, emo_files = interleaved_channel_files(
        folder, pattern, top_percent, channel_order
    )
    gcamp = np.stack(
        [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in gcamp_files],
        axis=-1,
    )
    emo = np.stack(
        [np.asarray(tifffile.imread(str(f)), dtype=np.float64) for f in emo_files],
        axis=-1,
    )
    return gcamp, emo
