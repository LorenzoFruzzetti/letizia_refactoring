"""Per-pixel dumps of the streaming intermediates.

The pipeline reduces every frame to ``n_roi`` box means and throws the pixels
away: in :func:`wfci.streaming.stream_trial_roi` the corrected DeltaF/F frame and
the two half-resolution raw-F frames live for one loop iteration and are then
overwritten. That is exactly the right default -- a 3000-frame recording is
~200 MB of pixels per volume and the ROI traces are ~0.5 MB -- but it means any
question that is not one of the predefined boxes (a different atlas, a seed-pixel
map, a per-pixel statistic, a check on the hemodynamic correction itself) can
only be answered by re-reading the raw TIFFs and re-running the pipeline.

:class:`PixelDump` is an optional sink that writes those pixels out *as they
stream*, so the pass count and the resident-memory profile are unchanged:

  * files are preallocated with ``np.lib.format.open_memmap`` and each frame is
    assigned into its own row, so nothing accumulates in RAM (invariant I2) and
    the write happens inside the existing pass 2 (invariant I10);
  * the result is a plain ``.npy`` that reads back with
    ``np.load(path, mmap_mode="r")``, so one frame or one pixel's time-course can
    be sliced without loading the whole volume.

**Axis order.** The dumps are ``[time, y, x]``, NOT the library's ``[y, x, time]``
(see LIBRARY.md section 3). This is a deliberate exception for *output files
only*, and the numeric path is untouched. Frame-major rows are contiguous, so
each frame is one sequential write; ``[y, x, time]`` would stride every frame
across the whole file, which for a 79 MB volume is the difference between a
sequential write and 6612 scattered ones. It is also ``tifffile``'s and ImageJ's
native order, so a dump opens as a stack without transposing.

**Region.** ``region`` crops to a rectangle on the final grid. The caller decides
what that rectangle means -- the *study* knows that its ROI boxes only ever
occupy part of the frame; the library only checks the window is inside the grid
and refuses otherwise rather than silently clipping.

**Dtypes.** ``dff_dtype``/``f_dtype`` are an on-disk storage choice, not a change
to the numeric path (which stays float64 throughout, as MATLAB parity requires).
Every value is range-checked against the target dtype before it is cast, so a
volume that does not fit raises instead of writing ``inf``.

**What the raw-F volumes do and do not let you recompute.** They are the exact
per-channel fluorescence on the final grid, so F0 drift, bleaching, SNR and the
reflectance channel itself can all be re-derived from them. They do NOT let you
reconstruct ``dff`` exactly. The pipeline corrects at HALF resolution and
downsamples the result, whereas anything computed from these volumes corrects
after the downsample; the correction is a ratio, and a box mean does not commute
with a ratio. The two agree closely on smooth real frames and diverge on
high-spatial-frequency noise. If you need the corrected quantity, use ``dff`` --
that is why it is dumped alongside rather than left to be derived.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .resize import imresize_box

# The three pixel-wise volumes a dump writes, and which stream each comes from.
# dff is the corrected DeltaF/F the ROI means are taken from; f_gcamp / f_emo are
# the raw fluorescence of the two channels, brought onto the same final grid.
VOLUME_NAMES = ("dff", "f_gcamp", "f_emo")


def _check_castable(name: str, arr: np.ndarray, dtype: np.dtype) -> None:
    """Raise if finite values of ``arr`` would not survive the cast to ``dtype``.

    Only overflow is checked, because that is the failure that is *silent*: a
    value above the dtype's maximum becomes ``inf`` and every downstream mean
    becomes ``inf`` with it. Precision loss is the documented, intended trade of
    choosing a narrow dtype and is not an error.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    limit = float(np.finfo(dtype).max)
    peak = float(np.abs(finite).max())
    if peak > limit:
        raise ValueError(
            f"{name}: value {peak:.6g} exceeds the maximum {limit:.6g} that "
            f"{np.dtype(dtype).name} can represent, so the cast would write inf. "
            f"Use a wider dtype for this volume."
        )


class PixelDump:
    """Incremental ``[time, y, x]`` .npy writer for one trial's pixel volumes.

    Parameters
    ----------
    out_dir:
        Directory to write into; created if missing.
    suffix:
        Appended to every file name, e.g. ``"full"`` -> ``pixels_dff_full.npy``.
        Mirrors the ``_debug``/``_full`` convention of the batch's other outputs.
    n_time:
        Number of frames that will be written. The files are preallocated to this
        length, so it must be the post-trim frame count.
    region:
        ``(row0, row1, col0, col1)`` half-open crop on the FINAL grid, or None for
        the whole frame. Validated against the first frame's shape.
    downsample:
        The pipeline's per-stage scale (0.5). Used to bring the half-resolution
        raw-F frames onto the final grid, so all three volumes share one geometry.
    dff_dtype, f_dtype:
        On-disk dtypes. See the module docstring.
    metadata:
        Extra provenance written verbatim into ``pixels_meta_<suffix>.npz``.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike,
        suffix: str,
        n_time: int,
        region: tuple[int, int, int, int] | None = None,
        downsample: float = 0.5,
        dff_dtype: Any = np.float16,
        f_dtype: Any = np.float32,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if n_time <= 0:
            raise ValueError(f"n_time must be positive, got {n_time}.")
        self.out_dir = str(out_dir)
        self.suffix = suffix
        self.n_time = int(n_time)
        self.region = None if region is None else tuple(int(v) for v in region)
        self.downsample = downsample
        self.dtypes = {
            "dff": np.dtype(dff_dtype),
            "f_gcamp": np.dtype(f_dtype),
            "f_emo": np.dtype(f_dtype),
        }
        self.metadata = dict(metadata or {})
        self._arrays: dict[str, np.memmap] = {}
        self._shape: tuple[int, int, int] | None = None
        self._baselines: tuple[np.ndarray, np.ndarray] | None = None
        self._written = 0
        self._closed = False
        os.makedirs(self.out_dir, exist_ok=True)

    # -- geometry ----------------------------------------------------------
    def path(self, name: str) -> str:
        """Where volume ``name`` is written."""
        return os.path.join(self.out_dir, f"pixels_{name}_{self.suffix}.npy")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.out_dir, f"pixels_meta_{self.suffix}.npz")

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        if self.region is None:
            return frame
        r0, r1, c0, c1 = self.region
        return frame[r0:r1, c0:c1]

    def _allocate(self, frame_shape: tuple[int, int]) -> None:
        """Preallocate the three .npy files once the frame shape is known."""
        if self.region is not None:
            r0, r1, c0, c1 = self.region
            if not (0 <= r0 < r1 <= frame_shape[0] and 0 <= c0 < c1 <= frame_shape[1]):
                raise ValueError(
                    f"Crop region (rows {r0}:{r1}, cols {c0}:{c1}) does not fit "
                    f"inside the {frame_shape[0]}x{frame_shape[1]} final grid. The "
                    f"region is not clipped -- widen the grid or move the window."
                )
            shape = (self.n_time, r1 - r0, c1 - c0)
        else:
            shape = (self.n_time, *frame_shape)
        self._shape = shape
        for name in VOLUME_NAMES:
            # open_memmap writes a real .npy header up front, so the file is a
            # valid array from the first frame on and never needs the whole
            # volume resident.
            self._arrays[name] = np.lib.format.open_memmap(
                self.path(name), mode="w+", dtype=self.dtypes[name], shape=shape
            )

    @property
    def shape(self) -> tuple[int, int, int] | None:
        """Written volume shape, or None before the first frame.

        Cached at allocation rather than read off the memmaps, so it still
        answers after :meth:`close` has released them.
        """
        return self._shape

    # -- sink protocol -----------------------------------------------------
    def baselines(self, mean_f: np.ndarray, mean_r: np.ndarray) -> None:
        """Record the two baseline images (pass 1's output).

        Kept at the half resolution the correction used them at, uncropped and in
        float64: they are two small images, and they are what makes the per-channel
        ratios (MATLAB's ``If2``/``Ir2``) exactly recoverable from the raw-F dumps.
        """
        self._baselines = (np.asarray(mean_f, dtype=np.float64).copy(),
                           np.asarray(mean_r, dtype=np.float64).copy())

    def frame(self, idx: int, g_half: np.ndarray, e_half: np.ndarray,
              dff_q: np.ndarray) -> None:
        """Write frame ``idx``, given exactly what pass 2 already holds.

        ``g_half``/``e_half`` are the trimmed, once-downsampled raw frames and
        ``dff_q`` the twice-downsampled corrected frame. The second downsample of
        the raw channels happens here rather than in the caller, so the streaming
        loop hands over its own locals and nothing about it changes. Per-frame
        resize equals whole-stack resize (invariant I11), so these are the same
        numbers a whole-stack downsample would give.
        """
        if self._closed:
            raise RuntimeError("PixelDump is closed; no more frames can be written.")
        if idx >= self.n_time:
            raise ValueError(
                f"Frame index {idx} is past the preallocated length {self.n_time}."
            )
        frames = {
            "dff": np.asarray(dff_q),
            "f_gcamp": imresize_box(g_half, self.downsample),
            "f_emo": imresize_box(e_half, self.downsample),
        }
        if not self._arrays:
            self._allocate(frames["dff"].shape)
        for name, frame in frames.items():
            cropped = self._crop(frame)
            _check_castable(f"{name}[{idx}]", cropped, self.dtypes[name])
            self._arrays[name][idx] = cropped.astype(self.dtypes[name])
        self._written = max(self._written, idx + 1)

    def close(self) -> None:
        """Flush the volumes and write the sidecar metadata."""
        if self._closed:
            return
        self._closed = True
        for arr in self._arrays.values():
            arr.flush()
        meta: dict[str, Any] = dict(self.metadata)
        meta.update(
            n_time=self.n_time,
            n_written=self._written,
            downsample=self.downsample,
            axis_order="time,y,x",
            volumes=np.array(VOLUME_NAMES),
            dtypes=np.array([self.dtypes[n].name for n in VOLUME_NAMES]),
        )
        if self.shape is not None:
            meta["shape"] = np.array(self.shape)
        if self.region is not None:
            # The crop origin is what maps a dumped pixel back to full-frame
            # coordinates: full_row = row0 + dumped_row.
            meta["region"] = np.array(self.region)
        if self._baselines is not None:
            meta["mean_f"] = self._baselines[0]
            meta["mean_r"] = self._baselines[1]
        np.savez(self.meta_path, **{k: np.asarray(v) for k, v in meta.items()})
        self._arrays.clear()

    def __enter__(self) -> "PixelDump":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
