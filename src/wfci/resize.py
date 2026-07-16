"""MATLAB-equivalent ``imresize(..., scale, 'box')``.

The original pipeline downsamples every image stack with MATLAB's
``imresize(A, 0.5, 'box')``. To get bit-for-bit comparable results we
reproduce MATLAB's exact resampling algorithm (the ``contributions`` routine
from ``images.internal.resize``) rather than approximating it.

Key facts, verified against MATLAB R2024a:
  * For EVEN input dimensions and scale 0.5 the 'box' method collapses to a
    plain 2x2 block mean (this repo's real data is 512 -> 256 -> 128, all even).
  * For ODD dimensions MATLAB uses an antialiasing box kernel with fractional
    contribution weights (e.g. a 3x3 -> [[3, 7.5], [4.5, 9]]). The general
    algorithm below reproduces both cases exactly.
"""

from __future__ import annotations

import numpy as np


def _box_kernel(x: np.ndarray) -> np.ndarray:
    # MATLAB box kernel: h(x) = 1 for -0.5 <= x < 0.5, else 0.
    return ((x >= -0.5) & (x < 0.5)).astype(np.float64)


def _contributions(in_length: int, scale: float):
    """Replicate MATLAB's ``contributions`` for the box kernel.

    Returns ``(weights, indices)`` where output pixel ``k`` is
    ``sum_p weights[k, p] * input[indices[k, p]]`` (indices are 0-based and
    already clamped to the valid range, which replicates edge pixels).
    """
    out_length = int(np.ceil(in_length * scale))

    kernel_width = 1.0
    if scale < 1:
        # Antialiasing: widen and rescale the kernel when downsampling.
        h = lambda t: scale * _box_kernel(scale * t)  # noqa: E731
        kernel_width = kernel_width / scale
    else:
        h = _box_kernel

    # Output-space coordinates (1-based, as in MATLAB).
    x = np.arange(1, out_length + 1, dtype=np.float64)
    # Inverse map: output 0.5 -> input 0.5 (pixel-center convention).
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)

    left = np.floor(u - kernel_width / 2.0)
    p = int(np.ceil(kernel_width)) + 2  # number of candidate input pixels
    indices = left[:, None] + np.arange(p)[None, :]  # 1-based input coords

    weights = h(u[:, None] - indices)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Clamp to valid range (replicates end pixels) and convert to 0-based.
    indices = np.clip(indices, 1, in_length).astype(np.intp) - 1
    return weights, indices


def _resize_axis(arr: np.ndarray, axis: int, scale: float) -> np.ndarray:
    n = arr.shape[axis]
    if scale == 0.5 and n % 2 == 0:
        # Fast path: an even-length axis at scale 0.5 'box' is exactly the mean
        # of adjacent pixel pairs (the general algorithm below assigns weights
        # 0.5/0.5 to the two contributors and 0 to the rest). This is the common
        # case for this repo's real data (512 -> 256 -> 128, all even) and the
        # dominant cost of the streaming path.
        #
        # Bit-identical to the general path: for a size-2 reduction that path
        # computes 0.5*a + 0.5*b while mean computes (a + b)/2, and both round to
        # round(0.5*(a + b)) because halving a double is exact. Splitting the
        # axis into (n//2, 2) is a pure stride view (no copy), so this is just
        # one fused reduction instead of a gather + weighted sum.
        split = arr.shape[:axis] + (n // 2, 2) + arr.shape[axis + 1:]
        return arr.reshape(split).mean(axis=axis + 1)

    weights, indices = _contributions(n, scale)
    moved = np.moveaxis(arr, axis, 0)               # (in_len, ...)
    gathered = moved[indices]                        # (out_len, p, ...)
    w = weights.reshape(weights.shape + (1,) * (gathered.ndim - 2))
    resized = (gathered * w).sum(axis=1)             # (out_len, ...)
    return np.moveaxis(resized, 0, axis)


def imresize_box(arr: np.ndarray, scale: float) -> np.ndarray:
    """Resize the first two axes of ``arr`` by ``scale`` using the box method.

    Mirrors MATLAB ``imresize(arr, scale, 'box')``: only the first two
    dimensions (rows, cols) are resized; any trailing dimensions (time, trial)
    are left untouched, exactly as MATLAB resizes each 2-D slice of a stack.
    Computation is done in float64 to match MATLAB's double arithmetic.
    """
    out = np.asarray(arr, dtype=np.float64)
    out = _resize_axis(out, 0, scale)  # rows
    out = _resize_axis(out, 1, scale)  # cols
    return out
