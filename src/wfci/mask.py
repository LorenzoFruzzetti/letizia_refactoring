"""Brain-mask stage -- restrict the analysis to pixels inside the brain.

Reproduces the masking half of
``Antea_scripts/(2)_Global_Signal_Regression_SCRIPT.txt``::

    Mask_resized = imresize(Mask, 0.5, 'box');
    for i = 1:size(Mask_resized,1)
        for j = 1:size(Mask_resized,2)
            if Mask_resized(i,j) == 0
                t_TEMP_resized(i,j,:,:) = NaN;
            end
        end
    end

i.e. pixels outside the mask become NaN for **all** frames and **all** trials,
and every later reduction (the ROI ``nanmean``, the global signal's ``nanmean``)
then ignores them.

Two things about this are easy to get wrong and are therefore explicit here:

**The mask is resized with 'box', not nearest-neighbour.** A binary mask
downsampled by a box filter comes back *fractional* -- a boundary pixel covering
half brain and half background lands at 0.5. MATLAB then tests ``== 0``, so a
pixel is excluded only when it has **no** brain in it at all; partial coverage is
kept. That is the behaviour reproduced by the default ``threshold=0.0``. It is a
real choice, not an accident of the code, so :func:`valid_from_mask` exposes it
rather than hiding it.

**"Outside the mask" is defined once.** The rest of the package never re-derives
it; it consumes NaN. This is what keeps the invalid set *static over time*, which
is the precondition the streaming GSR path depends on (see :mod:`wfci.gsr`).

Loading is an explicit path in / array out -- there is no ``uiopen`` equivalent
and no interactive file picker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .resize import imresize_box


def load_mask(path: str | Path) -> np.ndarray:
    """Load a brain mask image as a ``[y, x]`` float64 array.

    The mask's own scale is irrelevant -- 0/1, 0/255, logical, all behave the
    same, because everything downstream only asks whether a pixel is above
    :func:`valid_from_mask`'s threshold. Values are *not* binarised here: the box
    resize has to see the original numbers to compute fractional coverage.
    """
    arr = np.asarray(tifffile.imread(str(path)), dtype=np.float64)
    if arr.ndim == 3:
        # A single-page TIFF can still decode as [1, y, x] or [y, x, 1].
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 2:
            raise ValueError(
                f"Mask at {path} has shape {arr.shape}; expected a single 2-D image."
            )
        arr = squeezed
    if arr.ndim != 2:
        raise ValueError(
            f"Mask at {path} has shape {arr.shape}; expected a single 2-D image."
        )
    return arr


def resize_mask(mask: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downsample a mask with the same box filter used on the data.

    Deliberately the *same* :func:`~wfci.resize.imresize_box` as the image stacks,
    so mask and data land on identical pixel grids. Using a different resampler
    here (e.g. nearest neighbour) would shift the brain boundary by up to a pixel
    relative to the data it is masking.

    The result is fractional at the boundary; see the module docstring.
    """
    return imresize_box(np.asarray(mask, dtype=np.float64), scale)


def valid_from_mask(mask: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Boolean ``[y, x]`` map of pixels to KEEP: ``mask > threshold``.

    ``threshold=0.0`` reproduces the MATLAB (``if Mask_resized(i,j) == 0`` ->
    drop), keeping any pixel with a non-zero share of brain in it.

    Raise the threshold to be stricter about boundary pixels: ``0.5`` keeps only
    pixels at least half inside the brain (which is what MATLAB's ``imresize``
    would itself do to a *logical* mask, since it re-binarises the filtered
    result at 0.5). Passing the mask as logical vs numeric silently changes that
    in MATLAB; here it is an argument you can see.
    """
    return np.asarray(mask, dtype=np.float64) > threshold


def apply_mask(
    stack: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Set pixels outside the mask to NaN, across all frames and trials.

    Parameters
    ----------
    stack:
        ``[y, x, time]`` or ``[y, x, time, trial]``. Not modified; a masked copy
        is returned.
    mask:
        ``[y, x]``, already at ``stack``'s resolution (see :func:`resize_mask`).
        Its spatial shape must match exactly -- a mismatch here means the mask was
        drawn at a different resolution than the data was downsampled to, which
        would otherwise broadcast into silently wrong results.
    threshold:
        See :func:`valid_from_mask`.
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim not in (3, 4):
        raise ValueError(
            f"stack has shape {stack.shape}; expected [y, x, time] or "
            f"[y, x, time, trial]."
        )

    valid = valid_from_mask(mask, threshold)
    if valid.shape != stack.shape[:2]:
        raise ValueError(
            f"Mask shape {valid.shape} does not match the data's spatial shape "
            f"{stack.shape[:2]}. The mask must be at the SAME resolution as the "
            f"stack it masks -- the cortical pipeline masks the twice-downsampled "
            f"stack, so a mask drawn on the once-downsampled FOV needs "
            f"resize_mask(mask, 0.5) first."
        )
    if not valid.any():
        raise ValueError(
            f"Mask excludes every pixel (threshold={threshold}). Check the mask's "
            f"polarity: non-zero must mean 'inside the brain'."
        )

    out = stack.copy()
    # Broadcast [y, x] over the trailing time (and trial) axes: one decision per
    # pixel, applied to its whole time-series -- so the invalid set is static.
    out[~valid, ...] = np.nan
    return out
