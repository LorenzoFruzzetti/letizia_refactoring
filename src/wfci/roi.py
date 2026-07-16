"""Step 3 - ROI time-series extraction and functional connectivity.

Reproduces ``step3_ROI_functional_connectivity*.m``:
  1. average Delta F / F inside each of the four ROI boxes (nanmean over rows
     and cols) to get one time-series per region,
  2. assemble them column-wise as [Laterale_L, Verme_L, Laterale_R, Verme_R],
  3. per trial compute the 4x4 Pearson correlation matrix (over the full
     recording for resting-state, or over a stimulus window for stimulated),
  4. average the correlation matrices across trials.
"""

from __future__ import annotations

import numpy as np

from .config import Box, ROIConfig


def _box_slices(box: Box, y_1: int, x_2: int) -> tuple[slice, slice]:
    """Convert MATLAB 1-based inclusive offset ranges to Python 0-based slices.

    MATLAB ``img(y_1+a : y_1+b, ...)`` (1-based, inclusive) becomes
    ``arr[y_1+a-1 : y_1+b, ...]`` in Python (0-based, end-exclusive).
    """
    r0 = y_1 + box.row_start - 1
    r1 = y_1 + box.row_end        # exclusive end == inclusive end (b) since -1 cancels
    c0 = x_2 + box.col_start - 1
    c1 = x_2 + box.col_end
    return slice(r0, r1), slice(c0, c1)


def extract_roi_timeseries(dff_stack: np.ndarray, cfg: ROIConfig) -> np.ndarray:
    """Return ``TEMP_ROI`` of shape ``[time, 4, trial]``.

    ``dff_stack`` is ``[y, x, time, trial]`` (the output of step 1). Columns are
    ordered exactly as the MATLAB script: [Laterale_L, Verme_L, Laterale_R,
    Verme_R].
    """
    n_time = dff_stack.shape[2]
    n_trial = dff_stack.shape[3]
    names = list(cfg.boxes.keys())
    temp_roi = np.empty((n_time, len(names), n_trial), dtype=np.float64)

    for t in range(n_trial):
        img = dff_stack[:, :, :, t]
        for j, name in enumerate(names):
            rs, cs = _box_slices(cfg.boxes[name], cfg.y_1, cfg.x_2)
            patch = img[rs, cs, :]                     # [rows, cols, time]
            # nanmean over rows then cols -> one value per frame.
            temp_roi[:, j, t] = np.nanmean(patch, axis=(0, 1))
    return temp_roi


def _corrcoef_matlab(x: np.ndarray) -> np.ndarray:
    """Pearson correlation of columns, matching MATLAB ``corr(X)``.

    ``x`` is ``[time, regions]``; returns ``[regions, regions]``. Uses
    ``np.corrcoef`` (variables in columns), which yields the same Pearson
    coefficients as MATLAB ``corr`` (the N vs N-1 normalisation cancels).
    """
    return np.corrcoef(x, rowvar=False)


def functional_connectivity(
    temp_roi: np.ndarray,
    window: slice = slice(None),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-trial and trial-averaged connectivity.

    Parameters
    ----------
    temp_roi:
        ``[time, regions, trial]`` from :func:`extract_roi_timeseries`.
    window:
        Time window used for correlation. ``slice(None)`` (resting-state) uses
        the full recording; the stimulated variant uses ``slice(279, 300)``
        (MATLAB ``280:300``).

    Returns
    -------
    (R, R_mean, averaged_traces)
        ``R``: ``[regions, regions, trial]`` per-trial correlation matrices;
        ``R_mean``: ``[regions, regions]`` mean across trials;
        ``averaged_traces``: ``[time, regions]`` mean ROI traces across trials.
    """
    n_trial = temp_roi.shape[2]
    n_reg = temp_roi.shape[1]
    r = np.empty((n_reg, n_reg, n_trial), dtype=np.float64)
    for t in range(n_trial):
        r[:, :, t] = _corrcoef_matlab(temp_roi[window, :, t])
    r_mean = np.mean(r, axis=2)
    averaged_traces = np.mean(temp_roi, axis=2)
    return r, r_mean, averaged_traces
