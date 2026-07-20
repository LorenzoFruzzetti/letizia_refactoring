"""Step 3 - ROI time-series extraction and functional connectivity.

Reproduces ``step3_ROI_functional_connectivity*.m`` (cerebellar) and
``Antea_scripts/(4)_FOV128x128_corr_SCRIPT.txt`` (cortical) -- the two are the
same computation over a different number of boxes:
  1. average Delta F / F inside each ROI box (nanmean over rows and cols) to get
     one time-series per region,
  2. assemble them column-wise in ``cfg.boxes`` order,
  3. per trial compute the Pearson correlation matrix over a time window (the
     full recording for resting-state, a stimulus window for stimulated),
  4. average the correlation matrices across trials.

Nothing here is anatomy-specific or fixed at four regions: every output is sized
from ``len(cfg.boxes)``, so a 22-box cortical config yields ``[time, 22, trial]``
and a 22x22 ``R`` with no code change.
"""

from __future__ import annotations

import numpy as np

from .config import Box, ROIConfig


def _box_slices(box: Box, y_1: int, x_2: int) -> tuple[slice, slice]:
    """Convert MATLAB 1-based inclusive offset ranges to Python 0-based slices.

    MATLAB ``img(y_1+a : y_1+b, ...)`` (1-based, inclusive) becomes
    ``arr[y_1+a-1 : y_1+b, ...]`` in Python (0-based, end-exclusive).

    The result is NOT validated here -- use :func:`box_slices_for`, which knows the
    frame it will be applied to. An unvalidated slice from this function can be
    silently wrong; see that function for why.
    """
    r0 = y_1 + box.row_start - 1
    r1 = y_1 + box.row_end        # exclusive end == inclusive end (b) since -1 cancels
    c0 = x_2 + box.col_start - 1
    c1 = x_2 + box.col_end
    return slice(r0, r1), slice(c0, c1)


def box_slices_for(
    cfg: ROIConfig,
    frame_shape: tuple[int, int],
    expected_grid: tuple[int, int] | None = None,
) -> list[tuple[str, slice, slice]]:
    """Resolve every ROI box against a real frame, or refuse.

    Returns ``[(label, row_slice, col_slice), ...]`` in ``cfg.boxes`` order.

    **Why this exists.** ROI boxes are offsets from Bregma, so whether they land on
    the brain depends on the Bregma, the field of view and the downsampling -- none
    of which the box itself knows. When they do not fit, plain NumPy indexing does
    not complain; it does something worse:

    * a **negative** index counts from the far end, so a left-hemisphere ROI
      quietly averages the RIGHT side of the image and returns a perfectly
      plausible number (MATLAB raises here -- the port is more permissive than its
      source, and that is a hazard, not a feature);
    * an index past the end silently truncates, or yields an empty slice whose
      ``nanmean`` is NaN with only a RuntimeWarning.

    Both produce output that looks like data. So this raises instead, naming the
    ROI and the coordinate.

    ``expected_grid`` catches the case bounds-checking cannot: an atlas drawn for
    a *different* FOV whose boxes all still happen to fit. Every offset is then
    wrong by a scale factor while every box lands on real pixels. Only a declared
    grid (:attr:`wfci.atlases.Atlas.grid`) can detect that, which is why atlases
    carry one.
    """
    rows, cols = int(frame_shape[0]), int(frame_shape[1])

    if expected_grid is not None and (rows, cols) != tuple(expected_grid):
        raise ValueError(
            f"This atlas was drawn for a {expected_grid[0]}x{expected_grid[1]} "
            f"frame, but the data is {rows}x{cols}. ROI offsets are in pixels, so "
            f"they do not transfer between fields of view: every box would land on "
            f"the wrong anatomy while still looking like a valid result. Either use "
            f"an atlas drawn for this FOV, or -- if the offsets really are correct "
            f"here -- clear the check with dataclasses.replace(atlas, grid=None)."
        )

    resolved: list[tuple[str, slice, slice]] = []
    problems: list[str] = []
    for name, box in cfg.boxes.items():
        rs, cs = _box_slices(box, cfg.y_1, cfg.x_2)
        if rs.start < 0 or cs.start < 0 or rs.stop > rows or cs.stop > cols:
            problems.append(
                f"  {name}: rows {rs.start}:{rs.stop}, cols {cs.start}:{cs.stop}"
            )
        resolved.append((name, rs, cs))

    if problems:
        raise ValueError(
            f"{len(problems)} ROI box(es) fall outside the {rows}x{cols} frame, "
            f"with Bregma at (y_1={cfg.y_1}, x_2={cfg.x_2}):\n"
            + "\n".join(problems)
            + f"\nA box outside the frame does not fail loudly in NumPy -- a "
              f"negative index reads from the opposite edge, so the ROI would "
              f"average the wrong part of the brain and return a normal-looking "
              f"number. Check the Bregma (--bregma-row / --bregma-col) and that "
              f"the atlas matches this field of view."
        )
    return resolved


def extract_roi_timeseries(
    dff_stack: np.ndarray,
    cfg: ROIConfig,
    expected_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return ``TEMP_ROI`` of shape ``[time, n_rois, trial]``.

    ``dff_stack`` is ``[y, x, time, trial]`` (the output of step 1). Columns are
    ordered by ``cfg.boxes`` -- for the default cerebellar atlas that is
    [Laterale_L, Verme_L, Laterale_R, Verme_R], exactly as the MATLAB script.

    Every box is validated against the frame first (:func:`box_slices_for`): an
    ROI that does not fit is an error, not a quietly mis-indexed average.
    """
    n_time = dff_stack.shape[2]
    n_trial = dff_stack.shape[3]
    boxes = box_slices_for(cfg, dff_stack.shape[:2], expected_grid)
    temp_roi = np.empty((n_time, len(boxes), n_trial), dtype=np.float64)

    for t in range(n_trial):
        img = dff_stack[:, :, :, t]
        for j, (_name, rs, cs) in enumerate(boxes):
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
