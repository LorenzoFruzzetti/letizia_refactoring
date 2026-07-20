"""Step 2 - ROI placement check (visual overlay).

Reproduces ``step2_area_location_*.m`` / ``Antea_scripts/(3)_..._ROIposition.txt``:
paint the ROI boxes onto a representative Delta F / F frame and display it, so you
can confirm the boxes land where you expect relative to Bregma. This is a visual
check only, not part of the numeric pipeline.

**One atlas, drawn and averaged.** The MATLAB writes the box coordinates twice --
once in the step-2 overlay script, once in the step-3 averaging script -- and the
cerebellar pair has *drifted*: step 2 draws ``Laterale_L`` three columns from where
step 3 averages it, so the figure meant to verify the ROI shows a box only half
overlapping the data it came from. Here the overlay reads the same ``cfg.boxes``
the pipeline averages, so the picture cannot disagree with the numbers.
"""

from __future__ import annotations

import numpy as np

from .config import ROIConfig
from .roi import box_slices_for


def overlay_rois(
    frame: np.ndarray,
    cfg: ROIConfig,
    fill: float = 1.0,
    expected_grid: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with the ROI boxes filled with ``fill``.

    Boxes are validated against the frame (:func:`wfci.roi.box_slices_for`), so an
    ROI that does not fit raises here too -- an overlay that quietly painted the
    wrong pixels would be worse than no overlay, since its whole job is to be
    trusted as a check.
    """
    out = frame.copy()
    for _name, rs, cs in box_slices_for(cfg, frame.shape[:2], expected_grid):
        out[rs, cs] = fill
    return out


def show_roi_placement(
    dff_stack: np.ndarray,
    cfg: ROIConfig,
    frame_index: int = 302,
    average_trials: bool = False,
    clim: tuple[float, float] = (0.3, 3.0),
    ax=None,
):
    """Display one frame with ROI boxes overlaid (mirrors step 2).

    ``dff_stack`` is ``[y, x, time, trial]``. With ``average_trials=True`` the
    stimulated variant's trial-averaged frame is used. Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    if average_trials:
        frame = np.mean(dff_stack, axis=3)[:, :, frame_index]
    else:
        frame = dff_stack[:, :, frame_index, 0]

    overlaid = overlay_rois(frame, cfg)
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(overlaid, vmin=clim[0], vmax=clim[1])
    ax.figure.colorbar(im, ax=ax)
    return ax
