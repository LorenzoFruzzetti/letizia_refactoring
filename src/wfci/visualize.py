"""Step 2 - ROI placement sanity check (visual overlay).

Reproduces ``step2_area_location_*.m``: paint the four ROI boxes onto a
representative Delta F / F frame and display it, so you can confirm the boxes
land on vermis and lateral hemispheres relative to Bregma. This is a visual
check only, not part of the numeric pipeline.
"""

from __future__ import annotations

import numpy as np

from .config import ROIConfig
from .roi import _box_slices


def overlay_rois(frame: np.ndarray, cfg: ROIConfig, fill: float = 1.0) -> np.ndarray:
    """Return a copy of ``frame`` with the four ROI boxes filled with ``fill``."""
    out = frame.copy()
    for box in cfg.boxes.values():
        rs, cs = _box_slices(box, cfg.y_1, cfg.x_2)
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
