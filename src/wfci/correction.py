"""Step 1 - hemodynamic correction to Delta F / F.

Reproduces ``step1_correzione_emodinamica_*.m``. For one trial we have a GCaMP
fluorescence stack and an ``emo`` (hemodynamic / reflectance) stack, both
already trimmed of the first 20 frames and downsampled once by 0.5.

Correction (MATLAB):
    MIf = mean(gCaMP, 3);              # temporal mean image
    MIr = mean(emo,   3);
    If2 = gCaMP ./ MIf;               # per-pixel normalisation
    Ir2 = emo   ./ MIr;
    t_temp = If2 ./ Ir2;              # divide out hemodynamics
    dff    = (t_temp - 1) * 100;      # Delta F / F in percent

The only difference between the resting-state and stimulated variants is the
baseline window used for the temporal mean:
  * resting-state: the full recording  (baseline_slice = slice(None))
  * stimulated:    the pre-stimulus window only (e.g. slice(0, 278))
"""

from __future__ import annotations

import numpy as np

from .resize import imresize_box


def hemodynamic_correction(
    gcamp: np.ndarray,
    emo: np.ndarray,
    baseline_slice: slice = slice(None),
) -> np.ndarray:
    """Compute Delta F / F (%) for one trial.

    Parameters
    ----------
    gcamp, emo:
        ``[y, x, time]`` stacks (float64), already trimmed and downsampled once.
    baseline_slice:
        Temporal window (along axis 2) used for the mean baseline image.
        ``slice(None)`` for resting-state, ``slice(0, 278)`` for stimulated.
    """
    mean_f = np.mean(gcamp[:, :, baseline_slice], axis=2, keepdims=True)
    mean_r = np.mean(emo[:, :, baseline_slice], axis=2, keepdims=True)

    ratio_f = gcamp / mean_f
    ratio_r = emo / mean_r
    t_temp = ratio_f / ratio_r
    return (t_temp - 1.0) * 100.0  # Delta F / F in percent


def build_dff_stack(
    trials: list[tuple[np.ndarray, np.ndarray]],
    baseline_slice: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
) -> np.ndarray:
    """Full step 1: raw trial stacks -> ``t_TEMP_resize1`` = ``[y, x, time, trial]``.

    For each ``(gcamp_raw, emo_raw)`` trial this mirrors the MATLAB script:
      1. drop the first ``trim`` frames,
      2. downsample once by ``downsample`` (box),
      3. hemodynamic correction -> Delta F / F,
      4. downsample the corrected stack once more by ``downsample`` (box),
    then stacks all trials along a 4th axis.
    """
    corrected = []
    for gcamp_raw, emo_raw in trials:
        gcamp = imresize_box(gcamp_raw[:, :, trim:], downsample)
        emo = imresize_box(emo_raw[:, :, trim:], downsample)
        dff = hemodynamic_correction(gcamp, emo, baseline_slice)
        corrected.append(imresize_box(dff, downsample))
    return np.stack(corrected, axis=-1)
