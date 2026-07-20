"""End-to-end pipeline orchestration.

Ties the stages together::

    correction -> [mask] -> [GSR] -> ROI -> connectivity

Step 1 (hemodynamic correction) and step 3 (ROI functional connectivity) are
common to both wide-field pipelines. The two bracketed stages are optional and
belong to the cortical pipeline (``Antea_scripts/(2)_..._SCRIPT.txt``); the
cerebellar pipeline simply leaves them off, which is exactly its historical
behaviour. Step 2 is a visual check and lives in :mod:`wfci.visualize`.

The differences between pipelines are parameters, not code paths -- and a
:class:`~wfci.profiles.Profile` bundles them, so :func:`run_profile` is the usual
entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ROIConfig
from .correction import build_dff_stack
from .gsr import GSRConfig, regress_global
from .mask import apply_mask, resize_mask
from .profiles import Profile
from .roi import extract_roi_timeseries, functional_connectivity


@dataclass
class PipelineResult:
    dff_stack: np.ndarray        # t_TEMP_resize1  [y, x, time, trial]
    temp_roi: np.ndarray         # TEMP_ROI        [time, n_rois, trial]
    R: np.ndarray                # per-trial       [n_rois, n_rois, trial]
    R_mean: np.ndarray           # trial mean      [n_rois, n_rois]
    averaged_traces: np.ndarray  # trial mean      [time, n_rois]


def _regress_trials(dff: np.ndarray, gsr: GSRConfig) -> np.ndarray:
    """Apply GSR to each trial of a ``[y, x, time, trial]`` stack.

    Per trial, exactly as the MATLAB loops over ``size(t_TEMP_resized,4)``: the
    global signal is a property of one recording, so trials are never pooled.
    """
    return np.stack(
        [regress_global(dff[:, :, :, t], gsr) for t in range(dff.shape[3])],
        axis=-1,
    )


def run_pipeline(
    trials: list[tuple[np.ndarray, np.ndarray]],
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    corr_window: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
    mask: np.ndarray | None = None,
    gsr: GSRConfig | None = None,
    mask_threshold: float = 0.0,
    expected_grid: tuple[int, int] | None = None,
) -> PipelineResult:
    """Run the stage chain for a list of ``(gcamp_raw, emo_raw)`` trials.

    Parameters
    ----------
    trials, cfg, baseline_slice, corr_window, trim, downsample:
        As before -- the defaults are the cerebellar resting-state pipeline and
        are unchanged.
    mask:
        ``[y, x]`` brain mask **already at the corrected stack's resolution**
        (i.e. after both downsamples). Pixels outside it become NaN for every
        frame and trial. ``None`` (default) applies no mask.
        :func:`run_profile` handles the resizing; see :func:`wfci.mask.resize_mask`.
    gsr:
        Regress the global signal out of every pixel, per trial, after masking.
        ``None`` (default) skips it.
    mask_threshold:
        See :func:`wfci.mask.valid_from_mask`.
    expected_grid:
        The final ``(rows, cols)`` the ROI atlas was drawn for; see
        :func:`wfci.roi.box_slices_for`. ``None`` skips the check.

    Notes
    -----
    Stage order is not arbitrary: the mask must precede GSR because the global
    signal is the mean over *brain* pixels, and both must precede the ROI means
    so those average the regressed data.
    """
    dff = build_dff_stack(
        trials, baseline_slice=baseline_slice, trim=trim, downsample=downsample
    )
    if mask is not None:
        dff = apply_mask(dff, mask, threshold=mask_threshold)
    if gsr is not None:
        dff = _regress_trials(dff, gsr)
    temp_roi = extract_roi_timeseries(dff, cfg, expected_grid=expected_grid)
    R, R_mean, averaged_traces = functional_connectivity(temp_roi, window=corr_window)
    return PipelineResult(dff, temp_roi, R, R_mean, averaged_traces)


def prepare_mask(
    mask: np.ndarray | None,
    profile: Profile,
) -> np.ndarray | None:
    """Resize a raw mask onto the corrected stack's grid, per the profile.

    Shared by the in-memory and streaming paths so both interpret a mask
    identically. Returns None when the profile does not use a mask.
    """
    if not profile.use_mask:
        if mask is not None:
            raise ValueError(
                f"Profile {profile.name!r} does not use a brain mask, but one was "
                f"supplied. Use a profile with use_mask=True, or drop the mask."
            )
        return None
    if mask is None:
        raise ValueError(
            f"Profile {profile.name!r} requires a brain mask (use_mask=True) but "
            f"none was given. Pass mask=load_mask(path). Running it without one "
            f"would average skull and background into the global signal."
        )
    if profile.mask_downsample in (None, 1.0):
        return np.asarray(mask, dtype=np.float64)
    return resize_mask(mask, profile.mask_downsample)


def run_profile(
    trials: list[tuple[np.ndarray, np.ndarray]],
    cfg: ROIConfig,
    profile: Profile,
    mask: np.ndarray | None = None,
) -> PipelineResult:
    """Run the pipeline described by ``profile``.

    ``cfg`` supplies the per-animal Bregma; the profile supplies everything else,
    including the ROI atlas -- so ``cfg.boxes`` is overridden by ``profile.atlas``
    and need not be set by the caller.

    ``mask`` is the mask **as loaded** (see :func:`wfci.mask.load_mask`); the
    profile's ``mask_downsample`` puts it on the data's grid.
    """
    cfg = ROIConfig(y_1=cfg.y_1, x_2=cfg.x_2, boxes=dict(profile.atlas))
    return run_pipeline(
        trials,
        cfg,
        baseline_slice=profile.baseline,
        corr_window=profile.corr_window,
        trim=profile.trim,
        downsample=profile.downsample,
        mask=prepare_mask(mask, profile),
        gsr=profile.gsr,
        expected_grid=profile.atlas.grid,
    )


def run_resting_state(trials, cfg, **kwargs) -> PipelineResult:
    """Resting-state: full-recording baseline and full-recording correlation."""
    return run_pipeline(
        trials, cfg, baseline_slice=slice(None), corr_window=slice(None), **kwargs
    )


def run_stimulated(
    trials,
    cfg,
    baseline_slice: slice = slice(0, 278),
    corr_window: slice = slice(279, 300),
    **kwargs,
) -> PipelineResult:
    """Stimulated: pre-stimulus baseline (MATLAB 1:278) and stimulus-window
    correlation (MATLAB 280:300)."""
    return run_pipeline(
        trials, cfg, baseline_slice=baseline_slice, corr_window=corr_window, **kwargs
    )
