"""End-to-end pipeline orchestration (resting-state and stimulated).

Ties together step 1 (hemodynamic correction) and step 3 (ROI functional
connectivity). Step 2 is a visual check and lives in :mod:`wfci.visualize`.

The only differences between the resting-state and stimulated variants are two
windows, captured here as parameters:
  * baseline window for Delta F / F   (step 1)
  * correlation window                (step 3)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ROIConfig
from .correction import build_dff_stack
from .roi import extract_roi_timeseries, functional_connectivity


@dataclass
class PipelineResult:
    dff_stack: np.ndarray        # t_TEMP_resize1  [y, x, time, trial]
    temp_roi: np.ndarray         # TEMP_ROI        [time, 4, trial]
    R: np.ndarray                # per-trial       [4, 4, trial]
    R_mean: np.ndarray           # trial mean      [4, 4]
    averaged_traces: np.ndarray  # trial mean      [time, 4]


def run_pipeline(
    trials: list[tuple[np.ndarray, np.ndarray]],
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    corr_window: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
) -> PipelineResult:
    """Run steps 1 + 3 for a list of ``(gcamp_raw, emo_raw)`` trials."""
    dff = build_dff_stack(
        trials, baseline_slice=baseline_slice, trim=trim, downsample=downsample
    )
    temp_roi = extract_roi_timeseries(dff, cfg)
    R, R_mean, averaged_traces = functional_connectivity(temp_roi, window=corr_window)
    return PipelineResult(dff, temp_roi, R, R_mean, averaged_traces)


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
