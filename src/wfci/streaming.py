"""Constant-memory (streaming) variant of steps 1 + 3.

The in-memory path (:func:`wfci.pipeline.run_pipeline`) loads every trial's full
``[y, x, time]`` stack into RAM, exactly like the MATLAB script that reads the
whole multi-page TIFF into one ``double`` array before doing anything. At
512x512 x float64 = 2 MB/frame that is 10+ GB for a multi-GB acquisition, per
channel -- which is precisely the case the MATLAB comment
``ricorda che se pesa piu di 4gb non lo legge`` was warning about.

Nothing in the math actually needs the whole stack resident:

  * the baseline image is a *temporal mean*  -> a running sum is enough;
  * the DeltaF/F correction is *per-frame* once the two mean images are known;
  * both 0.5x box downsamples are *per-frame spatial* operations;
  * step 3 reduces each frame to four ROI ``nanmean`` values -> a tiny trace.

So we stream the file in TWO passes and keep only a couple of small images
resident at any time:

  pass 1: accumulate the baseline-window sum of the (half-resolution) frames
          -> mean images ``MIf`` / ``MIr``;
  pass 2: re-read frame by frame, correct -> DeltaF/F, downsample again, and
          reduce to the four ROI means -> one ``[time, 4]`` trace per trial.

The per-frame arithmetic is byte-for-byte the same operations as the in-memory
path (``imresize_box``, the correction formula, the ROI ``nanmean``), so results
match the MATLAB-validated pipeline up to floating-point summation order in the
baseline mean (~1e-13, negligible for DeltaF/F percentages). The trade-off is
two passes over the file instead of one, and no ``dff_stack`` is retained (it is
streamed away), so step-2 visualization needs the in-memory path instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROIConfig
from .io import FrameSource, tiff_frame_source
from .resize import imresize_box
from .roi import _box_slices, functional_connectivity


@dataclass
class StreamingResult:
    """Outputs of the streaming pipeline.

    Mirrors :class:`wfci.pipeline.PipelineResult` but without ``dff_stack``: the
    corrected stack is never materialised (that is the whole point of streaming).
    """

    temp_roi: np.ndarray         # TEMP_ROI        [time, 4, trial]
    R: np.ndarray                # per-trial       [4, 4, trial]
    R_mean: np.ndarray           # trial mean      [4, 4]
    averaged_traces: np.ndarray  # trial mean      [time, 4]


def _as_frame_source(channel: FrameSource | str | Path) -> FrameSource:
    """Coerce a channel argument to a :class:`FrameSource`.

    A :class:`FrameSource` passes through unchanged; a path (``str``/``Path``) is
    treated as a single multi-page TIFF -- the original streaming input -- so the
    older ``stream_trial_roi(gcamp_path, emo_path, ...)`` call style keeps working.
    """
    if isinstance(channel, FrameSource):
        return channel
    return tiff_frame_source(channel)


def _half_res_frames(source: FrameSource, trim: int, downsample: float):
    """Yield trimmed, once-downsampled ``[y, x]`` frames from a frame source.

    Reproduces the first two MATLAB steps per frame: drop the first ``trim``
    frames (``out(:,:,21:end)``), then ``imresize(...,0.5,'box')``. Resizing a
    single 2-D frame is identical to resizing the whole stack slice-by-slice,
    because ``imresize_box`` only touches the spatial axes. ``source.open()``
    gives a fresh iterator, so each call is one independent pass over the data.
    """
    gen = source.open()
    for _ in range(trim):
        next(gen, None)  # skip the trimmed lead-in frames
    for frame in gen:
        yield imresize_box(frame, downsample)


def stream_trial_roi(
    gcamp: FrameSource | str | Path,
    emo: FrameSource | str | Path,
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
) -> np.ndarray:
    """Stream one ``(gcamp, emo)`` trial from disk to its ``[time, 4]`` ROI trace.

    Constant memory: only the two running-sum images (pass 1) and the current
    frame plus the two mean images (pass 2) are ever resident.

    Parameters
    ----------
    gcamp, emo:
        The two channels as :class:`~wfci.io.FrameSource` objects, i.e. any
        re-iterable frame sequence (single multi-page TIFF, folder of single-page
        TIFFs, or one channel of an interleaved folder). A bare path is also
        accepted and treated as a multi-page TIFF, for the original call style.
        The storage format is thus decoupled from the streaming math here.
    cfg:
        ROI geometry (Bregma reference + the four boxes), as for the in-memory
        path. Box order defines the column order of the returned trace.
    baseline_slice:
        Temporal window (over the trimmed frames) for the mean baseline image.
        ``slice(None)`` for resting-state, ``slice(0, 278)`` for stimulated.
    trim, downsample:
        Same meaning as :func:`wfci.correction.build_dff_stack`.
    """
    gcamp_src = _as_frame_source(gcamp)
    emo_src = _as_frame_source(emo)
    # Frame count after trimming; both channels are assumed equal length, so we
    # take the shorter one to stay in lockstep.
    n_total = min(gcamp_src.count, emo_src.count)
    n_time = n_total - trim
    # Which trimmed-frame indices fall inside the baseline window.
    base_lo, base_hi, _ = baseline_slice.indices(n_time)
    base_start, base_stop = base_lo, base_hi

    # --- pass 1: baseline mean over half-resolution frames --------------------
    sum_f = sum_r = None
    count = 0
    g_stream = _half_res_frames(gcamp_src, trim, downsample)
    e_stream = _half_res_frames(emo_src, trim, downsample)
    for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
        if idx >= n_time:
            break
        if base_start <= idx < base_stop:
            if sum_f is None:
                sum_f = np.zeros_like(g_half)
                sum_r = np.zeros_like(e_half)
            sum_f += g_half  # running temporal sum, not the whole stack
            sum_r += e_half
            count += 1
    mean_f = sum_f / count  # MIf
    mean_r = sum_r / count  # MIr

    # Precompute ROI box slices once (they live on the final quarter-res frame).
    names = list(cfg.boxes.keys())
    box_slices = [_box_slices(cfg.boxes[name], cfg.y_1, cfg.x_2) for name in names]

    # --- pass 2: per-frame correction -> ROI means ---------------------------
    temp_roi = np.empty((n_time, len(names)), dtype=np.float64)
    g_stream = _half_res_frames(gcamp_src, trim, downsample)
    e_stream = _half_res_frames(emo_src, trim, downsample)
    for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
        if idx >= n_time:
            break
        # DeltaF/F (%) for this frame, exactly as hemodynamic_correction does it.
        dff_half = ((g_half / mean_f) / (e_half / mean_r) - 1.0) * 100.0
        # Second 0.5x box downsample (per-frame == whole-stack slice).
        dff_q = imresize_box(dff_half, downsample)
        for j, (rs, cs) in enumerate(box_slices):
            temp_roi[idx, j] = np.nanmean(dff_q[rs, cs])
    return temp_roi


def run_streaming(
    trial_sources: list[tuple[FrameSource | str | Path, FrameSource | str | Path]],
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    corr_window: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
) -> StreamingResult:
    """Streaming steps 1 + 3 over a list of ``(gcamp, emo)`` trials.

    Each trial is a pair of :class:`~wfci.io.FrameSource` objects (or bare TIFF
    paths, treated as multi-page files). It is streamed to its ``[time, 4]`` ROI
    trace (constant memory); only the tiny per-trial traces are stacked, then the
    usual per-trial 4x4 correlation and trial averaging run on them.
    """
    traces = [
        stream_trial_roi(g, e, cfg, baseline_slice, trim, downsample)
        for g, e in trial_sources
    ]
    # [time, 4, trial]
    temp_roi = np.stack(traces, axis=-1)
    R, R_mean, averaged_traces = functional_connectivity(temp_roi, window=corr_window)
    return StreamingResult(temp_roi, R, R_mean, averaged_traces)


def run_streaming_resting_state(trial_sources, cfg, **kwargs) -> StreamingResult:
    """Resting-state streaming: full-recording baseline and correlation."""
    return run_streaming(
        trial_sources, cfg, baseline_slice=slice(None), corr_window=slice(None), **kwargs
    )


def run_streaming_stimulated(
    trial_sources,
    cfg,
    baseline_slice: slice = slice(0, 278),
    corr_window: slice = slice(279, 300),
    **kwargs,
) -> StreamingResult:
    """Stimulated streaming: pre-stimulus baseline (MATLAB 1:278) and
    stimulus-window correlation (MATLAB 280:300)."""
    return run_streaming(
        trial_sources, cfg, baseline_slice=baseline_slice, corr_window=corr_window,
        **kwargs,
    )
