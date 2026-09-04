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

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ROIConfig
from .gsr import GSRConfig
from .io import FrameSource, tiff_frame_source
from .mask import valid_from_mask
from .profiles import Profile
from .resize import imresize_box
from .roi import box_slices_for, functional_connectivity


@dataclass
class StreamingResult:
    """Outputs of the streaming pipeline.

    Mirrors :class:`wfci.pipeline.PipelineResult` but without ``dff_stack``: the
    corrected stack is never materialised (that is the whole point of streaming).
    """

    temp_roi: np.ndarray         # TEMP_ROI        [time, n_rois, trial]
    R: np.ndarray                # per-trial       [n_rois, n_rois, trial]
    R_mean: np.ndarray           # trial mean      [n_rois, n_rois]
    averaged_traces: np.ndarray  # trial mean      [time, n_rois]


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
    mask: np.ndarray | None = None,
    gsr: GSRConfig | None = None,
    mask_threshold: float = 0.0,
    expected_grid: tuple[int, int] | None = None,
    pixel_dump=None,
) -> np.ndarray:
    """Stream one ``(gcamp, emo)`` trial from disk to its ``[time, n_rois]`` trace.

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
        ROI geometry (Bregma reference + the boxes), as for the in-memory path.
        Box order defines the column order of the returned trace.
    baseline_slice:
        Temporal window (over the trimmed frames) for the mean baseline image.
        ``slice(None)`` for resting-state, ``slice(0, 278)`` for stimulated.
    trim, downsample:
        Same meaning as :func:`wfci.correction.build_dff_stack`.
    mask:
        ``[y, x]`` brain mask at the final (twice-downsampled) resolution. Applied
        per frame, which is identical to masking the whole stack because the
        decision is per pixel and constant in time.
    gsr:
        Global signal regression config, or None.
    mask_threshold:
        See :func:`wfci.mask.valid_from_mask`.
    expected_grid:
        The final ``(rows, cols)`` the ROI atlas was drawn for; see
        :func:`wfci.roi.box_slices_for`. None skips the check.
    pixel_dump:
        Optional :class:`wfci.dump.PixelDump` (or anything with the same
        ``baselines``/``frame``/``close`` methods). When given, every frame's raw
        and corrected pixels are written to disk from *inside* pass 2 -- no extra
        read, no extra resident array, so the two-pass and one-frame invariants
        (I10, I2) are unaffected. None (the default) is the pixel-free path.
    """
    gcamp_src = _as_frame_source(gcamp)
    emo_src = _as_frame_source(emo)
    if pixel_dump is not None and gsr is not None:
        # A post-GSR per-pixel stack does not exist in this path by construction:
        # _stream_pass2_gsr accumulates only the sufficient statistics (a, b,
        # g_vec) precisely so it never has to hold one, and materialising it would
        # need a third pass over the file -- breaking invariant I10. Dumping the
        # PRE-GSR pixels under a name that implies otherwise would be worse than
        # refusing, so refuse.
        raise ValueError(
            "pixel_dump is not supported together with GSR: the streaming GSR "
            "path never materialises a post-GSR per-pixel frame (it accumulates "
            "only the OLS statistics), and producing one would require a third "
            "pass over the data. Dump without GSR, or use the in-memory path."
        )
    invalid = None if mask is None else ~valid_from_mask(mask, mask_threshold)
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

    # Precompute + validate the ROI box slices once, against the final
    # quarter-res grid they will be applied to. The frame shape is known now
    # (mean_f was built at that resolution) rather than only inside the loop, so a
    # geometry error surfaces before pass 2 reads the file a second time.
    final_shape = imresize_box(mean_f, downsample).shape
    resolved = box_slices_for(cfg, final_shape, expected_grid)
    names = [name for name, _, _ in resolved]
    box_slices = [(rs, cs) for _, rs, cs in resolved]

    if gsr is not None:
        return _stream_pass2_gsr(
            gcamp_src, emo_src, mean_f, mean_r, n_time, trim, downsample,
            box_slices, invalid, gsr,
        )

    # --- pass 2: per-frame correction -> ROI means ---------------------------
    temp_roi = np.empty((n_time, len(names)), dtype=np.float64)
    g_stream = _half_res_frames(gcamp_src, trim, downsample)
    e_stream = _half_res_frames(emo_src, trim, downsample)
    if pixel_dump is not None:
        pixel_dump.baselines(mean_f, mean_r)
    try:
        for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
            if idx >= n_time:
                break
            # DeltaF/F (%) for this frame, exactly as hemodynamic_correction does it.
            dff_half = ((g_half / mean_f) / (e_half / mean_r) - 1.0) * 100.0
            # Second 0.5x box downsample (per-frame == whole-stack slice).
            dff_q = imresize_box(dff_half, downsample)
            if invalid is not None:
                _check_mask_shape(invalid, dff_q)
                dff_q = np.where(invalid, np.nan, dff_q)
            if pixel_dump is not None:
                # Written from the loop's own locals, after masking, so the dump
                # holds exactly the pixels the ROI means below are taken from.
                pixel_dump.frame(idx, g_half, e_half, dff_q)
            for j, (rs, cs) in enumerate(box_slices):
                temp_roi[idx, j] = np.nanmean(dff_q[rs, cs])
    finally:
        if pixel_dump is not None:
            pixel_dump.close()
    return temp_roi


def _check_mask_shape(invalid: np.ndarray, frame: np.ndarray) -> None:
    if invalid.shape != frame.shape:
        raise ValueError(
            f"Mask shape {invalid.shape} does not match the corrected frame's "
            f"{frame.shape}. The mask must be at the FINAL (twice-downsampled) "
            f"resolution -- see wfci.pipeline.prepare_mask."
        )


def _stream_pass2_gsr(
    gcamp_src: FrameSource,
    emo_src: FrameSource,
    mean_f: np.ndarray,
    mean_r: np.ndarray,
    n_time: int,
    trim: int,
    downsample: float,
    box_slices: list[tuple[slice, slice]],
    invalid: np.ndarray | None,
    gsr: GSRConfig,
) -> np.ndarray:
    """Pass 2 with global signal regression, in constant memory.

    GSR looks fundamentally anti-streaming: it regresses **each pixel's whole
    time-series** against the global signal, which naively means holding
    ``[y, x, time]`` resident -- destroying the property that makes this module
    worth having. It doesn't, for two reasons.

    **1. Per-pixel OLS needs only sufficient statistics**, all accumulable one
    frame at a time: ``N``, ``sum(g)``, ``sum(g^2)`` (scalars) and ``sum(p)``,
    ``sum(g*p)`` (two ``[y, x]`` images). The global signal ``g(t)`` is a
    *spatial* mean of frame ``t``, so it is known at frame ``t`` -- no lookahead.

    **2. The ROI mean is linear and ``g(t)`` is one scalar per frame**, so the
    regressed ROI trace never needs the regressed *stack*::

        T_B(t) = mean_B( p(t) - a*g(t) - b )
               = mean_B(p(t)) - g(t)*mean_B(a) - mean_B(b)

    Every term is either accumulated per frame (``mean_B(p(t))``, ``g(t)``) or
    computed once at the end from the two stat images (``mean_B(a)``,
    ``mean_B(b)``).

    So GSR costs two extra ``[y, x]`` images and two ``[time]`` vectors, and the
    pass count stays at TWO -- which the decode-count test in
    ``tests/test_efficiency_invariants.py`` enforces.

    The precondition is that the invalid-pixel set is **static over time**; it is
    checked here rather than assumed. See :func:`_check_static_nans`.
    """
    n_rois = len(box_slices)
    sum_p = sum_gp = nan_count = None
    g_vec = np.empty(n_time, dtype=np.float64)
    raw_roi = np.empty((n_time, n_rois), dtype=np.float64)

    g_stream = _half_res_frames(gcamp_src, trim, downsample)
    e_stream = _half_res_frames(emo_src, trim, downsample)
    for idx, (g_half, e_half) in enumerate(zip(g_stream, e_stream)):
        if idx >= n_time:
            break
        dff_half = ((g_half / mean_f) / (e_half / mean_r) - 1.0) * 100.0
        dff_q = imresize_box(dff_half, downsample)
        if invalid is not None:
            _check_mask_shape(invalid, dff_q)
            dff_q = np.where(invalid, np.nan, dff_q)

        if sum_p is None:
            sum_p = np.zeros_like(dff_q)
            sum_gp = np.zeros_like(dff_q)
            nan_count = np.zeros(dff_q.shape, dtype=np.int64)

        bad = ~np.isfinite(dff_q)
        nan_count += bad

        # Two views of the same frame, so every reduction below excludes exactly
        # the non-finite pixels -- NaN *and* inf -- matching what the in-memory
        # path does (wfci.gsr drops any pixel that is not finite, and its global
        # signal averages only finite pixels). Plain np.nanmean would not: it
        # ignores NaN but propagates inf.
        clean = np.where(bad, 0.0, dff_q)      # for the running sums: finite
        masked = np.where(bad, np.nan, dff_q)  # for the nanmeans: inf -> NaN

        n_finite = dff_q.size - int(bad.sum())
        if n_finite == 0:
            raise ValueError(
                f"The global signal is undefined at frame {idx}: it has no valid "
                f"pixels at all. Check the brain mask and the input data."
            )
        g_t = float(clean.sum()) / n_finite    # the global signal, frame t
        g_vec[idx] = g_t

        with warnings.catch_warnings():
            # A fully-masked ROI box is all-NaN -- a legitimate NaN column, just as
            # in the in-memory path.
            warnings.simplefilter("ignore", RuntimeWarning)
            for j, (rs, cs) in enumerate(box_slices):
                raw_roi[idx, j] = np.nanmean(masked[rs, cs])

        # Zero-filled at the invalid pixels so the sums stay finite; their a/b are
        # overwritten with NaN below, so the zeros never reach a result.
        sum_p += clean
        sum_gp += g_t * clean

    _check_static_nans(nan_count, n_time)
    valid = nan_count == 0

    # The OLS fit, from the accumulated statistics. g(t) was kept in full (it is
    # only [time] floats), so the mean and variance of g are computed from it
    # directly rather than from sum(g)/sum(g^2): the centred form is what the
    # in-memory path uses, and reproducing it keeps the two paths agreeing to
    # roundoff instead of to catastrophic-cancellation error.
    g_mean = g_vec.mean()
    g_centred = g_vec - g_mean
    g_var = float(g_centred @ g_centred)
    if g_var <= gsr.min_variance:
        raise ValueError(
            f"The global signal has no variance (sum of squares {g_var:.3g}), so "
            f"no slope is identifiable. This usually means a constant or "
            f"single-frame recording."
        )

    # sum_t (g(t) - gbar) * p(t) == sum(g*p) - gbar*sum(p), i.e. N*cov(g, p).
    cov = sum_gp - g_mean * sum_p
    a = cov / g_var
    b = sum_p / n_time - a * g_mean
    a[~valid] = np.nan
    b[~valid] = np.nan

    # T_B(t) = raw_B(t) - g(t)*mean_B(a) - mean_B(b)
    temp_roi = np.empty((n_time, n_rois), dtype=np.float64)
    with warnings.catch_warnings():
        # A fully-masked box is all-NaN; that is a legitimate NaN column, exactly
        # as the in-memory path produces.
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, (rs, cs) in enumerate(box_slices):
            mean_a = np.nanmean(a[rs, cs])
            mean_b = np.nanmean(b[rs, cs])
            temp_roi[:, j] = raw_roi[:, j] - g_vec * mean_a - mean_b
    return temp_roi


def _check_static_nans(nan_count: np.ndarray, n_time: int) -> None:
    """Refuse to stream GSR when a pixel is NaN for only SOME frames.

    The two-pass identity computes each frame's ROI mean *during* pass 2, but
    ``nan_policy="drop_pixel"`` decides validity from the pixel's WHOLE
    time-series. If a pixel is valid in some frames and NaN in others, streaming
    would include it wherever it happens to be valid, while the in-memory path
    excludes it from every frame. The two would then disagree -- subtly, and only
    on real data.

    Under mask-only NaNs the invalid set is static and the identity is exact. A
    partially-NaN pixel means something else produced NaNs mid-recording (e.g. a
    zero in ``emo(t)`` making the ratio non-finite).

    The in-memory result is the one that is right (MERGING_PLAN.md P1), so this
    raises rather than silently returning different numbers.
    """
    partial = (nan_count > 0) & (nan_count < n_time)
    if not partial.any():
        return
    n_partial = int(partial.sum())
    ys, xs = np.nonzero(partial)
    example = f"(y={ys[0]}, x={xs[0]}) is NaN in {int(nan_count[ys[0], xs[0]])}/{n_time} frames"
    raise ValueError(
        f"Cannot stream GSR: {n_partial} pixel(s) are NaN in some frames but not "
        f"all -- e.g. {example}. Streaming GSR requires the invalid-pixel set to "
        f"be static over time (it holds for mask-only NaNs); a partially-NaN pixel "
        f"would be treated differently here than by the in-memory path, which "
        f"drops such pixels entirely. Re-run with streaming disabled "
        f"(--no-streaming) to get the correct, in-memory result."
    )


def run_streaming(
    trial_sources: list[tuple[FrameSource | str | Path, FrameSource | str | Path]],
    cfg: ROIConfig,
    baseline_slice: slice = slice(None),
    corr_window: slice = slice(None),
    trim: int = 20,
    downsample: float = 0.5,
    mask: np.ndarray | None = None,
    gsr: GSRConfig | None = None,
    mask_threshold: float = 0.0,
    expected_grid: tuple[int, int] | None = None,
    pixel_dump_factory=None,
) -> StreamingResult:
    """Streaming pipeline over a list of ``(gcamp, emo)`` trials.

    Each trial is a pair of :class:`~wfci.io.FrameSource` objects (or bare TIFF
    paths, treated as multi-page files). It is streamed to its ``[time, n_rois]``
    ROI trace (constant memory); only the tiny per-trial traces are stacked, then
    the usual per-trial correlation and trial averaging run on them.

    ``mask`` / ``gsr`` mirror :func:`wfci.pipeline.run_pipeline`'s optional stages;
    the mask must already be at the final resolution (see
    :func:`wfci.pipeline.prepare_mask`).

    ``pixel_dump_factory`` is an optional ``(trial_index) -> PixelDump | None``.
    It is a factory rather than a single sink because each trial needs its own
    files; returning None for a trial dumps nothing for it. See
    :mod:`wfci.dump`.
    """
    traces = [
        stream_trial_roi(
            g, e, cfg, baseline_slice, trim, downsample,
            mask=mask, gsr=gsr, mask_threshold=mask_threshold,
            expected_grid=expected_grid,
            pixel_dump=None if pixel_dump_factory is None else pixel_dump_factory(i),
        )
        for i, (g, e) in enumerate(trial_sources)
    ]
    # [time, n_rois, trial]
    temp_roi = np.stack(traces, axis=-1)
    R, R_mean, averaged_traces = functional_connectivity(temp_roi, window=corr_window)
    return StreamingResult(temp_roi, R, R_mean, averaged_traces)


def run_streaming_profile(
    trial_sources,
    cfg: ROIConfig,
    profile: Profile,
    mask: np.ndarray | None = None,
    pixel_dump_factory=None,
) -> StreamingResult:
    """Streaming counterpart of :func:`wfci.pipeline.run_profile`.

    Same profile, same numbers, constant memory -- the storage x memory x profile
    axes stay independent, so any profile runs either way. ``pixel_dump_factory``
    is passed straight through to :func:`run_streaming`.
    """
    from .pipeline import prepare_mask

    cfg = ROIConfig(y_1=cfg.y_1, x_2=cfg.x_2, boxes=dict(profile.atlas))
    return run_streaming(
        trial_sources,
        cfg,
        baseline_slice=profile.baseline,
        corr_window=profile.corr_window,
        trim=profile.trim,
        downsample=profile.downsample,
        mask=prepare_mask(mask, profile),
        gsr=profile.gsr,
        expected_grid=profile.atlas.grid,
        pixel_dump_factory=pixel_dump_factory,
    )


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
