"""Global signal regression (GSR) -- remove the brain-wide common signal.

Reproduces the regression half of
``Antea_scripts/(2)_Global_Signal_Regression_SCRIPT.txt``::

    global_signal = squeeze(nanmean(nanmean(data,1),2));   % one value per frame
    for row = 1:size(data,1)
        for col = 1:size(data,2)
            if ~isnan(data(row,col,:))                     % ALL frames finite
                pixel_signal = squeeze(data(row,col,:));
                mdl = fitlm(global_signal, pixel_signal);
                global_component = mdl.Coefficients.Estimate(2)*global_signal ...
                                 + mdl.Coefficients.Estimate(1);
                data_regressed(row,col,:) = pixel_signal - global_component;
            else
                data_regressed(row,col,:) = NaN;
            end
        end
    end

For every pixel: fit its time-series against the global signal by least squares,
subtract the fitted line, keep the residual.

**Why this is not 16 384 model fits.** ``fitlm(g, p)`` with a single predictor and
an intercept *is* ordinary least squares, and OLS with one regressor has a closed
form. So the whole double loop collapses to a handful of whole-image array ops
computed for every pixel at once::

    a = cov(g, p) / var(g)          slope      (per pixel)
    b = mean(p) - a * mean(g)       intercept  (per pixel)
    residual = p - a*g - b

This is the single most expensive step of the MATLAB pipeline and it becomes
essentially free. The speedup is a *consequence* of the vectorisation, not a
different algorithm -- ``tests/test_gsr.py`` pins it against an explicit
per-pixel ``lstsq`` reference at ~1e-16.

**NaN policy** (``GSRConfig.nan_policy``) is an explicit option from day one
rather than a default that might change later:

``"drop_pixel"`` (default)
    Reproduces MATLAB's ``if ~isnan(data(row,col,:))``. MATLAB's ``if`` over an
    array is true only when *every* element is -- so a pixel with even one NaN
    frame is dropped from **all** frames. Deterministic, matches the source
    script, and keeps the invalid set static over time, which is what lets the
    streaming path (:mod:`wfci.streaming`) reproduce this exactly.

``"per_frame"``
    Reserved for the general alternative (use each pixel's finite frames). Not
    implemented: it would make the valid set time-varying and break the streaming
    identity, so it needs its own design rather than a quiet default flip.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GSRConfig:
    """How to regress out the global signal.

    ``nan_policy``:
        ``"drop_pixel"`` (default) -- a pixel with any NaN frame is NaN
        everywhere, as in the MATLAB. ``"per_frame"`` is reserved; see the module
        docstring.
    ``min_variance``:
        A pixel whose global signal has (numerically) no variance has no
        identifiable slope -- ``a = 0/0``. Guarded rather than left to produce
        silent NaN/inf.
    """

    nan_policy: str = "drop_pixel"
    min_variance: float = 0.0

    def __post_init__(self) -> None:
        allowed = ("drop_pixel", "per_frame")
        if self.nan_policy not in allowed:
            raise ValueError(
                f"nan_policy={self.nan_policy!r} is not one of {allowed}."
            )
        if self.nan_policy == "per_frame":
            raise NotImplementedError(
                "nan_policy='per_frame' is reserved but not implemented: a "
                "time-varying valid set breaks the streaming GSR identity (see "
                "MERGING_PLAN.md §6.1). Use 'drop_pixel'."
            )


def global_signal(stack: np.ndarray) -> np.ndarray:
    """The spatial mean of each frame, over its FINITE pixels -> ``[time]``.

    MATLAB's ``squeeze(nanmean(nanmean(data,1),2))``, with one deliberate
    strengthening: non-finite means ``inf`` as well as ``nan``.

    ``nanmean`` ignores NaN but happily propagates ``inf``, so a single infinite
    pixel would turn the whole frame's global signal into ``inf`` and, through the
    regression, every pixel's output into garbage -- silently, since ``inf`` is not
    ``nan`` and nothing downstream tests for it. Infinities are reachable here:
    Delta F/F divides by the emo channel, so a zero in ``emo(t)`` produces one.
    MATLAB has the same hole (``~isnan(inf)`` is true, so it would feed the inf
    straight to ``fitlm``). Excluding non-finite values instead costs nothing on
    clean data -- where the two rules agree exactly -- and removes the failure
    mode on dirty data.

    Note this is a *spatial* reduction: ``g(t)`` depends only on frame ``t``, so
    it is knowable the moment that frame is read -- no lookahead. That is what
    makes the streaming path possible.
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"stack has shape {stack.shape}; expected [y, x, time].")

    finite = np.isfinite(stack)
    counts = finite.sum(axis=(0, 1))
    if (counts == 0).any():
        raise ValueError(
            "The global signal is undefined for at least one frame, i.e. that "
            "frame has no valid pixels at all. Check the brain mask and the "
            "input data."
        )
    sums = np.where(finite, stack, 0.0).sum(axis=(0, 1))
    return sums / counts


def _valid_pixels(stack: np.ndarray) -> np.ndarray:
    """``[y, x]`` bool: pixels finite in EVERY frame (MATLAB's ``~isnan(...)``)."""
    return np.isfinite(stack).all(axis=2)


def regress_global(
    stack: np.ndarray,
    cfg: GSRConfig | None = None,
    g: np.ndarray | None = None,
) -> np.ndarray:
    """Regress the global signal out of every pixel of one trial.

    Parameters
    ----------
    stack:
        ``[y, x, time]`` Delta F/F for one trial, already masked (NaN outside the
        brain). Not modified.
    cfg:
        See :class:`GSRConfig`. Defaults to ``GSRConfig()``.
    g:
        The global signal ``[time]``, if already computed. Defaults to
        :func:`global_signal` of ``stack`` -- pass it only to avoid recomputing
        the identical value.

    Returns
    -------
    ``[y, x, time]`` residuals, with dropped pixels NaN across all frames.
    """
    cfg = GSRConfig() if cfg is None else cfg
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"stack has shape {stack.shape}; expected [y, x, time].")

    g = global_signal(stack) if g is None else np.asarray(g, dtype=np.float64)
    if g.shape != (stack.shape[2],):
        raise ValueError(
            f"global signal has shape {g.shape}; expected ({stack.shape[2]},)."
        )

    valid = _valid_pixels(stack)
    if not valid.any():
        raise ValueError(
            "No pixel is finite across every frame, so nothing can be regressed. "
            "With nan_policy='drop_pixel' a single NaN frame drops the pixel "
            "entirely -- check for mid-recording NaNs (e.g. a zero in the emo "
            "channel making the ratio non-finite)."
        )

    a, b = _fit_slope_intercept(stack, g, valid, cfg.min_variance)

    # residual = p - (a*g + b), broadcasting the per-pixel line over time.
    out = stack - (a[:, :, None] * g[None, None, :] + b[:, :, None])
    # Dropped pixels are NaN for every frame, exactly as the MATLAB's else branch.
    out[~valid, :] = np.nan
    return out


def _fit_slope_intercept(
    stack: np.ndarray,
    g: np.ndarray,
    valid: np.ndarray,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel OLS of each pixel's trace on ``g`` -> ``(a, b)``, each ``[y, x]``.

    The closed form of ``fitlm(g, p)``: slope = cov(g,p)/var(g), intercept =
    mean(p) - slope*mean(g). Computed for all pixels at once.
    """
    n = g.size
    g_mean = g.mean()
    # Centred predictor: cov and var then reduce to plain dot products, and the
    # subtraction happens once instead of once per pixel.
    g_centred = g - g_mean
    g_var = float(g_centred @ g_centred)  # N * var(g), the OLS denominator

    if g_var <= min_variance:
        raise ValueError(
            f"The global signal has no variance (sum of squares {g_var:.3g}), so "
            f"no slope is identifiable. This usually means a constant or "
            f"single-frame recording."
        )

    # Dropped pixels hold NaN, which would poison these sums; zero them first and
    # restore NaN at the end (regress_global does that).
    clean = np.where(valid[:, :, None], stack, 0.0)

    p_mean = clean.sum(axis=2) / n
    # sum_t (g(t) - gbar) * p(t)  ==  N * cov(g, p): the centred predictor makes
    # the p-centring term vanish, so p never needs centring.
    cov = clean @ g_centred

    a = cov / g_var
    b = p_mean - a * g_mean

    a[~valid] = np.nan
    b[~valid] = np.nan
    return a, b
