"""GSR: the vectorised closed form must equal an honest per-pixel OLS.

MERGING_PLAN.md Phase 3 replaces MATLAB's 16 384 ``fitlm`` calls per trial with a
closed-form vectorised OLS. That is a legitimate rewrite only because ``fitlm``
with one predictor and an intercept *is* OLS -- so the way to earn it is to check
against a reference that fits each pixel separately and naively, the way the
MATLAB does.

That reference is ``np.linalg.lstsq`` per pixel with an explicit design matrix
``[g, 1]``: no shared algebra with the implementation under test, so an error in
the closed form cannot cancel out.

Scope (MERGING_PLAN.md P4/Phase 8): this validates the VECTORISATION, not the
TRANSCRIPTION. It proves the Python computes OLS correctly; it does not prove
that OLS-on-this-input is what the MATLAB script would have produced on real
data, because there is no MATLAB reference for the cortical path.
"""

from __future__ import annotations

import numpy as np
import pytest

from wfci.gsr import GSRConfig, global_signal, regress_global
from wfci.mask import apply_mask

SHAPE = (12, 10)
N_TIME = 40


def _stack(seed: int = 0, shape=SHAPE, n_time=N_TIME) -> np.ndarray:
    """A stack with a real shared global component, so GSR has something to remove."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 1.0, n_time)                       # the global signal
    gain = rng.uniform(0.5, 2.0, shape)                         # per-pixel coupling
    noise = rng.normal(0.0, 0.3, shape + (n_time,))
    offset = rng.uniform(-5.0, 5.0, shape)
    return gain[:, :, None] * common[None, None, :] + noise + offset[:, :, None]


def _mask(shape=SHAPE) -> np.ndarray:
    m = np.ones(shape, dtype=np.float64)
    m[:, -3:] = 0.0  # a background strip outside the brain
    return m


def _lstsq_reference(stack: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-pixel OLS the naive way -- one explicit fit per pixel, as MATLAB does.

    Design matrix [g, 1] == fitlm(g, p) with an intercept. Pixels with any NaN
    frame are dropped entirely, reproducing MATLAB's ``if ~isnan(data(row,col,:))``
    (an ``if`` over an array is true only when every element is).
    """
    y, x, t = stack.shape
    design = np.column_stack([g, np.ones(t)])
    out = np.full_like(stack, np.nan)
    for i in range(y):
        for j in range(x):
            pixel = stack[i, j, :]
            if not np.isfinite(pixel).all():
                continue  # dropped: NaN across all frames
            coef, *_ = np.linalg.lstsq(design, pixel, rcond=None)
            out[i, j, :] = pixel - (coef[0] * g + coef[1])
    return out


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------
def test_vectorised_gsr_equals_per_pixel_lstsq():
    """The whole point of Phase 3: same numbers, none of the 16 384 fits."""
    stack = apply_mask(_stack(), _mask())
    g = global_signal(stack)

    got = regress_global(stack)
    ref = _lstsq_reference(stack, g)

    finite = np.isfinite(ref)
    diff = np.abs(got[finite] - ref[finite]).max()
    print(f"  max|vectorised - per-pixel lstsq| = {diff:.3e}")
    assert diff < 1e-12, f"vectorised OLS diverges from a per-pixel fit: {diff:.3e}"
    # The NaN pattern must match too, not just the numbers.
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))


def test_vectorised_gsr_equals_per_pixel_lstsq_without_a_mask():
    """No masking: every pixel valid, so nothing hides behind NaN bookkeeping."""
    stack = _stack(seed=5)

    got = regress_global(stack)
    ref = _lstsq_reference(stack, global_signal(stack))

    diff = np.abs(got - ref).max()
    print(f"  max|vectorised - per-pixel lstsq| (unmasked) = {diff:.3e}")
    assert diff < 1e-12
    assert np.isfinite(got).all()


# ---------------------------------------------------------------------------
# The properties GSR is supposed to have
# ---------------------------------------------------------------------------
def test_residuals_are_orthogonal_to_the_global_signal():
    """After regressing g out, no pixel correlates with g -- that IS the job.

    An independent check of the result rather than of the arithmetic: OLS
    residuals are orthogonal to the predictor by construction, so if this fails
    the fit is wrong no matter what any reference says.
    """
    stack = apply_mask(_stack(seed=1), _mask())
    g = global_signal(stack)
    g_centred = g - g.mean()

    out = regress_global(stack)

    valid = np.isfinite(out).all(axis=2)
    residuals = out[valid]                       # [n_valid, time]
    projection = residuals @ g_centred           # dot with the centred predictor
    assert np.abs(projection).max() < 1e-9, "residuals still carry the global signal"
    # And the intercept is removed, so residuals are zero-mean in time.
    assert np.abs(residuals.mean(axis=1)).max() < 1e-9


def test_a_pixel_that_is_pure_global_signal_regresses_to_zero():
    """The clearest case: p(t) = 3*g(t) + 7 must leave nothing behind."""
    rng = np.random.default_rng(2)
    common = rng.normal(0.0, 1.0, N_TIME)
    stack = np.empty((2, 2, N_TIME))
    stack[:] = common  # every pixel identical -> global signal == common
    stack[0, 0, :] = 3.0 * common + 7.0

    out = regress_global(stack)

    assert np.abs(out).max() < 1e-9


def test_gsr_leaves_a_pixel_uncorrelated_with_g_almost_alone():
    """A pixel independent of g keeps its own signal (minus its mean)."""
    rng = np.random.default_rng(7)
    common = rng.normal(0.0, 1.0, N_TIME)
    private = rng.normal(0.0, 1.0, N_TIME)
    stack = np.empty((4, 4, N_TIME))
    stack[:] = common
    stack[0, 0, :] = private + common  # this pixel has BOTH

    out = regress_global(stack)

    kept = out[0, 0, :]
    # The private part survives; correlation with the original private signal is
    # high even though the shared part is gone.
    r = np.corrcoef(kept, private - private.mean())[0, 1]
    assert r > 0.9, f"the pixel's own signal was destroyed (r={r:.3f})"


def test_global_signal_is_the_spatial_mean_of_brain_pixels():
    """MATLAB's nanmean(nanmean(data,1),2), i.e. mask-aware."""
    stack = apply_mask(_stack(seed=3), _mask())

    g = global_signal(stack)

    assert g.shape == (N_TIME,)
    # Only the unmasked columns contribute.
    expected = stack[:, :-3, :].mean(axis=(0, 1))
    np.testing.assert_allclose(g, expected)


def test_global_signal_of_one_frame_uses_only_that_frame():
    """g(t) is a purely spatial reduction -- no lookahead.

    This is the property the streaming path is built on (MERGING_PLAN.md §2.2):
    g(t) is known the moment frame t is read.
    """
    stack = _stack(seed=8)

    g = global_signal(stack)

    for t in (0, 17, N_TIME - 1):
        np.testing.assert_allclose(g[t], stack[:, :, t].mean())


# ---------------------------------------------------------------------------
# NaN policy -- the MATLAB's drop-the-whole-pixel rule
# ---------------------------------------------------------------------------
def test_drop_pixel_drops_a_pixel_with_even_one_nan_frame():
    """MATLAB's `if ~isnan(data(row,col,:))` is ALL-frames-finite, not any."""
    stack = _stack(seed=4)
    stack[3, 3, 10] = np.nan  # a single bad frame

    out = regress_global(stack)

    assert np.isnan(out[3, 3, :]).all(), "one NaN frame must drop the whole pixel"
    # ...and only that pixel.
    others = np.ones(SHAPE, dtype=bool)
    others[3, 3] = False
    assert np.isfinite(out[others, :]).all()


def test_dropped_pixels_do_not_perturb_their_neighbours():
    """NaN must not leak into other pixels' fits through the vectorised sums.

    The implementation zero-fills dropped pixels to keep the array arithmetic
    finite; if that zero ever reached a *kept* pixel's fit, this would catch it.
    """
    stack = _stack(seed=6)
    clean = regress_global(stack.copy())

    holed = stack.copy()
    holed[5, 5, 3] = np.nan
    out = regress_global(holed)

    keep = np.ones(SHAPE, dtype=bool)
    keep[5, 5] = False
    # Every other pixel's residual is bit-for-bit what it was without the hole:
    # g is unchanged here because it is computed with nanmean over... (it is not:
    # the holed frame loses one pixel from its spatial mean). So compare against a
    # reference that sees the same g.
    ref = _lstsq_reference(holed, global_signal(holed))
    np.testing.assert_allclose(out[keep, :], ref[keep, :], atol=1e-12)
    assert np.isfinite(clean).all()


def test_per_frame_policy_is_refused_rather_than_silently_approximated():
    """P4: the option exists, but an unimplemented policy must not fall back."""
    with pytest.raises(NotImplementedError, match="per_frame"):
        GSRConfig(nan_policy="per_frame")


def test_an_unknown_nan_policy_is_rejected():
    with pytest.raises(ValueError, match="nan_policy"):
        GSRConfig(nan_policy="whatever")


# ---------------------------------------------------------------------------
# Degenerate inputs fail loudly instead of returning NaN
# ---------------------------------------------------------------------------
def test_a_constant_global_signal_is_rejected():
    """var(g) == 0 makes the slope 0/0; say so rather than emit NaN."""
    stack = np.ones((4, 4, 10))

    with pytest.raises(ValueError, match="no variance"):
        regress_global(stack)


def test_an_all_nan_frame_is_rejected():
    """A frame with no valid pixels makes g(t) NaN, which would poison every fit."""
    stack = _stack(seed=9)
    stack[:, :, 5] = np.nan

    with pytest.raises(ValueError, match="no valid pixels"):
        regress_global(stack)


def test_a_fully_nan_stack_is_rejected():
    stack = np.full((4, 4, 10), np.nan)

    with pytest.raises(ValueError, match="no valid pixels|not finite"):
        regress_global(stack)


def test_regress_global_does_not_modify_its_input():
    stack = apply_mask(_stack(), _mask())
    before = stack.copy()

    regress_global(stack)

    np.testing.assert_array_equal(np.isnan(stack), np.isnan(before))
    np.testing.assert_array_equal(stack[np.isfinite(stack)], before[np.isfinite(before)])


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match=r"expected \[y, x, time\]"):
        regress_global(np.ones((4, 4, 10, 2)))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
