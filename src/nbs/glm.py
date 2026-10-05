"""Edge-wise general linear model, vectorised over edges (NBS_ALGORITHM.md step 1).

Every edge is its own response variable, but all edges share one design, so one
``pinv(X) @ Y`` fits all of them at once. ``Y`` is always ``(n_obs, n_edges)``.

The design is split, per contrast, into the part being tested and the nuisance
part (Smith et al. 2007 / Winkler et al. 2014): the nuisance space is
``X @ null_space(C)``, the part of the model the contrast does not look at. That
partition is what Freedman-Lane permutes around (see :func:`freedman_lane_parts`),
and it is invariant to how the design is coded -- ``[1, g]`` with contrast
``[0, 1]`` and cell-means ``[g1, g2]`` with contrast ``[1, -1]`` both leave the
intercept as the nuisance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

TESTS = ("t", "F", "one_sample")


@dataclass(frozen=True)
class EdgeGLM:
    """A design + contrast, pre-factorised so each permutation is two matmuls."""

    design: np.ndarray            # (n_obs, n_pred)
    contrast: np.ndarray          # (n_con, n_pred); n_con == 1 for t
    test: str                     # "t" | "F" | "one_sample"
    pinv_design: np.ndarray       # (n_pred, n_obs)
    dof: int                      # n_obs - rank(X)
    contrast_precision: np.ndarray  # (n_con, n_con) = inv(C pinv(X'X) C')
    nuisance: np.ndarray          # (n_obs, n_nuis), may have 0 columns
    pinv_nuisance: np.ndarray     # (n_nuis, n_obs)

    @property
    def n_obs(self) -> int:
        return self.design.shape[0]

    @property
    def nuisance_is_constant(self) -> bool:
        """True when Freedman-Lane reduces to permuting ``Y`` itself.

        That holds when the nuisance space is empty or only the intercept:
        the nuisance fit is then the per-edge mean, which a row permutation
        leaves unchanged, so ``fit + P @ resid == P @ Y``.
        """
        return bool(np.allclose(self.nuisance - self.nuisance.mean(axis=0), 0.0))


def build_glm(design: np.ndarray | None, contrast, test: str, n_obs: int) -> EdgeGLM:
    """Validate the design/contrast and pre-compute everything permutation-invariant.

    ``test="one_sample"`` ignores ``design``/``contrast`` (they must be None): the
    model is an intercept and the contrast is ``[1]``.
    """
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}, got {test!r}")

    if test == "one_sample":
        if design is not None or contrast is not None:
            raise ValueError("one_sample takes no design/contrast (it is the intercept-only model)")
        design = np.ones((n_obs, 1))
        contrast = np.ones((1, 1))
    else:
        if design is None or contrast is None:
            raise ValueError(f"test={test!r} needs both a design and a contrast")

    design = np.asarray(design, dtype=np.float64)
    if design.ndim != 2 or design.shape[0] != n_obs:
        raise ValueError(f"design must be (n_obs={n_obs}, n_pred), got {design.shape}")
    if not np.all(np.isfinite(design)):
        raise ValueError("design contains NaN/inf")

    contrast = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
    if contrast.shape[1] != design.shape[1]:
        raise ValueError(
            f"contrast has {contrast.shape[1]} columns, design has {design.shape[1]} predictors"
        )
    if test != "F" and contrast.shape[0] != 1:
        raise ValueError(f"a {test} test takes one contrast row, got {contrast.shape[0]}")

    pinv_design = np.linalg.pinv(design)
    # A contrast is estimable iff it lies in the row space of X: C pinv(X) X == C.
    if not np.allclose(contrast @ pinv_design @ design, contrast, atol=1e-8):
        raise ValueError("contrast is not estimable from this design (not in its row space)")

    dof = n_obs - int(np.linalg.matrix_rank(design))
    if dof <= 0:
        raise ValueError(f"no residual degrees of freedom (n_obs={n_obs}, rank(X)={n_obs - dof})")

    # C (X'X)^+ C' is the covariance of C @ beta up to sigma^2.
    contrast_cov = contrast @ np.linalg.pinv(design.T @ design) @ contrast.T
    if np.linalg.matrix_rank(contrast_cov) < contrast.shape[0]:
        raise ValueError("contrast rows are linearly dependent; drop the redundant ones")
    contrast_precision = np.linalg.inv(contrast_cov)

    # Nuisance = the part of the model space the contrast ignores.
    nuisance = design @ null_space(contrast)
    pinv_nuisance = np.linalg.pinv(nuisance) if nuisance.shape[1] else np.zeros((0, n_obs))

    return EdgeGLM(
        design=design,
        contrast=contrast,
        test=test,
        pinv_design=pinv_design,
        dof=dof,
        contrast_precision=contrast_precision,
        nuisance=nuisance,
        pinv_nuisance=pinv_nuisance,
    )


def edge_statistics(glm: EdgeGLM, responses: np.ndarray) -> np.ndarray:
    """t (one-sided, as signed t) or F per edge; NaN (constant edges) -> 0.

    responses: (n_obs, n_edges). Returns (n_edges,).
    """
    beta = glm.pinv_design @ responses                         # (n_pred, n_edges)
    residuals = responses - glm.design @ beta                  # (n_obs, n_edges)
    sigma2 = np.einsum("oe,oe->e", residuals, residuals) / glm.dof  # (n_edges,)
    effect = glm.contrast @ beta                               # (n_con, n_edges)

    # A constant edge is fitted exactly, but roundoff leaves sigma2 ~ 1e-32 * Y^2
    # instead of 0, and effect/sqrt(sigma2) then turns noise into a large "t".
    # Treat residual variance below 1e-20 of the edge's mean square as exactly 0.
    mean_square = np.einsum("oe,oe->e", responses, responses) / glm.n_obs
    sigma2[sigma2 <= 1e-20 * mean_square] = 0.0

    # 0/0 on constant edges is expected and mapped to 0 below, as NBS does.
    with np.errstate(divide="ignore", invalid="ignore"):
        if glm.test == "F":
            n_con = glm.contrast.shape[0]
            quad = np.einsum("ce,ce->e", effect, glm.contrast_precision @ effect)
            stat = quad / n_con / sigma2
        else:
            # For one row, contrast_precision is 1 / (c pinv(X'X) c').
            stat = effect[0] * np.sqrt(glm.contrast_precision[0, 0] / sigma2)

    stat[~np.isfinite(stat)] = 0.0
    return stat


def freedman_lane_parts(glm: EdgeGLM, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``Y`` into nuisance fit and nuisance residuals.

    A permuted dataset is ``fit + residuals[perm]``. With no nuisance columns the
    fit is 0 and this is plain permutation of ``Y``.
    Returns (fit, residuals), both (n_obs, n_edges).
    """
    fit = glm.nuisance @ (glm.pinv_nuisance @ responses)
    return fit, responses - fit
