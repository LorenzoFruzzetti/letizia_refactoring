"""Network-Based Statistic: the whole procedure (NBS_ALGORITHM.md).

``nbs()`` runs steps 1-5; ``rethreshold()`` re-runs steps 2-5 at another primary
threshold from stored permutation statistics, without permuting again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .components import SIZE_MEASURES, component_sizes, max_component_size
from .glm import build_glm, edge_statistics, freedman_lane_parts
from .permutation import (
    block_indices,
    count_row_relabellings,
    enumerate_row_relabellings,
    enumerate_sign_flips,
    random_row_permutations,
    random_sign_flips,
)


@dataclass(frozen=True)
class Component:
    """One connected sub-network of supra-threshold edges."""

    nodes: tuple[int, ...]
    edges: np.ndarray        # (n_component_edges, 2) node pairs, i < j
    adjacency: np.ndarray    # (n_nodes, n_nodes) bool, symmetric
    size: float
    p_value: float           # FWER-corrected


@dataclass(frozen=True)
class NBSResult:
    """Everything a caller needs, plus what ``rethreshold`` needs to reuse the null."""

    stat_matrix: np.ndarray        # (n_nodes, n_nodes) symmetric, 0 on the diagonal
    components: list[Component]    # every observed component, largest first
    null_max_size: np.ndarray      # (n_null,) max component size per relabelling
    primary_threshold: float
    size_measure: str
    alpha: float
    exact: bool                    # True: every distinct relabelling enumerated
    n_relabellings_possible: int | None  # None when not a finite enumerable count
    edge_stats: np.ndarray         # (n_edges,) observed, upper-triangle order
    tested: np.ndarray             # (n_edges,) bool; False = zero in every observation
    rows: np.ndarray               # (n_edges,) upper-triangle row index
    cols: np.ndarray               # (n_edges,) upper-triangle column index
    null_stats: np.ndarray | None = field(default=None, repr=False)  # (n_null, n_edges)

    @property
    def significant(self) -> list[Component]:
        return [c for c in self.components if c.p_value < self.alpha]

    @property
    def n_nodes(self) -> int:
        return self.stat_matrix.shape[0]


def fisher_z(r: np.ndarray) -> np.ndarray:
    """``arctanh(r)`` off the diagonal; the diagonal (r = 1) is set to 0.

    Raises on |r| >= 1 off the diagonal rather than letting inf reach the GLM.
    r: (n_nodes, n_nodes, ...) with the node axes first.
    """
    r = np.array(r, dtype=np.float64)
    n_nodes = r.shape[0]
    diagonal = np.eye(n_nodes, dtype=bool).reshape((n_nodes, n_nodes) + (1,) * (r.ndim - 2))
    diagonal = np.broadcast_to(diagonal, r.shape)
    off = r[~diagonal]
    if np.any(np.abs(off) >= 1) or not np.all(np.isfinite(off)):
        raise ValueError("off-diagonal r must be finite and strictly inside (-1, 1)")
    r[diagonal] = 0.0
    return np.arctanh(r)


def upper_triangle(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n_nodes, n_nodes, n_obs) -> Y (n_obs, n_edges), rows, cols."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[0] != matrices.shape[1]:
        raise ValueError(f"matrices must be (n_nodes, n_nodes, n_obs), got {matrices.shape}")
    if not np.allclose(matrices, matrices.transpose(1, 0, 2), equal_nan=True):
        raise ValueError("every connectivity matrix must be symmetric")
    rows, cols = np.triu_indices(matrices.shape[0], k=1)
    responses = matrices[rows, cols, :].T                     # (n_obs, n_edges)
    if not np.all(np.isfinite(responses)):
        raise ValueError("matrices contain NaN/inf off the diagonal")
    return responses, rows, cols


def _edges_to_matrix(values: np.ndarray, rows, cols, n_nodes: int) -> np.ndarray:
    matrix = np.zeros((n_nodes, n_nodes), dtype=values.dtype)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def _observed_components(result_fields: dict, null_max_size: np.ndarray, exact: bool) -> list[Component]:
    """Steps 2, 3 and 5 on the observed statistics."""
    edge_stats = result_fields["edge_stats"]
    rows, cols = result_fields["rows"], result_fields["cols"]
    n_nodes = result_fields["n_nodes"]
    threshold = result_fields["primary_threshold"]
    kept = result_fields["tested"] & (edge_stats > threshold)
    edge_component, sizes = component_sizes(
        edge_stats, kept, rows, cols, n_nodes, result_fields["size_measure"], threshold
    )

    components = []
    n_null = null_max_size.size
    for i_comp in np.argsort(-sizes, kind="stable"):
        in_comp = edge_component == i_comp
        edges = np.column_stack([rows[in_comp], cols[in_comp]])
        # Exact: the observed relabelling is one of the n_null, so no +1.
        # Monte Carlo: (1 + #) / (1 + n) -- the observed counts as one draw.
        exceed = int(np.sum(null_max_size >= sizes[i_comp]))
        p_value = exceed / n_null if exact else (1 + exceed) / (1 + n_null)
        components.append(
            Component(
                nodes=tuple(int(n) for n in np.unique(edges)),
                edges=edges,
                adjacency=_edges_to_matrix(in_comp, rows, cols, n_nodes),
                size=float(sizes[i_comp]),
                p_value=float(p_value),
            )
        )
    return components


def nbs(
    matrices: np.ndarray,
    design: np.ndarray | None = None,
    contrast=None,
    *,
    test: str = "t",
    primary_threshold: float,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    size_measure: str = "extent",
    exchange_blocks: np.ndarray | None = None,
    exact: bool | None = None,
    seed: int | None = None,
    store_null_stats: bool = False,
) -> NBSResult:
    """Network-Based Statistic with FWER control at the component level.

    matrices: (n_nodes, n_nodes, n_obs) symmetric; Fisher-z them first if they are r.
    design: (n_obs, n_pred), intercept included explicitly if wanted.
    contrast: (n_pred,) for t, (n_con, n_pred) for F. The t test is one-sided
      (``c @ beta > 0``); negate the contrast for the other direction.
    test: "t" | "F" | "one_sample" (sign-flip test of mean > 0; no design/contrast).
    exchange_blocks: (n_obs,) labels; rows permute only within equal labels.
    exact: None = enumerate every distinct relabelling when there are at most
      ``n_permutations`` of them AND the permuted data is ``P @ Y`` (nuisance at
      most the intercept); True = demand it (raises if not possible); False = never.
    store_null_stats: keep the (n_null, n_edges) permuted statistics so that
      ``rethreshold`` can reuse them.
    """
    if size_measure not in SIZE_MEASURES:
        raise ValueError(f"size_measure must be one of {SIZE_MEASURES}, got {size_measure!r}")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    responses, rows, cols = upper_triangle(matrices)
    n_obs, _ = responses.shape
    n_nodes = matrices.shape[0]
    glm = build_glm(design, contrast, test, n_obs)

    # Edges that are zero in every observation are not tested at all.
    tested = np.any(responses != 0, axis=0)                     # (n_edges,)

    # Step 1 on the observed data.
    edge_stats = edge_statistics(glm, responses)
    edge_stats[~tested] = 0.0

    # Decide exact vs Monte Carlo, and build the relabelling stream.
    rng = np.random.default_rng(seed)
    if test == "one_sample":
        if exchange_blocks is not None:
            raise ValueError("exchange_blocks do not apply to the one_sample sign-flip test")
        n_possible: int | None = 2 ** n_obs
        can_enumerate = True
    else:
        blocks = block_indices(exchange_blocks, n_obs)
        # Counting distinct design-row arrangements is only the right count when
        # the permuted data is P @ Y (see permutation.py).
        can_enumerate = glm.nuisance_is_constant
        n_possible = count_row_relabellings(glm.design, blocks) if can_enumerate else None

    if exact is True and not can_enumerate:
        raise ValueError("exact enumeration needs a nuisance space of at most the intercept")
    if exact is None:
        exact = bool(can_enumerate and n_possible <= n_permutations)

    if test == "one_sample":
        relabellings = enumerate_sign_flips(n_obs) if exact else random_sign_flips(n_obs, n_permutations, rng)
        fit, residuals = np.zeros_like(responses), responses
    else:
        relabellings = (
            enumerate_row_relabellings(glm.design, blocks)
            if exact
            else random_row_permutations(blocks, n_obs, n_permutations, rng)
        )
        fit, residuals = freedman_lane_parts(glm, responses)

    # Step 4: one relabelling for all edges at once, keep the max component size.
    n_null = n_possible if exact else n_permutations
    null_max_size = np.zeros(n_null)
    null_stats = np.zeros((n_null, responses.shape[1])) if store_null_stats else None
    for i_perm, relabelling in enumerate(relabellings):
        if test == "one_sample":
            permuted = relabelling[:, None] * responses
        else:
            permuted = fit + residuals[relabelling]
        stats = edge_statistics(glm, permuted)
        stats[~tested] = 0.0
        if null_stats is not None:
            null_stats[i_perm] = stats
        null_max_size[i_perm] = max_component_size(
            stats, tested, rows, cols, n_nodes, size_measure, primary_threshold
        )

    fields = dict(
        edge_stats=edge_stats, rows=rows, cols=cols, n_nodes=n_nodes, tested=tested,
        primary_threshold=float(primary_threshold), size_measure=size_measure,
    )
    return NBSResult(
        stat_matrix=_edges_to_matrix(edge_stats, rows, cols, n_nodes),
        components=_observed_components(fields, null_max_size, exact),
        null_max_size=null_max_size,
        primary_threshold=float(primary_threshold),
        size_measure=size_measure,
        alpha=float(alpha),
        exact=exact,
        n_relabellings_possible=n_possible,
        edge_stats=edge_stats,
        tested=tested,
        rows=rows,
        cols=cols,
        null_stats=null_stats,
    )


def rethreshold(
    result: NBSResult, primary_threshold: float, *, size_measure: str | None = None
) -> NBSResult:
    """Steps 2-5 again at a new threshold, reusing the stored permutation statistics.

    Identical to calling ``nbs`` again with the same seed, without the GLM refits.
    Needs a result made with ``store_null_stats=True``.
    """
    if result.null_stats is None:
        raise ValueError("rethreshold needs a result computed with store_null_stats=True")
    size_measure = result.size_measure if size_measure is None else size_measure
    if size_measure not in SIZE_MEASURES:
        raise ValueError(f"size_measure must be one of {SIZE_MEASURES}, got {size_measure!r}")

    null_max_size = np.array([
        max_component_size(stats, result.tested, result.rows, result.cols,
                           result.n_nodes, size_measure, primary_threshold)
        for stats in result.null_stats
    ])
    fields = dict(
        edge_stats=result.edge_stats, rows=result.rows, cols=result.cols,
        n_nodes=result.n_nodes, tested=result.tested,
        primary_threshold=float(primary_threshold), size_measure=size_measure,
    )
    return NBSResult(
        stat_matrix=result.stat_matrix,
        components=_observed_components(fields, null_max_size, result.exact),
        null_max_size=null_max_size,
        primary_threshold=float(primary_threshold),
        size_measure=size_measure,
        alpha=result.alpha,
        exact=result.exact,
        n_relabellings_possible=result.n_relabellings_possible,
        edge_stats=result.edge_stats,
        tested=result.tested,
        rows=result.rows,
        cols=result.cols,
        null_stats=result.null_stats,
    )
