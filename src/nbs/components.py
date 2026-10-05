"""Connected components of the supra-threshold edge graph (NBS_ALGORITHM.md steps 2-3)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SIZE_MEASURES = ("extent", "intensity", "intensity_excess")


def component_sizes(
    edge_stats: np.ndarray,
    kept: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_nodes: int,
    size_measure: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Label the kept edges by component and size each component.

    edge_stats, kept: (n_edges,); rows/cols: (n_edges,) upper-triangle node indices.
    Returns (edge_component, sizes): edge_component is (n_edges,) with -1 for
    edges that were not kept, and sizes is (n_components,). Isolated nodes carry
    no edge, so they never form a component.

    Size: extent = edge count; intensity = sum of the statistics (the manual's
    definition); intensity_excess = sum of ``stat - threshold`` (common variant).
    """
    edge_component = np.full(edge_stats.shape, -1, dtype=np.intp)
    if not kept.any():
        return edge_component, np.zeros(0)

    graph = coo_matrix(
        (np.ones(int(kept.sum())), (rows[kept], cols[kept])), shape=(n_nodes, n_nodes)
    )
    _, node_component = connected_components(graph, directed=False)

    # Renumber so only components that own at least one edge get an id 0..k-1.
    raw = node_component[rows[kept]]
    _, dense = np.unique(raw, return_inverse=True)
    edge_component[kept] = dense.ravel()

    if size_measure == "extent":
        weights = None
    elif size_measure == "intensity":
        weights = edge_stats[kept]
    elif size_measure == "intensity_excess":
        weights = edge_stats[kept] - threshold
    else:
        raise ValueError(f"size_measure must be one of {SIZE_MEASURES}, got {size_measure!r}")
    sizes = np.bincount(edge_component[kept], weights=weights).astype(np.float64)
    return edge_component, sizes


def max_component_size(
    edge_stats: np.ndarray,
    tested: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    n_nodes: int,
    size_measure: str,
    threshold: float,
) -> float:
    """Size of the largest component, 0 if no edge survives the threshold."""
    kept = tested & (edge_stats > threshold)
    _, sizes = component_sizes(edge_stats, kept, rows, cols, n_nodes, size_measure, threshold)
    return float(sizes.max()) if sizes.size else 0.0
