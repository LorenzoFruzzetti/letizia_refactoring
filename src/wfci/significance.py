"""Generic significance visualisation -- mask a matrix by an adjacency, draw the network.

LIBRARY, not policy (MERGING_PLAN.md P2). This ingests an **adjacency matrix** of
significant connections (e.g. exported from NBS, or from a future permutation
test) and a **value matrix** (e.g. a group difference ``DIFF``), and produces:

  * the value matrix masked to only its significant entries,
  * a node-link **network figure**,
  * per-node **bar plots** of how many significant connections each region has.

Colour scales and node positions are **arguments with sensible defaults** -- they
are study choices, not library constants. A study passes its own anatomical node
coordinates and, if it wants, its own colour scheme.

**NBS itself stays out** (MERGING_PLAN.md §4.3, an explicit non-goal). Computing
the network-based statistic is a separate project; this consumes its result. The
adjacency can come from anywhere that yields an ``[n, n]`` matrix whose non-zero
entries mark significant edges.

This is the layer MATLAB step (6) corresponds to. Two documented quirks of that
script are deliberately **not** reproduced:

  * **Node sizes.** There, ``degrees = connectivity_matrix`` (a whole 2-D matrix)
    is divided by its max and used as ``node_sizes`` -- a per-node quantity taken
    from an ``[n, n]`` array. Here node size is an honest per-node vector: each
    node's weighted degree (:func:`node_strength`).
  * **Edge colours.** There, ``COLOR_EDGE`` is a variable reused between the HYPER
    and HYPO figure cells, so a shorter edge list can read stale colours left over
    from the previous figure. Here every edge colour is computed locally, from the
    edge being drawn.
"""

from __future__ import annotations

import numpy as np


def mask_by_adjacency(
    value_matrix: np.ndarray,
    adjacency: np.ndarray,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """Keep ``value_matrix`` only where ``adjacency`` marks a significant edge.

    Reproduces MATLAB's ``matrix_onlysign`` loop followed by
    ``triu(...) + triu(...,1)'``: entries where ``adjacency > 0`` keep their value,
    everything else becomes 0, and (with ``symmetrize``) the upper triangle is
    mirrored so the result is symmetric.

    Parameters
    ----------
    value_matrix:
        ``[n, n]`` values to display (e.g. a group ``DIFF``).
    adjacency:
        ``[n, n]`` significance mask; any entry ``> 0`` is a kept edge. A boolean
        matrix works too.
    symmetrize:
        Mirror the upper triangle onto the lower, as the MATLAB does, so an
        adjacency that only filled one triangle still yields a symmetric network.
    """
    value_matrix = np.asarray(value_matrix, dtype=np.float64)
    adjacency = np.asarray(adjacency)
    if value_matrix.shape != adjacency.shape:
        raise ValueError(
            f"value_matrix {value_matrix.shape} and adjacency {adjacency.shape} "
            f"must have the same shape."
        )
    if value_matrix.ndim != 2 or value_matrix.shape[0] != value_matrix.shape[1]:
        raise ValueError(f"expected a square [n, n] matrix; got {value_matrix.shape}.")

    masked = np.where(adjacency > 0, value_matrix, 0.0)
    masked = np.nan_to_num(masked, nan=0.0)
    if symmetrize:
        upper = np.triu(masked)
        masked = upper + np.triu(masked, 1).T
    return masked


def node_strength(masked: np.ndarray) -> np.ndarray:
    """Per-node weighted degree: ``sum_j |masked[i, j]|`` -> ``[n]``.

    An honest per-node vector (see the module note on the MATLAB's matrix-valued
    ``node_sizes``). Used to scale node markers: a region with many strong
    significant connections draws larger.
    """
    return np.abs(np.asarray(masked, dtype=np.float64)).sum(axis=1)


def count_significant_edges(masked: np.ndarray, sign: str = "negative") -> np.ndarray:
    """Per-node count of significant edges of one sign -> ``[n]``.

    The input to the step-6 bar plots: for each region, how many of its
    significant connections are increases (``"positive"``) or decreases
    (``"negative"``). ``sign="both"`` counts all non-zero edges.
    """
    masked = np.asarray(masked, dtype=np.float64)
    if sign == "negative":
        hits = masked < 0
    elif sign == "positive":
        hits = masked > 0
    elif sign == "both":
        hits = masked != 0
    else:
        raise ValueError(f"sign must be 'negative', 'positive' or 'both'; got {sign!r}.")
    return hits.sum(axis=1).astype(np.float64)


def circular_layout(n: int, radius: float = 1.0) -> np.ndarray:
    """``[n, 2]`` node positions evenly spaced on a circle.

    A generic default when a study has no anatomical coordinates. Nodes go
    clockwise from the top, so a left/right-blocked label order (all left, then all
    right) lands the two hemispheres on opposite sides.
    """
    angles = np.pi / 2 - np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def hemispheric_layout(n_left: int, n_right: int) -> np.ndarray:
    """``[n_left + n_right, 2]`` positions: left column, then right column.

    Matches the ``[all-left, all-right]`` block order the cortical atlas uses, so
    the two hemispheres sit as two vertical columns -- a readable default for a
    left/right study without real coordinates.
    """
    def _column(count: int, x: float) -> np.ndarray:
        ys = np.linspace(1.0, -1.0, count) if count > 1 else np.array([0.0])
        return np.column_stack([np.full(count, x), ys])

    return np.vstack([_column(n_left, -1.0), _column(n_right, 1.0)])


def _diverging_norm(values: np.ndarray):
    """A matplotlib norm centred on 0 and symmetric about it.

    A group difference is signed and its meaning is symmetric (an increase of
    +0.2 and a decrease of -0.2 are equal and opposite), so the colour scale must
    be too, or the zero point drifts and the colours mislead.
    """
    import matplotlib.colors as mcolors

    limit = float(np.nanmax(np.abs(values))) if np.any(values) else 1.0
    limit = limit or 1.0
    return mcolors.Normalize(vmin=-limit, vmax=limit)


def network_figure(
    masked: np.ndarray,
    *,
    node_positions: np.ndarray | None = None,
    node_values: np.ndarray | None = None,
    labels: list[str] | None = None,
    cmap: str = "RdBu_r",
    norm=None,
    node_size_range: tuple[float, float] = (60.0, 600.0),
    edge_width_range: tuple[float, float] = (0.5, 6.0),
    annotate: bool = True,
    ax=None,
):
    """Draw the significant-connection network.

    Parameters
    ----------
    masked:
        ``[n, n]`` masked value matrix (see :func:`mask_by_adjacency`). Non-zero
        entries are the edges to draw; their value sets edge colour and width.
    node_positions:
        ``[n, 2]`` coordinates. Defaults to :func:`circular_layout`. A study passes
        anatomical (e.g. MNI) coordinates here -- exactly the ``node_positions`` the
        MATLAB loaded from a file.
    node_values:
        ``[n]`` value colouring each node. Defaults to each node's mean over its
        row of ``masked``. Pass ``DIFF.mean(axis=1)`` to match the MATLAB, which
        colours nodes by their mean over the *full* difference row.
    labels:
        Optional ``[n]`` names, annotated next to each node when ``annotate``.
    cmap, norm:
        Colour scale for both nodes and edges. ``norm`` defaults to a symmetric,
        zero-centred diverging norm -- appropriate for a signed difference. Colour
        thresholds are therefore an argument (change ``cmap``/``norm``), not the
        library's fixed ``> 0.6 -> dark red`` ladder.
    node_size_range, edge_width_range:
        ``(min, max)`` marker area / line width; scaled by node strength and
        ``|edge value|`` respectively.
    annotate, ax:
        Draw labels; target Axes (created if None).

    Returns
    -------
    The matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    masked = np.asarray(masked, dtype=np.float64)
    n = masked.shape[0]
    if node_positions is None:
        node_positions = circular_layout(n)
    node_positions = np.asarray(node_positions, dtype=np.float64)
    if node_positions.shape != (n, 2):
        raise ValueError(
            f"node_positions must be [{n}, 2] to match the {n}-node matrix; got "
            f"{node_positions.shape}."
        )
    if node_values is None:
        # Each node coloured by the mean over its significant edges; a node with
        # none is left at 0 (an all-NaN row -> nanmean warns, hence the guard).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows = np.where(masked != 0, masked, np.nan)
            node_values = np.nan_to_num(np.nanmean(rows, axis=1), nan=0.0)
    node_values = np.asarray(node_values, dtype=np.float64)

    cmap_obj = plt.get_cmap(cmap)
    edge_norm = norm if norm is not None else _diverging_norm(masked[masked != 0] if np.any(masked) else masked)
    node_norm = norm if norm is not None else _diverging_norm(node_values)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # -- edges (drawn first, so nodes sit on top) --------------------------
    # Only the upper triangle, so each undirected edge is drawn once.
    iu, ju = np.triu_indices(n, k=1)
    edge_vals = masked[iu, ju]
    nonzero = edge_vals != 0
    if nonzero.any():
        max_abs = float(np.abs(edge_vals[nonzero]).max()) or 1.0
        lw_lo, lw_hi = edge_width_range
        for i, j, v in zip(iu[nonzero], ju[nonzero], edge_vals[nonzero]):
            width = lw_lo + (lw_hi - lw_lo) * (abs(v) / max_abs)
            # Colour computed HERE, from this edge -- never a reused variable.
            ax.plot(
                [node_positions[i, 0], node_positions[j, 0]],
                [node_positions[i, 1], node_positions[j, 1]],
                color=cmap_obj(edge_norm(v)),
                linewidth=width,
                zorder=1,
                solid_capstyle="round",
            )

    # -- nodes --------------------------------------------------------------
    strength = node_strength(masked)
    s_lo, s_hi = node_size_range
    if strength.max() > 0:
        sizes = s_lo + (s_hi - s_lo) * (strength / strength.max())
    else:
        sizes = np.full(n, s_lo)
    ax.scatter(
        node_positions[:, 0], node_positions[:, 1],
        s=sizes, c=node_values, cmap=cmap_obj, norm=node_norm,
        edgecolors="none", zorder=2,
    )

    if annotate and labels is not None:
        for (x, y), name in zip(node_positions, labels):
            ax.annotate(name, (x, y), fontsize=7, ha="center", va="center", zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def significance_barplot(
    masked: np.ndarray,
    labels: list[str] | None = None,
    *,
    sign: str = "negative",
    ax=None,
    color=None,
):
    """Horizontal bar plot of per-node significant-edge counts (MATLAB step-6 bars).

    One bar per region, its length the number of significant connections of the
    given ``sign``. ``sign="negative"`` reproduces the ``NEG_sums`` / ``barh`` plot;
    ``"positive"`` the ``POS_sums`` one. Regions read top-to-bottom in label order.
    """
    import matplotlib.pyplot as plt

    counts = count_significant_edges(masked, sign)
    n = counts.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    y = np.arange(n)
    ax.barh(y, counts, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels if labels is not None else [str(i) for i in range(n)],
                       fontsize=7)
    ax.invert_yaxis()  # first region at the top, as the MATLAB's YDir reverse
    ax.set_xlabel(f"significant {sign} connections")
    return ax
