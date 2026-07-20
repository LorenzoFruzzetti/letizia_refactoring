"""The significance/figure layer: masking, per-node reductions, headless figures.

`wfci.significance` ingests an adjacency (significant edges, e.g. from NBS) and a
value matrix (e.g. a group DIFF) and draws the network + bar plots. It never
computes the network statistic itself -- that is a non-goal. These tests check the
maths of the masking and per-node reductions exactly, and that every figure
renders headless without error.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wfci import (  # noqa: E402
    circular_layout,
    count_significant_edges,
    hemispheric_layout,
    mask_by_adjacency,
    network_figure,
    node_strength,
    significance_barplot,
)


def _diff(n: int = 4) -> np.ndarray:
    d = np.array([
        [0.0, 0.5, 0.0, -0.3],
        [0.5, 0.0, 0.2, 0.0],
        [0.0, 0.2, 0.0, 0.4],
        [-0.3, 0.0, 0.4, 0.0],
    ])
    return d


def _adjacency() -> np.ndarray:
    # Mark edges (0,1), (0,3), (2,3) as significant.
    a = np.zeros((4, 4))
    for i, j in [(0, 1), (0, 3), (2, 3)]:
        a[i, j] = a[j, i] = 1.0
    return a


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def test_mask_keeps_only_significant_edges():
    masked = mask_by_adjacency(_diff(), _adjacency())

    # Kept edges carry their DIFF value...
    assert masked[0, 1] == 0.5
    assert masked[0, 3] == -0.3
    assert masked[2, 3] == 0.4
    # ...and everything else is zero, including the significant-but-absent (1,2).
    assert masked[1, 2] == 0.0
    assert masked[0, 2] == 0.0


def test_mask_result_is_symmetric():
    masked = mask_by_adjacency(_diff(), _adjacency())

    np.testing.assert_array_equal(masked, masked.T)


def test_mask_treats_nan_as_zero():
    diff = _diff()
    diff[0, 1] = diff[1, 0] = np.nan
    adj = _adjacency()

    masked = mask_by_adjacency(diff, adj)

    assert masked[0, 1] == 0.0  # NaN edge does not propagate


def test_mask_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        mask_by_adjacency(np.zeros((4, 4)), np.zeros((3, 3)))


def test_a_boolean_adjacency_works():
    masked = mask_by_adjacency(_diff(), _adjacency().astype(bool))

    assert masked[0, 1] == 0.5


# ---------------------------------------------------------------------------
# Per-node reductions
# ---------------------------------------------------------------------------
def test_node_strength_is_a_per_node_vector():
    """Not a matrix -- the MATLAB step-6 quirk this deliberately does not repeat."""
    masked = mask_by_adjacency(_diff(), _adjacency())

    strength = node_strength(masked)

    assert strength.shape == (4,)
    # Node 0 has edges to 1 (0.5) and 3 (0.3 abs) -> 0.8.
    assert strength[0] == pytest.approx(0.8)
    assert strength[1] == pytest.approx(0.5)


def test_count_significant_edges_by_sign():
    masked = mask_by_adjacency(_diff(), _adjacency())

    neg = count_significant_edges(masked, "negative")
    pos = count_significant_edges(masked, "positive")

    # Node 0: one negative edge (to 3, -0.3), one positive (to 1, 0.5).
    assert neg[0] == 1 and pos[0] == 1
    # Node 2: only the positive edge to 3 (0.4).
    assert neg[2] == 0 and pos[2] == 1
    assert count_significant_edges(masked, "both")[0] == 2


def test_count_rejects_a_bad_sign():
    with pytest.raises(ValueError, match="sign"):
        count_significant_edges(np.zeros((3, 3)), "sideways")


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def test_circular_layout_shape_and_radius():
    pos = circular_layout(8, radius=2.0)

    assert pos.shape == (8, 2)
    np.testing.assert_allclose(np.hypot(pos[:, 0], pos[:, 1]), 2.0)


def test_hemispheric_layout_splits_left_and_right():
    pos = hemispheric_layout(11, 11)

    assert pos.shape == (22, 2)
    assert (pos[:11, 0] < 0).all(), "left block should be on the left"
    assert (pos[11:, 0] > 0).all(), "right block should be on the right"


# ---------------------------------------------------------------------------
# Figures render headless
# ---------------------------------------------------------------------------
def test_network_figure_renders():
    masked = mask_by_adjacency(_diff(), _adjacency())

    ax = network_figure(masked, labels=["a", "b", "c", "d"])

    assert ax is not None
    assert ax.collections, "no nodes were drawn"
    plt.close("all")


def test_network_figure_uses_supplied_positions():
    masked = mask_by_adjacency(_diff(), _adjacency())
    positions = hemispheric_layout(2, 2)

    ax = network_figure(masked, node_positions=positions)

    pts = ax.collections[0].get_offsets()
    np.testing.assert_allclose(np.asarray(pts), positions)
    plt.close("all")


def test_network_figure_rejects_wrong_sized_positions():
    masked = mask_by_adjacency(_diff(), _adjacency())

    with pytest.raises(ValueError, match=r"\[4, 2\]"):
        network_figure(masked, node_positions=np.zeros((3, 2)))
    plt.close("all")


def test_a_node_with_no_significant_edges_does_not_warn(recwarn):
    """An isolated node (all-zero row) must colour to 0, silently."""
    masked = np.zeros((4, 4))
    masked[0, 1] = masked[1, 0] = 0.5  # only nodes 0,1 connected

    network_figure(masked)

    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
    plt.close("all")


def test_barplot_renders_both_signs():
    masked = mask_by_adjacency(_diff(), _adjacency())

    ax = significance_barplot(masked, ["a", "b", "c", "d"], sign="negative")

    assert len(ax.patches) == 4  # one bar per node
    plt.close("all")


# ---------------------------------------------------------------------------
# The worked study script end to end (the Phase 6 acceptance)
# ---------------------------------------------------------------------------
def test_study_script_contrasts_match_a_hand_computation():
    """compute_contrasts on the demo cohort must equal a direct numpy average."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import healthy_vs_disease_day4 as study

    table = study.demo_cohort(seed=0)
    contrasts = study.compute_contrasts(table)

    healthy = [r.matrix for r in table if r.metadata["group"] == "healthy"]
    disease = [r.matrix for r in table if r.metadata["group"] == "disease"]
    expected_diff = np.mean(healthy, axis=0) - np.mean(disease, axis=0)

    np.testing.assert_allclose(contrasts["diff"].matrix, expected_diff)

    # The injected signal (0.4 per group, opposite signs) dominates DIFF at its
    # edges: ~0.8 plus the residual of the random base, which does not fully
    # cancel over 5+5 animals. So it is large and positive, and clearly the
    # strongest edge -- which is what demo_adjacency relies on to find it.
    diff = contrasts["diff"].matrix
    signal_edges = [(0, 11), (2, 13), (4, 15), (8, 19)]
    for i, j in signal_edges:
        assert diff[i, j] > 0.5, f"injected signal at ({i},{j}) lost: {diff[i, j]:.3f}"
    off_diag = diff[~np.eye(diff.shape[0], dtype=bool)]
    assert diff[0, 11] >= np.percentile(off_diag, 95), "signal edge is not among the strongest"


def test_study_script_writes_three_figures(tmp_path):
    """The whole study runs headless and produces its figures (Phase 6 acceptance)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import healthy_vs_disease_day4 as study

    cfg = dict(study.RUN_CONFIG)
    cfg["output_dir"] = str(tmp_path)
    original = study.RUN_CONFIG
    study.RUN_CONFIG = cfg
    try:
        study.main()
    finally:
        study.RUN_CONFIG = original

    for name in ("diff_matrix.png", "network.png", "barplots.png"):
        assert (tmp_path / name).exists(), f"{name} was not written"
        assert (tmp_path / name).stat().st_size > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
