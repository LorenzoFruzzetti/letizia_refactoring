"""The `nbs` package against NBS_ALGORITHM.md's validation checklist.

Checklist -> test:
  null FWER ~ alpha ................ test_null_family_wise_error_rate
  planted effect recovered ......... test_planted_component_is_recovered
  t == scipy.ttest_ind ............. test_t_matches_scipy_ttest_ind
  Freedman-Lane w/o nuisance ....... test_freedman_lane_with_intercept_equals_plain_permutation
  exact vs Monte Carlo ............. test_exact_and_monte_carlo_agree
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from nbs import build_glm, edge_statistics, fisher_z, nbs, rethreshold, upper_triangle
from nbs.permutation import (
    block_indices,
    count_row_relabellings,
    enumerate_row_relabellings,
    random_row_permutations,
)


def random_matrices(n_nodes: int, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """(n_nodes, n_nodes, n_obs) symmetric Gaussian matrices, zero diagonal."""
    raw = rng.standard_normal((n_nodes, n_nodes, n_obs))
    sym = (raw + raw.transpose(1, 0, 2)) / np.sqrt(2)
    sym[np.arange(n_nodes), np.arange(n_nodes), :] = 0.0
    return sym


def two_group_design(n1: int, n2: int) -> np.ndarray:
    """Intercept + group-1 indicator, (n1 + n2, 2)."""
    group = np.r_[np.ones(n1), np.zeros(n2)]
    return np.column_stack([np.ones(n1 + n2), group])


# ---------------------------------------------------------------- edge statistics

def test_t_matches_scipy_ttest_ind():
    rng = np.random.default_rng(0)
    n1, n2 = 7, 9
    matrices = random_matrices(6, n1 + n2, rng)
    responses, _, _ = upper_triangle(matrices)
    glm = build_glm(two_group_design(n1, n2), [0, 1], "t", n1 + n2)
    expected = stats.ttest_ind(responses[:n1], responses[n1:], axis=0).statistic
    np.testing.assert_allclose(edge_statistics(glm, responses), expected, rtol=1e-10)


def test_one_sample_matches_scipy_ttest_1samp():
    rng = np.random.default_rng(1)
    responses = rng.standard_normal((12, 15)) + 0.3
    glm = build_glm(None, None, "one_sample", 12)
    expected = stats.ttest_1samp(responses, 0.0, axis=0).statistic
    np.testing.assert_allclose(edge_statistics(glm, responses), expected, rtol=1e-10)


def test_f_matches_scipy_f_oneway_and_equals_t_squared():
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1, 2], [5, 6, 7])
    responses = rng.standard_normal((18, 10))
    cell_means = np.eye(3)[labels]                              # (18, 3)
    glm_f = build_glm(cell_means, [[1, -1, 0], [0, 1, -1]], "F", 18)
    expected = stats.f_oneway(*(responses[labels == g] for g in range(3)), axis=0).statistic
    np.testing.assert_allclose(edge_statistics(glm_f, responses), expected, rtol=1e-10)

    # A one-row F is t squared.
    glm_t = build_glm(cell_means, [1, -1, 0], "t", 18)
    glm_f1 = build_glm(cell_means, [[1, -1, 0]], "F", 18)
    np.testing.assert_allclose(
        edge_statistics(glm_f1, responses), edge_statistics(glm_t, responses) ** 2, rtol=1e-10
    )


def test_constant_edge_statistic_is_zero():
    responses = np.ones((10, 3))
    responses[:, 1] = np.arange(10)
    glm = build_glm(two_group_design(5, 5), [0, 1], "t", 10)
    assert edge_statistics(glm, responses)[0] == 0.0


def test_non_estimable_contrast_raises():
    group = np.r_[np.ones(4), np.zeros(4)]
    design = np.column_stack([np.ones(8), group, 1 - group])    # rank 2 of 3 columns
    with pytest.raises(ValueError, match="estimable"):
        build_glm(design, [0, 1, 0], "t", 8)


def test_fisher_z_zeroes_diagonal_and_rejects_unit_r():
    r = np.array([[1.0, 0.5], [0.5, 1.0]])[:, :, None]
    z = fisher_z(r)
    assert z[0, 0, 0] == 0.0 and z[0, 1, 0] == pytest.approx(np.arctanh(0.5))
    with pytest.raises(ValueError):
        fisher_z(np.array([[1.0, 1.0], [1.0, 1.0]]))


# ---------------------------------------------------------------- permutations

def test_relabelling_count_is_binomial_and_multiplies_over_blocks():
    design = two_group_design(5, 7)
    assert count_row_relabellings(design, block_indices(None, 12)) == math.comb(12, 5)
    blocks = np.r_[np.zeros(6), np.ones(6)]
    # Block 0 holds 5 of group 1 and 1 of group 0; block 1 holds 0 and 6.
    assert count_row_relabellings(design, block_indices(blocks, 12)) == math.comb(6, 5) * 1


def test_enumeration_is_complete_and_distinct():
    design = two_group_design(3, 4)
    blocks = block_indices(None, 7)
    arrangements = {
        tuple(design[np.argsort(perm), 1]) for perm in enumerate_row_relabellings(design, blocks)
    }
    assert len(arrangements) == math.comb(7, 3)


def test_random_permutations_stay_inside_exchange_blocks():
    blocks_labels = np.repeat([0, 1, 2], 4)
    blocks = block_indices(blocks_labels, 12)
    for perm in random_row_permutations(blocks, 12, 50, np.random.default_rng(3)):
        np.testing.assert_array_equal(blocks_labels[perm], blocks_labels)


# ---------------------------------------------------------------- full procedure

def test_planted_component_is_recovered():
    rng = np.random.default_rng(4)
    n_nodes, n1, n2 = 12, 15, 15
    matrices = random_matrices(n_nodes, n1 + n2, rng)
    planted = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]   # i < j, as components report
    for i, j in planted:
        matrices[i, j, :n1] += 2.0
        matrices[j, i, :n1] += 2.0

    result = nbs(matrices, two_group_design(n1, n2), [0, 1],
                 primary_threshold=3.0, n_permutations=500, seed=0)
    assert len(result.significant) >= 1
    best = result.significant[0]
    found = {tuple(edge) for edge in best.edges}
    assert set(planted) <= found
    assert best.p_value < 0.05
    np.testing.assert_array_equal(best.adjacency, best.adjacency.T)

    # The one-sided test in the other direction sees nothing.
    reverse = nbs(matrices, two_group_design(n1, n2), [0, -1],
                  primary_threshold=3.0, n_permutations=200, seed=0)
    assert not reverse.significant


def test_null_family_wise_error_rate():
    """Random labels, random data: P(any significant component) ~ alpha."""
    rng = np.random.default_rng(5)
    n_runs, alpha = 200, 0.05
    hits = 0
    for i_run in range(n_runs):
        matrices = random_matrices(10, 20, rng)
        result = nbs(matrices, two_group_design(10, 10), [0, 1], primary_threshold=2.0,
                     n_permutations=99, alpha=alpha, seed=i_run, exact=False)
        hits += bool(result.significant)
    rate = hits / n_runs
    # Binomial sd at 200 runs is ~0.015; a broken null gives ~0 or >> 0.1.
    assert 0.01 <= rate <= 0.10, rate


def test_freedman_lane_with_intercept_equals_plain_permutation():
    """[1, g] contrast [0, 1] (nuisance = intercept, FL path) vs cell-means [1, -1]."""
    rng = np.random.default_rng(6)
    matrices = random_matrices(8, 16, rng)
    group = np.r_[np.ones(8), np.zeros(8)]
    common = dict(primary_threshold=1.5, n_permutations=300, seed=11, exact=False,
                  store_null_stats=True)
    with_intercept = nbs(matrices, np.column_stack([np.ones(16), group]), [0, 1], **common)
    cell_means = nbs(matrices, np.column_stack([group, 1 - group]), [1, -1], **common)
    np.testing.assert_allclose(with_intercept.null_stats, cell_means.null_stats, atol=1e-10)
    np.testing.assert_array_equal(with_intercept.null_max_size, cell_means.null_max_size)


def test_freedman_lane_controls_a_nuisance_that_drives_the_edges():
    """A covariate correlated with group drives every edge; FL must not blame the group."""
    rng = np.random.default_rng(7)
    n_obs, hits = 24, 0
    for i_run in range(40):
        group = np.r_[np.ones(12), np.zeros(12)]
        covariate = group + rng.standard_normal(n_obs)
        matrices = random_matrices(8, n_obs, rng) + 1.5 * covariate[None, None, :]
        design = np.column_stack([np.ones(n_obs), group, covariate])
        result = nbs(matrices, design, [0, 1, 0], primary_threshold=2.0,
                     n_permutations=99, seed=i_run)
        hits += bool(result.significant)
    assert hits / 40 <= 0.15


def test_exact_and_monte_carlo_agree():
    rng = np.random.default_rng(8)
    matrices = random_matrices(8, 10, rng)
    matrices[0, 1, :5] += 1.2
    matrices[1, 0, :5] += 1.2
    design = two_group_design(5, 5)
    exact = nbs(matrices, design, [0, 1], primary_threshold=1.5, n_permutations=5000)
    assert exact.exact and exact.null_max_size.size == math.comb(10, 5)
    monte_carlo = nbs(matrices, design, [0, 1], primary_threshold=1.5,
                      n_permutations=5000, exact=False, seed=0)
    assert not monte_carlo.exact
    for c_exact, c_mc in zip(exact.components, monte_carlo.components):
        assert c_exact.size == c_mc.size
        assert abs(c_exact.p_value - c_mc.p_value) < 0.03
    # The observed labelling is inside the enumeration, so p >= 1 / count.
    assert min(c.p_value for c in exact.components) >= 1 / math.comb(10, 5)


def test_one_sample_sign_flip_exact():
    rng = np.random.default_rng(9)
    matrices = random_matrices(6, 8, rng)
    matrices[2, 3, :] += 3.0
    matrices[3, 2, :] += 3.0
    result = nbs(matrices, test="one_sample", primary_threshold=3.0, n_permutations=1000)
    assert result.exact and result.null_max_size.size == 2 ** 8
    assert (2, 3) in {tuple(e) for c in result.components for e in c.edges}


def test_rethreshold_equals_a_fresh_run():
    rng = np.random.default_rng(10)
    matrices = random_matrices(10, 20, rng)
    design = two_group_design(10, 10)
    base = nbs(matrices, design, [0, 1], primary_threshold=2.0, n_permutations=200,
               seed=3, exact=False, store_null_stats=True)
    for threshold, measure in [(1.5, "extent"), (2.5, "intensity"), (1.8, "intensity_excess")]:
        reused = rethreshold(base, threshold, size_measure=measure)
        fresh = nbs(matrices, design, [0, 1], primary_threshold=threshold, n_permutations=200,
                    seed=3, exact=False, size_measure=measure)
        np.testing.assert_allclose(reused.null_max_size, fresh.null_max_size)
        assert [c.p_value for c in reused.components] == [c.p_value for c in fresh.components]


def test_edges_zero_everywhere_are_not_tested():
    rng = np.random.default_rng(11)
    matrices = random_matrices(5, 10, rng)
    matrices[0, 4, :] = matrices[4, 0, :] = 0.0
    result = nbs(matrices, two_group_design(5, 5), [0, 1], primary_threshold=-10.0,
                 n_permutations=20, exact=False, seed=0)
    # With a threshold of -10 every TESTED edge is kept; the untested one is not.
    assert result.stat_matrix[0, 4] == 0.0
    assert not any(component.adjacency[0, 4] for component in result.components)


def test_intensity_sums_the_statistics():
    rng = np.random.default_rng(12)
    matrices = random_matrices(6, 12, rng)
    result = nbs(matrices, two_group_design(6, 6), [0, 1], primary_threshold=0.5,
                 size_measure="intensity", n_permutations=20, exact=False, seed=0)
    for component in result.components:
        stats_in = result.stat_matrix[component.edges[:, 0], component.edges[:, 1]]
        assert component.size == pytest.approx(stats_in.sum())
