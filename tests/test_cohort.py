"""The generic cohort layer, and the library/experiment boundary it protects.

`wfci.cohort` is pure mechanism: stack per-animal matrices, select subsets by
study-defined metadata, average, difference. The tests hand-compute the same
group means with plain numpy and require the primitives to match -- a real check
of what `.mean()` / `DIFF` do, not a tautology.

The last test is the one that matters most for the design: it enforces P2 -- that
no study term ever leaks into `src/wfci/`. If that fails, the boundary the whole
cohort layer exists to draw has been crossed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from wfci import AnimalResult, CohortTable, LabeledMatrix, load_results

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("a", "b", "c")


def _sym(seed: int, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1.0, 1.0, (n, n))
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)
    return m


def _cohort() -> CohortTable:
    """Four animals, two groups, tagged with a study-defined 'group' and 'tag'."""
    return CohortTable([
        AnimalResult(_sym(0), LABELS, {"group": "healthy", "tag": "X"}),
        AnimalResult(_sym(1), LABELS, {"group": "healthy", "tag": "Y"}),
        AnimalResult(_sym(2), LABELS, {"group": "disease", "tag": "X"}),
        AnimalResult(_sym(3), LABELS, {"group": "disease", "tag": "Y"}),
    ])


# ---------------------------------------------------------------------------
# The maths, against a hand computation
# ---------------------------------------------------------------------------
def test_group_mean_matches_a_plain_numpy_average():
    """.select(...).mean() must equal np.mean over the same animals' matrices."""
    table = _cohort()

    got = table.select(group="healthy").mean()

    expected = np.mean([_sym(0), _sym(1)], axis=0)
    np.testing.assert_allclose(got.matrix, expected)
    assert got.labels == LABELS


def test_diff_is_the_difference_of_the_group_means():
    """DIFF = mean_healthy - mean_disease, exactly as MATLAB step (5)."""
    table = _cohort()

    diff = table.select(group="healthy").mean() - table.select(group="disease").mean()

    expected = np.mean([_sym(0), _sym(1)], axis=0) - np.mean([_sym(2), _sym(3)], axis=0)
    np.testing.assert_allclose(diff.matrix, expected)
    assert isinstance(diff, LabeledMatrix)
    assert diff.labels == LABELS


def test_stack_is_cat3_over_animals():
    """.stack() is MATLAB's cat(3, ...): [n_roi, n_roi, n_animal]."""
    table = _cohort()

    stacked = table.stack()

    assert stacked.shape == (3, 3, 4)
    np.testing.assert_array_equal(stacked[:, :, 0], _sym(0))


def test_a_two_key_selection_is_an_and():
    table = _cohort()

    sel = table.select(group="healthy", tag="X")

    assert len(sel) == 1
    np.testing.assert_array_equal(sel[0].matrix, _sym(0))


def test_mean_ignores_nan_cells():
    """A NaN cell (e.g. a fully-masked ROI in one animal) drops out, not poisons."""
    a = _sym(0)
    b = _sym(1)
    b[0, 1] = b[1, 0] = np.nan
    table = CohortTable([
        AnimalResult(a, LABELS, {"group": "g"}),
        AnimalResult(b, LABELS, {"group": "g"}),
    ])

    m = table.select(group="g").mean().matrix

    # The NaN cell averages over just the finite animal.
    assert m[0, 1] == pytest.approx(a[0, 1])
    assert m[1, 2] == pytest.approx(np.mean([a[1, 2], b[1, 2]]))


# ---------------------------------------------------------------------------
# Selection edge cases fail usefully
# ---------------------------------------------------------------------------
def test_an_empty_selection_raises_with_the_available_values():
    table = _cohort()

    with pytest.raises(ValueError, match="No animals match"):
        table.select(group="healthy", tag="Z")


def test_a_missing_key_is_a_non_match_not_an_error():
    """A key only some animals carry must exclude the others, not crash."""
    table = CohortTable([
        AnimalResult(_sym(0), LABELS, {"group": "g", "extra": "yes"}),
        AnimalResult(_sym(1), LABELS, {"group": "g"}),  # no 'extra'
    ])

    sel = table.select(extra="yes")

    assert len(sel) == 1


def test_groupby_splits_by_a_key():
    table = _cohort()

    groups = table.groupby("group")

    assert set(groups) == {"healthy", "disease"}
    assert len(groups["healthy"]) == 2


def test_filter_is_the_predicate_escape_hatch():
    table = _cohort()

    sel = table.filter(lambda r: r.metadata["tag"] == "X")

    assert len(sel) == 2


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
def test_pooling_mismatched_labels_is_refused():
    """Averaging matrices with different ROI orders would compare wrong regions."""
    with pytest.raises(ValueError, match="same ROI label order"):
        CohortTable([
            AnimalResult(_sym(0), ("a", "b", "c"), {}),
            AnimalResult(_sym(1), ("a", "c", "b"), {}),  # reordered
        ])


def test_an_empty_cohort_is_refused():
    with pytest.raises(ValueError, match="empty"):
        CohortTable([])


def test_labeled_matrix_refuses_to_combine_different_labels():
    a = LabeledMatrix(_sym(0), ("a", "b", "c"))
    b = LabeledMatrix(_sym(1), ("a", "c", "b"))

    with pytest.raises(ValueError, match="index different regions"):
        a - b


def test_a_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        AnimalResult(np.zeros((3, 4)), ("a", "b", "c"), {})


def test_label_count_must_match_the_matrix():
    with pytest.raises(ValueError, match="labels"):
        AnimalResult(_sym(0, n=3), ("a", "b"), {})


# ---------------------------------------------------------------------------
# load_results reads run_pipeline.py output
# ---------------------------------------------------------------------------
def test_load_results_reads_npz_and_attaches_study_metadata(tmp_path):
    """The library reads the matrix; the study's callback says what it is."""
    for i, group in enumerate(["healthy", "disease"]):
        np.savez(
            tmp_path / f"animal_{i}_{group}.npz",
            R_mean=_sym(i),
            roi_labels=np.array(LABELS),
        )

    def meta(path):
        return {"group": "healthy" if "healthy" in path.stem else "disease"}

    results = load_results(str(tmp_path / "*.npz"), metadata_from=meta)
    table = CohortTable(results)

    assert len(table) == 2
    assert set(table.values("group")) == {"healthy", "disease"}
    assert table.labels == LABELS


def test_load_results_errors_clearly_on_a_missing_key(tmp_path):
    np.savez(tmp_path / "x.npz", something_else=_sym(0))

    with pytest.raises(KeyError, match="R_mean"):
        load_results(str(tmp_path / "x.npz"))


# ---------------------------------------------------------------------------
# P2: the library/experiment boundary, enforced
# ---------------------------------------------------------------------------
def test_no_study_term_leaks_into_the_library():
    """MERGING_PLAN.md P2, made a test: src/wfci/ must contain no study knowledge.

    The cohort layer exists precisely so that a study's groups, animals and splits
    live in experiments/, never in the package. If one of these terms appears in a
    library source file, that boundary has been crossed -- so the grep the plan
    specified is run here on every CI pass, not just once by hand.
    """
    pattern = re.compile(r"sani|pd_|macchi|day4|sex", re.IGNORECASE)
    offenders = []
    for py in (REPO_ROOT / "src" / "wfci").rglob("*.py"):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "study terms leaked into the library (P2 violation):\n" + "\n".join(offenders)
    )


def test_the_experiment_script_is_where_the_study_terms_live():
    """The counterpart: the study knowledge really is in experiments/, not nowhere.

    A boundary is only meaningful if the thing it excludes exists on the other
    side. This confirms the worked study script carries the design the library
    refuses to.
    """
    script = REPO_ROOT / "experiments" / "healthy_vs_disease_day4.py"
    text = script.read_text(encoding="utf-8")

    assert "COHORT" in text
    assert '"group"' in text and '"sex"' in text
    assert "MACCHI" in text  # the real animal ids live here


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
