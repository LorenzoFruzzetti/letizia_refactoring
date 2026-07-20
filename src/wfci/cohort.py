"""Generic cohort layer -- stack per-animal results, select subsets, difference means.

LIBRARY, not policy (MERGING_PLAN.md P2). **Nothing here knows a group name, an
animal, a condition, a day or a disease model.** It provides mechanism: load
per-animal connectivity matrices, tag each with whatever metadata the *study*
defines, select subsets by that metadata, and average or difference them. Which
animals exist, which group each belongs to, and how they split is the study
script's business -- see ``experiments/``.

This is the layer MATLAB step (5) corresponds to. There, a group mean was written
as ``mean(cat(3, <this animal>, <that animal>, ...), 3)`` with every animal in
the group named, by hand, in the code -- the experimental design encoded in
variable names. Here that design is **data the study fills in**, and the same
primitives serve the next study with entirely different animals, unchanged.

Worked example::

    results = load_results("outputs/*.npz", metadata_from=my_parser)
    table   = CohortTable(results)
    healthy = table.select(group="healthy").mean()   # generic selection + mean
    disease = table.select(group="disease").mean()
    diff    = healthy - disease                       # LabeledMatrix, keeps labels
"""

from __future__ import annotations

import glob as _glob
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnimalResult:
    """One animal's connectivity matrix plus the metadata a study tags it with.

    Attributes
    ----------
    matrix:
        ``[n_roi, n_roi]`` connectivity (e.g. a per-animal ``R_mean``).
    labels:
        ROI names in row/column order. Two results can only be pooled if their
        labels match exactly -- otherwise the matrices index different regions and
        averaging them is meaningless. Stored as a tuple so it is hashable and
        cannot drift.
    metadata:
        Study-defined tags: ``{"group": "healthy", "condition": "F", "animal":
        "A12", ...}``. The library never inspects the *values*; it only matches on
        them in :meth:`CohortTable.select`. What keys exist is entirely the
        study's choice.
    source:
        Where this came from (a file path), for debugging. Not used in any maths.
    """

    matrix: np.ndarray
    labels: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "labels", tuple(self.labels))
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"AnimalResult matrix must be square [n_roi, n_roi]; got {m.shape}."
            )
        if len(self.labels) != m.shape[0]:
            raise ValueError(
                f"AnimalResult has {len(self.labels)} labels but a {m.shape[0]}x"
                f"{m.shape[1]} matrix."
            )


@dataclass(frozen=True)
class LabeledMatrix:
    """A matrix that carries its ROI labels through arithmetic.

    Group means and their difference (``DIFF``) are matrices whose rows and columns
    only mean something with their labels attached. Returning a bare ndarray would
    drop them at exactly the point they are needed (the figures). ``__sub__`` /
    ``__add__`` check the labels match, so a difference between two differently
    ordered matrices raises instead of silently comparing the wrong regions.
    """

    matrix: np.ndarray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", np.asarray(self.matrix, dtype=np.float64))
        object.__setattr__(self, "labels", tuple(self.labels))

    def _check(self, other: "LabeledMatrix") -> None:
        if self.labels != other.labels:
            raise ValueError(
                "LabeledMatrix labels differ, so these matrices index different "
                "regions and cannot be combined. Left starts "
                f"{self.labels[:3]}..., right starts {other.labels[:3]}..."
            )

    def __sub__(self, other: "LabeledMatrix") -> "LabeledMatrix":
        self._check(other)
        return LabeledMatrix(self.matrix - other.matrix, self.labels)

    def __add__(self, other: "LabeledMatrix") -> "LabeledMatrix":
        self._check(other)
        return LabeledMatrix(self.matrix + other.matrix, self.labels)

    def __array__(self, dtype=None):
        # So np.asarray(labeled_matrix) and matplotlib imshow(...) just work.
        return np.asarray(self.matrix, dtype=dtype)


def _resolve_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Expand a glob string, a single path, or an iterable of paths to a list."""
    if isinstance(paths, (str, Path)):
        text = str(paths)
        if any(ch in text for ch in "*?[") and not Path(text).exists():
            matched = sorted(_glob.glob(text))
            if not matched:
                raise FileNotFoundError(f"No files match {text!r}.")
            return [Path(p) for p in matched]
        return [Path(paths)]
    return [Path(p) for p in paths]


def load_results(
    paths: str | Path | Iterable[str | Path],
    *,
    metadata_from: Callable[[Path], dict[str, Any]] | None = None,
    matrix_key: str = "R_mean",
    labels_key: str = "roi_labels",
) -> list[AnimalResult]:
    """Load per-animal ``.npz`` results (as written by ``run_pipeline.py``).

    Parameters
    ----------
    paths:
        A glob (``"outputs/*.npz"``), one path, or an iterable of paths.
    metadata_from:
        ``callable(Path) -> dict`` that supplies each file's study metadata --
        typically by parsing the filename, or looking the animal up in a table the
        study owns. Defaults to no metadata (an empty dict per file). This callback
        is the whole seam: the *library* reads matrices, the *study* says what each
        one is.
    matrix_key, labels_key:
        Which arrays to read. Defaults match ``run_pipeline.py``'s output
        (``R_mean`` and ``roi_labels``).

    Returns
    -------
    A list of :class:`AnimalResult`, in the order ``paths`` resolved to.
    """
    results = []
    for path in _resolve_paths(paths):
        with np.load(path, allow_pickle=False) as z:
            if matrix_key not in z:
                raise KeyError(
                    f"{path} has no array {matrix_key!r} (found {list(z.keys())}). "
                    f"Pass matrix_key= for a different layout."
                )
            matrix = np.asarray(z[matrix_key], dtype=np.float64)
            labels = (
                tuple(str(s) for s in z[labels_key]) if labels_key in z
                else tuple(str(i) for i in range(matrix.shape[0]))
            )
        metadata = dict(metadata_from(path)) if metadata_from is not None else {}
        results.append(AnimalResult(matrix, labels, metadata, source=str(path)))
    return results


class CohortTable:
    """An ordered collection of :class:`AnimalResult`, selectable by metadata.

    The table itself is the "cohort table" of MERGING_PLAN.md §4.1: a study fills
    it with per-animal results and whatever columns (metadata keys) it likes, then
    selects and averages subsets. The library never learns what the columns mean.

    All results must share one label order -- otherwise the matrices are not
    comparable and a group mean would average different regions together. This is
    checked on construction, not discovered later in a figure.
    """

    def __init__(self, results: Iterable[AnimalResult]):
        self._results: list[AnimalResult] = list(results)
        if not self._results:
            raise ValueError(
                "CohortTable is empty. It needs at least one AnimalResult; check "
                "that load_results matched any files."
            )
        first = self._results[0].labels
        for r in self._results:
            if r.labels != first:
                raise ValueError(
                    "Every animal must share the same ROI label order to be pooled. "
                    f"{r.source or 'an animal'} has {r.labels[:3]}... but the first "
                    f"has {first[:3]}.... Were these produced with the same atlas?"
                )
        self._labels = first

    # -- read-only sequence behaviour --------------------------------------
    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[AnimalResult]:
        return iter(self._results)

    def __getitem__(self, i: int) -> AnimalResult:
        return self._results[i]

    def __repr__(self) -> str:
        return f"CohortTable({len(self._results)} animals, {len(self._labels)} ROIs)"

    # -- selection ----------------------------------------------------------
    def select(self, **criteria: Any) -> "CohortTable":
        """A sub-table of animals whose metadata matches ALL given key=value pairs.

        ``table.select(group="healthy", condition="M")`` keeps animals tagged both
        ``group="healthy"`` and ``condition="M"``. A key an animal does not have is
        treated as a non-match (excluded), never an error -- a study may tag only
        some animals with a given key.

        Raises if the selection is empty: an empty group is almost always a typo in
        a criterion, and silently averaging nothing yields NaN with no clue why.
        """
        kept = [
            r for r in self._results
            if all(r.metadata.get(k, _MISSING) == v for k, v in criteria.items())
        ]
        if not kept:
            raise ValueError(
                f"No animals match {criteria}. Available values: "
                f"{ {k: sorted(self.values(k)) for k in criteria} }."
            )
        return CohortTable(kept)

    def filter(self, predicate: Callable[[AnimalResult], bool]) -> "CohortTable":
        """A sub-table of animals for which ``predicate(result)`` is true.

        The escape hatch for anything :meth:`select` cannot express with equality
        (ranges, membership, computed conditions). Still generic: the *predicate*
        is the study's, the filtering is the library's.
        """
        kept = [r for r in self._results if predicate(r)]
        if not kept:
            raise ValueError("No animals satisfy the predicate.")
        return CohortTable(kept)

    def groupby(self, key: str) -> dict[Any, "CohortTable"]:
        """Split into sub-tables by the value of one metadata key.

        ``table.groupby("group")`` -> ``{"healthy": CohortTable, "disease":
        CohortTable}``. Order of first appearance is preserved.
        """
        buckets: dict[Any, list[AnimalResult]] = {}
        for r in self._results:
            if key in r.metadata:
                buckets.setdefault(r.metadata[key], []).append(r)
        return {value: CohortTable(rs) for value, rs in buckets.items()}

    def values(self, key: str) -> set[Any]:
        """The set of values present for a metadata key (for building selections)."""
        return {r.metadata[key] for r in self._results if key in r.metadata}

    # -- reductions ---------------------------------------------------------
    def stack(self) -> np.ndarray:
        """The animals stacked as ``[n_roi, n_roi, n_animal]`` (MATLAB ``cat(3,...)``)."""
        return np.stack([r.matrix for r in self._results], axis=-1)

    def mean(self) -> LabeledMatrix:
        """Element-wise mean over the animals -> a :class:`LabeledMatrix`.

        ``nanmean``: a NaN cell (e.g. a fully-masked ROI in one animal) drops out
        of that cell's average rather than poisoning it, matching how the rest of
        the pipeline treats masked-out regions.
        """
        return LabeledMatrix(np.nanmean(self.stack(), axis=2), self._labels)

    def to_dataframe(self):
        """A pandas DataFrame of the metadata, one row per animal (lazy import).

        Convenience for inspecting a cohort ("who is in it, tagged how"); the
        maths never needs it, so pandas stays an optional, lazily-imported extra.
        """
        import pandas as pd

        rows = [{"source": r.source, **r.metadata} for r in self._results]
        return pd.DataFrame(rows)


# Sentinel so select() distinguishes "key absent" from "value is None".
_MISSING = object()
