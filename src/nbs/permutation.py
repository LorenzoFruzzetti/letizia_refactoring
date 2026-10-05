"""Relabellings for the permutation null (NBS_ALGORITHM.md steps 4 and "Permutation count").

Two kinds of relabelling:

* **row permutations** ``perm`` (n_obs,) -- the permuted dataset is
  ``fit + residuals[perm]`` (Freedman-Lane; plain permutation when fit is 0).
  With exchange blocks a row only moves among rows sharing its block label.
* **sign flips** ``signs`` (n_obs,) of +-1 -- for the one-sample test.

Exact enumeration lists every DISTINCT relabelling once. For row permutations,
two permutations are the same relabelling when they put the same design rows in
the same places; so the count is, per block, the multinomial coefficient of the
block's distinct design rows (two groups n1, n2 -> C(n1 + n2, n1)), multiplied
over blocks. That equivalence only holds when the permuted data is ``P @ Y``,
i.e. when the nuisance space is at most the intercept -- the caller checks.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np


def block_indices(exchange_blocks: np.ndarray | None, n_obs: int) -> list[np.ndarray]:
    """Row indices of each exchange block; one block of all rows if None."""
    if exchange_blocks is None:
        return [np.arange(n_obs)]
    exchange_blocks = np.asarray(exchange_blocks)
    if exchange_blocks.shape != (n_obs,):
        raise ValueError(f"exchange_blocks must be ({n_obs},), got {exchange_blocks.shape}")
    return [np.flatnonzero(exchange_blocks == label) for label in np.unique(exchange_blocks)]


def row_labels(design: np.ndarray) -> np.ndarray:
    """Integer label per observation: equal labels = identical design rows. (n_obs,)"""
    _, labels = np.unique(np.round(design, 12), axis=0, return_inverse=True)
    return labels.ravel()


def count_row_relabellings(design: np.ndarray, blocks: list[np.ndarray]) -> int:
    """Number of distinct design-row arrangements reachable within the blocks."""
    labels = row_labels(design)
    total = 1
    for rows in blocks:
        counts = np.bincount(labels[rows])
        total *= math.factorial(len(rows)) // math.prod(math.factorial(int(c)) for c in counts)
    return total


def _multiset_arrangements(labels: list[int]) -> Iterator[tuple[int, ...]]:
    """Every distinct ordering of a multiset of labels, each exactly once."""
    remaining = {label: labels.count(label) for label in sorted(set(labels))}
    n = len(labels)
    current: list[int] = []

    def extend() -> Iterator[tuple[int, ...]]:
        if len(current) == n:
            yield tuple(current)
            return
        for label, count in remaining.items():
            if count:
                remaining[label] -= 1
                current.append(label)
                yield from extend()
                current.pop()
                remaining[label] += 1

    yield from extend()


def enumerate_row_relabellings(design: np.ndarray, blocks: list[np.ndarray]) -> Iterator[np.ndarray]:
    """Yield one row permutation per distinct relabelling (the observed one included).

    For a target arrangement A of design-row labels, we want ``Y[perm]`` to give
    the statistic of ``Y`` against a design whose row i has label A[i]. Since the
    statistic is unchanged when Y and X are permuted together,
    ``stat(Y[perm], X) == stat(Y, X[argsort(perm)])``; so build the source map
    ``source[i]`` = some unused row carrying label A[i], and ``perm = argsort(source)``.
    """
    labels = row_labels(design)
    per_block = [list(_multiset_arrangements(labels[rows].tolist())) for rows in blocks]
    for combination in itertools.product(*per_block):
        source = np.empty(len(labels), dtype=np.intp)
        for rows, arrangement in zip(blocks, combination):
            # Unused source rows of each label, consumed in order.
            pools = {label: list(rows[labels[rows] == label]) for label in set(arrangement)}
            for position, label in zip(rows, arrangement):
                source[position] = pools[label].pop(0)
        yield np.argsort(source)


def random_row_permutations(
    blocks: list[np.ndarray], n_obs: int, n_permutations: int, rng: np.random.Generator
) -> Iterator[np.ndarray]:
    """Yield ``n_permutations`` random permutations, each shuffling within blocks only."""
    for _ in range(n_permutations):
        perm = np.arange(n_obs)
        for rows in blocks:
            perm[rows] = rng.permutation(rows)
        yield perm


def enumerate_sign_flips(n_obs: int) -> Iterator[np.ndarray]:
    """All 2**n_obs sign vectors (the observed all-plus one included)."""
    for signs in itertools.product((1.0, -1.0), repeat=n_obs):
        yield np.array(signs)


def random_sign_flips(n_obs: int, n_permutations: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Yield ``n_permutations`` random +-1 vectors."""
    for _ in range(n_permutations):
        yield rng.choice((1.0, -1.0), size=n_obs)
