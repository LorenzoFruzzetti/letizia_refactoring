"""nbs - Network-Based Statistic (Zalesky, Fornito & Bullmore 2010) in numpy/scipy.

Implements NBS_ALGORITHM.md: edge-wise GLM (t / F / one-sample), primary
threshold, connected components sized by extent or intensity, and a permutation
null of the maximum component size giving FWER-corrected p-values per component.
Freedman-Lane handles nuisance covariates, exchange blocks restrict permutations
for repeated measures, and small designs are enumerated exactly.

Kept separate from ``wfci`` on purpose: ``wfci`` scopes NBS out (MERGING_PLAN.md
§4.3) and only *ingests* an adjacency (``wfci.significance``). A significant
component's ``adjacency`` is exactly that input.
"""

from __future__ import annotations

from .components import SIZE_MEASURES
from .core import Component, NBSResult, fisher_z, nbs, rethreshold, upper_triangle
from .glm import TESTS, EdgeGLM, build_glm, edge_statistics

__all__ = [
    "SIZE_MEASURES",
    "TESTS",
    "Component",
    "EdgeGLM",
    "NBSResult",
    "build_glm",
    "edge_statistics",
    "fisher_z",
    "nbs",
    "rethreshold",
    "upper_triangle",
]
