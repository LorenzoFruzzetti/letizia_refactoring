"""Worked study script -- group connectivity contrast (Antea steps 5-6).

This is the POLICY layer (MERGING_PLAN.md P2). Everything the `wfci` library
deliberately does not know lives here: which animals exist, which group and sex
each belongs to, how the cohort splits, which edges count as significant, and how
the figures are coloured and laid out. The library supplies only mechanism
(`wfci.cohort`, `wfci.significance`); this file supplies the experiment.

It corresponds to the MATLAB scripts `Antea_scripts/(5)_matrici_e_figure.txt` and
`(6)_visualizzazione_connettivita_sign.txt`, where the same design was encoded in
variable names (`mean_R_PV_F_MACCHI_DX_day4`, `mean_SANI`, ...). Here it is a
table this script fills in, so the next study edits THIS file, never the library.

What it does:
  1. builds a cohort table (per-animal R_mean + metadata),
  2. group means and their difference DIFF = healthy - disease,
  3. the male / female sub-contrasts,
  4. the figures: the DIFF matrix, the significant-connection network, the
     per-node bar plots.

Run it (writes PNGs to experiments/output/):
    conda run -n letizia python experiments/healthy_vs_disease_day4.py

By default it runs on a SYNTHETIC cohort with a known injected difference, so the
script is runnable and testable without real data. Point `RUN_CONFIG["results_dir"]`
at a folder of real `run_pipeline.py` .npz outputs to run on those instead.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Headless: render to files without a display. Set before importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make wfci importable when run straight from the repo without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wfci import (  # noqa: E402
    CORTEX_22,
    CohortTable,
    AnimalResult,
    load_results,
    mask_by_adjacency,
    network_figure,
    significance_barplot,
    hemispheric_layout,
)

# ---------------------------------------------------------------------------
# THE EXPERIMENTAL DESIGN -- this is the study knowledge the library must not hold.
# ---------------------------------------------------------------------------
# One row per animal: its id, its group, its sex. This is exactly what MATLAB
# step (5) spelled out in variable names like `mean_R_PV_F_MACCHI_DX_day4`.
COHORT: list[dict[str, str]] = [
    {"animal": "PV_F_MACCHI_DX",   "group": "healthy", "sex": "F"},
    {"animal": "PV_F_MACCHI_NM",   "group": "healthy", "sex": "F"},
    {"animal": "PV_M_MACCHI_DX",   "group": "healthy", "sex": "M"},
    {"animal": "PV_M_MACCHI_DXSX", "group": "healthy", "sex": "M"},
    {"animal": "PV_M_MACCHI_NM",   "group": "healthy", "sex": "M"},
    {"animal": "CR1F_NM",          "group": "disease", "sex": "F"},
    {"animal": "CR1M_DX",          "group": "disease", "sex": "M"},
    {"animal": "CR1M_DXSX",        "group": "disease", "sex": "M"},
    {"animal": "CR1M_NM",          "group": "disease", "sex": "M"},
    {"animal": "CR1M_SX",          "group": "disease", "sex": "M"},
]

# Display names for the 22 cortical ROIs, in the atlas's column order. The atlas
# keys are the transcription labels (M2L_alta, ...); these are the anatomical
# names the MATLAB figures used. A study-owned relabelling -- pure presentation.
DISPLAY_LABELS = [
    "MOs-a_L", "MOs-p_L", "MOp-a_L", "MOp-p_L", "SSp-bfd_L", "SSp-tr_L",
    "SSp-fl_L", "SSp-hl_L", "RSP_L", "VISa_L", "VISp_L",
    "MOs-a_R", "MOs-p_R", "MOp-a_R", "MOp-p_R", "SSp-bfd_R", "SSp-tr_R",
    "SSp-fl_R", "SSp-hl_R", "RSP_R", "VISa_R", "VISp_R",
]

# ---------------------------------------------------------------------------
# Edit to run from the editor without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Folder of per-animal run_pipeline.py .npz outputs. None -> synthetic demo.
    "results_dir": None,
    "output_dir": str(Path(__file__).resolve().parent / "output"),
    # Fraction of the strongest |DIFF| edges the demo treats as "significant".
    # In a real study this comes from NBS, not a threshold -- see demo_adjacency.
    "significant_fraction": 0.10,
    "seed": 0,
}


# ---------------------------------------------------------------------------
# Loading a REAL cohort (per-animal .npz from run_pipeline.py)
# ---------------------------------------------------------------------------
def _metadata_from_filename(path: Path) -> dict[str, str]:
    """Look an animal up in COHORT by matching its id against the file name.

    This is the study's own parser -- the seam wfci.cohort.load_results leaves for
    exactly this: the library reads the matrix, the study says what it is.
    """
    stem = path.stem
    for row in COHORT:
        if re.search(re.escape(row["animal"]), stem):
            return dict(row)
    raise ValueError(
        f"{path.name} does not match any animal in COHORT. Add it to COHORT or "
        f"rename the file to contain the animal id."
    )


def load_cohort(results_dir: str | Path) -> CohortTable:
    """Load real per-animal results, tagging each from COHORT."""
    results = load_results(
        str(Path(results_dir) / "*.npz"), metadata_from=_metadata_from_filename
    )
    return CohortTable(results)


# ---------------------------------------------------------------------------
# A SYNTHETIC cohort, so the script runs and is testable without real data
# ---------------------------------------------------------------------------
def demo_cohort(seed: int = 0) -> CohortTable:
    """A synthetic cohort with a KNOWN injected group difference.

    Every animal gets a symmetric, correlation-like matrix. Healthy and disease
    animals differ by a fixed signal on a handful of edges, so the group DIFF has
    real structure for the significance figures -- and the group means are an
    ordinary numpy average of these matrices, which is what the test hand-checks.
    """
    labels = list(CORTEX_22)          # the atlas column order
    n = len(labels)
    rng = np.random.default_rng(seed)

    # A fixed signal: a few edges that separate the groups (symmetric, zero diag).
    signal = np.zeros((n, n))
    for i, j in [(0, 11), (2, 13), (4, 15), (8, 19)]:
        signal[i, j] = signal[j, i] = 0.4

    results = []
    for k, row in enumerate(COHORT):
        base = rng.uniform(-0.3, 0.3, (n, n))
        base = (base + base.T) / 2.0
        sign = +1.0 if row["group"] == "healthy" else -1.0
        matrix = base + sign * signal
        np.fill_diagonal(matrix, 1.0)
        results.append(AnimalResult(matrix, tuple(labels), dict(row),
                                    source=f"demo::{row['animal']}"))
    return CohortTable(results)


def demo_adjacency(diff: np.ndarray, fraction: float) -> np.ndarray:
    """Stand-in for an NBS significance mask: the strongest |DIFF| edges.

    A REAL study loads its adjacency from NBS (or another permutation test) --
    wfci does not compute the network statistic (MERGING_PLAN.md non-goal). This
    demo just marks the top `fraction` of edges so the figures have something to
    draw. It is policy, so it lives here, not in wfci.significance.
    """
    n = diff.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    strengths = np.abs(diff[iu, ju])
    if strengths.size == 0:
        return np.zeros_like(diff)
    keep = max(1, int(round(fraction * strengths.size)))
    threshold = np.sort(strengths)[-keep]
    adj = np.zeros((n, n))
    for i, j, s in zip(iu, ju, strengths):
        if s >= threshold:
            adj[i, j] = adj[j, i] = 1.0
    return adj


# ---------------------------------------------------------------------------
# The contrasts -- generic selections, study-defined groups
# ---------------------------------------------------------------------------
def compute_contrasts(table: CohortTable) -> dict[str, Any]:
    """Group means and their differences, overall and per sex.

    Each line is one selection the study chooses; the library only knows how to
    average a subset. Change a group definition here and nothing in wfci moves.
    """
    healthy = table.select(group="healthy").mean()
    disease = table.select(group="disease").mean()
    diff = healthy - disease                      # DIFF = mean_SANI - mean_PD

    # Male sub-contrast.
    male_diff = (
        table.select(group="healthy", sex="M").mean()
        - table.select(group="disease", sex="M").mean()
    )

    # Female sub-contrast. NOTE: the MATLAB step (5) female block selected the
    # groups INCONSISTENTLY with its own lines 6-7 -- it treated CR1F as healthy
    # and PV_F_MACCHI as disease, i.e. swapped. Under this design there is no
    # library behaviour to port wrongly: the grouping is one line the study owns.
    # The CONSISTENT selection is below; if the original swap was intended, swap
    # the two `.select(...)` calls. Either way it is a change to THIS file only.
    female_diff = (
        table.select(group="healthy", sex="F").mean()
        - table.select(group="disease", sex="F").mean()
    )

    return {
        "healthy": healthy,
        "disease": disease,
        "diff": diff,
        "male_diff": male_diff,
        "female_diff": female_diff,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_diff_matrix(diff, labels, path: Path) -> None:
    """The DIFF correlation matrix (MATLAB step 5 imagesc)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    limit = float(np.abs(diff.matrix).max()) or 1.0
    im = ax.imshow(diff.matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("DIFF = healthy - disease")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def figure_network(diff, adjacency, labels, path: Path) -> None:
    """The significant-connection network (MATLAB step 6)."""
    masked = mask_by_adjacency(diff.matrix, adjacency)
    # Two hemispheres as two columns (the atlas is 11 left, then 11 right).
    positions = hemispheric_layout(11, 11)
    fig, ax = plt.subplots(figsize=(7, 7))
    network_figure(masked, node_positions=positions, labels=labels, ax=ax)
    ax.set_title("Significant connections (demo adjacency)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def figure_barplots(diff, adjacency, labels, path: Path) -> None:
    """Per-node significant increases / decreases (MATLAB step 6 barh)."""
    masked = mask_by_adjacency(diff.matrix, adjacency)
    fig, (ax_neg, ax_pos) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    significance_barplot(masked, labels, sign="negative", ax=ax_neg, color="#c0392b")
    significance_barplot(masked, labels, sign="positive", ax=ax_pos, color="#2c6fbb")
    ax_neg.set_title("significant decreases")
    ax_pos.set_title("significant increases")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = dict(RUN_CONFIG)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if cfg["results_dir"]:
        table = load_cohort(cfg["results_dir"])
        print(f"Loaded {len(table)} animals from {cfg['results_dir']}")
    else:
        table = demo_cohort(cfg["seed"])
        print(f"Synthetic demo cohort: {len(table)} animals, {len(table.labels)} ROIs")

    contrasts = compute_contrasts(table)
    diff = contrasts["diff"]
    adjacency = demo_adjacency(diff.matrix, cfg["significant_fraction"])
    n_sig = int((np.triu(adjacency, 1) > 0).sum())

    print(f"  healthy: {len(table.select(group='healthy'))}  "
          f"disease: {len(table.select(group='disease'))}")
    print(f"  DIFF range: [{diff.matrix.min():.3f}, {diff.matrix.max():.3f}]")
    print(f"  significant edges (demo): {n_sig}")

    figure_diff_matrix(diff, DISPLAY_LABELS, out / "diff_matrix.png")
    figure_network(diff, adjacency, DISPLAY_LABELS, out / "network.png")
    figure_barplots(diff, adjacency, DISPLAY_LABELS, out / "barplots.png")
    print(f"Wrote figures -> {out}")


if __name__ == "__main__":
    main()
