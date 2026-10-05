"""Network-Based Statistic on synthetic 22-node connectivity: plant, recover, plot.

Builds one Fisher-z connectivity matrix per animal for two groups, adds an effect
to a known connected set of edges in group A, runs `nbs.nbs` at three primary
thresholds (NBS_ALGORITHM.md: report a range, do not pick one) and draws the
significant component with `wfci.significance.network_figure`.

Run:  conda run --no-capture-output -n letizia python examples/nbs_synthetic_example.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nbs import fisher_z, nbs, rethreshold
from wfci import CORTEX_22, atlas_labels
from wfci.significance import circular_layout, mask_by_adjacency, network_figure

# ---------------------------------------------------------------- parameters
n_animals_group_a = 12
n_animals_group_b = 12
baseline_r = 0.4                     # mean off-diagonal Pearson r of every matrix
noise_sd_z = 0.15                    # per-animal edge noise, in Fisher-z units
effect_z = 0.25                      # added to the planted edges in group A
planted_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]  # node pairs, i < j
primary_thresholds = [2.5, 3.1, 3.5]  # t cut-offs to report together
n_permutations = 5000
alpha = 0.05
seed = 0
output_dir = Path(__file__).resolve().parent / "output" / "nbs"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- synthetic data
# r matrices -> Fisher z, as the real analysis must do before the GLM.
labels = atlas_labels(CORTEX_22)
n_nodes = len(labels)
n_animals = n_animals_group_a + n_animals_group_b
rng = np.random.default_rng(seed)

z_baseline = np.arctanh(baseline_r)
noise = rng.normal(0.0, noise_sd_z, (n_nodes, n_nodes, n_animals))
noise = (noise + noise.transpose(1, 0, 2)) / np.sqrt(2)
connectivity_z = z_baseline + noise                              # (n_nodes, n_nodes, n_animals)
for i_node, j_node in planted_edges:
    connectivity_z[i_node, j_node, :n_animals_group_a] += effect_z
    connectivity_z[j_node, i_node, :n_animals_group_a] += effect_z
connectivity_r = np.tanh(connectivity_z)
connectivity_r[np.arange(n_nodes), np.arange(n_nodes), :] = 1.0
matrices = fisher_z(connectivity_r)                              # (n_nodes, n_nodes, n_animals)

# ---------------------------------------------------------------- NBS
# Design: intercept + group-A indicator; contrast [0, 1] tests A > B.
group_a = np.r_[np.ones(n_animals_group_a), np.zeros(n_animals_group_b)]
design = np.column_stack([np.ones(n_animals), group_a])          # (n_animals, 2)
first = nbs(matrices, design, [0, 1], primary_threshold=primary_thresholds[0],
            n_permutations=n_permutations, alpha=alpha, seed=seed, store_null_stats=True)
# The other thresholds reuse the same permutations (no GLM refits).
results = [first] + [rethreshold(first, threshold) for threshold in primary_thresholds[1:]]

print(f"planted edges: {[(labels[i], labels[j]) for i, j in planted_edges]}")
print(f"null: {first.null_max_size.size} permutations, exact={first.exact}")
for result in results:
    print(f"\nprimary threshold t > {result.primary_threshold:g}: "
          f"{len(result.components)} component(s), {len(result.significant)} significant")
    for component in result.components[:3]:
        edge_names = [f"{labels[i]}-{labels[j]}" for i, j in component.edges]
        print(f"  size {component.size:g}  p = {component.p_value:.4f}  edges: {edge_names}")

# ---------------------------------------------------------------- plot
# One panel per threshold: the t map masked to the significant component(s).
fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6),
                         constrained_layout=True, squeeze=False)
for ax, result in zip(axes[0], results):
    significant_adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    for component in result.significant:
        significant_adjacency |= component.adjacency
    masked_t = mask_by_adjacency(result.stat_matrix, significant_adjacency)
    network_figure(masked_t, node_positions=circular_layout(n_nodes), labels=labels, ax=ax)
    ax.set_title(f"NBS A > B, t > {result.primary_threshold:g} "
                 f"({len(result.significant)} significant, {first.null_max_size.size} perms)")
fig.savefig(output_dir / "nbs_synthetic_components.png", dpi=200)
plt.close(fig)
print(f"\nfigure: {output_dir / 'nbs_synthetic_components.png'}")
