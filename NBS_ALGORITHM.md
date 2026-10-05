# Network-Based Statistic (NBS): algorithm for implementation

Source: *Reference Manual for NBS Connectome v1.2* (Zalesky et al., 2012),
`Reference_Manual_NBS_v1.2.pdf`, sections 2 and 4. Original method: Zalesky,
Fornito & Bullmore (2010), NeuroImage 53:1197-1207.

NBS tests one hypothesis at every edge of a network. It controls the family-wise
error rate (FWER) at the level of **connected sub-networks (components)**, not
individual edges. A significant component means the network as a whole shows the
effect. No single edge inside it can be declared significant on its own.

## Inputs

| name | shape | meaning |
|---|---|---|
| `matrices` | `(n_nodes, n_nodes, n_obs)` | one symmetric connectivity matrix per observation (subject) |
| `design` | `(n_obs, n_pred)` | GLM design matrix; row k = observation k; intercept column added explicitly if needed |
| `contrast` | `(n_pred,)` | hypothesis tested, e.g. `[1, -1]` = group 1 > group 2 |
| `test` | `"t"`, `"F"`, `"one_sample"` | t is one-sided |
| `primary_threshold` | float | statistic cut-off for keeping an edge (e.g. t > 3.1) |
| `n_permutations` | int | default 5000 |
| `alpha` | float | FWER level, default 0.05 |
| `size_measure` | `"extent"`, `"intensity"` | component size (see step 3) |
| `exchange_blocks` | `(n_obs,)` or None | permute only within equal labels (repeated measures) |

Work on the upper triangle only: `Y` has shape `(n_obs, n_edges)`, where
`n_edges = n_nodes * (n_nodes - 1) / 2`. Edges that are zero in every observation
are excluded from testing.

## Algorithm

1. **Edge-wise GLM.** Fit `Y[:, e] = X @ beta + eps` for every edge e (all at once:
   `beta = pinv(X) @ Y`). The statistic per edge is:
   - t: `t = c @ beta / sqrt(sigma2 * c @ inv(X.T @ X) @ c)`, with
     `sigma2 = RSS / (n_obs - rank(X))`
   - F: the standard reduced vs full model F for the contrast
   - one-sample: t-test of the mean > 0

   Set NaN statistics (constant edges) to 0.
2. **Primary threshold.** Keep the edges with `stat > primary_threshold`.
3. **Connected components.** Build a graph from the kept edges and find its
   connected components (BFS/DFS, or `scipy.sparse.csgraph.connected_components`).
   Isolated nodes are not components. Size per component:
   - extent = number of edges (suits weak, widespread effects)
   - intensity = sum of the edge statistics (suits strong, focal effects).
     A common variant sums `stat - threshold`. The manual only says "sum of test
     statistic values".
4. **Permutation null.** For each permutation, repeat steps 1-3 on permuted data
   and store **only the size of the largest component** (0 if none):
   - use **one permutation vector for every edge**; this preserves the dependence
     between edges;
   - if the design has nuisance covariates, use **Freedman-Lane**: regress the
     nuisance columns out of Y, permute the residuals, add back the nuisance fit,
     then refit the full model;
   - `one_sample` flips the sign of each observation instead of permuting;
   - with `exchange_blocks`, shuffle only among rows that share a block label.
5. **FWER p-value.** For each observed component of size `s`:
   `p = (1 + #{perm: max_size >= s}) / (1 + n_permutations)`.
   The manual uses the uncorrected ratio `#{...} / n_permutations`; the `+1` form
   is the conservative, standard one. Report the components with `p < alpha`.

### Why the maximum (FWER)

```
FWER = P(any component s_i > t_alpha | H0) = 1 - P(max_i s_i <= t_alpha | H0) = alpha
```

The permutation maxima estimate the null distribution of `max_i s_i`, and
`t_alpha` is its (1 - alpha) quantile.

## Outputs

- A list of significant components, each with its edge set (binary upper-triangular
  adjacency), its size and its FWER p-value.
- The edge statistic matrix `(n_nodes, n_nodes)`.
- The null distribution of the maximum component size `(n_permutations,)`. Keep
  it: permutations can be reused when only the primary threshold changes, if the
  per-permutation statistic maps are stored.

## Permutation count

Two groups of sizes n1, n2 have exactly `C(n1 + n2, n1)` distinct relabellings,
including the observed one. The one-sample test has `2**n`. Exchange blocks allow
the product of the within-block counts. The smallest p-value possible is
`1 / count`. If `count <= n_permutations`, enumerate every relabelling and get an
exact test instead of random draws; random draws would repeat.

## Pitfalls for this dataset

- **Unit of observation = animal** (or at least session), never the recording.
  The 5 recordings of a session are not independent (see CLAUDE.md 9.30). Average
  them to one matrix per unit before a between-group test, or use exchange blocks
  for within-animal contrasts.
- Apply the **Fisher z transform** (`arctanh(r)`) to Pearson r before the GLM. It
  is standard practice, though not in the manual.
- NBS does not see how the matrices were made. Group differences in preprocessing
  (hemodynamic correction, baseline drift, GSR) become "connectivity". Model them
  as nuisance covariates where possible.
- The primary threshold is the one tuning parameter. FWER holds at any value.
  Report results over a small range (e.g. t = 2.5, 3.1, 3.5) instead of picking one
  after seeing them.
- For focal, single-edge effects, use the manual's alternative: edge-wise FDR on
  permutation p-values (it needs many more permutations, about 50000).

## Validation checklist

- Null data (random matrices, random labels): the rate of any significant
  component over many runs is about alpha.
- Planted effect: add +delta to a known connected edge set in one group; it is
  recovered with p < alpha.
- The edge statistics match `scipy.stats.ttest_ind` (two groups, no covariates).
- The Freedman-Lane path with no nuisance columns reduces to plain permutation.
- The exact enumeration and Monte Carlo p-values agree for small groups.
