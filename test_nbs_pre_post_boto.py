"""NBS on the 22x22 ROI connectivity: preBoto (day_index 1-3) vs postBoto (day_index >= 4).

Same data and session averaging as test_nbs_change_over_time.py: every recording's
`R_roi` (roi_pixel_connectivity.py) is Fisher-z transformed and averaged over the
t# of its session (one animal on one day). Factors come from
`pixel_data/experimental_design.csv`; only groups R and T of the C57 line are kept.

Instead of a time slope, each session gets a period label:

    preBoto   day_index <  post_start_day_index   (1, 2, 3)
    postBoto  day_index >= post_start_day_index   (4, 5, 6)

Four analyses, each run in both directions and reported at three primary thresholds:

    pooled   z_edge ~ animal + post                    contrast: post
    R        same, R animals only                      contrast: post
    T        same, T animals only                      contrast: post
    R - T    z_edge ~ animal + post:R + post:T         contrast: post:R - post:T

plus, between groups inside each period:

    R_vs_T_preBoto    one obs per animal (its preBoto mean)    z ~ R + T, contrast R - T
    R_vs_T_postBoto   one obs per animal (its postBoto mean)   z ~ R + T, contrast R - T

For the within-animal analyses "increase" means stronger in postBoto than in preBoto;
for R_vs_T it means stronger in R than in T. Between groups the animal is the unit
(its sessions in the period are averaged first) and group labels are permuted freely
across animals, enumerated exactly. postBoto has only 3 R and 3 T animals, i.e. 20
relabellings: the smallest attainable p is 1/20 = 0.05, so R_vs_T_postBoto CANNOT be
significant at alpha = 0.05 whatever the data; read it descriptively.

For the within-animal analyses the animal dummies make the
comparison WITHIN-animal: only animals with sessions in both periods carry the
pre/post contrast. Animals with preBoto sessions only still enter the model (their
session-to-session scatter informs the error variance) but cannot drive the effect.
Sessions are permuted only within an animal, so the null keeps each animal's own
number of pre and post sessions.

Caveat: postBoto is also "later", so the contrast cannot separate the treatment from
anything else that changes with session number (drift, habituation, ageing).

Outputs (outputs/nbs_pre_post_boto/):
    components.csv              every component: analysis, direction, threshold, size, p
    sessions.csv                one row per session with its period
    animal_periods.csv          sessions per animal and period (who carries the contrast)
    animal_period_means.csv     the per-animal period means used by the R_vs_T tests
    edge_t_<analysis>.csv       22x22 observed t map (positive = higher postBoto / higher R)
    nbs_<analysis>.png          significant components, rows = direction, cols = threshold
    edge_t_heatmaps.png         observed t of every edge per analysis, significant edges boxed
    null_distributions.png      permutation null of the max component size vs the observed
    pre_post_mean_matrices.png  per group: preBoto mean z, postBoto mean z, post - pre
    r_vs_t_mean_matrices.png    per period: R mean z, T mean z, R - T
    pre_post_by_animal.png      per animal: preBoto vs postBoto mean z (all edges, and each
                                significant component)

Run: conda run --no-capture-output -n letizia python test_nbs_pre_post_boto.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbs import fisher_z, nbs, rethreshold
from wfci.significance import circular_layout, mask_by_adjacency, network_figure

# ---------------------------------------------------------------- parameters
repo_root = Path(__file__).resolve().parent
design_path = repo_root / "pixel_data" / "experimental_design.csv"
# Which roi_pixel_connectivity.py output folder to read: "pixels_dff_full" (mean baseline),
# "pixels_dff_full_gsr" (+ global signal regression), "pixels_median_dff_20s_full"
# (running-median baseline, no drift), "pixels_median_dff_20s_full_gsr" (both).
# The NBS_CONNECTIVITY_VARIANT environment variable overrides it for batch reruns.
connectivity_variant = os.environ.get("NBS_CONNECTIVITY_VARIANT", "pixels_dff_full")
connectivity_root = repo_root / "outputs" / "roi_pixel_connectivity" / connectivity_variant
# The original variant keeps its original output folder; any other gets a suffix.
variant_suffix = "" if connectivity_variant == "pixels_dff_full" else f"_{connectivity_variant}"
connectivity_file = "roi_pixel_connectivity.npz"   # per recording, holds R_roi (22, 22)
groups = ["R", "T"]                    # experimental groups kept
mouse_lines = ["C57"]                  # PV-CRE left out for now
post_start_day_index = 4               # first day_index counted as postBoto
period_names = ["preBoto", "postBoto"]
primary_thresholds = [2.5, 3.1, 3.5]   # t cut-offs, reported together
n_permutations = 5000                  # upper bound; exact enumeration when fewer relabellings exist
alpha = 0.05
size_measure = "extent"                # component size = number of edges
seed = 0
heatmap_t_limit = 4.0                  # shared colour scale of the t heatmaps
difference_z_limit = 0.4               # colour scale of the post - pre mean-z difference maps
output_dir = repo_root / "outputs" / f"nbs_pre_post_boto{variant_suffix}"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loading
design = pd.read_csv(design_path)
design = design[design["group"].isin(groups) & design["mouse_line"].isin(mouse_lines)]
design = design.reset_index(drop=True)

# Fisher z of every recording's ROI x ROI matrix.
recording_z = []
roi_labels = None
for row in design.itertuples():
    data = np.load(connectivity_root / row.pixel_dir / connectivity_file)
    if roi_labels is None:
        roi_labels = [str(label) for label in data["roi_labels"]]
    if [str(label) for label in data["roi_labels"]] != roi_labels:
        raise ValueError(f"{row.recording_id}: ROI labels differ from the first recording")
    recording_z.append(fisher_z(data["R_roi"]))
recording_z = np.stack(recording_z, axis=-1)                 # (n_rois, n_rois, n_recordings)
n_rois = len(roi_labels)

# ---------------------------------------------------------------- sessions
# One observation per animal-day: the Fisher-z mean over that session's t#.
design["session"] = design["day"].astype(str) + "_" + design["animal"]
sessions = (
    design.groupby("session", sort=False)
    .agg(day=("day", "first"), animal=("animal", "first"), group=("group", "first"),
         day_index=("day_index", "first"), n_recordings=("recording", "size"))
    .reset_index()
)
session_z = np.stack(
    [recording_z[:, :, design.index[design["session"] == name]].mean(axis=-1)
     for name in sessions["session"]],
    axis=-1,
)                                                            # (n_rois, n_rois, n_sessions)

# Period label of every session: 0 = preBoto, 1 = postBoto.
post = (sessions["day_index"] >= post_start_day_index).to_numpy().astype(float)
sessions["period"] = np.where(post == 1, period_names[1], period_names[0])

# Which animals have sessions in both periods (only these carry the contrast).
animal_periods = pd.crosstab([sessions["group"], sessions["animal"]], sessions["period"])
animal_periods = animal_periods.reindex(columns=period_names, fill_value=0)
animal_periods["carries_contrast"] = (animal_periods[period_names] > 0).all(axis=1)
animal_periods.to_csv(output_dir / "animal_periods.csv")

print(f"{len(design)} recordings -> {len(sessions)} sessions, "
      f"{sessions['animal'].nunique()} animals, postBoto = day_index >= {post_start_day_index}")
print(animal_periods)

# ---------------------------------------------------------------- designs
def within_animal_design(table: pd.DataFrame, effect_columns: dict[str, np.ndarray]):
    """Animal dummies (one per animal, no intercept) followed by the effect columns.

    Returns (design (n_sessions, n_animals + n_effects), column names).
    """
    animals = sorted(table["animal"].unique())
    dummies = (table["animal"].to_numpy()[:, None] == np.array(animals)[None, :]).astype(float)
    columns = [f"animal_{a}" for a in animals] + list(effect_columns)
    return np.column_stack([dummies, *effect_columns.values()]), columns


# ---------------------------------------------------------------- per-animal period means
# Each animal's mean z in each period (sessions averaged within the period). This is
# the observation of the between-group tests (the animal is the independent unit
# there) and is used by the descriptive plots, so every animal counts once.
animal_period_rows = []
animal_period_z = []
for (group, animal, period), period_sessions in sessions.groupby(["group", "animal", "period"]):
    animal_period_rows.append(dict(group=group, animal=animal, period=period,
                                   n_sessions=len(period_sessions)))
    animal_period_z.append(session_z[:, :, period_sessions.index].mean(axis=-1))
animal_period_table = pd.DataFrame(animal_period_rows)
animal_period_z = np.stack(animal_period_z, axis=-1)         # (n_rois, n_rois, n_animal_periods)
animal_period_table.to_csv(output_dir / "animal_period_means.csv", index=False)

# Every analysis names its observations (matrices), its exchange blocks (None = free
# permutation), what a positive t means, and the animals it uses.
analyses = {}
is_r = (sessions["group"] == "R").to_numpy().astype(float)
within_positive = f"higher {period_names[1]}"


def within_analysis(rows: np.ndarray, design_matrix: np.ndarray, contrast: np.ndarray) -> dict:
    """Session-level within-animal analysis: permutations stay inside each animal."""
    return dict(matrices=session_z[:, :, rows], design=design_matrix, contrast=contrast,
                exchange_blocks=sessions.loc[rows, "animal"].to_numpy(),
                animals=sessions.loc[rows, "animal"].to_numpy(), positive=within_positive,
                description=f"{period_names[1]} vs {period_names[0]}")


# pooled: one common within-animal pre/post shift over R and T.
all_rows = np.ones(len(sessions), bool)
matrix, columns = within_animal_design(sessions, {"post": post})
analyses["pooled"] = within_analysis(all_rows, matrix, np.r_[np.zeros(len(columns) - 1), 1.0])

# One group at a time.
for group in groups:
    rows = (sessions["group"] == group).to_numpy()
    matrix, columns = within_animal_design(sessions[rows], {"post": post[rows]})
    analyses[group] = within_analysis(rows, matrix, np.r_[np.zeros(len(columns) - 1), 1.0])

# R - T: separate pre/post shifts, contrast is their difference.
matrix, columns = within_animal_design(
    sessions, {"post_R": post * is_r, "post_T": post * (1 - is_r)}
)
analyses["R_minus_T"] = within_analysis(all_rows, matrix, np.r_[np.zeros(len(columns) - 2), 1.0, -1.0])

# R vs T inside each period: one observation per animal (its period mean), design =
# one mean per group, contrast R - T. Free permutation of group labels across animals;
# the nuisance is only the grand mean, so the null is enumerated exactly when possible.
for period in period_names:
    selected = (animal_period_table["period"] == period).to_numpy()
    animal_is_r = (animal_period_table.loc[selected, "group"] == "R").to_numpy().astype(float)
    analyses[f"R_vs_T_{period}"] = dict(
        matrices=animal_period_z[:, :, selected],
        design=np.column_stack([animal_is_r, 1 - animal_is_r]), contrast=np.array([1.0, -1.0]),
        exchange_blocks=None, animals=animal_period_table.loc[selected, "animal"].to_numpy(),
        positive="higher in R", description=f"R vs T, {period}",
    )

# ---------------------------------------------------------------- NBS
# Each analysis x direction: one permutation run, the other thresholds reuse it.
component_rows = []
results = {}                           # (analysis, direction) -> [NBSResult per threshold]
for name, analysis in analyses.items():
    animals = analysis["animals"]
    for direction, sign in [("increase", 1.0), ("decrease", -1.0)]:
        first = nbs(analysis["matrices"], analysis["design"], sign * analysis["contrast"],
                    primary_threshold=primary_thresholds[0], n_permutations=n_permutations,
                    alpha=alpha, size_measure=size_measure,
                    exchange_blocks=analysis["exchange_blocks"],
                    seed=seed, store_null_stats=True)
        per_threshold = [first] + [rethreshold(first, t) for t in primary_thresholds[1:]]
        results[(name, direction)] = per_threshold

        if direction == "increase":
            pd.DataFrame(first.stat_matrix, index=roi_labels, columns=roi_labels).to_csv(
                output_dir / f"edge_t_{name}.csv"
            )
        for result in per_threshold:
            for rank, component in enumerate(result.components, start=1):
                component_rows.append(dict(
                    analysis=name, direction=direction, primary_threshold=result.primary_threshold,
                    rank=rank, size=component.size, p_value=component.p_value,
                    significant=component.p_value < alpha,
                    n_observations=analysis["matrices"].shape[-1], n_animals=len(np.unique(animals)),
                    exact=result.exact, n_null=result.null_max_size.size,
                    nodes=" ".join(roi_labels[i] for i in component.nodes),
                    edges=" ".join(f"{roi_labels[i]}-{roi_labels[j]}" for i, j in component.edges),
                ))

components = pd.DataFrame(component_rows)
components.to_csv(output_dir / "components.csv", index=False)

# Console summary: the largest component of every run. "increase" = positive t
# (see each analysis's `positive`). With an exact null of N relabellings the smallest
# attainable p is 1/N, so N < 1/alpha means the analysis cannot reach significance.
print(f"\nsize = {size_measure}; increase = postBoto higher (within) or R higher (R_vs_T)")
for (name, direction), per_threshold in results.items():
    for result in per_threshold:
        best = result.components[0] if result.components else None
        text = (f"largest {best.size:g} edges, p = {best.p_value:.4f}" if best else "no edge above threshold")
        flag = "  *" if result.significant else ""
        null_text = f"exact, {result.null_max_size.size} relabellings" if result.exact else \
            f"{result.null_max_size.size} random permutations"
        print(f"  {name:16s} {direction:8s} t > {result.primary_threshold:g}: {text}{flag}  ({null_text})")
for name in analyses:
    first = results[(name, "increase")][0]
    if first.exact and 1 / first.null_max_size.size >= alpha:
        print(f"  NOTE {name}: only {first.null_max_size.size} distinct relabellings, smallest "
              f"attainable p = {1 / first.null_max_size.size:.3f} >= alpha; it cannot be significant")

# ---------------------------------------------------------------- plots
# One figure per analysis: rows = direction, columns = threshold, significant edges only.
node_positions = circular_layout(n_rois)
for name in analyses:
    fig, axes = plt.subplots(2, len(primary_thresholds), figsize=(6 * len(primary_thresholds), 12),
                             constrained_layout=True, squeeze=False)
    for i_row, direction in enumerate(["increase", "decrease"]):
        for i_col, result in enumerate(results[(name, direction)]):
            significant_adjacency = np.zeros((n_rois, n_rois), dtype=bool)
            for component in result.significant:
                significant_adjacency |= component.adjacency
            masked_t = mask_by_adjacency(result.stat_matrix, significant_adjacency)
            ax = axes[i_row, i_col]
            network_figure(masked_t, node_positions=node_positions, labels=roi_labels, ax=ax)
            best_p = min((c.p_value for c in result.components), default=np.nan)
            ax.set_title(f"{name} {direction} ({analyses[name]['positive'] if direction == 'increase' else 'opposite'}), "
                         f"t > {result.primary_threshold:g}\n"
                         f"{len(result.significant)} significant (min p = {best_p:.3f})")
    null_text = "within-animal permutations" if analyses[name]["exchange_blocks"] is not None else \
        f"group-label permutations across animals ({'exact' if results[(name, 'increase')][0].exact else 'random'}, " \
        f"n = {results[(name, 'increase')][0].null_max_size.size})"
    fig.suptitle(f"NBS {analyses[name]['description']} ({period_names[1]} = day_index >= "
                 f"{post_start_day_index}), C57 {'/'.join(groups)}, {null_text}")
    fig.savefig(output_dir / f"nbs_{name}.png", dpi=200)
    plt.close(fig)

# Observed t of every edge, one heatmap per analysis (positive = the analysis's
# `positive`). Edges of any significant component at the lowest threshold are boxed:
# black = increase, white = decrease.
analysis_names = list(analyses)
heatmap_columns = 3
heatmap_rows = int(np.ceil(len(analysis_names) / heatmap_columns))
fig, axes = plt.subplots(heatmap_rows, heatmap_columns, figsize=(6.5 * heatmap_columns, 6 * heatmap_rows),
                         constrained_layout=True, squeeze=False)
for ax in axes.flat[len(analysis_names):]:
    ax.axis("off")
for ax, name in zip(axes.flat, analysis_names):
    t_matrix = results[(name, "increase")][0].stat_matrix          # (n_rois, n_rois)
    image = ax.imshow(t_matrix, cmap="RdBu_r", vmin=-heatmap_t_limit, vmax=heatmap_t_limit)
    for direction, edge_color in [("increase", "black"), ("decrease", "white")]:
        for component in results[(name, direction)][0].significant:
            for i_node, j_node in component.edges:
                for row_index, col_index in [(i_node, j_node), (j_node, i_node)]:
                    ax.add_patch(plt.Rectangle((col_index - 0.5, row_index - 0.5), 1, 1,
                                               fill=False, edgecolor=edge_color, linewidth=1.5))
    ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rois), roi_labels, fontsize=7)
    ax.set_title(f"{name}: t, {analyses[name]['description']}\n"
                 f"(boxed = significant at t > {primary_thresholds[0]:g})")
    fig.colorbar(image, ax=ax, shrink=0.7, extend="both", label=f"t (positive = {analyses[name]['positive']})")
fig.savefig(output_dir / "edge_t_heatmaps.png", dpi=200)
plt.close(fig)

# Permutation null of the maximum component size, observed largest component marked.
fig, axes = plt.subplots(2 * len(analysis_names), len(primary_thresholds),
                         figsize=(4.5 * len(primary_thresholds), 2.6 * 2 * len(analysis_names)),
                         constrained_layout=True, squeeze=False)
for i_row, (name, direction) in enumerate(results):
    for i_col, result in enumerate(results[(name, direction)]):
        ax = axes[i_row, i_col]
        null_sizes = result.null_max_size                           # (n_null,)
        bins = np.arange(null_sizes.max() + 2) - 0.5
        ax.hist(null_sizes, bins=bins, color="0.6")
        critical_size = np.quantile(null_sizes, 1 - alpha)
        ax.axvline(critical_size, color="black", linestyle="--", label=f"{1 - alpha:.0%} null")
        if result.components:
            best = result.components[0]
            ax.axvline(best.size, color="red", label=f"observed {best.size:g} (p={best.p_value:.3f})")
        ax.set_yscale("log")
        null_text = "exact" if result.exact else "random"
        ax.set_title(f"{name} {direction}, t > {result.primary_threshold:g} "
                     f"({null_text}, n = {null_sizes.size})", fontsize=9)
        ax.set_xlabel("max component size (edges)", fontsize=8)
        ax.legend(fontsize=7)
fig.savefig(output_dir / "null_distributions.png", dpi=150)
plt.close(fig)

# Descriptive group matrices: mean z per period and post - pre, averaged over the
# animals that carry the contrast (each animal weighted once).
fig, axes = plt.subplots(len(groups), 3, figsize=(19, 6 * len(groups)),
                         constrained_layout=True, squeeze=False)
for i_row, group in enumerate(groups):
    contrast_animals = animal_periods.loc[group].index[animal_periods.loc[group, "carries_contrast"]]
    period_means = {}
    for period in period_names:
        selected = (
            (animal_period_table["group"] == group)
            & (animal_period_table["period"] == period)
            & animal_period_table["animal"].isin(contrast_animals)
        ).to_numpy()
        period_means[period] = animal_period_z[:, :, selected].mean(axis=-1)   # (n_rois, n_rois)
    difference = period_means[period_names[1]] - period_means[period_names[0]]
    both_means = np.concatenate([period_means[p][~np.eye(n_rois, dtype=bool)] for p in period_names])
    z_low, z_high = np.percentile(both_means, [2, 98])
    panels = [
        (period_means[period_names[0]], period_names[0], "viridis", z_low, z_high, "mean Fisher z"),
        (period_means[period_names[1]], period_names[1], "viridis", z_low, z_high, "mean Fisher z"),
        (difference, f"{period_names[1]} - {period_names[0]}", "RdBu_r",
         -difference_z_limit, difference_z_limit, "difference in Fisher z"),
    ]
    for ax, (matrix, title, cmap, low, high, label) in zip(axes[i_row], panels):
        shown = matrix.copy()
        np.fill_diagonal(shown, np.nan)
        image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)
        ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n_rois), roi_labels, fontsize=7)
        ax.set_title(f"group {group}: {title}\n(animals with both periods: {', '.join(contrast_animals)})",
                     fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.7, extend="both", label=label)
fig.savefig(output_dir / "pre_post_mean_matrices.png", dpi=200)
plt.close(fig)

# Descriptive between-group matrices: per period, the R mean, the T mean and R - T,
# each over the animal period means used by the R_vs_T tests.
fig, axes = plt.subplots(len(period_names), 3, figsize=(19, 6 * len(period_names)),
                         constrained_layout=True, squeeze=False)
for i_row, period in enumerate(period_names):
    group_means = {}
    group_animals = {}
    for group in groups:
        selected = ((animal_period_table["group"] == group) & (animal_period_table["period"] == period)).to_numpy()
        group_means[group] = animal_period_z[:, :, selected].mean(axis=-1)            # (n_rois, n_rois)
        group_animals[group] = animal_period_table.loc[selected, "animal"].tolist()
    difference = group_means[groups[0]] - group_means[groups[1]]
    both_means = np.concatenate([group_means[g][~np.eye(n_rois, dtype=bool)] for g in groups])
    z_low, z_high = np.percentile(both_means, [2, 98])
    panels = [
        (group_means[groups[0]], f"{period}: group {groups[0]} ({', '.join(group_animals[groups[0]])})",
         "viridis", z_low, z_high, "mean Fisher z"),
        (group_means[groups[1]], f"{period}: group {groups[1]} ({', '.join(group_animals[groups[1]])})",
         "viridis", z_low, z_high, "mean Fisher z"),
        (difference, f"{period}: {groups[0]} - {groups[1]}", "RdBu_r",
         -difference_z_limit, difference_z_limit, "difference in Fisher z"),
    ]
    for ax, (matrix, title, cmap, low, high, label) in zip(axes[i_row], panels):
        shown = matrix.copy()
        np.fill_diagonal(shown, np.nan)
        image = ax.imshow(shown, cmap=cmap, vmin=low, vmax=high)
        ax.set_xticks(range(n_rois), roi_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n_rois), roi_labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.7, extend="both", label=label)
fig.savefig(output_dir / "r_vs_t_mean_matrices.png", dpi=200)
plt.close(fig)

# Per-animal pre vs post: mean over all off-diagonal edges, then over each significant
# component (lowest threshold). Animals with one period only are drawn as a lone point.
off_diagonal_rows, off_diagonal_cols = np.triu_indices(n_rois, k=1)
edge_sets = [("all edges", off_diagonal_rows, off_diagonal_cols)]
for (name, direction), per_threshold in results.items():
    for component in per_threshold[0].significant:
        edge_sets.append((
            f"{name} {direction} component ({component.size:g} edges, "
            f"t > {per_threshold[0].primary_threshold:g}, p = {component.p_value:.3f})",
            component.edges[:, 0], component.edges[:, 1],
        ))
fig, axes = plt.subplots(len(edge_sets), len(groups), figsize=(5 * len(groups), 4.2 * len(edge_sets)),
                         sharey="row", constrained_layout=True, squeeze=False)
for i_row, (title, edge_rows, edge_cols) in enumerate(edge_sets):
    edge_mean_z = animal_period_z[edge_rows, edge_cols, :].mean(axis=0)   # (n_animal_periods,)
    for ax, group in zip(axes[i_row], groups):
        in_group = animal_period_table["group"] == group
        for animal, animal_rows in animal_period_table[in_group].groupby("animal"):
            x_positions = [period_names.index(p) for p in animal_rows["period"]]
            ax.plot(x_positions, edge_mean_z[animal_rows.index], "o-", label=animal)
        ax.set_xticks(range(len(period_names)), period_names)
        ax.set_xlim(-0.4, len(period_names) - 0.6)
        ax.set_title(f"{title}\ngroup {group}", fontsize=9)
        ax.legend(fontsize=7)
    axes[i_row, 0].set_ylabel("mean Fisher z (per animal and period)")
fig.savefig(output_dir / "pre_post_by_animal.png", dpi=200)
plt.close(fig)

sessions.to_csv(output_dir / "sessions.csv", index=False)
print(f"\noutputs: {output_dir}")
