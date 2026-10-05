"""NBS on the 22x22 ROI connectivity: how does it change across sessions in R and T?

Observations are SESSIONS (one animal on one day). Each session's `R_roi` matrices
(roi_pixel_connectivity.py, one per `t#`, the same ROI x ROI Pearson matrix
run_botox_batch computes) are Fisher-z averaged over its recordings. Factors come
from `pixel_data/experimental_design.csv`. Only groups R and T of the C57 line are
kept (PV-CRE animals are left out for now).

Four analyses, each run in both directions (the NBS t is one-sided) and reported
at three primary thresholds (NBS_ALGORITHM.md: report a range, do not pick one):

    pooled   z_edge ~ animal + time                    contrast: time
    R        same, R animals only                      contrast: time
    T        same, T animals only                      contrast: time
    R - T    z_edge ~ animal + time:R + time:T         contrast: time:R - time:T

The animal dummies make every slope WITHIN-animal: an animal's own sessions are
compared with each other, so animals differing in overall connectivity cannot
look like a change over time. Permutations shuffle sessions only within an animal
(exchange blocks), Freedman-Lane removing the animal means (and, for R - T, the
common slope) first. An animal with a single session carries no time information
and only adds its own dummy.

`time` is `day_index` (session number within the animal, 1..6) or, with
`time_variable = "days_since_first_session"`, calendar days since that animal's
first session.

Outputs (outputs/nbs_change_over_time/<time_variable>/):
    components.csv              every component: analysis, direction, threshold,
                                size, p, nodes, edges
    edge_t_<analysis>.csv       22x22 observed t map of each analysis (increase direction)
    nbs_<analysis>.png          significant components, rows = direction, cols = threshold
    edge_t_heatmaps.png         observed t of every edge per analysis, significant edges boxed
    null_distributions.png      permutation null of the max component size vs the observed
    significant_component_trajectories.png  mean z of each significant component per
                                session and animal, both groups
    mean_connectivity_by_session.png  per-animal mean off-diagonal z across sessions

Run: conda run --no-capture-output -n letizia python test_nbs_change_over_time.py
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
time_variable = os.environ.get("NBS_TIME_VARIABLE", "day_index")  # "day_index" | "days_since_first_session"
primary_thresholds = [2.5, 3.1, 3.5]   # t cut-offs, reported together
n_permutations = 5000
alpha = 0.05
size_measure = "extent"                # component size = number of edges
seed = 0
heatmap_t_limit = 4.0                  # shared colour scale of the t heatmaps (|t| beyond saturates)
output_dir = repo_root / "outputs" / f"nbs_change_over_time{variant_suffix}" / time_variable
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

# Calendar days since each animal's first session (day is yymmdd).
session_date = pd.to_datetime(sessions["day"].astype(str), format="%y%m%d")
first_date = session_date.groupby(sessions["animal"]).transform("min")
sessions["days_since_first_session"] = (session_date - first_date).dt.days
sessions["time"] = sessions[time_variable].astype(float)

print(f"{len(design)} recordings -> {len(sessions)} sessions, "
      f"{sessions['animal'].nunique()} animals")
print(pd.crosstab([sessions["group"], sessions["animal"]], sessions["day_index"]))


# ---------------------------------------------------------------- designs
def within_animal_design(table: pd.DataFrame, slope_columns: dict[str, np.ndarray]):
    """Animal dummies (one per animal, no intercept) followed by the slope columns.

    Returns (design (n_sessions, n_animals + n_slopes), column names).
    """
    animals = sorted(table["animal"].unique())
    dummies = (table["animal"].to_numpy()[:, None] == np.array(animals)[None, :]).astype(float)
    columns = [f"animal_{a}" for a in animals] + list(slope_columns)
    return np.column_stack([dummies, *slope_columns.values()]), columns


analyses = {}
time = sessions["time"].to_numpy()
is_r = (sessions["group"] == "R").to_numpy().astype(float)

# pooled: one common within-animal slope over R and T.
matrix, columns = within_animal_design(sessions, {"time": time})
analyses["pooled"] = dict(rows=np.ones(len(sessions), bool), design=matrix,
                          contrast=np.r_[np.zeros(len(columns) - 1), 1.0])

# One group at a time.
for group in groups:
    rows = (sessions["group"] == group).to_numpy()
    matrix, columns = within_animal_design(sessions[rows], {"time": time[rows]})
    analyses[group] = dict(rows=rows, design=matrix,
                           contrast=np.r_[np.zeros(len(columns) - 1), 1.0])

# R - T: separate slopes, contrast is their difference.
matrix, columns = within_animal_design(
    sessions, {"time_R": time * is_r, "time_T": time * (1 - is_r)}
)
analyses["R_minus_T"] = dict(rows=np.ones(len(sessions), bool), design=matrix,
                             contrast=np.r_[np.zeros(len(columns) - 2), 1.0, -1.0])

# ---------------------------------------------------------------- NBS
# Each analysis x direction: one permutation run, the other thresholds reuse it.
component_rows = []
results = {}                           # (analysis, direction) -> [NBSResult per threshold]
for name, analysis in analyses.items():
    rows = analysis["rows"]
    animals = sessions.loc[rows, "animal"].to_numpy()        # exchange blocks
    for direction, sign in [("increase", 1.0), ("decrease", -1.0)]:
        first = nbs(session_z[:, :, rows], analysis["design"], sign * analysis["contrast"],
                    primary_threshold=primary_thresholds[0], n_permutations=n_permutations,
                    alpha=alpha, size_measure=size_measure, exchange_blocks=animals,
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
                    n_sessions=int(rows.sum()), n_animals=len(np.unique(animals)),
                    exact=result.exact, n_null=result.null_max_size.size,
                    nodes=" ".join(roi_labels[i] for i in component.nodes),
                    edges=" ".join(f"{roi_labels[i]}-{roi_labels[j]}" for i, j in component.edges),
                ))

components = pd.DataFrame(component_rows)
components.to_csv(output_dir / "components.csv", index=False)

# Console summary: the largest component of every run.
print(f"\ntime = {time_variable}, {n_permutations} within-animal permutations, size = {size_measure}")
for (name, direction), per_threshold in results.items():
    for result in per_threshold:
        best = result.components[0] if result.components else None
        text = (f"largest {best.size:g} edges, p = {best.p_value:.4f}" if best else "no edge above threshold")
        flag = "  *" if result.significant else ""
        print(f"  {name:10s} {direction:8s} t > {result.primary_threshold:g}: {text}{flag}")

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
            ax.set_title(f"{name} {direction}, t > {result.primary_threshold:g}\n"
                         f"{len(result.significant)} significant (min p = {best_p:.3f})")
    fig.suptitle(f"NBS change over {time_variable}, C57 {'/'.join(groups)}, "
                 f"{n_permutations} within-animal permutations")
    fig.savefig(output_dir / f"nbs_{name}.png", dpi=200)
    plt.close(fig)

# Observed t of every edge, one heatmap per analysis (increase direction; the
# decrease t is its negative). Edges of any significant component at the lowest
# threshold are boxed: black = increase, white = decrease.
analysis_names = list(analyses)
fig, axes = plt.subplots(1, len(analysis_names), figsize=(6.5 * len(analysis_names), 6),
                         constrained_layout=True, squeeze=False)
for ax, name in zip(axes[0], analysis_names):
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
    ax.set_title(f"{name}: t of the time slope\n(boxed = significant at t > {primary_thresholds[0]:g})")
    fig.colorbar(image, ax=ax, shrink=0.7, extend="both", label="t (positive = increases over time)")
fig.savefig(output_dir / "edge_t_heatmaps.png", dpi=200)
plt.close(fig)

# Permutation null of the maximum component size, observed largest component marked.
fig, axes = plt.subplots(2 * len(analysis_names), len(primary_thresholds),
                         figsize=(4.5 * len(primary_thresholds), 2.6 * 2 * len(analysis_names)),
                         constrained_layout=True, squeeze=False)
for i_row, (name, direction) in enumerate(results):
    for i_col, result in enumerate(results[(name, direction)]):
        ax = axes[i_row, i_col]
        null_sizes = result.null_max_size                           # (n_permutations,)
        bins = np.arange(null_sizes.max() + 2) - 0.5
        ax.hist(null_sizes, bins=bins, color="0.6")
        critical_size = np.quantile(null_sizes, 1 - alpha)
        ax.axvline(critical_size, color="black", linestyle="--", label=f"{1 - alpha:.0%} null")
        if result.components:
            best = result.components[0]
            ax.axvline(best.size, color="red", label=f"observed {best.size:g} (p={best.p_value:.3f})")
        ax.set_yscale("log")
        ax.set_title(f"{name} {direction}, t > {result.primary_threshold:g}", fontsize=9)
        ax.set_xlabel("max component size (edges)", fontsize=8)
        ax.legend(fontsize=7)
fig.savefig(output_dir / "null_distributions.png", dpi=150)
plt.close(fig)

# Each significant component (lowest threshold it appears at): the mean Fisher z
# over its edges, per session, one line per animal, plus the across-animal mean.
significant_runs = [
    (name, direction, result)
    for (name, direction), per_threshold in results.items()
    for result in per_threshold[:1]
    if result.significant
]
if significant_runs:
    fig, axes = plt.subplots(len(significant_runs), len(groups),
                             figsize=(6 * len(groups), 4.5 * len(significant_runs)),
                             sharey="row", constrained_layout=True, squeeze=False)
    for i_row, (name, direction, result) in enumerate(significant_runs):
        component = result.significant[0]
        component_z = session_z[component.edges[:, 0], component.edges[:, 1], :].mean(axis=0)  # (n_sessions,)
        for ax, group in zip(axes[i_row], groups):
            in_group = (sessions["group"] == group).to_numpy()
            for animal in sorted(sessions.loc[in_group, "animal"].unique()):
                animal_rows = np.flatnonzero((sessions["animal"] == animal).to_numpy())
                order = animal_rows[np.argsort(time[animal_rows])]
                ax.plot(time[order], component_z[order], "o-", alpha=0.6, label=animal)
            group_mean = pd.Series(component_z[in_group]).groupby(time[in_group]).mean()
            ax.plot(group_mean.index, group_mean.to_numpy(), "k-", linewidth=3, label="mean")
            ax.set_title(f"{name} {direction} component ({component.size:g} edges, "
                         f"t > {result.primary_threshold:g}, p = {component.p_value:.3f})\n"
                         f"group {group}", fontsize=9)
            ax.set_xlabel(time_variable)
            ax.legend(fontsize=7)
        axes[i_row, 0].set_ylabel("mean Fisher z over component edges")
    fig.savefig(output_dir / "significant_component_trajectories.png", dpi=200)
    plt.close(fig)

# Global check: mean off-diagonal z per session, one line per animal.
off_diagonal = ~np.eye(n_rois, dtype=bool)
sessions["mean_offdiag_z"] = session_z[off_diagonal].mean(axis=0)
fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), sharey=True,
                         constrained_layout=True, squeeze=False)
for ax, group in zip(axes[0], groups):
    for animal, animal_sessions in sessions[sessions["group"] == group].groupby("animal"):
        animal_sessions = animal_sessions.sort_values("time")
        ax.plot(animal_sessions["time"], animal_sessions["mean_offdiag_z"], "o-", label=animal)
    ax.set_title(f"group {group}: mean off-diagonal Fisher z per session")
    ax.set_xlabel(time_variable)
    ax.legend(fontsize=8)
axes[0, 0].set_ylabel("mean off-diagonal z")
fig.savefig(output_dir / "mean_connectivity_by_session.png", dpi=200)
plt.close(fig)
sessions.to_csv(output_dir / "sessions.csv", index=False)
print(f"\noutputs: {output_dir}")
