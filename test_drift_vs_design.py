"""Does the within-recording decline of the hemisphere traces depend on the design?

Many recordings in outputs/hemisphere_traces show the mean-baseline dF/F sliding
down over the 5 min (the running-median panels remove it by construction). This
script turns that into one number per recording and asks which columns of
pixel_data/experimental_design.csv it follows: group, mouse_line, animal, day,
day_index / animal_day (the same thing, as a number and as a label), recording.

Metrics, one per recording, hemispheres averaged (left and right are nearly
identical in every figure):
    drift_emo_pct_per_min     OLS slope of the emo-corrected mean-baseline dF/F0.
    drift_gcamp_pct_per_min   same on the GCaMP-only trace (no reflectance
                              correction), i.e. closest to raw bleaching.
    fluct_log2_ratio          log2(SD last third / SD first third) of the emo
                              20 s running-median dF/F: does the FAST activity
                              shrink too, or only the baseline?

Statistics:
    1. Variance partition (intercept-only mixed model, ML): animal, session
       (animal x day) within animal, recording within session (residual).
       This answers "is the decline a property of the session?".
    2. Mixed model with the design columns as fixed effects, random animal and
       session; each term tested by a likelihood-ratio test (drop one term).
    3. Plain non-parametric tests on session means (Kruskal-Wallis per factor)
       and a Friedman test of t1..t5 within session.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# ---------------------------------------------------------------- parameters
repo_root = Path(__file__).resolve().parent
design_path = repo_root / "pixel_data" / "experimental_design.csv"
traces_root = repo_root / "outputs" / "hemisphere_traces"  # <day>_<animal>/<t#>/hemisphere_traces*.csv
output_dir = repo_root / "outputs" / "drift_vs_design"

emo_trace_file = "hemisphere_traces.csv"
gcamp_trace_file = "hemisphere_traces_gcamp_only.csv"
# 260828_PV7/t2: one-frame reflectance glitch plus a persistent -4 % GCaMP step
# (CLAUDE.md 9.23 / 9.27) -- its slope measures the artefact, not the recording.
exclude_recordings = ["260828_PV7_t2"]
fluct_fraction = 1 / 3  # first / last fraction of the recording compared by fluct_log2_ratio

metric_labels = {  # metric column -> axis label
    "drift_emo_pct_per_min": "drift, emo-corrected (% dF/F per min)",
    "drift_gcamp_pct_per_min": "drift, GCaMP only (% dF/F per min)",
    "fluct_log2_ratio": "log2 SD ratio, last / first third",
}
group_colors = {"PV": "#2a78d6", "R": "#eb6834", "T": "#1baf7a"}  # categorical slots 1-3
group_markers = {"PV": "o", "R": "s", "T": "^"}  # secondary encoding, not colour alone

output_dir.mkdir(parents=True, exist_ok=True)


def fit_nested_model(formula: str, data: pd.DataFrame, reml: bool):
    """Random intercept per animal plus a variance component per session within animal.

    statsmodels' default optimizer (bfgs) stops short on this model and returns a
    worse likelihood -- a reduced model then "beats" the full one. lbfgs, powell and
    nm agree with each other when they converge. They are tried in that order
    (statsmodels moves to the next on failure) and a non-converged fit is an error.
    """
    model = smf.mixedlm(formula, data, groups="animal", re_formula="1",
                        vc_formula={"session": "0 + C(session)"}).fit(reml=reml, method=["lbfgs", "powell", "nm"])
    if not model.converged:
        raise RuntimeError(f"Mixed model did not converge: {formula}")
    return model

# ---------------------------------------------------------------- loading
design = pd.read_csv(design_path)
design = design[~design["recording_id"].isin(exclude_recordings)].reset_index(drop=True)
design["session"] = design["day"].astype(str) + "_" + design["animal"]  # one animal on one day

# ---------------------------------------------------------------- per-recording metrics
# One row per recording: the three metrics above.
metric_rows = []
for row in design.itertuples():
    unit_dir = traces_root / f"{row.day}_{row.animal}" / row.recording
    emo = pd.read_csv(unit_dir / emo_trace_file)
    gcamp = pd.read_csv(unit_dir / gcamp_trace_file)
    time_min = emo["time_s"].to_numpy() / 60.0  # (n_time,)

    emo_dff = 0.5 * (emo["dff_mean_left"] + emo["dff_mean_right"]).to_numpy()  # (n_time,) %
    gcamp_dff = 0.5 * (gcamp["dff_mean_left"] + gcamp["dff_mean_right"]).to_numpy()  # (n_time,) %
    emo_fast = 0.5 * (emo["median_20s_left"] + emo["median_20s_right"]).to_numpy()  # (n_time,) %

    n_edge = int(len(emo_fast) * fluct_fraction)  # frames in the first / last third
    fluct_ratio = np.std(emo_fast[-n_edge:]) / np.std(emo_fast[:n_edge])

    metric_rows.append({
        "recording_id": row.recording_id,
        "drift_emo_pct_per_min": np.polyfit(time_min, emo_dff, 1)[0],
        "drift_gcamp_pct_per_min": np.polyfit(time_min, gcamp_dff, 1)[0],
        "fluct_log2_ratio": np.log2(fluct_ratio),
    })
metrics = design.merge(pd.DataFrame(metric_rows), on="recording_id")
metrics.to_csv(output_dir / "recording_drift_metrics.csv", index=False)
print(f"{len(metrics)} recordings, {metrics['session'].nunique()} sessions, "
      f"{metrics['animal'].nunique()} animals")
# How universal the decline is, and its shape across t1..t5 within a session.
for metric in metric_labels:
    within_session = metrics[metric] - metrics.groupby("session")[metric].transform("mean")
    print(f"  {metric}: mean {metrics[metric].mean():+.3f}, sd {metrics[metric].std():.3f}, "
          f"negative in {(metrics[metric] < 0).mean():.1%} of recordings; "
          f"within-session median by t: "
          + ", ".join(f"{t} {v:+.3f}" for t, v in within_session.groupby(metrics['recording']).median().items()))

# ---------------------------------------------------------------- variance partition
# Intercept-only model: how the metric's variance splits between animal, session
# within animal, and recording within session.
partition_rows = []
for metric in metric_labels:
    null_model = fit_nested_model(f"{metric} ~ 1", metrics, reml=True)
    var_animal = float(null_model.cov_re.iloc[0, 0])
    var_session = float(null_model.vcomp[0])
    var_recording = float(null_model.scale)
    var_total = var_animal + var_session + var_recording
    partition_rows.append({
        "metric": metric,
        "var_animal": var_animal, "var_session": var_session, "var_recording": var_recording,
        "frac_animal": var_animal / var_total,
        "frac_session": var_session / var_total,
        "frac_recording": var_recording / var_total,
        # Share of variance two recordings of the same session have in common.
        "icc_session": (var_animal + var_session) / var_total,
    })
partition = pd.DataFrame(partition_rows)
partition.to_csv(output_dir / "variance_partition.csv", index=False)
print("\nVariance partition (fractions):")
print(partition[["metric", "frac_animal", "frac_session", "frac_recording", "icc_session"]]
      .round(3).to_string(index=False))

# ---------------------------------------------------------------- mixed model, design terms
# Every design column as a fixed effect; animal and session stay random, so a
# session-level factor is not credited with 5 replicates per session.
# animal_day is day_index as a label and animal/day are the random structure,
# so they are not separate terms here (see the Kruskal-Wallis tests below).
fixed_terms = {
    "group": "C(group)",
    "mouse_line": "C(mouse_line)",
    "day_index": "day_index",
    # Categorical: within a session the order effect is not linear (t2 declines
    # most, t5 least), so a single slope would hide it.
    "recording": "C(recording)",
}
term_rows = []
for metric in metric_labels:
    full_formula = f"{metric} ~ " + " + ".join(fixed_terms.values())
    full_model = fit_nested_model(full_formula, metrics, reml=False)
    for term_name, term in fixed_terms.items():
        reduced_formula = f"{metric} ~ " + " + ".join(t for t in fixed_terms.values() if t != term)
        reduced_model = fit_nested_model(reduced_formula, metrics, reml=False)
        lr_stat = 2 * (full_model.llf - reduced_model.llf)
        df_diff = len(full_model.fe_params) - len(reduced_model.fe_params)
        # Slope-type terms get their coefficient; categorical ones their level contrasts.
        coefficients = {name: value for name, value in full_model.fe_params.items()
                        if name.startswith(term)}
        term_rows.append({
            "metric": metric, "term": term_name, "lr_chi2": lr_stat, "df": df_diff,
            "p_value": stats.chi2.sf(lr_stat, df_diff),
            "coefficients": "; ".join(f"{name}={value:+.4f}" for name, value in coefficients.items()),
        })
terms = pd.DataFrame(term_rows)
terms.to_csv(output_dir / "mixed_model_terms.csv", index=False)
print("\nMixed model, likelihood-ratio test per design term:")
print(terms.round(4).to_string(index=False))

# ---------------------------------------------------------------- non-parametric tests
# Session-level factors are tested on session means (109 independent-ish values,
# not 545 pseudo-replicates); recording is tested within session (Friedman).
session_means = (metrics.groupby(["session", "animal", "day", "group", "mouse_line",
                                  "day_index", "animal_day"], as_index=False)[list(metric_labels)]
                 .mean())
factor_rows = []
for metric in metric_labels:
    for factor in ["group", "mouse_line", "animal", "day", "day_index", "animal_day"]:
        samples = [values[metric].to_numpy() for _, values in session_means.groupby(factor)]
        samples = [sample for sample in samples if len(sample) > 0]
        h_stat, p_value = stats.kruskal(*samples)
        factor_rows.append({"metric": metric, "factor": factor, "unit": "session mean",
                            "n_levels": len(samples), "statistic": h_stat, "p_value": p_value})
    # recordings: sessions x t1..t5 table
    by_recording = metrics.pivot(index="session", columns="recording", values=metric).dropna()
    chi2_stat, p_value = stats.friedmanchisquare(*[by_recording[col] for col in by_recording])
    factor_rows.append({"metric": metric, "factor": "recording", "unit": "within session",
                        "n_levels": by_recording.shape[1], "statistic": chi2_stat, "p_value": p_value})
factor_tests = pd.DataFrame(factor_rows)
factor_tests.to_csv(output_dir / "factor_tests.csv", index=False)
print("\nKruskal-Wallis on session means / Friedman within session:")
print(factor_tests.round(4).to_string(index=False))

# Is the decline a session property? mouse_line within the groups that have both lines.
print("\nmouse_line within group (Mann-Whitney on session means):")
for group_name in ["R", "T"]:
    subset = session_means[session_means["group"] == group_name]
    for metric in metric_labels:
        c57 = subset.loc[subset["mouse_line"] == "C57", metric]
        pv_cre = subset.loc[subset["mouse_line"] == "PV-CRE", metric]
        u_stat, p_value = stats.mannwhitneyu(c57, pv_cre)
        print(f"  {group_name} {metric}: C57 median {c57.median():+.3f} (n={len(c57)}) vs "
              f"PV-CRE {pv_cre.median():+.3f} (n={len(pv_cre)}), p={p_value:.4f}")

# ---------------------------------------------------------------- figure 1: metric by factor
# One figure per metric: every recording as a dot, coloured and shaped by group,
# with the median of each level as a black bar.
animal_order = (metrics[["group", "animal"]].drop_duplicates()
                .sort_values(["group", "animal"])["animal"].tolist())
factor_orders = {
    "group": sorted(metrics["group"].unique()),
    "mouse_line": sorted(metrics["mouse_line"].unique()),
    "day_index": sorted(metrics["day_index"].unique()),
    "recording": sorted(metrics["recording"].unique()),
    "animal": animal_order,
    "day": sorted(metrics["day"].unique()),
}
rng = np.random.default_rng(0)  # jitter only
for metric, metric_label in metric_labels.items():
    fig, axes = plt.subplot_mosaic(
        [["group", "mouse_line", "day_index", "recording"], ["animal"] * 4, ["day"] * 4],
        figsize=(14, 11), constrained_layout=True)
    for factor, levels in factor_orders.items():
        ax = axes[factor]
        for i_level, level in enumerate(levels):
            level_rows = metrics[metrics[factor] == level]
            for group_name, group_rows in level_rows.groupby("group"):
                jitter = rng.uniform(-0.25, 0.25, len(group_rows))
                ax.scatter(i_level + jitter, group_rows[metric], s=10, alpha=0.6,
                           color=group_colors[group_name], marker=group_markers[group_name],
                           linewidths=0)
            ax.hlines(level_rows[metric].median(), i_level - 0.35, i_level + 0.35,
                      color="black", linewidth=2)
        ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
        ax.set_xticks(range(len(levels)), [str(level) for level in levels],
                      rotation=90 if factor in ("animal", "day") else 0)
        ax.set_xlabel(factor)
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", color="0.92")
    handles = [plt.Line2D([], [], color=color, marker=group_markers[name], linestyle="", label=name)
               for name, color in group_colors.items()]
    axes["group"].legend(handles=handles, title="group", fontsize=8)
    fig.suptitle(f"{metric_label} by design column ({len(metrics)} recordings, "
                 f"black bar = median)")
    fig.savefig(output_dir / f"{metric}_by_factor.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- figure 2: variance partition
fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
component_colors = {"frac_animal": "#2a78d6", "frac_session": "#eb6834", "frac_recording": "#b0b0a8"}
component_labels = {"frac_animal": "animal", "frac_session": "session within animal",
                    "frac_recording": "recording within session"}
for i_metric, metric in enumerate(metric_labels):
    left = 0.0
    for component, color in component_colors.items():
        width = partition.loc[partition["metric"] == metric, component].item()
        ax.barh(i_metric, width, left=left, color=color, edgecolor="white", linewidth=2,
                label=component_labels[component] if i_metric == 0 else None)
        if width >= 0.05:  # narrower segments have no room for a label
            ax.text(left + width / 2, i_metric, f"{width:.0%}", ha="center", va="center", fontsize=8)
        left += width
ax.set_yticks(range(len(metric_labels)), list(metric_labels.values()))
ax.set_xlim(0, 1)
ax.set_xlabel("fraction of variance")
ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8)
ax.set_title("Variance partition of each metric (mixed model, REML)")
fig.savefig(output_dir / "variance_partition.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 3: t1..t5 within session
# One thin line per session across its five recordings, group median on top.
fig, axes = plt.subplots(1, len(metric_labels), figsize=(15, 4.5), constrained_layout=True,
                         squeeze=False)
for i_metric, (metric, metric_label) in enumerate(metric_labels.items()):
    ax = axes[0, i_metric]
    for session, session_rows in metrics.groupby("session"):
        session_rows = session_rows.sort_values("recording_index")
        ax.plot(session_rows["recording_index"], session_rows[metric], linewidth=0.6, alpha=0.35,
                color=group_colors[session_rows["group"].iloc[0]])
    for group_name, group_rows in metrics.groupby("group"):
        group_median = group_rows.groupby("recording_index")[metric].median()
        ax.plot(group_median.index, group_median.to_numpy(), linewidth=2.5,
                marker=group_markers[group_name], color=group_colors[group_name],
                label=f"{group_name} median")
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_xticks(range(1, 6), [f"t{i}" for i in range(1, 6)])
    ax.set_xlabel("recording within session")
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", color="0.92")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Recording order within each session (thin: one session)")
fig.savefig(output_dir / "metrics_by_recording_order.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 4: animal x day map
# Session means of the emo drift, so a session-specific decline shows as one cell.
heatmap_metric = "drift_emo_pct_per_min"
heatmap = (session_means.pivot(index="animal", columns="day_index", values=heatmap_metric)
           .reindex(animal_order))  # (n_animals, n_day_index)
color_limit = np.nanmax(np.abs(heatmap.to_numpy()))
fig, ax = plt.subplots(figsize=(6, 9), constrained_layout=True)
image = ax.imshow(heatmap.to_numpy(), cmap="coolwarm", vmin=-color_limit, vmax=color_limit,
                  aspect="auto")
for (i_row, i_col), value in np.ndenumerate(heatmap.to_numpy()):
    if np.isfinite(value):
        ax.text(i_col, i_row, f"{value:+.2f}", ha="center", va="center", fontsize=7)
group_of_animal = metrics.drop_duplicates("animal").set_index("animal")["group"]
ax.set_yticks(range(len(animal_order)), [f"{animal} ({group_of_animal[animal]})" for animal in animal_order])
ax.set_xticks(range(heatmap.shape[1]), [f"day{day}" for day in heatmap.columns])
ax.set_xlabel("day_index")
fig.colorbar(image, ax=ax, label=metric_labels[heatmap_metric], shrink=0.6)
ax.set_title(f"Session mean {heatmap_metric}\n(mean of t1..t5)")
fig.savefig(output_dir / "drift_emo_animal_by_day.png", dpi=200)
plt.close(fig)

print(f"\nOutputs in {output_dir}")
