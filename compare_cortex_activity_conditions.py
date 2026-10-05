"""Exposure-aware condition comparisons from saved cortex tables, no image rereads."""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix

ROOT = Path(__file__).resolve().parent
RUN_CONFIG = dict(output_root=str(ROOT/"outputs/cortex_activity_emo"), bootstrap=500, seed=1729)


def aggregate_exposure(frame, keys):
    rows = []
    for key, data in frame.groupby(keys, observed=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        exposure = data.valid_seconds.sum()
        row = dict(zip(keys,key), valid_seconds=exposure,
                   occupied_seconds=data.occupied_seconds.sum(), event_count=data.event_count.sum())
        row["burden"] = row["occupied_seconds"]/exposure if exposure else np.nan
        row["events_per_minute"] = 60*row["event_count"]/exposure if exposure else np.nan
        for name in ("residual_rms", "mean_positive_pixel_activity", "mean_recruitment",
                     "mean_event_amplitude", "mean_peak_recruitment"):
            if name in data:
                good = data[name].notna() & (data.valid_seconds > 0)
                weights = data.loc[good,"event_count"] if name.startswith("mean_event") or name == "mean_peak_recruitment" else data.loc[good,"valid_seconds"]
                row[name] = np.average(data.loc[good,name],weights=weights) if weights.sum()>0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_contrasts(animal, n_boot, rng):
    """Animal-balanced descriptive contrasts: resample whole animal summaries."""
    results = []
    for endpoint in ("burden", "events_per_minute", "mean_event_amplitude", "mean_peak_recruitment", "residual_rms"):
        for a,b in combinations(sorted(animal.group.unique()),2):
            x = animal.loc[animal.group == a,endpoint].dropna().to_numpy()
            y = animal.loc[animal.group == b,endpoint].dropna().to_numpy()
            draws = [rng.choice(x,len(x)).mean()-rng.choice(y,len(y)).mean() for _ in range(n_boot)] if min(len(x),len(y))>=2 else []
            ci = np.quantile(draws,[.025,.975]) if draws else [np.nan,np.nan]
            results.append(dict(endpoint=endpoint, group_a=a, group_b=b, contrast="a minus b",
                estimate=x.mean()-y.mean() if len(x) and len(y) else np.nan, ci_low=ci[0],ci_high=ci[1],
                animals_a=len(x),animals_b=len(y), adjustment="unadjusted animal-balanced",
                status="ok" if draws else "insufficient animals for bootstrap"))
    return results


def fit_models(data, n_boot, rng, root):
    """Independence-working GEE, robust animal covariance; shared mouse-line support."""
    report, contrasts, trajectories, coefficients = [], [], [], []
    pd.DataFrame(columns=["endpoint","term","estimate","robust_se"]).to_csv(root/"model_coefficients.csv",index=False)
    pd.DataFrame(columns=["endpoint","time_min","group","estimate"]).to_csv(root/"adjusted_trajectories.csv",index=False)
    groups = sorted(data.group.unique())
    if len(groups)<2:
        return [], [{"status":"need at least two groups"}]
    # No predictions in PV x C57 cells: restrict to lines observed in EVERY group.
    lines = set.intersection(*(set(data.loc[data.group==g,"mouse_line"]) for g in groups))
    d = data[data.mouse_line.isin(lines) & (data.valid_seconds>0)].copy()
    cell_columns = ["mouse_line","day_index","recording_index"]
    cells = set.intersection(*(set(map(tuple,d.loc[d.group==g,cell_columns].to_numpy())) for g in groups))
    d = d.loc[np.array([tuple(cell) in cells for cell in d[cell_columns].to_numpy()], dtype=bool)].copy()
    counts = d.groupby("group").animal.nunique()
    audit = dict(common_mouse_lines=sorted(lines), supported_covariate_cells=len(cells), animals_per_group=counts.to_dict())
    if len(counts)!=len(groups) or counts.min()<3 or d.animal.nunique()<8:
        return [], [dict(audit,status="insufficient common-support animals: require >=3/group and >=8 total")]
    d["time_min"] = d.midpoint_s/60
    terms = ["C(group) * time_min"]
    for col,term in (("mouse_line","C(mouse_line)"),("day_index","day_index"),("recording_index","C(recording_index)")):
        if d[col].nunique()>1:
            terms.append(term)
    rhs = " + ".join(terms)
    design_info = dmatrix(rhs, d).design_info
    # Equal animal weighting for marginal predictions, same supported distribution for all groups.
    population = d.copy()
    popweights = 1/population.groupby("animal").animal.transform("size").to_numpy()
    popweights /= popweights.sum()
    common_time = sorted(set.intersection(*(set(d.loc[d.group==g,"time_min"]) for g in groups)))
    if not common_time:
        return [], [dict(audit,status="no common time support")]
    for endpoint, family in (("event_count",sm.families.Poisson()),
                             ("burden",sm.families.Binomial()),
                             ("event_count_nb",sm.families.NegativeBinomial(alpha=1.0))):
        target = "event_count" if endpoint.startswith("event_count") else endpoint
        offset = np.log(d.valid_seconds.to_numpy()/60) if target=="event_count" else None
        weights = d.valid_seconds.to_numpy()/d.valid_seconds.mean() if target=="burden" else None
        try:
            if d[target].nunique() < 2:
                raise ValueError("constant outcome; model not identifiable")
            model = smf.gee(f"{target} ~ {rhs}","animal",d, family=family, offset=offset, weights=weights,
                            cov_struct=sm.cov_struct.Independence())
            if list(design_info.column_names) != list(model.exog_names):
                raise ValueError("Formula design columns differ between model and prediction")
            if np.linalg.matrix_rank(model.exog)<model.exog.shape[1]:
                raise ValueError("rank-deficient design; no adjusted contrasts reported")
            with warnings.catch_warnings(record=True) as caught:
                fit = model.fit(maxiter=100)
            if not fit.converged or not np.isfinite(fit.params).all():
                raise ValueError("GEE did not converge to finite parameters")
            report.append(dict(audit,endpoint=endpoint,status="ok",formula=f"{target} ~ {rhs}",
                warnings=[str(w.message) for w in caught],
                pearson_dispersion=float(np.sum(fit.resid_pearson**2)/max(1,fit.df_resid)),
                negative_binomial_alpha=1.0 if endpoint.endswith("nb") else None))
            coefficients.extend(dict(endpoint=endpoint,term=k,estimate=fit.params[k],robust_se=fit.bse[k]) for k in fit.params.index)
            designs = {}
            for time in common_time:
                for group in groups:
                    pred = population.assign(group=group,time_min=time)
                    matrix = np.asarray(build_design_matrices([design_info], pred)[0])
                    designs[(time,group)] = matrix
                    value = float(np.average(family.link.inverse(matrix@fit.params.to_numpy()),weights=popweights))
                    trajectories.append(dict(endpoint=endpoint,time_min=time,group=group,estimate=value))
            # Resample complete animals within group; independence GEE and weighted GLM share coefficients.
            # Full visits/recordings remain intact in each bootstrap draw.
            draws = {key:[] for key in designs}
            for _ in range(n_boot):
                blocks = []
                for group in groups:
                    animals = d.loc[d.group==group,"animal"].unique()
                    blocks.extend(d[d.animal==a] for a in rng.choice(animals,len(animals),replace=True))
                sample = pd.concat(blocks,ignore_index=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bootmodel = smf.glm(f"{target} ~ {rhs}",sample,family=family,
                            offset=np.log(sample.valid_seconds/60) if target=="event_count" else None,
                            var_weights=sample.valid_seconds/d.valid_seconds.mean() if target=="burden" else None)
                        if list(bootmodel.exog_names)!=list(model.exog_names) or np.linalg.matrix_rank(bootmodel.exog)<bootmodel.exog.shape[1]:
                            continue
                        boot = bootmodel.fit(maxiter=100)
                    if not boot.converged or not np.isfinite(boot.params).all():
                        continue
                    for key,matrix in designs.items():
                        draws[key].append(float(np.average(family.link.inverse(matrix@boot.params.to_numpy()),weights=popweights)))
                except (ValueError, np.linalg.LinAlgError):
                    continue
            for time in common_time:
                for a,b in combinations(groups,2):
                    delta = np.array(draws[(time,a)])-np.array(draws[(time,b)])
                    stable = len(delta)>=max(30,.8*n_boot)
                    ci = np.quantile(delta,[.025,.975]) if stable else [np.nan,np.nan]
                    estimates = {g:np.average(family.link.inverse(designs[(time,g)]@fit.params.to_numpy()),weights=popweights) for g in (a,b)}
                    contrasts.append(dict(endpoint=endpoint,time_min=time,group_a=a,group_b=b,estimate=estimates[a]-estimates[b],
                        ci_low=ci[0],ci_high=ci[1],adjustment="GEE common-line support, animal-standardized",
                        bootstrap_success=len(delta),status="ok" if stable else "unstable bootstrap"))
        except (ValueError, np.linalg.LinAlgError) as exc:
            report.append(dict(audit,endpoint=endpoint,status=str(exc)))
    pd.DataFrame(coefficients).to_csv(root/"model_coefficients.csv",index=False)
    pd.DataFrame(trajectories).to_csv(root/"adjusted_trajectories.csv",index=False)
    return contrasts,report


def run(output_root=RUN_CONFIG["output_root"], bootstrap=500, seed=1729):
    if bootstrap<1:
        raise ValueError("bootstrap must be positive")
    root = Path(output_root)
    if not (root/"run_complete.json").exists():
        raise ValueError("No completed analysis manifest; finish the runner before comparing conditions")
    rec = pd.read_csv(root/"recording_metrics.csv")
    win = pd.concat([chunk[chunk.region == "cortex"] for chunk in
                     pd.read_csv(root/"window_metrics.csv",chunksize=100_000)],ignore_index=True)
    completed = json.loads((root/"run_complete.json").read_text())
    if set(rec.recording_id) != set(completed["recording_ids"]) or set(win.recording_id) != set(completed["recording_ids"]):
        raise ValueError("Recording tables do not match completed run")
    design = rec.drop_duplicates("recording_id")
    pd.crosstab([design.group,design.mouse_line],design.day.astype(str).str[:4]).to_csv(root/"design_overlap.csv")
    keys = ["group","animal","day","day_index","mouse_line","method","support","threshold_factor","region"]
    day = aggregate_exposure(rec, keys)
    day.to_csv(root/"animal_day_metrics.csv",index=False)
    animal = aggregate_exposure(rec,[k for k in keys if k not in ("day","day_index")])
    animal.to_csv(root/"animal_metrics.csv",index=False)
    selected = (animal.method=="additive_60s") & (animal.support=="common") & (animal.threshold_factor==1) & (animal.region=="cortex")
    rng = np.random.default_rng(seed)
    contrasts = bootstrap_contrasts(animal[selected],bootstrap,rng)
    primary = win[(win.method=="additive_60s") & (win.support=="common") & (win.threshold_factor==1) & (win.region=="cortex") & (win.valid_seconds>0)]
    adjusted,report = fit_models(primary,bootstrap,rng,root)
    pd.DataFrame(contrasts+adjusted).to_csv(root/"condition_contrasts.csv",index=False)
    (root/"model_diagnostics.json").write_text(json.dumps(dict(models=report,bootstrap=bootstrap,seed=seed,
        interpretation="Exploratory candidate calcium events; thresholds unvalidated; shared mouse-line contrasts only"),indent=2),encoding="utf-8")
    trajectories = aggregate_exposure(win[win.region=="cortex"],["group","animal","method","support","threshold_factor","midpoint_s"])
    trajectories.to_csv(root/"animal_time_metrics.csv",index=False)
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for group,data in trajectories[(trajectories.method=="additive_60s") & (trajectories.support=="common") & (trajectories.threshold_factor==1)].groupby("group"):
        for endpoint,ax in zip(("burden","events_per_minute"),axes):
            summary = data.groupby("midpoint_s")[endpoint].agg(["mean","sem"])
            ax.errorbar(summary.index,summary["mean"],yerr=summary["sem"].fillna(0),label=group)
            ax.set(xlabel="seconds in recording",ylabel=endpoint)
            ax.legend()
    fig.suptitle("Sampled cortical ROI union | animal means +/- SEM | exploratory thresholds")
    fig.tight_layout(); fig.savefig(root/"condition_time.png",dpi=150); plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4))
    for (group,method),d in animal[(animal.region=="cortex") & (animal.support=="common")].groupby(["group","method"]):
        summary = d.groupby("threshold_factor").burden.mean()
        ax.plot(summary.index,summary.values,marker="o",label=f"{group} {method}")
    ax.set(xlabel="common native amplitude threshold multiplier",ylabel="animal mean event burden")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(root/"threshold_sensitivity.png",dpi=150); plt.close(fig)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",default=RUN_CONFIG["output_root"])
    p.add_argument("--bootstrap",type=int,default=RUN_CONFIG["bootstrap"])
    p.add_argument("--seed",type=int,default=RUN_CONFIG["seed"])
    run(**vars(p.parse_args()))
