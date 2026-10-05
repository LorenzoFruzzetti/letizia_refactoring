"""Plot completed additive cortex/emo results without rereading raw image volumes.

Run directly from the editor using RUN_CONFIG, or use --input-root on the CLI.
Auto selection prefers a completed cohort, otherwise the completed pilot.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN_CONFIG = dict(input_root=None, baseline_s=60.0, threshold_factor=1.0, dpi=180)
COLORS = {"PV": "#2a78d6", "R": "#eb6834", "T": "#1baf7a"}
ENDPOINTS = {
    "burden": "Candidate-event burden (fraction)",
    "events_per_minute": "Candidate events / valid minute",
    "residual_rms": "Residual RMS (F/emo ratio)",
    "mean_event_amplitude": "Mean event peak (F/emo ratio)",
    "mean_peak_recruitment": "Mean event-peak recruitment (fraction)",
    "mean_positive_pixel_activity": "Positive pixel activity (F/emo ratio)",
}


def aggregate(data, keys):
    """Pool valid exposure within animal; never count recordings as animals."""
    rows = []
    for key, block in data.groupby(keys, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        exposure = block.valid_seconds.sum()
        row = dict(zip(keys, key), valid_seconds=exposure)
        row["burden"] = block.occupied_seconds.sum()/exposure if exposure else np.nan
        row["events_per_minute"] = 60*block.event_count.sum()/exposure if exposure else np.nan
        for endpoint in ENDPOINTS:
            if endpoint in ("burden", "events_per_minute") or endpoint not in block:
                continue
            weights = block.event_count if endpoint in ("mean_event_amplitude", "mean_peak_recruitment") else block.valid_seconds
            good = np.isfinite(block[endpoint]) & (weights > 0)
            values = block.loc[good, endpoint].to_numpy()
            if endpoint == "residual_rms":
                values = values**2
            value = np.average(values, weights=weights[good]) if good.any() else np.nan
            row[endpoint] = np.sqrt(value) if endpoint == "residual_rms" else value
        rows.append(row)
    return pd.DataFrame(rows)


def run(input_root=None, baseline_s=60.0, threshold_factor=1.0, dpi=180, allow_incomplete=False):
    if input_root is None:
        candidates = [ROOT/"outputs/cortex_activity_emo", ROOT/"outputs/cortex_activity_emo_pilot"]
        input_root = next((p for p in candidates if (p/"run_complete.json").exists()), None)
        if input_root is None:
            raise FileNotFoundError("No completed cortex/emo run found")
    source = Path(input_root)
    config = json.loads((source/"config.json").read_text())
    omissions = []
    if (source/"run_complete.json").exists():
        manifest = json.loads((source/"run_complete.json").read_text())
        table_roots = [source]
    elif allow_incomplete:
        from analyze_cortex_activity_emo import cache_matches, RUN_CONFIG as ANALYSIS_CONFIG
        settings = dict(ANALYSIS_CONFIG, **config["config"])
        settings["output_root"] = str(source.resolve())
        settings["overwrite"] = False
        design = pd.read_csv(Path(settings["pixel_root"])/"experimental_design.csv",dtype={"day":str})
        requested = set(config["recording_ids"])
        table_roots, completed = [], []
        for row in design[design.recording_id.isin(requested)].to_dict("records"):
            if cache_matches(row,settings):
                completed.append(row["recording_id"])
                table_roots.append(source/row["pixel_dir"])
            else:
                omissions.append(row["recording_id"])
        if not completed:
            raise ValueError("No compatible completed recording caches found")
        manifest = dict(recording_ids=completed,selection_mode="compatible completed subset",
                        requested_recordings=len(requested),omitted_recording_ids=omissions)
        print(f"Plotting {len(completed)}/{len(requested)} recordings; omitted: {omissions}",flush=True)
    else:
        raise ValueError("Run incomplete. Use --allow-incomplete to explicitly plot validated completed caches.")
    rec = pd.concat([pd.read_csv(folder/"recording_metrics.csv") for folder in table_roots],ignore_index=True)
    if set(rec.recording_id) != set(manifest["recording_ids"]):
        raise ValueError("Recording metrics do not match completed run")
    if ((rec.occupied_seconds > rec.valid_seconds+1e-8) | (rec.valid_seconds < 0)).any():
        raise ValueError("Invalid exposure in saved metrics")
    method = f"additive_{baseline_s:g}s"
    primary = rec[(rec.method == method) & (rec.support == "common") &
                  np.isclose(rec.threshold_factor, threshold_factor)].copy()
    if primary.empty:
        raise ValueError("Requested method / threshold is absent from the saved results")
    out = source/"plots"/f"{baseline_s:g}s_factor{threshold_factor:g}"
    out.mkdir(parents=True, exist_ok=True)
    groups = sorted(primary.group.unique(), key=lambda g: (g not in COLORS, g))
    counts = primary.drop_duplicates("recording_id").groupby("group").agg(
        animals=("animal", "nunique"), recordings=("recording_id", "nunique"))
    cohort = primary.drop_duplicates("recording_id")
    small = counts.animals.min() < 3
    status = "PILOT / descriptive only" if small else "Descriptive animal summaries"
    if omissions:
        status += f"; {len(omissions)} incomplete recordings omitted"
    subtitle = f"{status} | {len(cohort)} recordings, {cohort.animal.nunique()} animals | {baseline_s:g}s median | threshold x{threshold_factor:g}"
    footnote = "Sampled cortical ROI union; common valid support; exploratory candidate-event thresholds"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlesize": 11, "svg.fonttype": "none"})
    gallery = []

    def save(fig, stem, title, description):
        fig.suptitle(title+"\n"+subtitle, fontsize=13)
        fig.supxlabel(footnote, fontsize=9)
        fig.savefig(out/f"{stem}.png", dpi=dpi)
        fig.savefig(out/f"{stem}.svg")
        plt.close(fig)
        gallery.append(dict(stem=stem, title=title, description=description))

    # One dot per animal, preserving the earlier study's blue/orange/green palette.
    animals = aggregate(primary[primary.region == "cortex"], ["group", "animal"])
    animals.to_csv(out/"plotted_animal_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True, squeeze=False)
    for ax, (endpoint, label) in zip(axes.flat, ENDPOINTS.items()):
        for x, group in enumerate(groups):
            data = animals[animals.group == group].dropna(subset=[endpoint])
            offsets = np.linspace(-.15, .15, len(data)) if len(data)>1 else [0]*len(data)
            ax.scatter(x+np.array(offsets), data[endpoint], color=COLORS.get(group, "gray"), s=45)
            if len(data):
                ax.plot([x-.24, x+.24], [data[endpoint].median()]*2, color="black", lw=2)
            if small:
                for offset, (_, row) in zip(offsets, data.iterrows()):
                    ax.annotate(row.animal, (x+offset, row[endpoint]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xticks(range(len(groups)), [f"{g}\n{counts.loc[g,'animals']} animal(s)\n{counts.loc[g,'recordings']} rec." for g in groups])
        ax.set_title(label)
        ax.grid(axis="y", alpha=.15)
        ax.set_ylim(bottom=0)
    save(fig, "activity_by_group", "Activity by group: dots = animals, black bars = medians",
         "Exposure pooled within each animal; event peaks weighted by event count. No inferential error bars for this descriptive plot.")

    # Read large cohort tables in chunks, retaining only the selected cortical rows.
    chunks = []
    for folder in table_roots:
        for chunk in pd.read_csv(folder/"window_metrics.csv", chunksize=100_000):
            chunks.append(chunk[(chunk.region == "cortex") & (chunk.method == method) &
                                (chunk.support == "common") & np.isclose(chunk.threshold_factor, threshold_factor) &
                                (chunk.valid_seconds > 0)])
    windows = pd.concat(chunks, ignore_index=True)
    if set(windows.recording_id)-set(manifest["recording_ids"]):
        raise ValueError("Unexpected recordings in window metrics")
    trajectories = aggregate(windows, ["group", "animal", "midpoint_s"])
    trajectories.to_csv(out/"plotted_animal_time.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, squeeze=False)
    for ax, endpoint in zip(axes.flat, ("burden", "events_per_minute", "residual_rms", "mean_positive_pixel_activity")):
        for group in groups:
            data = trajectories[trajectories.group == group]
            for _, animal in data.groupby("animal"):
                ax.plot(animal.midpoint_s, animal[endpoint], color=COLORS.get(group), alpha=.25, lw=1)
            summary = data.groupby("midpoint_s")[endpoint].mean()
            ax.plot(summary.index, summary, color=COLORS.get(group), marker="o", label=group, lw=2)
        ax.set(xlabel="Time in recording (s; bin centers)", ylabel=ENDPOINTS[endpoint])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15); ax.legend()
    save(fig, "activity_over_time", "Activity during the recording: thin = animals, thick = group means",
         "Only bins with valid exposure are shown. Group means weight animals equally; RMS is pooled from squared residual RMS.")

    roi = aggregate(primary[~primary.region.isin(["cortex", "left", "right"])], ["group", "animal", "region"])
    roi.to_csv(out/"plotted_roi_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 10), constrained_layout=True, squeeze=False)
    for ax, endpoint in zip(axes.flat, ("burden", "residual_rms")):
        table = roi.groupby(["region", "group"])[endpoint].mean().unstack().reindex(columns=groups)
        im = ax.imshow(np.ma.masked_invalid(table.to_numpy()), aspect="auto", cmap="Blues", vmin=0,
                       vmax=1 if endpoint == "burden" else None)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_yticks(range(len(table)), table.index)
        ax.set_title(ENDPOINTS[endpoint]); fig.colorbar(im, ax=ax, shrink=.75)
    save(fig, "activity_by_roi", "Regional activity: animal-balanced group means",
         "Individual saved ROI boxes. ROI event intervals can overlap and must not be summed as independent cortical events.")

    sensitivity = aggregate(rec[(rec.region == "cortex") & (rec.support == "common")],
                            ["group", "animal", "baseline_s", "threshold_factor"])
    sensitivity.to_csv(out/"plotted_sensitivity.csv", index=False)
    methods = sorted(sensitivity.baseline_s.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 4.8), constrained_layout=True, squeeze=False)
    for ax, seconds in zip(axes.flat, methods):
        for group in groups:
            data = sensitivity[(sensitivity.group == group) & (sensitivity.baseline_s == seconds)]
            for _, animal in data.groupby("animal"):
                ax.plot(animal.threshold_factor, animal.burden, color=COLORS.get(group), alpha=.2)
            summary = data.groupby("threshold_factor").burden.mean()
            ax.plot(summary.index, summary, marker="o", color=COLORS.get(group), label=group)
        ax.set(title=f"{seconds:g} s baseline", xlabel="Pixel threshold multiplier", ylabel="Candidate-event burden", ylim=(0, 1))
        ax.legend(); ax.grid(alpha=.15)
    save(fig, "baseline_threshold_sensitivity", "Sensitivity to baseline window and detection threshold",
         "Same common-support setting for each method; thresholds are not selected to maximize condition differences.")

    # Select one example per group deterministically, never by largest activity.
    examples = cohort.sort_values("recording_id").groupby("group", sort=False).head(1)
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 3.1*len(examples)+1), constrained_layout=True, squeeze=False)
    example_rows = []
    for row_index, (_, recording) in enumerate(examples.iterrows()):
        with np.load(source/recording.pixel_dir/f"traces_{baseline_s:g}s.npz", allow_pickle=False) as trace:
            j = list(trace["region_names"]).index("cortex")
            time = trace["time_s"]
            good = trace["common_support"] & trace["common_valid"][:,j] & trace["valid"][:,j]
            color = COLORS.get(recording.group, "gray")
            axes[row_index, 0].plot(time, trace["q"][:,j], color=color, lw=.6)
            axes[row_index, 0].plot(time, trace["b"][:,j], color="black", lw=1.4, label="Baseline")
            axes[row_index, 1].plot(time, trace["corrected"][:,j], color=color, lw=.6)
            axes[row_index, 1].axhline(trace["mean_b"][j], color="black", ls="--", lw=1)
            # Same y limits for raw and corrected make amplitude preservation visible.
            limits = np.r_[trace["q"][:,j], trace["corrected"][:,j]]
            finite = limits[np.isfinite(limits)]
            if len(finite):
                span = max(np.ptp(finite), 1e-6)
                for ax in axes[row_index, :2]:
                    ax.set_ylim(finite.min()-.05*span, finite.max()+.05*span)
            factors = trace["threshold_factors"]
            factor_index = np.flatnonzero(np.isclose(factors, threshold_factor))[0]
            axes[row_index, 2].plot(time, np.where(good, trace["recruitment"][factor_index,:,j], np.nan), color=color, lw=.7)
            for ax, title in zip(axes[row_index], ("Raw ratio + running baseline", "Corrected ratio + mean baseline", "Pixel recruitment on valid support")):
                ax.set(title=title, xlabel="Time in recording (s)")
            axes[row_index, 0].set_ylabel(f"{recording.recording_id}\nF/emo ratio")
            axes[row_index, 1].set_ylabel("F/emo ratio")
            axes[row_index, 2].set(ylabel="Active-pixel fraction", ylim=(0, 1))
            example_rows.append(dict(recording_id=recording.recording_id, group=recording.group,
                                     valid_seconds=good.sum()/config["config"]["sampling_rate_hz"]))
    save(fig, "correction_examples", "Baseline correction examples: first recording ID in each group",
         "Per-pixel correction followed by unique-pixel ROI-union averaging. Raw and corrected panels share a vertical scale within each recording.")

    qc = pd.concat([pd.read_csv(folder/"recording_qc.csv") for folder in table_roots],ignore_index=True)
    qc = qc[(qc.status == "included") & np.isclose(qc.baseline_s, baseline_s) & (qc.channel == "ratio")]
    drift = qc.drop_duplicates("recording_id")[["recording_id", "baseline_slope_per_min"]]
    scatter = primary[primary.region == "cortex"].merge(drift, on="recording_id", validate="one_to_one")
    scatter.to_csv(out/"plotted_drift_activity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, squeeze=False)
    for ax, endpoint in zip(axes.flat, ("burden", "residual_rms")):
        for group in groups:
            d = scatter[scatter.group == group]
            ax.scatter(d.baseline_slope_per_min, d[endpoint], label=group, color=COLORS.get(group), alpha=.65)
            if small:
                for _, row in d.iterrows():
                    ax.annotate(row.animal, (row.baseline_slope_per_min, row[endpoint]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.axvline(0, color="gray", lw=.8, ls="--")
        ax.set(xlabel="Ratio baseline slope (F/emo ratio per minute)", ylabel=ENDPOINTS[endpoint])
        ax.legend(); ax.grid(alpha=.15)
    save(fig, "drift_vs_activity", "Baseline decline versus residual activity: dots = recordings",
         "QC slope uses the median of the spatial-mean ratio, as labelled in the saved diagnostic table. No regression or independent-recording inference is applied.")

    cards = []
    for item in gallery:
        stem = item["stem"]
        cards.append(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["description"])}</p>'
                     f'<a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(item["title"])}"></a>'
                     f'<p><a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></p></section>')
    (out/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Cortex activity plots</title>'
        '<style>body{max-width:1350px;margin:35px auto;padding:0 24px;font:16px system-ui;color:#263238;background:#fafafa}'
        'section{background:white;padding:20px;margin:24px 0;border:1px solid #ddd;border-radius:8px}img{width:100%}a{color:#1769aa}</style>'
        '<h1>Cortex activity relative to emo</h1><p>'+html.escape(subtitle)+'</p><p>'+html.escape(footnote)+'</p>'
        '<p>These are descriptive plots of the completed selection. Calcium candidate-event burden is not a validated epilepsy severity score.</p>'
        +'<p>Omitted recordings: '+html.escape(', '.join(omissions) or 'none from selected run')+'</p>'+''.join(cards)+'</html>', encoding="utf-8")
    (out/"plot_manifest.json").write_text(json.dumps(dict(input_root=str(source.resolve()),
        recording_ids=manifest["recording_ids"], omitted_recording_ids=omissions,
        selection_mode=manifest.get("selection_mode","completed run"), baseline_s=baseline_s, threshold_factor=threshold_factor,
        groups=counts.reset_index().to_dict("records"), examples=example_rows, plots=gallery), indent=2), encoding="utf-8")
    print(f"Saved {len(gallery)} figures (PNG + SVG) and gallery: {out/'index.html'}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=RUN_CONFIG["input_root"])
    parser.add_argument("--baseline-s", type=float, default=RUN_CONFIG["baseline_s"])
    parser.add_argument("--threshold-factor", type=float, default=RUN_CONFIG["threshold_factor"])
    parser.add_argument("--dpi", type=int, default=RUN_CONFIG["dpi"])
    parser.add_argument("--allow-incomplete", action="store_true", help="Plot compatible completed caches and explicitly report missing recordings")
    run(**vars(parser.parse_args()))
