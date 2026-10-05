"""Candidate calcium-event burden in the sampled cortical ROI union.

Editor: edit RUN_CONFIG. CLI: python analyze_cortex_activity_emo.py --help.
All signal operations precede spatial averaging; no dF/F, GSR or second detrend.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import hashlib
import json
import platform
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from epileptic_by_area_animal_day_pixels import (
    crop_slices, median_window_frames, recording_roi_set, resolve_boxes,
)
from epileptic_by_active_pixels import hemisphere_masks

ROOT = Path(__file__).resolve().parent
RUN_CONFIG = dict(
    pixel_root=str(ROOT / "pixel_data"),
    output_root=str(ROOT / "outputs/cortex_activity_emo"),
    sampling_rate_hz=10.0, baseline_windows_s=[60.0, 20.0, 120.0],
    analysis_margin_s=60.0, skip_start_s=20.0, bin_seconds=30.0,
    pixel_threshold=0.01, threshold_factors=[0.5, 1.0, 2.0],
    recruitment_on=0.10, recruitment_off=0.05,
    min_amplitude=0.002, min_prominence=0.001,
    min_duration_s=0.2, max_rise_s=5.0, merge_gap_s=0.3,
    denominator_floor=1e-6, max_invalid_fraction=0.01,
    step_fraction=0.05, chunk_pixels=128,
    exclude_recordings=["260828_PV7/t2"], artifact_csv=None,
    recording_ids=[], recording_limit=None, overwrite=False,
    training_animal_ids=[], workers=2,
)
VERSION = "1"
# Exact pre-multiprocessing implementation; numerical processing is unchanged.
COMPATIBLE_IMPLEMENTATIONS = {"0131286bcf676ef8065bb8690b76209e0cb699c94019eab1c210d4326b22ae10"}
PRODUCTS = ["window_metrics.csv", "events.csv", "recording_metrics.csv", "recording_qc.csv"]
EVENT_COLUMNS = ["event_id", "onset_s", "offset_s", "peak_s", "duration_s",
                 "rise_s", "amplitude", "prominence", "positive_auc",
                 "peak_recruitment", "left_recruitment", "right_recruitment",
                 "bilateral", "long_event", "boundary_censored"]


def fingerprint(path):
    """Large source files use path/size/mtime; small geometry files also use SHA256."""
    path = Path(path)
    stat = path.stat()
    result = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size < 2_000_000:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def correct_signal(q, window_frames, invalid=None):
    """Full-recording partial-edge median; ANY gap invalidates its whole window."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not len(q) or window_frames < 1 or window_frames % 2 != 1:
        raise ValueError("Expected nonempty [time,pixel] and positive odd window")
    bad = ~np.isfinite(q)
    if invalid is not None:
        bad |= np.broadcast_to(np.asarray(invalid, bool), q.shape)
    clean = np.where(bad, np.nan, q)
    rolling = pd.DataFrame(clean).rolling(window_frames, center=True, min_periods=1)
    baseline = rolling.median().to_numpy(copy=True)
    contaminated = (pd.DataFrame(bad).rolling(window_frames, center=True, min_periods=1)
                    .max().to_numpy().astype(bool))
    baseline[contaminated] = np.nan
    counts = np.isfinite(baseline).sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("No valid baseline samples remain for at least one pixel")
    mean_b = np.nansum(baseline, axis=0) / counts
    residual = clean - baseline
    support = np.minimum(np.arange(len(q)) + window_frames // 2 + 1, len(q)) - np.maximum(
        np.arange(len(q)) - window_frames // 2, 0)
    return dict(q=clean, b=baseline, mean_b=mean_b, residual=residual,
                corrected=residual + mean_b, baseline_count=counts,
                window_support=support, valid=np.isfinite(residual))


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_events(trace, recruitment, valid, fs, config, baseline_s):
    """Hysteresis on recruitment, then merge only through valid time. [start,stop)."""
    valid = np.asarray(valid) & np.isfinite(trace) & np.isfinite(recruitment)
    candidates = []
    for a, b in intervals(valid & (recruitment >= config["recruitment_off"])):
        if np.any(recruitment[a:b] >= config["recruitment_on"]):
            if (candidates and (a - candidates[-1][1]) / fs <= config["merge_gap_s"]
                    and valid[candidates[-1][1]:a].all()):
                candidates[-1] = (candidates[-1][0], b)
            else:
                candidates.append((a, b))
    accepted = []
    for a, b in candidates:
        peak = a + int(np.argmax(trace[a:b]))
        flank = np.r_[trace[max(0, a-1):a], trace[b:min(len(trace), b+1)]]
        flank = flank[np.isfinite(flank)]
        prominence = trace[peak] - max(0.0, float(np.max(flank)) if len(flank) else 0.0)
        rise = (peak-a) / fs
        if ((b-a)/fs < config["min_duration_s"] or rise > config["max_rise_s"]
                or trace[peak] < config["min_amplitude"]
                or prominence < config["min_prominence"]):
            continue
        accepted.append(dict(start=a, stop=b, peak=peak, onset_s=a/fs, offset_s=b/fs,
                             peak_s=peak/fs, duration_s=(b-a)/fs, rise_s=rise,
                             amplitude=float(trace[peak]), prominence=float(prominence),
                             positive_auc=float(np.maximum(trace[a:b], 0).sum()/fs),
                             peak_recruitment=float(recruitment[peak]),
                             long_event=(b-a)/fs >= baseline_s/2,
                             boundary_censored=bool(a == 0 or b == len(trace)
                                 or not valid[a-1] or (b < len(trace) and not valid[b]))))
    return accepted


def metric_row(trace, positive, recruitment, valid, events, start, stop, fs):
    selected = valid[start:stop]
    x = trace[start:stop][selected]
    exposure = len(x)/fs
    occupied = np.zeros(stop-start, bool)
    count = 0
    for event in events:
        a, b = max(start, event["start"]), min(stop, event["stop"])
        if a < b:
            occupied[a-start:b-start] = True
        count += start <= event["start"] < stop
    seconds = np.count_nonzero(occupied & selected)/fs
    return dict(start_s=start/fs, stop_s=stop/fs, midpoint_s=(start+stop)/(2*fs),
                valid_seconds=exposure, event_count=count, occupied_seconds=seconds,
                burden=seconds/exposure if exposure else np.nan,
                events_per_minute=60*count/exposure if exposure else np.nan,
                residual_rms=float(np.sqrt(np.mean(x*x))) if len(x) else np.nan,
                residual_mad=float(1.4826*np.median(np.abs(x-np.median(x)))) if len(x) else np.nan,
                mean_positive_pixel_activity=float(np.mean(positive[start:stop][selected])) if len(x) else np.nan,
                mean_recruitment=float(np.mean(recruitment[start:stop][selected])) if len(x) else np.nan)


def validate_config(c):
    for key in ("sampling_rate_hz", "bin_seconds", "pixel_threshold", "denominator_floor",
                "chunk_pixels", "min_duration_s", "max_rise_s"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 < c["recruitment_off"] <= c["recruitment_on"] <= 1:
        raise ValueError("Require 0 < recruitment_off <= recruitment_on <= 1")
    if not 0 <= c["max_invalid_fraction"] < 1:
        raise ValueError("max_invalid_fraction must be in [0,1)")
    if c["recording_limit"] is not None and c["recording_limit"] < 1:
        raise ValueError("recording_limit must be positive")
    if len(set(c["baseline_windows_s"])) != len(c["baseline_windows_s"]) or len(set(c["threshold_factors"])) != len(c["threshold_factors"]):
        raise ValueError("Duplicate windows or threshold factors")
    if not c["baseline_windows_s"] or not c["threshold_factors"]:
        raise ValueError("At least one baseline window and threshold factor required")
    for value in c["baseline_windows_s"] + c["threshold_factors"]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Windows and threshold factors must be finite and positive")
    for key in ("analysis_margin_s", "skip_start_s", "merge_gap_s", "min_amplitude", "min_prominence", "step_fraction"):
        if not np.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")


def load_recording(row, c):
    folder = Path(c["pixel_root"]) / row["pixel_dir"]
    mp = folder / "pixels_meta_full.npz"
    with np.load(mp, allow_pickle=False) as data:
        meta = {k: data[k].tolist() for k in data.files if k not in ("mean_f", "mean_r")}
    if meta["axis_order"] != "time,y,x" or meta["n_written"] != meta["n_time"]:
        raise ValueError(f"Invalid or incomplete dump: {folder}")
    atlas_path, atlas = recording_roi_set(meta)
    boxes = resolve_boxes(atlas, all_rois=True, custom_boxes=False)
    region = meta["region"]
    shape = tuple(meta["shape"])
    fpath, epath = [folder / f"pixels_f_{channel}_full.npy" for channel in ("gcamp", "emo")]
    f, e = [np.load(p, mmap_mode="r") for p in (fpath, epath)]
    if shape[0] != meta["n_time"] or f.shape != shape or e.shape != shape or f.dtype != np.float32 or e.dtype != np.float32:
        raise ValueError(f"Unexpected pixel shape/dtype: {folder}")
    masks = hemisphere_masks(boxes, meta["y_1"], meta["x_2"], region, shape[1:])
    masks = {"cortex": masks["L"] | masks["R"], "left": masks["L"], "right": masks["R"]}
    for name, box in boxes.items():
        m = np.zeros(shape[1:], bool)
        m[crop_slices(box, meta["y_1"], meta["x_2"], region)] = True
        masks[name] = m
    ids = np.flatnonzero(masks["cortex"])
    # Read selected pixels only; per-pixel median working memory is chunk bounded.
    f = np.asarray(f.reshape(shape[0], -1)[:, ids], dtype=np.float64)
    e = np.asarray(e.reshape(shape[0], -1)[:, ids], dtype=np.float64)
    bad = ~np.isfinite(f) | ~np.isfinite(e) | (e <= c["denominator_floor"]) | (f <= 0)
    keep = bad.mean(axis=0) <= c["max_invalid_fraction"]
    if not keep.any():
        raise ValueError("No quality pixels remain")
    denominator_invalid = int((~np.isfinite(e) | (e <= c["denominator_floor"])).sum())
    nonfinite_gcamp = int((~np.isfinite(f)).sum())
    ids, f, e, bad = ids[keep], f[:, keep], e[:, keep], bad[:, keep]
    membership = {name: mask.ravel()[ids] for name, mask in masks.items()}
    if any(not member.any() for member in membership.values()):
        raise ValueError("A region has no quality pixels")
    frame_bad = np.zeros(shape[0], bool)
    if c["artifact_csv"]:
        annotations = pd.read_csv(c["artifact_csv"])
        for a in annotations.loc[annotations.recording_id == row["recording_id"]].to_dict("records"):
            frame_bad |= (np.arange(shape[0])/c["sampling_rate_hz"] >= a["start_s"]) & (
                np.arange(shape[0])/c["sampling_rate_hz"] < a["stop_s"])
    # Conservative frame-step flag in either raw channel. Also save indices for review.
    raw_means = np.column_stack([np.nanmedian(np.where(bad, np.nan, x), axis=1) for x in (f, e)])
    relative_step = np.abs(np.diff(raw_means, axis=0)) / np.maximum(np.abs(raw_means[:-1]), c["denominator_floor"])
    step = np.r_[False, np.any(relative_step > c["step_fraction"], axis=1)] if c["step_fraction"] else np.zeros(shape[0], bool)
    frame_bad |= step
    bad |= frame_bad[:, None]
    q = np.full_like(f, np.nan)
    np.divide(f, e, out=q, where=~bad)
    sources = [fingerprint(p) for p in (mp, atlas_path, fpath, epath)]
    timing_keys = {k: meta[k] for k in ("sampling_rate_hz", "channel_order", "channel_offset_s") if k in meta}
    if "sampling_rate_hz" in timing_keys and timing_keys["sampling_rate_hz"] != c["sampling_rate_hz"]:
        raise ValueError("Configured per-channel rate differs from metadata")
    qc = dict(invalid_denominator_samples=denominator_invalid, nonfinite_gcamp_samples=nonfinite_gcamp, mask_pixels=len(ids), original_union_pixels=int(masks["cortex"].sum()),
              removed_pixels=int((~keep).sum()), invalid_samples=int(bad.sum()),
              flagged_frames=int(frame_bad.sum()), step_frames=json.dumps(np.flatnonzero(step).tolist()),
              timing_metadata=json.dumps(timing_keys),
              timing_assumption="Configured per-channel rate; index-aligned channels; no timestamp interpolation",
              original_sensor_clipping="not assessable from downsampled dumps")
    return q, f, e, bad, membership, ids, shape, sources, qc


def recording_signature(row, c):
    """Read small metadata and file stats only; never map raw volumes for cache checks."""
    folder = Path(c["pixel_root"]) / row["pixel_dir"]
    mp = folder / "pixels_meta_full.npz"
    with np.load(mp, allow_pickle=False) as data:
        meta = {k: data[k].tolist() for k in data.files if k not in ("mean_f", "mean_r")}
    atlas_path, _ = recording_roi_set(meta)
    sources = [fingerprint(path) for path in (mp, atlas_path,
        folder/"pixels_f_gcamp_full.npy", folder/"pixels_f_emo_full.npy")]
    signature = dict(version=VERSION, config={k: v for k, v in c.items() if k not in (
        "overwrite", "recording_limit", "recording_ids", "output_root", "workers")}, sources=sources,
        design=row, implementation=fingerprint(__file__),
        helpers=[fingerprint(ROOT/name) for name in ("epileptic_by_area_animal_day_pixels.py","epileptic_by_active_pixels.py")],
        versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
    if c["artifact_csv"]:
        signature["artifacts"] = fingerprint(c["artifact_csv"])
    return signature


def cache_matches(row, c, signature=None):
    if c["overwrite"]:
        return False
    out = Path(c["output_root"]) / row["pixel_dir"]
    done = out/"complete.json"
    required = PRODUCTS + [f"{stem}_{seconds:g}s.{ext}" for seconds in c["baseline_windows_s"]
        for stem,ext in (("traces","npz"),("channel_diagnostics","csv"),("diagnostics","png"),("survival","csv"),("spatial","npz"),("spatial","png"))]
    if not done.exists() or not all((out/name).exists() for name in required):
        return False
    signature = signature or recording_signature(row,c)
    try:
        saved = json.loads(done.read_text())
        previous = saved["provenance"]
        # A marker must be internally consistent; do not accept edited provenance.
        if saved["key"] != hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest():
            return False
        accepted = COMPATIBLE_IMPLEMENTATIONS | {signature["implementation"]["sha256"]}
        if previous["implementation"].get("sha256") not in accepted:
            return False
        return {k:v for k,v in previous.items() if k != "implementation"} == {
            k:v for k,v in signature.items() if k != "implementation"}
    except (KeyError, ValueError, TypeError):
        return False


def process_recording(row, c):
    out = Path(c["output_root"]) / row["pixel_dir"]
    signature = recording_signature(row,c)
    key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    done = out/"complete.json"
    products = PRODUCTS
    if cache_matches(row,c,signature):
        return {name:pd.read_csv(out/name) for name in products}
    q, f, e, bad, members, ids, shape, sources, qc = load_recording(row,c)
    out.mkdir(parents=True, exist_ok=True)
    if done.exists():
        done.unlink()  # remove completion marker only; a failed run cannot appear complete
    fs, n = c["sampling_rate_hz"], len(q)
    t = np.arange(n)/fs
    windows, recordings, event_rows, qc_rows = [], [], [], []
    longest_window = median_window_frames(max(c["baseline_windows_s"]), fs)
    longest_bad = pd.DataFrame(bad).rolling(longest_window,center=True,min_periods=1).max().to_numpy().astype(bool)
    common_valid = np.column_stack([~longest_bad[:,member].any(axis=1) for member in members.values()])
    for seconds in c["baseline_windows_s"]:
        w = median_window_frames(seconds, fs)
        names = list(members)
        fields = {k: np.zeros((n, len(names))) for k in ("q", "b", "residual", "corrected", "positive")}
        valid = np.ones((n, len(names)), bool)
        mean_b = np.zeros(len(names))
        active = {factor: np.zeros((n, len(names))) for factor in c["threshold_factors"]}
        noise_null = np.zeros(len(names))
        pixel_noise, pixel_means, baseline_counts, pixel_rms, pixel_active = [], [], [], [], []
        common_margin = max(c["analysis_margin_s"], max(c["baseline_windows_s"])/2)
        common = (t >= max(common_margin, c["skip_start_s"])) & (t < n/fs-common_margin)
        for start in range(0, q.shape[1], c["chunk_pixels"]):
            stop = start+c["chunk_pixels"]
            result = correct_signal(q[:, start:stop], w, bad[:, start:stop])
            res = result["residual"]
            # First-difference MAD is a diagnostic high-frequency noise proxy, not a detector scale.
            diff = np.diff(res, axis=0)
            sigma = 1.4826*np.nanmedian(np.abs(diff-np.nanmedian(diff, axis=0)), axis=0)/np.sqrt(2)
            pixel_noise.extend(sigma)
            common_res = np.where(longest_bad[:,start:stop],np.nan,res)[common]
            pixel_rms.extend(np.sqrt(np.nanmean(common_res**2,axis=0)))
            exposure = np.isfinite(common_res).sum(axis=0)
            pixel_active.extend(np.divide(np.sum(common_res >= c["pixel_threshold"],axis=0), exposure,
                out=np.full(res.shape[1],np.nan), where=exposure>0))
            pixel_means.extend(result["mean_b"])
            baseline_counts.extend(result["baseline_count"])
            for j, name in enumerate(names):
                take = members[name][start:stop]
                if not take.any():
                    continue
                for field in ("q", "b", "residual", "corrected"):
                    fields[field][:, j] += np.nan_to_num(result[field][:, take]).sum(axis=1)
                fields["positive"][:, j] += np.nansum(np.maximum(res[:, take], 0), axis=1)
                valid[:, j] &= result["valid"][:, take].all(axis=1)
                mean_b[j] += result["mean_b"][take].sum()
                noise_null[j] += np.nansum(sigma[take])/np.sqrt(2*np.pi)
                for factor in active:
                    active[factor][:, j] += np.sum(res[:, take] >= c["pixel_threshold"]*factor, axis=1)
        area = np.array([m.sum() for m in members.values()])
        for field in fields.values():
            field /= area
            field[~valid] = np.nan
        # Preserve observed q even when a nearby gap invalidates the estimated baseline.
        for j,name in enumerate(names):
            fields["q"][:,j] = np.mean(q[:,members[name]],axis=1)
        mean_b /= area
        noise_null /= area
        for fraction in active.values():
            fraction /= area
            fraction[~valid] = np.nan
        support = result["window_support"]
        full = support == w
        common_margin = max(c["analysis_margin_s"], max(c["baseline_windows_s"])/2)
        common = (t >= max(common_margin, c["skip_start_s"])) & (t < n/fs-common_margin)
        own = (t >= c["skip_start_s"]) & full
        np.savez_compressed(out/f"traces_{seconds:g}s.npz", time_s=t, region_names=names,
            **fields, mean_b=mean_b, valid=valid, window_support=support,
            common_support=common, common_valid=common_valid, method_support=own, pixel_indices=ids, crop_shape=shape[1:],
            pixel_mean_b=pixel_means, pixel_baseline_count=baseline_counts, pixel_noise_proxy=pixel_noise,
            region_area=area, gaussian_positive_noise_null=noise_null,
            recruitment=np.stack(list(active.values())), threshold_factors=list(active),
            equal_roi_residual=fields["residual"][:,3:].mean(axis=1),
            equal_roi_corrected=fields["corrected"][:,3:].mean(axis=1),
            equal_roi_mean_b=mean_b[3:].mean(),
            processing="per-pixel median then fixed-mask spatial mean")
        plot_events = []
        for support_name, support_mask in (("common", common), ("method", own)):
            for factor, fractions in active.items():
                region_events = {}
                for j, name in enumerate(names):
                    good = valid[:, j] & support_mask
                    if support_name == "common":
                        good &= common_valid[:,j]
                    region_events[name] = detect_events(fields["residual"][:, j], fractions[:, j], good, fs, c, seconds)
                # Cortex events are the UNION of regional intervals: a local event is counted once.
                union = np.zeros(n, bool)
                for events in region_events.values():
                    for ev in events:
                        union[ev["start"]:ev["stop"]] = True
                cortex_valid = valid[:, 0] & support_mask
                if support_name == "common":
                    cortex_valid &= common_valid[:,0]
                union &= cortex_valid
                merged = []
                for a, b in intervals(union):
                    if merged and (a-merged[-1][1])/fs <= c["merge_gap_s"] and cortex_valid[merged[-1][1]:a].all():
                        merged[-1] = (merged[-1][0], b)
                    else:
                        merged.append((a,b))
                cortex_events = []
                for a,b in merged:
                    peak = a+int(np.argmax(fields["residual"][a:b, 0]))
                    cortex_events.append(dict(start=a, stop=b, peak=peak, onset_s=a/fs, offset_s=b/fs,
                        peak_s=peak/fs, duration_s=(b-a)/fs, rise_s=(peak-a)/fs,
                        amplitude=float(fields["residual"][peak, 0]), prominence=np.nan,
                        positive_auc=float(np.maximum(fields["residual"][a:b, 0],0).sum()/fs),
                        peak_recruitment=float(fractions[peak,0]), long_event=(b-a)/fs >= seconds/2,
                        boundary_censored=bool(a==0 or b==n or not cortex_valid[a-1] or (b<n and not cortex_valid[b]))))
                region_events["cortex"] = cortex_events
                if support_name == "common" and factor == 1.0:
                    plot_events = cortex_events
                for j, name in enumerate(names):
                    good = valid[:, j] & support_mask
                    if support_name == "common":
                        good &= common_valid[:,j]
                    common_keys = dict(row, method=f"additive_{seconds:g}s", baseline_s=seconds,
                                       support=support_name, threshold_factor=factor, region=name, mask_pixels=int(area[j]))
                    events = region_events[name]
                    for index, ev in enumerate(events):
                        peak = ev["peak"]
                        event_rows.append(dict(common_keys, **{k:v for k,v in ev.items() if k not in ("start","stop","peak")},
                            event_id=f"{row['recording_id']}:{seconds:g}:{support_name}:{factor:g}:{name}:{index}",
                            left_recruitment=fractions[peak,1], right_recruitment=fractions[peak,2],
                            bilateral=bool(min(fractions[peak,1:3]) >= c["recruitment_on"])))
                    args = (fields["residual"][:,j], fields["positive"][:,j], fractions[:,j], good, events)
                    for a in range(0,n,max(1,round(c["bin_seconds"]*fs))):
                        windows.append(dict(common_keys, **metric_row(*args, a, min(n,a+round(c["bin_seconds"]*fs)), fs)))
                    rec = dict(common_keys, **metric_row(*args,0,n,fs), mean_baseline=mean_b[j],
                        gaussian_positive_noise_null=noise_null[j],
                        mean_event_amplitude=float(np.mean([ev["amplitude"] for ev in events])) if events else np.nan,
                        mean_peak_recruitment=float(np.mean([ev["peak_recruitment"] for ev in events])) if events else np.nan)
                    recordings.append(rec)
        # Independent channel diagnostics: same additive operation, no optical rescaling.
        diagnostic = {"time_s":t}
        for channel, data in (("gcamp", f), ("emo", e), ("ratio", q)):
            raw = np.where(bad, np.nan, data)
            # labelled trace-level diagnostic, never used for the per-pixel endpoints
            trace = np.mean(raw, axis=1)[:,None]
            result_d = correct_signal(trace, w)
            diagnostic[channel+"_q"] = trace[:,0]
            diagnostic[channel+"_b"] = result_d["b"][:,0]
            diagnostic[channel+"_residual"] = result_d["residual"][:,0]
            finite = np.isfinite(result_d["b"][:,0]) & common
            slope = np.polyfit(t[finite]/60, result_d["b"][finite,0],1)[0] if finite.sum()>1 else np.nan
            for part, selection in zip(("early","middle","late"), np.array_split(np.flatnonzero(common),3)):
                x = result_d["residual"][selection,0]
                x = x[np.isfinite(x)]
                qc_rows.append(dict(row, **qc, baseline_s=seconds, channel=channel, period=part,
                    diagnostic_processing="median of spatial mean (diagnostic only)", baseline_slope_per_min=slope,
                    residual_rms=float(np.sqrt(np.mean(x*x))) if len(x) else np.nan,
                    residual_p95=float(np.quantile(x,.95)) if len(x) else np.nan,
                    noise_difference_mad=float(1.4826*np.nanmedian(np.abs(np.diff(result_d["residual"][selection,0])-np.nanmedian(np.diff(result_d["residual"][selection,0]))))/np.sqrt(2)) if len(selection)>1 else np.nan,
                    baseline_relative_change=float((result_d["b"][finite,0][-1]-result_d["b"][finite,0][0])/np.mean(result_d["b"][finite,0])) if finite.any() else np.nan,
                    valid_seconds=float((valid[:,0] & common_valid[:,0])[common].sum()/fs),
                    excluded_seconds=float(n/fs-(valid[:,0] & common_valid[:,0])[common].sum()/fs), status="included"))
        pd.DataFrame(diagnostic).to_csv(out/f"channel_diagnostics_{seconds:g}s.csv",index=False)
        fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
        axes[0].plot(t,fields["q"][:,0],lw=.5,label="q")
        axes[0].plot(t,fields["b"][:,0],label="pixel baseline mean")
        axes[1].plot(t,fields["corrected"][:,0],lw=.5,label="corrected")
        axes[1].axhline(mean_b[0],color="k",ls="--",label="mean baseline")
        plot_factor = 1.0 if 1.0 in active else c["threshold_factors"][0]
        axes[2].plot(t,active[plot_factor][:,0],lw=.6,label=f"recruitment, factor {plot_factor}")
        for ev in plot_events:
            for ax in axes:
                ax.axvspan(ev["onset_s"],ev["offset_s"],color="red",alpha=.15)
        for ax in axes:
            ax.legend(loc="upper right")
        axes[0].set_ylabel("F/emo ratio")
        axes[1].set_ylabel("F/emo ratio")
        axes[2].set_ylabel("active fraction")
        axes[2].set_xlabel("seconds in recording")
        fig.suptitle(f"{row['recording_id']} | sampled cortical ROI union | {seconds:g}s additive median")
        fig.tight_layout()
        fig.savefig(out/f"diagnostics_{seconds:g}s.png",dpi=130)
        plt.close(fig)
        # Empirical survival curves retain all valid native residual amplitudes.
        survival = []
        for j,name in enumerate(names):
            values = np.sort(fields["residual"][:,j][valid[:,j] & common_valid[:,j] & common])
            survival.extend(dict(region=name, amplitude=x, survival=(len(values)-i)/len(values)) for i,x in enumerate(values))
        pd.DataFrame(survival,columns=["region","amplitude","survival"]).to_csv(out/f"survival_{seconds:g}s.csv",index=False)
        np.savez_compressed(out/f"spatial_{seconds:g}s.npz", pixel_indices=ids, crop_shape=shape[1:],
                            mean_baseline=pixel_means, noise_proxy=pixel_noise, residual_rms=pixel_rms,
                            active_fraction=pixel_active)
        fig, axes = plt.subplots(1,2,figsize=(9,4))
        for ax,values,title in zip(axes,(pixel_rms,pixel_active),("Residual RMS (F/emo ratio)","Active-time fraction")):
            spatial = np.full(np.prod(shape[1:]),np.nan)
            spatial[ids] = values
            im = ax.imshow(spatial.reshape(shape[1:]),interpolation="nearest")
            ax.set_title(title); fig.colorbar(im,ax=ax)
        fig.suptitle("Sampled cortical ROI union; common valid support")
        fig.tight_layout(); fig.savefig(out/f"spatial_{seconds:g}s.png",dpi=130); plt.close(fig)
    tables = dict(zip(products, [pd.DataFrame(windows), pd.DataFrame(event_rows,columns=list(row)+[
        "method","baseline_s","support","threshold_factor","region","mask_pixels"]+EVENT_COLUMNS),
        pd.DataFrame(recordings), pd.DataFrame(qc_rows)]))
    for name, table in tables.items():
        table.to_csv(out/name,index=False)
    done.write_text(json.dumps(dict(key=key, provenance=signature),indent=2),encoding="utf-8")
    return tables


def process_job(row, config):
    """Worker writes only its recording directory; large tables never cross processes."""
    try:
        process_recording(row,config)
        return dict(recording_id=row["recording_id"], status="computed", error="")
    except Exception as exc:
        import traceback
        return dict(recording_id=row["recording_id"], status="failed",
                    error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())


def run(config=None):
    c = dict(RUN_CONFIG, **(config or {}))
    validate_config(c)
    if not isinstance(c["workers"], int) or c["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(c["output_root"])
    root.mkdir(parents=True,exist_ok=True)
    run_done = root/"run_complete.json"
    if run_done.exists():
        run_done.unlink()
    design_path = Path(c["pixel_root"])/"experimental_design.csv"
    design = pd.read_csv(design_path,dtype={"day":str})
    if design.recording_id.duplicated().any() or design.pixel_dir.duplicated().any():
        raise ValueError("Duplicate recording IDs or paths in design")
    if c["recording_ids"]:
        missing = set(c["recording_ids"])-set(design.recording_id)
        if missing:
            raise ValueError(f"Unknown recording IDs: {missing}")
        design = design[design.recording_id.isin(c["recording_ids"])]
    if c["artifact_csv"]:
        artifacts = pd.read_csv(c["artifact_csv"])
        required = {"recording_id","start_s","stop_s"}
        if not required.issubset(artifacts.columns):
            raise ValueError("Artifact CSV requires recording_id,start_s,stop_s")
        if (not np.isfinite(artifacts[["start_s","stop_s"]]).all().all()
                or (artifacts.start_s < 0).any() or (artifacts.stop_s <= artifacts.start_s).any()):
            raise ValueError("Artifact intervals must be finite, nonnegative and nonempty")
    excluded = design[design.pixel_dir.isin(c["exclude_recordings"])]
    design = design[~design.pixel_dir.isin(c["exclude_recordings"])].head(c["recording_limit"])
    if design.empty:
        raise ValueError("No recordings selected")
    manifest = dict(config=c, implementation_version=VERSION, formula="q=F/emo; b=running_median(q); residual=q-b; corrected=residual+mean_valid(b)",
        units="F_gcamp/F_emo ratio; AUC ratio-seconds", mask="sampled cortical ROI union, fixed quality mask",
        calibration="exploratory common native thresholds; not validated epileptiform labels",
        edge_policy="partial recording edges included in mean_b; all gap-intersecting windows invalid",
        design=fingerprint(design_path), python=platform.python_version(), numpy=np.__version__,
        pandas=pd.__version__, scipy=scipy.__version__, recording_ids=design.recording_id.tolist())
    (root/"config.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    rows = design.to_dict("records")
    pending, results = [], []
    for row in rows:
        if cache_matches(row,c):
            results.append(dict(recording_id=row["recording_id"],status="cached",error=""))
        else:
            pending.append(row)
    print(f"Reusing {len(results)}/{len(rows)} completed recordings; processing {len(pending)} with {c['workers']} workers",flush=True)

    def record_result(result):
        results.append(result)
        pd.DataFrame(results).to_csv(root/"processing_status.csv",index=False)
        print(f"[{len(results)}/{len(rows)}] {result['recording_id']} {result['status']} {result['error']}",flush=True)

    pd.DataFrame(results,columns=["recording_id","status","error"]).to_csv(root/"processing_status.csv",index=False)
    if c["workers"] == 1:
        for row in pending:
            record_result(process_job(row,c))
    elif pending:
        with ProcessPoolExecutor(max_workers=c["workers"],mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(process_job,row,c):row for row in pending}
            for future in as_completed(futures):
                record_result(future.result())
    failures = [result for result in results if result["status"] == "failed"]
    (root/"processing_failures.json").write_text(json.dumps(failures,indent=2),encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} recording(s) failed QC/processing; completed caches preserved. See processing_failures.json. Cohort is incomplete.")
    # Only the parent writes aggregate tables, in original design order.
    written = set()
    for row in rows:
        for name in PRODUCTS:
            table = pd.read_csv(root/row["pixel_dir"]/name)
            if name == "recording_qc.csv":
                table["reason"] = ""
            table.to_csv(root/(name+".partial"),mode="a" if name in written else "w",
                         header=name not in written,index=False)
            written.add(name)
    if len(excluded):
        qc_path = root/"recording_qc.csv.partial"
        columns = pd.read_csv(qc_path,nrows=0).columns
        excluded.assign(status="excluded",reason="prespecified acquisition artifact").reindex(
            columns=columns).to_csv(qc_path,mode="a",header=False,index=False)
    for name in written:
        (root/(name+".partial")).replace(root/name)
    run_done.write_text(json.dumps(dict(recording_ids=design.recording_id.tolist(),
        tables={name:fingerprint(root/name) for name in sorted(written)}),indent=2),encoding="utf-8")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,help="JSON overrides of RUN_CONFIG")
    parser.add_argument("--pixel-root")
    parser.add_argument("--output-root")
    parser.add_argument("--workers",type=int)
    parser.add_argument("--recording-limit",type=int)
    parser.add_argument("--recording-ids",nargs="+")
    parser.add_argument("--baseline-windows-s",nargs="+",type=float)
    parser.add_argument("--pixel-threshold",type=float)
    parser.add_argument("--artifact-csv")
    parser.add_argument("--overwrite",action="store_true",default=None)
    args = vars(parser.parse_args())
    config_path = args.pop("config")
    config = json.loads(config_path.read_text()) if config_path else {}
    unknown = set(config)-set(RUN_CONFIG)
    if unknown:
        parser.error(f"Unknown config keys: {unknown}")
    config.update({k:v for k,v in args.items() if v is not None})
    run(config)


if __name__ == "__main__":
    main()
