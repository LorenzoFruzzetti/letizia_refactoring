"""Analyze completed median-dF/F caches and report against the full study design.

Uses the current epileptic_by_active_pixels.RUN_CONFIG and shared cohort detector.
No caches are built. The cutoff is pooled only over the included cached subset.
Run: conda run --no-capture-output -n letizia python analyze_cached_active_pixels.py
"""

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cache_median_dff import cache_paths
from epileptic_by_active_pixels import RUN_CONFIG, ActivePixelTraceLoader, SIGNAL_UNIT
from epileptic_by_area_animal_day import run_analysis

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/active_pixels_cached_30s_design"
DESIGN = ROOT / "pixel_data/experimental_design.csv"
COUNTS = ["n_peaks", "n_confirmed", "n_epileptic", "n_epileptic_L", "n_epileptic_R"]


def aggregate(results, design, keys):
    """Keep unobserved design cells missing and pool exposure only over observed files."""
    expected = design.groupby(keys, as_index=False, sort=False).agg(
        n_recordings_expected=("recording_id", "size"), n_animals_expected=("animal", "nunique"))
    measured = results.groupby(keys, as_index=False, sort=False).agg(
        n_recordings_analyzed=("recording_id", "size"), n_animals_analyzed=("animal", "nunique"),
        recorded_min=("recorded_min", "sum"),
        **{column: (column, "sum") for column in COUNTS})
    table = expected.merge(measured, on=keys, how="left", validate="one_to_one")
    for column in ["n_recordings_analyzed", "n_animals_analyzed"]:
        table[column] = table[column].fillna(0).astype(int)
    table["coverage_fraction"] = table.n_recordings_analyzed / table.n_recordings_expected
    table["hemisphere_min"] = 2 * table.recorded_min
    table["epileptic_per_hemisphere_min"] = table.n_epileptic / table.hemisphere_min
    return table


def main():
    config = RUN_CONFIG
    assert config["dff_source"] == "median_dff", "This runner selects completed median caches"
    assert config["baseline_window_s"] == 30, "Choose a new output folder for another baseline"
    design = pd.read_csv(DESIGN, dtype={"day": str})
    assert not design.recording_id.duplicated().any()
    assert not design.pixel_dir.duplicated().any()
    assert not design[["day", "animal", "group"]].drop_duplicates().duplicated(["day", "animal"]).any()
    excluded = set(config["exclude_recordings"])
    paths, durations, statuses = [], {}, []
    for row in design.itertuples(index=False):
        folder = (Path(config["pixel_root"]) / row.pixel_dir).resolve()
        assert folder.is_relative_to(Path(config["pixel_root"]).resolve())
        volume, marker = cache_paths(folder, config["baseline_window_s"])
        if row.pixel_dir in excluded:
            statuses.append("excluded_known_artifact")
        elif not (volume.is_file() and marker.is_file()):
            statuses.append("no_completed_30s_cache")
        else:
            cache = json.loads(marker.read_text(encoding="utf-8"))
            assert cache["window_s"] == config["baseline_window_s"]
            assert cache["sampling_rate_hz"] == config["detection"]["sampling_rate_hz"]
            with np.load(folder / "pixels_meta_full.npz", allow_pickle=False) as meta:
                n_frames = int(meta["n_time"])
                assert int(meta["n_written"]) == n_frames
            durations[row.recording_id] = n_frames / config["detection"]["sampling_rate_hz"] / 60
            paths.append(folder / "pixels_meta_full.npz")
            statuses.append("analyzed")
    # A cached dump absent from the design must not silently disappear.
    known = set(design.pixel_dir)
    for marker in Path(config["pixel_root"]).glob("*/*/pixels_median_dff_30s_meta.json"):
        assert f"{marker.parent.parent.name}/{marker.parent.name}" in known, str(marker)
    assert paths, "No completed caches"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    included = design.loc[np.array(statuses) == "analyzed"].copy()
    included.to_csv(OUTPUT / "included_recordings.csv", index=False)
    groups_dir = OUTPUT / "design_groups"
    groups_dir.mkdir(exist_ok=True)
    design[["day", "animal", "group"]].drop_duplicates().to_csv(groups_dir / "batch_summary.csv", index=False)
    loader = ActivePixelTraceLoader(OUTPUT, **{key: config[key] for key in (
        "pixel_z_threshold", "dff_source", "baseline_window_s", "pixel_scale", "pixel_scale_percentile")})
    print(f"Analyzing {len(paths)} completed caches; design contains {len(design)} recordings", flush=True)
    run_analysis(
        input_root=groups_dir, output_dir=OUTPUT, recording_paths=sorted(paths), recording_limit=None,
        trace_loader=loader, diagnostics_root=OUTPUT, diagnostic_plot_count=0,
        diagnostic_animals_per_group=None, roi_metadata=loader.categories,
        signal_unit=SIGNAL_UNIT, workers=config["workers"],
        provenance=dict(experimental_design=str(DESIGN), selection="completed 30-second caches only",
                        calibration_scope="included cached recordings only",
                        included_recordings=included.pixel_dir.tolist(), excluded_recordings=sorted(excluded),
                        **{key: config[key] for key in ("dff_source", "baseline_window_s",
                           "pixel_z_threshold", "pixel_scale", "pixel_scale_percentile")}),
        **config["detection"],
    )
    events = pd.read_csv(OUTPUT / "all_peaks.csv", dtype={"date": str})
    events["recording_id"] = events.date + "_" + events.animal + "_" + events.recording
    counts = events.groupby("recording_id").agg(
        n_peaks=("frame", "size"), n_confirmed=("confirmed", "sum"), n_epileptic=("epileptic", "sum"))
    results = included.merge(counts, on="recording_id", how="left", validate="one_to_one")
    for side in ("L", "R"):
        side_counts = events.loc[(events.roi == f"Cortex{side}_active") & events.epileptic].groupby("recording_id").size()
        results[f"n_epileptic_{side}"] = results.recording_id.map(side_counts)
    results[COUNTS] = results[COUNTS].fillna(0).astype(int)
    results["recorded_min"] = results.recording_id.map(durations)
    results["hemisphere_min"] = 2 * results.recorded_min
    results["epileptic_per_hemisphere_min"] = results.n_epileptic / results.hemisphere_min
    assert len(results) == len(paths)
    assert results.n_peaks.sum() == len(events)
    assert results.n_epileptic.sum() == events.epileptic.sum()
    assert (results.n_epileptic == results.n_epileptic_L + results.n_epileptic_R).all()
    coverage = design.copy()
    coverage["analysis_status"] = statuses
    coverage = coverage.merge(results[["recording_id", *COUNTS, "recorded_min", "hemisphere_min",
                                      "epileptic_per_hemisphere_min"]], on="recording_id", how="left", validate="one_to_one")
    # Join the canonical day index rather than re-numbering the available subset.
    event_details = events.drop(columns=["date", "animal", "group", "recording", "recording_day"])
    event_details = event_details.merge(included, on="recording_id", how="left", validate="many_to_one")
    tables = {
        "recording_results": results,
        "animal_day_results": aggregate(results, design, ["group", "mouse_line", "animal", "day", "day_index", "animal_day"]),
        "animal_results": aggregate(results, design, ["group", "mouse_line", "animal"]),
        "group_line_day_results": aggregate(results, design, ["group", "mouse_line", "day_index", "animal_day"]),
        "group_line_results": aggregate(results, design, ["group", "mouse_line"]),
        "design_coverage": coverage,
        "events_with_design": event_details,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)
    notes = (
        "Results follow pixel_data/experimental_design.csv, preserving its day_index and animal_day.\n"
        "Only completed 30-second caches were analyzed; no new cache was built.\n"
        "The top-1% cutoff is pooled over this cached subset, not the full cohort.\n"
        "Unanalyzed recordings and design cells have missing event counts, not zeros.\n"
        "n_epileptic is the sum of left and right hemisphere detections; bilateral events are not deduplicated.\n"
        "Rates divide that sum by hemisphere_min = 2 * recorded_min.\n"
        "Group/line/day summaries are descriptive pooled exposure rates, not independent-animal inference.\n"
        "Use the *_results tables for canonical design days; the detector's original tables number only observed days.\n"
    )
    (OUTPUT / "RESULTS_README.txt").write_text(notes, encoding="utf-8")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("Excel engine unavailable; all tables saved as CSV.", flush=True)
    else:
        with pd.ExcelWriter(OUTPUT / "cached_active_pixel_results.xlsx", engine="openpyxl") as writer:
            for name, table in tables.items():
                table.to_excel(writer, sheet_name=name, index=False)
                sheet = writer.sheets[name]
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
            pd.DataFrame({"notes": notes.splitlines()}).to_excel(writer, sheet_name="notes", index=False)
    print(tables["group_line_results"].to_string(index=False), flush=True)
    print(f"Verified {len(results)} recordings, {len(events)} peaks, {int(events.epileptic.sum())} epileptiform detections.", flush=True)
    print(f"Tables saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
