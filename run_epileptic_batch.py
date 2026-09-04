"""Run the epileptiform detection over every recording of an outputs tree.

Walks `<input_root>/<day>_<animal>/<t#>/roi_fluorescence_full.csv`, calls
`test_epileptic_detection.main` on each one - so every recording still gets its
own `epileptic_detection/` folder with `roi_events.csv`,
`roi_epileptic_events.csv` and the trace figure - and then tallies the flagged
events into two tables:

  epileptic_counts_by_recording.csv   one row per t#, so t1..t5 stay separate,
  epileptic_counts_by_animal.csv      one row per day+animal, t1..t5 summed.

Both carry one column per cortical ROI plus a `total`, with `n_peaks` and
`n_confirmed` alongside so a count can be read against how many peaks the
detector saw in the first place.

Run straight from the editor: edit RUN_CONFIG below, no CLI flags needed.
"""

import os

# MUST run before numpy/scipy/matplotlib are imported anywhere in the process.
# Same MKL delay-load fault as test_epileptic_detection documents; setting it
# here as well means this file works whether it is the entrypoint or imported.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import by path so the script runs from the editor regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_epileptic_detection as detection

# Edit this section to run the batch without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    # Tree holding one folder per day+animal, each holding one folder per
    # recording (t1..t5), each holding the per-ROI fluorescence CSV.
    "input_root": "outputs/botox_restani_rebuilt",
    # Where the two summary tables go. The per-recording detection outputs are
    # NOT written here - they stay next to each recording's CSV, exactly where
    # running test_epileptic_detection.py on that file alone would put them.
    "output_dir": "outputs/epileptic_summary",
    "csv_name": "roi_fluorescence_full.csv",
    # Plots are ~90% of the runtime (about 1.4 s of the 1.5 s per recording).
    # Turn them off for a counts-only pass over the whole tree.
    "make_plots": True,
    # Reuse an existing roi_events.csv instead of recomputing it. Only safe when
    # the existing files were produced by the SAME detection settings - the
    # tables would otherwise mix cut-offs - so it is off by default.
    "skip_existing": False,
    # Stop after this many recordings; None runs all of them. For smoke tests.
    "limit": None,
    # Calibration pass for epileptic_criterion "calibrated_z": sweep every
    # recording's frames first, pool them, and solve for the one z threshold
    # that `calibration_percent` of all frames exceed. Costs about a minute for
    # 355 recordings and is cached, so a re-run reuses it.
    "calibrate": True,
    "calibration_percent": 1.0,
    # None -> "<output_dir>/calibration.json". Delete that file to recalibrate.
    "calibration_path": None,
    # Calibrate on this many evenly-spaced recordings instead of all of them.
    # None uses every recording, which is what the shipped calibration did.
    "calibration_sample": None,
    # Overrides forwarded into test_epileptic_detection.RUN_CONFIG, e.g.
    # {"epileptic_scope": "recording", "epileptic_top_percent": 5.0}. Anything
    # left out keeps that file's value, so the two scripts cannot drift apart.
    "detection_overrides": {},
}


def discover_recordings(root: Path, csv_name: str) -> list[dict[str, Any]]:
    """Every `<day>_<animal>/<t#>/<csv_name>` under `root`, in sorted order.

    The day+animal split is on the FIRST underscore only: days are numeric and
    animal names carry the letters (260618_PV3 -> 260618, PV3).
    """
    recordings: list[dict[str, Any]] = []
    for csv_path in sorted(root.glob(f"*/*/{csv_name}")):
        unit = csv_path.parent.parent.name
        if "_" not in unit:
            raise ValueError(f"folder {unit!r} is not <day>_<animal>: {csv_path}")
        day, animal = unit.split("_", 1)
        recordings.append({
            "day": day,
            "animal": animal,
            "recording": csv_path.parent.name,
            "csv_path": csv_path,
        })
    return recordings


def roi_columns(csv_path: Path) -> list[str]:
    """ROI names of one fluorescence CSV, read from the header alone."""
    header = pd.read_csv(csv_path, nrows=0)
    return list(header.columns[detection.STARTING_COLUMN:])


def tally_events(events: pd.DataFrame, roi_names: list[str]) -> dict[str, Any]:
    """Flagged-event count per ROI, plus the totals that give them context.

    Every ROI of the recording gets an entry even when it has no events, so the
    table is rectangular and a zero means "looked at, found nothing" rather than
    "missing".
    """
    counts: dict[str, Any] = {roi: 0 for roi in roi_names}
    n_peaks = len(events)
    n_confirmed = 0
    if n_peaks:
        n_confirmed = int(events["confirmed"].sum())
        flagged = events.loc[events["epileptic"], "roi"].value_counts()
        unknown = sorted(set(flagged.index) - set(roi_names))
        if unknown:
            raise ValueError(f"events in ROIs absent from the CSV header: {unknown}")
        counts.update({roi: int(n) for roi, n in flagged.items()})
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "n_peaks": n_peaks,
        "n_confirmed": n_confirmed,
    }


def load_existing_events(events_path: Path) -> pd.DataFrame | None:
    """Reuse a previous roi_events.csv, or None when there is nothing usable.

    A recording that produced no peaks at all writes a header-only file; that is
    a valid result and comes back as an empty frame, not None.
    """
    if not events_path.is_file() or events_path.stat().st_size == 0:
        return None
    return pd.read_csv(events_path)


def calibrate_z_threshold(
    recordings: list[dict[str, Any]],
    detection_cfg: dict[str, Any],
    top_percent: float,
    sample: int | None = None,
) -> dict[str, Any]:
    """Solve for the one z threshold that `top_percent` of all frames exceed.

    Every recording contributes a histogram of its own robust-z frames; the
    histograms are summed and the threshold read off the pooled tail. Pooling
    matters: a single recording offers only about 300 effective samples (a
    10-frame rolling mean and the calcium decay leave nowhere near 2980
    independent ones) and its 22 ROIs are correlated with each other at r ~ 0.74,
    so a per-recording tail estimate is far noisier than its frame count
    suggests. Across the dataset the tail is set by millions of frames.

    The result is a constant, so it does not move with a recording's own
    activity - which is the point. A recording that is half event ranks high
    against it instead of quietly raising its own bar.
    """
    chosen = recordings
    if sample is not None and sample < len(recordings):
        # Evenly spaced rather than random, so the subset spans every day and
        # animal in discovery order and the result is reproducible.
        step = len(recordings) / float(sample)
        chosen = [recordings[int(i * step)] for i in range(sample)]

    counts = np.zeros(len(detection.Z_HISTOGRAM_BINS) - 1, dtype=np.int64)
    started = time.time()
    for index, recording in enumerate(chosen, start=1):
        z = detection.frame_z_for_csv(recording["csv_path"], detection_cfg)
        counts += detection.z_histogram(z)
        if index % 50 == 0 or index == len(chosen):
            print(f"  calibrating [{index}/{len(chosen)}] "
                  f"{int(counts.sum()):,} frames  {time.time() - started:.0f}s",
                  flush=True)

    threshold = detection.z_threshold_from_histogram(counts, top_percent)
    total = int(counts.sum())
    above = int(counts[detection.Z_HISTOGRAM_BINS[:-1] >= threshold].sum())
    return {
        "z_threshold": threshold,
        "target_percent": top_percent,
        "achieved_percent": 100.0 * above / total,
        "n_recordings": len(chosen),
        "n_frames": total,
        "smooth_window_s": detection_cfg["smooth_window_s"],
        "sampling_rate_hz": detection_cfg["sampling_rate_hz"],
        "bin_width": float(detection.Z_HISTOGRAM_BINS[1] - detection.Z_HISTOGRAM_BINS[0]),
    }


def build_tables(
    rows: list[dict[str, Any]], roi_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two output tables: t1..t5 separate, and t1..t5 summed per animal."""
    value_columns = roi_names + ["total", "n_peaks", "n_confirmed"]
    by_recording = pd.DataFrame(
        rows, columns=["day", "animal", "recording"] + value_columns)

    # sort=False keeps day/animal in discovery order, which is already sorted.
    # n_recordings says how many t# went into each summed row, so an animal
    # missing a session is visible rather than looking like a quiet one.
    grouped = by_recording.groupby(["day", "animal"], sort=False)
    by_animal = grouped[value_columns].sum().reset_index()
    by_animal.insert(2, "n_recordings", grouped.size().to_numpy())
    return by_recording, by_animal


def main(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = {**RUN_CONFIG, **(cfg or {})}

    input_root = Path(cfg["input_root"])
    output_dir = Path(cfg["output_dir"])
    if not input_root.is_dir():
        raise FileNotFoundError(f"input_root does not exist: {input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # One detection config for the whole batch, so every recording is measured
    # the same way. verbose is off: the per-recording report would be 355 blocks
    # of eight lines, so the progress line below replaces it.
    detection_cfg = {
        **detection.RUN_CONFIG,
        **cfg["detection_overrides"],
        "make_plots": bool(cfg["make_plots"]),
        "verbose": False,
    }
    recordings = discover_recordings(input_root, cfg["csv_name"])
    if cfg["limit"] is not None:
        recordings = recordings[: int(cfg["limit"])]
    if not recordings:
        raise FileNotFoundError(
            f"no {cfg['csv_name']} under {input_root}/<day>_<animal>/<t#>/")

    print(f"{len(recordings)} recordings under {input_root}")

    # The calibrated threshold has to come from the same frames the detection
    # will run on, so it is solved here rather than assumed. Cached, because a
    # sweep of 355 recordings is a minute and the answer only changes when the
    # smoothing or the target percent does.
    if detection_cfg["epileptic_criterion"] == "calibrated_z" and cfg["calibrate"]:
        calibration_path = (Path(cfg["calibration_path"])
                            if cfg["calibration_path"] is not None
                            else output_dir / "calibration.json")
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            print(f"  reusing calibration from {calibration_path}")
        else:
            calibration = calibrate_z_threshold(
                recordings, detection_cfg, float(cfg["calibration_percent"]),
                cfg["calibration_sample"])
            calibration_path.write_text(
                json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
        detection_cfg["epileptic_z_threshold"] = calibration["z_threshold"]
        detection_cfg["epileptic_frame_percent"] = calibration["target_percent"]
        print(f"  calibrated z >= {calibration['z_threshold']:.2f} from "
              f"{calibration['n_frames']:,} frames of "
              f"{calibration['n_recordings']} recordings "
              f"(target top {calibration['target_percent']:g}%, "
              f"achieved {calibration['achieved_percent']:.3f}%)")

    # Provenance for the tables: which thresholds produced these counts.
    (output_dir / "detection_config.json").write_text(
        json.dumps({k: v for k, v in detection_cfg.items()
                    if k not in {"csv_path", "output_dir"}},
                   indent=2, sort_keys=True),
        encoding="utf-8")

    if detection_cfg["epileptic_criterion"] == "peak_percentile":
        print(f"  criterion peak_percentile: top "
              f"{detection_cfg['epileptic_top_percent']:g}% by "
              f"{detection_cfg['epileptic_rank_metric']}, "
              f"scope {detection_cfg['epileptic_scope']}")
    else:
        print(f"  criterion {detection_cfg['epileptic_criterion']}, "
              f"target top {detection_cfg['epileptic_frame_percent']:g}% of frames")
    print(f"  plots {'on' if detection_cfg['make_plots'] else 'off'}")

    roi_names: list[str] = []
    rows: list[dict[str, Any]] = []
    by_recording_path = output_dir / "epileptic_counts_by_recording.csv"
    started = time.time()

    for index, recording in enumerate(recordings, start=1):
        csv_path = recording["csv_path"]
        names = roi_columns(csv_path)
        if not roi_names:
            roi_names = names
        elif names != roi_names:
            # ROI columns are the table's schema; a different set (or a
            # different order) would file counts under the wrong area.
            raise ValueError(
                f"{csv_path} has ROI columns {names}, expected {roi_names}")

        run_cfg = {**detection_cfg, "csv_path": str(csv_path), "output_dir": None}
        events = None
        reused = False
        if cfg["skip_existing"]:
            events = load_existing_events(
                detection.resolve_output_dir(run_cfg) / "roi_events.csv")
            reused = events is not None
        if events is None:
            events = detection.main(run_cfg)

        tally = tally_events(events, roi_names)
        rows.append({
            "day": recording["day"],
            "animal": recording["animal"],
            "recording": recording["recording"],
            **tally["counts"],
            "total": tally["total"],
            "n_peaks": tally["n_peaks"],
            "n_confirmed": tally["n_confirmed"],
        })

        # Rewritten after every recording, so an interrupted batch still leaves a
        # valid table of everything finished so far (CLAUDE.md 9.6).
        by_recording, _ = build_tables(rows, roi_names)
        by_recording.to_csv(by_recording_path, index=False)

        elapsed = time.time() - started
        print(f"[{index}/{len(recordings)}] {recording['day']}_{recording['animal']}"
              f"/{recording['recording']}: {tally['total']} epileptiform of "
              f"{tally['n_peaks']} peaks ({tally['n_confirmed']} confirmed)"
              + ("  [reused]" if reused else "")
              + f"  {elapsed:.0f}s", flush=True)

    by_recording, by_animal = build_tables(rows, roi_names)
    by_recording.to_csv(by_recording_path, index=False)
    by_animal_path = output_dir / "epileptic_counts_by_animal.csv"
    by_animal.to_csv(by_animal_path, index=False)

    print(f"{len(by_recording)} recordings, {len(by_animal)} day+animal units, "
          f"{int(by_recording['total'].sum())} epileptiform events "
          f"of {int(by_recording['n_peaks'].sum())} peaks "
          f"in {time.time() - started:.0f}s")
    print(f"  wrote {by_recording_path}")
    print(f"  wrote {by_animal_path}")
    return by_recording, by_animal


if __name__ == "__main__":
    main()
