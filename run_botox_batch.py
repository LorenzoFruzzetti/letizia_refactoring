"""Batch resting-state FC over the BOTOX_RESTANI manifest, merging recordings.

This is a *study script* (P2 boundary: dataset policy lives here, mechanism lives
in ``wfci``). It reads the manifest produced by ``scan_botox_dataset.py`` and, for
each (day, animal) row, runs ONE cerebellar resting-state connectivity analysis
over ALL of that animal's ``t1..tn`` recordings *concatenated into a single
continuous trial* -- because for this dataset the consecutive ``t#`` recordings
are one resting-state session split across files.

MERGE = CONCATENATION (not trial-averaging). Each ``t#`` folder is split into its
two interleaved channels, the first ``trim`` frames of each block are dropped (the
per-acquisition onset transient), and the trimmed blocks are concatenated
frame-by-frame into one long gcamp stream + one long emo stream. That single
stream is then run as ONE trial (with the pipeline's own ``trim`` set to 0, since
we already trimmed per block), so there is one baseline, one ROI time-series, and
one correlation over the whole concatenated recording. The alternative -- treating
each ``t#`` as a separate trial and averaging their correlation matrices -- would
be right only if the blocks were separate acquisitions; here they are continuous.

Per (day, animal) it writes, under ``<output_root>\\<day>_<animal>\\``:
    * connectivity[_debug].npz  -- temp_roi, R, R_mean, averaged_traces, labels,
                                   the merged recording paths, Bregma, profile.
    * roi_traces[_debug].png    -- trial-averaged ROI traces.
    * roi_overlay_debug.png     -- ROI placement (debug/in-memory mode only).
And one summary row per unit is appended to ``<output_root>\\batch_summary.csv``.

Two modes, mirroring ``run_intermingle_rs.py``:
    * debug = True  -- fast smoke test. Loads only ``debug_max_frames`` frames per
      channel per recording into RAM (after the per-block trim), concatenates the
      blocks into one trial, and runs ``run_profile``, optionally capping the
      number of concatenated recordings (``debug_max_recordings``). Keeps
      ``dff_stack`` so it can draw the ROI-placement overlay. Use to validate setup.
    * debug = False -- the real run. Streams the concatenation of every frame of
      every recording in constant memory (``run_streaming_profile``, two passes
      over the one long trial). A full unit is ~15 GB across 5 recordings, so
      streaming is mandatory. No overlay.

ROI geometry: each unit uses its profile's atlas and the manifest's Bregma unless a
ROI set drawn with ``roi_editor.py`` is configured -- ``roi_set_dir`` for per-unit
files (``<day>_<animal>.yaml``) or ``roi_set`` for one file shared by every unit.

Row selection:
    * ``select`` -- list of "day/animal" keys to run (e.g. ["260611/R1"]).
      Empty list = every row in the manifest.
    * ``max_rows`` -- cap the number of rows actually run (after ``select``).

Run it:
    conda run -n letizia python run_botox_batch.py                  # editor mode, uses RUN_CONFIG
    conda run -n letizia python run_botox_batch.py --full           # force full streaming
    conda run -n letizia python run_botox_batch.py --select 260611/R1 260611/R2

Edit ``RUN_CONFIG`` and run -- no CLI flags needed. Errors are left to surface.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import replace
from typing import Any

import numpy as np

from wfci import (
    ROIConfig,
    folder_frame_source,
    get_profile,
    interleaved_channel_files,
    run_profile,
    run_streaming_profile,
)

# Reuse the single-folder helpers instead of duplicating them -- run_intermingle_rs
# guards its own run behind ``if __name__ == "__main__"``, so importing is inert.
from run_intermingle_rs import (
    _load_limited_trial,
    _save_overlay,
    _save_traces,
    apply_roi_set,
)

# ---------------------------------------------------------------------------
# Edit this section to run without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Manifest written by scan_botox_dataset.py.
    "manifest": r"manifests\botox_restani_manifest.csv",
    # Root under which per-unit output folders are created.
    "output_root": r"outputs\botox_restani",
    # Which units to run: "day/animal" keys. Empty list = all rows in the manifest.
    "select": ["260611/R1", "260611/R2"],
    # Cap on rows actually run after `select` (None = no cap). Handy for a quick test.
    "max_rows": None,
    # ROI geometry (boxes + Bregma) drawn with roi_editor.py. Two ways to use it:
    #   * roi_set_dir -- per-unit files, <roi_set_dir>\<day>_<animal>.yaml. Each
    #     animal gets its own ROIs/Bregma. A unit with no file falls back to
    #     roi_set, and if that is None too, to the profile atlas + the manifest's
    #     Bregma. This is the "check every session" workflow.
    #   * roi_set     -- ONE file used for every unit ("the same ROIs for all
    #     sessions"), e.g. r"roi_sets\shared_roi_set.yaml".
    # Set both to None to keep the previous behaviour exactly.
    "roi_set_dir": None,
    "roi_set": None,
    # Debug smoke test (in-memory, limited frames) vs full streaming run.
    "debug": True,
    # Debug only: frames PER CHANNEL PER RECORDING to load into RAM.
    "debug_max_frames": 60,
    # Cap how many t# recordings are concatenated (None = all). Applies to BOTH
    # modes -- e.g. set to 2 to run/time only t1+t2. The real analysis uses all 5.
    "max_recordings": None,
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Minimal CLI overrides for the knobs worth flipping from the terminal."""
    p = argparse.ArgumentParser(description="Batch RS-FC over the BOTOX manifest.")
    p.add_argument("--manifest", default=defaults["manifest"], help="Manifest CSV.")
    p.add_argument("--output-root", default=defaults["output_root"])
    p.add_argument("--select", nargs="*", default=defaults["select"],
                   metavar="DAY/ANIMAL", help="Units to run; omit for all.")
    p.add_argument("--max-rows", type=int, default=defaults["max_rows"])
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"], metavar="DIR",
                   help="Per-unit ROI sets: DIR\\<day>_<animal>.yaml (roi_editor.py).")
    p.add_argument("--roi-set", default=defaults["roi_set"], metavar="PATH",
                   help="One ROI set for every unit (fallback when --roi-set-dir has no file).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", dest="debug", action="store_false", default=defaults["debug"],
                      help="Full streaming run over every frame of every recording.")
    mode.add_argument("--debug", dest="debug", action="store_true",
                      help="Debug smoke test on the first --debug-frames frames.")
    p.add_argument("--debug-frames", type=int, default=defaults["debug_max_frames"],
                   metavar="N", help="Debug mode: frames per channel per recording.")
    p.add_argument("--max-recordings", type=int, default=defaults["max_recordings"],
                   metavar="K", help="Cap concatenated recordings (both modes); e.g. 2 = t1+t2.")
    ns = p.parse_args()
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Prefer CLI args when any are given, else fall back to RUN_CONFIG (editor mode)."""
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        manifest=config["manifest"],
        output_root=config["output_root"],
        select=list(config["select"]),
        max_rows=config["max_rows"],
        roi_set_dir=config["roi_set_dir"],
        roi_set=config["roi_set"],
        debug=bool(config["debug"]),
        debug_frames=config["debug_max_frames"],
        max_recordings=config["max_recordings"],
    )


def load_manifest(path: str) -> list[dict[str, str]]:
    """Read the manifest CSV into a list of row dicts (order preserved)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select_rows(rows: list[dict[str, str]], keys: list[str],
                max_rows: int | None) -> list[dict[str, str]]:
    """Filter manifest rows by "day/animal" keys, then cap to ``max_rows``."""
    if keys:
        wanted = set(keys)
        chosen = [r for r in rows if f"{r['day']}/{r['animal']}" in wanted]
        missing = wanted - {f"{r['day']}/{r['animal']}" for r in chosen}
        if missing:
            raise ValueError(f"select keys not found in manifest: {sorted(missing)}")
    else:
        chosen = list(rows)
    if max_rows is not None:
        chosen = chosen[:max_rows]
    return chosen


def resolve_roi_set(day: str, animal: str, args: argparse.Namespace) -> str | None:
    """Which ROI-set file this unit should use, or None for the profile default.

    Per-unit file first (``<roi_set_dir>\\<day>_<animal>.yaml``), then the single
    shared ``roi_set``. A missing per-unit file is NOT an error when a shared file
    is configured -- that is the intended "same ROIs for all sessions, except the
    ones I drew" mix -- but a ``roi_set_dir`` with no file and no shared fallback
    means this unit silently keeps the preset geometry, so say so out loud.
    """
    if args.roi_set_dir:
        per_unit = os.path.join(args.roi_set_dir, f"{day}_{animal}.yaml")
        if os.path.isfile(per_unit):
            return per_unit
        print(f"  no ROI set at {per_unit}"
              + (f" -- falling back to {args.roi_set}" if args.roi_set
                 else " -- using the profile's atlas and the manifest Bregma"))
    return args.roi_set


def _split_and_trim(folder: str, channel_order: str, trim: int):
    """Split one recording into (gcamp_files, emo_files) with the onset trimmed.

    ``interleaved_channel_files`` decodes exactly 2 images of THIS folder to decide
    which interleaved group is GCaMP (so each block resolves its own channel
    order), then the first ``trim`` files of each channel are dropped -- the
    per-acquisition onset transient that the pipeline would otherwise trim itself.
    Returns the trimmed, equal-length channel file lists.
    """
    gcamp_files, emo_files = interleaved_channel_files(folder, channel_order=channel_order)
    n = min(len(gcamp_files), len(emo_files))
    if n <= trim:
        raise ValueError(
            f"{folder}: only {n} frames/channel, at or below the {trim}-frame "
            f"onset trim; nothing left after trimming this block."
        )
    return list(gcamp_files[trim:n]), list(emo_files[trim:n])


def _build_concat_streaming_source(rec_paths: list[str], channel_order: str, trim: int):
    """Concatenate all trimmed blocks into ONE (gcamp_src, emo_src) streaming trial.

    Each block is split + onset-trimmed, then the per-block channel file lists are
    concatenated in recording order into one long ordered list per channel. Wrapped
    as a single re-iterable ``FrameSource`` pair, this is the one continuous trial
    the streaming pipeline runs (with the pipeline's own trim set to 0, since the
    blocks are already trimmed).
    """
    all_gcamp: list = []
    all_emo: list = []
    for folder in rec_paths:
        g_files, e_files = _split_and_trim(folder, channel_order, trim)
        all_gcamp += g_files
        all_emo += e_files
    return folder_frame_source(all_gcamp), folder_frame_source(all_emo)


def _build_concat_debug_trial(rec_paths: list[str], channel_order: str, limit: int,
                              trim: int):
    """Load ``limit`` post-trim frames/channel from each block, concatenated to one trial.

    Mirrors the streaming concatenation but in-memory and clipped, for a fast smoke
    test: each block is split + onset-trimmed, its first ``limit`` remaining frames
    per channel are decoded into RAM, and the blocks are concatenated along time
    into a single ``(gcamp, emo)`` ``[y, x, time]`` trial.
    """
    gcamp_parts: list[np.ndarray] = []
    emo_parts: list[np.ndarray] = []
    for folder in rec_paths:
        g_files, e_files = _split_and_trim(folder, channel_order, trim)
        take = min(limit, len(g_files))
        g, e = _load_limited_trial(g_files, e_files, take)
        gcamp_parts.append(g)
        emo_parts.append(e)
    # Concatenate along the time axis (axis=-1 of [y, x, time]) into one trial.
    return np.concatenate(gcamp_parts, axis=-1), np.concatenate(emo_parts, axis=-1)


def run_unit(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    """Run one (day, animal) unit: merge its recordings and save the results."""
    day, animal = row["day"], row["animal"]
    unit = f"{day}/{animal}"
    rec_paths = [p for p in row["recording_paths"].split(";") if p]
    bregma_row, bregma_col = int(row["bregma_row"]), int(row["bregma_col"])
    channel_order = row["channel_order"]

    profile = get_profile(row["profile"])
    # An optional ROI set (drawn with roi_editor.py) overrides the profile's atlas
    # and the manifest's Bregma -- boxes and Bregma always travel together.
    roi_set = resolve_roi_set(day, animal, args)
    profile, bregma_row, bregma_col = apply_roi_set(profile, bregma_row, bregma_col, roi_set)
    # The profile supplies the ROI atlas; cfg carries only the per-animal Bregma.
    cfg = ROIConfig.from_bregma(bregma_row, bregma_col, boxes=dict(profile.atlas))

    out_dir = os.path.join(args.output_root, f"{day}_{animal}")
    os.makedirs(out_dir, exist_ok=True)

    # Which blocks to concatenate (cap applies to both modes; e.g. 2 -> t1+t2).
    merged = rec_paths if args.max_recordings is None else rec_paths[:args.max_recordings]
    # We trim each block's onset ourselves, then concatenate, so the pipeline's own
    # per-trial trim must be disabled -- otherwise it would trim the concatenation
    # once more at its very start.
    block_trim = profile.trim
    run_prof = replace(profile, trim=0)

    print(f"\n=== {unit} (group {row['group']}) ===")
    print(f"Bregma : row={bregma_row}, col={bregma_col} "
          f"(downsampled y_1={cfg.y_1}, x_2={cfg.x_2})")

    t0 = time.perf_counter()
    if args.debug:
        print(f"Mode   : DEBUG (in-memory) -- concatenate {len(merged)}/{len(rec_paths)} "
              f"recordings, first {args.debug_frames} frames/channel each "
              f"(onset trim {block_trim})")
        trial = _build_concat_debug_trial(merged, channel_order, args.debug_frames, block_trim)
        result = run_profile([trial], cfg, run_prof)
        overlay_path = os.path.join(out_dir, "roi_overlay_debug.png")
        _save_overlay(result.dff_stack, cfg, profile, overlay_path)
        print(f"Overlay: {overlay_path}  <- CHECK the boxes sit on the anatomy")
    else:
        print(f"Mode   : FULL (streaming) -- concatenate {len(merged)}/{len(rec_paths)} "
              f"recordings into one trial (onset trim {block_trim})")
        source = _build_concat_streaming_source(merged, channel_order, block_trim)
        result = run_streaming_profile([source], cfg, run_prof)
    elapsed = time.perf_counter() - t0
    print(f"Time   : {elapsed:.1f}s for {len(merged)} recording(s)")

    labels = profile.labels
    print(f"R_mean ({profile.n_rois}x{profile.n_rois}), ROI order {labels}:")
    print(np.array2string(result.R_mean, precision=4, suppress_small=True))

    traces_path = os.path.join(out_dir, "roi_traces_debug.png" if args.debug
                               else "roi_traces_full.png")
    _save_traces(result.averaged_traces, labels, traces_path)

    npz_path = os.path.join(out_dir, "connectivity_debug.npz" if args.debug
                            else "connectivity_full.npz")
    arrays = dict(
        temp_roi=result.temp_roi,
        R=result.R,
        R_mean=result.R_mean,
        averaged_traces=result.averaged_traces,
        roi_labels=np.array(labels),
        profile=np.array(profile.name),
        day=np.array(day),
        animal=np.array(animal),
        group=np.array(row["group"]),
        bregma=np.array([bregma_row, bregma_col]),
        merged_recordings=np.array(merged),
        # Which geometry produced these numbers ("" = the profile's own atlas).
        roi_set=np.array(roi_set or ""),
    )
    if getattr(result, "dff_stack", None) is not None:
        arrays["dff_stack"] = result.dff_stack
    np.savez(npz_path, **arrays)
    print(f"Saved  : {npz_path}")

    # One flat summary row: the off-diagonal mean is a crude single-number handle
    # on the connectivity strength, useful for eyeballing the batch at a glance.
    n = result.R_mean.shape[0]
    off_diag = result.R_mean[~np.eye(n, dtype=bool)]
    return {
        "day": day,
        "animal": animal,
        "group": row["group"],
        "n_recordings_merged": len(merged),
        "profile": profile.name,
        "bregma_row": bregma_row,
        "bregma_col": bregma_col,
        "roi_set": roi_set or "",
        "mean_offdiag_R": float(np.nanmean(off_diag)),
        "mode": "debug" if args.debug else "full",
        "elapsed_s": round(elapsed, 1),
        "npz_path": npz_path,
    }


def main() -> None:
    args = build_runtime_args()
    rows = load_manifest(args.manifest)
    chosen = select_rows(rows, args.select, args.max_rows)

    keys = [f"{r['day']}/{r['animal']}" for r in chosen]
    print(f"Manifest : {args.manifest} ({len(rows)} rows)")
    print(f"Selected : {len(chosen)} unit(s) -> {keys}")
    print(f"Mode     : {'DEBUG' if args.debug else 'FULL streaming'}")

    os.makedirs(args.output_root, exist_ok=True)
    summaries = [run_unit(row, args) for row in chosen]

    summary_csv = os.path.join(args.output_root, "batch_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nBatch summary ({len(summaries)} units) -> {summary_csv}")


if __name__ == "__main__":
    main()
