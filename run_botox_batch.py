"""Batch resting-state FC over the BOTOX_RESTANI manifest, per recording.

This is a *study script* (P2 boundary: dataset policy lives here, mechanism lives
in ``wfci``). It reads the manifest produced by ``scan_botox_dataset.py`` and, for
each (day, animal) row, runs a cerebellar resting-state connectivity analysis over
that animal's ``t1..tn`` interleaved recordings.

SEPARATE (default, ``merge_recordings=False``). Each ``t#`` subfolder is its own
analysis: its own channel split, its own ``trim``, baseline, ROI time-series and
correlation matrix, written to its own output subfolder. Nothing is pooled across
recordings, so per-recording results stay comparable and one bad block cannot
contaminate the others.

MERGE = CONCATENATION (``merge_recordings=True``), for when the consecutive ``t#``
recordings really are one continuous session split across files. Each ``t#`` folder
is split into its two interleaved channels, the first ``trim`` frames of each block
are dropped (the per-acquisition onset transient), and the trimmed blocks are
concatenated frame-by-frame into one long gcamp stream + one long emo stream. That
single stream runs as ONE trial (with the pipeline's own ``trim`` set to 0, since
we already trimmed per block), so there is one baseline, one ROI time-series and
one correlation over the whole concatenation. Note this is concatenation, not
trial-averaging: averaging the blocks' correlation matrices would be right only if
they were separate acquisitions.

Per (day, animal) it writes, under ``<output_root>\\<day>_<animal>\\`` -- and, in
the default separate mode, under a further ``<t#>\\`` subfolder per recording:
    * connectivity[_debug].npz  -- temp_roi, R, R_mean, averaged_traces, labels,
                                   the recording path(s), Bregma, profile.
    * roi_traces[_debug].png    -- trial-averaged ROI traces.
    * roi_overlay_debug.png     -- ROI placement (debug/in-memory mode only).
One summary row per recording (or per unit when merging) is appended to
``<output_root>\\batch_summary.csv``.

Two modes, mirroring ``run_intermingle_rs.py``:
    * debug = True  -- fast smoke test. Loads only ``debug_max_frames`` frames per
      channel per recording into RAM and runs ``run_profile``, optionally capping
      the number of recordings used (``max_recordings``). Keeps ``dff_stack`` so it
      can draw the ROI-placement overlay. Use to validate setup.
    * debug = False -- the real run. Streams every frame of every recording in
      constant memory (``run_streaming_profile``, two passes per trial). A single
      recording is ~3 GB and a full unit ~15 GB across 5, so streaming is
      mandatory. No overlay.

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
    conda run -n letizia python run_botox_batch.py --merge-recordings   # one concatenated trial per animal

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
    # roi_set is the 22-box cortical layout (wfci.atlases.CORTEX_22 -- the MATLAB's
    # img_av(y_1+.., x_2+..) boxes), shared by every unit. Only the ATLAS is
    # replaced: the manifest's profile column stays cerebellar_rs, so the pipeline
    # chain (trim, no mask, no GSR) is unchanged -- just 22 ROIs instead of 4.
    # Each file's own Bregma wins over the manifest's, so once you have drawn
    # per-animal sets with roi_editor.py, set roi_set_dir = r"roi_sets" and those
    # take precedence over this shared file.
    "roi_set_dir": None,
    "roi_set": r"roi_sets\cortex22_roi_set.yaml",
    # How the animal's t1..tn recordings are treated:
    #   False (default) -- each t# subfolder is its own analysis, with its own
    #     trim/baseline/correlation and its own <day>_<animal>\<t#>\ output folder.
    #   True            -- all t# are concatenated into ONE continuous trial
    #     (see the module docstring); one output folder per animal.
    "merge_recordings": False,
    # Debug smoke test (in-memory, limited frames) vs full streaming run.
    "debug": True,
    # Debug only: frames PER CHANNEL PER RECORDING to load into RAM.
    "debug_max_frames": 60,
    # Cap how many t# recordings are used (None = all). Applies to BOTH modes and
    # both merge settings -- e.g. 2 = only t1+t2. The real analysis uses all 5.
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
                   metavar="K", help="Cap the recordings used (both modes); e.g. 2 = t1+t2.")
    p.add_argument("--merge-recordings", action=argparse.BooleanOptionalAction,
                   default=defaults["merge_recordings"],
                   help="Concatenate an animal's t# recordings into ONE trial. "
                        "Default (--no-merge-recordings): analyse each t# separately.")
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
        merge_recordings=bool(config["merge_recordings"]),
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


def _run_one_recording(folder: str, cfg, profile, channel_order: str,
                       args: argparse.Namespace):
    """Run ONE t# recording as its own trial -- the default, unmerged path.

    Nothing is shared with the animal's other recordings: this folder's own
    interleaved split feeds the profile unchanged, so the profile's own ``trim``
    drops this acquisition's onset transient and the baseline/correlation windows
    are computed over this recording alone. Same math as
    ``run_intermingle_rs.py`` on a single folder.
    """
    gcamp_files, emo_files = interleaved_channel_files(folder, channel_order=channel_order)
    n_per_channel = min(len(gcamp_files), len(emo_files))
    gcamp_files, emo_files = list(gcamp_files[:n_per_channel]), list(emo_files[:n_per_channel])

    if args.debug:
        limit = min(args.debug_frames, n_per_channel)
        if limit <= profile.trim:
            raise ValueError(
                f"{folder}: debug frames={limit} is at or below the profile's "
                f"{profile.trim}-frame trim; nothing left to correlate."
            )
        print(f"Mode   : DEBUG (in-memory) -- first {limit} frames/channel "
              f"of {n_per_channel} (trim {profile.trim})")
        trial = _load_limited_trial(gcamp_files, emo_files, limit)
        return run_profile([trial], cfg, profile)

    print(f"Mode   : FULL (streaming) -- {n_per_channel} frames/channel "
          f"(trim {profile.trim})")
    sources = [(folder_frame_source(gcamp_files), folder_frame_source(emo_files))]
    return run_streaming_profile(sources, cfg, profile)


def _run_merged_recordings(rec_paths: list[str], cfg, profile, channel_order: str,
                           args: argparse.Namespace):
    """Run an animal's recordings concatenated into ONE continuous trial.

    We trim each block's onset ourselves, then concatenate, so the pipeline's own
    per-trial trim must be disabled -- otherwise it would trim the concatenation
    once more at its very start.
    """
    block_trim = profile.trim
    run_prof = replace(profile, trim=0)
    if args.debug:
        print(f"Mode   : DEBUG (in-memory) -- concatenate {len(rec_paths)} recordings, "
              f"first {args.debug_frames} frames/channel each (onset trim {block_trim})")
        trial = _build_concat_debug_trial(
            rec_paths, channel_order, args.debug_frames, block_trim
        )
        return run_profile([trial], cfg, run_prof)

    print(f"Mode   : FULL (streaming) -- concatenate {len(rec_paths)} recordings into "
          f"one trial (onset trim {block_trim})")
    source = _build_concat_streaming_source(rec_paths, channel_order, block_trim)
    return run_streaming_profile([source], cfg, run_prof)


def _save_outputs(result, out_dir: str, args: argparse.Namespace, cfg, profile,
                  meta: dict[str, Any]) -> str:
    """Write the overlay/traces/npz for one analysis. Returns the .npz path.

    Shared by the separate and merged paths so both write the same files with the
    same names -- only the directory (and ``meta``) differ.
    """
    os.makedirs(out_dir, exist_ok=True)
    labels = profile.labels

    if getattr(result, "dff_stack", None) is not None:
        # RAM (debug) mode only: the placement overlay needs the DFF stack.
        overlay_path = os.path.join(out_dir, "roi_overlay_debug.png")
        _save_overlay(result.dff_stack, cfg, profile, overlay_path)
        print(f"Overlay: {overlay_path}  <- CHECK the boxes sit on the anatomy")

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
        **{k: np.array(v) for k, v in meta.items()},
    )
    if getattr(result, "dff_stack", None) is not None:
        arrays["dff_stack"] = result.dff_stack
    np.savez(npz_path, **arrays)
    print(f"Saved  : {npz_path}")
    return npz_path


def run_unit(row: dict[str, str], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run one (day, animal) unit and save its results.

    Returns one summary row per analysis: one per recording in the default
    separate mode, or a single merged row when ``--merge-recordings`` is set.
    """
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

    unit_dir = os.path.join(args.output_root, f"{day}_{animal}")
    # Which recordings to use (cap applies to both modes; e.g. 2 -> t1+t2).
    used = rec_paths if args.max_recordings is None else rec_paths[:args.max_recordings]

    print(f"\n=== {unit} (group {row['group']}) ===")
    print(f"Bregma : row={bregma_row}, col={bregma_col} "
          f"(downsampled y_1={cfg.y_1}, x_2={cfg.x_2})")
    print(f"Recs   : {len(used)}/{len(rec_paths)} "
          f"{'CONCATENATED into one trial' if args.merge_recordings else 'analysed SEPARATELY'}")

    # Common per-unit provenance saved into every .npz alongside the arrays.
    base_meta = dict(
        day=day,
        animal=animal,
        group=row["group"],
        bregma=[bregma_row, bregma_col],
        # Which geometry produced these numbers ("" = the profile's own atlas).
        roi_set=roi_set or "",
    )
    base_summary = dict(
        day=day,
        animal=animal,
        group=row["group"],
        profile=profile.name,
        bregma_row=bregma_row,
        bregma_col=bregma_col,
        roi_set=roi_set or "",
        mode="debug" if args.debug else "full",
    )

    if args.merge_recordings:
        t0 = time.perf_counter()
        result = _run_merged_recordings(used, cfg, profile, channel_order, args)
        elapsed = time.perf_counter() - t0
        print(f"Time   : {elapsed:.1f}s for {len(used)} recording(s)")
        npz_path = _save_outputs(result, unit_dir, args, cfg, profile,
                                 dict(base_meta, merged_recordings=used))
        return [dict(base_summary, recording="merged", n_recordings=len(used),
                     mean_offdiag_R=_mean_offdiag(result.R_mean),
                     elapsed_s=round(elapsed, 1), npz_path=npz_path)]

    summaries = []
    for folder in used:
        # Each recording keeps its own identity in the output tree: the t# folder
        # name becomes a subfolder, so t1 and t2 never overwrite each other.
        rec_name = os.path.basename(folder.rstrip("\\/")) or "recording"
        print(f"\n--- {unit} / {rec_name} ---")
        print(f"Folder : {folder}")
        t0 = time.perf_counter()
        result = _run_one_recording(folder, cfg, profile, channel_order, args)
        elapsed = time.perf_counter() - t0
        print(f"Time   : {elapsed:.1f}s")
        npz_path = _save_outputs(result, os.path.join(unit_dir, rec_name), args, cfg,
                                 profile, dict(base_meta, recording=rec_name,
                                               recording_path=folder))
        summaries.append(dict(base_summary, recording=rec_name, n_recordings=1,
                              mean_offdiag_R=_mean_offdiag(result.R_mean),
                              elapsed_s=round(elapsed, 1), npz_path=npz_path))
    return summaries


def _mean_offdiag(R_mean: np.ndarray) -> float:
    """Mean of the off-diagonal correlations -- a crude single-number handle on
    connectivity strength, useful for eyeballing the batch summary at a glance."""
    n = R_mean.shape[0]
    return float(np.nanmean(R_mean[~np.eye(n, dtype=bool)]))


def main() -> None:
    args = build_runtime_args()
    rows = load_manifest(args.manifest)
    chosen = select_rows(rows, args.select, args.max_rows)

    keys = [f"{r['day']}/{r['animal']}" for r in chosen]
    print(f"Manifest : {args.manifest} ({len(rows)} rows)")
    print(f"Selected : {len(chosen)} unit(s) -> {keys}")
    print(f"Mode     : {'DEBUG' if args.debug else 'FULL streaming'}")
    print("Recs     : " + ("merged into one trial per animal" if args.merge_recordings
                           else "kept separate (one analysis per t#)"))

    os.makedirs(args.output_root, exist_ok=True)
    # One row per analysis: per recording when separate, per unit when merging.
    summaries = [s for row in chosen for s in run_unit(row, args)]

    summary_csv = os.path.join(args.output_root, "batch_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nBatch summary ({len(summaries)} row(s)) -> {summary_csv}")


if __name__ == "__main__":
    main()
