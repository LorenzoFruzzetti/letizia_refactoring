"""Resting-state functional connectivity on one interleaved ("intermingle") folder.

This is a *study script* (P2 boundary: policy lives here, mechanism lives in
``wfci``). It runs the cerebellar resting-state pipeline on a single folder of
interleaved single-page TIFFs -- odd-positioned images are one channel, even the
other -- and writes the 4x4 connectivity matrix plus a couple of check figures.

Target dataset (edit ``RUN_CONFIG`` for another):
    \\146.48.88.209\share2\BOTOX_RESTANI\260611\R1\t1
    6000 single-page 512x512 TIFFs = 3000 frames/channel, ~3 GB.

Two run modes, selected by ``RUN_CONFIG["debug"]``:

* ``debug = True``  -- a fast smoke test on the first ``debug_max_frames`` frames
  PER CHANNEL, loaded into RAM (``run_profile``). RAM mode keeps ``dff_stack``, so
  it can also draw the step-2 ROI-placement overlay -- the visual check that the
  Bregma coordinates put the 4 boxes on the right anatomy. Use this to validate
  the setup before committing to the full recording.
* ``debug = False`` -- the real run. Streams the whole folder frame-by-frame in
  constant memory (``run_streaming_profile``, two passes), which is what a ~3 GB
  recording needs. No ``dff_stack`` is retained in this mode (streamed away), so
  no overlay -- confirm placement in the debug run first.

ROI geometry comes from the profile's atlas + ``bregma_row``/``bregma_col`` unless
``RUN_CONFIG["roi_set"]`` (or ``--roi-set``) points at a ROI-set YAML drawn with
``roi_editor.py``; that file carries both the boxes and the Bregma they were drawn
from, and then supplies both.

Run it:
    conda run -n letizia python run_intermingle_rs.py                 # editor mode, uses RUN_CONFIG
    conda run -n letizia python run_intermingle_rs.py --full          # force the full streaming run
    conda run -n letizia python run_intermingle_rs.py --debug-frames 80
    conda run -n letizia python run_intermingle_rs.py --roi-set roi_sets\260611_R1.yaml --full

No CLI flags are needed -- edit ``RUN_CONFIG`` and run. Errors are intentionally
left to surface (no try/except) so a bad path or shape fails loudly.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
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

# The ROI-set file format (boxes + the Bregma they were drawn from) is owned by
# the editor utility; importing it here is inert (its run is __main__-guarded).
from roi_editor import load_roi_set

# ---------------------------------------------------------------------------
# Edit this section to run without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # The single interleaved folder for one trial (odd/even = the two channels).
    # UNC path as a raw string so the backslashes are not read as escapes.
    "folder": r"\\146.48.88.209\share2\BOTOX_RESTANI\260611\R1\t1",
    # Which pipeline: cerebellar resting state (4 ROIs, full-recording baseline
    # and correlation window). The intermingle modality is the cerebellar port.
    "profile": "cerebellar_rs",
    # Channel assignment for the interleaved split. "auto" picks the brighter
    # group (top-10% pixel intensity) as GCaMP; for this dataset the odd frames
    # (R11_00001, 00003, ...) are the bright GCaMP channel, so "auto" is correct.
    "channel_order": "auto",
    # Per-animal Bregma at FULL resolution (floor(.../2) is applied internally to
    # land on the downsampled grid). VERIFY these against the ROI overlay the debug
    # run writes before trusting the full run's numbers -- ROI placement is the
    # one thing this script cannot check for you.
    "bregma_row": 121,
    "bregma_col": 134,
    # Optional ROI set written by roi_editor.py: a YAML holding the ROI boxes AND
    # the Bregma they were drawn from. When given it REPLACES the profile's atlas
    # and (if the file carries one) the bregma_row/bregma_col above -- the boxes
    # are offsets from that Bregma, so the two travel together. None = the
    # profile's own atlas with the coordinates above.
    #   e.g. r"roi_sets\260611_R1.yaml"      (this animal)
    #        r"roi_sets\shared_roi_set.yaml" (the same layout for every session)
    "roi_set": None,
    # Debug smoke test vs the real streaming run.
    "debug": True,
    # Debug only: frames PER CHANNEL to load (reads the folder's first 2*N images,
    # since they alternate). Must exceed the profile's 20-frame trim; keep it small
    # for a quick check but large enough that a correlation means something.
    "debug_max_frames": 60,
    # Where the .npz result and check figures go (created if missing).
    "output_dir": "outputs/intermingle_R1_t1",
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Minimal CLI overrides for the few knobs worth flipping from the terminal."""
    p = argparse.ArgumentParser(description="Resting-state FC on one interleaved folder.")
    p.add_argument("--folder", default=defaults["folder"], help="Interleaved TIFF folder.")
    p.add_argument("--bregma-row", type=int, default=defaults["bregma_row"])
    p.add_argument("--bregma-col", type=int, default=defaults["bregma_col"])
    p.add_argument("--output-dir", default=defaults["output_dir"])
    p.add_argument("--roi-set", default=defaults["roi_set"],
                   metavar="PATH", help="ROI set YAML from roi_editor.py (boxes + Bregma).")
    # --full and --debug are mutually exclusive ways to pick the mode; without
    # either, the RUN_CONFIG["debug"] value stands.
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", dest="debug", action="store_false", default=defaults["debug"],
                      help="Full streaming run over every frame.")
    mode.add_argument("--debug", dest="debug", action="store_true",
                      help="Debug smoke test on the first --debug-frames frames.")
    p.add_argument("--debug-frames", type=int, default=defaults["debug_max_frames"],
                   metavar="N", help="Debug mode: frames per channel.")
    ns = p.parse_args()
    ns.profile = defaults["profile"]
    ns.channel_order = defaults["channel_order"]
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Prefer CLI args when any are given, else fall back to RUN_CONFIG (editor mode)."""
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        folder=config["folder"],
        profile=config["profile"],
        channel_order=config["channel_order"],
        bregma_row=config["bregma_row"],
        bregma_col=config["bregma_col"],
        roi_set=config["roi_set"],
        debug=bool(config["debug"]),
        debug_frames=config["debug_max_frames"],
        output_dir=config["output_dir"],
    )


def apply_roi_set(profile, bregma_row: int, bregma_col: int, roi_set: str | None):
    """Swap in the geometry from a ROI-set file. Returns ``(profile, row, col)``.

    ``roi_set`` is a file written by ``roi_editor.py``: the ROI boxes plus the
    Bregma they were drawn from. The boxes replace the profile's atlas (via
    ``dataclasses.replace``, so the atlas's declared ``grid`` still guards against
    a layout drawn for another field of view), and the file's Bregma replaces the
    caller's -- the two are one measurement, and using boxes from one animal with
    the Bregma of another would move every ROI while still producing numbers.
    ``None`` leaves everything as it was.
    """
    if not roi_set:
        return profile, bregma_row, bregma_col
    atlas, file_row, file_col = load_roi_set(roi_set)
    profile = replace(profile, atlas=atlas)
    if file_row is not None:
        bregma_row, bregma_col = file_row, file_col
    print(f"ROI set: {roi_set} -> {len(atlas)} ROIs {list(atlas)}, grid {atlas.grid}, "
          f"Bregma row={bregma_row} col={bregma_col}")
    return profile, bregma_row, bregma_col


def _load_limited_trial(gcamp_files, emo_files, limit):
    """Load only the first ``limit`` frames per channel into RAM (debug mode).

    Goes through the lazy folder sources and ``islice`` so exactly ``limit`` images
    per channel are decoded off the (network) disk -- not the whole 6000-file
    folder -- then stacks them to the ``[y, x, time]`` array the in-memory pipeline
    expects. Returns one ``(gcamp, emo)`` trial.
    """
    gcamp_src = folder_frame_source(gcamp_files)
    emo_src = folder_frame_source(emo_files)
    gcamp = np.stack(list(itertools.islice(gcamp_src.open(), limit)), axis=-1)
    emo = np.stack(list(itertools.islice(emo_src.open(), limit)), axis=-1)
    return gcamp, emo


def _save_overlay(dff_stack, cfg, profile, out_path):
    """Draw the step-2 ROI-placement overlay on a representative DFF frame.

    This is the Bregma sanity check: the 4 boxes are drawn from the same ``cfg``
    the pipeline averages, so if they land off the anatomy the coordinates are
    wrong. Only available in the RAM (debug) mode, which keeps ``dff_stack``.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    from wfci import show_roi_placement

    # overlay_frame may exceed a short debug clip; clamp to a valid frame index.
    frame_index = min(profile.overlay_frame, dff_stack.shape[2] - 1)
    ax = show_roi_placement(dff_stack, cfg, frame_index=frame_index, average_trials=False)
    ax.set_title(f"{profile.name}: ROI placement @ Bregma "
                 f"(frame {frame_index}) -- verify boxes sit on the anatomy")
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def _save_traces(averaged_traces, labels, out_path):
    """Plot the trial-averaged ROI DFF traces -- a quick 'did signals come out' look."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for j, label in enumerate(labels):
        ax.plot(averaged_traces[:, j], label=label, lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel(r"$\Delta$F/F (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Trial-averaged ROI traces")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_runtime_args()
    os.makedirs(args.output_dir, exist_ok=True)

    profile = get_profile(args.profile)
    # An optional ROI set overrides both the atlas and the Bregma (see apply_roi_set).
    profile, bregma_row, bregma_col = apply_roi_set(
        profile, args.bregma_row, args.bregma_col, args.roi_set
    )
    # The profile supplies the ROI atlas; cfg carries only the per-animal Bregma.
    cfg = ROIConfig.from_bregma(bregma_row, bregma_col, boxes=dict(profile.atlas))

    # Split the interleaved folder into two channel file lists. Decodes exactly two
    # images to decide which group is brighter (=GCaMP); nothing large is read here.
    gcamp_files, emo_files = interleaved_channel_files(
        args.folder, channel_order=args.channel_order
    )
    n_per_channel = min(len(gcamp_files), len(emo_files))
    print(f"Folder : {args.folder}")
    print(f"Split  : {len(gcamp_files)} gcamp + {len(emo_files)} emo images "
          f"({n_per_channel} frames/channel), channel_order={args.channel_order}")
    print(f"Bregma : row={bregma_row}, col={bregma_col}  "
          f"(downsampled y_1={cfg.y_1}, x_2={cfg.x_2})")

    if args.debug:
        limit = min(args.debug_frames, n_per_channel)
        if limit <= profile.trim:
            raise ValueError(
                f"debug_frames={limit} is at or below the profile's {profile.trim}-frame "
                f"trim; nothing left to correlate. Use a larger value."
            )
        print(f"Mode   : DEBUG (in-memory) -- first {limit} frames/channel")
        trials = [_load_limited_trial(gcamp_files, emo_files, limit)]
        result = run_profile(trials, cfg, profile)
        # RAM mode keeps dff_stack -> we can draw the placement overlay.
        overlay_path = os.path.join(args.output_dir, "roi_overlay_debug.png")
        _save_overlay(result.dff_stack, cfg, profile, overlay_path)
        print(f"Overlay: {overlay_path}  <- CHECK the boxes sit on the anatomy")
    else:
        print("Mode   : FULL (streaming, constant memory, two passes)")
        sources = [(folder_frame_source(gcamp_files), folder_frame_source(emo_files))]
        result = run_streaming_profile(sources, cfg, profile)

    labels = profile.labels
    print(f"\nR_mean ({profile.n_rois}x{profile.n_rois}), ROI order {labels}:")
    print(np.array2string(result.R_mean, precision=4, suppress_small=True))

    traces_path = os.path.join(args.output_dir, "roi_traces_debug.png" if args.debug
                               else "roi_traces_full.png")
    _save_traces(result.averaged_traces, labels, traces_path)
    print(f"Traces : {traces_path}")

    # Save the arrays. Streaming has no dff_stack, so only include it when present.
    npz_path = os.path.join(args.output_dir,
                            "connectivity_debug.npz" if args.debug else "connectivity_full.npz")
    arrays = dict(
        temp_roi=result.temp_roi,
        R=result.R,
        R_mean=result.R_mean,
        averaged_traces=result.averaged_traces,
        roi_labels=np.array(labels),
        profile=np.array(profile.name),
        bregma=np.array([bregma_row, bregma_col]),
        # Which geometry produced these numbers ("" = the profile's own atlas).
        roi_set=np.array(args.roi_set or ""),
    )
    if getattr(result, "dff_stack", None) is not None:
        arrays["dff_stack"] = result.dff_stack
    np.savez(npz_path, **arrays)
    print(f"Saved  : {npz_path}")


if __name__ == "__main__":
    main()
