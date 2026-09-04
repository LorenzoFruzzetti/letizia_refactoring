"""Resting-state functional connectivity on interleaved ("intermingle") folders.

This is a *study script* (P2 boundary: policy lives here, mechanism lives in
``wfci``). It runs the cerebellar resting-state pipeline on folders of interleaved
single-page TIFFs -- odd-positioned images are one channel, even the other -- and
writes the 4x4 connectivity matrix plus a couple of check figures.

``RUN_CONFIG["folder"]`` may be either a single recording folder or any parent of
several: the script walks down and treats EVERY folder that holds the TIFFs as its
own recording. Each one is analysed SEPARATELY -- its own channel split, trim,
baseline and correlation -- and gets its own output folder. Recordings of the same
animal are never concatenated (use ``run_botox_batch.py --merge-recordings`` if
that is what you want).

Target dataset (edit ``RUN_CONFIG`` for another) -- one acquisition day:
    \\146.48.88.209\share2\BOTOX_RESTANI\260611\<animal>\<t#>\*.tif
       day     ->      animal (R1, R2, T4) -> recording (t1..t5)
    6000 single-page 512x512 TIFFs per recording = 3000 frames/channel, ~3 GB.

Outputs mirror that tree under ``output_dir``, so nothing overwrites anything:
    outputs\intermingle_260611\R1\t1\{roi_overlay_debug.png, roi_traces_*.png,
                                     connectivity_*.npz}
(A single recording folder as input writes straight into ``output_dir``.)

Two run modes, selected by ``RUN_CONFIG["debug"]``:

* ``debug = True``  -- a fast smoke test on the first ``debug_max_frames`` frames
  PER CHANNEL, loaded into RAM (``run_profile``). RAM mode keeps ``dff_stack``, so
  it can also draw the step-2 ROI-placement overlay -- the visual check that the
  Bregma coordinates put the 4 boxes on the right anatomy. Use this to validate
  the setup before committing to the full recording.
* ``debug = False`` -- the real run. Streams each folder frame-by-frame in
  constant memory (``run_streaming_profile``, two passes), which is what a ~3 GB
  recording needs. No ``dff_stack`` is retained in this mode (streamed away), so
  no overlay -- confirm placement in the debug run first.

ROI geometry comes from the profile's atlas + ``bregma_row``/``bregma_col`` unless
``RUN_CONFIG["roi_set"]`` (or ``--roi-set``) points at a ROI-set YAML drawn with
``roi_editor.py``; that file carries both the boxes and the Bregma they were drawn
from, and then supplies both. One geometry is used for EVERY discovered recording,
which is right for the t# of one animal but not across animals -- for per-animal
Bregma, run one animal folder at a time (or use ``run_botox_batch.py``, whose
manifest carries a Bregma per animal).

Run it:
    conda run -n letizia python run_intermingle_rs.py                 # editor mode, uses RUN_CONFIG
    conda run -n letizia python run_intermingle_rs.py --full          # force the full streaming run
    conda run -n letizia python run_intermingle_rs.py --debug-frames 80
    conda run -n letizia python run_intermingle_rs.py --folder \\146.48.88.209\share2\BOTOX_RESTANI\260611\R1
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
from pathlib import Path
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
from roi_editor import folder_key, load_roi_set

# ---------------------------------------------------------------------------
# Edit this section to run without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Where the interleaved TIFFs live (odd/even = the two channels). Either ONE
    # recording folder, or a parent of several -- an animal (R1 -> t1..t5) or a
    # whole day (260611 -> R1/R2/T4 -> t1..t5). Every folder holding TIFFs below
    # this one is analysed on its own. UNC path as a raw string so the backslashes
    # are not read as escapes.
    "folder": r"\\146.48.88.209\share2\BOTOX_RESTANI\260611",
    # Glob used both to find the recording folders and to read them.
    "pattern": "*.tif",
    # Which pipeline: cerebellar resting state (4 ROIs, full-recording baseline
    # and correlation window). The intermingle modality is the cerebellar port.
    "profile": "cerebellar_rs",
    # Channel assignment for the interleaved split. "auto" picks the DIMMER
    # group (top-10% pixel intensity) as GCaMP -- on this rig the reflectance
    # (emo) channel is the brighter of the two.
    "channel_order": "auto",
    # Bregma at FULL resolution (floor(.../2) is applied internally to land on the
    # downsampled grid). VERIFY these against the ROI overlay the debug run writes
    # before trusting the full run's numbers -- ROI placement is the one thing this
    # script cannot check for you. NOTE these coordinates apply to EVERY recording
    # discovered below "folder": right for one animal's t1..tn, but across animals
    # run one animal folder at a time (Bregma moves between animals).
    "bregma_row": 121,
    "bregma_col": 134,
    # Optional ROI set written by roi_editor.py: a YAML holding the ROI boxes AND
    # the Bregma they were drawn from. When given it REPLACES the profile's atlas
    # and (if the file carries one) the bregma_row/bregma_col above -- the boxes
    # are offsets from that Bregma, so the two travel together. None = the
    # profile's own atlas with the coordinates above.
    #   e.g. r"roi_sets\260611_R1.yaml"      (this animal)
    #        r"roi_sets\shared_roi_set.yaml" (the same layout for every session)
    # Set to the 22-box cortical layout (wfci.atlases.CORTEX_22 -- the MATLAB's
    # img_av(y_1+.., x_2+..) boxes). Only the ATLAS is replaced: the profile above
    # stays cerebellar_rs, so trim/baseline/correlation and the no-mask, no-GSR
    # chain are unchanged -- just 22 ROIs instead of 4, giving a 22x22 R.
    # Re-draw it per animal with roi_editor.py and point this at that file.
    "roi_set": r"roi_sets\cortex22_roi_set.yaml",
    # Folder of per-recording ROI sets written by batch_roi_select.py.  When set,
    # each recording looks for <roi_set_dir>/<key>.yaml (where key = last 3 path
    # components, e.g. 260611_R1_t1) and uses that file if it exists, falling back
    # to the shared roi_set above.  None = use one roi_set for every recording.
    "roi_set_dir": None,
    # Debug smoke test vs the real streaming run.
    "debug": True,
    # Debug only: frames PER CHANNEL to load (reads the folder's first 2*N images,
    # since they alternate). Must exceed the profile's 20-frame trim; keep it small
    # for a quick check but large enough that a correlation means something.
    "debug_max_frames": 60,
    # Root for the results. Each recording gets its own subfolder mirroring its
    # path below "folder" (e.g. R1\t1), so recordings never overwrite each other.
    # A single recording folder as input writes straight into this directory.
    "output_dir": "outputs/intermingle_260611",
    "prefer_cli_args": True,
}


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Minimal CLI overrides for the few knobs worth flipping from the terminal."""
    p = argparse.ArgumentParser(description="Resting-state FC on interleaved folders.")
    p.add_argument("--folder", default=defaults["folder"],
                   help="Interleaved TIFF folder, or a parent of several (day/animal).")
    p.add_argument("--pattern", default=defaults["pattern"], help="TIFF glob.")
    p.add_argument("--bregma-row", type=int, default=defaults["bregma_row"])
    p.add_argument("--bregma-col", type=int, default=defaults["bregma_col"])
    p.add_argument("--output-dir", default=defaults["output_dir"])
    p.add_argument("--roi-set", default=defaults["roi_set"],
                   metavar="PATH", help="ROI set YAML from roi_editor.py (boxes + Bregma).")
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"],
                   metavar="DIR",
                   help="Folder of per-recording ROI sets from batch_roi_select.py; "
                        "each recording looks for <dir>/<key>.yaml before falling back "
                        "to --roi-set.")
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
        pattern=config["pattern"],
        profile=config["profile"],
        channel_order=config["channel_order"],
        bregma_row=config["bregma_row"],
        bregma_col=config["bregma_col"],
        roi_set=config["roi_set"],
        roi_set_dir=config["roi_set_dir"],
        debug=bool(config["debug"]),
        debug_frames=config["debug_max_frames"],
        output_dir=config["output_dir"],
    )


def resolve_roi_set_for(folder: Path, args) -> str | None:
    """Return the ROI set path for one recording folder, or None.

    Priority: <roi_set_dir>/<key>.yaml (per-recording) > args.roi_set (shared).
    Falls back to None when neither is configured or the per-recording file does
    not exist yet (run batch_roi_select.py first to create it).
    """
    if args.roi_set_dir:
        candidate = Path(args.roi_set_dir) / f"{folder_key(folder)}.yaml"
        if candidate.is_file():
            return str(candidate)
    return args.roi_set or None


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


def discover_recordings(root: str | Path, pattern: str) -> list[Path]:
    """Find every interleaved recording folder at or below ``root``, sorted.

    A folder counts as a recording when it directly holds at least 2 images
    matching ``pattern`` -- the minimum an interleaved split needs. That single
    rule handles all three input levels of this dataset without hard-coding the
    ``day\\animal\\t#`` depth: a recording folder returns itself, an animal folder
    returns its t1..tn, a day folder returns every animal's recordings. Folders
    holding images are never descended into, so a stray subfolder inside a
    recording cannot turn into a second one.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    if len(sorted(root.glob(pattern))) >= 2:
        return [root]
    found: list[Path] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        found.extend(discover_recordings(sub, pattern))
    return found


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


def run_one_recording(folder: Path, cfg, profile, args):
    """Analyse ONE interleaved folder end to end and return the pipeline result.

    Everything here is per-recording: the channel split, the profile's own trim,
    the baseline and the correlation window. Nothing is shared with the other
    recordings of the same animal -- that separation is the point (see the module
    docstring); ``run_botox_batch.py --merge-recordings`` is the concatenating one.
    """
    # Split the interleaved folder into two channel file lists. Decodes exactly two
    # images to decide which group is dimmer (=GCaMP); nothing large is read here.
    gcamp_files, emo_files = interleaved_channel_files(
        folder, pattern=args.pattern, channel_order=args.channel_order
    )
    n_per_channel = min(len(gcamp_files), len(emo_files))
    print(f"  Split  : {len(gcamp_files)} gcamp + {len(emo_files)} emo images "
          f"({n_per_channel} frames/channel), channel_order={args.channel_order}")

    if args.debug:
        limit = min(args.debug_frames, n_per_channel)
        if limit <= profile.trim:
            raise ValueError(
                f"debug_frames={limit} is at or below the profile's {profile.trim}-frame "
                f"trim; nothing left to correlate. Use a larger value."
            )
        print(f"  Mode   : DEBUG (in-memory) -- first {limit} frames/channel")
        trials = [_load_limited_trial(gcamp_files, emo_files, limit)]
        return run_profile(trials, cfg, profile)

    print("  Mode   : FULL (streaming, constant memory, two passes)")
    sources = [(folder_frame_source(gcamp_files), folder_frame_source(emo_files))]
    return run_streaming_profile(sources, cfg, profile)


def save_outputs(result, out_dir: str, cfg, profile, args, bregma_row, bregma_col,
                 *, roi_set_path: str | None = None) -> str:
    """Write this recording's figures and .npz into ``out_dir``; return the npz path."""
    os.makedirs(out_dir, exist_ok=True)
    labels = profile.labels

    # Only the RAM (debug) mode keeps dff_stack, so only it can draw the overlay.
    if getattr(result, "dff_stack", None) is not None:
        overlay_path = os.path.join(out_dir, "roi_overlay_debug.png")
        _save_overlay(result.dff_stack, cfg, profile, overlay_path)
        print(f"  Overlay: {overlay_path}  <- CHECK the boxes sit on the anatomy")

    traces_path = os.path.join(out_dir, "roi_traces_debug.png" if args.debug
                               else "roi_traces_full.png")
    _save_traces(result.averaged_traces, labels, traces_path)
    print(f"  Traces : {traces_path}")

    npz_path = os.path.join(out_dir,
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
        roi_set=np.array(roi_set_path or args.roi_set or ""),
    )
    if getattr(result, "dff_stack", None) is not None:
        arrays["dff_stack"] = result.dff_stack
    np.savez(npz_path, **arrays)
    print(f"  Saved  : {npz_path}")
    return npz_path


def _mean_offdiag(R_mean) -> float:
    """Mean of the off-diagonal correlations -- one number to compare recordings by."""
    n = R_mean.shape[0]
    off = ~np.eye(n, dtype=bool)
    return float(np.nanmean(R_mean[off]))


def main() -> None:
    args = build_runtime_args()

    base_profile = get_profile(args.profile)

    root = Path(args.folder)
    recordings = discover_recordings(root, args.pattern)
    if not recordings:
        raise FileNotFoundError(
            f"No folder holding at least 2 {args.pattern!r} images found at or below {root}."
        )

    print(f"Input  : {root}")
    print(f"Found  : {len(recordings)} recording(s), analysed separately")
    if args.roi_set_dir:
        print(f"ROI sets: {args.roi_set_dir}/<key>.yaml  (per-recording; "
              f"run batch_roi_select.py to create them)")

    summary: list[tuple[str, float, str]] = []
    for i, folder in enumerate(recordings, start=1):
        # Mirror the input tree under output_dir so recordings never collide. When
        # the input IS the recording folder its relative path has no parts, so it
        # writes straight into output_dir, as this script always did.
        rel = folder.relative_to(root)
        label = "/".join(rel.parts) or root.name
        out_dir = os.path.join(args.output_dir, *rel.parts)

        # Resolve per-recording ROI set (from roi_set_dir) or fall back to the
        # shared roi_set.  apply_roi_set with None is a no-op, so the profile's
        # own atlas and the config's Bregma are used when nothing is configured.
        roi_set_path = resolve_roi_set_for(folder, args)
        profile, bregma_row, bregma_col = apply_roi_set(
            base_profile, args.bregma_row, args.bregma_col, roi_set_path
        )
        cfg = ROIConfig.from_bregma(bregma_row, bregma_col, boxes=dict(profile.atlas))

        print(f"\n[{i}/{len(recordings)}] {label}  ({folder})")
        print(f"  Bregma : row={bregma_row}, col={bregma_col}  "
              f"(downsampled y_1={cfg.y_1}, x_2={cfg.x_2})")
        result = run_one_recording(folder, cfg, profile, args)

        print(f"  R_mean ({profile.n_rois}x{profile.n_rois}), ROI order {profile.labels}:")
        print(np.array2string(result.R_mean, precision=4, suppress_small=True))

        npz_path = save_outputs(result, out_dir, cfg, profile, args, bregma_row, bregma_col,
                                roi_set_path=roi_set_path)
        summary.append((label, _mean_offdiag(result.R_mean), npz_path))

    print(f"\nDone: {len(summary)} recording(s) -> {args.output_dir}")
    for label, mean_r, npz_path in summary:
        print(f"  {label:<12} mean off-diagonal R = {mean_r:+.4f}   {npz_path}")


if __name__ == "__main__":
    main()
