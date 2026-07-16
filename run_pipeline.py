"""Editor / CLI entrypoint for the wide-field connectivity pipeline.

Run directly from the editor (no CLI flags needed) by editing ``RUN_CONFIG``
below, or pass flags on the command line to override.

Two independent knobs describe a run (see ``wfci.io`` / ``wfci.streaming``):

``source`` -- how the two channels are stored on disk:
  * ``"stack_file"``: each trial is two multi-page TIFF files (gcamp, emo).
  * ``"frame_folder"``: each trial is two folders of single-page TIFFs, one
    GCaMP folder and one emo folder.
  * ``"interleaved_folder"``: each trial is a SINGLE folder of single-page TIFFs
    holding both channels acquired alternately (odd-positioned images are one
    channel, even-positioned the other). The loader splits them and assigns the
    brighter group (by top-10% pixel intensity of the first image in each group)
    to gcamp, the dimmer to emo. This is the layout of the sample ``data/``
    folder. Each ``trials`` entry is a single folder path, not a (gcamp, emo)
    pair.

``streaming`` -- how much is held in memory:
  * ``False``: load each trial's full ``[y, x, time]`` stacks into RAM (fast,
    keeps ``dff_stack`` so the step-2 overlay is available).
  * ``True``: read frame-by-frame in constant memory (two passes over each
    channel). Use this for large recordings that do not fit in RAM. No
    ``dff_stack`` is retained, so the step-2 overlay is unavailable in this mode.

These two axes are orthogonal: any ``source`` can be run streaming or not, so
e.g. an ``interleaved_folder`` recording too large for RAM streams by setting
``streaming=True`` -- the odd/even split is done up front from the file list and
each channel's frames are then streamed, without ever stacking the folder.

Example ``RUN_CONFIG`` for a real acquisition (edit paths, then run):
    trials = [("/path/animal/t1/gcamp.tif", "/path/animal/t1/emo.tif"), ...]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from typing import Any

import numpy as np

from wfci import (
    FrameSource,
    ROIConfig,
    folder_frame_source,
    frame_folder_source,
    interleaved_channel_files,
    load_frame_folder,
    load_interleaved_folder,
    load_stack,
    run_resting_state,
    run_streaming_resting_state,
    run_streaming_stimulated,
    run_stimulated,
    tiff_frame_source,
)

SOURCE_CHOICES = ["stack_file", "frame_folder", "interleaved_folder"]

# ---------------------------------------------------------------------------
# Edit this section to run the pipeline without passing CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Trial paths. For "stack_file"/"frame_folder" each entry is a (gcamp, emo)
    # pair (two files or two folders). For "interleaved_folder" each entry is
    # instead a SINGLE folder holding both channels interleaved (odd/even
    # images); the loader splits them and picks the brighter group as gcamp.
    "trials": [
        # stack_file/frame_folder: ("data/animal_t1_gcamp.tif", "data/animal_t1_emo.tif"),
        # UNC paths must be raw strings (r"...") or the backslashes are read as
        # escapes -- "\t1" would become a TAB character.
        r"\\server\share\ANIMAL\t1",
    ],
    "source": "interleaved_folder",  # "stack_file" | "frame_folder" | "interleaved_folder"
    "streaming": True,             # False = load into RAM; True = constant-memory stream
    "mode": "resting_state",        # "resting_state" | "stimulated"
    # Per-animal Bregma (full-resolution row, col); floor(.../2) is applied.
    "bregma_row": 121,
    "bregma_col": 134,
    "output_path": "outputs/connectivity.npz",  # | None to skip saving
    # Debug mode: keep only the first N frames PER CHANNEL of each trial, for a
    # fast smoke run on a subset of a recording. None = use every frame.
    # For "interleaved_folder" a limit of N reads the folder's first 2*N images
    # (they alternate between the two channels), so each channel gets N frames.
    # N must exceed the pipeline's 20-frame trim, and "stimulated" mode uses a
    # fixed 0:278 baseline / 279:300 correlation window, so N must be >= ~320
    # there for the result to mean anything.
    "debug_max_frames": None,
    "prefer_cli_args": True,
}


def _interleaved_folder(entry):
    """Extract the single folder path from an interleaved-source trial entry.

    Accepts a bare string or a 1-tuple/list (the CLI and config both occur).
    """
    return entry[0] if isinstance(entry, (tuple, list)) else entry


def _limited_source(src, limit):
    """Cap a :class:`FrameSource` at its first ``limit`` frames (debug mode).

    The wrapper keeps the source lazy and re-iterable: ``open()`` still returns a
    fresh iterator (the streaming path makes two passes), it just stops early, so
    only the frames actually consumed are ever decoded.
    """
    if limit is None or limit >= src.count:
        return src
    return FrameSource(limit, lambda: itertools.islice(src.open(), limit))


def _stack_source(src):
    """Materialise a FrameSource's frames into one ``[y, x, time]`` array."""
    return np.stack(list(src.open()), axis=-1)


def _load_trials(trial_paths, source, limit=None):
    """Load each trial fully into RAM as a ``(gcamp, emo)`` ``[y, x, time]`` pair."""
    if limit is not None:
        # Debug mode: go through the (lazy) frame sources so only the first
        # ``limit`` frames per channel are read off disk, then stack those.
        return [
            (_stack_source(gcamp_src), _stack_source(emo_src))
            for gcamp_src, emo_src in _build_streaming_sources(trial_paths, source, limit)
        ]
    trials = []
    for entry in trial_paths:
        if source == "interleaved_folder":
            # One folder holds both channels interleaved (odd/even); the loader
            # splits them and assigns the brighter group to gcamp.
            trials.append(load_interleaved_folder(_interleaved_folder(entry)))
        elif source == "frame_folder":
            gcamp_path, emo_path = entry
            trials.append((load_frame_folder(gcamp_path), load_frame_folder(emo_path)))
        else:  # stack_file
            gcamp_path, emo_path = entry
            trials.append((load_stack(gcamp_path), load_stack(emo_path)))
    return trials


def _build_streaming_sources(trial_paths, source, limit=None):
    """Build each trial's ``(gcamp_src, emo_src)`` FrameSource pair for streaming.

    Nothing large is loaded here: the sources decode frames lazily. For an
    interleaved folder the odd/even split + brighter-group decision is done up
    front from the file list (two small images read), then each channel's file
    list becomes a streaming source.

    ``limit`` (debug mode) caps each channel at its first ``limit`` frames.
    """
    sources = []
    for entry in trial_paths:
        if source == "interleaved_folder":
            gcamp_files, emo_files = interleaved_channel_files(_interleaved_folder(entry))
            pair = (folder_frame_source(gcamp_files), folder_frame_source(emo_files))
        elif source == "frame_folder":
            gcamp_path, emo_path = entry
            pair = (frame_folder_source(gcamp_path), frame_folder_source(emo_path))
        else:  # stack_file
            gcamp_path, emo_path = entry
            pair = (tiff_frame_source(gcamp_path), tiff_frame_source(emo_path))
        sources.append(tuple(_limited_source(src, limit) for src in pair))
    return sources


def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default=defaults["source"],
        help="How the two channels are stored on disk.",
    )
    p.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=defaults["streaming"],
        help="Constant-memory streaming (--streaming) vs load into RAM "
             "(--no-streaming).",
    )
    p.add_argument(
        "--mode",
        choices=["resting_state", "stimulated"],
        default=defaults["mode"],
    )
    p.add_argument("--bregma-row", type=int, default=defaults["bregma_row"])
    p.add_argument("--bregma-col", type=int, default=defaults["bregma_col"])
    p.add_argument("--output-path", default=defaults["output_path"])
    p.add_argument(
        "--debug-max-frames",
        type=int,
        default=defaults["debug_max_frames"],
        metavar="N",
        help="Debug mode: use only the first N frames per channel of each trial "
             "(interleaved_folder reads the folder's first 2*N images). Must be "
             "> the 20-frame trim. Omit for a full run.",
    )
    # Trials on the CLI: "gcamp,emo" pairs for stack_file/frame_folder, or a
    # single folder path for interleaved_folder. Repeat --trial for more trials.
    p.add_argument(
        "--trial",
        action="append",
        default=None,
        metavar="GCAMP,EMO | FOLDER",
        help="A trial as 'gcamp_path,emo_path' (stack_file/frame_folder) or a "
             "single folder path (interleaved_folder). Repeat for more trials.",
    )
    ns = p.parse_args()
    if ns.trial is not None:
        if ns.source == "interleaved_folder":
            # Each trial is one folder holding both interleaved channels.
            ns.trials = [t for t in ns.trial]
        else:
            ns.trials = [tuple(t.split(",", 1)) for t in ns.trial]
    else:
        ns.trials = defaults["trials"]
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    prefer_cli_args = bool(config.get("prefer_cli_args", True))
    if prefer_cli_args and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        trials=config["trials"],
        source=config["source"],
        streaming=bool(config["streaming"]),
        mode=config["mode"],
        bregma_row=config["bregma_row"],
        bregma_col=config["bregma_col"],
        output_path=config["output_path"],
        debug_max_frames=config["debug_max_frames"],
    )


def main() -> None:
    args = build_runtime_args()
    if not args.trials:
        print(
            "No trials configured. Edit RUN_CONFIG['trials'] in run_pipeline.py "
            "or pass --trial GCAMP,EMO. See tests/ for a runnable parity example."
        )
        return

    cfg = ROIConfig.from_bregma(args.bregma_row, args.bregma_col)

    limit = args.debug_max_frames
    if limit is not None and limit <= 20:
        # 20 frames are trimmed off the front of every trial by the pipeline, so
        # a smaller debug limit leaves nothing to correlate.
        raise ValueError(
            f"debug_max_frames={limit} is below the pipeline's 20-frame trim; "
            "use a larger value (>= 50 for a useful smoke run)."
        )

    if args.streaming:
        # Constant-memory path: trial *sources* stream straight to ROI traces,
        # nothing large is loaded. No dff_stack is produced (see RUN_CONFIG doc).
        sources = _build_streaming_sources(args.trials, args.source, limit)
        runner = (
            run_streaming_stimulated
            if args.mode == "stimulated"
            else run_streaming_resting_state
        )
        result = runner(sources, cfg)
        n_trials = len(sources)
    else:
        trials = _load_trials(args.trials, args.source, limit)
        runner = run_stimulated if args.mode == "stimulated" else run_resting_state
        result = runner(trials, cfg)
        n_trials = len(trials)

    mem = "stream" if args.streaming else "in-memory"
    debug = f"  |  DEBUG: first {limit} frames/channel" if limit is not None else ""
    print(
        f"Mode: {args.mode}  |  source: {args.source}  |  memory: {mem}  "
        f"|  trials: {n_trials}{debug}"
    )
    print("R_mean (4x4 functional connectivity):")
    print(np.array2string(result.R_mean, precision=4, suppress_small=True))

    if args.output_path:
        import os

        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        arrays = dict(
            temp_roi=result.temp_roi,
            R=result.R,
            R_mean=result.R_mean,
            averaged_traces=result.averaged_traces,
        )
        # The streaming path never materialises dff_stack; only save it if present.
        if getattr(result, "dff_stack", None) is not None:
            arrays["dff_stack"] = result.dff_stack
        np.savez(args.output_path, **arrays)
        print(f"Saved results -> {args.output_path}")


if __name__ == "__main__":
    main()
