"""Interactive ROI / Bregma editor for interleaved wide-field folders.

This is a *utility* study script (P2 boundary: policy and UI live here, the maths
lives in ``wfci``). It answers the one question the run scripts cannot answer for
you -- **do the ROI boxes sit on the right anatomy for THIS animal?** -- and lets
you fix it by dragging, then writes the result to a file the run scripts load.

What it shows
-------------
For every session (one interleaved folder), it decodes only the first
``preview_frames`` GCaMP images and puts them through the *same* two 0.5x box
downsamples the pipeline uses (512 -> 256 -> 128), so what you see is exactly the
grid the ROI offsets are resolved against. No correction, no DFF: a raw anatomy
image is a far better background for placing boxes than a noisy single DFF frame.

What it writes: a **ROI set** file
----------------------------------
One YAML per session (or one shared file for all of them)::

    name: 260611_R1
    grid: [128, 128]            # the FINAL frame these offsets were drawn for
    source: drawn with roi_editor.py ...
    bregma_row: 120             # per-animal Bregma, in RUN_CONFIG units (2 x grid)
    bregma_col: 134
    lambda_row_offset: 30       # Bregma -> Lambda, in FINAL-grid rows (= box units)
    boxes:                      # ORDER IS THE COLUMN ORDER OF R -- do not sort
      Laterale_L: {row_start: 21, row_end: 26, col_start: -34, col_end: -29}
      ...

This is a **superset of the library's atlas file format**: ``wfci.load_atlas``
reads it unchanged and simply ignores the three extra top-level keys. They are
there because a box is only meaningful together with the Bregma it was drawn from
-- keeping them in one file stops the two halves of the geometry drifting apart.
:func:`load_roi_set` returns the atlas and the Bregma; the Lambda distance is
read separately (:func:`load_lambda_offset`) because it is a record of the *scale
the boxes were already drawn at*, not something the pipeline has to apply.
``run_intermingle_rs.py`` and ``run_botox_batch.py`` take a ``roi_set`` path in
their ``RUN_CONFIG``.

The two landmarks
-----------------
**Bregma** (magenta +) is the origin: every box is stored as an offset from it, so
moving it translates the whole constellation rigidly.

**Lambda** (green x) sits on the midline, ``lambda_row_offset`` rows posterior to
Bregma, and is a *ruler* rather than an anchor. Its distance from Bregma is the
scale the layout was drawn at, so dragging it does not move one landmark -- it
rescales the whole atlas about Bregma. That is the per-animal knob for "this
brain sits bigger/smaller in the field of view than the one the atlas came from":
stretch the axis by 10% and every box moves 10% further out. Box *sizes* stay put
by default -- see :func:`scale_boxes` for why.

Controls (also printed at the bottom of the window)
---------------------------------------------------
Moving the WHOLE layout -- Bregma and every box together -- is the primary gesture,
because the boxes are stored as offsets *from* Bregma: move the anchor and all of
them translate rigidly, keeping their shape and sizes. Three ways to do it:

    drag the magenta +       drag Bregma + all boxes   double-click  put Bregma here
    ctrl+arrows              move Bregma + all boxes 1 px
    ctrl+shift+arrows        move Bregma + all boxes 5 px

SCALING the layout, with the Bregma -> Lambda distance:

    drag the green x          rescale about Bregma   , / .    distance -1 / +1 px
    < / >                     distance -5 / +5 px    l        back to scale 1.000x

Adjusting ONE box, once the layout as a whole sits right:

    drag inside a box        move it            drag a corner handle  resize it
    arrows / shift+arrows    nudge box 1 / 5    + / -                 grow / shrink

Everything else:

    n / p                    next / prev session
    s / S                    save this session / save every session
    w                        write the SHARED roi set (one file for all sessions)
    a / A                    apply this layout to all sessions (A also copies Bregma)
    r                        reset this session to how it started
    [ / ]                    display contrast    h    print this help    q  quit

Run it (``--no-capture-output`` matters: plain ``conda run`` holds every print until
the process exits, so an interactive script looks like it is doing nothing):
    conda run --no-capture-output -n letizia python roi_editor.py       # uses RUN_CONFIG
    conda run --no-capture-output -n letizia python roi_editor.py --folders "\\\\host\\share\\...\\R1\\t1"
    conda run --no-capture-output -n letizia python roi_editor.py --manifest manifests\\botox_restani_manifest.csv --scope recording
From VS Code: the "ROI editor (roi_editor.py)" launch config (external terminal).

This script NEVER runs the analysis -- it only draws geometry and writes a file. The
pipeline scripts are the ones that compute, and they do not stop to ask: they read
the file this one writes.

Built on **pyqtgraph** (Qt). Each box is a ``pg.RectROI`` and each landmark a
``pg.TargetItem``, so dragging, corner handles and whole-pixel snapping are Qt's job
rather than hand-rolled hit-testing. Right-drag / scroll zoom and pan the view;
right-click is pyqtgraph's own menu (view range, export image).

Needs a display (it opens a Qt window; it can appear BEHIND the terminal). Qt is
imported inside :meth:`ROIEditor._build_ui`, never at module level, so the run
scripts' ``from roi_editor import load_roi_set`` keeps working on a headless machine.

Errors are left to surface (no try/except); the one thing that is *checked* rather
than raised is a box that has been dragged off the frame -- it turns red and the save
is refused with a message, because in a GUI a refusal you can act on beats a
traceback.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wfci import (
    Box,
    ROIConfig,
    box_slices_for,
    folder_frame_source,
    get_profile,
    imresize_box,
    interleaved_channel_files,
    load_atlas,
)

# ---------------------------------------------------------------------------
# Edit this section to run without CLI flags.
# ---------------------------------------------------------------------------
RUN_CONFIG: dict[str, Any] = {
    # Where the sessions come from. Give a manifest (from scan_botox_dataset.py)
    # OR set it to None and list folders explicitly below.
    "manifest": None, #r"manifests\botox_restani_manifest.csv",
    # Manifest only: one editor page per animal ("animal" -- previews that
    # animal's FIRST recording, which is what the per-animal Bregma applies to)
    # or one page per t# recording ("recording" -- check every folder's first
    # image; more pages, same Bregma unless you change it per page).
    "scope": "animal",
    # Used when "manifest" is None: the interleaved folders to inspect.
    "folders": [
        r"data\data_Atea",
    ],
    # Pipeline whose ROI atlas is the starting layout (and whose downsampling
    # defines the preview grid). "cortical_gsr" carries the 22-box CORTEX_22
    # atlas (M2/M1/BFD/Tr/FL/HL/RS/V1a/V1, left then right) -- the layout the
    # MATLAB draws as img_av(y_1+.., x_2+..). Only the atlas and downsample are
    # read here: the editor never runs a pipeline, so this profile's mask/GSR
    # settings do not apply. "cerebellar_rs" gives the 4-box CEREBELLUM_4 instead.
    "profile": "cortical_gsr",
    # Interleaved split: "auto" picks the dimmer group as GCaMP. Only the GCaMP
    # channel is previewed.
    "channel_order": "auto",
    # How many GCaMP images to average into the preview. 1 = literally the first
    # image of the folder; a handful (5-20) is the same anatomy with less noise.
    "preview_frames": 1,
    # Fallback Bregma (RUN_CONFIG units, i.e. 2 x the final-grid pixel) used in
    # "folders" mode, or for a manifest row that carries none.
    "bregma_row": 121,
    "bregma_col": 134,
    # Bregma -> Lambda distance, in FINAL-GRID rows -- the same units the box
    # offsets use, NOT the doubled bregma_* units above. It is the layout's scale
    # reference: the atlas as shipped is declared to be drawn at this distance, and
    # dragging Lambda to D rescales every box by D / this. Only the starting value
    # lives here; a ROI set that carries lambda_row_offset overrides it.
    "lambda_offset": 55,
    # Whether rescaling also scales each box's SIZE, or only its position.
    # False (default) keeps every box the size it was drawn -- so the same number of
    # pixels is averaged for every animal, and per-ROI noise stays comparable across
    # a group. True is the true similarity transform: use it when the boxes are meant
    # to cover a fixed fraction of each animal's cortex rather than a fixed area.
    "lambda_scales_box_size": False,
    # Where ROI sets are read from and written to.
    "roi_set_dir": r"roi_sets",
    # Filename of the ONE shared set written by the 'w' key -- the "same ROIs for
    # every session" option.
    "shared_name": "shared_roi_set.yaml",
    # Start each session from <roi_set_dir>/<key>.yaml when that file exists, so
    # re-running the editor picks up where you left off.
    "load_existing": False,
    # Seed EVERY session from this one ROI set instead of the profile's atlas
    # (e.g. the shared file, to adjust it per animal). None = profile atlas.
    "start_from": None,
    "prefer_cli_args": True,
}

# The ROI-set file's box fields, in the order the library's loader expects.
BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")

# The Bregma -> Lambda distance the shipped atlases are taken to be drawn at, in
# final-grid rows. It is only a *reference*: what matters is the ratio between it
# and the distance measured on a given animal, which is the factor the layout is
# scaled by. See Session.rescale_to.
DEFAULT_LAMBDA_OFFSET = 30

# Smallest Bregma -> Lambda distance the editor will accept. Not just a guard
# against dividing by zero: the distance is a ruler, and at 5 px one pixel of drag
# would rescale the whole layout by 20%.
MIN_LAMBDA_OFFSET = 5

HELP = (
    "MOVE THE WHOLE LAYOUT: drag the magenta + (Bregma) -- every box comes "
    "with it, keeping its shape\n"
    "   also: double-click = put Bregma there    ctrl+arrows / ctrl+shift+arrows = "
    "move it 1 / 5 px\n"
    "SCALE THE LAYOUT: drag the green x (Lambda) -- its distance from Bregma is the "
    "scale\n"
    "   also: , / . = distance -1 / +1 px    < / > = -5 / +5 px    "
    "l = back to scale 1.000x\n"
    "ONE BOX: left-drag it to move, left-drag a corner to resize, "
    "arrows / shift+arrows to nudge 1 / 5 px\n"
    "+ / -: grow / shrink box    "
    "[ / ]: contrast    r: reset page\n"
    "n / p: next / prev session    s: save this session    S: save all    "
    "w: write SHARED set\n"
    "a: apply boxes + scale to all sessions    A: also copy Bregma    "
    "h: help    q: quit"
)


# ---------------------------------------------------------------------------
# ROI-set files (the atlas + the Bregma it was drawn from)
# ---------------------------------------------------------------------------
def save_roi_set(
    path: str | Path,
    boxes: dict[str, Box],
    bregma_row: int,
    bregma_col: int,
    grid: tuple[int, int],
    name: str,
    source: str,
    lambda_offset: int | None = None,
    downsample: float = 0.5,
) -> Path:
    """Write a ROI set (boxes + Bregma + Lambda + grid) to YAML (or JSON, by extension).

    The layout is the library's atlas file plus ``bregma_row`` / ``bregma_col`` /
    ``downsample`` / ``lambda_row_offset``, so ``wfci.load_atlas`` reads the same
    file and ignores those metadata keys. ``sort_keys=False`` is not cosmetic: box order IS the column
    order of ``R``, so alphabetising the file would silently relabel every matrix
    built from it.

    ``lambda_offset`` is written as *documentation of the scale the boxes are
    already at*, not as an instruction: the offsets in this file are the final,
    scaled ones, so a pipeline that ignores the key still gets the right geometry.
    It is what lets the editor reopen the file and keep scaling from where it left off.
    """
    payload = {
        "name": name,
        "grid": [int(grid[0]), int(grid[1])],
        "source": source,
        "bregma_row": int(bregma_row),
        "bregma_col": int(bregma_col),
        "downsample": float(downsample),
        **({} if lambda_offset is None else {"lambda_row_offset": int(lambda_offset)}),
        "boxes": {
            label: {f: int(getattr(box, f)) for f in BOX_FIELDS}
            for label, box in boxes.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        text = json.dumps(payload, indent=2)
    else:
        # default_flow_style=None keeps each box on one hand-editable line.
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=None)
    path.write_text(text, encoding="utf-8")
    return path


def load_roi_set(path: str | Path) -> tuple[Any, int | None, int | None]:
    """Load a ROI set: ``(atlas, bregma_row, bregma_col)``.

    The boxes go through ``wfci.load_atlas`` so the file gets the library's own
    validation (missing/typo'd/non-integer fields, the grid check). The Bregma is
    read separately because it is *not* part of an atlas -- it is per animal.
    Both Bregma keys are optional, but they come as a pair: a file with only one
    of them is a half-edited file, and half a coordinate is not usable.
    """
    path = Path(path)
    atlas = load_atlas(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)

    bregma_row = data.get("bregma_row")
    bregma_col = data.get("bregma_col")
    if (bregma_row is None) != (bregma_col is None):
        raise ValueError(
            f"{path}: has only one of bregma_row / bregma_col. Give both (the "
            f"Bregma the boxes were drawn from) or neither (the run script's own "
            f"Bregma is then used)."
        )
    if bregma_row is None:
        return atlas, None, None
    return atlas, int(bregma_row), int(bregma_col)


def load_lambda_offset(path: str | Path) -> int | None:
    """The ``lambda_row_offset`` a ROI set was drawn at, or ``None`` if it has none.

    Deliberately *not* part of :func:`load_roi_set`'s tuple. The boxes in the file
    are already scaled to this distance, so the pipeline needs nothing from it --
    making the run scripts unpack and carry a value they must not act on would be
    an invitation to apply the scale twice. Only the editor reads it, to resume
    scaling from the same reference.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    value = data.get("lambda_row_offset")
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# Preview image: the first frame(s) on the pipeline's final grid
# ---------------------------------------------------------------------------
def preview_image(
    folder: str | Path,
    channel_order: str = "auto",
    n_frames: int = 1,
    downsample: float = 0.5,
) -> np.ndarray:
    """Mean of the first ``n_frames`` GCaMP images, on the FINAL analysis grid.

    Two 0.5x box downsamples (512 -> 256 -> 128) -- the same pair the pipeline
    applies around the hemodynamic correction -- so an ROI offset means the same
    number of pixels here as it does in the analysis. Only ``n_frames`` images
    (plus the two the channel split reads) are decoded off the network share.
    """
    gcamp_files, _emo_files = interleaved_channel_files(folder, channel_order=channel_order)
    source = folder_frame_source(gcamp_files)
    frames = list(itertools.islice(source.open(), n_frames))
    raw = np.mean(frames, axis=0)                      # [y, x]
    return imresize_box(imresize_box(raw, downsample), downsample)


# ---------------------------------------------------------------------------
# Geometry helpers: MATLAB-style offsets <-> 0-based pixel bounds
# ---------------------------------------------------------------------------
# A box is stored as inclusive 1-based offsets from Bregma, i.e. MATLAB
# ``img(y_1+row_start : y_1+row_end, ...)``. NumPy is 0-based, so the first pixel
# of that range is index ``y_1 + row_start - 1`` -- the same "-1" wfci.roi applies.
# Bregma itself (offset 0) therefore sits at 0-based pixel (y_1 - 1, x_2 - 1).
def pixel_bounds(box: Box, y_1: int, x_2: int) -> tuple[int, int, int, int]:
    """``(r0, r1, c0, c1)`` 0-based pixel bounds, both ends INCLUSIVE."""
    return (
        y_1 + box.row_start - 1,
        y_1 + box.row_end - 1,
        x_2 + box.col_start - 1,
        x_2 + box.col_end - 1,
    )


def box_from_pixels(r0: int, r1: int, c0: int, c1: int, y_1: int, x_2: int) -> Box:
    """Inverse of :func:`pixel_bounds` -- pixel bounds back to Bregma offsets."""
    return Box(
        row_start=r0 - y_1 + 1,
        row_end=r1 - y_1 + 1,
        col_start=c0 - x_2 + 1,
        col_end=c1 - x_2 + 1,
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def scale_boxes(boxes: dict[str, Box], factor: float,
                scale_size: bool = False) -> dict[str, Box]:
    """Scale every box's offsets about Bregma by ``factor``.

    This is the Bregma -> Lambda knob's only effect on geometry. Because the offsets
    are measured *from* Bregma, multiplying them scales the constellation about
    Bregma without moving it -- the same reason moving Bregma translates the layout.

    ``scale_size=False`` (the default) scales each box's **centre** and keeps its
    width and height. A box is an averaging window, and holding its area fixed keeps
    the number of pixels behind every ROI mean -- and therefore its noise level --
    identical across animals, so a group comparison is not confounded by how big
    each brain happened to sit in the field of view. ``scale_size=True`` is the true
    similarity transform, for when the boxes are meant to cover a fixed *fraction*
    of cortex instead.

    Rounding is Python's round-half-to-even, which is symmetric about zero: a
    mirrored left/right pair scales to a still-mirrored pair, so the atlas cannot
    quietly lose its symmetry at a half-pixel. ``factor == 1.0`` is exactly the
    identity (every centre is an integer or a half-integer, both exact in binary),
    so a no-op Lambda change moves nothing.
    """
    scaled: dict[str, Box] = {}
    for label, box in boxes.items():
        if scale_size:
            r0, r1 = int(round(box.row_start * factor)), int(round(box.row_end * factor))
            c0, c1 = int(round(box.col_start * factor)), int(round(box.col_end * factor))
            # Shrinking must not invert a box: Box requires end >= start.
            scaled[label] = Box(r0, max(r0, r1), c0, max(c0, c1))
            continue
        span_r = box.row_end - box.row_start
        span_c = box.col_end - box.col_start
        r0 = int(round((box.row_start + box.row_end) / 2 * factor - span_r / 2))
        c0 = int(round((box.col_start + box.col_end) / 2 * factor - span_c / 2))
        scaled[label] = Box(r0, r0 + span_r, c0, c0 + span_c)
    return scaled


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """One editor page: a folder to preview and the geometry being edited."""

    key: str                      # filename stem of its ROI set, e.g. 260611_R1
    folder: str                   # the interleaved folder previewed
    y_1: int                      # Bregma row on the FINAL grid (MATLAB 1-based)
    x_2: int                      # Bregma col on the FINAL grid
    boxes: dict[str, Box]
    # Bregma -> Lambda, in final-grid rows (the box offsets' own units). Lambda is
    # on the midline, so it needs no column of its own: it is always at x_2.
    lambda_offset: int = DEFAULT_LAMBDA_OFFSET
    # The layout the current scale is measured FROM: `base_boxes` as drawn at
    # `base_lambda`. Scaling always derives from this, never from the boxes it last
    # produced, so 30 -> 40 -> 30 lands back on the exact same integers instead of
    # drifting by a rounding step in each direction. Any direct edit to a box
    # re-anchors it here (see `rebase`).
    base_lambda: int = 0
    base_boxes: dict[str, Box] | None = None
    seed: tuple[int, int, int, dict[str, Box]] | None = field(default=None)  # 'r' = reset
    image: np.ndarray | None = None   # decoded lazily, when the page is opened
    saved_to: str = ""                # last file written for this session
    dirty: bool = False

    def __post_init__(self) -> None:
        if self.base_boxes is None:
            self.rebase()
        if self.seed is None:
            self.seed = (self.y_1, self.x_2, self.lambda_offset, dict(self.boxes))

    @property
    def bregma_row(self) -> int:
        """Bregma in RUN_CONFIG units: the scripts apply ``// 2`` to reach y_1."""
        return 2 * self.y_1

    @property
    def bregma_col(self) -> int:
        return 2 * self.x_2

    @property
    def lambda_row(self) -> int:
        """Lambda's row on the final grid, in the same 1-based frame as ``y_1``."""
        return self.y_1 + self.lambda_offset

    @property
    def lambda_scale(self) -> float:
        """How much the layout is currently stretched relative to its reference."""
        return self.lambda_offset / self.base_lambda

    def rebase(self) -> None:
        """Declare the current boxes to be the reference layout at the current Lambda.

        Called after any edit that changes a box directly (drag, resize, nudge):
        those pixels are what the user *wants* at this distance, so they become the
        thing later rescaling is derived from. Without this, a hand-adjusted box
        would spring back to a scaled version of its old self on the next Lambda move.
        """
        self.base_boxes = dict(self.boxes)
        self.base_lambda = self.lambda_offset

    def rescale_to(self, lambda_offset: int, scale_size: bool = False) -> None:
        """Set the Bregma -> Lambda distance and rescale the layout to match."""
        self.lambda_offset = int(lambda_offset)
        self.boxes = scale_boxes(self.base_boxes, self.lambda_offset / self.base_lambda,
                                 scale_size)


def _seed_layout(key: str, args: argparse.Namespace, default_boxes: dict[str, Box],
                 bregma_row: int, bregma_col: int) -> tuple[int, int, int, dict[str, Box]]:
    """Starting geometry for one session: existing file > --start-from > profile.

    Returns ``(y_1, x_2, lambda_offset, boxes)``. A ROI set that carries a Bregma
    overrides the manifest's -- it is the Bregma those boxes were drawn from, and
    separating them would silently move every box. Same for its Lambda distance:
    the boxes in the file are already at that scale, so adopting the boxes without
    it would leave the editor rescaling from the wrong reference.
    """
    candidates = []
    if args.load_existing:
        candidates.append(Path(args.roi_set_dir) / f"{key}.yaml")
    if args.start_from:
        candidates.append(Path(args.start_from))

    for path in candidates:
        if path.is_file():
            atlas, file_row, file_col = load_roi_set(path)
            file_lambda = load_lambda_offset(path)
            row = bregma_row if file_row is None else file_row
            col = bregma_col if file_col is None else file_col
            lam = args.lambda_offset if file_lambda is None else file_lambda
            print(f"  {key}: seeded from {path}")
            return row // 2, col // 2, lam, dict(atlas)

    return bregma_row // 2, bregma_col // 2, args.lambda_offset, dict(default_boxes)


def sessions_from_manifest(args: argparse.Namespace,
                           default_boxes: dict[str, Box]) -> list[Session]:
    """One session per manifest row ("animal") or per t# recording ("recording")."""
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    sessions: list[Session] = []
    for row in rows:
        day, animal = row["day"], row["animal"]
        paths = [p for p in row["recording_paths"].split(";") if p]
        names = [n for n in row["recordings"].split(";") if n]
        # The manifest carries a per-animal Bregma; fall back to the config's.
        bregma_row = int(row["bregma_row"]) if row.get("bregma_row") else args.bregma_row
        bregma_col = int(row["bregma_col"]) if row.get("bregma_col") else args.bregma_col

        # The key is also the ROI-set filename stem, and run_botox_batch.py looks
        # units up as "<day>_<animal>.yaml" -- keep the two spellings identical.
        if args.scope == "animal":
            entries = [(_safe(f"{day}_{animal}"), paths[0])]
        elif args.scope == "recording":
            entries = [(_safe(f"{day}_{animal}_{names[i]}"), p) for i, p in enumerate(paths)]
        else:
            raise ValueError(f"scope={args.scope!r} is not 'animal' or 'recording'.")

        for key, folder in entries:
            y_1, x_2, lam, boxes = _seed_layout(key, args, default_boxes,
                                                bregma_row, bregma_col)
            sessions.append(Session(key=key, folder=folder, y_1=y_1, x_2=x_2,
                                    boxes=boxes, lambda_offset=lam))
    return sessions


def _safe(name: str) -> str:
    """Filename-safe form of a key component (keys become file paths)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def folder_key(folder: str | Path) -> str:
    """A filename-safe session key from a folder path: its last 3 components.

    ``...\\260611\\R1\\t1`` -> ``260611_R1_t1``. The drive/UNC anchor is dropped and
    anything outside ``[A-Za-z0-9._-]`` is replaced, because this key is joined to
    ``roi_set_dir`` to form a path -- an unsanitised ``h:\\letizia\\data`` would make
    ``Path(dir) / key`` absolute again and write the file somewhere else entirely.
    """
    path = Path(folder)
    parts = path.parts[1:] if path.anchor else path.parts
    return "_".join(_safe(p) for p in parts[-3:])


def sessions_from_folders(args: argparse.Namespace,
                          default_boxes: dict[str, Box]) -> list[Session]:
    """One session per explicitly listed folder; key = last 3 path components."""
    sessions: list[Session] = []
    for folder in args.folders:
        key = folder_key(folder)
        y_1, x_2, lam, boxes = _seed_layout(key, args, default_boxes,
                                            args.bregma_row, args.bregma_col)
        sessions.append(Session(key=key, folder=str(folder), y_1=y_1, x_2=x_2,
                                boxes=boxes, lambda_offset=lam))
    return sessions


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------
# Qt and pyqtgraph are imported INSIDE ROIEditor._build_ui, not here. The run
# scripts do `from roi_editor import load_roi_set`, and that import must stay
# free of any GUI stack: it has to work on a headless batch machine, and a
# 15 GB overnight run must not fail because Qt could not find a display.
#
# Geometry, once, because it is the only thing here that can be silently wrong:
#   * A box is inclusive 0-based pixels r0..r1, c0..c1 (see pixel_bounds).
#   * pyqtgraph's ImageItem draws pixel i over the data interval [i, i+1), so an
#     inclusive r0..r1 box is a rect at y=r0 with height r1-r0+1. (Matplotlib's
#     imshow centred pixels instead, [i-0.5, i+0.5] -- hence no more 0.5 offsets.)
#   * Bregma (offset 0) is pixel (y_1-1, x_2-1); its marker goes at that pixel's
#     CENTRE, (x_2-0.5, y_1-0.5), so the crosshair sits on the pixel, not its edge.
#   * Lambda is offset `lambda_offset`, i.e. pixel (y_1+lambda_offset-1, x_2-1), so
#     its marker is exactly `lambda_offset` pixels below Bregma's. Sharing the
#     offset convention with the boxes is what makes the distance directly readable.
#   * The view is y-inverted so row 0 is at the top, matching every other image in
#     this project. Without that the whole atlas would appear mirrored and every
#     box would be dragged to the wrong place.
class ROIEditor:
    """pyqtgraph ROI editor over a list of :class:`Session`.

    Each box is a :class:`pyqtgraph.RectROI` -- Qt does the dragging, resizing and
    corner handles, and ``translateSnap``/``scaleSnap`` keep every edit on whole
    pixels, so a box can never acquire a fractional offset. The two landmarks are
    :class:`pyqtgraph.TargetItem` s, draggable by construction.

    What is not delegated is the *coupling*, and there are two of them, both
    following from boxes being stored as offsets from Bregma:

    * moving the **Bregma** target repositions every box (:meth:`_sync_boxes`)
      without touching ``session.boxes`` -- that is what keeps the layout rigid;
    * moving the **Lambda** target changes the distance the offsets are measured in,
      so it rescales them about Bregma (:meth:`_set_lambda`). This one *does* rewrite
      ``session.boxes``, always from ``session.base_boxes``, never from the last
      scaled result.
    """

    NUDGE_BIG = 5        # shift+arrow step, in pixels
    HANDLE_SIZE = 7      # corner-handle radius in SCREEN px (zoom-independent)

    def __init__(self, sessions: list[Session], args: argparse.Namespace,
                 profile_name: str) -> None:
        self.sessions = sessions
        self.args = args
        self.profile_name = profile_name
        self.index = 0
        self.selected: str | None = None
        self.clip = [1.0, 99.5]      # display percentiles
        self.message = ""
        # True while WE are moving items programmatically. Qt emits the same
        # "region changed" signal for a user drag and for a setPos() call, so
        # without this guard a Bregma move would recurse: sync boxes -> each box
        # reports a change -> read it back as a user edit -> sync again.
        self._syncing = False
        self._rois: dict[str, Any] = {}      # label -> RectROI
        self._labels: dict[str, Any] = {}    # label -> TextItem
        self._build_ui()

    # -- Qt construction -----------------------------------------------------
    def _build_ui(self) -> None:
        import pyqtgraph as pg
        # Import Qt through pyqtgraph's shim, never as PySide6/PyQt5 directly, so
        # the editor runs on whichever binding a machine happens to have.
        from pyqtgraph.Qt import QtCore, QtWidgets

        # row-major: index images as [row, col] like NumPy and like the rest of
        # this codebase. pyqtgraph's default is the transpose, which would silently
        # swap the row/col meaning of every ROI offset.
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

        self.pg = pg
        self.QtCore = QtCore
        self.app = pg.mkQApp("ROI editor")

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("ROI / Bregma editor")
        self.window.resize(900, 1000)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.header)

        self.graphics = pg.GraphicsLayoutWidget()
        # NoFocus: let arrow keys reach the window's keyPressEvent instead of being
        # eaten by the view (where they would pan it).
        self.graphics.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.graphics, stretch=1)

        self.plot = self.graphics.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)                  # row 0 at the top, like imshow
        self.plot.showAxes(True, showValues=True)
        self.plot.setLabel("left", "row (pixels, final grid)")
        self.plot.setLabel("bottom", "col (pixels, final grid)")
        # The image is the backdrop: never let a stray drag pick it up.
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)

        self.msg_label = QtWidgets.QLabel()
        self.msg_label.setTextFormat(QtCore.Qt.PlainText)
        self.msg_label.setWordWrap(True)
        self.msg_label.setStyleSheet("color: #1f77b4; font-family: monospace;")
        layout.addWidget(self.msg_label)

        help_label = QtWidgets.QLabel(HELP)
        help_label.setTextFormat(QtCore.Qt.PlainText)
        help_label.setStyleSheet("color: #555; font-family: monospace; font-size: 11px;")
        layout.addWidget(help_label)

        self.window.setCentralWidget(central)

        # Bregma: a crosshair that is draggable out of the box. Drawn last / on top.
        self.target = pg.TargetItem(
            pos=(0, 0), size=18, movable=True, pen=pg.mkPen("m", width=2),
            hoverPen=pg.mkPen("y", width=3), label=None,
        )
        self.target.setZValue(100)
        self.plot.addItem(self.target)
        self.target.sigPositionChanged.connect(self._on_target_moved)

        # Lambda: the second landmark. A different symbol and colour on purpose --
        # it looks like Bregma's twin but behaves completely differently (dragging it
        # RESCALES the layout instead of moving it), so it must not be mistaken for
        # a second anchor.
        self.lambda_target = pg.TargetItem(
            pos=(0, 0), size=16, movable=True, symbol="x",
            pen=pg.mkPen("g", width=2), hoverPen=pg.mkPen("y", width=3), label=None,
        )
        self.lambda_target.setZValue(100)
        self.plot.addItem(self.lambda_target)
        self.lambda_target.sigPositionChanged.connect(self._on_lambda_moved)

        # The midline segment between the landmarks: draws the distance the whole
        # layout is scaled by, so it can be lined up against the actual skull.
        self.axis_line = pg.PlotDataItem(
            pen=pg.mkPen("g", width=1, style=QtCore.Qt.DashLine))
        self.axis_line.setZValue(99)
        self.plot.addItem(self.axis_line)

        # Double-click puts Bregma somewhere directly, for when it is far from where
        # it belongs and dragging is tedious. The matplotlib version used RIGHT-click
        # for this; in pyqtgraph right-click is the view's own menu (zoom, export),
        # which is worth keeping, so the gesture moved to double-click.
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

        # Keys go to the window, so one handler serves the whole editor.
        self.window.keyPressEvent = self._on_key          # type: ignore[method-assign]
        self.window.setFocusPolicy(QtCore.Qt.StrongFocus)

    # -- state ---------------------------------------------------------------
    @property
    def session(self) -> Session:
        return self.sessions[self.index]

    def _ensure_image(self, session: Session) -> None:
        """Decode the preview for a page the first time it is opened."""
        if session.image is None:
            # flush: over a network share this is the slow step, and an unflushed
            # line makes the editor look like it is doing nothing.
            print(f"Loading preview: {session.folder}", flush=True)
            session.image = preview_image(session.folder, self.args.channel_order,
                                          self.args.preview_frames)

    def _out_of_frame(self, session: Session) -> list[str]:
        """Labels whose box does not fit the frame (drawn red, save refused).

        Same condition ``wfci.roi.box_slices_for`` raises on -- checked here so the
        GUI can say which box and let you drag it back, instead of dying.
        """
        rows, cols = session.image.shape
        bad = []
        for label, box in session.boxes.items():
            r0, r1, c0, c1 = pixel_bounds(box, session.y_1, session.x_2)
            if r0 < 0 or c0 < 0 or r1 >= rows or c1 >= cols:
                bad.append(label)
        return bad

    def say(self, text: str) -> None:
        """Put a line in the window's status area and on stdout."""
        self.message = text
        print(text)
        self._refresh_status()

    # -- building / syncing the scene ---------------------------------------
    def _make_roi(self, label: str, box: Box):
        """One RectROI for one box, snapped to whole pixels."""
        pg = self.pg
        session = self.session
        r0, r1, c0, c1 = pixel_bounds(box, session.y_1, session.x_2)
        roi = pg.RectROI(
            pos=(c0, r0), size=(c1 - c0 + 1, r1 - r0 + 1),
            pen=pg.mkPen("c", width=2), hoverPen=pg.mkPen("y", width=2),
            # Whole-pixel edits only: an ROI offset is an integer by definition, and
            # a fractional drag would be silently floored on save.
            translateSnap=True, scaleSnap=True, snapSize=1.0,
            # No rotation: the boxes are axis-aligned slices, and wfci has no way to
            # express a rotated one. Not offering it beats saving a rotation that
            # gets dropped.
            rotatable=False, resizable=True, removable=False,
        )
        # RectROI ships one bottom-right scale handle; add the other three corners
        # so the box can be resized from whichever side is wrong.
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        for handle in roi.getHandles():
            handle.radius = self.HANDLE_SIZE
            handle.buildPath()
        roi.sigRegionChanged.connect(lambda _roi, lbl=label: self._on_roi_changed(lbl))
        roi.sigClicked.connect(lambda _roi, _ev, lbl=label: self._select(lbl))
        return roi

    def _rebuild_scene(self) -> None:
        """Recreate every item for the current session (page switch, reset, apply)."""
        session = self.session
        self._ensure_image(session)

        self._syncing = True
        try:
            for item in list(self._rois.values()) + list(self._labels.values()):
                self.plot.removeItem(item)
            self._rois.clear()
            self._labels.clear()

            self.image_item.setImage(session.image, autoLevels=False)
            self._apply_levels()
            rows, cols = session.image.shape
            self.plot.setRange(xRange=(0, cols), yRange=(0, rows), padding=0.02)

            for label, box in session.boxes.items():
                roi = self._make_roi(label, box)
                self.plot.addItem(roi)
                self._rois[label] = roi

                text = self.pg.TextItem(label, color="c", anchor=(0, 1))
                self.plot.addItem(text)
                self._labels[label] = text

            self._sync_landmarks()
        finally:
            self._syncing = False
        self._restyle()

    def _sync_landmarks(self) -> None:
        """Put both markers and the axis line where the session says they are.

        Only called from inside a ``_syncing`` block: ``setPos`` emits the same
        signal a drag does, so an unguarded call here would be read back as the user
        having moved the landmark.
        """
        session = self.session
        bregma_y = session.y_1 - 0.5              # pixel y_1-1, at its centre
        lambda_y = session.lambda_row - 0.5       # exactly lambda_offset px below
        self.target.setPos(session.x_2 - 0.5, bregma_y)
        self.lambda_target.setPos(session.x_2 - 0.5, lambda_y)
        self.axis_line.setData([session.x_2 - 0.5] * 2, [bregma_y, lambda_y])

    def _sync_boxes(self) -> None:
        """Push ``session.boxes`` + both landmarks back onto the existing items.

        Used after anything that changes geometry without a user drag: a Bregma
        move, a rescale, a keyboard nudge, a resize. ``session.boxes`` is the single
        source of truth; the items only ever mirror it.
        """
        session = self.session
        self._syncing = True
        try:
            for label, box in session.boxes.items():
                r0, r1, c0, c1 = pixel_bounds(box, session.y_1, session.x_2)
                roi = self._rois[label]
                roi.setPos(c0, r0, finish=False)
                roi.setSize((c1 - c0 + 1, r1 - r0 + 1), finish=False)
            self._sync_landmarks()
        finally:
            self._syncing = False
        self._restyle()

    def _apply_levels(self) -> None:
        session = self.session
        lo, hi = np.percentile(session.image, self.clip)
        self.image_item.setLevels((lo, hi))

    def _restyle(self) -> None:
        """Colour the boxes (red = off frame, yellow = selected) and place labels."""
        session = self.session
        bad = set(self._out_of_frame(session))
        for label, roi in self._rois.items():
            colour = ("r" if label in bad else
                      "y" if label == self.selected else "c")
            width = 3 if label == self.selected else 2
            roi.setPen(self.pg.mkPen(colour, width=width))
            r0, _r1, c0, _c1 = pixel_bounds(session.boxes[label], session.y_1, session.x_2)
            text = self._labels[label]
            text.setColor(colour)
            text.setPos(c0, r0)          # anchor=(0,1) puts it just above the box
        self._refresh_header()
        self._refresh_status(bad)

    def _refresh_header(self) -> None:
        session = self.session
        rows, cols = session.image.shape
        self.header.setText(
            f"[{self.index + 1}/{len(self.sessions)}] {session.key}"
            f"{'  *unsaved*' if session.dirty else ''}\n"
            f"{session.folder}\n"
            f"{rows}x{cols} grid | Bregma row={session.bregma_row} "
            f"col={session.bregma_col} (y_1={session.y_1}, x_2={session.x_2}) | "
            f"Lambda +{session.lambda_offset} px = scale {session.lambda_scale:.3f}x "
            f"(ref {session.base_lambda}) | "
            f"profile {self.profile_name} | {len(session.boxes)} ROIs"
        )
        self.window.setWindowTitle(f"ROI editor -- {session.key}")

    def _refresh_status(self, bad: set | None = None) -> None:
        if bad is None:
            bad = set(self._out_of_frame(self.session)) if self.session.image is not None else set()
        # Spell the off-frame condition out rather than only recolouring the line:
        # dragging the whole layout runs into the frame edge often, and "some boxes
        # went red" is easy to miss until a save is refused.
        note = (f"   |   {len(bad)} box(es) OFF FRAME: {sorted(bad)} -- saving is "
                f"refused until they fit" if bad else "")
        self.msg_label.setText(self.message + note)
        self.msg_label.setStyleSheet(
            f"color: {'#d62728' if bad else '#1f77b4'}; font-family: monospace;")

    def _select(self, label: str | None) -> None:
        self.selected = label
        self._restyle()

    # -- signals from the user ----------------------------------------------
    def _on_roi_changed(self, label: str) -> None:
        """A box was dragged or resized: read the item back into the session."""
        if self._syncing:
            return
        session = self.session
        roi = self._rois[label]
        pos, size = roi.pos(), roi.size()
        # Snapping keeps these integral; round anyway so a fractional value can
        # never reach a Box (int() would truncate -0.5 to 0 and shift the ROI).
        c0, r0 = int(round(pos.x())), int(round(pos.y()))
        width, height = int(round(size.x())), int(round(size.y()))
        # A zero-width rect is reachable by dragging a handle across the box; clamp
        # to 1 px rather than let row_end < row_start into a Box.
        width, height = max(1, width), max(1, height)
        session.boxes[label] = box_from_pixels(r0, r0 + height - 1, c0, c0 + width - 1,
                                              session.y_1, session.x_2)
        # A hand-placed box is the new truth at this Lambda distance, so it becomes
        # part of the reference the next rescale derives from -- otherwise the very
        # next Lambda nudge would throw the adjustment away.
        session.rebase()
        session.dirty = True
        self.selected = label
        self._restyle()

    def _on_scene_clicked(self, event) -> None:
        """Double-click on the image: put Bregma there, boxes follow."""
        if not event.double():
            return
        view_pos = self.plot.vb.mapSceneToView(event.scenePos())
        # A click at data y lands in pixel floor(y), and pixel i is offset-0 index
        # i+1 -- the inverse of the "y_1 - 0.5" the marker is drawn at.
        self._move_bregma(math.floor(view_pos.y()) + 1, math.floor(view_pos.x()) + 1)
        session = self.session
        self.say(f"Bregma -> row={session.bregma_row}, col={session.bregma_col} "
                 f"({len(session.boxes)} boxes moved with it)")
        self._sync_boxes()

    def _on_target_moved(self) -> None:
        """Bregma was dragged: move it, and carry every box with it."""
        if self._syncing:
            return
        pos = self.target.pos()
        # The marker sits at the pixel CENTRE, so undo the half-pixel to get y_1/x_2.
        self._move_bregma(int(round(pos.y() + 0.5)), int(round(pos.x() + 0.5)))
        session = self.session
        self.say(f"Bregma -> row={session.bregma_row}, col={session.bregma_col} "
                 f"({len(session.boxes)} boxes moved with it)")
        self._sync_boxes()

    def _on_lambda_moved(self) -> None:
        """Lambda was dragged: take its distance from Bregma as the new scale."""
        if self._syncing:
            return
        pos = self.lambda_target.pos()
        # Only the ROW is read. Lambda is a midline landmark, and one dragged off the
        # midline would describe a ROTATED brain -- which a Box cannot express, for
        # the same reason the RectROIs are not rotatable. A sideways drag therefore
        # snaps back to x_2 on the next sync rather than being silently honoured.
        self._set_lambda(int(round(pos.y() + 0.5)) - self.session.y_1)

    # -- edits ---------------------------------------------------------------
    def _nudge_box(self, dr: int, dc: int) -> None:
        if self.selected is None:
            self.say("No ROI selected -- click one first.")
            return
        session = self.session
        rows, cols = session.image.shape
        r0, r1, c0, c1 = pixel_bounds(session.boxes[self.selected], session.y_1, session.x_2)
        dr = _clamp(dr, -r0, rows - 1 - r1)          # keep the whole box in frame
        dc = _clamp(dc, -c0, cols - 1 - c1)
        session.boxes[self.selected] = box_from_pixels(
            r0 + dr, r1 + dr, c0 + dc, c1 + dc, session.y_1, session.x_2)
        session.rebase()             # see _on_roi_changed
        session.dirty = True
        self._sync_boxes()

    def _resize_box(self, delta: int) -> None:
        """Grow (+1) or shrink (-1) the selected box by ``delta`` px on all sides."""
        if self.selected is None:
            self.say("No ROI selected -- click one first.")
            return
        session = self.session
        rows, cols = session.image.shape
        r0, r1, c0, c1 = pixel_bounds(session.boxes[self.selected], session.y_1, session.x_2)
        r0, r1 = _clamp(r0 - delta, 0, rows - 1), _clamp(r1 + delta, 0, rows - 1)
        c0, c1 = _clamp(c0 - delta, 0, cols - 1), _clamp(c1 + delta, 0, cols - 1)
        if r1 < r0 or c1 < c0:      # a box may not shrink past a single pixel
            self.say("Box is already 1 px -- cannot shrink further.")
            return
        session.boxes[self.selected] = box_from_pixels(r0, r1, c0, c1,
                                                      session.y_1, session.x_2)
        session.rebase()             # see _on_roi_changed
        session.dirty = True
        self._sync_boxes()

    def _set_lambda(self, lambda_offset: int) -> None:
        """Set the Bregma -> Lambda distance and rescale the layout about Bregma.

        This is the only thing in the editor that rewrites every box at once. The
        distance is a per-animal ruler: a brain whose Bregma-Lambda axis measures 10%
        longer than the atlas's sits 10% larger in the field of view, so every offset
        grows by 10%. Bregma does not move and (by default) no box changes size --
        see :func:`scale_boxes`.

        Clamped, unlike the boxes it produces: Lambda is a landmark you point at
        something visible, so it stays on the frame, whereas a box pushed off the
        frame by the rescale is left off it, turns red and blocks the save. Same
        division of labour as :meth:`_move_bregma`.
        """
        session = self.session
        rows, _cols = session.image.shape
        lambda_offset = _clamp(lambda_offset, MIN_LAMBDA_OFFSET, rows - session.y_1)
        session.rescale_to(lambda_offset, self.args.lambda_scales_box_size)
        session.dirty = True
        self.say(f"Lambda {lambda_offset} px below Bregma -> scale "
                 f"{session.lambda_scale:.3f}x of the reference layout "
                 f"({len(session.boxes)} boxes rescaled)")
        self._sync_boxes()

    def _move_bregma(self, y_1: int, x_2: int) -> None:
        """Move Bregma; every box translates with it, rigidly.

        This is the whole point of storing boxes as *offsets* rather than pixels:
        ``session.boxes`` is not touched here, so the constellation cannot deform
        and no box can change size -- one assignment moves the entire layout.
        Every way of moving Bregma (drag, ctrl+arrows) comes through here, so they
        all behave identically.

        Lambda travels along too, because it is stored as a distance rather than a
        position: translating the layout must not change its scale.

        Only Bregma itself is clamped to the frame, not the boxes it carries: a
        layout dragged too near an edge pushes boxes out, and those turn red and
        block the save (see :meth:`_out_of_frame`). Clamping the boxes instead
        would mean silently *deforming* the layout to fit -- the one thing offsets
        exist to prevent.
        """
        session = self.session
        rows, cols = session.image.shape
        session.y_1 = _clamp(y_1, 1, rows)
        session.x_2 = _clamp(x_2, 1, cols)
        session.dirty = True

    # -- saving --------------------------------------------------------------
    def _save(self, session: Session, path: Path, name: str) -> bool:
        """Write one ROI set. Refuses (returns False) if a box is off the frame."""
        self._ensure_image(session)
        bad = self._out_of_frame(session)
        if bad:
            self.say(f"NOT saved -- {len(bad)} box(es) off the frame: {bad}. "
                     f"Drag them back or move Bregma.")
            return False
        # Final gate: the library's own validation, on the geometry as saved. It
        # raises loudly if anything got past the check above.
        cfg = ROIConfig(y_1=session.y_1, x_2=session.x_2, boxes=dict(session.boxes))
        box_slices_for(cfg, session.image.shape)

        source = (f"drawn with roi_editor.py on the {self.profile_name} preview grid; "
                  f"folder {session.folder}")
        written = save_roi_set(path, session.boxes, session.bregma_row, session.bregma_col,
                               grid=session.image.shape, name=name, source=source,
                               lambda_offset=session.lambda_offset,
                               downsample=get_profile(self.profile_name).downsample)
        session.saved_to = str(written)
        session.dirty = False
        self.say(f"Saved {name} -> {written}")
        return True

    def _save_current(self) -> None:
        session = self.session
        self._save(session, Path(self.args.roi_set_dir) / f"{session.key}.yaml", session.key)

    def _save_all(self) -> None:
        # Loads the preview of any page not yet opened: a save is validated against
        # the real frame, so every session must have one.
        saved = sum(
            self._save(s, Path(self.args.roi_set_dir) / f"{s.key}.yaml", s.key)
            for s in self.sessions
        )
        self.say(f"Saved {saved}/{len(self.sessions)} session ROI sets to "
                 f"{self.args.roi_set_dir}")

    def _write_shared(self) -> None:
        """One file for every session -- the 'same ROIs everywhere' option."""
        session = self.session
        path = Path(self.args.roi_set_dir) / self.args.shared_name
        self._save(session, path, Path(self.args.shared_name).stem)

    def _apply_to_all(self, with_bregma: bool) -> None:
        session = self.session
        for other in self.sessions:
            if other is session:
                continue
            other.boxes = {label: box for label, box in session.boxes.items()}
            # The Lambda distance goes with the boxes, not with Bregma: these offsets
            # ARE the ones drawn at this distance, so leaving the old value behind
            # would make the copy claim a scale it is not at, and the next rescale on
            # that page would be measured from the wrong reference.
            other.lambda_offset = session.lambda_offset
            other.rebase()
            if with_bregma:
                other.y_1, other.x_2 = session.y_1, session.x_2
            other.dirty = True
        what = ("boxes, scale and Bregma" if with_bregma
                else "boxes and scale (each keeps its Bregma)")
        self.say(f"Applied {session.key}'s {what} to all {len(self.sessions)} sessions "
                 f"-- press S to write them, or w for one shared file.")

    # -- keyboard ------------------------------------------------------------
    def _on_key(self, event) -> None:
        """Qt key handler, installed as the window's ``keyPressEvent``."""
        Qt = self.QtCore.Qt
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)
        session = self.session
        rebuild = False

        arrows = {Qt.Key_Up: (-1, 0), Qt.Key_Down: (1, 0),
                  Qt.Key_Left: (0, -1), Qt.Key_Right: (0, 1)}

        if key in arrows:
            dr, dc = arrows[key]
            # ctrl -> the anchor (and therefore every box); no modifier -> the one
            # selected box. shift scales the step the same way in both cases.
            step = self.NUDGE_BIG if shift else 1
            if ctrl:
                self._move_bregma(session.y_1 + dr * step, session.x_2 + dc * step)
                self.say(f"Bregma -> row={session.bregma_row}, col={session.bregma_col} "
                         f"({len(session.boxes)} boxes moved with it)")
                self._sync_boxes()
            else:
                self._nudge_box(dr * step, dc * step)
            return

        # , / . change the Bregma -> Lambda distance and so rescale the layout. Both
        # the unshifted keys and the shifted < / > are listed because which of the two
        # Qt reports for "shift+comma" depends on the keyboard layout.
        if key in (Qt.Key_Comma, Qt.Key_Less, Qt.Key_Period, Qt.Key_Greater):
            step = self.NUDGE_BIG if shift else 1
            sign = -1 if key in (Qt.Key_Comma, Qt.Key_Less) else 1
            self._set_lambda(session.lambda_offset + sign * step)
            return

        if key == Qt.Key_N:
            self.index = (self.index + 1) % len(self.sessions)
            self.selected, self.message = None, ""
            rebuild = True
        elif key == Qt.Key_P:
            self.index = (self.index - 1) % len(self.sessions)
            self.selected, self.message = None, ""
            rebuild = True
        elif key == Qt.Key_S:
            # Qt reports the letter and the modifier separately; matplotlib used to
            # hand over a pre-cased "s" / "S".
            self._save_all() if shift else self._save_current()
        elif key == Qt.Key_W:
            self._write_shared()
        elif key == Qt.Key_A:
            self._apply_to_all(with_bregma=shift)
        elif key == Qt.Key_L:
            # Undo all scaling: back to the reference layout, at scale 1.000x. Not the
            # same as 'r' -- hand edits to individual boxes are part of the reference
            # (they rebased it), so this only takes out the stretch.
            self._set_lambda(session.base_lambda)
        elif key == Qt.Key_R:
            y_1, x_2, lam, boxes = session.seed
            session.y_1, session.x_2, session.boxes = y_1, x_2, dict(boxes)
            session.lambda_offset = lam
            session.rebase()          # the starting layout is the reference again
            session.dirty = False
            self.selected = None
            self.say(f"{session.key}: reset to the starting layout.")
            rebuild = True
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._resize_box(+1)
        elif key == Qt.Key_Minus:
            self._resize_box(-1)
        elif key in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            # Widen / narrow the display stretch; pure cosmetics, no geometry.
            self.clip[1] = _clamp_float(
                self.clip[1] + (2.0 if key == Qt.Key_BracketRight else -2.0),
                self.clip[0] + 1.0, 100.0)
            self._apply_levels()
            self.say(f"Contrast: {self.clip[0]:.1f}-{self.clip[1]:.1f} percentile")
        elif key == Qt.Key_H:
            print(HELP)
            self.say("Controls printed to the terminal.")
        elif key == Qt.Key_Q:
            self.window.close()
            return
        else:
            return

        if rebuild:
            self._rebuild_scene()

    def run(self) -> None:
        self._rebuild_scene()
        self.window.show()
        self.app.exec()


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------
def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """Minimal CLI overrides; everything else stays in RUN_CONFIG."""
    p = argparse.ArgumentParser(description="Interactive ROI / Bregma editor.")
    p.add_argument("--manifest", default=defaults["manifest"],
                   help="Manifest CSV; omit --folders to use it.")
    p.add_argument("--folders", nargs="*", default=defaults["folders"],
                   help="Interleaved folders to edit (ignores --manifest when given).")
    p.add_argument("--scope", choices=("animal", "recording"), default=defaults["scope"])
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"])
    p.add_argument("--shared-name", default=defaults["shared_name"])
    p.add_argument("--start-from", default=defaults["start_from"],
                   help="Seed every session from this ROI set file.")
    p.add_argument("--preview-frames", type=int, default=defaults["preview_frames"])
    p.add_argument("--bregma-row", type=int, default=defaults["bregma_row"])
    p.add_argument("--bregma-col", type=int, default=defaults["bregma_col"])
    p.add_argument("--lambda-offset", type=int, default=defaults["lambda_offset"],
                   help="Bregma->Lambda distance in FINAL-grid rows (the scale "
                        "reference; box-offset units, not doubled).")
    p.add_argument("--lambda-scales-box-size", action="store_true",
                   default=defaults["lambda_scales_box_size"],
                   help="Rescaling also scales each box's size, not just its position.")
    p.add_argument("--no-load-existing", dest="load_existing", action="store_false",
                   default=defaults["load_existing"],
                   help="Ignore ROI sets already in --roi-set-dir.")
    ns = p.parse_args()
    ns.profile = defaults["profile"]
    ns.channel_order = defaults["channel_order"]
    # An explicit --folders list wins over the manifest.
    if ns.folders and "--folders" in sys.argv:
        ns.manifest = None
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Prefer CLI args when any are given, else fall back to RUN_CONFIG (editor mode)."""
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(
        manifest=config["manifest"],
        folders=list(config["folders"]),
        scope=config["scope"],
        profile=config["profile"],
        channel_order=config["channel_order"],
        preview_frames=config["preview_frames"],
        bregma_row=config["bregma_row"],
        bregma_col=config["bregma_col"],
        lambda_offset=config["lambda_offset"],
        lambda_scales_box_size=bool(config["lambda_scales_box_size"]),
        roi_set_dir=config["roi_set_dir"],
        shared_name=config["shared_name"],
        load_existing=bool(config["load_existing"]),
        start_from=config["start_from"],
    )


def main() -> None:
    args = build_runtime_args()
    profile = get_profile(args.profile)
    default_boxes = dict(profile.atlas)

    # A tiny reference distance makes every rescale wild and a zero one is a division
    # by zero, so this is refused up front rather than at the first drag.
    if args.lambda_offset < MIN_LAMBDA_OFFSET:
        raise ValueError(
            f"lambda_offset={args.lambda_offset} is below the minimum "
            f"{MIN_LAMBDA_OFFSET} px. It is the distance the layout's scale is "
            f"measured against, so it has to be a usable ruler."
        )

    print(f"Profile   : {profile.name} ({profile.n_rois} ROIs: {profile.labels})")
    print(f"ROI sets  : {args.roi_set_dir}")
    print(f"Lambda    : {args.lambda_offset} px below Bregma (scale reference; "
          f"box sizes {'scale too' if args.lambda_scales_box_size else 'stay fixed'})")
    if args.manifest:
        print(f"Manifest  : {args.manifest} (scope={args.scope})")
        sessions = sessions_from_manifest(args, default_boxes)
    else:
        sessions = sessions_from_folders(args, default_boxes)
    if not sessions:
        raise ValueError("No sessions to edit: give a manifest or a non-empty folders list.")
    print(f"Sessions  : {len(sessions)} -> {[s.key for s in sessions]}")
    print(HELP)
    print("\nOpening the editor window (it may open BEHIND this terminal). The first "
          "preview is\ndecoded now -- over a network share that takes a few seconds. "
          "Nothing is analysed\nhere: draw the ROIs, press 's' to save, then run the "
          "pipeline with --roi-set.\n", flush=True)

    ROIEditor(sessions, args, profile.name).run()
    print("Editor closed.")


if __name__ == "__main__":
    main()
