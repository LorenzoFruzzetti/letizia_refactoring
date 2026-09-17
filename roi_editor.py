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

ROTATING the constellation around Bregma (boxes remain axis-aligned):

    z / x                     rotate left / right 1 degree
    Z / X                     rotate left / right 5 degrees

Adjusting ONE box, once the layout as a whole sits right:

    drag inside a box        move it            drag a corner handle  resize it
    arrows / shift+arrows    nudge box 1 / 5    + / -                 grow / shrink

Everything else:

    n / p                    next / prev session
    c                        copy the PREVIOUS page's layout onto this one
    s / S                    save this session / save every session
    ctrl+s                   save only the sessions with unsaved changes
    w                        write the SHARED roi set (one file for all sessions)
    a / A                    apply this layout to all sessions (A also copies Bregma)
    r                        reset this session to how it started
    Resume (button)          reload this page from the saved set on disk
    [ / ]                    display contrast    h    print this help    q  quit

Paging carries work forward: the first time you open a page it takes the layout you
are looking at now, so an adjustment made once follows you through a run of similar
recordings. A page that already has its own saved ROI set is the exception -- those
boxes were drawn for that recording and are kept. `c` overrides either way.

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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
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
    # True (default): editing one box of a bilateral pair moves its twin to the exact
    # mirror image, so a layout cannot leave the editor asymmetric. Every ROI set drawn
    # before this lock existed lost its mirror symmetry on V1 and M2 by 2-3 px, always
    # on one hemisphere, and 'c' then propagated the slip to every later page.
    # Turn it off ('m' in the window, or --no-mirror-lock) only to draw a deliberately
    # one-sided layout.
    "mirror_lock": True,
    # Where ROI sets are WRITTEN (and, unless roi_seed_dir is set, also read from).
    "roi_set_dir": r"roi_sets",
    # Optional read-only directory of starting layouts. When set, a page with no saved
    # set of its own in roi_set_dir is seeded from <roi_seed_dir>/<key>.yaml instead of
    # the profile atlas, and saves still go to roi_set_dir -- so a verified reference
    # set can be opened, inspected and adjusted without ever being overwritten.
    # None = read starting layouts from roi_set_dir, the original single-directory behaviour.
    "roi_seed_dir": None,
    # Filename of the ONE shared set written by the 'w' key -- the "same ROIs for
    # every session" option.
    "shared_name": "shared_roi_set.yaml",
    # Start each session from <roi_set_dir>/<key>.yaml when that file exists, so
    # re-running the editor picks up where you left off.
    "load_existing": False,
    # Seed EVERY session from this one ROI set instead of the profile's atlas
    # (e.g. the shared file, to adjust it per animal). None = profile atlas.
    "start_from": None,
    # Pre-computed 128×128 anatomy TIFFs, one per session, named <key>.tif.
    # When set, sessions are built directly from these images -- no interleaved
    # recording folder is needed. Overrides "manifest" and "folders".
    # roi_sets/alignment_images already holds one image per recorded session.
    "image_dir": r"roi_sets\alignment_images",
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
    "ROTATE CONSTELLATION: z / x = left / right 1 degree    "
    "Z / X = left / right 5 degrees\n"
    "FLIP CONSTELLATION: f = mirror entire layout left-right (around y-axis)    "
    "v = mirror up-down (around x-axis)\n"
    "ONE BOX: left-drag it to move, left-drag a corner to resize, "
    "arrows / shift+arrows to nudge 1 / 5 px\n"
    "+ / -: grow / shrink box    "
    "[ / ]: contrast    r: reset page\n"
    "MIRROR LOCK (on): editing one box of an L/R pair mirrors its twin about Bregma"
    "    m: toggle it off for a one-sided edit\n"
    "n / p: next / prev session    c: copy the PREVIOUS page's layout onto this one\n"
    "s: save this session    S: save all    ctrl+s: save only the modified ones    "
    "w: write SHARED set\n"
    "a: apply boxes + scale to all sessions    A: also copy Bregma    "
    "r: reset this session    R: reset ALL sessions    h: help    q: quit\n"
    "BUTTON ONLY -- Resume: reload this page from its saved ROI set, or from the "
    "most recently saved one when it has none yet"
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


def round_half_away(value: float) -> int:
    """Round to nearest, ties away from zero.

    Odd -- ``f(-x) == -f(x)`` -- so the two hemispheres round the same way, which
    Python's half-to-even ``round`` is too (``round(0.5) == 0`` but ``round(-0.5)``
    is also ``0``... as ``-0``; the failure is elsewhere). It is translation
    invariant, ``f(x + n) == f(x) + n``, everywhere EXCEPT across a sign change at a
    tie, and no rounding rule can have both properties at once -- see
    :func:`scale_boxes`, which is why that function reflects explicitly rather than
    relying on either.
    """
    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


def mirror_box(box: Box) -> Box:
    """``box`` reflected across the midline that runs through Bregma.

    Offsets are measured from Bregma, so the reflection is just a negate-and-swap of
    the column pair; the rows are untouched because the midline is vertical.
    """
    return Box(row_start=box.row_start, row_end=box.row_end,
               col_start=-box.col_end, col_end=-box.col_start)


def mirror_twin(label: str, labels: Iterable[str]) -> str | None:
    """The bilateral counterpart of ``label``, or None when it has none.

    The side letter is the first ``L``/``R`` whose swap names another box in the same
    atlas: ``M2L_alta -> M2R_alta``, ``FLL -> FLR``, ``RSL_alta -> RSR_alta``. Asking
    for a real label rather than assuming a position is what keeps ``RSL_alta``'s
    leading ``R`` from being read as the side (``LSL_alta`` is not a box).
    """
    labels = set(labels)
    for i, ch in enumerate(label):
        if ch not in "LR":
            continue
        twin = label[:i] + ("R" if ch == "L" else "L") + label[i + 1:]
        if twin in labels:
            return twin
    return None


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

    A mirrored left/right pair must scale to a still-mirrored pair, or the atlas
    quietly loses its symmetry at a half-pixel and every left-vs-right contrast drawn
    from it is confounded. NO rounding rule delivers that on its own. The mirror needs
    ``f(-x) == -f(x)`` (the hemispheres must round the same way) *and*
    ``f(x + n) == f(x) + n`` for integer ``n`` (the half-span shift that turns a centre
    into an edge must not move one side and not the other), and the two are
    contradictory at a tie: oddness gives ``f(-0.5) == -f(0.5)``, translation gives
    ``f(-0.5) == f(0.5) - 1``, so ``f(0.5)`` would have to be ``0.5``. Half-to-even
    fails the second (``round(5.5) - round(0.5) == 6``); ties-away fails it across zero
    (``round_half_away(4.5) - 5 == 0`` but ``round_half_away(-0.5) == -1``).

    So the reflection is done rather than hoped for: a box on the LEFT of Bregma (a
    negative column centre) is computed as the exact mirror of the same box reflected
    onto the right, ``-(right_edge)``. Whatever the rounding does, it does the same
    thing to both hemispheres, at every factor and every span. Rows are rounded
    plainly -- the midline is vertical, so rows carry no reflection, and bilateral
    twins hold identical row offsets and therefore scale identically anyway.

    ``factor == 1.0`` is exactly the identity (every centre is an integer or a
    half-integer, both exact in binary), so a no-op Lambda change moves nothing.
    """
    scaled: dict[str, Box] = {}
    for label, box in boxes.items():
        if scale_size:
            r0, r1 = (round_half_away(box.row_start * factor),
                      round_half_away(box.row_end * factor))
            c0, c1 = (round_half_away(box.col_start * factor),
                      round_half_away(box.col_end * factor))
            # Shrinking must not invert a box: Box requires end >= start.
            scaled[label] = Box(r0, max(r0, r1), c0, max(c0, c1))
            continue
        span_r = box.row_end - box.row_start
        span_c = box.col_end - box.col_start
        r0 = round_half_away((box.row_start + box.row_end) / 2 * factor - span_r / 2)
        centre_c = (box.col_start + box.col_end) / 2 * factor
        if centre_c >= 0:
            c0 = round_half_away(centre_c - span_c / 2)
        else:
            # Reflect the right-hand box and negate, so the pair is mirrored by
            # construction instead of by a property no rounding rule actually has.
            c0 = -(round_half_away(-centre_c - span_c / 2) + span_c)
        scaled[label] = Box(r0, r0 + span_r, c0, c0 + span_c)
    return scaled


def rotate_boxes(boxes: dict[str, Box], degrees: float) -> dict[str, Box]:
    """Rotate box centres around Bregma while keeping the boxes axis-aligned.

    ROI sets describe rectangular array slices and therefore cannot represent a
    tilted rectangle. Rotation moves each rectangle's centre as a constellation;
    its pixel width and height remain unchanged.
    """
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    rotated: dict[str, Box] = {}
    for label, box in boxes.items():
        span_r = box.row_end - box.row_start
        span_c = box.col_end - box.col_start
        centre_r = (box.row_start + box.row_end) / 2
        centre_c = (box.col_start + box.col_end) / 2
        # Rows increase downward, so this is a clockwise-positive screen rotation.
        new_r = centre_r * cosine + centre_c * sine
        new_c = centre_c * cosine - centre_r * sine
        r0 = int(round(new_r - span_r / 2))
        c0 = int(round(new_c - span_c / 2))
        rotated[label] = Box(r0, r0 + span_r, c0, c0 + span_c)
    return rotated


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
    image: np.ndarray | None = None   # baked before the editing loop starts
    saved_to: str = ""                # last file written for this session
    dirty: bool = False
    # True when this page's geometry came from, or has been written to, its OWN
    # <key>.yaml -- i.e. boxes drawn for THIS recording rather than inherited. It is
    # what stops page-to-page carry-over overwriting saved work (see ROIEditor._navigate).
    has_own_set: bool = False

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
                 bregma_row: int, bregma_col: int
                 ) -> tuple[int, int, int, dict[str, Box], bool]:
    """Starting geometry for one session: existing file > --start-from > profile.

    Returns ``(y_1, x_2, lambda_offset, boxes, has_own_set)``. A ROI set that carries
    a Bregma overrides the manifest's -- it is the Bregma those boxes were drawn from,
    and separating them would silently move every box. Same for its Lambda distance:
    the boxes in the file are already at that scale, so adopting the boxes without
    it would leave the editor rescaling from the wrong reference.

    ``has_own_set`` is True for either of the two per-recording candidates below:
    both hold boxes belonging to THIS recording, so the editor must not let
    page-to-page carry-over overwrite them. ``--start-from`` seeds every page from the
    same file and so is not "its own" -- it is a starting guess like the atlas.

    Candidate order:

    * ``<roi_set_dir>/<key>.yaml`` -- a set already saved for this recording in the
      output directory. It wins because it is the work in progress: re-running the
      editor must pick up where the last run left off.
    * ``<roi_seed_dir>/<key>.yaml`` -- this recording's layout in a read-only reference
      directory (e.g. the rebuilt sets). Only consulted when nothing has been saved
      for this page yet, and never written to; saves always go to ``roi_set_dir``.
    * ``--start-from`` -- one file seeding every page.
    * the profile atlas.
    """
    own_path = Path(args.roi_set_dir) / f"{key}.yaml"
    seed_dir = getattr(args, "roi_seed_dir", None)
    seed_path = Path(seed_dir) / f"{key}.yaml" if seed_dir else None

    candidates = []
    if args.load_existing:
        candidates.append(own_path)
        # A seed directory is a starting layout, not an output: consulted only when the
        # page has nothing saved of its own, so re-runs never fall back past your edits.
        if seed_path is not None and seed_path != own_path:
            candidates.append(seed_path)
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
            return row // 2, col // 2, lam, dict(atlas), path in (own_path, seed_path)

    return bregma_row // 2, bregma_col // 2, args.lambda_offset, dict(default_boxes), False


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
            y_1, x_2, lam, boxes, own = _seed_layout(key, args, default_boxes,
                                                     bregma_row, bregma_col)
            sessions.append(Session(key=key, folder=folder, y_1=y_1, x_2=x_2,
                                    boxes=boxes, lambda_offset=lam, has_own_set=own))
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
        y_1, x_2, lam, boxes, own = _seed_layout(key, args, default_boxes,
                                                 args.bregma_row, args.bregma_col)
        sessions.append(Session(key=key, folder=str(folder), y_1=y_1, x_2=x_2,
                                boxes=boxes, lambda_offset=lam, has_own_set=own))
    return sessions


def sessions_from_images(args: argparse.Namespace,
                         default_boxes: dict[str, Box]) -> list[Session]:
    """One session per pre-computed anatomy TIFF in image_dir; no interleaved folder needed.

    Each ``<key>.tif`` must be a 128×128 float32 image already on the final analysis
    grid (as saved by a previous bake_all_images run). The image is loaded up front
    so bake_all_images never tries to decode a recording folder.
    """
    image_dir = Path(args.image_dir)
    sessions: list[Session] = []
    for image_path in sorted(image_dir.glob("*.tif")):
        key = image_path.stem
        y_1, x_2, lam, boxes, own = _seed_layout(key, args, default_boxes,
                                                 args.bregma_row, args.bregma_col)
        session = Session(key=key, folder=str(image_path), y_1=y_1, x_2=x_2,
                          boxes=boxes, lambda_offset=lam, has_own_set=own)
        session.image = np.asarray(tifffile.imread(image_path))
        sessions.append(session)
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
        # Editing one box of a bilateral pair rewrites its twin as the exact mirror.
        # Defaulted here as well as in RUN_CONFIG so a caller that builds `args` by
        # hand gets the safe behaviour rather than the historical one.
        self.mirror_lock = bool(getattr(args, "mirror_lock", True))
        self.clip = [1.0, 99.5]      # display percentiles
        self.message = ""
        # Pages already shown this run -- the first visit to a page carries the
        # CURRENT layout onto it instead of resetting to its own seed (see
        # _navigate); a page visited again keeps whatever it was left at.
        self._visited: set[int] = {0}
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
        self.QtWidgets = QtWidgets
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

        button_bar = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Previous")
        self.btn_rotate_left = QtWidgets.QPushButton("Rotate left")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_resume = QtWidgets.QPushButton("Resume")
        self.btn_rotate_right = QtWidgets.QPushButton("Rotate right")
        self.btn_next = QtWidgets.QPushButton("Next >>")
        for btn in (self.btn_prev, self.btn_rotate_left, self.btn_save,
                    self.btn_resume, self.btn_rotate_right, self.btn_next):
            btn.setFocusPolicy(QtCore.Qt.NoFocus)  # keep arrow keys reaching the window
            button_bar.addWidget(btn)
        layout.addLayout(button_bar)
        self.btn_prev.clicked.connect(lambda: self._navigate(-1))
        self.btn_next.clicked.connect(lambda: self._navigate(+1))
        self.btn_rotate_left.clicked.connect(lambda: self._rotate_constellation(-1))
        self.btn_rotate_right.clicked.connect(lambda: self._rotate_constellation(+1))
        self.btn_save.clicked.connect(self._on_click_save)
        self.btn_resume.clicked.connect(self._on_click_resume)

        # Second row: the two batch gestures. Kept apart from the per-page row above
        # because they reach beyond the page being looked at -- one pulls geometry in
        # from another page, the other writes several files at once.
        batch_bar = QtWidgets.QHBoxLayout()
        self.btn_copy_prev = QtWidgets.QPushButton("Copy previous page's ROIs")
        self.btn_save_modified = QtWidgets.QPushButton("Save modified")
        for btn in (self.btn_copy_prev, self.btn_save_modified):
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            batch_bar.addWidget(btn)
        layout.addLayout(batch_bar)
        self.btn_copy_prev.clicked.connect(self._on_click_copy_previous)
        self.btn_save_modified.clicked.connect(self._on_click_save_modified)

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
        """Decode one preview, retaining a guard for direct/test callers."""
        if session.image is None:
            # flush: over a network share this is the slow step, and an unflushed
            # line makes the editor look like it is doing nothing.
            print(f"Loading preview: {session.folder}", flush=True)
            session.image = preview_image(session.folder, self.args.channel_order,
                                          self.args.preview_frames)

    def bake_all_images(self) -> None:
        """Decode and save every preview before entering the editing loop.

        Baking up front keeps page changes responsive. Existing TIFFs are loaded
        directly; only missing previews are decoded from their recordings and
        written. The final-grid images use float32: they are alignment aids, not
        analysis inputs, so retaining the full source resolution is wasteful.
        """
        total = len(self.sessions)
        image_dir = Path(self.args.roi_set_dir) / "alignment_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        dlg = self.QtWidgets.QProgressDialog(
            "Preparing previews...", None, 0, total, self.window)
        dlg.setWindowTitle("ROI editor")
        dlg.setMinimumWidth(420)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        self.app.processEvents()
        try:
            for number, session in enumerate(self.sessions, start=1):
                image_path = image_dir / f"{session.key}.tif"
                if image_path.is_file():
                    dlg.setLabelText(
                        f"Loading preview {number}/{total}: {session.key}")
                    self.app.processEvents()
                    session.image = np.asarray(tifffile.imread(image_path))
                    print(f"Loaded alignment preview: {image_path}", flush=True)
                else:
                    dlg.setLabelText(
                        f"Preparing preview {number}/{total}: {session.key}")
                    self.app.processEvents()
                    # A session may already carry an in-memory preview (for example
                    # in tests or image-directory mode). Save it when the cache is
                    # missing; decode only when no preview is available yet.
                    self._ensure_image(session)
                    tifffile.imwrite(image_path, session.image.astype(np.float32))
                    print(f"Saved alignment preview: {image_path}", flush=True)
                dlg.setValue(number)
                self.app.processEvents()
        finally:
            dlg.close()

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

    def _popup(self, text: str, title: str = "ROI editor", warn: bool = False) -> None:
        """A modal feedback box -- for button actions, where the status line alone
        is easy to miss (unlike keyboard shortcuts, a button click has no other
        confirmation that anything happened)."""
        box = self.QtWidgets.QMessageBox.warning if warn else self.QtWidgets.QMessageBox.information
        box(self.window, title, text)

    def _navigate(self, delta: int) -> None:
        """Switch page by +-1 (wraps around), shared by the n/p keys and the buttons.

        A page not yet visited this run is seeded from the CURRENT layout rather
        than its own starting atlas/file -- paging through many similar sessions
        should carry an adjustment forward, not throw it away on every 'next'. A
        page visited before keeps whatever it was left at (edited or not); 'r'
        still resets a page to its own true original layout regardless.

        The one exception is a page that already has its OWN ``<key>.yaml``
        (``has_own_set``): those boxes were drawn for this recording and are better
        than any neighbour's, so arriving on the page must not overwrite them. Carry
        the neighbour's layout over deliberately with 'c' if that is what you want.
        """
        source = self.session
        self.index = (self.index + delta) % len(self.sessions)
        target = self.session
        note = ""
        if self.index not in self._visited:
            self._visited.add(self.index)
            if target.has_own_set:
                note = (f"{target.key}: kept its own saved ROI set "
                        f"(not carried over from {source.key}); press 'c' to copy "
                        f"the previous page's layout over it.")
            else:
                target.boxes = dict(source.boxes)
                target.lambda_offset = source.lambda_offset
                target.rebase()
                target.dirty = True
        self.selected, self.message = None, ""
        self._rebuild_scene()
        if note:
            self.say(note)

    def _copy_from_previous(self) -> bool:
        """Put the PREVIOUS page's whole layout on this page.

        The manual counterpart of what :meth:`_navigate` does automatically the
        first time a page is opened: consecutive recordings are usually the same
        animal under the same camera, so the layout just adjusted next door is a
        far better starting point than whatever this page was left at. Boxes, the
        Lambda distance AND Bregma all come across -- the point is for the two
        pages to look identical, and a copy that left Bregma behind would silently
        translate every box it just brought in.

        "Previous" is the page ``p`` / ``<< Previous`` goes to, wrap-around
        included, so on the first page it copies from the last one.
        """
        if len(self.sessions) < 2:
            self.say("Only one session loaded -- there is no previous page to copy from.")
            return False
        source = self.sessions[(self.index - 1) % len(self.sessions)]
        target = self.session
        target.boxes = dict(source.boxes)
        target.lambda_offset = source.lambda_offset
        target.y_1, target.x_2 = source.y_1, source.x_2
        # The copied boxes are the ones drawn at the copied Lambda distance, so they
        # become this page's reference too (see Session.rebase).
        target.rebase()
        target.dirty = True
        # This page was marked visited when it was opened; nothing to do there.
        self.selected = None
        self._rebuild_scene()
        self.say(f"{target.key}: layout copied from {source.key} "
                 f"({len(target.boxes)} boxes, Bregma row={target.bregma_row} "
                 f"col={target.bregma_col}, Lambda +{target.lambda_offset} px)")
        return True

    def _on_click_copy_previous(self) -> None:
        ok = self._copy_from_previous()
        self._popup(self.message, title="Copied from previous page" if ok
                    else "Nothing to copy", warn=not ok)

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

    def _sync_one(self, label: str) -> None:
        """Push ONE box back onto its rect, without touching the others.

        Used while a box is being dragged: the dragged rect is already where the mouse
        put it, and re-setting it mid-drag would fight the drag. Its mirrored twin,
        though, was changed by us and has to be moved.
        """
        session = self.session
        r0, r1, c0, c1 = pixel_bounds(session.boxes[label], session.y_1, session.x_2)
        roi = self._rois[label]
        self._syncing = True
        try:
            roi.setPos(c0, r0, finish=False)
            roi.setSize((c1 - c0 + 1, r1 - r0 + 1), finish=False)
        finally:
            self._syncing = False
        self._restyle()

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
        # The button carries the count so the size of the pending batch is visible
        # without paging through every session to look for "*unsaved*" headers.
        n_dirty = sum(s.dirty for s in self.sessions)
        self.btn_save_modified.setText(f"Save modified ({n_dirty})")
        self.btn_save_modified.setEnabled(n_dirty > 0)
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
        twin = self._mirror_to_twin(label)
        session.rebase()
        session.dirty = True
        self.selected = label
        if twin:
            # Only the twin: the dragged rect is already where the mouse left it, and
            # re-setting it here would fight the drag. _restyle only recolours, so the
            # twin's rect would otherwise stay behind its box.
            self.say(f"{label} moved, {twin} mirrored with it (m unlocks the pair)")
            self._sync_one(twin)
        else:
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
    def _mirror_to_twin(self, label: str) -> str | None:
        """Rewrite ``label``'s bilateral twin as its exact mirror. Returns the twin.

        Called after every edit that touches ONE box -- keyboard nudge, resize, mouse
        drag, handle drag -- so the pair cannot come apart in the first place. Copying
        the whole mirrored box rather than applying the mirrored *delta* is deliberate:
        it makes the twin exact regardless of how the edit was expressed, and it also
        repairs a pair that was already off.

        The twin is not clamped to the frame. A box pushed off it turns red and blocks
        the save, which is the same visible refusal an off-frame drag already gets --
        better than silently squashing the twin and breaking the symmetry to fit.
        """
        if not self.mirror_lock:
            return None
        twin = mirror_twin(label, self.session.boxes)
        if twin is None:
            return None
        self.session.boxes[twin] = mirror_box(self.session.boxes[label])
        return twin

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
        twin = self._mirror_to_twin(self.selected)
        session.rebase()             # see _on_roi_changed
        session.dirty = True
        if twin:
            self.say(f"{self.selected} moved, {twin} mirrored with it "
                     f"(m unlocks the pair)")
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
        twin = self._mirror_to_twin(self.selected)
        session.rebase()             # see _on_roi_changed
        session.dirty = True
        if twin:
            self.say(f"{self.selected} resized, {twin} mirrored with it "
                     f"(m unlocks the pair)")
        self._sync_boxes()

    def _rotate_constellation(self, degrees: float) -> None:
        """Rotate every ROI centre around Bregma, preserving box dimensions."""
        session = self.session
        session.boxes = rotate_boxes(session.boxes, degrees)
        session.rebase()
        session.dirty = True
        self.say(f"Constellation rotated {degrees:+g} degrees around Bregma "
                 f"({len(session.boxes)} axis-aligned boxes moved)")
        self._sync_boxes()

    def _flip_constellation(self, horizontal: bool) -> None:
        """Mirror Bregma and all boxes about the image centre on one axis."""
        session = self.session
        rows, cols = session.image.shape
        if horizontal:
            # Reflect across the y-axis: negate and swap col offsets.
            session.x_2 = _clamp(cols - (session.x_2 - 1), 1, cols)
            session.boxes = {
                label: Box(row_start=box.row_start, row_end=box.row_end,
                           col_start=-box.col_end, col_end=-box.col_start)
                for label, box in session.boxes.items()
            }
            msg = f"flipped left-right (Bregma col={session.bregma_col})"
        else:
            # Reflect across the x-axis: negate and swap row offsets.
            session.y_1 = _clamp(rows - (session.y_1 - 1), 1, rows)
            session.boxes = {
                label: Box(row_start=-box.row_end, row_end=-box.row_start,
                           col_start=box.col_start, col_end=box.col_end)
                for label, box in session.boxes.items()
            }
            msg = f"flipped up-down (Bregma row={session.bregma_row})"
        session.rebase()
        session.dirty = True
        self.say(f"Constellation {msg}, {len(session.boxes)} boxes mirrored")
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
    def _save(self, session: Session, path: Path, name: str,
              own_set: bool = True) -> bool:
        """Write one ROI set. Refuses (returns False) if a box is off the frame.

        ``own_set=False`` for the shared file, which is one layout for every page and
        so does not make any single page's geometry "its own" (see Session.has_own_set).
        """
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
        if own_set:
            # Only a file under this page's own key clears the unsaved state: the
            # shared set is a different file, and a page whose geometry went only
            # there still has nothing saved under its own name.
            session.dirty = False
            # It now holds geometry drawn for THIS recording, so a later first visit
            # must not carry a neighbour's over it. This matters for 'S' / "Save
            # modified", which write pages that have not been opened yet.
            session.has_own_set = True
        # The header's "*unsaved*" marker and the modified-count button both read
        # `dirty`, so they have to be redrawn here -- a save is the one edit-like
        # action that does not go through _restyle.
        self._refresh_header()
        self.say(f"Saved {name} -> {written}")
        return True

    def _save_current(self) -> bool:
        session = self.session
        return self._save(session, Path(self.args.roi_set_dir) / f"{session.key}.yaml", session.key)

    def _on_click_save(self) -> None:
        """Save button: same as the 's' key, plus a popup so the save is unmissable."""
        ok = self._save_current()
        self._popup(self.message, title="Saved" if ok else "Save failed", warn=not ok)

    def _latest_saved_set(self) -> Path | None:
        """The most recently written ROI set in ``roi_set_dir``, whichever page it is for.

        "Most recent" is by modification time rather than by page order: it is the
        geometry last committed to disk, which in a long run is the layout that has
        had the most work put into it. Only used as Resume's fallback.
        """
        directory = Path(self.args.roi_set_dir)
        if not directory.is_dir():
            return None
        candidates = [p for p in directory.glob("*.yaml") if p.is_file()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _resume_session(self) -> bool:
        """Reload the current page from a saved ROI set. True when one was found.

        Distinct from 'r' (reset): 'r' goes back to how the page STARTED this run
        (the profile atlas, or whatever it was seeded from); this goes back to what
        was actually written to disk, discarding only the edits made since -- the
        "undo my mistakes, not my whole session" button.

        Two sources, in order:

        * ``<roi_set_dir>/<key>.yaml`` -- this page's own saved set. The page then
          matches its file exactly, so it is no longer unsaved and 'r' is re-pointed
          at this layout.
        * ``<roi_seed_dir>/<key>.yaml`` -- this page's layout in the read-only
          reference directory, when one is configured. Still this recording's own
          geometry, so it counts as "own": it is exactly the "discard my edits and go
          back to the reference" button.
        * failing that, the most recently saved set in ``roi_set_dir`` (see
          :meth:`_latest_saved_set`). With hundreds of pages and a handful saved, the
          per-page file usually does not exist yet, and the last layout drawn is a far
          better starting point than the atlas. That geometry belongs to a DIFFERENT
          page, so nothing has been written under this key: the page stays *unsaved*
          (and stays in the "Save modified" batch), and 'r' still returns to the
          layout this page opened with.
        """
        session = self.session
        own_path = Path(self.args.roi_set_dir) / f"{session.key}.yaml"
        seed_dir = getattr(self.args, "roi_seed_dir", None)
        seed_path = Path(seed_dir) / f"{session.key}.yaml" if seed_dir else None

        path = own_path if own_path.is_file() else None
        if path is None and seed_path is not None and seed_path.is_file():
            path = seed_path
        own = path is not None
        if path is None:
            path = self._latest_saved_set()
        if path is None:
            self.say(f"Nothing to resume from: {session.key} has no saved ROI set, and "
                     f"{self.args.roi_set_dir} does not hold a single one yet.")
            return False

        atlas, file_row, file_col = load_roi_set(path)
        lam = load_lambda_offset(path)
        if file_row is not None:
            session.y_1, session.x_2 = file_row // 2, file_col // 2
        session.boxes = dict(atlas)
        if lam is not None:
            session.lambda_offset = lam
        session.rebase()

        if own:
            session.seed = (session.y_1, session.x_2, session.lambda_offset,
                            dict(session.boxes))
            session.dirty = False
            session.has_own_set = True
            self.say(f"{session.key}: resumed from {path}")
        else:
            session.dirty = True
            saved_at = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %H:%M")
            self.say(f"{session.key} has no ROI set of its own -- resumed from the most "
                     f"recently saved one, {path.name} (saved {saved_at}). Check it "
                     f"against THIS anatomy, then press 's' to save it as "
                     f"{session.key}.yaml.")
        self.selected = None
        self._rebuild_scene()
        return True

    def _on_click_resume(self) -> None:
        ok = self._resume_session()
        self._popup(self.message, title="Resume" if ok else "Nothing to resume",
                    warn=not ok)

    def _save_many(self, sessions: list[Session], what: str) -> list[str]:
        """Write one ROI set per session; return the keys that were REFUSED.

        Every preview was baked at startup, so each save can validate against its
        real frame without doing I/O here. A refusal (a box off the frame) only
        skips that one session -- the rest are still written, and the keys come back
        so the summary can name them instead of leaving the count to be puzzled over.
        """
        results = [
            (s.key, self._save(s, Path(self.args.roi_set_dir) / f"{s.key}.yaml", s.key))
            for s in sessions
        ]
        refused = [key for key, ok in results if not ok]
        note = (f"  --  REFUSED (boxes off frame): {refused}" if refused else "")
        self.say(f"Saved {len(results) - len(refused)}/{len(results)} {what} ROI sets "
                 f"to {self.args.roi_set_dir}{note}")
        return refused

    def _save_all(self) -> None:
        self._save_many(self.sessions, "session")

    def _save_modified(self) -> list[str]:
        """Save every page with unsaved changes, and only those.

        The batch counterpart of 's': after paging through a run and adjusting some
        of the sessions, this writes exactly the ones that were touched. Unlike 'S'
        it does not rewrite files for pages that were only looked at, so their
        on-disk ``source`` line and timestamps keep saying when they were really
        drawn. ``dirty`` is set by every edit and cleared by a successful save, so
        it is the same "*unsaved*" marker the header shows.
        """
        pending = [s for s in self.sessions if s.dirty]
        if not pending:
            self.say("Nothing to save -- no session has unsaved changes.")
            return []
        return self._save_many(pending, "modified")

    def _on_click_save_modified(self) -> None:
        pending = any(s.dirty for s in self.sessions)
        refused = self._save_modified()
        self._popup(self.message,
                    title="Saved modified" if pending and not refused else "Save modified",
                    warn=bool(refused))

    def _write_shared(self) -> None:
        """One file for every session -- the 'same ROIs everywhere' option."""
        session = self.session
        path = Path(self.args.roi_set_dir) / self.args.shared_name
        self._save(session, path, Path(self.args.shared_name).stem, own_set=False)

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

        if key in (Qt.Key_Z, Qt.Key_X):
            step = self.NUDGE_BIG if shift else 1
            self._rotate_constellation((-step if key == Qt.Key_Z else step))
            return

        if key == Qt.Key_M:
            self.mirror_lock = not self.mirror_lock
            self.say("Mirror lock ON -- editing one box of an L/R pair now mirrors "
                     "its twin about Bregma."
                     if self.mirror_lock else
                     "Mirror lock OFF -- boxes move independently. Anything you edit "
                     "one-sided from here stays asymmetric, and 'c' carries it forward.")
            return

        if key == Qt.Key_F:
            self._flip_constellation(horizontal=True)
            return

        if key == Qt.Key_V:
            self._flip_constellation(horizontal=False)
            return

        if key == Qt.Key_N:
            self._navigate(+1)
            return
        elif key == Qt.Key_P:
            self._navigate(-1)
            return
        elif key == Qt.Key_C:
            self._copy_from_previous()
            return
        elif key == Qt.Key_S:
            # Qt reports the letter and the modifier separately; matplotlib used to
            # hand over a pre-cased "s" / "S".
            if ctrl:
                self._save_modified()
            elif shift:
                self._save_all()
            else:
                self._save_current()
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
            if shift:
                for s in self.sessions:
                    y_1, x_2, lam, boxes = s.seed
                    s.y_1, s.x_2, s.boxes = y_1, x_2, dict(boxes)
                    s.lambda_offset = lam
                    s.rebase()
                    s.dirty = False
                self.selected = None
                self.say(f"All {len(self.sessions)} sessions reset to starting layouts.")
                rebuild = True
            else:
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
        self.bake_all_images()
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
    p.add_argument("--roi-set-dir", default=defaults["roi_set_dir"],
                   help="Where ROI sets are written.")
    p.add_argument("--roi-seed-dir", default=defaults["roi_seed_dir"],
                   help="Read-only directory of starting layouts; saves still go to "
                        "--roi-set-dir.")
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
    p.add_argument("--no-mirror-lock", dest="mirror_lock", action="store_false",
                   default=defaults["mirror_lock"],
                   help="Let one box of a bilateral pair be edited without its twin "
                        "following. On by default: an asymmetric pair is almost always "
                        "a slip, not a decision.")
    p.add_argument("--no-load-existing", dest="load_existing", action="store_false",
                   default=defaults["load_existing"],
                   help="Ignore ROI sets already in --roi-set-dir.")
    p.add_argument("--image-dir", default=defaults.get("image_dir"), metavar="DIR",
                   help="Directory of pre-computed 128x128 anatomy TIFFs (<key>.tif). "
                        "Bypasses interleaved folder decoding; overrides --manifest/--folders.")
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
        mirror_lock=bool(config["mirror_lock"]),
        roi_set_dir=config["roi_set_dir"],
        roi_seed_dir=config["roi_seed_dir"],
        shared_name=config["shared_name"],
        load_existing=bool(config["load_existing"]),
        start_from=config["start_from"],
        image_dir=config.get("image_dir"),
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
    print(f"ROI sets  : {args.roi_set_dir}  (saves go here)")
    if getattr(args, "roi_seed_dir", None):
        print(f"Seeded from: {args.roi_seed_dir}  (read-only reference layouts)")
    print(f"Lambda    : {args.lambda_offset} px below Bregma (scale reference; "
          f"box sizes {'scale too' if args.lambda_scales_box_size else 'stay fixed'})")
    lock = "LOCKED (m unlocks)" if getattr(args, "mirror_lock", True) else "UNLOCKED (m locks)"
    print(f"Mirror    : {lock} -- editing one box of an L/R pair mirrors its twin")
    if getattr(args, "image_dir", None):
        print(f"Images    : {args.image_dir} (pre-computed anatomy previews)")
        sessions = sessions_from_images(args, default_boxes)
    elif args.manifest:
        print(f"Manifest  : {args.manifest} (scope={args.scope})")
        sessions = sessions_from_manifest(args, default_boxes)
    else:
        sessions = sessions_from_folders(args, default_boxes)
    if not sessions:
        raise ValueError("No sessions to edit: give a manifest or a non-empty folders list.")
    print(f"Sessions  : {len(sessions)} -> {[s.key for s in sessions]}")
    print(HELP)
    print("\nOpening the editor window (it may open BEHIND this terminal). All alignment "
          "previews are\ndecoded and saved under <roi_set_dir>/alignment_images "
          "before editing starts.\nNothing is analysed here: draw the ROIs, press "
          "'s' to save, then run the pipeline\nwith --roi-set.\n", flush=True)

    ROIEditor(sessions, args, profile.name).run()
    print("Editor closed.")


if __name__ == "__main__":
    main()
