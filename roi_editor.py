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
    boxes:                      # ORDER IS THE COLUMN ORDER OF R -- do not sort
      Laterale_L: {row_start: 21, row_end: 26, col_start: -34, col_end: -29}
      ...

This is a **superset of the library's atlas file format**: ``wfci.load_atlas``
reads it unchanged and simply ignores the two ``bregma_*`` keys. The extra keys
are there because a box is only meaningful together with the Bregma it was drawn
from -- keeping them in one file stops the two halves of the geometry drifting
apart. :func:`load_roi_set` returns both halves; ``run_intermingle_rs.py`` and
``run_botox_batch.py`` take a ``roi_set`` path in their ``RUN_CONFIG``.

Controls (also printed at the bottom of the window)
---------------------------------------------------
    left-drag inside a box   move it            left-drag a corner   resize it
    right-click              put Bregma here    ctrl+arrows          nudge Bregma
    arrows / shift+arrows    nudge box 1 / 5    + / -                grow / shrink
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

Needs a display (it opens a Tk window; it can appear BEHIND the terminal). Errors are
left to surface (no try/except);
the one thing that is *checked* rather than raised is a box that has been dragged
off the frame -- it turns red and the save is refused with a message, because in a
GUI a refusal you can act on beats a traceback.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
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
    "manifest": r"manifests\botox_restani_manifest.csv",
    # Manifest only: one editor page per animal ("animal" -- previews that
    # animal's FIRST recording, which is what the per-animal Bregma applies to)
    # or one page per t# recording ("recording" -- check every folder's first
    # image; more pages, same Bregma unless you change it per page).
    "scope": "animal",
    # Used when "manifest" is None: the interleaved folders to inspect.
    "folders": [
        r"\\146.48.88.209\share2\BOTOX_RESTANI\260611\R1\t1",
    ],
    # Pipeline whose ROI atlas is the starting layout (and whose downsampling
    # defines the preview grid).
    "profile": "cerebellar_rs",
    # Interleaved split: "auto" picks the brighter group as GCaMP. Only the GCaMP
    # channel is previewed.
    "channel_order": "auto",
    # How many GCaMP images to average into the preview. 1 = literally the first
    # image of the folder; a handful (5-20) is the same anatomy with less noise.
    "preview_frames": 1,
    # Fallback Bregma (RUN_CONFIG units, i.e. 2 x the final-grid pixel) used in
    # "folders" mode, or for a manifest row that carries none.
    "bregma_row": 121,
    "bregma_col": 134,
    # Where ROI sets are read from and written to.
    "roi_set_dir": r"roi_sets",
    # Filename of the ONE shared set written by the 'w' key -- the "same ROIs for
    # every session" option.
    "shared_name": "shared_roi_set.yaml",
    # Start each session from <roi_set_dir>/<key>.yaml when that file exists, so
    # re-running the editor picks up where you left off.
    "load_existing": True,
    # Seed EVERY session from this one ROI set instead of the profile's atlas
    # (e.g. the shared file, to adjust it per animal). None = profile atlas.
    "start_from": None,
    "prefer_cli_args": True,
}

# The ROI-set file's box fields, in the order the library's loader expects.
BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")

HELP = (
    "left-drag box: move   left-drag corner: resize   right-click: set Bregma   "
    "ctrl+arrows: move Bregma\n"
    "arrows / shift+arrows: nudge box 1 / 5 px    + / -: grow / shrink    "
    "[ / ]: contrast    r: reset page\n"
    "n / p: next / prev session    s: save this session    S: save all    "
    "w: write SHARED set\n"
    "a: apply these boxes to all sessions    A: apply boxes AND Bregma to all    "
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
) -> Path:
    """Write a ROI set (boxes + Bregma + grid) to YAML (or JSON, by extension).

    The layout is the library's atlas file plus ``bregma_row`` / ``bregma_col``,
    so ``wfci.load_atlas`` reads the same file and ignores those two keys.
    ``sort_keys=False`` is not cosmetic: box order IS the column order of ``R``,
    so alphabetising the file would silently relabel every matrix built from it.
    """
    payload = {
        "name": name,
        "grid": [int(grid[0]), int(grid[1])],
        "source": source,
        "bregma_row": int(bregma_row),
        "bregma_col": int(bregma_col),
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
    seed: tuple[int, int, dict[str, Box]] = field(default=None)  # for 'r' (reset)
    image: np.ndarray | None = None   # decoded lazily, when the page is opened
    saved_to: str = ""                # last file written for this session
    dirty: bool = False

    @property
    def bregma_row(self) -> int:
        """Bregma in RUN_CONFIG units: the scripts apply ``// 2`` to reach y_1."""
        return 2 * self.y_1

    @property
    def bregma_col(self) -> int:
        return 2 * self.x_2


def _seed_layout(key: str, args: argparse.Namespace, default_boxes: dict[str, Box],
                 bregma_row: int, bregma_col: int) -> tuple[int, int, dict[str, Box]]:
    """Starting geometry for one session: existing file > --start-from > profile.

    Returns ``(y_1, x_2, boxes)``. A ROI set that carries a Bregma overrides the
    manifest's -- it is the Bregma those boxes were drawn from, and separating
    them would silently move every box.
    """
    candidates = []
    if args.load_existing:
        candidates.append(Path(args.roi_set_dir) / f"{key}.yaml")
    if args.start_from:
        candidates.append(Path(args.start_from))

    for path in candidates:
        if path.is_file():
            atlas, file_row, file_col = load_roi_set(path)
            row = bregma_row if file_row is None else file_row
            col = bregma_col if file_col is None else file_col
            print(f"  {key}: seeded from {path}")
            return row // 2, col // 2, dict(atlas)

    return bregma_row // 2, bregma_col // 2, dict(default_boxes)


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
            y_1, x_2, boxes = _seed_layout(key, args, default_boxes, bregma_row, bregma_col)
            sessions.append(Session(key=key, folder=folder, y_1=y_1, x_2=x_2,
                                    boxes=boxes, seed=(y_1, x_2, dict(boxes))))
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
        y_1, x_2, boxes = _seed_layout(key, args, default_boxes,
                                       args.bregma_row, args.bregma_col)
        sessions.append(Session(key=key, folder=str(folder), y_1=y_1, x_2=x_2,
                                boxes=boxes, seed=(y_1, x_2, dict(boxes))))
    return sessions


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------
class ROIEditor:
    """Matplotlib-based ROI/Bregma editor over a list of :class:`Session`."""

    HANDLE_PX = 1.6      # how close to a corner a click must be to resize
    NUDGE_BIG = 5        # shift+arrow step

    def __init__(self, sessions: list[Session], args: argparse.Namespace,
                 profile_name: str) -> None:
        self.sessions = sessions
        self.args = args
        self.profile_name = profile_name
        self.index = 0
        self.selected: str | None = None
        self.drag: dict[str, Any] | None = None
        self.clip = [1.0, 99.5]      # display percentiles
        self.message = ""

        import matplotlib
        matplotlib.use("TkAgg")
        # Matplotlib's own single-key shortcuts collide with almost every key this
        # editor uses (s=save figure, p=pan, left/right=nav, h/r=home, l/k=scales,
        # f=fullscreen, g=grid, o=zoom). Clear them rather than pick worse keys.
        for keymap in ("save", "pan", "home", "back", "forward", "zoom", "grid",
                       "grid_minor", "yscale", "xscale", "fullscreen", "quit_all"):
            matplotlib.rcParams[f"keymap.{keymap}"] = []
        import matplotlib.pyplot as plt

        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(8.5, 9.0))
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.20)
        self.fig.text(0.02, 0.015, HELP, fontsize=7.5, va="bottom", family="monospace")
        self.msg_text = self.fig.text(0.02, 0.155, "", fontsize=9, va="bottom",
                                      color="tab:blue", family="monospace")

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

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

    # -- drawing -------------------------------------------------------------
    def draw(self) -> None:
        session = self.session
        self._ensure_image(session)
        image = session.image
        bad = set(self._out_of_frame(session))

        self.ax.clear()
        lo, hi = np.percentile(image, self.clip)
        self.ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")

        from matplotlib.patches import Rectangle

        for label, box in session.boxes.items():
            r0, r1, c0, c1 = pixel_bounds(box, session.y_1, session.x_2)
            color = ("red" if label in bad else
                     "yellow" if label == self.selected else "cyan")
            # Pixel i spans [i-0.5, i+0.5] in imshow data coords, so an inclusive
            # r0..r1 box starts at r0-0.5 and is (r1-r0+1) tall.
            self.ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1,
                                        fill=False, edgecolor=color, linewidth=1.6))
            self.ax.text(c0 - 0.5, r0 - 1.2, label, color=color, fontsize=7.5,
                         ha="left", va="bottom")

        # Bregma: offset 0 is MATLAB index y_1 -> 0-based pixel y_1 - 1.
        self.ax.plot([session.x_2 - 1], [session.y_1 - 1], marker="+", markersize=16,
                     markeredgewidth=1.6, color="magenta")

        rows, cols = image.shape
        self.ax.set_title(
            f"[{self.index + 1}/{len(self.sessions)}] {session.key}"
            f"{'  *unsaved*' if session.dirty else ''}\n"
            f"{session.folder}\n"
            f"{rows}x{cols} grid | Bregma row={session.bregma_row} col={session.bregma_col} "
            f"(y_1={session.y_1}, x_2={session.x_2}) | profile {self.profile_name}",
            fontsize=9,
        )
        self.msg_text.set_text(self.message)
        self.msg_text.set_color("tab:red" if bad else "tab:blue")
        self.fig.canvas.draw_idle()

    def say(self, text: str) -> None:
        """Put a line in the window's status area and on stdout."""
        self.message = text
        print(text)

    # -- hit testing ---------------------------------------------------------
    def _hit(self, x: float, y: float):
        """``(label, corner)`` under the cursor, or None. corner=(row_side, col_side)."""
        session = self.session
        # Reverse order so the most recently drawn (topmost) box wins an overlap.
        for label in reversed(list(session.boxes)):
            r0, r1, c0, c1 = pixel_bounds(session.boxes[label], session.y_1, session.x_2)
            if not (c0 - 0.5 <= x <= c1 + 0.5 and r0 - 0.5 <= y <= r1 + 0.5):
                continue
            for row_side, ry in ((0, r0), (1, r1)):
                for col_side, cx in ((0, c0), (1, c1)):
                    if abs(y - ry) <= self.HANDLE_PX and abs(x - cx) <= self.HANDLE_PX:
                        return label, (row_side, col_side)
            return label, None
        return None

    # -- edits ---------------------------------------------------------------
    def _set_box(self, label: str, r0: int, r1: int, c0: int, c1: int) -> None:
        session = self.session
        session.boxes[label] = box_from_pixels(r0, r1, c0, c1, session.y_1, session.x_2)
        session.dirty = True

    def _nudge_box(self, dr: int, dc: int) -> None:
        if self.selected is None:
            self.say("No ROI selected -- click one first.")
            return
        session = self.session
        rows, cols = session.image.shape
        r0, r1, c0, c1 = pixel_bounds(session.boxes[self.selected], session.y_1, session.x_2)
        dr = _clamp(dr, -r0, rows - 1 - r1)          # keep the whole box in frame
        dc = _clamp(dc, -c0, cols - 1 - c1)
        self._set_box(self.selected, r0 + dr, r1 + dr, c0 + dc, c1 + dc)

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
        self._set_box(self.selected, r0, r1, c0, c1)

    def _move_bregma(self, y_1: int, x_2: int) -> None:
        """Move Bregma; boxes are offsets, so they all translate with it."""
        session = self.session
        rows, cols = session.image.shape
        session.y_1 = _clamp(y_1, 1, rows)
        session.x_2 = _clamp(x_2, 1, cols)
        session.dirty = True

    # -- mouse ---------------------------------------------------------------
    def on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button == 3:                       # right click -> set Bregma
            self._move_bregma(int(round(event.ydata)) + 1, int(round(event.xdata)) + 1)
            self.say(f"Bregma -> row={self.session.bregma_row}, col={self.session.bregma_col}")
            self.draw()
            return
        if event.button != 1:
            return

        hit = self._hit(event.xdata, event.ydata)
        if hit is None:
            self.selected = None
            self.draw()
            return
        label, corner = hit
        self.selected = label
        session = self.session
        self.drag = {
            "label": label,
            "corner": corner,
            "x": event.xdata,
            "y": event.ydata,
            "bounds": pixel_bounds(session.boxes[label], session.y_1, session.x_2),
        }
        self.draw()

    def on_motion(self, event) -> None:
        if self.drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        session = self.session
        rows, cols = session.image.shape
        r0, r1, c0, c1 = self.drag["bounds"]
        corner = self.drag["corner"]

        if corner is None:
            # Move: translate by the cursor delta, clamped so the box stays whole.
            dr = _clamp(int(round(event.ydata - self.drag["y"])), -r0, rows - 1 - r1)
            dc = _clamp(int(round(event.xdata - self.drag["x"])), -c0, cols - 1 - c1)
            self._set_box(self.drag["label"], r0 + dr, r1 + dr, c0 + dc, c1 + dc)
        else:
            # Resize: the grabbed corner follows the cursor, the opposite one is
            # the anchor; min() / max() stop the box inverting through it.
            y = _clamp(int(round(event.ydata)), 0, rows - 1)
            x = _clamp(int(round(event.xdata)), 0, cols - 1)
            row_side, col_side = corner
            nr0, nr1 = (min(y, r1), r1) if row_side == 0 else (r0, max(y, r0))
            nc0, nc1 = (min(x, c1), c1) if col_side == 0 else (c0, max(x, c0))
            self._set_box(self.drag["label"], nr0, nr1, nc0, nc1)
        self.draw()

    def on_release(self, event) -> None:
        self.drag = None

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
                               grid=session.image.shape, name=name, source=source)
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
            if with_bregma:
                other.y_1, other.x_2 = session.y_1, session.x_2
            other.dirty = True
        what = "boxes and Bregma" if with_bregma else "boxes (each keeps its Bregma)"
        self.say(f"Applied {session.key}'s {what} to all {len(self.sessions)} sessions "
                 f"-- press S to write them, or w for one shared file.")

    # -- keyboard ------------------------------------------------------------
    def on_key(self, event) -> None:
        key = event.key
        if key is None:
            return
        session = self.session

        if key in ("n", "p"):
            step = 1 if key == "n" else -1
            self.index = (self.index + step) % len(self.sessions)
            self.selected = None
            self.message = ""
        elif key == "s":
            self._save_current()
        elif key == "S":
            self._save_all()
        elif key == "w":
            self._write_shared()
        elif key in ("a", "A"):
            self._apply_to_all(with_bregma=(key == "A"))
        elif key == "r":
            y_1, x_2, boxes = session.seed
            session.y_1, session.x_2, session.boxes = y_1, x_2, dict(boxes)
            session.dirty = False
            self.say(f"{session.key}: reset to the starting layout.")
        elif key in ("+", "="):
            self._resize_box(+1)
        elif key == "-":
            self._resize_box(-1)
        elif key in ("[", "]"):
            # Widen / narrow the display stretch; pure cosmetics, no geometry.
            self.clip[1] = _clamp_float(self.clip[1] + (2.0 if key == "]" else -2.0),
                                        self.clip[0] + 1.0, 100.0)
            self.say(f"Contrast: {self.clip[0]:.1f}-{self.clip[1]:.1f} percentile")
        elif key == "h":
            print(HELP)
            self.say("Controls printed to the terminal.")
        elif key.endswith(("up", "down", "left", "right")):
            dr = -1 if key.endswith("up") else 1 if key.endswith("down") else 0
            dc = -1 if key.endswith("left") else 1 if key.endswith("right") else 0
            if key.startswith("ctrl+"):
                self._move_bregma(session.y_1 + dr, session.x_2 + dc)
                self.say(f"Bregma -> row={session.bregma_row}, col={session.bregma_col}")
            else:
                step = self.NUDGE_BIG if key.startswith("shift+") else 1
                self._nudge_box(dr * step, dc * step)
        else:
            return
        self.draw()

    def run(self) -> None:
        self.draw()
        self.plt.show()


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
        roi_set_dir=config["roi_set_dir"],
        shared_name=config["shared_name"],
        load_existing=bool(config["load_existing"]),
        start_from=config["start_from"],
    )


def main() -> None:
    args = build_runtime_args()
    profile = get_profile(args.profile)
    default_boxes = dict(profile.atlas)

    print(f"Profile   : {profile.name} ({profile.n_rois} ROIs: {profile.labels})")
    print(f"ROI sets  : {args.roi_set_dir}")
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
