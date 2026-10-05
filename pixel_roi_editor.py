"""Interactive ROI editor on the saved pixel dumps (after the pixel computation).

roi_editor.py places ROIs BEFORE the batch, on a raw anatomy preview, and the
batch then reduces every frame to those boxes. Once run_botox_batch.py has
written the per-pixel dumps (`pixel_data/<day>_<animal>/<t#>/`), boxes no longer
need to be fixed in advance: any rectangle inside the saved crop can be averaged
from the dumped pixels. This tool is for choosing those rectangles while LOOKING
AT THE COMPUTED SIGNAL, and it writes them in the same ROI-set format so the
pixel scripts use them in place of each dump's own set:

    python epileptic_by_area_animal_day_pixels.py --roi-set-dir roi_sets/pixel_selected
    python epileptic_by_active_pixels.py         --roi-set-dir roi_sets/pixel_selected
    python roi_pixel_connectivity.py              --roi-set-dir roi_sets/pixel_selected

Each of those reads `<roi-set-dir>/<day>_<animal>_<t#>.yaml` for every recording
it analyses, raises if one is missing (no silent fallback, CLAUDE.md 9.4), and
writes to an output folder suffixed `_roi_<folder name>`, so the default runs are
never overwritten. The run scripts do the computing; this editor never runs them.

What it shows
-------------
One page per recording. The image is the saved crop (76x87, Bregma-relative,
CLAUDE.md 9.15) drawn at its FULL-GRID position, so the axes and box offsets are
the same numbers roi_editor.py shows. Backgrounds, cycled with `b`:

    mean F      time-mean of the raw GCaMP dump (pixels_f_gcamp_full.npy): anatomy
    SD          temporal SD of the trace volume: where the signal moves
    seed r      Pearson r of every pixel with the SELECTED box's trace
    frame       one frame of the trace volume; drag the yellow line in the trace
                panel to choose it

Below the image, the selected box's mean trace (yellow) and its bilateral twin's
(cyan), from `trace_volume` (default the 20 s median dF/F cache, the signal the
detectors run on). They update while a box is dragged.

What it writes
--------------
`<out_dir>/<day>_<animal>_<t#>.yaml`, written by roi_editor.save_roi_set: the
dump's own grid, Bregma and downsample (the pixel scripts refuse a set whose
Bregma differs from the dump's), and the boxes as Bregma offsets. Box ORDER is
the trace/column order downstream. Bregma cannot be moved here: the pixels were
cropped around it, so moving it would only shift every box, which ctrl+arrows
already does.

Box names must put the hemisphere letter at the end of the part before the first
`_` (`S1L`, `M2R_alta`): that is how the detectors assign a hemisphere
(epileptic_by_area_animal_day_pixels.roi_categories). The bilateral twin of a
name swaps that letter.

Controls
--------
    click a box              select it (its trace is plotted)
    drag inside / a corner   move / resize it
    arrows, shift+arrows     nudge selected box 1 / 5 px
    ctrl+arrows (+shift)     move EVERY box 1 / 5 px (left/right refused under mirror lock)
    + / -                    grow / shrink selected box
    double-click, i          add a box (named in a dialog; with mirror lock its twin too)
    Delete / Backspace       delete selected box (and its twin under mirror lock)
    e                        rename selected box (and its twin under mirror lock)
    m                        toggle mirror lock
    b / B                    next / previous background      [ / ]  contrast
    n / p, the drop-down     next / previous / any recording
    A                        this page becomes the TEMPLATE: its boxes are placed on EVERY
                             recording by that recording's Bregma and Lambda (see below)
    a                        the same, only on the t# of this animal-day
    s / S / ctrl+s           save this page / every page / only the modified pages
    r                        reset this page to how it opened     h  help    q  quit

Pages start from `<out_dir>/<key>.yaml` when it exists (resume), else from
`<seed_dir>/<key>.yaml` when `seed_dir` is set, else from the dump's own ROI set.

Select once, use everywhere
---------------------------
Draw the ROIs on ONE recording, press `A`, check a few other pages, press `S`.
Every recording's Bregma and Lambda were already placed with roi_editor.py
(`bregma_row/col`, `lambda_row_offset` in its roi_sets/rebuilt set). A box is
stored as an offset from Bregma, so the template transfers as

    offset on recording k = template offset x (Lambda_k / Lambda_template)

scaled about Bregma by roi_editor.scale_boxes (box sizes fixed unless
`lambda_scales_box_size`; mirrored pairs stay exactly mirrored; the same Lambda is
the identity). Saving also writes `<out_dir>/_template.yaml` (the template boxes
and the Lambda they were drawn at). To regenerate every set from it without the
window, e.g. after new recordings are dumped:

    python pixel_roi_editor.py --apply-template roi_sets/pixel_selected/_template.yaml

That checks every recording first and writes nothing if a scaled box would leave
a crop. Every recording then carries the same labels in the same order, which
roi_pixel_connectivity.py needs to average maps. A downstream run needs a file
for EVERY recording it analyses.

Run it (`--no-capture-output`, or prints are held until the window closes):
    conda run --no-capture-output -n letizia python pixel_roi_editor.py
    conda run --no-capture-output -n letizia python pixel_roi_editor.py --recordings "260611_PV5/*"
    conda run --no-capture-output -n letizia python pixel_roi_editor.py --trace-volume pixels_dff_full.npy
    conda run --no-capture-output -n letizia python pixel_roi_editor.py --apply-template roi_sets/pixel_selected/_template.yaml

Qt is imported inside PixelROIEditor._build_ui, so the helpers here import on a
headless machine (tests/test_pixel_roi_editor.py). Errors are left to surface; a
box outside the crop is the one condition that is refused with a message instead
(drawn red, save blocked), as in roi_editor.py.
"""

from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse
import fnmatch
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from epileptic_by_area_animal_day_pixels import crop_slices, recording_roi_set
import yaml

from roi_editor import box_from_pixels, mirror_box, pixel_bounds, save_roi_set, scale_boxes
from roi_pixel_connectivity import read_meta
from wfci import Box

REPO_ROOT = Path(__file__).resolve().parent

# Edit this section to run the editor without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "pixel_root": REPO_ROOT / "pixel_data",  # Contains <day>_<animal>/<t#>/pixels_meta_full.npz
    "out_dir": REPO_ROOT / "roi_sets/pixel_selected",  # Where ROI sets are written (and resumed from).
    # None: each page starts from its dump's own ROI set (roi_sets/rebuilt). A folder:
    # start from <seed_dir>/<day>_<animal>_<t#>.yaml instead (read-only; saves go to out_dir).
    "seed_dir": None,
    # Per-pixel volume the traces, SD, seed-r and frame views are computed from.
    # pixels_median_dff_<N>s_full.npy (cache_median_dff.py) or pixels_dff_full.npy.
    "trace_volume": "pixels_median_dff_20s_full.npy",
    # None: every dump. Otherwise fnmatch patterns on "<day>_<animal>/<t#>", e.g. ["260611_PV5/*"].
    "recordings": None,
    "sampling_rate_hz": 10.0,  # Per-channel frame rate, for the trace time axis.
    "new_box_span_px": 5,  # A new box is span+1 px square (5 -> 6x6, the cortex22 box size).
    # True: editing one box of an L/R pair rewrites its twin as the exact mirror about
    # Bregma (as in roi_editor.py). `m` toggles it in the window.
    "mirror_lock": True,
    # Transfer to other recordings (`A`, `a`, apply_template) scales every box's
    # offset from Bregma by target Lambda / template Lambda. False keeps each box's
    # SIZE (same pixel count behind every ROI mean, as roi_editor.py's default);
    # True scales sizes too.
    "lambda_scales_box_size": False,
    # None: open the window. A template YAML (the <out_dir>/_template.yaml written
    # by `A` + save): no window; write every selected recording's set from it and exit.
    "apply_template": None,
    "prefer_cli_args": True,  # False: ignore CLI arguments and use this block.
}

TEMPLATE_NAME = "_template.yaml"  # never matches a <day>_<animal>_<t#>.yaml key

BACKGROUNDS = ("mean F", "SD", "seed r", "frame")

# Hemisphere rule of the detectors: side letter at the end of the prefix before "_".
ROI_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]+[LR](_[A-Za-z0-9_]+)?$")

HELP = (
    "ONE BOX: click to select, drag to move, drag a corner to resize, "
    "arrows / shift+arrows nudge 1 / 5 px, + / - grow / shrink\n"
    "ALL BOXES: ctrl+arrows / ctrl+shift+arrows move 1 / 5 px    "
    "ADD: double-click or i    DELETE: Del    RENAME: e\n"
    "m: mirror lock    b / B: background (mean F, SD, seed r, frame)    [ / ]: contrast    "
    "frame view: drag the yellow line in the trace panel\n"
    "n / p: next / prev recording    A: THIS PAGE = TEMPLATE, placed on EVERY recording by its Bregma + Lambda    "
    "a: same, this animal-day only\n"
    "s: save page    S: save all    ctrl+s: save modified    r: reset page    h: help    q: quit"
)


# ---------------------------------------------------------------------------
# Geometry and naming helpers (no Qt; tested in tests/test_pixel_roi_editor.py)
# ---------------------------------------------------------------------------
def is_valid_roi_name(label: str) -> bool:
    """True when the detectors can read a hemisphere from ``label`` (see module docstring)."""
    return bool(ROI_NAME_PATTERN.match(label))


def hemisphere_twin(label: str) -> str | None:
    """Bilateral counterpart of ``label`` by the detectors' rule, or None without a side letter.

    `M2L_alta -> M2R_alta`, `RSL_alta -> RSR_alta`: the letter swapped is the last
    one of the prefix, so the leading `R` of `RSL_alta` is never taken for the side.
    """
    prefix, separator, suffix = label.partition("_")
    if prefix[-1:] not in ("L", "R"):
        return None
    return prefix[:-1] + ("R" if prefix[-1] == "L" else "L") + separator + suffix


def boxes_outside_crop(boxes: dict[str, Box], y_1: int, x_2: int, region) -> list[str]:
    """Labels whose box does not lie wholly inside the saved crop (the pixels that exist)."""
    row_0, row_1, col_0, col_1 = map(int, region)
    outside = []
    for label, box in boxes.items():
        r0, r1, c0, c1 = pixel_bounds(box, y_1, x_2)
        if r0 < row_0 or c0 < col_0 or r1 >= row_1 or c1 >= col_1:
            outside.append(label)
    return outside


def cross_hemisphere_overlaps(boxes: dict[str, Box]) -> list[tuple[str, str]]:
    """(left, right) box pairs that share pixels.

    epileptic_by_active_pixels.py raises on these (a pixel would count for both
    hemispheres); the ROI-mean scripts do not care. So the editor warns, not refuses.
    """
    sides = {label: label.partition("_")[0][-1:] for label in boxes}
    overlaps = []
    for left in (label for label in boxes if sides[label] == "L"):
        for right in (label for label in boxes if sides[label] == "R"):
            a, b = boxes[left], boxes[right]
            if (a.row_start <= b.row_end and b.row_start <= a.row_end
                    and a.col_start <= b.col_end and b.col_start <= a.col_end):
                overlaps.append((left, right))
    return overlaps


def box_trace(volume: np.ndarray, box: Box, y_1: int, x_2: int, region) -> np.ndarray:
    """Mean trace of one box over the crop volume (n_time, rows, cols) -> (n_time,) float64."""
    row_slice, col_slice = crop_slices(vars(box), y_1, x_2, region)
    return np.nanmean(volume[:, row_slice, col_slice], axis=(1, 2), dtype=np.float64)


def recording_key(folder: Path) -> str:
    """`pixel_data/260611_PV5/t1` -> `260611_PV5_t1`, the ROI-set file stem the pixel scripts read."""
    return f"{folder.parent.name}_{folder.name}"


def discover_recordings(pixel_root: Path, patterns: list[str] | None) -> list[Path]:
    """Dump folders under ``pixel_root``, optionally filtered by `<day>_<animal>/<t#>` patterns.

    A pattern that matches nothing is a typo and raises, like `exclude_recordings`
    in the pixel scripts.
    """
    folders = sorted(path.parent for path in Path(pixel_root).glob("*/*/pixels_meta_full.npz"))
    if not folders:
        raise FileNotFoundError(f"No full pixel dumps found under {pixel_root}")
    if patterns is None:
        return folders
    names = {folder: f"{folder.parent.name}/{folder.name}" for folder in folders}
    unmatched = [p for p in patterns if not any(fnmatch.fnmatch(n, p.replace("\\", "/")) for n in names.values())]
    if unmatched:
        raise ValueError(f"Recording patterns match no dump under {pixel_root}: {unmatched}")
    return [folder for folder in folders
            if any(fnmatch.fnmatch(names[folder], p.replace("\\", "/")) for p in patterns)]


def save_page_roi_set(out_dir: Path, key: str, boxes: dict[str, Box], meta: dict[str, Any],
                      lambda_offset: int | None, source: str) -> Path:
    """Write one recording's ROI set with the dump's own grid, Bregma and downsample.

    Refuses boxes outside the crop: the pixel scripts would raise on them anyway,
    but only after the batch has started.
    """
    outside = boxes_outside_crop(boxes, int(meta["y_1"]), int(meta["x_2"]), meta["region"])
    if outside:
        raise ValueError(f"{key}: boxes outside the saved crop {meta['region']}: {outside}")
    if not boxes:
        raise ValueError(f"{key}: no boxes to save")
    return save_roi_set(Path(out_dir) / f"{key}.yaml", boxes, int(meta["bregma_row"]),
                        int(meta["bregma_col"]), tuple(meta["grid"]), name=key, source=source,
                        lambda_offset=lambda_offset, downsample=float(meta["downsample"]))


# ---------------------------------------------------------------------------
# One selection for every recording: transfer by Bregma and Lambda
# ---------------------------------------------------------------------------
def dump_lambda(meta: dict[str, Any]) -> int:
    """Bregma -> Lambda distance (final-grid rows) of the recording, from its own ROI set.

    That set was drawn with roi_editor.py, where Bregma and Lambda were placed on
    this animal; the dump's crop is centred on the same Bregma.
    """
    path, atlas = recording_roi_set(meta, None)
    value = atlas.get("lambda_row_offset")
    if value is None or int(value) <= 0:
        raise ValueError(f"{path} has no positive lambda_row_offset; cannot scale a template to it")
    return int(value)


def transfer_boxes(boxes: dict[str, Box], source_lambda: int, target_lambda: int,
                   scale_size: bool = False) -> dict[str, Box]:
    """Boxes drawn at ``source_lambda`` placed on a recording whose Lambda is ``target_lambda``.

    Offsets are already relative to Bregma, so only the scale changes: every
    offset is multiplied by target / source about Bregma (roi_editor.scale_boxes,
    which keeps mirrored L/R pairs exactly mirrored). The same Lambda is exactly
    the identity. Always transfer from the template, never from a transferred
    copy, or the rounding would accumulate.
    """
    if source_lambda <= 0 or target_lambda <= 0:
        raise ValueError(f"Lambda distances must be positive: {source_lambda}, {target_lambda}")
    return scale_boxes(boxes, target_lambda / source_lambda, scale_size=scale_size)


def save_template(out_dir: Path, boxes: dict[str, Box], meta: dict[str, Any], key: str,
                  template_lambda: int, scale_size: bool) -> Path:
    """Write the one selection every recording is derived from, with the Lambda it was drawn at."""
    return save_roi_set(Path(out_dir) / TEMPLATE_NAME, boxes, int(meta["bregma_row"]),
                        int(meta["bregma_col"]), tuple(meta["grid"]), name=f"template from {key}",
                        source=(f"pixel_roi_editor.py template drawn on {key}; every recording's set = "
                                f"these offsets x (its lambda_row_offset / {template_lambda}), "
                                f"box sizes {'scaled' if scale_size else 'fixed'}"),
                        lambda_offset=template_lambda, downsample=float(meta["downsample"]))


def load_template(path: Path) -> tuple[dict[str, Box], int]:
    """(boxes, lambda) of a template written by :func:`save_template` (or any ROI set with a Lambda)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("lambda_row_offset") is None:
        raise ValueError(f"{path} has no lambda_row_offset: a template must say the Lambda it was drawn at")
    return {label: Box(**spec) for label, spec in data["boxes"].items()}, int(data["lambda_row_offset"])


def apply_template(template_boxes: dict[str, Box], template_lambda: int, folders: list[Path],
                   out_dir: Path, scale_size: bool, source: str) -> list[Path]:
    """Write every recording's ROI set from the template; nothing is written if one does not fit.

    Checking all recordings before writing any means a failure never leaves a
    half-updated folder that the pixel scripts would read without complaint.
    """
    planned, outside = [], {}
    for folder in folders:
        meta = read_meta(folder / "pixels_meta_full.npz")
        target_lambda = dump_lambda(meta)
        boxes = transfer_boxes(template_boxes, template_lambda, target_lambda, scale_size)
        bad = boxes_outside_crop(boxes, int(meta["y_1"]), int(meta["x_2"]), meta["region"])
        if bad:
            outside[recording_key(folder)] = bad
        planned.append((recording_key(folder), boxes, meta, target_lambda))
    if outside:
        raise ValueError(f"{len(outside)} recording(s) would get boxes outside their saved crop after "
                         f"Lambda scaling; nothing written: {dict(list(outside.items())[:10])}")
    return [save_page_roi_set(out_dir, key, boxes, meta, target_lambda,
                              f"{source}; scale {target_lambda}/{template_lambda}")
            for key, boxes, meta, target_lambda in planned]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@dataclass
class Page:
    """One recording: its dump folder and the boxes being edited.

    Geometry is filled in by `load` on first use, so opening the editor over all
    545 dumps does not read 545 metadata files up front.
    """

    folder: Path
    key: str
    meta: dict[str, Any] | None = None
    boxes: dict[str, Box] = field(default_factory=dict)
    start_boxes: dict[str, Box] = field(default_factory=dict)  # what `r` returns to
    start_source: str = ""
    # Bregma -> Lambda of THIS recording (from its roi_editor.py set): the scale its
    # boxes are at, written to the saved set and used to transfer templates.
    own_lambda: int | None = None
    dirty: bool = False

    @property
    def unit(self) -> str:
        return self.folder.parent.name

    def load(self, out_dir: Path, seed_dir: Path | None) -> None:
        """Read the dump metadata and the starting boxes (saved > seed > dump's own)."""
        if self.meta is not None:
            return
        self.meta = read_meta(self.folder / "pixels_meta_full.npz")
        saved_path = Path(out_dir) / f"{self.key}.yaml"
        if saved_path.is_file():
            path, atlas = recording_roi_set(self.meta, out_dir)
            self.start_source = f"resumed from {path}"
        elif seed_dir is not None:
            path, atlas = recording_roi_set(self.meta, seed_dir)
            self.start_source = f"seeded from {path}"
        else:
            path, atlas = recording_roi_set(self.meta, None)
            self.start_source = f"dump's own set {path}"
        self.boxes = {label: Box(**spec) for label, spec in atlas["boxes"].items()}
        self.start_boxes = dict(self.boxes)
        self.own_lambda = dump_lambda(self.meta)

    @property
    def y_1(self) -> int:
        return int(self.meta["y_1"])

    @property
    def x_2(self) -> int:
        return int(self.meta["x_2"])

    @property
    def region(self) -> list[int]:
        return [int(v) for v in self.meta["region"]]


@dataclass
class PageData:
    """The current page's pixels and derived images (only one page is held at a time)."""

    volume: np.ndarray        # (n_time, rows, cols) float32, the trace volume
    mean_f: np.ndarray        # (rows, cols) time-mean raw GCaMP
    sd: np.ndarray            # (rows, cols) temporal SD of the trace volume
    frame_levels: tuple[float, float]
    centred: np.ndarray | None = None      # (n_time, rows*cols) float64, built on first seed-r view
    pixel_norm: np.ndarray | None = None   # (rows*cols,)


def load_page_data(folder: Path, trace_volume: str) -> PageData:
    """Read the trace volume into RAM and derive the static background images."""
    # volume: (n_time, rows, cols) float16 on disk -> float32 (~79 MB for 2980 x 76 x 87)
    volume = np.asarray(np.load(folder / trace_volume, mmap_mode="r"), dtype=np.float32)
    raw_f = np.load(folder / "pixels_f_gcamp_full.npy", mmap_mode="r")
    if raw_f.shape != volume.shape:
        raise ValueError(f"{trace_volume} {volume.shape} and pixels_f_gcamp_full.npy {raw_f.shape} differ: {folder}")
    mean_f = raw_f.mean(axis=0, dtype=np.float64)
    sd = np.nanstd(volume, axis=0, dtype=np.float64)
    # Fixed levels for the frame view, so contrast does not jump from frame to frame.
    frame_levels = tuple(float(v) for v in np.nanpercentile(volume[::10], [1.0, 99.5]))
    return PageData(volume=volume, mean_f=mean_f, sd=sd, frame_levels=frame_levels)


def seed_r_map(data: PageData, trace: np.ndarray) -> np.ndarray:
    """Pearson r of every pixel with ``trace`` -> (rows, cols).

    Same explicit formula as roi_pixel_connectivity.pixel_correlation, via einsum
    so no BLAS call is made (CLAUDE.md 9.11/9.17). The centred pixels are cached
    on the page, so only the seed side is recomputed per edit.
    """
    n_time, n_row, n_col = data.volume.shape
    if data.centred is None:
        pixels = data.volume.reshape(n_time, -1).astype(np.float64)
        data.centred = pixels - pixels.mean(axis=0, keepdims=True)
        data.pixel_norm = np.sqrt(np.einsum("tp,tp->p", data.centred, data.centred))
    seed_centred = trace - trace.mean()
    cross = np.einsum("t,tp->p", seed_centred, data.centred)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cross / (np.sqrt(seed_centred @ seed_centred) * data.pixel_norm)
    return r.reshape(n_row, n_col)


# ---------------------------------------------------------------------------
# The editor window
# ---------------------------------------------------------------------------
class PixelROIEditor:
    """pyqtgraph editor over a list of :class:`Page` (same widgets as roi_editor.ROIEditor).

    `page.boxes` is the single source of truth; the RectROIs only mirror it. Qt
    emits the same signal for a user drag and for our own setPos, so `_syncing`
    marks programmatic moves, exactly as in roi_editor.py.
    """

    NUDGE_BIG = 5
    HANDLE_SIZE = 7

    def __init__(self, pages: list[Page], args: argparse.Namespace) -> None:
        self.pages = pages
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.seed_dir = None if args.seed_dir is None else Path(args.seed_dir)
        self.index = 0
        self.selected: str | None = None
        self.mirror_lock = bool(args.mirror_lock)
        self.background = 0          # index into BACKGROUNDS
        self.clip = [1.0, 99.5]      # display percentiles
        self.frame = 0               # frame shown by the "frame" background
        self.message = ""
        self.data: PageData | None = None
        # Set by `A`/`a`: the page whose boxes every other page was derived from.
        self.template: dict[str, Any] | None = None
        self._syncing = False
        self._rois: dict[str, Any] = {}
        self._labels: dict[str, Any] = {}
        self._build_ui()

    # -- Qt construction -----------------------------------------------------
    def _build_ui(self) -> None:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        # row-major: images indexed [row, col] like NumPy (see roi_editor._build_ui).
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
        self.pg, self.QtCore, self.QtWidgets = pg, QtCore, QtWidgets
        self.app = pg.mkQApp("Pixel ROI editor")

        self.window = QtWidgets.QMainWindow()
        self.window.resize(1000, 1050)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.header)

        bar = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<< Previous")
        self.page_combo = QtWidgets.QComboBox()
        self.page_combo.addItems([page.key for page in self.pages])
        self.btn_next = QtWidgets.QPushButton("Next >>")
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save_modified = QtWidgets.QPushButton("Save modified")
        self.btn_background = QtWidgets.QPushButton("Background")
        for widget in (self.btn_prev, self.page_combo, self.btn_next, self.btn_background,
                       self.btn_save, self.btn_save_modified):
            widget.setFocusPolicy(QtCore.Qt.NoFocus)  # keep arrow keys reaching the window
            bar.addWidget(widget)
        layout.addLayout(bar)
        self.btn_prev.clicked.connect(lambda: self._go_to(self.index - 1))
        self.btn_next.clicked.connect(lambda: self._go_to(self.index + 1))
        self.page_combo.currentIndexChanged.connect(self._go_to)
        self.btn_background.clicked.connect(lambda: self._cycle_background(+1))
        self.btn_save.clicked.connect(lambda: self._save_pages([self.page], "this page"))
        self.btn_save_modified.clicked.connect(self._save_modified)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setFocusPolicy(QtCore.Qt.NoFocus)
        layout.addWidget(self.graphics, stretch=1)

        self.plot = self.graphics.addPlot(row=0, col=0)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setLabel("left", "row (pixels, full 128 grid)")
        self.plot.setLabel("bottom", "col (pixels, full 128 grid)")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)

        # Bregma is fixed (the crop was cut around it), so its marker is not movable.
        self.bregma_marker = pg.TargetItem(pos=(0, 0), size=16, movable=False,
                                           pen=pg.mkPen("m", width=2), label=None)
        self.bregma_marker.setZValue(100)
        self.plot.addItem(self.bregma_marker)
        # The midline through Bregma: the axis every mirrored pair is reflected about.
        self.midline = pg.InfiniteLine(angle=90, movable=False,
                                       pen=pg.mkPen("m", width=1, style=QtCore.Qt.DashLine))
        self.plot.addItem(self.midline)
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

        self.trace_plot = self.graphics.addPlot(row=1, col=0)
        self.trace_plot.setLabel("bottom", "time (s)")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.2)
        self.twin_curve = self.trace_plot.plot(pen=pg.mkPen("c", width=1))
        self.trace_curve = self.trace_plot.plot(pen=pg.mkPen("y", width=1))
        # Frame cursor: drag it to choose the frame shown by the "frame" background.
        self.frame_line = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen("y", width=2))
        self.frame_line.sigPositionChanged.connect(self._on_frame_moved)
        self.trace_plot.addItem(self.frame_line)
        self.graphics.ci.layout.setRowStretchFactor(0, 3)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)

        self.msg_label = QtWidgets.QLabel()
        self.msg_label.setTextFormat(QtCore.Qt.PlainText)
        self.msg_label.setWordWrap(True)
        layout.addWidget(self.msg_label)
        help_label = QtWidgets.QLabel(HELP)
        help_label.setTextFormat(QtCore.Qt.PlainText)
        help_label.setStyleSheet("color: #555; font-family: monospace; font-size: 11px;")
        layout.addWidget(help_label)
        self.window.setCentralWidget(central)

        self.window.keyPressEvent = self._on_key            # type: ignore[method-assign]
        self.window.closeEvent = self._on_close              # type: ignore[method-assign]
        self.window.setFocusPolicy(QtCore.Qt.StrongFocus)

    # -- state ---------------------------------------------------------------
    @property
    def page(self) -> Page:
        return self.pages[self.index]

    def _load(self, page: Page) -> None:
        page.load(self.out_dir, self.seed_dir)

    def say(self, text: str) -> None:
        self.message = text
        print(text, flush=True)
        self._refresh_status()

    def _popup(self, text: str, warn: bool = False) -> None:
        box = self.QtWidgets.QMessageBox.warning if warn else self.QtWidgets.QMessageBox.information
        box(self.window, "Pixel ROI editor", text)

    # -- page switching ------------------------------------------------------
    def _go_to(self, index: int) -> None:
        index = max(0, min(len(self.pages) - 1, int(index)))
        if index == self.index and self.data is not None:
            return
        self.index = index
        self._show_page()

    def _show_page(self) -> None:
        """Load the current page's pixels and rebuild every item on it."""
        page = self.page
        print(f"Loading {page.folder} ({self.args.trace_volume})", flush=True)
        self._load(page)
        self.data = load_page_data(page.folder, self.args.trace_volume)
        self.frame = min(self.frame, self.data.volume.shape[0] - 1)
        if self.selected not in page.boxes:
            self.selected = next(iter(page.boxes), None)

        self._syncing = True
        self.page_combo.blockSignals(True)
        self.page_combo.setCurrentIndex(self.index)
        self.page_combo.blockSignals(False)
        for item in list(self._rois.values()) + list(self._labels.values()):
            self.plot.removeItem(item)
        self._rois.clear()
        self._labels.clear()
        for label in page.boxes:
            self._add_item(label)
        self.bregma_marker.setPos(page.x_2 - 0.5, page.y_1 - 0.5)
        self.midline.setPos(page.x_2 - 0.5)
        row_0, row_1, col_0, col_1 = page.region
        # The crop is drawn at its full-grid position, so pixel_bounds applies unchanged.
        # One crop pixel is one grid pixel, so a pure translation places it. Not setRect:
        # that scales by rect / current image size, and on the first page no image is set
        # yet (pyqtgraph then uses size 1), which blew each pixel up to the whole region.
        self.image_item.resetTransform()
        self.image_item.setPos(col_0, row_0)
        self.plot.setRange(xRange=(col_0, col_1), yRange=(row_0, row_1), padding=0.02)
        n_time = self.data.volume.shape[0]
        self.frame_line.setBounds((0, (n_time - 1) / self.args.sampling_rate_hz))
        self.frame_line.setPos(self.frame / self.args.sampling_rate_hz)
        self._syncing = False

        self.say(f"{page.key}: {len(page.boxes)} boxes, {page.start_source}")
        self._refresh_background()
        self._refresh_trace()
        self._restyle()

    # -- items ---------------------------------------------------------------
    def _add_item(self, label: str) -> None:
        """One snapped RectROI + its text label for ``label`` (see roi_editor._make_roi)."""
        pg, page = self.pg, self.page
        r0, r1, c0, c1 = pixel_bounds(page.boxes[label], page.y_1, page.x_2)
        roi = pg.RectROI(pos=(c0, r0), size=(c1 - c0 + 1, r1 - r0 + 1),
                         pen=pg.mkPen("c", width=2), hoverPen=pg.mkPen("y", width=2),
                         translateSnap=True, scaleSnap=True, snapSize=1.0,
                         rotatable=False, resizable=True, removable=False)
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        for handle in roi.getHandles():
            handle.radius = self.HANDLE_SIZE
            handle.buildPath()
        # pyqtgraph ROIs ignore clicks unless told otherwise; needed for click-to-select.
        roi.setAcceptedMouseButtons(self.QtCore.Qt.MouseButton.LeftButton)
        roi.sigRegionChanged.connect(lambda _roi, lbl=label: self._on_roi_changed(lbl, final=False))
        roi.sigRegionChangeFinished.connect(lambda _roi, lbl=label: self._on_roi_changed(lbl, final=True))
        roi.sigClicked.connect(lambda _roi, _ev, lbl=label: self._select(lbl))
        self.plot.addItem(roi)
        self._rois[label] = roi
        text = pg.TextItem(label, color="c", anchor=(0, 1))
        self.plot.addItem(text)
        self._labels[label] = text

    def _remove_item(self, label: str) -> None:
        self.plot.removeItem(self._rois.pop(label))
        self.plot.removeItem(self._labels.pop(label))

    def _sync_items(self, labels=None) -> None:
        """Push ``page.boxes`` onto the RectROIs (all, or only ``labels``)."""
        page = self.page
        self._syncing = True
        for label in (page.boxes if labels is None else labels):
            r0, r1, c0, c1 = pixel_bounds(page.boxes[label], page.y_1, page.x_2)
            self._rois[label].setPos(c0, r0, finish=False)
            self._rois[label].setSize((c1 - c0 + 1, r1 - r0 + 1), finish=False)
        self._syncing = False
        self._restyle()

    def _restyle(self) -> None:
        """Colour boxes (red = outside crop, yellow = selected) and refresh the texts."""
        page = self.page
        outside = set(boxes_outside_crop(page.boxes, page.y_1, page.x_2, page.region))
        for label, roi in self._rois.items():
            colour = "r" if label in outside else "y" if label == self.selected else "c"
            roi.setPen(self.pg.mkPen(colour, width=3 if label == self.selected else 2))
            r0, _r1, c0, _c1 = pixel_bounds(page.boxes[label], page.y_1, page.x_2)
            self._labels[label].setColor(colour)
            self._labels[label].setPos(c0, r0)
        self._refresh_header()
        self._refresh_status()

    def _refresh_header(self) -> None:
        page = self.page
        n_dirty = sum(p.dirty for p in self.pages)
        self.btn_save_modified.setText(f"Save modified ({n_dirty})")
        self.btn_save_modified.setEnabled(n_dirty > 0)
        self.btn_background.setText(f"Background: {BACKGROUNDS[self.background]}")
        self.header.setText(
            f"[{self.index + 1}/{len(self.pages)}] {page.key}  group {page.meta['group']}"
            f"{'  *unsaved*' if page.dirty else ''}\n"
            f"{page.folder}  |  traces: {self.args.trace_volume}\n"
            f"crop rows {page.region[0]}..{page.region[1] - 1}, cols {page.region[2]}..{page.region[3] - 1} | "
            f"Bregma y_1={page.y_1} x_2={page.x_2} | Lambda +{page.own_lambda} px"
            + (f" = {page.own_lambda / self.template['lambda_']:.3f}x template {self.template['key']}"
               if self.template is not None else "")
            + f" | {len(page.boxes)} ROIs | "
            f"mirror lock {'ON' if self.mirror_lock else 'OFF'} | saves to {self.out_dir}")
        self.window.setWindowTitle(f"Pixel ROI editor -- {page.key}")

    def _refresh_status(self) -> None:
        page = self.page
        outside = boxes_outside_crop(page.boxes, page.y_1, page.x_2, page.region)
        overlaps = cross_hemisphere_overlaps(page.boxes)
        notes = []
        if outside:
            notes.append(f"{len(outside)} box(es) OUTSIDE THE CROP {sorted(outside)}: saving refused")
        if overlaps:
            notes.append(f"L/R boxes overlap {overlaps}: epileptic_by_active_pixels.py will refuse this set")
        self.msg_label.setText(self.message + "".join(f"   |   {note}" for note in notes))
        colour = "#d62728" if outside else "#ff7f0e" if overlaps else "#1f77b4"
        self.msg_label.setStyleSheet(f"color: {colour}; font-family: monospace;")

    # -- background and trace ------------------------------------------------
    def _selected_trace(self, label: str | None) -> np.ndarray | None:
        """Trace of ``label``, or None when there is none or the box leaves the crop."""
        page = self.page
        if label is None or label not in page.boxes:
            return None
        if boxes_outside_crop({label: page.boxes[label]}, page.y_1, page.x_2, page.region):
            return None
        return box_trace(self.data.volume, page.boxes[label], page.y_1, page.x_2, page.region)

    def _refresh_trace(self) -> None:
        page = self.page
        n_time = self.data.volume.shape[0]
        time_s = np.arange(n_time) / self.args.sampling_rate_hz
        trace = self._selected_trace(self.selected)
        twin = hemisphere_twin(self.selected) if self.selected else None
        twin_trace = self._selected_trace(twin) if twin in page.boxes else None
        if trace is None:
            self.trace_curve.setData([], [])
        else:
            self.trace_curve.setData(time_s, trace)
        if twin_trace is None:
            self.twin_curve.setData([], [])
        else:
            self.twin_curve.setData(time_s, twin_trace)
        title = "no box selected" if self.selected is None else (
            f"{self.selected} (yellow)" + (f" vs {twin} (cyan)" if twin_trace is not None else "")
            + ("  -- outside the crop" if trace is None else ""))
        self.trace_plot.setTitle(f"{title}   [{self.args.trace_volume}]")

    def _refresh_background(self) -> None:
        mode = BACKGROUNDS[self.background]
        if mode == "mean F":
            image, colormap = self.data.mean_f, None
        elif mode == "SD":
            image, colormap = self.data.sd, None
        elif mode == "frame":
            image, colormap = self.data.volume[self.frame], "viridis"
        else:
            trace = self._selected_trace(self.selected)
            if trace is None:
                image = np.full(self.data.sd.shape, np.nan)
            else:
                image = seed_r_map(self.data, trace)
            colormap = "viridis"
        self.image_item.setImage(image, autoLevels=False)
        if colormap is None:
            self.image_item.setLookupTable(None)
        else:
            self.image_item.setLookupTable(self.pg.colormap.get(colormap).getLookupTable())
        if mode == "frame":
            levels = self.data.frame_levels
        elif mode == "seed r":
            levels = (0.0, 1.0)  # without GSR almost every r is positive (roi_pixel_connectivity)
        else:
            levels = tuple(np.nanpercentile(image, self.clip))
        self.image_item.setLevels(levels)
        if mode == "frame":
            self.plot.setTitle(f"frame {self.frame} ({self.frame / self.args.sampling_rate_hz:.1f} s)")
        elif mode == "seed r":
            self.plot.setTitle(f"Pearson r with {self.selected} (0..1)")
        else:
            self.plot.setTitle(mode)

    def _cycle_background(self, step: int) -> None:
        self.background = (self.background + step) % len(BACKGROUNDS)
        self._refresh_background()
        self._refresh_header()

    def _on_frame_moved(self) -> None:
        if self._syncing or self.data is None:
            return
        n_time = self.data.volume.shape[0]
        self.frame = max(0, min(n_time - 1, int(round(self.frame_line.value() * self.args.sampling_rate_hz))))
        if BACKGROUNDS[self.background] == "frame":
            self._refresh_background()

    # -- edits ---------------------------------------------------------------
    def _select(self, label: str | None) -> None:
        self.selected = label
        self._refresh_trace()
        if BACKGROUNDS[self.background] == "seed r":
            self._refresh_background()
        self._restyle()

    def _mirror(self, label: str) -> str | None:
        """Under mirror lock, rewrite ``label``'s twin as its exact mirror; return the twin."""
        twin = hemisphere_twin(label)
        if not self.mirror_lock or twin not in self.page.boxes:
            return None
        self.page.boxes[twin] = mirror_box(self.page.boxes[label])
        return twin

    def _on_roi_changed(self, label: str, final: bool) -> None:
        """A box is being dragged (final=False) or was released (final=True)."""
        if self._syncing:
            return
        page = self.page
        roi = self._rois[label]
        c0, r0 = int(round(roi.pos().x())), int(round(roi.pos().y()))
        width, height = max(1, int(round(roi.size().x()))), max(1, int(round(roi.size().y())))
        page.boxes[label] = box_from_pixels(r0, r0 + height - 1, c0, c0 + width - 1, page.y_1, page.x_2)
        page.dirty = True
        twin = self._mirror(label)
        if twin:
            self._sync_items([twin])
        self.selected = label
        # The trace is a cheap box mean, so it follows the drag; seed r waits for the release.
        self._refresh_trace()
        if final and BACKGROUNDS[self.background] == "seed r":
            self._refresh_background()
        self._restyle()

    def _edited(self, text: str) -> None:
        """Common tail of a keyboard edit to the selected box."""
        self.page.dirty = True
        twin = self._mirror(self.selected)
        self._sync_items()
        self._refresh_trace()
        if BACKGROUNDS[self.background] == "seed r":
            self._refresh_background()
        self.say(text + (f", {twin} mirrored" if twin else ""))

    def _nudge_box(self, dr: int, dc: int) -> None:
        if self.selected is None:
            self.say("No box selected -- click one first.")
            return
        box = self.page.boxes[self.selected]
        self.page.boxes[self.selected] = Box(box.row_start + dr, box.row_end + dr,
                                             box.col_start + dc, box.col_end + dc)
        self._edited(f"{self.selected} moved ({dr:+d}, {dc:+d})")

    def _resize_box(self, delta: int) -> None:
        if self.selected is None:
            self.say("No box selected -- click one first.")
            return
        box = self.page.boxes[self.selected]
        if box.row_end - box.row_start + 2 * delta < 0 or box.col_end - box.col_start + 2 * delta < 0:
            self.say("Box is already 1 px -- cannot shrink further.")
            return
        self.page.boxes[self.selected] = Box(box.row_start - delta, box.row_end + delta,
                                             box.col_start - delta, box.col_end + delta)
        self._edited(f"{self.selected} {'grown' if delta > 0 else 'shrunk'}")

    def _move_all(self, dr: int, dc: int) -> None:
        """Shift every box; sideways is refused under mirror lock (it would break every pair)."""
        if dc and self.mirror_lock:
            self.say("Moving every box sideways breaks every mirrored pair -- press m to unlock first.")
            return
        page = self.page
        page.boxes = {label: Box(b.row_start + dr, b.row_end + dr, b.col_start + dc, b.col_end + dc)
                      for label, b in page.boxes.items()}
        page.dirty = True
        self._sync_items()
        self._refresh_trace()
        if BACKGROUNDS[self.background] == "seed r":
            self._refresh_background()
        self.say(f"All {len(page.boxes)} boxes moved ({dr:+d}, {dc:+d})")

    def _ask_name(self, title: str, default: str) -> str | None:
        """Ask for a box name until it is valid and unused; None when cancelled."""
        QtWidgets = self.QtWidgets
        prompt = ("Name: area + hemisphere letter, then an optional _suffix\n"
                  "(e.g. S1L, M2R_alta). The side letter ends the part before '_'.")
        name = default
        while True:
            name, ok = QtWidgets.QInputDialog.getText(self.window, title, prompt,
                                                      QtWidgets.QLineEdit.EchoMode.Normal, name)
            if not ok:
                return None
            name = name.strip()
            if not is_valid_roi_name(name):
                self._popup(f"{name!r} is not a valid name: letters/digits, ending the prefix in L or R.",
                            warn=True)
            elif name in self.page.boxes:
                self._popup(f"{name!r} already exists on this page.", warn=True)
            else:
                return name

    def _add_box(self, row: int, col: int) -> None:
        """Add a box centred on full-grid pixel (row, col); under mirror lock its twin too."""
        page = self.page
        name = self._ask_name("New ROI", "")
        if name is None:
            return
        twin = hemisphere_twin(name)
        if self.mirror_lock and twin in page.boxes:
            self._popup(f"Its twin {twin} already exists; under mirror lock the pair must be "
                        f"added together. Pick another name or press m.", warn=True)
            return
        span = int(self.args.new_box_span_px)
        r0, c0 = row - span // 2, col - span // 2
        page.boxes[name] = box_from_pixels(r0, r0 + span, c0, c0 + span, page.y_1, page.x_2)
        self._add_item(name)
        added = [name]
        if self.mirror_lock:
            page.boxes[twin] = mirror_box(page.boxes[name])
            self._add_item(twin)
            added.append(twin)
        page.dirty = True
        self._select(name)
        self.say(f"Added {added} (box order = column order downstream; new boxes go last)")

    def _delete_box(self) -> None:
        if self.selected is None:
            self.say("No box selected -- click one first.")
            return
        page = self.page
        doomed = [self.selected]
        twin = hemisphere_twin(self.selected)
        if self.mirror_lock and twin in page.boxes:
            doomed.append(twin)
        for label in doomed:
            del page.boxes[label]
            self._remove_item(label)
        page.dirty = True
        self._select(next(iter(page.boxes), None))
        self.say(f"Deleted {doomed}")

    def _rename_box(self) -> None:
        if self.selected is None:
            self.say("No box selected -- click one first.")
            return
        page = self.page
        old = self.selected
        new = self._ask_name(f"Rename {old}", old)
        if new is None:
            return
        renames = {old: new}
        old_twin, new_twin = hemisphere_twin(old), hemisphere_twin(new)
        if self.mirror_lock and old_twin in page.boxes:
            if new_twin is None or new_twin in page.boxes:
                self._popup(f"Cannot rename the twin {old_twin} to {new_twin}.", warn=True)
                return
            renames[old_twin] = new_twin
        # Rebuild the dict so every box keeps its position in the column order.
        page.boxes = {renames.get(label, label): box for label, box in page.boxes.items()}
        for label in renames:
            self._remove_item(label)
        for label in renames.values():
            self._add_item(label)
        page.dirty = True
        self._select(new)
        self.say(f"Renamed {renames}")

    def _on_scene_clicked(self, event) -> None:
        """Double-click on the image (not the trace panel): add a box there."""
        if not event.double() or not self.plot.vb.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self.plot.vb.mapSceneToView(event.scenePos())
        self._add_box(math.floor(point.y()), math.floor(point.x()))

    # -- batch gestures ------------------------------------------------------
    def _apply_template_to(self, targets: list[Page], what: str) -> None:
        """Make this page the template and place it on ``targets`` by their Bregma and Lambda.

        Each target gets the template's Bregma offsets scaled by its own Lambda /
        this page's Lambda (transfer_boxes). Targets whose boxes then leave their
        crop are listed and drawn red; saving refuses them.
        """
        source = self.page
        boxes = dict(source.boxes)
        self.template = dict(boxes=boxes, lambda_=source.own_lambda, key=source.key, meta=source.meta)
        scale_size = bool(self.args.lambda_scales_box_size)
        outside, factors = [], []
        for number, page in enumerate(targets, start=1):
            self._load(page)
            page.boxes = transfer_boxes(boxes, source.own_lambda, page.own_lambda, scale_size)
            page.dirty = True
            factors.append(page.own_lambda / source.own_lambda)
            if boxes_outside_crop(page.boxes, page.y_1, page.x_2, page.region):
                outside.append(page.key)
            if number % 50 == 0:
                print(f"  placed on {number}/{len(targets)}", flush=True)
        self._refresh_header()
        self.say(f"Template = {source.key} (Lambda {source.own_lambda} px, {len(boxes)} boxes) placed on "
                 f"{len(targets)} recording(s) ({what}), scale {min(factors):.3f}x-{max(factors):.3f}x; "
                 f"not saved yet" + (f"; {len(outside)} have boxes OUTSIDE their crop: {outside[:10]}"
                                     if outside else ""))
        if outside:
            self._popup(f"{len(outside)} recording(s) get boxes outside their crop after Lambda scaling "
                        f"and cannot be saved:\n{outside[:20]}\nMove those boxes inward on the template "
                        f"page and press A again.", warn=True)

    def _save_pages(self, pages: list[Page], what: str) -> None:
        """Save ``pages``; pages with a box outside their crop are skipped and listed."""
        written, refused = [], []
        for page in pages:
            self._load(page)
            if boxes_outside_crop(page.boxes, page.y_1, page.x_2, page.region) or not page.boxes:
                refused.append(page.key)
                continue
            source = (f"pixel_roi_editor.py on {self.args.trace_volume}, {page.start_source}, "
                      f"{datetime.now():%Y-%m-%d %H:%M}")
            save_page_roi_set(self.out_dir, page.key, page.boxes, page.meta, page.own_lambda, source)
            page.dirty = False
            written.append(page.key)
        if self.template is not None and written:
            # The one selection everything else derives from; re-apply it later with --apply-template.
            path = save_template(self.out_dir, self.template["boxes"], self.template["meta"],
                                 self.template["key"], self.template["lambda_"],
                                 bool(self.args.lambda_scales_box_size))
            print(f"Template written: {path}", flush=True)
        self._refresh_header()
        self.say(f"Saved {len(written)} ROI set(s) ({what}) to {self.out_dir}"
                 + (f"; REFUSED {len(refused)} with boxes outside the crop: {refused[:10]}" if refused else ""))
        if refused:
            self._popup(f"{len(refused)} page(s) not saved -- a box is outside the crop:\n{refused[:20]}",
                        warn=True)

    def _save_modified(self) -> None:
        self._save_pages([page for page in self.pages if page.dirty], "modified pages")

    # -- keys and window -----------------------------------------------------
    def _on_key(self, event) -> None:
        Qt = self.QtCore.Qt
        key, mods = event.key(), event.modifiers()
        ctrl, shift = bool(mods & Qt.ControlModifier), bool(mods & Qt.ShiftModifier)
        arrows = {Qt.Key_Up: (-1, 0), Qt.Key_Down: (1, 0), Qt.Key_Left: (0, -1), Qt.Key_Right: (0, 1)}
        if key in arrows:
            dr, dc = arrows[key]
            step = self.NUDGE_BIG if shift else 1
            (self._move_all if ctrl else self._nudge_box)(dr * step, dc * step)
        elif key == Qt.Key_N:
            self._go_to(self.index + 1)
        elif key == Qt.Key_P:
            self._go_to(self.index - 1)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._resize_box(+1)
        elif key == Qt.Key_Minus:
            self._resize_box(-1)
        elif key == Qt.Key_I:
            row_0, row_1, col_0, col_1 = self.page.region
            centre = self.plot.vb.viewRect().center()
            self._add_box(min(max(int(centre.y()), row_0), row_1 - 1), min(max(int(centre.x()), col_0), col_1 - 1))
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_box()
        elif key == Qt.Key_E:
            self._rename_box()
        elif key == Qt.Key_M:
            self.mirror_lock = not self.mirror_lock
            self._refresh_header()
            self.say(f"Mirror lock {'ON' if self.mirror_lock else 'OFF -- pairs now move independently'}")
        elif key == Qt.Key_B:
            self._cycle_background(-1 if shift else +1)
        elif key in (Qt.Key_BracketLeft, Qt.Key_BracketRight):
            step = 2.0 if key == Qt.Key_BracketRight else -2.0
            self.clip[1] = max(self.clip[0] + 1.0, min(100.0, self.clip[1] + step))
            self._refresh_background()
            self.say(f"Contrast: {self.clip[0]:.1f}-{self.clip[1]:.1f} percentile (mean F and SD views)")
        elif key == Qt.Key_A:
            if shift:
                self._apply_template_to(self.pages, "every recording")
            else:
                self._apply_template_to([p for p in self.pages if p.unit == self.page.unit], self.page.unit)
        elif key == Qt.Key_S:
            if ctrl:
                self._save_modified()
            elif shift:
                self._save_pages(self.pages, "every page")
            else:
                self._save_pages([self.page], "this page")
        elif key == Qt.Key_R:
            page = self.page
            page.boxes, page.dirty = dict(page.start_boxes), False
            self.selected = None
            self._show_page()
            self.say(f"{page.key}: reset to how it opened ({page.start_source})")
        elif key == Qt.Key_H:
            print(HELP, flush=True)
            self.say("Controls printed to the terminal.")
        elif key == Qt.Key_Q:
            self.window.close()

    def _on_close(self, event) -> None:
        """Ask before discarding unsaved pages."""
        dirty = [page.key for page in self.pages if page.dirty]
        if dirty:
            QMessageBox = self.QtWidgets.QMessageBox
            answer = QMessageBox.question(
                self.window, "Pixel ROI editor",
                f"{len(dirty)} page(s) have unsaved changes (e.g. {dirty[:5]}).\nQuit without saving?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()

    def run(self) -> None:
        self._show_page()
        self.window.show()
        self.app.exec()


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------
def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pixel-root", type=Path, default=defaults["pixel_root"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"],
                        help="Where ROI sets are written and resumed from")
    parser.add_argument("--seed-dir", type=Path, default=defaults["seed_dir"],
                        help="Read-only starting sets <day>_<animal>_<t#>.yaml; default: each dump's own")
    parser.add_argument("--trace-volume", default=defaults["trace_volume"],
                        help="Per-pixel volume file in each dump folder used for traces and views")
    parser.add_argument("--recordings", nargs="*", default=defaults["recordings"], metavar="DAY_ANIMAL/T#",
                        help="fnmatch patterns selecting the pages, e.g. '260611_PV5/*'")
    parser.add_argument("--sampling-rate-hz", type=float, default=defaults["sampling_rate_hz"])
    parser.add_argument("--new-box-span-px", type=int, default=defaults["new_box_span_px"])
    parser.add_argument("--mirror-lock", action=argparse.BooleanOptionalAction, default=defaults["mirror_lock"])
    parser.add_argument("--lambda-scales-box-size", action=argparse.BooleanOptionalAction,
                        default=defaults["lambda_scales_box_size"],
                        help="Template transfer scales box sizes too, not only their offsets from Bregma")
    parser.add_argument("--apply-template", type=Path, default=defaults["apply_template"],
                        help="No window: write every selected recording's set from this template YAML and exit")
    return parser.parse_args()


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{key: value for key, value in config.items() if key != "prefer_cli_args"})


def main() -> None:
    args = build_runtime_args()
    if args.new_box_span_px < 0:
        raise ValueError("new_box_span_px must be >= 0")
    folders = discover_recordings(Path(args.pixel_root), args.recordings)
    if args.apply_template is not None:
        # Headless: one template -> every recording, scaled by its own Lambda.
        template_boxes, template_lambda = load_template(args.apply_template)
        print(f"Template   : {args.apply_template} ({len(template_boxes)} boxes at Lambda {template_lambda} px)")
        written = apply_template(template_boxes, template_lambda, folders, Path(args.out_dir),
                                 bool(args.lambda_scales_box_size),
                                 f"pixel_roi_editor.py --apply-template {args.apply_template}")
        print(f"Wrote {len(written)} ROI sets to {args.out_dir}")
        return
    missing = [folder for folder in folders if not (folder / args.trace_volume).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} recordings have no {args.trace_volume}, e.g. {missing[0]}; "
                                f"run cache_median_dff.py or pick another --trace-volume")
    pages = [Page(folder=folder, key=recording_key(folder)) for folder in folders]
    print(f"Recordings : {len(pages)} under {args.pixel_root}")
    print(f"Traces     : {args.trace_volume}")
    print(f"Start from : {'saved set in out_dir, else ' + str(args.seed_dir) if args.seed_dir else 'saved set in out_dir, else each dump own ROI set'}")
    print(f"Saves to   : {args.out_dir}")
    print(HELP)
    print("\nOpening the editor window (it may open BEHIND this terminal). Then run a pixel script with\n"
          f"--roi-set-dir {args.out_dir}\n", flush=True)
    PixelROIEditor(pages, args).run()
    print("Editor closed.")


if __name__ == "__main__":
    main()
