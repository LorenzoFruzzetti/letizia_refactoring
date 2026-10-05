"""Interactive viewer for the PV3/PV4/PV5 connectivity matrices with live colour scales.

Made for matching the look of published figures whose scripts and parameters are not
available: drag the colour limits and switch colormaps until the matrices look like the
reference, then save. The settings that produced the match are written next to the PNG.

Data (same as test_pv_antea_difference.py, i.e. Antea's script 5):
    mean_R (per animal-day)  plain mean of the 5 recordings' Pearson r (`R_roi`), no Fisher z
    "mean" row               equal-weight mean of the three animals' mean_R
    difference               mean_R day N - mean_R day 1 (baseline_day_index)

Window:
    tab "Pearson r"    rows PV3/PV4/PV5/mean, columns day 1..5
    tab "Difference"   same rows, columns day 2..5 minus day 1
    controls           variant, colormap (+ reverse), min/max slider and spin box for
                       each tab's scale (update live), "symmetric about 0", hemisphere
                       lines, a reference image pane, and Save
Save writes, for the visible tab, <output_dir>/<tab>_<variant>.png (matplotlib, rotated
labels, the current colormap and limits) and <tab>_<variant>_settings.json.

Run (editor): press Run; edit RUN_CONFIG below.
Run (terminal):
    conda run --no-capture-output -n letizia python pv_matrix_viewer.py
    conda run --no-capture-output -n letizia python pv_matrix_viewer.py --variant pixels_dff_full --reference path\\to\\figure.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")   # Save renders off-screen; the live view is pyqtgraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parula_colormap import parula

REPO_ROOT = Path(__file__).resolve().parent
VARIANTS = [
    "pixels_dff_full_gsr",          # Antea's chain: mean-baseline dF/F + GSR
    "pixels_dff_full",
    "pixels_median_dff_20s_full",
    "pixels_median_dff_20s_full_gsr",
]
# Offered colormaps; any other matplotlib name can be given in RUN_CONFIG / --colormap.
COLORMAP_NAMES = ["parula", "viridis", "jet", "turbo", "RdBu_r", "bwr", "coolwarm",
                  "seismic", "hot", "gray"]
# Our ROI label -> the Allen-style label Antea's script 5 uses.
ALLEN_LABEL_OF = {
    "M2L_alta": "MOs-a_L", "M2L_bassa": "MOs-p_L", "M1L_alta": "MOp-a_L", "M1L_bassa": "MOp-p_L",
    "BFDL": "SSp-bfd_L", "TrL": "SSp-tr_L", "FLL": "SSp-fL_L", "HLL": "SSp-hl_L",
    "RSL_alta": "RSP_L", "V1aL": "VISa_L", "V1L": "VISp_L",
    "M2R_alta": "MOs-a_R", "M2R_bassa": "MOs-p_R", "M1R_alta": "MOp-a_R", "M1R_bassa": "MOp-p_R",
    "BFDR": "SSp-bfd_R", "TrR": "SSp-tr_R", "FLR": "SSp-fL_R", "HLR": "SSp-hl_R",
    "RSR_alta": "RSP_R", "V1aR": "VISa_R", "V1R": "VISp_R",
}

# Edit this section to run the viewer without passing CLI flags.
RUN_CONFIG: dict[str, Any] = {
    "design_path": REPO_ROOT / "pixel_data" / "experimental_design.csv",
    "connectivity_root": REPO_ROOT / "outputs" / "roi_pixel_connectivity",
    "output_dir": REPO_ROOT / "outputs" / "pv_matrix_viewer",
    "animals": ["PV3", "PV4", "PV5"],
    "baseline_day_index": 1,
    "variant": "pixels_dff_full_gsr",
    "colormap": "parula",
    "r_levels": (-1.0, 1.0),            # starting colour limits of the Pearson r tab
    "difference_levels": (-1.0, 1.0),   # starting colour limits of the difference tab
    "difference_limit": 2.0,            # slider range of the difference; r - r spans +-2
    "allen_labels": True,               # Antea's MOs-a_L ... labels instead of M2L_alta ...
    "reference": None,                  # image of the figure to match, shown beside (| None)
    "prefer_cli_args": True,
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_session_means(design_path: Path, connectivity_root: Path, variant: str,
                       animals: list[str]) -> tuple[dict, dict, list[str]]:
    """Plain mean of R_roi over each session's recordings (Antea's mean_R = mean(R, 3)).

    Returns mean_r[animal][day_index] -> (n_rois, n_rois), sessions[animal][day_index]
    -> "<day>_<animal>", and the ROI labels in matrix order.
    """
    design = pd.read_csv(design_path, dtype={"day": str})
    design["session"] = design["day"] + "_" + design["animal"]
    mean_r, sessions, roi_labels = {}, {}, None
    for animal in animals:
        by_day = design[design["animal"] == animal].groupby("day_index")["session"].unique()
        if any(len(s) != 1 for s in by_day):
            raise ValueError(f"{animal}: more than one session on a day_index: {by_day.to_dict()}")
        sessions[animal] = {int(day): s[0] for day, s in by_day.items()}
        mean_r[animal] = {}
        for day, session in sessions[animal].items():
            recordings = design.loc[design["session"] == session, "recording"].sort_values()
            matrices = []
            for recording in recordings:
                data = np.load(connectivity_root / variant / session / recording
                               / "roi_pixel_connectivity.npz")
                labels = list(data["roi_labels"])
                if roi_labels is None:
                    roi_labels = labels
                if labels != roi_labels:
                    raise ValueError(f"ROI labels differ in {variant}/{session}/{recording}")
                matrices.append(data["R_roi"])
            mean_r[animal][day] = np.mean(matrices, axis=0)
    return mean_r, sessions, roi_labels


def build_panels(mean_r: dict, sessions: dict, animals: list[str],
                 baseline_day: int) -> dict[str, dict]:
    """Panel grids of both tabs: {"r": {...}, "difference": {...}}.

    Each has `rows` (row names), `columns` (day indices) and `cells`:
    (i_row, i_col) -> (title, matrix).
    """
    days = sorted({d for a in animals for d in sessions[a]})
    later_days = [d for d in days if d != baseline_day]
    for animal in animals:
        if baseline_day not in sessions[animal]:
            raise ValueError(f"{animal}: no day_index {baseline_day} session")
    rows = animals + ["mean"]

    def matrix_of(name: str, day: int) -> np.ndarray:
        # The mean row averages the animals' mean_R, as Antea's group means do.
        if name == "mean":
            return np.mean([mean_r[a][day] for a in animals], axis=0)
        return mean_r[name][day]

    r_cells, difference_cells = {}, {}
    for i_row, name in enumerate(rows):
        for i_col, day in enumerate(days):
            label = f"mean: day {day}" if name == "mean" else f"{name}: {sessions[name][day]} (day {day})"
            r_cells[(i_row, i_col)] = (label, matrix_of(name, day))
        for i_col, day in enumerate(later_days):
            label = (f"mean: day {day} - day {baseline_day}" if name == "mean"
                     else f"{name}: day {day} - day {baseline_day}")
            difference_cells[(i_row, i_col)] = (label, matrix_of(name, day) - matrix_of(name, baseline_day))
    return {"r": {"rows": rows, "columns": days, "cells": r_cells},
            "difference": {"rows": rows, "columns": later_days, "cells": difference_cells}}


def matplotlib_colormap(name: str, reverse: bool):
    """The named colormap as a matplotlib Colormap (parula from parula_colormap.py)."""
    colormap = parula if name == "parula" else matplotlib.colormaps[name]
    return colormap.reversed() if reverse else colormap


def save_figure(grid: dict, tick_labels: list[str], colormap, levels: tuple[float, float],
                colorbar_label: str, title: str, hemisphere_lines: bool, path: Path) -> None:
    """Render one tab's panel grid with matplotlib at the given colormap and limits."""
    n_rows, n_cols = len(grid["rows"]), len(grid["columns"])
    n_rois = len(tick_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows),
                             constrained_layout=True, squeeze=False)
    for (i_row, i_col), (panel_title, matrix) in grid["cells"].items():
        ax = axes[i_row, i_col]
        image = ax.imshow(matrix, cmap=colormap, vmin=levels[0], vmax=levels[1],
                          interpolation="nearest")
        if hemisphere_lines:
            ax.axhline(n_rois / 2 - 0.5, color="w", linewidth=0.6)
            ax.axvline(n_rois / 2 - 0.5, color="w", linewidth=0.6)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xticks(range(n_rois), tick_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(n_rois), tick_labels, fontsize=5)
        fig.colorbar(image, ax=ax, shrink=0.75, label=colorbar_label)
    fig.suptitle(title)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class LevelControl:
    """Min and max of one colour scale: a slider and a spin box each, kept in sync.

    `on_change((low, high))` is called on every change, so dragging updates live.
    """

    def __init__(self, QtWidgets, QtCore, title: str, limit: float,
                 initial: tuple[float, float], on_change: Callable, step: float = 0.01):
        self.limit, self.step, self.on_change = limit, step, on_change
        self.initial = (float(initial[0]), float(initial[1]))
        self.values = list(self.initial)
        self.group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(self.group)
        self.sliders, self.spins = [], []
        for i_row, name in enumerate(("min", "max")):
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(-round(limit / step), round(limit / step))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-limit, limit)
            spin.setDecimals(3)
            spin.setSingleStep(step)
            # Default-argument binding: each lambda keeps its own row index.
            slider.valueChanged.connect(lambda value, i=i_row: self._set(i, value * self.step))
            spin.valueChanged.connect(lambda value, i=i_row: self._set(i, value))
            grid.addWidget(QtWidgets.QLabel(name), i_row, 0)
            grid.addWidget(slider, i_row, 1)
            grid.addWidget(spin, i_row, 2)
            self.sliders.append(slider)
            self.spins.append(spin)
        self.symmetric = QtWidgets.QCheckBox("symmetric about 0")
        self.symmetric.toggled.connect(lambda checked: checked and self._set(1, self.values[1]))
        reset = QtWidgets.QPushButton("reset")
        reset.clicked.connect(self.reset)
        grid.addWidget(self.symmetric, 2, 0, 1, 2)
        grid.addWidget(reset, 2, 2)
        self._show()

    def _set(self, index: int, value: float) -> None:
        """Apply one edited end, keep min < max (and symmetry), then notify."""
        values = list(self.values)
        values[index] = float(np.clip(value, -self.limit, self.limit))
        if self.symmetric.isChecked():
            magnitude = max(abs(values[index]), self.step)
            values = [-magnitude, magnitude]
        if values[0] >= values[1]:
            # Push the OTHER end so the one being dragged follows the hand.
            if index == 0:
                values[1] = min(values[0] + self.step, self.limit)
                values[0] = values[1] - self.step
            else:
                values[0] = max(values[1] - self.step, -self.limit)
                values[1] = values[0] + self.step
        self.values = values
        self._show()
        self.on_change(tuple(self.values))

    def _show(self) -> None:
        # Signals blocked so writing the widgets does not re-enter _set.
        for widget_index, value in enumerate(self.values):
            for widget, shown in ((self.sliders[widget_index], round(value / self.step)),
                                  (self.spins[widget_index], value)):
                widget.blockSignals(True)
                widget.setValue(shown)
                widget.blockSignals(False)

    def reset(self) -> None:
        self.symmetric.setChecked(False)
        self.values = list(self.initial)
        self._show()
        self.on_change(tuple(self.values))


class MatrixViewer:
    """Window: two tabs of matrix grids, a control column, an optional reference image."""

    def __init__(self, args: argparse.Namespace):
        import pyqtgraph as pg
        # Qt through pyqtgraph's shim, as in roi_editor.py, so any binding works.
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

        # row-major: image[row, col] like NumPy (pyqtgraph's default is the transpose).
        # White background like the matplotlib/MATLAB figures being matched.
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False,
                            background="w", foreground="k")
        self.pg, self.QtCore, self.QtGui, self.QtWidgets = pg, QtCore, QtGui, QtWidgets
        self.args = args
        self.app = pg.mkQApp("PV matrix viewer")
        self.cache: dict[str, tuple] = {}           # variant -> (panels, roi_labels)
        self.tabs_state: dict[str, dict] = {}       # "r"/"difference" -> images, bar, lines

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("PV connectivity matrices: live colour scale")
        self.window.resize(1700, 1000)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.window.setCentralWidget(splitter)

        # -- controls column
        controls = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(controls)
        form = QtWidgets.QFormLayout()
        self.variant_box = QtWidgets.QComboBox()
        self.variant_box.addItems(VARIANTS)
        self.variant_box.setCurrentText(args.variant)
        form.addRow("variant", self.variant_box)
        self.colormap_box = QtWidgets.QComboBox()
        self.colormap_box.setEditable(True)            # any matplotlib name can be typed
        self.colormap_box.addItems(COLORMAP_NAMES)
        self.colormap_box.setCurrentText(args.colormap)
        form.addRow("colormap", self.colormap_box)
        self.reverse_box = QtWidgets.QCheckBox("reverse colormap")
        form.addRow(self.reverse_box)
        self.lines_box = QtWidgets.QCheckBox("hemisphere lines")
        form.addRow(self.lines_box)
        column.addLayout(form)
        self.r_control = LevelControl(QtWidgets, QtCore, "Pearson r scale", 1.0,
                                      tuple(args.r_levels), lambda v: self._apply_levels("r", v))
        self.difference_control = LevelControl(
            QtWidgets, QtCore, "Difference scale", float(args.difference_limit),
            tuple(args.difference_levels), lambda v: self._apply_levels("difference", v))
        column.addWidget(self.r_control.group)
        column.addWidget(self.difference_control.group)
        reference_button = QtWidgets.QPushButton("open reference image...")
        reference_button.clicked.connect(self._choose_reference)
        save_button = QtWidgets.QPushButton("save visible tab (PNG + settings)")
        save_button.clicked.connect(self.save_visible_tab)
        column.addWidget(reference_button)
        column.addWidget(save_button)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        column.addStretch(1)
        controls.setMaximumWidth(380)
        splitter.addWidget(controls)

        # -- matrix tabs
        self.tab_widget = QtWidgets.QTabWidget()
        # Each tab: a fixed-height strip with the colour bar above the panel grid. The
        # bar lives in its own widget so it can neither steal the grid's height nor be
        # pushed off-screen by the number of columns.
        self.layouts, self.bar_layouts = {}, {}
        for key, tab_title in (("r", "Pearson r"), ("difference", "Difference (day N - day 1)")):
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            self.bar_layouts[key] = pg.GraphicsLayoutWidget()
            self.bar_layouts[key].setFixedHeight(80)
            self.layouts[key] = pg.GraphicsLayoutWidget()
            page_layout.addWidget(self.bar_layouts[key])
            page_layout.addWidget(self.layouts[key], 1)
            self.tab_widget.addTab(page, tab_title)
        splitter.addWidget(self.tab_widget)

        # -- reference image pane (empty until an image is opened)
        self.reference_label = QtWidgets.QLabel("no reference image")
        self.reference_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.reference_label.setMinimumWidth(200)
        splitter.addWidget(self.reference_label)
        splitter.setSizes([340, 1100, 300])

        self.variant_box.currentTextChanged.connect(self._rebuild)
        self.colormap_box.currentTextChanged.connect(self._apply_colormap)
        self.reverse_box.toggled.connect(self._apply_colormap)
        self.lines_box.toggled.connect(self._apply_lines)
        self._rebuild(args.variant)
        if args.reference is not None:
            self.load_reference(Path(args.reference))

    # -- data -> grid --------------------------------------------------------
    def _data(self, variant: str) -> tuple[dict, list[str]]:
        """Panels and tick labels of one variant, loaded once and cached."""
        if variant not in self.cache:
            mean_r, sessions, roi_labels = load_session_means(
                Path(self.args.design_path), Path(self.args.connectivity_root), variant,
                list(self.args.animals))
            panels = build_panels(mean_r, sessions, list(self.args.animals),
                                  int(self.args.baseline_day_index))
            labels = [ALLEN_LABEL_OF[l] for l in roi_labels] if self.args.allen_labels else roi_labels
            self.cache[variant] = (panels, labels)
        return self.cache[variant]

    def _rebuild(self, variant: str) -> None:
        """(Re)draw both tabs for a variant; colormap and levels are re-applied."""
        pg = self.pg
        panels, labels = self._data(variant)
        n_rois = len(labels)
        tick_font = self.QtGui.QFont()
        tick_font.setPointSize(6)
        for key, colorbar_label in (("r", "Pearson r"), ("difference", "difference in r")):
            layout, grid = self.layouts[key], panels[key]
            layout.clear()
            self.bar_layouts[key].clear()
            images, lines = [], []
            # Horizontal colour bar, shown in the strip above the grid.
            control = self.r_control if key == "r" else self.difference_control
            bar = pg.ColorBarItem(values=tuple(control.values), interactive=False, width=12,
                                  orientation="horizontal")
            bar.axis.setLabel(colorbar_label)
            self.bar_layouts[key].addItem(bar, row=0, col=0)
            for (i_row, i_col), (title, matrix) in grid["cells"].items():
                # Short titles: a title's text sets the panel's minimum width, and the
                # full "<day>_<animal>" titles pushed the last columns off-screen. The
                # full title is the tooltip and is used in the saved PNG.
                day = grid["columns"][i_col]
                short_title = (f"{grid['rows'][i_row]} · day {day}" if key == "r"
                               else f"{grid['rows'][i_row]} · d{day}-d{self.args.baseline_day_index}")
                plot = layout.addPlot(row=i_row, col=i_col)
                plot.setTitle(short_title, size="8pt")
                plot.setToolTip(title)
                plot.setAspectLocked(True)
                plot.invertY(True)                      # row 0 at the top, like imshow
                plot.setMouseEnabled(x=False, y=False)
                plot.hideButtons()
                image = pg.ImageItem(matrix)            # pixel (r, c) spans [c, c+1] x [r, r+1]
                plot.addItem(image)
                images.append(image)
                # ROI names on the left column, ROI numbers along the bottom row
                # (pyqtgraph cannot rotate tick text); full labels are in the saved PNG.
                left, bottom = plot.getAxis("left"), plot.getAxis("bottom")
                left.setTicks([[(i + 0.5, labels[i]) for i in range(n_rois)]])
                bottom.setTicks([[(i + 0.5, str(i + 1)) for i in range(n_rois)]])
                for axis in (left, bottom):
                    axis.setStyle(tickFont=tick_font, tickLength=0)
                # Axes elsewhere would only reserve empty margin and squeeze the panels.
                if i_col > 0:
                    plot.hideAxis("left")
                if i_row < len(grid["rows"]) - 1:
                    plot.hideAxis("bottom")
                for angle in (0, 90):
                    line = pg.InfiniteLine(pos=n_rois / 2, angle=angle, pen=pg.mkPen("w", width=1))
                    line.setZValue(10)
                    plot.addItem(line)
                    lines.append(line)
            bar.setImageItem(images)
            self.tabs_state[key] = {"images": images, "bar": bar, "lines": lines}
        self._apply_colormap()
        self._apply_lines()
        self._apply_levels("r", tuple(self.r_control.values))
        self._apply_levels("difference", tuple(self.difference_control.values))

    # -- live updates --------------------------------------------------------
    def _current_colormap(self):
        return matplotlib_colormap(self.colormap_box.currentText(), self.reverse_box.isChecked())

    def _apply_colormap(self, *_) -> None:
        name = self.colormap_box.currentText()
        if name != "parula" and name not in matplotlib.colormaps:
            self.status.setText(f"unknown colormap '{name}'")   # still typing a name
            return
        rgba = self._current_colormap()(np.linspace(0.0, 1.0, 256))   # (256, 4) floats
        colormap = self.pg.ColorMap(np.linspace(0.0, 1.0, 256), (rgba * 255).astype(np.ubyte))
        for state in self.tabs_state.values():
            state["bar"].setColorMap(colormap)
        self._update_status()

    def _apply_levels(self, key: str, levels: tuple[float, float]) -> None:
        if key in self.tabs_state:
            self.tabs_state[key]["bar"].setLevels(levels)   # propagates to every image
        self._update_status()

    def _apply_lines(self, *_) -> None:
        for state in self.tabs_state.values():
            for line in state["lines"]:
                line.setVisible(self.lines_box.isChecked())

    def _update_status(self) -> None:
        r_low, r_high = self.r_control.values if hasattr(self, "r_control") else (0, 0)
        d_low, d_high = (self.difference_control.values
                         if hasattr(self, "difference_control") else (0, 0))
        self.status.setText(
            f"colormap {self.colormap_box.currentText()}"
            f"{' (reversed)' if self.reverse_box.isChecked() else ''}\n"
            f"Pearson r: vmin={r_low:.3f}, vmax={r_high:.3f}\n"
            f"difference: vmin={d_low:.3f}, vmax={d_high:.3f}")

    # -- reference image -----------------------------------------------------
    def _choose_reference(self) -> None:
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Reference figure", str(REPO_ROOT),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.load_reference(Path(path))

    def load_reference(self, path: Path) -> None:
        pixmap = self.QtGui.QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"could not read reference image {path}")
        self.reference_label.setPixmap(pixmap.scaledToWidth(
            600, self.QtCore.Qt.TransformationMode.SmoothTransformation))
        self.reference_label.setToolTip(str(path))

    # -- save ----------------------------------------------------------------
    def save_visible_tab(self) -> Path:
        """Write the visible tab as a matplotlib PNG plus the settings that made it."""
        key = "r" if self.tab_widget.currentIndex() == 0 else "difference"
        variant = self.variant_box.currentText()
        panels, labels = self._data(variant)
        control = self.r_control if key == "r" else self.difference_control
        levels = tuple(control.values)
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{key}_{variant}"
        colorbar_label = "Pearson r" if key == "r" else "difference in Pearson r"
        save_figure(panels[key], labels, self._current_colormap(), levels, colorbar_label,
                    f"{variant}: {colorbar_label} (colormap {self.colormap_box.currentText()}, "
                    f"vmin {levels[0]:g}, vmax {levels[1]:g})",
                    self.lines_box.isChecked(), output_dir / f"{stem}.png")
        settings = {"tab": key, "variant": variant, "colormap": self.colormap_box.currentText(),
                    "reverse_colormap": self.reverse_box.isChecked(),
                    "vmin": levels[0], "vmax": levels[1],
                    "hemisphere_lines": self.lines_box.isChecked()}
        (output_dir / f"{stem}_settings.json").write_text(json.dumps(settings, indent=2))
        self.status.setText(self.status.text() + f"\nsaved {output_dir / (stem + '.png')}")
        print(f"saved {output_dir / (stem + '.png')}: {settings}")
        return output_dir / f"{stem}.png"


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------
def parse_args(defaults: dict[str, Any]) -> argparse.Namespace:
    """CLI overrides of RUN_CONFIG."""
    p = argparse.ArgumentParser(description="Live colour-scale viewer for the PV matrices.")
    p.add_argument("--variant", choices=VARIANTS, default=defaults["variant"])
    p.add_argument("--colormap", default=defaults["colormap"],
                   help="parula or any matplotlib colormap name.")
    p.add_argument("--r-levels", type=float, nargs=2, default=defaults["r_levels"],
                   metavar=("VMIN", "VMAX"))
    p.add_argument("--difference-levels", type=float, nargs=2,
                   default=defaults["difference_levels"], metavar=("VMIN", "VMAX"))
    p.add_argument("--reference", default=defaults["reference"],
                   help="Image of the figure to match, shown beside the matrices.")
    p.add_argument("--output-dir", default=defaults["output_dir"])
    p.add_argument("--our-labels", dest="allen_labels", action="store_false",
                   default=defaults["allen_labels"],
                   help="Use M2L_alta-style labels instead of Antea's Allen names.")
    ns = p.parse_args()
    for key in ("design_path", "connectivity_root", "animals", "baseline_day_index",
                "difference_limit"):
        setattr(ns, key, defaults[key])
    return ns


def build_runtime_args(config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Prefer CLI args when any are given, else fall back to RUN_CONFIG (editor mode)."""
    config = dict(RUN_CONFIG if config is None else config)
    if bool(config.get("prefer_cli_args", True)) and len(sys.argv) > 1:
        return parse_args(defaults=config)
    return argparse.Namespace(**{k: v for k, v in config.items() if k != "prefer_cli_args"})


def main() -> None:
    args = build_runtime_args()
    viewer = MatrixViewer(args)
    viewer.window.show()
    viewer.pg.exec()


if __name__ == "__main__":
    main()
