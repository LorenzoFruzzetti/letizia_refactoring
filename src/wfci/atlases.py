"""ROI atlases -- named, self-describing box layouts.

An *atlas* is an ordered set of ROI boxes given as offsets from a per-animal
Bregma reference (see :mod:`wfci.config`), plus the conditions under which those
offsets are meaningful.

**The boxes are not anatomy.** They are anatomy projected through one optical
setup: a particular field of view, magnification and downsampling. The same
region sits at different pixel offsets on a different rig. So an atlas that does
not know which grid it was drawn for cannot be checked, and an unchecked atlas
fails *silently* -- the boxes still land somewhere, the means still compute, the
correlation matrix still looks publishable. That is why :class:`Atlas` carries
``grid`` and why :func:`wfci.roi.box_slices_for` refuses to guess.

**Order is significant**: it is the column order of ``TEMP_ROI`` and therefore the
row/column order of the correlation matrix ``R``.

**These are presets, not a closed set.** Nothing in the package special-cases
them: every output is sized from ``len(atlas)``. A study with its own layout
builds its own -- in Python, or in a YAML/JSON file it owns
(:func:`load_atlas` / :func:`save_atlas`) -- and never edits this module.

Two presets ship:

``CEREBELLUM_4``
    The cerebellar resting-state / stimulated pipeline (``matlab/step3_*.m``).
    MATLAB-validated to machine precision -- see ``tests/test_parity.py``.

``CORTEX_22``
    The cortical GSR pipeline (``Antea_scripts/``). Transcribed from the MATLAB;
    see the provenance note on that constant before trusting the coordinates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .config import Box

# The final analysis grid both shipped atlases were drawn on: 512x512 raw frames
# after the pipeline's two 0.5x box downsamples. The cortical scripts say so in
# their own filenames ("FOV128-128", "FOV128x128").
_FOV_128 = (128, 128)


@dataclass(frozen=True, eq=False)
class Atlas(Mapping):
    """An ordered ``{label: Box}`` mapping that knows where it is valid.

    Behaves as a read-only mapping, so it drops straight into anything that took
    a plain dict: ``dict(atlas)``, ``list(atlas)``, ``atlas["V1R"]``, ``len(atlas)``,
    ``atlas.items()``, and ``atlas == {...}`` all work.

    Attributes
    ----------
    name:
        Identifier, used in error messages.
    boxes:
        ``{label: Box}``. **Order is the column order** of ``TEMP_ROI`` / ``R``.
        Copied on construction, so the atlas cannot be mutated out from under a
        caller (nor can a caller mutate a shared preset).
    grid:
        ``(rows, cols)`` of the FINAL analysis frame these offsets were drawn for,
        or ``None`` for "unknown, do not check". Checked against the real data by
        :func:`wfci.roi.box_slices_for`. ``None`` is honest for an ad-hoc atlas but
        gives up the one check that catches a *scaled* atlas whose boxes all still
        fit -- see that function.
    source:
        Free-text provenance: where these numbers came from and who is
        accountable for them.
    """

    name: str
    boxes: dict[str, Box] = field(default_factory=dict)
    grid: tuple[int, int] | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.boxes:
            raise ValueError(f"Atlas {self.name!r} has no boxes.")
        object.__setattr__(self, "boxes", dict(self.boxes))
        if self.grid is not None:
            grid = tuple(int(v) for v in self.grid)
            if len(grid) != 2 or grid[0] <= 0 or grid[1] <= 0:
                raise ValueError(
                    f"Atlas {self.name!r}: grid must be (rows, cols) of positive "
                    f"ints, got {self.grid!r}."
                )
            object.__setattr__(self, "grid", grid)

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Box:
        return self.boxes[key]

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self) -> int:
        return len(self.boxes)

    def __eq__(self, other) -> bool:
        # Compare as a mapping of boxes: an Atlas and the plain dict with the same
        # boxes are the same layout. Name/grid/source are metadata about the
        # layout, not part of it.
        if isinstance(other, Atlas):
            return self.boxes == other.boxes
        if isinstance(other, Mapping):
            return self.boxes == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        where = f", grid={self.grid}" if self.grid else ""
        return f"Atlas({self.name!r}, {len(self.boxes)} boxes{where})"

    @property
    def labels(self) -> list[str]:
        """Labels in column order."""
        return list(self.boxes)


# ---------------------------------------------------------------------------
# Cerebellum -- 4 ROIs
# ---------------------------------------------------------------------------
# Copied verbatim from step3_ROI_functional_connectivity.m (MATLAB 1-based,
# inclusive offsets from Bregma):
#     Verme_R    = img(y_1+22 : y_1+27, x_2+0  : x_2+5)
#     Verme_L    = img(y_1+22 : y_1+27, x_2-12 : x_2-7)
#     Laterale_R = img(y_1+21 : y_1+26, x_2+29 : x_2+34)
#     Laterale_L = img(y_1+21 : y_1+26, x_2-34 : x_2-29)
# The dict order below is the MATLAB's TEMP_ROI column order, which is NOT the
# order the boxes are defined in above.
#
# NOTE: step2_area_location_RS.m draws Laterale_L at x_2-37:x_2-32 -- three
# columns away from where step3 *averages* it. The two MATLAB scripts have
# drifted. step3 is the one that produced the numbers, so step3 wins here; the
# overlay figure in that repo shows a box only half-overlapping the data it was
# meant to verify. This is the reason wfci keeps ONE atlas and derives the
# overlay from it (see wfci.visualize) instead of repeating the coordinates.
CEREBELLUM_4 = Atlas(
    name="cerebellum_4",
    boxes={
        "Laterale_L": Box(21, 26, -34, -29),
        "Verme_L": Box(22, 27, -12, -7),
        "Laterale_R": Box(21, 26, 29, 34),
        "Verme_R": Box(22, 27, 0, 5),
    },
    grid=_FOV_128,
    source="matlab/step3_ROI_functional_connectivity.m; MATLAB-validated "
           "(tests/test_parity.py)",
)

# ---------------------------------------------------------------------------
# Cortex -- 22 ROIs
# ---------------------------------------------------------------------------
# PROVENANCE / TRUST (read before relying on these numbers):
#   Transcribed from the Antea cortical scripts, which define the same 22 boxes
#   TWICE -- once to draw them, once to average them:
#     * Antea_scripts/(3)_FOV128-128_ROIposition.txt   (the overlay)
#     * Antea_scripts/(4)_FOV128x128_corr_SCRIPT.txt   (the ROI means)
#   Both copies were cross-checked box-for-box and are IDENTICAL -- unlike the
#   cerebellar pair above, which HAS drifted. tests/test_atlas_transcription.py
#   re-checks both facts, against the real .txt files, on every run.
#   (3) supplies no names, so every label here comes from (4)'s variable names.
#
#   Unlike CEREBELLUM_4 there is NO MATLAB reference for this path -- the
#   coordinates are a careful reading of the scripts, not a proven match. See
#   MERGING_PLAN.md Phase 8. The `_alta` / `_bassa` suffixes ("upper" / "lower")
#   are the source scripts' own; two boxes per region, not a naming accident.
#
# ORDER: taken from (4)'s
#     regioni_L = cat(2, M2L_alta, M2L_bassa, M1L_alta, M1L_bassa, BFDL, TrL,
#                        FLL, HLL, RSL_alta, V1aL, V1L)
#     regioni_R = cat(2, M2R_alta, ...)          % same sequence, right side
#     ALL       = cat(2, regioni_L, regioni_R)   % <- all 11 left, then all 11 right
# so R is block-structured: left hemisphere first, then right, each in
# anterior-to-posterior order (motor -> somatosensory -> retrosplenial -> visual).
# Changing this order silently permutes every published matrix; don't.
CORTEX_22 = Atlas(
    name="cortex_22",
    boxes={
        # -- left hemisphere (regioni_L) --
        "M2L_alta": Box(-28, -23, -15, -10),    # secondary motor, upper
        "M2L_bassa": Box(-10, -5, -10, -5),     # secondary motor, lower
        "M1L_alta": Box(-17, -12, -32, -27),    # primary motor, upper
        "M1L_bassa": Box(-6, -1, -20, -15),     # primary motor, lower
        "BFDL": Box(17, 22, -42, -37),          # barrel field
        "TrL": Box(18, 23, -26, -21),           # trunk
        "FLL": Box(0, 5, -32, -27),             # forelimb
        "HLL": Box(7, 12, -22, -17),            # hindlimb
        "RSL_alta": Box(25, 30, -9, -4),        # retrosplenial, upper
        "V1aL": Box(27, 32, -28, -23),          # primary visual, anterior
        "V1L": Box(41, 46, -30, -25),           # primary visual
        # -- right hemisphere (regioni_R) --
        "M2R_alta": Box(-28, -23, 10, 15),
        "M2R_bassa": Box(-10, -5, 5, 10),
        "M1R_alta": Box(-17, -12, 27, 32),
        "M1R_bassa": Box(-6, -1, 15, 20),
        "BFDR": Box(17, 22, 37, 42),
        "TrR": Box(18, 23, 21, 26),
        "FLR": Box(0, 5, 27, 32),
        "HLR": Box(7, 12, 17, 22),
        "RSR_alta": Box(25, 30, 4, 9),
        "V1aR": Box(27, 32, 23, 28),
        "V1R": Box(41, 46, 25, 30),
    },
    grid=_FOV_128,
    source="Antea_scripts/(3)_FOV128-128_ROIposition.txt + "
           "(4)_FOV128x128_corr_SCRIPT.txt; NO MATLAB parity reference "
           "(MERGING_PLAN.md Phase 8)",
)

ATLASES: dict[str, Atlas] = {
    CEREBELLUM_4.name: CEREBELLUM_4,
    CORTEX_22.name: CORTEX_22,
}


def atlas_labels(atlas: Mapping) -> list[str]:
    """The atlas's labels in column order (i.e. ``TEMP_ROI`` / ``R`` order)."""
    return list(atlas.keys())


def as_atlas(boxes: Mapping, name: str = "custom") -> Atlas:
    """Coerce a plain ``{label: Box}`` dict to an :class:`Atlas`.

    An :class:`Atlas` passes through unchanged. A bare dict has no declared grid,
    so it is wrapped with ``grid=None`` -- honest, but it forgoes the scaled-atlas
    check. Set a grid explicitly when you know it.
    """
    if isinstance(boxes, Atlas):
        return boxes
    return Atlas(name=name, boxes=dict(boxes), grid=None)


# ---------------------------------------------------------------------------
# Atlases as files a study owns
# ---------------------------------------------------------------------------
_BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")


def load_atlas(path: str | Path) -> Atlas:
    """Load an atlas from a YAML or JSON file (by extension).

    This is how a study owns its own geometry: the ROI layout lives with the
    experiment -- next to its Bregma values and trial list -- instead of being
    edited into the library. The presets remain as validated defaults.

    Expected structure::

        name: my_cortex
        grid: [128, 128]          # optional but recommended; see Atlas.grid
        source: "drawn by AB, 2026-03; 4x objective, 512px sensor"
        boxes:                    # ORDER IS PRESERVED -> column order of R
          M2L_alta:  {row_start: -28, row_end: -23, col_start: -15, col_end: -10}
          M2L_bassa: {row_start: -10, row_end:  -5, col_start: -10, col_end:  -5}

    Offsets are MATLAB-style: 1-based and **inclusive**, relative to Bregma —
    exactly as written in the original scripts, so a file can be checked against
    them by eye.

    Round-trips with :func:`save_atlas`; ``save_atlas(CORTEX_22, "x.yaml")`` gives
    a working starting point to edit.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        import yaml

        # safe_load: an atlas file is data, and must never be able to execute.
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported atlas format {suffix!r} for {path}. Use .yaml, .yml or "
            f".json."
        )

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got "
                         f"{type(data).__name__}.")

    raw_boxes = data.get("boxes")
    if not isinstance(raw_boxes, dict) or not raw_boxes:
        raise ValueError(
            f"{path}: needs a non-empty 'boxes' mapping of label -> "
            f"{{{', '.join(_BOX_FIELDS)}}}."
        )

    boxes: dict[str, Box] = {}
    for label, spec in raw_boxes.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"{path}: box {label!r} must be a mapping with "
                f"{', '.join(_BOX_FIELDS)}; got {spec!r}."
            )
        missing = [f for f in _BOX_FIELDS if f not in spec]
        if missing:
            raise ValueError(f"{path}: box {label!r} is missing {missing}.")
        extra = [k for k in spec if k not in _BOX_FIELDS]
        if extra:
            # A typo'd key would otherwise be silently ignored and the box would
            # take a default it never declared.
            raise ValueError(
                f"{path}: box {label!r} has unexpected key(s) {extra}; allowed: "
                f"{list(_BOX_FIELDS)}."
            )
        try:
            boxes[str(label)] = Box(**{f: int(spec[f]) for f in _BOX_FIELDS})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: box {label!r} has non-integer offsets: {exc}") from None

    grid = data.get("grid")
    if grid is not None:
        grid = tuple(int(v) for v in grid)

    return Atlas(
        name=str(data.get("name", path.stem)),
        boxes=boxes,
        grid=grid,
        source=str(data.get("source", f"loaded from {path}")),
    )


def save_atlas(atlas: Mapping, path: str | Path) -> Path:
    """Write an atlas to YAML or JSON (by extension). Round-trips :func:`load_atlas`.

    Handy for exporting a preset as an editable starting point::

        save_atlas(CORTEX_22, "my_study/atlas.yaml")
    """
    atlas = as_atlas(atlas)
    path = Path(path)
    payload = {
        "name": atlas.name,
        "grid": list(atlas.grid) if atlas.grid else None,
        "source": atlas.source,
        "boxes": {
            label: {f: getattr(box, f) for f in _BOX_FIELDS}
            for label, box in atlas.boxes.items()
        },
    }
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        # sort_keys=False: the box order IS the column order of R, so alphabetising
        # the file would quietly relabel every matrix built from it.
        # default_flow_style=None: put leaf collections inline, so each box is one
        # readable line ("M2L_alta: {row_start: -28, ...}") instead of five. This
        # file is meant to be edited and diffed by hand -- 22 boxes as 110 lines of
        # block YAML is not.
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=None)
    elif suffix == ".json":
        text = json.dumps(payload, indent=2)
    else:
        raise ValueError(
            f"Unsupported atlas format {suffix!r} for {path}. Use .yaml, .yml or "
            f".json."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
