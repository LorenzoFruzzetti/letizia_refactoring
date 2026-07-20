"""ROI geometry configuration (Bregma reference + the ROI boxes).

The MATLAB scripts hard-code, per animal, a downsampled Bregma position
(``y_1``, ``x_2``) and a set of ROI boxes given as *offsets* from Bregma. We keep
them in MATLAB terms here (1-based, inclusive ranges) and convert to Python
indexing at extraction time, so the numbers stay auditable against the original
scripts.

Nothing here is anatomy-specific: the boxes are an **argument**. The default is
the cerebellar 4-ROI layout because that is what this package shipped with, but
:mod:`wfci.atlases` also provides the cortical 22-ROI layout, and a study with
its own layout just passes its own dict. Everything downstream sizes itself from
``len(boxes)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """One ROI box as MATLAB-style inclusive offsets from Bregma (y_1, x_2)."""

    row_start: int  # offset added to y_1 (inclusive, 1-based MATLAB range start)
    row_end: int    # offset added to y_1 (inclusive, 1-based MATLAB range end)
    col_start: int  # offset added to x_2
    col_end: int    # offset added to x_2


def _default_boxes() -> dict[str, Box]:
    """The cerebellar 4-ROI atlas, i.e. this package's original default.

    Imported lazily: :mod:`wfci.atlases` imports :class:`Box` from this module,
    so a module-level import here would be circular.
    """
    from .atlases import CEREBELLUM_4

    return dict(CEREBELLUM_4)


@dataclass
class ROIConfig:
    """Per-animal Bregma reference and the ROI boxes to average.

    ``y_1`` / ``x_2`` are the downsampled Bregma coordinates, i.e.
    ``floor(bregma_row / 2)`` and ``floor(bregma_col / 2)`` in the MATLAB code.

    ``boxes`` order is significant: it is the column order of ``TEMP_ROI`` and
    hence the row/column order of the correlation matrix ``R``. The default is
    :data:`wfci.atlases.CEREBELLUM_4` -- ``[Laterale_L, Verme_L, Laterale_R,
    Verme_R]``, matching the MATLAB. Pass :data:`wfci.atlases.CORTEX_22` (or any
    dict) for a different layout; no other code changes.
    """

    y_1: int
    x_2: int
    boxes: dict[str, Box] = field(default_factory=_default_boxes)

    @property
    def labels(self) -> list[str]:
        """ROI names in column order -- the labels of ``TEMP_ROI`` / ``R``.

        Derived from ``boxes`` rather than stored alongside it: a second list
        would be free to disagree with the boxes it names (wrong length, stale
        order), and a mislabelled-but-valid correlation matrix is precisely the
        error nothing downstream can detect. One source of truth instead.
        """
        return list(self.boxes.keys())

    @property
    def n_rois(self) -> int:
        return len(self.boxes)

    @classmethod
    def from_bregma(cls, bregma_row: int, bregma_col: int, **kwargs) -> "ROIConfig":
        """Build from full-resolution Bregma coords (applies floor(.../2))."""
        return cls(y_1=bregma_row // 2, x_2=bregma_col // 2, **kwargs)
