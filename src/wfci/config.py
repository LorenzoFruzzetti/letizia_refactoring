"""ROI geometry configuration (Bregma reference + the four ROI boxes).

The MATLAB scripts hard-code, per animal, a downsampled Bregma position
(``y_1``, ``x_2``) and four ROI boxes given as *offsets* from Bregma. The box
extents below are copied verbatim from ``step3_ROI_functional_connectivity.m``
using MATLAB's 1-based, inclusive ranges. We keep them in MATLAB terms here and
convert to Python indexing at extraction time, so the numbers stay auditable
against the original script.

MATLAB box definitions (rows, cols as offsets from y_1 / x_2):
    Verme_R    = img(y_1+22 : y_1+27, x_2+0  : x_2+5)
    Verme_L    = img(y_1+22 : y_1+27, x_2-12 : x_2-7)
    Laterale_R = img(y_1+21 : y_1+26, x_2+29 : x_2+34)
    Laterale_L = img(y_1+21 : y_1+26, x_2-34 : x_2-29)
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


@dataclass
class ROIConfig:
    """Per-animal Bregma reference and the four ROI boxes.

    ``y_1`` / ``x_2`` are the downsampled Bregma coordinates, i.e.
    ``floor(bregma_row / 2)`` and ``floor(bregma_col / 2)`` in the MATLAB code.
    """

    y_1: int
    x_2: int
    # Order matters: this is the column order of TEMP_ROI produced by step 3,
    # namely [Laterale_L, Verme_L, Laterale_R, Verme_R].
    boxes: dict[str, Box] = field(
        default_factory=lambda: {
            "Laterale_L": Box(21, 26, -34, -29),
            "Verme_L": Box(22, 27, -12, -7),
            "Laterale_R": Box(21, 26, 29, 34),
            "Verme_R": Box(22, 27, 0, 5),
        }
    )

    @classmethod
    def from_bregma(cls, bregma_row: int, bregma_col: int, **kwargs) -> "ROIConfig":
        """Build from full-resolution Bregma coords (applies floor(.../2))."""
        return cls(y_1=bregma_row // 2, x_2=bregma_col // 2, **kwargs)
