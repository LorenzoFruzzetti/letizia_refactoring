"""Pin the ROI atlases to the MATLAB scripts they were transcribed from.

There is no MATLAB reference for the cortical pipeline (MERGING_PLAN.md P4), so
``CORTEX_22`` rests entirely on 22 boxes having been copied out of a .txt file by
hand -- 88 integers, each of which is silently plausible if wrong. A transposed
digit would not crash anything; it would just quietly average the wrong pixels
and produce a publishable-looking correlation matrix.

So instead of trusting the transcription, this parses the original MATLAB and
compares. It also re-runs, on every test run, the cross-check that the plan
called for at transcription time: the Antea scripts define the same 22 boxes
TWICE -- once to draw them (script 3) and once to average them (script 4) -- and
those two copies must agree. The equivalent cerebellar pair HAS drifted in this
repo, which is why duplication is treated as guilty until proven innocent.

If these tests fail, believe the MATLAB, not the Python.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wfci.atlases import CEREBELLUM_4, CORTEX_22
from wfci.config import Box

ANTEA = Path(__file__).resolve().parents[1] / "Antea_scripts"
SCRIPT_3 = ANTEA / "(3)_FOV128-128_ROIposition.txt"
SCRIPT_4 = ANTEA / "(4)_FOV128x128_corr_SCRIPT.txt"

# img_av(y_1+17 : y_1+22, x_2+37 : x_2+42, :)  -- MATLAB inclusive offset ranges.
# "y_1-0" appears in the source, hence the sign is captured rather than assumed.
_BOX_RE = re.compile(
    r"y_1\s*([+-]\s*\d+)\s*:\s*y_1\s*([+-]\s*\d+)\s*,\s*"
    r"x_2\s*([+-]\s*\d+)\s*:\s*x_2\s*([+-]\s*\d+)"
)
# BFD_bis = img_av(<box>);
_ASSIGN_RE = re.compile(r"^\s*(\w+)\s*=\s*img_av\s*\(")
# BFDR = nanmean(nanmean(BFD_bis,1),2);
_MEAN_RE = re.compile(r"^\s*(\w+)\s*=\s*nanmean\s*\(\s*nanmean\s*\(\s*(\w+)\s*,\s*1\s*\)\s*,\s*2\s*\)")
# regioni_L = cat(2,M2L_alta',M2L_bassa',...);
# Deliberately NOT anchored to the line start: the source runs statements
# together ("HLL = HLL(:,:);regioni_R = cat(2,...);" is one physical line), and
# regioni_R is defined BEFORE regioni_L. Both are why this reads the whole text.
_CAT_RE = re.compile(r"(regioni_[LR])\s*=\s*cat\s*\(\s*2\s*,([^)]+?)\)\s*;")
# ALL = cat(2,regioni_L,regioni_R);  -- the final column order.
_ALL_RE = re.compile(r"\bALL\s*=\s*cat\s*\(\s*2\s*,([^)]+?)\)\s*;")


def _int(token: str) -> int:
    return int(token.replace(" ", ""))


def _boxes_in(line: str) -> list[Box]:
    return [
        Box(_int(a), _int(b), _int(c), _int(d))
        for a, b, c, d in _BOX_RE.findall(line)
    ]


@pytest.fixture(scope="module")
def script_3_boxes() -> list[Box]:
    """Every box drawn by the overlay script, in source order."""
    if not SCRIPT_3.exists():
        pytest.skip(f"MATLAB source not found: {SCRIPT_3}")
    boxes = []
    for line in SCRIPT_3.read_text(encoding="utf-8", errors="replace").splitlines():
        # Only the overlay assignments, not the img_av = ... setup line.
        if "img_av(" in line and line.rstrip().endswith("=1;"):
            boxes.extend(_boxes_in(line))
    return boxes


@pytest.fixture(scope="module")
def script_4() -> dict:
    """The correlation script's label -> Box map plus its output column order."""
    if not SCRIPT_4.exists():
        pytest.skip(f"MATLAB source not found: {SCRIPT_4}")
    text = SCRIPT_4.read_text(encoding="utf-8", errors="replace")

    temp_boxes: dict[str, Box] = {}   # e.g. BFD_bis -> Box
    labels: dict[str, Box] = {}       # e.g. BFDR    -> Box

    for line in text.splitlines():
        # A temp var is assigned a slice of img_av. NOTE: the script REUSES some
        # temp names (M2R_bis_2 is assigned twice, for _alta then _bassa), so the
        # map must be updated in order, never treated as write-once.
        m = _ASSIGN_RE.match(line)
        if m:
            found = _boxes_in(line)
            if found:
                temp_boxes[m.group(1)] = found[0]

        m = _MEAN_RE.match(line)
        if m:
            label, temp = m.group(1), m.group(2)
            assert temp in temp_boxes, f"{label} averages {temp}, never assigned"
            labels[label] = temp_boxes[temp]

    def _names(group: str) -> list[str]:
        return [t.strip().rstrip("'") for t in group.split(",") if t.strip()]

    groups = {m.group(1): _names(m.group(2)) for m in _CAT_RE.finditer(text)}

    # Expand ALL = cat(2, regioni_L, regioni_R) into the real column order rather
    # than assuming which side comes first.
    all_m = _ALL_RE.search(text)
    assert all_m, "script (4) has no ALL = cat(2, ...) line"
    column_order: list[str] = []
    for name in _names(all_m.group(1)):
        assert name in groups, f"ALL references {name}, which is never built"
        column_order.extend(groups[name])

    return {"labels": labels, "groups": groups, "column_order": column_order}


# ---------------------------------------------------------------------------
# The two MATLAB copies must agree with each other
# ---------------------------------------------------------------------------
def test_the_two_matlab_copies_define_the_same_boxes(script_3_boxes, script_4):
    """Script (3)'s overlay and script (4)'s ROI means must draw the same boxes.

    They are maintained by hand as two independent lists; if they ever diverge,
    the figure showing where the ROIs are stops describing where the numbers came
    from -- which is the failure mode you cannot see in the output.
    """
    drawn = sorted(script_3_boxes, key=lambda b: (b.row_start, b.col_start))
    averaged = sorted(script_4["labels"].values(), key=lambda b: (b.row_start, b.col_start))

    assert len(drawn) == 22, f"script (3) defines {len(drawn)} boxes, expected 22"
    assert len(averaged) == 22, f"script (4) defines {len(averaged)} boxes, expected 22"
    assert drawn == averaged, (
        "the overlay script and the correlation script disagree about the ROI "
        "boxes; the MATLAB duplication has drifted"
    )


# ---------------------------------------------------------------------------
# The Python transcription must match the MATLAB
# ---------------------------------------------------------------------------
def test_cortex_22_matches_the_matlab_coordinates(script_4):
    """Every CORTEX_22 box equals the box the MATLAB averages under that name."""
    matlab = script_4["labels"]

    assert set(CORTEX_22) == set(matlab), (
        f"label mismatch.\n  only in Python: {sorted(set(CORTEX_22) - set(matlab))}"
        f"\n  only in MATLAB: {sorted(set(matlab) - set(CORTEX_22))}"
    )
    wrong = {
        name: (box, matlab[name])
        for name, box in CORTEX_22.items()
        if box != matlab[name]
    }
    assert not wrong, "\n".join(
        f"{name}: python={py} matlab={ml}" for name, (py, ml) in wrong.items()
    )


def test_cortex_22_column_order_matches_the_matlab(script_4):
    """Dict order must equal MATLAB's ``ALL = cat(2, regioni_L, regioni_R)``.

    ``extract_roi_timeseries`` uses ``cfg.boxes`` key order as TEMP_ROI's column
    order, so this ordering IS the row/column meaning of every R matrix. Reorder
    the dict and the matrix permutes silently -- same shape, same values, wrong
    labels.
    """
    expected = script_4["column_order"]

    assert len(expected) == 22, f"MATLAB's ALL has {len(expected)} columns, expected 22"
    assert list(CORTEX_22) == expected, (
        f"CORTEX_22 order does not match MATLAB's ALL concatenation.\n"
        f"  python: {list(CORTEX_22)}\n  matlab: {expected}"
    )


def test_cortex_22_is_left_right_symmetric():
    """Each L/R pair must mirror: same rows, mirrored columns about Bregma.

    Independent of the parsing above -- it checks the anatomy the coordinates are
    supposed to encode, so a digit copied from the wrong line is caught even if
    both MATLAB copies happened to carry it.
    """
    for left, right in [
        ("M2L_alta", "M2R_alta"), ("M2L_bassa", "M2R_bassa"),
        ("M1L_alta", "M1R_alta"), ("M1L_bassa", "M1R_bassa"),
        ("BFDL", "BFDR"), ("TrL", "TrR"), ("FLL", "FLR"), ("HLL", "HLR"),
        ("RSL_alta", "RSR_alta"), ("V1aL", "V1aR"), ("V1L", "V1R"),
    ]:
        l, r = CORTEX_22[left], CORTEX_22[right]
        assert (l.row_start, l.row_end) == (r.row_start, r.row_end), (
            f"{left}/{right} sit at different heights: rows {l.row_start}..{l.row_end} "
            f"vs {r.row_start}..{r.row_end}"
        )
        # Mirrored: left spans [-b, -a] where right spans [a, b].
        assert (l.col_start, l.col_end) == (-r.col_end, -r.col_start), (
            f"{left}/{right} are not mirrored about Bregma: cols "
            f"{l.col_start}..{l.col_end} vs {r.col_start}..{r.col_end}"
        )


# ---------------------------------------------------------------------------
# The cerebellar atlas must not have moved (P1: today's outputs are the contract)
# ---------------------------------------------------------------------------
def test_cerebellum_4_is_unchanged():
    """CEREBELLUM_4 is the MATLAB-validated layout; pin it literally.

    test_parity.py already proves the numbers match MATLAB, but it builds its
    config from the reference .mat rather than from this default -- so the
    default itself is untested there. This pins it.
    """
    assert CEREBELLUM_4 == {
        "Laterale_L": Box(21, 26, -34, -29),
        "Verme_L": Box(22, 27, -12, -7),
        "Laterale_R": Box(21, 26, 29, 34),
        "Verme_R": Box(22, 27, 0, 5),
    }
    assert list(CEREBELLUM_4) == ["Laterale_L", "Verme_L", "Laterale_R", "Verme_R"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
