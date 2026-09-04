"""The ROI editor's geometry contract, driven through the real Qt widgets.

The editor is the one place where a wrong number is *invisible*: the boxes still
land somewhere, the means still compute, and the correlation matrix still looks
publishable. So the mapping is what gets tested, in both directions:

    session.boxes  --pixel_bounds-->  pg.RectShape on screen  --read back-->  session.boxes

plus the two invariants the whole design rests on, both consequences of **boxes being
offsets from Bregma**:

* moving Bregma translates every box rigidly, and must never rewrite an offset or
  resize a box;
* moving Lambda changes the unit those offsets are measured in, so it scales the
  constellation about Bregma -- and must do so reversibly, from a fixed reference
  rather than from its own last output.

Runs headless: ``QT_QPA_PLATFORM=offscreen`` is set before Qt is imported, so no
window opens and no display is needed. Skipped entirely when pyqtgraph is absent.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Must precede any Qt import, hence before roi_editor's editor is built.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pyqtgraph", reason="roi_editor's window needs pyqtgraph + a Qt binding")

from roi_editor import (  # noqa: E402
    DEFAULT_LAMBDA_OFFSET,
    MIN_LAMBDA_OFFSET,
    ROIEditor,
    Session,
    pixel_bounds,
    rotate_boxes,
    scale_boxes,
)
from wfci.atlases import CORTEX_22  # noqa: E402

GRID = (128, 128)
BREGMA_Y1, BREGMA_X2 = 60, 67           # manifest Bregma 121/134, halved
BOX_FIELDS = ("row_start", "row_end", "col_start", "col_end")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_editor(n_sessions: int = 1, y_1: int = BREGMA_Y1, x_2: int = BREGMA_X2,
                scale_size: bool = False):
    """An editor over synthetic sessions: image preset, so nothing is decoded."""
    rng = np.random.default_rng(0)
    sessions = [
        # seed / base_* are filled in by Session.__post_init__.
        Session(key=f"sess{i}", folder="(synthetic)", y_1=y_1, x_2=x_2,
                boxes=dict(CORTEX_22.boxes), image=rng.random(GRID))
        for i in range(n_sessions)
    ]
    args = types.SimpleNamespace(channel_order="auto", preview_frames=1,
                                 roi_set_dir="roi_sets", shared_name="shared.yaml",
                                 lambda_offset=DEFAULT_LAMBDA_OFFSET,
                                 lambda_scales_box_size=scale_size)
    editor = ROIEditor(sessions, args, "cortical_gsr")
    editor._rebuild_scene()
    return editor


def bounds(session) -> dict[str, tuple[int, int, int, int]]:
    """Every box as inclusive 0-based pixel bounds."""
    return {label: pixel_bounds(box, session.y_1, session.x_2)
            for label, box in session.boxes.items()}


def offsets(session) -> dict[str, tuple[int, ...]]:
    """Every box as its raw Bregma offsets -- what actually gets saved."""
    return {label: tuple(getattr(box, f) for f in BOX_FIELDS)
            for label, box in session.boxes.items()}


def test_rotate_boxes_moves_centres_around_bregma_and_preserves_size():
    from wfci import Box

    boxes = {"roi": Box(9, 13, 19, 25)}
    rotated = rotate_boxes(boxes, 90)["roi"]

    assert rotated.row_end - rotated.row_start == 4
    assert rotated.col_end - rotated.col_start == 6
    assert (rotated.row_start + rotated.row_end) / 2 == 22
    assert (rotated.col_start + rotated.col_end) / 2 == -11


def test_rotation_shortcuts_rotate_the_whole_constellation():
    editor = make_editor()
    before = offsets(editor.session)

    press(editor, "X", shift=True)

    assert offsets(editor.session) != before
    assert editor.session.dirty


def test_bake_all_images_saves_alignment_tiffs(tmp_path, monkeypatch):
    editor = make_editor(n_sessions=2)
    editor.args.roi_set_dir = str(tmp_path)
    writes = []

    def fake_imwrite(path, image):
        writes.append((Path(path), image.copy()))

    monkeypatch.setattr("roi_editor.tifffile.imwrite", fake_imwrite)
    editor.bake_all_images()

    assert [path for path, _image in writes] == [
        tmp_path / "alignment_images" / "sess0.tif",
        tmp_path / "alignment_images" / "sess1.tif",
    ]
    assert all(image.dtype == np.float32 for _path, image in writes)
    assert all(image.shape == GRID for _path, image in writes)


def test_bake_all_images_loads_existing_tiff_without_rewriting(tmp_path, monkeypatch):
    editor = make_editor()
    editor.args.roi_set_dir = str(tmp_path)
    image_dir = tmp_path / "alignment_images"
    image_dir.mkdir()
    cached = np.full(GRID, 17, dtype=np.float32)
    cached_path = image_dir / "sess0.tif"
    import tifffile
    tifffile.imwrite(cached_path, cached)

    monkeypatch.setattr(
        "roi_editor.tifffile.imwrite",
        lambda *_args, **_kwargs: pytest.fail("an existing preview was rewritten"),
    )
    editor.bake_all_images()

    np.testing.assert_array_equal(editor.sessions[0].image, cached)


def press(editor, key_name: str, ctrl: bool = False, shift: bool = False) -> None:
    """Fire a real QKeyEvent at the editor's key handler."""
    from pyqtgraph.Qt import QtCore, QtGui

    Qt = QtCore.Qt
    mods = Qt.NoModifier
    if ctrl:
        mods |= Qt.ControlModifier
    if shift:
        mods |= Qt.ShiftModifier
    editor._on_key(QtGui.QKeyEvent(QtCore.QEvent.KeyPress,
                                   getattr(Qt, f"Key_{key_name}"), mods))


def drag_target(editor, d_row: int, d_col: int) -> None:
    """Move the Bregma marker as a user drag would (emits sigPositionChanged)."""
    session = editor.session
    editor.target.setPos(session.x_2 - 0.5 + d_col, session.y_1 - 0.5 + d_row)


def drag_lambda(editor, row: float, col: float | None = None) -> None:
    """Move the Lambda marker as a user drag would, to data (col, row)."""
    session = editor.session
    editor.lambda_target.setPos(session.x_2 - 0.5 if col is None else col, row)


def centres(session) -> dict[str, tuple[float, float]]:
    """Every box's centre, in Bregma-offset units -- what a rescale multiplies."""
    return {label: ((box.row_start + box.row_end) / 2, (box.col_start + box.col_end) / 2)
            for label, box in session.boxes.items()}


def sizes(session) -> dict[str, tuple[int, int]]:
    return {label: (box.row_end - box.row_start + 1, box.col_end - box.col_start + 1)
            for label, box in session.boxes.items()}


# ---------------------------------------------------------------------------
# The import must stay GUI-free
# ---------------------------------------------------------------------------
def test_importing_roi_editor_does_not_pull_in_qt():
    """``from roi_editor import load_roi_set`` must work on a headless machine.

    Both run scripts do exactly that to read a ROI set. If Qt were imported at
    module level, an unattended 15 GB batch would die on a box with no display --
    which is why the editor imports pyqtgraph inside ``_build_ui``.
    """
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import roi_editor;"
        "gui = sorted({m.split('.')[0] for m in sys.modules"
        " if m.split('.')[0] in ('PySide6','PySide2','PyQt5','PyQt6','pyqtgraph')});"
        "print(','.join(gui));"
        "assert callable(roi_editor.load_roi_set)" % REPO_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "", f"importing roi_editor pulled in {out.stdout.strip()}"


# ---------------------------------------------------------------------------
# session.boxes <-> the on-screen items
# ---------------------------------------------------------------------------
def test_every_box_is_placed_on_the_pixels_pixel_bounds_says():
    """pyqtgraph draws pixel i over [i, i+1), so r0..r1 inclusive is y=r0, h=r1-r0+1.

    Off-by-one here would shift every ROI by a pixel while still producing numbers.
    """
    editor = make_editor()
    expected = bounds(editor.session)
    for label, roi in editor._rois.items():
        r0, r1, c0, c1 = expected[label]
        pos, size = roi.pos(), roi.size()
        assert (int(pos.y()), int(pos.x())) == (r0, c0), f"{label} position"
        assert (int(size.y()), int(size.x())) == (r1 - r0 + 1, c1 - c0 + 1), f"{label} size"


def test_reading_the_items_back_is_the_identity():
    """A no-op user edit must not move anything.

    ``_on_roi_changed`` is what every drag and resize goes through, so if it does not
    round-trip, merely touching a box would nudge it.
    """
    editor = make_editor()
    before = offsets(editor.session)
    for label in list(editor._rois):
        editor._on_roi_changed(label)
    assert offsets(editor.session) == before


def test_bregma_marker_sits_on_the_bregma_pixel_centre():
    """Offset 0 is pixel (y_1-1, x_2-1); its centre is (x_2-0.5, y_1-0.5)."""
    editor = make_editor()
    session = editor.session
    pos = editor.target.pos()
    assert (pos.x(), pos.y()) == (session.x_2 - 0.5, session.y_1 - 0.5)


def test_double_click_places_bregma_on_the_clicked_pixel():
    """The inverse of the half-pixel the marker is drawn at."""
    editor = make_editor()
    view_pos = types.SimpleNamespace(x=lambda: 70.8, y=lambda: 55.2)
    editor.plot.vb.mapSceneToView = lambda _p: view_pos
    editor._on_scene_clicked(types.SimpleNamespace(double=lambda: True,
                                                  scenePos=lambda: None))
    # floor(55.2) = pixel 55 -> y_1 = 56;  floor(70.8) = pixel 70 -> x_2 = 71
    assert (editor.session.y_1, editor.session.x_2) == (56, 71)


def test_single_click_does_not_move_bregma():
    """Only a *double* click teleports Bregma -- a stray click must not move it."""
    editor = make_editor()
    before = (editor.session.y_1, editor.session.x_2)
    editor._on_scene_clicked(types.SimpleNamespace(double=lambda: False,
                                                  scenePos=lambda: None))
    assert (editor.session.y_1, editor.session.x_2) == before


# ---------------------------------------------------------------------------
# The invariant: Bregma carries every box, rigidly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d_row, d_col", [(7, -4), (-3, 5), (0, 2), (-1, 0)])
def test_dragging_bregma_translates_every_box_rigidly(d_row, d_col):
    editor = make_editor()
    session = editor.session
    before = bounds(session)
    drag_target(editor, d_row, d_col)
    after = bounds(session)

    shifts = {(after[l][0] - before[l][0], after[l][2] - before[l][2]) for l in before}
    assert shifts == {(d_row, d_col)}, "boxes did not all translate by the same amount"
    sizes = lambda b: {l: (v[1] - v[0], v[3] - v[2]) for l, v in b.items()}
    assert sizes(before) == sizes(after), "a box changed size"


def test_moving_bregma_never_rewrites_a_box_offset():
    """The offsets are what gets SAVED; a Bregma move must leave them alone."""
    editor = make_editor()
    before = offsets(editor.session)
    drag_target(editor, 6, -5)
    press(editor, "Down", ctrl=True)
    press(editor, "Right", ctrl=True, shift=True)
    assert offsets(editor.session) == before


def test_the_on_screen_items_follow_the_model():
    """Not just the data: the RectROIs themselves must move, or the view lies."""
    editor = make_editor()
    drag_target(editor, 7, -4)
    expected = bounds(editor.session)
    for label, roi in editor._rois.items():
        r0, _r1, c0, _c1 = expected[label]
        pos = roi.pos()
        assert (int(pos.y()), int(pos.x())) == (r0, c0), f"{label} did not follow"


def test_programmatic_sync_is_never_read_back_as_a_user_edit():
    """Qt emits sigRegionChanged for a setPos() exactly as it does for a drag.

    So the signal *will* fire while we reposition boxes after a Bregma move; every
    one of those deliveries must find ``_syncing`` set. Otherwise the sync is read
    back as an edit and recurses.
    """
    editor = make_editor()
    guard_seen: list[bool] = []
    original = editor._on_roi_changed
    editor._on_roi_changed = lambda label: (guard_seen.append(editor._syncing),
                                            original(label))[1]
    drag_target(editor, 3, 0)
    assert guard_seen, "expected sigRegionChanged to fire during the sync"
    assert all(guard_seen), "an emit slipped through unguarded -- feedback loop"


# ---------------------------------------------------------------------------
# Lambda: the scale reference
# ---------------------------------------------------------------------------
def test_lambda_marker_sits_the_declared_distance_below_bregma():
    """The distance on screen must BE lambda_offset, or the ruler lies.

    Both markers are drawn at a pixel centre, so the gap between them is exactly the
    offset -- that is the whole reason Lambda shares the boxes' offset convention.
    """
    editor = make_editor()
    session = editor.session
    gap = editor.lambda_target.pos().y() - editor.target.pos().y()
    assert gap == session.lambda_offset
    assert editor.lambda_target.pos().x() == editor.target.pos().x(), "off the midline"


def test_scaling_is_the_identity_at_factor_one():
    """A no-op Lambda change must not move a single box by a rounding step."""
    editor = make_editor()
    session = editor.session
    before = offsets(session)
    editor._set_lambda(session.lambda_offset)
    assert offsets(session) == before
    assert scale_boxes(dict(CORTEX_22.boxes), 1.0) == dict(CORTEX_22.boxes)


def test_stretching_lambda_scales_every_box_centre_about_bregma():
    """Centres land within half a pixel of factor x their old position.

    Half a pixel is the floor, not slack: every CORTEX_22 box is 6 px wide, and a box
    with an even size can only ever be centred on a half-integer -- so an exactly
    doubled centre is often not a reachable position at all.
    """
    editor = make_editor()
    session = editor.session
    before, before_sizes = centres(session), sizes(session)
    editor._set_lambda(2 * session.lambda_offset)       # factor exactly 2

    assert session.lambda_scale == 2.0
    for label, (row_c, col_c) in centres(session).items():
        assert abs(row_c - 2 * before[label][0]) <= 0.5, label
        assert abs(col_c - 2 * before[label][1]) <= 0.5, label
    assert sizes(session) == before_sizes, "box sizes must not change by default"


def test_bregma_does_not_move_when_the_layout_is_rescaled():
    """Scaling is about Bregma: the anchor is the one thing that stays put."""
    editor = make_editor()
    session = editor.session
    anchor = (session.y_1, session.x_2)
    editor._set_lambda(session.lambda_offset + 9)
    assert (session.y_1, session.x_2) == anchor


def test_rescaling_keeps_the_left_right_symmetry():
    """CORTEX_22 is 11 mirrored pairs; a rescale that breaks that is a rescale
    that has silently moved one hemisphere relative to the other."""
    editor = make_editor()
    editor._set_lambda(37)                              # a factor with no exact halves
    boxes = editor.session.boxes
    for left, right in zip(CORTEX_22.labels[:11], CORTEX_22.labels[11:]):
        l_box, r_box = boxes[left], boxes[right]
        assert (l_box.row_start, l_box.row_end) == (r_box.row_start, r_box.row_end)
        assert l_box.col_start == -r_box.col_end, f"{left}/{right} no longer mirrored"
        assert l_box.col_end == -r_box.col_start, f"{left}/{right} no longer mirrored"


def test_scaling_out_and_back_returns_the_exact_same_integers():
    """The reason a reference layout is kept instead of rescaling in place.

    Deriving each scale from the previous *result* would round twice and drift; every
    scale is derived from base_boxes, so this round-trips exactly.
    """
    editor = make_editor()
    session = editor.session
    before = offsets(session)
    for distance in (31, 44, 17, 33, DEFAULT_LAMBDA_OFFSET):
        editor._set_lambda(distance)
    assert offsets(session) == before


def test_a_hand_edited_box_survives_the_next_rescale():
    """A box the user placed by hand becomes part of the reference (rebase).

    Otherwise the next Lambda nudge would recompute it from the atlas and throw the
    adjustment away -- the layout would fight back.
    """
    editor = make_editor()
    session = editor.session
    editor._select("V1R")
    press(editor, "Down")                    # nudge V1R one pixel down
    nudged = session.boxes["V1R"]
    assert session.base_boxes["V1R"] == nudged, "the edit was not taken as reference"

    editor._set_lambda(session.lambda_offset + 3)
    editor._set_lambda(session.lambda_offset - 3)
    assert session.boxes["V1R"] == nudged


def test_dragging_lambda_sideways_does_not_tilt_the_layout():
    """Only the row is read: an off-midline Lambda would mean a rotated brain."""
    editor = make_editor()
    session = editor.session
    target_row = session.y_1 + 40 - 0.5
    drag_lambda(editor, row=target_row, col=session.x_2 + 25)

    assert session.lambda_offset == 40
    # ... and the marker snaps back onto the midline rather than staying where dropped.
    assert editor.lambda_target.pos().x() == session.x_2 - 0.5


def test_lambda_is_clamped_to_a_usable_ruler_and_to_the_frame():
    editor = make_editor()
    session = editor.session
    editor._set_lambda(0)                                 # would divide by zero
    assert session.lambda_offset == MIN_LAMBDA_OFFSET
    editor._set_lambda(10_000)                            # far below the frame
    assert session.lambda_row == GRID[0], "Lambda left the frame"


def test_lambda_travels_with_bregma_without_changing_the_scale():
    """It is stored as a distance, so translating the layout must not rescale it."""
    editor = make_editor()
    session = editor.session
    scale, offset = session.lambda_scale, session.lambda_offset
    drag_target(editor, 9, -6)

    assert (session.lambda_scale, session.lambda_offset) == (scale, offset)
    gap = editor.lambda_target.pos().y() - editor.target.pos().y()
    assert gap == offset


def test_scale_size_option_scales_the_boxes_too():
    """The opt-in true similarity transform, for boxes meant to cover a fraction
    of cortex rather than a fixed area."""
    editor = make_editor(scale_size=True)
    session = editor.session
    before = sizes(session)
    editor._set_lambda(2 * session.lambda_offset)
    grown = sizes(session)
    assert grown != before
    # V1R is 6x6 at offsets 41..46 / 25..30; doubled that is 82..92 / 50..60 = 11x11.
    assert grown["V1R"] == (11, 11)


@pytest.mark.parametrize("key_name, shift, expected_delta", [
    ("Period", False, 1), ("Greater", True, 5),
    ("Comma", False, -1), ("Less", True, -5),
])
def test_comma_and_period_change_the_lambda_distance(key_name, shift, expected_delta):
    editor = make_editor()
    session = editor.session
    before = session.lambda_offset
    press(editor, key_name, shift=shift)
    assert session.lambda_offset - before == expected_delta


def test_l_returns_the_layout_to_scale_one():
    editor = make_editor()
    session = editor.session
    before = offsets(session)
    editor._set_lambda(41)
    assert offsets(session) != before
    press(editor, "L")
    assert session.lambda_scale == 1.0
    assert offsets(session) == before


def test_reset_restores_the_lambda_distance_too():
    editor = make_editor()
    session = editor.session
    before = (session.lambda_offset, offsets(session))
    editor._set_lambda(45)
    press(editor, "R")
    assert (session.lambda_offset, offsets(session)) == before
    assert session.lambda_scale == 1.0, "the reference was not re-anchored"


def test_apply_to_all_carries_the_scale_with_the_boxes():
    """The copied offsets ARE the scaled ones, so the distance describing them must
    come along or the other page would claim a scale it is not at."""
    editor = make_editor(n_sessions=2)
    editor._set_lambda(39)
    editor._apply_to_all(with_bregma=False)

    other = editor.sessions[1]
    assert other.lambda_offset == 39
    assert other.boxes == editor.sessions[0].boxes
    assert other.lambda_scale == 1.0, "the copy must be its own reference"


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key_name, shift, expected", [
    ("Down", False, (1, 0)), ("Down", True, (5, 0)),
    ("Up", False, (-1, 0)), ("Up", True, (-5, 0)),
    ("Right", False, (0, 1)), ("Right", True, (0, 5)),
    ("Left", False, (0, -1)), ("Left", True, (0, -5)),
])
def test_ctrl_arrows_move_bregma_and_all_boxes(key_name, shift, expected):
    editor = make_editor()
    session = editor.session
    before, y_1, x_2 = bounds(session), session.y_1, session.x_2
    press(editor, key_name, ctrl=True, shift=shift)

    assert (session.y_1 - y_1, session.x_2 - x_2) == expected
    after = bounds(session)
    shifts = {(after[l][0] - before[l][0], after[l][2] - before[l][2]) for l in before}
    assert shifts == {expected}, "boxes did not track Bregma"


def test_plain_arrows_move_only_the_selected_box():
    editor = make_editor()
    session = editor.session
    editor._select("V1R")
    before, anchor = bounds(session), (session.y_1, session.x_2)
    press(editor, "Down")

    assert (session.y_1, session.x_2) == anchor, "a plain arrow moved Bregma"
    moved = [l for l in before if before[l] != bounds(session)[l]]
    assert moved == ["V1R"]


def test_arrows_with_nothing_selected_say_so_instead_of_guessing():
    editor = make_editor()
    before = offsets(editor.session)
    press(editor, "Down")
    assert offsets(editor.session) == before
    assert "No ROI selected" in editor.message


def test_shift_disambiguates_save_one_from_save_all():
    """Qt reports the letter and the modifier separately (matplotlib gave 's'/'S')."""
    editor = make_editor(n_sessions=2)
    seen: list[str] = []
    editor._save_current = lambda: seen.append("current")
    editor._save_all = lambda: seen.append("all")
    press(editor, "S")
    press(editor, "S", shift=True)
    assert seen == ["current", "all"]


def test_shift_disambiguates_apply_boxes_from_apply_boxes_and_bregma():
    editor = make_editor(n_sessions=2)
    seen: list[bool] = []
    editor._apply_to_all = lambda with_bregma: seen.append(with_bregma)
    press(editor, "A")
    press(editor, "A", shift=True)
    assert seen == [False, True]


def test_n_and_p_page_through_sessions_and_rebuild_the_scene():
    editor = make_editor(n_sessions=3)
    press(editor, "N")
    assert editor.index == 1
    assert len(editor._rois) == len(CORTEX_22), "scene not rebuilt for the new page"
    press(editor, "P")
    press(editor, "P")
    assert editor.index == 2, "paging should wrap"


def test_reset_restores_the_starting_layout():
    editor = make_editor()
    session = editor.session
    before_offsets, before_anchor = offsets(session), (session.y_1, session.x_2)
    drag_target(editor, 8, 8)
    editor._select("V1R")
    press(editor, "Plus")
    assert offsets(session) != before_offsets or (session.y_1, session.x_2) != before_anchor

    press(editor, "R")
    assert offsets(session) == before_offsets
    assert (session.y_1, session.x_2) == before_anchor
    assert session.dirty is False


def test_grow_then_shrink_round_trips_and_offsets_stay_integers():
    editor = make_editor()
    session = editor.session
    editor._select("V1R")
    before = bounds(session)["V1R"]
    press(editor, "Plus")
    r0, r1, c0, c1 = bounds(session)["V1R"]
    assert (r1 - r0, c1 - c0) == (before[1] - before[0] + 2, before[3] - before[2] + 2)
    press(editor, "Minus")
    assert bounds(session)["V1R"] == before

    box = session.boxes["V1R"]
    assert all(isinstance(getattr(box, f), int) for f in BOX_FIELDS), \
        "a fractional drag reached a Box"


def test_contrast_keys_change_levels_but_not_geometry():
    editor = make_editor()
    before = offsets(editor.session)
    # getLevels() hands back an ndarray, so compare as a tuple of floats.
    levels_before = tuple(float(v) for v in editor.image_item.getLevels())
    press(editor, "BracketRight")
    levels_after = tuple(float(v) for v in editor.image_item.getLevels())
    assert levels_after != levels_before
    assert offsets(editor.session) == before


# ---------------------------------------------------------------------------
# Off-frame boxes: refused, not silently saved
# ---------------------------------------------------------------------------
def test_off_frame_boxes_are_flagged_reddened_and_block_the_save(tmp_path):
    """An off-frame box is a *negative index* in NumPy, not a crash.

    A left-hemisphere ROI would quietly average the right side of the brain, so the
    save has to refuse -- and say which boxes, so it can be acted on.
    """
    editor = make_editor()
    session = editor.session
    editor.target.setPos(4.5, 4.5)              # slam the layout into the corner

    bad = editor._out_of_frame(session)
    assert bad, "expected boxes off the frame near the corner"
    assert "OFF FRAME" in editor.msg_label.text()
    assert str(len(bad)) in editor.msg_label.text()

    reddened = {l for l, roi in editor._rois.items()
                if roi.pen.color().name() == "#ff0000"}
    assert reddened == set(bad), "the red boxes must be exactly the off-frame ones"

    out = tmp_path / "refused.yaml"
    assert editor._save(session, out, "refused") is False
    assert not out.exists(), "a refused save still wrote a file"


def test_a_valid_layout_saves_and_round_trips(tmp_path):
    editor = make_editor()
    session = editor.session
    editor._set_lambda(34)                  # save a RESCALED layout, not the pristine one
    out = tmp_path / "ok.yaml"
    assert editor._save(session, out, "ok") is True
    assert out.is_file()

    from roi_editor import load_lambda_offset, load_roi_set

    atlas, row, col = load_roi_set(out)
    assert atlas.labels == CORTEX_22.labels, "box ORDER is the column order of R"
    # The file's boxes are the SCALED ones -- the pipeline must not have to rescale.
    assert dict(atlas) == dict(session.boxes)
    assert dict(atlas) != dict(CORTEX_22.boxes)
    assert (row, col) == (session.bregma_row, session.bregma_col)
    assert load_lambda_offset(out) == 34
    import yaml
    assert yaml.safe_load(out.read_text(encoding="utf-8"))["downsample"] == 0.5
    assert session.dirty is False


def test_the_shipped_22_box_set_declares_the_distance_it_was_drawn_at(tmp_path):
    """roi_sets/cortex22_roi_set.yaml is what the run scripts point at, and what the
    editor seeds from -- reopening it must resume at scale 1.000x, not rescale it."""
    from roi_editor import load_lambda_offset, load_roi_set

    path = REPO_ROOT / "roi_sets" / "cortex22_roi_set.yaml"
    atlas, _row, _col = load_roi_set(path)
    assert dict(atlas) == dict(CORTEX_22.boxes), "the shipped set is CORTEX_22 unscaled"
    assert load_lambda_offset(path) == DEFAULT_LAMBDA_OFFSET
