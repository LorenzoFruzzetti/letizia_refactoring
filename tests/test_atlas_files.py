"""An atlas a study owns: load_atlas / save_atlas.

ROI geometry is experimental design -- it depends on the preparation, the
objective and the field of view, not on the analysis maths. So a study must be
able to keep its layout in a file it owns, next to its Bregma values and trial
list, rather than editing coordinates into the library. The presets stay as
validated defaults; they are a starting point, not a requirement.

The file format is deliberately dumb data: MATLAB-style 1-based inclusive offsets,
written out plainly so a file can be diffed against the original scripts by eye.
It is parsed with yaml.safe_load -- an atlas is data, and must never be able to
execute.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wfci import CEREBELLUM_4, CORTEX_22, ROIConfig, load_atlas, save_atlas
from wfci.atlases import Atlas, as_atlas
from wfci.config import Box
from wfci.roi import extract_roi_timeseries


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_a_preset_round_trips_through_a_file(tmp_path, suffix):
    """save -> load must return the same atlas, in the same order."""
    path = tmp_path / f"atlas{suffix}"

    save_atlas(CORTEX_22, path)
    loaded = load_atlas(path)

    assert loaded == CORTEX_22, "boxes changed through the round trip"
    assert list(loaded) == list(CORTEX_22), "column ORDER changed through the round trip"
    assert loaded.grid == CORTEX_22.grid
    assert loaded.name == CORTEX_22.name


def test_the_saved_order_is_the_atlas_order_not_alphabetical(tmp_path):
    """The box order IS the column order of R.

    A serialiser that sorted keys would silently relabel every matrix built from
    the file -- same shape, same values, wrong meaning. Alphabetical would put
    BFDL first; the atlas starts with M2L_alta.
    """
    path = tmp_path / "atlas.yaml"
    save_atlas(CORTEX_22, path)

    text = path.read_text(encoding="utf-8")
    positions = [text.index(f"{label}:") for label in CORTEX_22]

    assert positions == sorted(positions), "labels are not in atlas order in the file"
    assert list(load_atlas(path))[0] == "M2L_alta"


def test_a_hand_written_yaml_atlas_works(tmp_path):
    """The format has to be writable by a human, not just by save_atlas."""
    path = tmp_path / "mine.yaml"
    path.write_text(
        "name: my_two_roi\n"
        "grid: [128, 128]\n"
        "source: drawn by AB, 4x objective\n"
        "boxes:\n"
        "  left_M1:  {row_start: -17, row_end: -12, col_start: -32, col_end: -27}\n"
        "  right_M1: {row_start: -17, row_end: -12, col_start:  27, col_end:  32}\n",
        encoding="utf-8",
    )

    atlas = load_atlas(path)

    assert list(atlas) == ["left_M1", "right_M1"]
    assert atlas["left_M1"] == Box(-17, -12, -32, -27)
    assert atlas.grid == (128, 128)
    assert atlas.source == "drawn by AB, 4x objective"


def test_choosing_fewer_rois_does_not_change_the_remaining_values(tmp_path):
    """A study's ROI selection is a *view*, not a different analysis.

    Everything upstream of the ROI means -- correction, masking, the global signal
    -- is computed over the whole brain, so dropping ROIs from the atlas must
    leave the surviving ones bit-for-bit unchanged. If it did not, the atlas would
    be silently coupled to the maths and "own your geometry" would be a trap.
    """
    from wfci import CORTICAL_GSR
    from wfci.pipeline import run_profile
    from dataclasses import replace as dc_replace

    rng = np.random.default_rng(4)
    common = 1.0 + 0.2 * np.sin(np.linspace(0, 7.0, 6))
    gain = rng.uniform(0.5, 1.5, (512, 512))
    trials = [(
        rng.uniform(100.0, 140.0, (512, 512, 6)) + 40.0 * gain[:, :, None] * common,
        rng.uniform(600.0, 700.0, (512, 512, 6)),
    )]
    mask = np.zeros((256, 256))
    mask[20:-20, 20:-20] = 1.0

    keep = ["M2L_alta", "M1R_bassa", "V1L"]
    subset = Atlas(
        name="subset",
        boxes={k: CORTEX_22[k] for k in keep},
        grid=CORTEX_22.grid,
    )

    full = run_profile(trials, ROIConfig(y_1=63, x_2=63), CORTICAL_GSR, mask=mask)
    few = run_profile(
        trials, ROIConfig(y_1=63, x_2=63),
        dc_replace(CORTICAL_GSR, atlas=subset), mask=mask,
    )

    all_labels = list(CORTEX_22)
    for j, label in enumerate(keep):
        np.testing.assert_array_equal(
            few.temp_roi[:, j, :], full.temp_roi[:, all_labels.index(label), :],
            err_msg=f"{label} changed when other ROIs were dropped",
        )


def test_a_loaded_atlas_drives_the_pipeline(tmp_path):
    """The point of the feature: a study's file replaces the built-in layout."""
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({
        "name": "three_roi",
        "grid": [128, 128],
        "boxes": {
            "a": {"row_start": 1, "row_end": 6, "col_start": 1, "col_end": 6},
            "b": {"row_start": 1, "row_end": 6, "col_start": -6, "col_end": -1},
            "c": {"row_start": 10, "row_end": 15, "col_start": -3, "col_end": 2},
        },
    }), encoding="utf-8")

    atlas = load_atlas(path)
    cfg = ROIConfig(y_1=63, x_2=63, boxes=dict(atlas))
    stack = np.random.default_rng(0).uniform(1.0, 2.0, (128, 128, 5, 2))

    temp_roi = extract_roi_timeseries(stack, cfg, expected_grid=atlas.grid)

    assert temp_roi.shape == (5, 3, 2)
    assert cfg.labels == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Bad files fail loudly
# ---------------------------------------------------------------------------
def test_a_missing_box_field_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: x\nboxes:\n  a: {row_start: 1, row_end: 6, col_start: 1}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing.*col_end"):
        load_atlas(path)


def test_a_typo_in_a_box_key_is_rejected_not_ignored(tmp_path):
    """A silently-ignored key would leave the box with an offset it never declared."""
    path = tmp_path / "typo.yaml"
    path.write_text(
        "name: x\nboxes:\n"
        "  a: {row_start: 1, row_end: 6, col_start: 1, col_end: 6, rowstart: 9}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected key"):
        load_atlas(path)


def test_a_file_with_no_boxes_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("name: x\ngrid: [128, 128]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty 'boxes'"):
        load_atlas(path)


def test_non_integer_offsets_are_rejected(tmp_path):
    path = tmp_path / "float.yaml"
    path.write_text(
        "name: x\nboxes:\n  a: {row_start: hello, row_end: 6, col_start: 1, col_end: 6}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-integer"):
        load_atlas(path)


def test_an_unknown_extension_is_rejected(tmp_path):
    path = tmp_path / "atlas.txt"
    path.write_text("name: x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported atlas format"):
        load_atlas(path)


def test_yaml_is_parsed_safely(tmp_path):
    """An atlas file is data. It must never be able to construct objects.

    yaml.load would happily instantiate arbitrary Python from a !!python/object
    tag; safe_load refuses. Atlas files may be shared between labs, so this is a
    real boundary, not a theoretical one.
    """
    import yaml

    path = tmp_path / "evil.yaml"
    path.write_text(
        "name: x\n"
        "boxes: !!python/object/apply:os.system ['echo pwned']\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.YAMLError):
        load_atlas(path)


# ---------------------------------------------------------------------------
# Atlas as a mapping (drop-in for the plain dicts it replaced)
# ---------------------------------------------------------------------------
def test_atlas_behaves_like_the_dict_it_replaced():
    """Everything that took a dict of boxes must keep working."""
    assert dict(CEREBELLUM_4) == {
        "Laterale_L": Box(21, 26, -34, -29),
        "Verme_L": Box(22, 27, -12, -7),
        "Laterale_R": Box(21, 26, 29, 34),
        "Verme_R": Box(22, 27, 0, 5),
    }
    assert list(CEREBELLUM_4) == ["Laterale_L", "Verme_L", "Laterale_R", "Verme_R"]
    assert CEREBELLUM_4["Verme_R"] == Box(22, 27, 0, 5)
    assert len(CEREBELLUM_4) == 4
    assert "Verme_L" in CEREBELLUM_4
    assert CEREBELLUM_4 == dict(CEREBELLUM_4), "Atlas must compare equal to its dict"
    assert [k for k, _ in CEREBELLUM_4.items()] == CEREBELLUM_4.labels


def test_an_atlas_cannot_be_mutated_through_a_caller():
    """A preset is shared process-wide; a caller must not be able to edit it."""
    boxes = {"a": Box(1, 2, 1, 2)}
    atlas = Atlas(name="x", boxes=boxes)

    boxes["a"] = Box(9, 9, 9, 9)      # mutate what we passed in
    boxes["b"] = Box(1, 1, 1, 1)      # and add to it

    assert atlas["a"] == Box(1, 2, 1, 2), "atlas aliased the caller's dict"
    assert "b" not in atlas
    with pytest.raises(TypeError):
        atlas["c"] = Box(1, 1, 1, 1)  # Mapping: no __setitem__


def test_as_atlas_wraps_a_plain_dict_without_inventing_a_grid():
    """grid=None is honest for an ad-hoc dict; a guessed grid would be worse."""
    wrapped = as_atlas({"a": Box(1, 2, 1, 2)})

    assert wrapped.grid is None
    assert as_atlas(CORTEX_22) is CORTEX_22, "an Atlas must pass through unchanged"


def test_an_empty_atlas_is_rejected():
    with pytest.raises(ValueError, match="no boxes"):
        Atlas(name="empty", boxes={})


def test_a_bad_grid_is_rejected():
    with pytest.raises(ValueError, match="grid must be"):
        Atlas(name="x", boxes={"a": Box(1, 2, 1, 2)}, grid=(0, 128))


# ---------------------------------------------------------------------------
# The presets declare what they are
# ---------------------------------------------------------------------------
def test_the_shipped_atlases_declare_their_grid_and_provenance():
    """An atlas without a declared grid cannot be checked; the presets know theirs.

    Both were drawn on the 128x128 final frame (512x512 raw, two 0.5x
    downsamples) -- the cortical scripts say so in their own filenames.
    """
    for atlas in (CEREBELLUM_4, CORTEX_22):
        assert atlas.grid == (128, 128), f"{atlas.name} has no declared grid"
        assert atlas.source, f"{atlas.name} has no provenance"

    # And the provenance must not overstate the cortical path's pedigree.
    assert "MATLAB-validated" in CEREBELLUM_4.source
    assert "NO MATLAB parity reference" in CORTEX_22.source


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-v"]))
