"""pixel_roi_editor.py helpers, and the --roi-set-dir override it feeds in the pixel scripts."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from epileptic_by_active_pixels import ActivePixelTraceLoader
from epileptic_by_area_animal_day_pixels import PixelTraceLoader, recording_roi_set, roi_output_suffix
from pixel_roi_editor import (
    TEMPLATE_NAME, Page, apply_template, box_trace, boxes_outside_crop, cross_hemisphere_overlaps,
    discover_recordings, hemisphere_twin, is_valid_roi_name, load_page_data, load_template, recording_key,
    save_page_roi_set, save_template, seed_r_map, transfer_boxes,
)
from roi_editor import mirror_box
from roi_editor import save_roi_set
from roi_pixel_connectivity import analyze_recording, pixel_correlation
from wfci import Box

REGION = [30, 70, 40, 90]  # full-grid rows 30..69, cols 40..89 (a 40 x 50 crop)
Y_1, X_2 = 45, 65          # Bregma on the full grid, MATLAB 1-based
N_TIME = 60


def make_dump(root: Path, animal: str = "PV5", lambda_offset: int = 49) -> Path:
    """A small synthetic pixel dump folder with its own ROI set, like run_botox_batch writes."""
    folder = root / "pixel_data" / f"260611_{animal}" / "t1"
    folder.mkdir(parents=True)
    own_set = save_roi_set(root / "rebuilt" / f"260611_{animal}_t1.yaml",
                           {"V1L": Box(10, 15, -10, -5), "V1R": Box(10, 15, 5, 10)},
                           Y_1 * 2, X_2 * 2, (128, 128), name=f"260611_{animal}_t1", source="test",
                           lambda_offset=lambda_offset)
    rng = np.random.default_rng(0)
    shape = (N_TIME, REGION[1] - REGION[0], REGION[3] - REGION[2])
    f_gcamp = (1000 + 50 * rng.standard_normal(shape)).astype(np.float32)
    f_emo = (5000 + 80 * rng.standard_normal(shape)).astype(np.float32)
    np.save(folder / "pixels_f_gcamp_full.npy", f_gcamp)
    np.save(folder / "pixels_f_emo_full.npy", f_emo)
    np.save(folder / "pixels_dff_full.npy", rng.standard_normal(shape).astype(np.float16))
    np.savez(folder / "pixels_meta_full.npz", axis_order="time,y,x", n_time=N_TIME, n_written=N_TIME,
             roi_set=str(own_set), grid=np.array([128, 128]), bregma_row=Y_1 * 2, bregma_col=X_2 * 2,
             downsample=0.5, region=np.array(REGION), y_1=Y_1, x_2=X_2, shape=np.array(shape),
             day="260611", animal=animal, recording="t1", group="P", roi_labels=np.array(["V1L", "V1R"]),
             mean_f=np.zeros((4, 4)), mean_r=np.zeros((4, 4)))
    return folder


def test_names_and_twins_follow_the_detectors_hemisphere_rule():
    assert hemisphere_twin("M2L_alta") == "M2R_alta"
    assert hemisphere_twin("RSL_alta") == "RSR_alta"   # leading R is not the side
    assert hemisphere_twin("S1R") == "S1L"
    assert hemisphere_twin("Midline") is None
    assert is_valid_roi_name("S1L") and is_valid_roi_name("M2R_alta_2")
    for bad in ("L", "S1", "S1L-x", "S1 L", "", "S1L_"):
        assert not is_valid_roi_name(bad), bad


def test_crop_and_overlap_checks():
    # Crop rows 30..69: row_start -14 puts the first pixel at 0-based row 30, the crop edge.
    inside, outside = Box(-14, -10, 0, 3), Box(-16, -12, 0, 3)
    assert boxes_outside_crop({"AL": inside, "BL": outside}, Y_1, X_2, REGION) == ["BL"]
    boxes = {"AL": Box(0, 5, -3, 1), "AR": Box(0, 5, -1, 3), "CL": Box(20, 25, -9, -4)}
    assert cross_hemisphere_overlaps(boxes) == [("AL", "AR")]


def test_discover_recordings_and_key(tmp_path):
    folder = make_dump(tmp_path)
    assert discover_recordings(tmp_path / "pixel_data", None) == [folder]
    assert discover_recordings(tmp_path / "pixel_data", ["260611_PV5/*"]) == [folder]
    assert recording_key(folder) == "260611_PV5_t1"
    with pytest.raises(ValueError, match="match no dump"):
        discover_recordings(tmp_path / "pixel_data", ["999999_X/*"])


def test_page_starts_from_own_set_then_resumes_from_saved(tmp_path):
    folder = make_dump(tmp_path)
    out_dir = tmp_path / "selected"
    page = Page(folder=folder, key=recording_key(folder))
    page.load(out_dir, None)
    assert list(page.boxes) == ["V1L", "V1R"] and page.own_lambda == 49
    new_boxes = {"S1L": Box(0, 3, -12, -9), "S1R": Box(0, 3, 9, 12)}
    save_page_roi_set(out_dir, page.key, new_boxes, page.meta, page.own_lambda, "test")
    resumed = Page(folder=folder, key=page.key)
    resumed.load(out_dir, None)
    assert resumed.boxes == new_boxes and resumed.start_source.startswith("resumed")
    saved = yaml.safe_load((out_dir / "260611_PV5_t1.yaml").read_text())
    assert (saved["bregma_row"], saved["bregma_col"], saved["downsample"]) == (90, 130, 0.5)


def test_save_refuses_boxes_outside_the_crop(tmp_path):
    folder = make_dump(tmp_path)
    page = Page(folder=folder, key=recording_key(folder))
    page.load(tmp_path / "selected", None)
    with pytest.raises(ValueError, match="outside the saved crop"):
        save_page_roi_set(tmp_path / "selected", page.key, {"XL": Box(-30, -25, 0, 3)}, page.meta, None, "t")


def test_override_is_what_the_pixel_scripts_compute_on(tmp_path):
    folder = make_dump(tmp_path)
    out_dir = tmp_path / "selected"
    page = Page(folder=folder, key=recording_key(folder))
    page.load(out_dir, None)
    new_boxes = {"S1L": Box(0, 3, -12, -9), "S1R": Box(0, 3, 9, 12), "M2L": Box(-10, -8, -4, -2),
                 "M2R": Box(-10, -8, 2, 4)}
    save_page_roi_set(out_dir, page.key, new_boxes, page.meta, page.own_lambda, "test")
    meta_path = folder / "pixels_meta_full.npz"

    # ROI-mean detector input: F * mean_t(R) / R averaged over each NEW box.
    loader = PixelTraceLoader(tmp_path / "out", all_rois=True, signal_mode="reflectance_ratio",
                              roi_set_dir=out_dir)
    table = loader(meta_path)
    assert list(table.columns) == ["trial", "frame", *new_boxes]
    f = np.load(folder / "pixels_f_gcamp_full.npy").astype(np.float64)
    r = np.load(folder / "pixels_f_emo_full.npy").astype(np.float64)
    corrected = f * r.mean(axis=0) / r
    for label, box in new_boxes.items():
        np.testing.assert_allclose(table[label], box_trace(corrected, box, Y_1, X_2, REGION), rtol=1e-12)

    # Seed connectivity: seeds are the new boxes' means of the chosen volume.
    result_path = tmp_path / "conn" / "roi_pixel_connectivity.npz"
    analyze_recording(dict(folder=str(folder), output_path=str(result_path),
                           volume_file="pixels_dff_full.npy", roi_set_dir=out_dir))
    with np.load(result_path) as result:
        assert result["roi_labels"].tolist() == list(new_boxes)
        dff = np.load(folder / "pixels_dff_full.npy").astype(np.float64)
        np.testing.assert_allclose(result["seed_traces"][:, 0], box_trace(dff, new_boxes["S1L"], Y_1, X_2, REGION))

    # Active pixels: hemisphere masks are the union of the NEW boxes.
    active = ActivePixelTraceLoader(tmp_path / "active", dff_source="pipeline_dff", pixel_z_threshold=1.0,
                                    roi_set_dir=out_dir)
    active(meta_path)
    geometry = json.loads((tmp_path / "active" / "260611_PV5" / "t1" / "active_pixel_geometry.json").read_text())
    assert set(geometry["boxes"]) == set(new_boxes)
    assert geometry["n_mask_pixels"] == {"CortexL_active": 16 + 9, "CortexR_active": 16 + 9}


def test_override_never_falls_back_and_checks_bregma(tmp_path):
    folder = make_dump(tmp_path)
    meta = Page(folder=folder, key="k")
    meta.load(tmp_path / "selected", None)
    with pytest.raises(FileNotFoundError, match="No ROI set"):
        recording_roi_set(meta.meta, tmp_path / "empty")
    save_roi_set(tmp_path / "shifted" / "260611_PV5_t1.yaml", {"S1L": Box(0, 3, -12, -9)},
                 Y_1 * 2 + 2, X_2 * 2, (128, 128), name="x", source="t")
    with pytest.raises(ValueError, match="bregma_row"):
        recording_roi_set(meta.meta, tmp_path / "shifted")
    assert roi_output_suffix(None) == "" and roi_output_suffix(tmp_path / "pixel_selected") == "_roi_pixel_selected"


def test_seed_r_map_matches_pixel_correlation(tmp_path):
    folder = make_dump(tmp_path)
    data = load_page_data(folder, "pixels_dff_full.npy")
    trace = box_trace(data.volume, Box(0, 3, -12, -9), Y_1, X_2, REGION)
    pixels = data.volume.reshape(N_TIME, -1).astype(np.float64)
    expected = pixel_correlation(trace[:, None], pixels).reshape(data.sd.shape)
    np.testing.assert_allclose(seed_r_map(data, trace), expected, atol=1e-12)
    # Pixels inside the seed box correlate with it positively.
    assert seed_r_map(data, trace)[Y_1 - 1 - REGION[0], X_2 - 13 - REGION[2]] > 0


def test_transfer_scales_offsets_about_bregma_and_keeps_pairs_mirrored():
    template = {"S1L": Box(0, 5, -20, -15), "S1R": mirror_box(Box(0, 5, -20, -15)), "M2L": Box(-20, -15, -8, -3),
                "M2R": Box(-20, -15, 3, 8)}
    assert transfer_boxes(template, 49, 49) == template          # same Lambda: identity
    for target in range(42, 57):
        moved = transfer_boxes(template, 49, target)
        assert moved["S1R"] == mirror_box(moved["S1L"]) and moved["M2R"] == mirror_box(moved["M2L"])
        for label, box in moved.items():               # sizes fixed, centres scaled
            original = template[label]
            assert box.row_end - box.row_start == original.row_end - original.row_start
            centre = (box.col_start + box.col_end) / 2
            assert abs(centre - (original.col_start + original.col_end) / 2 * target / 49) <= 0.5
    with pytest.raises(ValueError, match="positive"):
        transfer_boxes(template, 0, 49)


def test_one_template_writes_every_recording_by_its_own_lambda(tmp_path):
    reference = make_dump(tmp_path, "PV5", 49)
    other = make_dump(tmp_path, "R1", 56)
    out_dir = tmp_path / "selected"
    template = {"S1L": Box(0, 3, -12, -9), "S1R": Box(0, 3, 9, 12)}
    page = Page(folder=reference, key=recording_key(reference))
    page.load(out_dir, None)
    save_template(out_dir, template, page.meta, page.key, page.own_lambda, scale_size=False)
    boxes, template_lambda = load_template(out_dir / TEMPLATE_NAME)
    assert boxes == template and template_lambda == 49
    written = apply_template(boxes, template_lambda, [reference, other], out_dir, False, "test")
    assert [path.name for path in written] == ["260611_PV5_t1.yaml", "260611_R1_t1.yaml"]
    for folder, lambda_offset in ((reference, 49), (other, 56)):
        saved = yaml.safe_load((out_dir / f"{recording_key(folder)}.yaml").read_text())
        assert saved["lambda_row_offset"] == lambda_offset
        expected = transfer_boxes(template, 49, lambda_offset)
        assert {label: Box(**spec) for label, spec in saved["boxes"].items()} == expected
    # The pixel scripts read the transferred sets.
    table = PixelTraceLoader(tmp_path / "out", all_rois=True, signal_mode="reflectance_ratio",
                             roi_set_dir=out_dir)(other / "pixels_meta_full.npz")
    assert list(table.columns) == ["trial", "frame", "S1L", "S1R"]


def test_template_that_leaves_a_crop_writes_nothing(tmp_path):
    reference = make_dump(tmp_path, "PV5", 42)
    other = make_dump(tmp_path, "R1", 56)
    out_dir = tmp_path / "selected"
    # Col offset -24 fits the crop (cols 40..89, Bregma col index 64) at Lambda 42, not at 56.
    template = {"XL": Box(0, 3, -24, -21)}
    with pytest.raises(ValueError, match="nothing written"):
        apply_template(template, 42, [reference, other], out_dir, False, "test")
    assert not out_dir.exists() or not list(out_dir.glob("*.yaml"))
