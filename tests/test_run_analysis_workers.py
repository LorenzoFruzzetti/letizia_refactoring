"""run_analysis: the pool gives the serial result, and the minimum-signal gate works."""

import numpy as np
import pandas as pd
import pytest

from epileptic_by_area_animal_day import run_analysis


def _write_cohort(root, n_recordings=3, n_frames=1200, seed=0):
    # Two bilateral "ROIs" of noise with a few large, fast events on top.
    rng = np.random.default_rng(seed)
    paths = []
    for i_rec in range(n_recordings):
        folder = root / "260101_T1" / f"t{i_rec + 1}"
        folder.mkdir(parents=True)
        traces = 0.3 + 0.01 * rng.standard_normal((n_frames, 2))
        for start in rng.choice(np.arange(50, n_frames - 50), 6, replace=False):
            traces[start:start + 10] += rng.uniform(0.05, 0.4)
        table = pd.DataFrame({"trial": 0, "frame": np.arange(n_frames),
                              "M1L": traces[:, 0], "M1R": traces[:, 1]})
        path = folder / "roi_fluorescence_full.csv"
        table.to_csv(path, index=False)
        paths.append(path)
    pd.DataFrame({"day": ["260101"], "animal": ["T1"], "group": ["T"]}).to_csv(
        root / "batch_summary.csv", index=False)
    return paths


def _run(root, out, **options):
    # The synthetic events are far above the noise, so the default (-10, 30) is too narrow.
    options.setdefault("z_histogram_range", (-10.0, 1000.0))
    run_analysis(input_root=root, output_dir=out, diagnostic_plot_count=1,
                 diagnostic_width_scale=1.0, **options)
    return pd.read_csv(out / "all_peaks.csv")


def test_pool_matches_serial_and_gate_filters(tmp_path):
    root = tmp_path / "in"
    paths = _write_cohort(root)
    serial = _run(root, tmp_path / "serial", workers=1)
    pooled = _run(root, tmp_path / "pooled", workers=2)
    pd.testing.assert_frame_equal(serial, pooled)
    assert serial["epileptic"].any()
    # signal_at_peak is the loaded value itself, not the detrended one.
    loaded = pd.read_csv(paths[0]).set_index("frame")
    first = serial[serial["recording"] == "t1"].iloc[0]
    assert first["signal_at_peak"] == loaded.loc[first["frame"], first["roi"]]

    gate = float(serial.loc[serial["epileptic"], "signal_at_peak"].median())
    gated = _run(root, tmp_path / "gated", workers=1, epileptic_min_signal=gate)
    expected = serial["epileptic"] & (serial["signal_at_peak"] >= gate)
    pd.testing.assert_series_equal(gated["epileptic"], expected, check_names=False)
    assert 0 < gated["epileptic"].sum() < serial["epileptic"].sum()


def test_cutoff_at_the_top_of_the_histogram_raises(tmp_path):
    # More than 1% of frames lie above z = 0.5, so the top bin is not a percentile.
    root = tmp_path / "in"
    _write_cohort(root, n_recordings=1)
    with pytest.raises(ValueError, match="widen z_histogram_range"):
        _run(root, tmp_path / "out", z_histogram_range=(-10.0, 0.5))
