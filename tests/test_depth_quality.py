import numpy as np
import pytest

from msc_cali.depth_quality import analyze_depth_sequence


def test_depth_quality_reports_coverage_fps_and_noise() -> None:
    rng = np.random.default_rng(42)
    depths = 1.0 + rng.normal(0.0, 0.002, size=(30, 20, 30))
    depths[:, :2, :] = 0.0
    timestamps = np.arange(30) * (1000.0 / 30.0)
    frame_numbers = np.arange(100, 130)
    report = analyze_depth_sequence(depths, timestamps, frame_numbers, requested_fps=30)

    assert report["measured_fps"] == pytest.approx(30.0)
    assert report["dropped_frames_from_frame_numbers"] == 0
    assert report["valid_depth_ratio_mean"] == pytest.approx(0.9)
    assert report["center_valid_depth_ratio_mean"] == pytest.approx(1.0)
    assert 1.0 < report["center_temporal_std_median_mm"] < 3.0


def test_depth_quality_detects_frame_number_gap() -> None:
    depths = np.ones((4, 4, 4), dtype=np.float64)
    timestamps = [0.0, 33.0, 66.0, 99.0]
    report = analyze_depth_sequence(depths, timestamps, [1, 2, 5, 6], requested_fps=30)
    assert report["dropped_frames_from_frame_numbers"] == 2

