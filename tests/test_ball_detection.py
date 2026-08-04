import numpy as np
import pytest

from msc_cali.ball_detection import BallDetectorConfig, BallNotFoundError, CameraIntrinsics, detect_ball


def _synthetic_ball() -> tuple[np.ndarray, np.ndarray]:
    height, width = 100, 120
    yy, xx = np.ogrid[:height, :width]
    circle = (xx - 70) ** 2 + (yy - 40) ** 2 <= 12**2
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = [45, 45, 45]
    rgb[circle] = [20, 180, 70]
    depth = np.full((height, width), 1.2, dtype=np.float64)
    depth[circle] = 0.8
    return rgb, depth


def test_detect_ball_returns_aligned_3d_point() -> None:
    rgb, depth = _synthetic_ball()
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=60.0, cy=50.0)
    result = detect_ball(rgb, depth, intrinsics)

    assert result.center_uv == pytest.approx((70.0, 40.0), abs=0.5)
    assert result.point_camera_m == pytest.approx([0.08, -0.08, 0.8], abs=0.005)
    assert result.area_px > 400
    assert result.confidence > 0.7


def test_detect_ball_rejects_missing_target() -> None:
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    depth = np.ones((30, 40), dtype=np.float64)
    with pytest.raises(BallNotFoundError):
        detect_ball(
            rgb,
            depth,
            CameraIntrinsics(100.0, 100.0, 20.0, 15.0),
            BallDetectorConfig(min_area_px=10),
        )

