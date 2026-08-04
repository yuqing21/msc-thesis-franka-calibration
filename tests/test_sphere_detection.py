import numpy as np
import pytest

from msc_cali.ball_detection import BallDetectorConfig, CameraIntrinsics
from msc_cali.sphere_detection import fit_known_radius_sphere


def test_fit_known_radius_sphere_recovers_center() -> None:
    height, width = 180, 240
    intrinsics = CameraIntrinsics(fx=220.0, fy=220.0, cx=120.0, cy=90.0)
    radius_m = 0.05
    true_center = np.asarray([0.03, -0.02, 0.80])
    yy, xx = np.indices((height, width))
    rays = np.stack(
        ((xx - intrinsics.cx) / intrinsics.fx, (yy - intrinsics.cy) / intrinsics.fy, np.ones_like(xx)),
        axis=-1,
    )
    a = np.sum(rays * rays, axis=-1)
    b = -2.0 * np.sum(rays * true_center, axis=-1)
    c = float(true_center @ true_center - radius_m**2)
    discriminant = b * b - 4.0 * a * c
    visible = discriminant >= 0.0
    depth = np.full((height, width), 1.5, dtype=np.float64)
    depth[visible] = (-b[visible] - np.sqrt(discriminant[visible])) / (2.0 * a[visible])
    rgb = np.full((height, width, 3), 45, dtype=np.uint8)
    rgb[visible] = [20, 180, 70]

    result = fit_known_radius_sphere(
        rgb,
        depth,
        intrinsics,
        radius_m,
        BallDetectorConfig(min_area_px=50),
    )

    assert result.center_camera_m == pytest.approx(true_center, abs=2e-4)
    assert result.fit_rmse_m < 1e-5
    assert result.surface_point_count > 100
