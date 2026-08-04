from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ball_detection import BallDetectorConfig, BallNotFoundError, CameraIntrinsics, find_ball_pixels


@dataclass(frozen=True)
class SphereDetection:
    center_camera_m: np.ndarray
    known_radius_m: float
    surface_point_count: int
    fit_rmse_m: float
    fit_median_abs_m: float
    center_uv: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_camera_m": self.center_camera_m.tolist(),
            "known_radius_m": self.known_radius_m,
            "surface_point_count": self.surface_point_count,
            "fit_rmse_m": self.fit_rmse_m,
            "fit_median_abs_m": self.fit_median_abs_m,
            "center_uv": list(self.center_uv),
        }


def fit_known_radius_sphere(
    rgb: Any,
    depth_m: Any,
    intrinsics: CameraIntrinsics,
    known_radius_m: float,
    config: BallDetectorConfig | None = None,
) -> SphereDetection:
    """Fit a known-radius sphere to aligned depth samples inside the colour mask."""
    if not np.isfinite(known_radius_m) or known_radius_m <= 0:
        raise ValueError("known_radius_m must be positive and finite")
    config = config or BallDetectorConfig()
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth_m, dtype=np.float64)
    if depth_array.shape != rgb_array.shape[:2]:
        raise ValueError("depth_m must be aligned with RGB and have shape (H, W)")

    rows, columns = find_ball_pixels(rgb_array, config)
    depths = depth_array[rows, columns]
    valid = (
        np.isfinite(depths)
        & (depths >= config.min_depth_m)
        & (depths <= config.max_depth_m)
    )
    rows = rows[valid]
    columns = columns[valid]
    depths = depths[valid]
    if len(depths) < 20:
        raise BallNotFoundError("not enough valid depth samples for sphere fitting")

    median_depth = float(np.median(depths))
    depth_gate_m = max(0.040, 1.6 * known_radius_m)
    foreground = np.abs(depths - median_depth) <= depth_gate_m
    rows = rows[foreground]
    columns = columns[foreground]
    depths = depths[foreground]
    if len(depths) < 20:
        raise BallNotFoundError("too few foreground depth samples remain for sphere fitting")

    points = np.column_stack(
        (
            (columns - intrinsics.cx) * depths / intrinsics.fx,
            (rows - intrinsics.cy) * depths / intrinsics.fy,
            depths,
        )
    )
    center_u = float(np.median(columns))
    center_v = float(np.median(rows))
    surface_center = np.asarray(
        [
            (center_u - intrinsics.cx) * median_depth / intrinsics.fx,
            (center_v - intrinsics.cy) * median_depth / intrinsics.fy,
            median_depth,
        ],
        dtype=np.float64,
    )
    viewing_ray = surface_center / np.linalg.norm(surface_center)
    center = surface_center + viewing_ray * (known_radius_m / np.sqrt(2.0))

    inliers = np.ones(len(points), dtype=bool)
    for _ in range(20):
        selected = points[inliers]
        vectors = center - selected
        distances = np.linalg.norm(vectors, axis=1)
        distances = np.maximum(distances, 1e-9)
        residuals = distances - known_radius_m
        jacobian = vectors / distances[:, None]
        delta, *_ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        center += delta

        all_residuals = np.linalg.norm(center - points, axis=1) - known_radius_m
        median_residual = float(np.median(all_residuals))
        mad = float(np.median(np.abs(all_residuals - median_residual)))
        threshold = max(0.003, 3.5 * 1.4826 * mad)
        new_inliers = np.abs(all_residuals - median_residual) <= threshold
        if int(new_inliers.sum()) < 20:
            new_inliers = np.ones(len(points), dtype=bool)
        if np.linalg.norm(delta) < 1e-7 and np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers

    final_residuals = np.linalg.norm(center - points[inliers], axis=1) - known_radius_m
    return SphereDetection(
        center_camera_m=center,
        known_radius_m=float(known_radius_m),
        surface_point_count=int(inliers.sum()),
        fit_rmse_m=float(np.sqrt(np.mean(final_residuals**2))),
        fit_median_abs_m=float(np.median(np.abs(final_residuals))),
        center_uv=(center_u, center_v),
    )
