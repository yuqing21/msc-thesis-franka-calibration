from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_points(value: Any, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains NaN or infinity")
    return points


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform mapping column-vector points from camera to robot base."""

    rotation: np.ndarray
    translation_m: np.ndarray
    source_frame: str = "camera"
    target_frame: str = "robot_base"

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation_m must have shape (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("transform contains NaN or infinity")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("rotation must be proper with determinant +1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    def apply(self, points_m: Any) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        if points.shape == (3,):
            if not np.all(np.isfinite(points)):
                raise ValueError("point contains NaN or infinity")
            return self.rotation @ points + self.translation_m
        points = _as_points(points, name="points_m")
        return points @ self.rotation.T + self.translation_m

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.T
        inverse_translation = -(inverse_rotation @ self.translation_m)
        return RigidTransform(
            rotation=inverse_rotation,
            translation_m=inverse_translation,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "rigid-transform/v1",
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "units": "m",
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation_m.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RigidTransform":
        if value.get("schema") != "rigid-transform/v1" or value.get("units") != "m":
            raise ValueError("unsupported transform schema or units")
        return cls(
            rotation=np.asarray(value["rotation"], dtype=np.float64),
            translation_m=np.asarray(value["translation_m"], dtype=np.float64),
            source_frame=str(value["source_frame"]),
            target_frame=str(value["target_frame"]),
        )


@dataclass(frozen=True)
class CalibrationResult:
    transform: RigidTransform
    residuals_m: np.ndarray
    rmse_m: float
    median_m: float
    p95_m: float
    max_m: float
    sample_count: int
    geometry_singular_values_m: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        output = self.transform.to_dict()
        output["quality"] = {
            "sample_count": self.sample_count,
            "rmse_m": self.rmse_m,
            "median_m": self.median_m,
            "p95_m": self.p95_m,
            "max_m": self.max_m,
            "geometry_singular_values_m": self.geometry_singular_values_m.tolist(),
            "residuals_m": self.residuals_m.tolist(),
        }
        return output


def solve_rigid_transform(
    camera_points_m: Any,
    robot_points_m: Any,
    *,
    min_samples: int = 6,
    min_geometry_ratio: float = 1e-3,
) -> CalibrationResult:
    """Solve robot_point = R @ camera_point + t using the Kabsch method.

    A full-rank, non-coplanar point set is required because this project uses
    depth-camera 3D measurements and needs observable depth-axis geometry.
    """

    camera = _as_points(camera_points_m, name="camera_points_m")
    robot = _as_points(robot_points_m, name="robot_points_m")
    if camera.shape != robot.shape:
        raise ValueError("camera and robot point arrays must have identical shape")
    if len(camera) < min_samples:
        raise ValueError(f"at least {min_samples} correspondences are required")

    camera_center = camera.mean(axis=0)
    robot_center = robot.mean(axis=0)
    camera_centered = camera - camera_center
    robot_centered = robot - robot_center

    geometry_singular_values = np.linalg.svd(camera_centered, compute_uv=False)
    if geometry_singular_values[0] <= np.finfo(np.float64).eps:
        raise ValueError("calibration points have no spatial extent")
    geometry_ratio = geometry_singular_values[-1] / geometry_singular_values[0]
    if geometry_ratio < min_geometry_ratio:
        raise ValueError(
            "calibration geometry is nearly planar or collinear; collect points at varied x, y, and z"
        )

    covariance = camera_centered.T @ robot_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = robot_center - rotation @ camera_center
    transform = RigidTransform(
        rotation=rotation,
        translation_m=translation,
        source_frame="camera_depth_optical_frame",
        target_frame="panda_link0",
    )
    predicted_robot = transform.apply(camera)
    residuals = np.linalg.norm(predicted_robot - robot, axis=1)
    return CalibrationResult(
        transform=transform,
        residuals_m=residuals,
        rmse_m=float(np.sqrt(np.mean(residuals**2))),
        median_m=float(np.median(residuals)),
        p95_m=float(np.percentile(residuals, 95)),
        max_m=float(np.max(residuals)),
        sample_count=len(camera),
        geometry_singular_values_m=geometry_singular_values,
    )

