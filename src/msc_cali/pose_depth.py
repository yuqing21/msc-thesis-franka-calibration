from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np

from .ball_detection import CameraIntrinsics


POSE_JOINTS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}

RIGHT_ARM_JOINTS = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
}

UPPER_BODY_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)

RIGHT_ARM_CONNECTIONS = (
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
)


@dataclass(frozen=True)
class DepthSample:
    depth_m: float
    valid_ratio: float
    mad_m: float


@dataclass(frozen=True)
class JointObservation:
    name: str
    pixel_uv: tuple[int, int]
    visibility: float
    presence: float
    depth: DepthSample | None
    camera_xyz_m: np.ndarray | None
    filtered_camera_xyz_m: np.ndarray | None
    valid: bool
    reason: str


def robust_depth_at(
    depth_m: Any,
    u: int,
    v: int,
    *,
    radius_px: int = 6,
    min_depth_m: float = 0.3,
    max_depth_m: float = 4.0,
    minimum_valid_ratio: float = 0.25,
) -> DepthSample | None:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional aligned depth image")
    if radius_px < 1:
        raise ValueError("radius_px must be positive")
    height, width = depth.shape
    if not (0 <= u < width and 0 <= v < height):
        return None
    x0, x1 = max(0, u - radius_px), min(width, u + radius_px + 1)
    y0, y1 = max(0, v - radius_px), min(height, v + radius_px + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
    values = depth[y0:y1, x0:x1][disk]
    valid = np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)
    valid_ratio = float(np.mean(valid)) if len(values) else 0.0
    if valid_ratio < minimum_valid_ratio:
        return None
    samples = values[valid]
    median = float(np.median(samples))
    absolute_deviations = np.abs(samples - median)
    mad = float(np.median(absolute_deviations))
    gate = max(0.025, 3.0 * 1.4826 * mad)
    inliers = samples[absolute_deviations <= gate]
    if len(inliers) < max(5, int(0.5 * len(samples))):
        return None
    return DepthSample(float(np.median(inliers)), valid_ratio, mad)


def deproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be positive and finite")
    return np.asarray(
        [
            (u - intrinsics.cx) * depth_m / intrinsics.fx,
            (v - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def joint_angle_deg(first: Any, vertex: Any, third: Any) -> float:
    """Return the unsigned 3-D angle first-vertex-third in degrees."""
    first_vector = np.asarray(first, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    third_vector = np.asarray(third, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(third_vector))
    if denominator <= 1e-12:
        raise ValueError("joint-angle segments must have non-zero length")
    cosine = float(np.dot(first_vector, third_vector) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class OneEuroVectorFilter:
    """One-Euro low-pass filter for stable 3-D landmarks with low lag."""

    def __init__(self, *, min_cutoff_hz: float = 1.2, beta: float = 0.03, derivative_cutoff_hz: float = 1.0) -> None:
        if min_cutoff_hz <= 0 or derivative_cutoff_hz <= 0 or beta < 0:
            raise ValueError("One-Euro filter parameters are invalid")
        self.min_cutoff_hz = min_cutoff_hz
        self.beta = beta
        self.derivative_cutoff_hz = derivative_cutoff_hz
        self._timestamp_s: float | None = None
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff_hz: float, dt_s: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff_hz)
        return 1.0 / (1.0 + tau / dt_s)

    def reset(self) -> None:
        self._timestamp_s = None
        self._value = None
        self._derivative = None

    def update(self, value: Any, timestamp_s: float) -> np.ndarray:
        sample = np.asarray(value, dtype=np.float64)
        if sample.shape != (3,) or not np.all(np.isfinite(sample)):
            raise ValueError("One-Euro input must be a finite 3-vector")
        if self._timestamp_s is None or self._value is None or self._derivative is None:
            self._timestamp_s = float(timestamp_s)
            self._value = sample.copy()
            self._derivative = np.zeros(3, dtype=np.float64)
            return self._value.copy()
        dt_s = float(timestamp_s) - self._timestamp_s
        if dt_s <= 0:
            return self._value.copy()
        raw_derivative = (sample - self._value) / dt_s
        derivative_alpha = self._alpha(self.derivative_cutoff_hz, dt_s)
        self._derivative += derivative_alpha * (raw_derivative - self._derivative)
        cutoff = self.min_cutoff_hz + self.beta * float(np.linalg.norm(self._derivative))
        value_alpha = self._alpha(cutoff, dt_s)
        self._value += value_alpha * (sample - self._value)
        self._timestamp_s = float(timestamp_s)
        return self._value.copy()


class PoseDepthTracker:
    def __init__(
        self,
        *,
        minimum_visibility: float = 0.70,
        minimum_presence: float = 0.70,
        depth_radius_px: int = 6,
        joint_indices: dict[str, int] | None = None,
    ) -> None:
        self.minimum_visibility = minimum_visibility
        self.minimum_presence = minimum_presence
        self.depth_radius_px = depth_radius_px
        self.joint_indices = dict(joint_indices or POSE_JOINTS)
        self.filters = {name: OneEuroVectorFilter() for name in self.joint_indices}

    def process(
        self,
        landmarks: list[Any],
        depth_m: Any,
        intrinsics: CameraIntrinsics,
        timestamp_s: float,
    ) -> dict[str, JointObservation]:
        height, width = np.asarray(depth_m).shape
        observations: dict[str, JointObservation] = {}
        for name, index in self.joint_indices.items():
            landmark = landmarks[index]
            u = int(round(float(landmark.x) * (width - 1)))
            v = int(round(float(landmark.y) * (height - 1)))
            visibility = float(landmark.visibility or 0.0)
            presence = float(landmark.presence or 0.0)
            reason = "ok"
            depth_sample = None
            xyz = None
            filtered = None
            valid = True
            if not (0 <= u < width and 0 <= v < height):
                valid, reason = False, "out_of_frame"
            elif visibility < self.minimum_visibility or presence < self.minimum_presence:
                valid, reason = False, "low_confidence"
            else:
                depth_sample = robust_depth_at(depth_m, u, v, radius_px=self.depth_radius_px)
                if depth_sample is None:
                    valid, reason = False, "no_valid_depth"
                else:
                    xyz = deproject_pixel(u, v, depth_sample.depth_m, intrinsics)
                    filtered = self.filters[name].update(xyz, timestamp_s)
            observations[name] = JointObservation(
                name=name,
                pixel_uv=(u, v),
                visibility=visibility,
                presence=presence,
                depth=depth_sample,
                camera_xyz_m=xyz,
                filtered_camera_xyz_m=filtered,
                valid=valid,
                reason=reason,
            )
        return observations
