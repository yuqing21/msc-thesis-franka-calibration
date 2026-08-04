"""Calibration and vision-side utilities for the MSc thesis setup."""

from .ball_detection import (
    BallDetection,
    BallDetectorConfig,
    BallNotFoundError,
    CameraIntrinsics,
    detect_ball,
    find_ball_pixels,
)
from .sphere_detection import SphereDetection, fit_known_radius_sphere
from .geometry import CalibrationResult, RigidTransform, solve_rigid_transform
from .realsense import RealSenseConfig, RealSenseD435f, RealSenseFrame, RealSenseUnavailableError

__all__ = [
    "BallDetection",
    "BallDetectorConfig",
    "BallNotFoundError",
    "CalibrationResult",
    "CameraIntrinsics",
    "RigidTransform",
    "RealSenseConfig",
    "RealSenseD435f",
    "RealSenseFrame",
    "RealSenseUnavailableError",
    "detect_ball",
    "find_ball_pixels",
    "SphereDetection",
    "fit_known_radius_sphere",
    "solve_rigid_transform",
]
