from dataclasses import dataclass

import numpy as np
import pytest

from msc_cali.ball_detection import CameraIntrinsics
from msc_cali.pose_depth import (
    RIGHT_ARM_JOINTS,
    OneEuroVectorFilter,
    PoseDepthTracker,
    deproject_pixel,
    joint_angle_deg,
    robust_depth_at,
)


@dataclass
class Landmark:
    x: float
    y: float
    visibility: float = 0.99
    presence: float = 0.99


def test_robust_depth_uses_local_foreground_median() -> None:
    depth = np.full((40, 50), 2.0)
    depth[15:26, 20:31] = 1.2
    depth[20, 25] = 0.0
    sample = robust_depth_at(depth, 25, 20, radius_px=5)
    assert sample is not None
    assert sample.depth_m == pytest.approx(1.2)
    assert sample.valid_ratio > 0.9


def test_deprojection_uses_rgb_intrinsics() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=200.0, cx=50.0, cy=40.0)
    assert deproject_pixel(60, 60, 2.0, intrinsics) == pytest.approx([0.2, 0.2, 2.0])


def test_one_euro_filter_reduces_stationary_jitter() -> None:
    rng = np.random.default_rng(5)
    raw = np.asarray([1.0, 0.0, 2.0]) + rng.normal(0.0, 0.01, size=(60, 3))
    filter_ = OneEuroVectorFilter(min_cutoff_hz=1.0, beta=0.0)
    filtered = np.asarray([filter_.update(value, index / 30.0) for index, value in enumerate(raw)])
    assert np.std(filtered[15:, 0]) < np.std(raw[15:, 0])


def test_pose_tracker_reports_six_valid_upper_body_joints() -> None:
    landmarks = [Landmark(0.5, 0.5) for _ in range(33)]
    depth = np.full((100, 200), 1.5)
    tracker = PoseDepthTracker()
    result = tracker.process(
        landmarks,
        depth,
        CameraIntrinsics(fx=150, fy=150, cx=100, cy=50),
        1.0,
    )
    assert len(result) == 6
    assert all(item.valid for item in result.values())


def test_right_arm_mode_reports_only_three_joints() -> None:
    landmarks = [Landmark(0.5, 0.5) for _ in range(33)]
    tracker = PoseDepthTracker(joint_indices=RIGHT_ARM_JOINTS)
    result = tracker.process(
        landmarks,
        np.full((100, 200), 1.5),
        CameraIntrinsics(fx=150, fy=150, cx=100, cy=50),
        1.0,
    )
    assert set(result) == set(RIGHT_ARM_JOINTS)


def test_joint_angle_is_unsigned_three_dimensional_angle() -> None:
    assert joint_angle_deg([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert joint_angle_deg([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
