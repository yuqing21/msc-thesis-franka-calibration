import numpy as np
import pytest

from msc_cali.robot_state import FrankaStateOnce, average_transforms, rotation_difference_rad


def _state_payload() -> dict:
    transform = np.eye(4)
    transform[:3, 3] = [0.4, -0.1, 0.5]
    return {
        "schema": "franka-state-once/v1",
        "O_T_EE": transform.reshape(16, order="F").tolist(),
        "q": [0.0] * 7,
        "dq": [0.0] * 7,
        "max_abs_dq_rad_s": 0.0,
        "current_errors_empty": True,
        "last_motion_errors_empty": True,
    }


def test_franka_state_parses_column_major_transform() -> None:
    state = FrankaStateOnce.from_dict(_state_payload())
    assert state.transform_base_ee[:3, 3] == pytest.approx([0.4, -0.1, 0.5])


def test_average_transforms_and_rotation_difference() -> None:
    first = np.eye(4)
    second = np.eye(4)
    angle = np.radians(10.0)
    second[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    second[:3, 3] = [0.2, 0.0, 0.0]
    average = average_transforms(first, second)
    assert np.degrees(rotation_difference_rad(first, second)) == pytest.approx(10.0)
    assert average[:3, 3] == pytest.approx([0.1, 0.0, 0.0])
