from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrankaStateOnce:
    transform_base_ee: np.ndarray
    joints_rad: np.ndarray
    joint_velocities_rad_s: np.ndarray
    max_abs_dq_rad_s: float
    current_errors_empty: bool
    last_motion_errors_empty: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrankaStateOnce":
        if value.get("schema") != "franka-state-once/v1":
            raise ValueError("unsupported Franka state schema")
        transform = np.asarray(value["O_T_EE"], dtype=np.float64)
        if transform.shape != (16,):
            raise ValueError("O_T_EE must contain 16 column-major values")
        transform = transform.reshape((4, 4), order="F")
        if not np.all(np.isfinite(transform)) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("O_T_EE is not a finite homogeneous transform")
        return cls(
            transform_base_ee=transform,
            joints_rad=np.asarray(value["q"], dtype=np.float64),
            joint_velocities_rad_s=np.asarray(value["dq"], dtype=np.float64),
            max_abs_dq_rad_s=float(value["max_abs_dq_rad_s"]),
            current_errors_empty=bool(value["current_errors_empty"]),
            last_motion_errors_empty=bool(value["last_motion_errors_empty"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "O_T_EE_column_major": self.transform_base_ee.reshape(16, order="F").tolist(),
            "q_rad": self.joints_rad.tolist(),
            "dq_rad_s": self.joint_velocities_rad_s.tolist(),
            "max_abs_dq_rad_s": self.max_abs_dq_rad_s,
            "current_errors_empty": self.current_errors_empty,
            "last_motion_errors_empty": self.last_motion_errors_empty,
        }


def rotation_difference_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def average_transforms(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    u, _, vt = np.linalg.svd(first[:3, :3] + second[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = (first[:3, 3] + second[:3, 3]) / 2.0
    return result
