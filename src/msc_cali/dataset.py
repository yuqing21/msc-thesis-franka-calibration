from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationDataset:
    sample_ids: tuple[str, ...]
    camera_points_m: np.ndarray
    robot_points_m: np.ndarray


def load_calibration_jsonl(path: Path) -> CalibrationDataset:
    sample_ids: list[str] = []
    camera_points: list[Any] = []
    robot_points: list[Any] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not sample.get("enabled", True):
                continue
            if sample.get("units", "m") != "m":
                raise ValueError(f"{path}:{line_number}: only metre units are accepted")
            sample_id = str(sample.get("sample_id", f"sample_{line_number:03d}"))
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id!r}")
            seen_ids.add(sample_id)
            sample_ids.append(sample_id)
            camera_points.append(sample["camera_point_m"])
            robot_points.append(sample["robot_point_m"])

    if not sample_ids:
        raise ValueError(f"{path} contains no enabled calibration samples")
    camera = np.asarray(camera_points, dtype=np.float64)
    robot = np.asarray(robot_points, dtype=np.float64)
    if camera.shape != (len(sample_ids), 3) or robot.shape != (len(sample_ids), 3):
        raise ValueError("every calibration point must contain exactly three coordinates")
    if not np.all(np.isfinite(camera)) or not np.all(np.isfinite(robot)):
        raise ValueError("calibration samples contain NaN or infinity")
    return CalibrationDataset(tuple(sample_ids), camera, robot)

