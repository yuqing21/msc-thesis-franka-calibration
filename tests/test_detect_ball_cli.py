import json
from pathlib import Path

import numpy as np
import pytest

from msc_cali.cli import detect_ball_npz


def test_capture_metadata_intrinsics_are_accepted(tmp_path: Path, monkeypatch, capsys) -> None:
    height, width = 60, 80
    yy, xx = np.ogrid[:height, :width]
    circle = (xx - 45) ** 2 + (yy - 25) ** 2 <= 10**2
    rgb = np.full((height, width, 3), 40, dtype=np.uint8)
    rgb[circle] = [20, 180, 70]
    depth = np.full((height, width), 1.0, dtype=np.float32)
    depth[circle] = 0.7
    arrays_path = tmp_path / "capture.npz"
    metadata_path = tmp_path / "capture.json"
    np.savez_compressed(arrays_path, rgb=rgb, depth_m=depth)
    metadata_path.write_text(
        json.dumps({"schema": "realsense-aligned-frame/v1", "intrinsics": {"fx": 100, "fy": 100, "cx": 40, "cy": 30}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["detect_ball_npz", "--input", str(arrays_path), "--intrinsics", str(metadata_path)],
    )
    assert detect_ball_npz.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["depth_m"] == pytest.approx(0.7)
