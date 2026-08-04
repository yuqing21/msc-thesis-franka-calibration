from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from msc_cali.ball_detection import BallDetectorConfig, CameraIntrinsics, detect_ball
from msc_cali.geometry import RigidTransform


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect the coloured ball in saved aligned RGB/depth arrays.")
    parser.add_argument("--input", type=Path, required=True, help="NPZ containing rgb and depth_m")
    parser.add_argument("--intrinsics", type=Path, required=True, help="JSON with fx, fy, cx, cy")
    parser.add_argument("--transform", type=Path, help="Optional camera-to-base transform JSON")
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as arrays:
        rgb = arrays["rgb"]
        depth_m = arrays["depth_m"]
    intrinsics_data = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    if "intrinsics" in intrinsics_data:
        intrinsics_data = intrinsics_data["intrinsics"]
    intrinsics = CameraIntrinsics(**intrinsics_data)
    detection = detect_ball(rgb, depth_m, intrinsics, BallDetectorConfig())
    output = detection.to_dict()
    if args.transform:
        transform_data = json.loads(args.transform.read_text(encoding="utf-8"))
        transform = RigidTransform.from_dict(transform_data)
        output["point_base_m"] = transform.apply(detection.point_camera_m).tolist()
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
