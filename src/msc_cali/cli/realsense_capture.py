from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from msc_cali.realsense import RealSenseConfig, RealSenseD435f


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one aligned RGB/depth frame from an Intel RealSense D435f.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path")
    parser.add_argument("--metadata", type=Path, help="Output metadata JSON; defaults beside NPZ")
    parser.add_argument("--serial", help="Optional camera serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()

    config = RealSenseConfig(
        serial=args.serial,
        color_width=args.width,
        color_height=args.height,
        depth_width=args.width,
        depth_height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    with RealSenseD435f(config) as camera:
        frame = camera.capture()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rgb=frame.rgb, depth_m=frame.depth_m)
    metadata_path = args.metadata or args.output.with_suffix(".json")
    metadata = {
        "schema": "realsense-aligned-frame/v1",
        "camera_model_expected": "Intel RealSense Depth Camera D435f",
        "device": frame.device.to_dict(),
        "host_timestamp_s": frame.host_timestamp_s,
        "camera_timestamp_ms": frame.camera_timestamp_ms,
        "frame_number": frame.frame_number,
        "intrinsics": {
            "fx": frame.intrinsics.fx,
            "fy": frame.intrinsics.fy,
            "cx": frame.intrinsics.cx,
            "cy": frame.intrinsics.cy,
        },
        "rgb_channel_order": "RGB",
        "depth_units": "m",
        "depth_aligned_to_rgb": True,
        "array_file": args.output.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(metadata_path.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

