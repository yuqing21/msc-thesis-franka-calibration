from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from msc_cali.depth_quality import analyze_depth_sequence
from msc_cali.realsense import RealSenseConfig, RealSenseD435f


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure D435f frame delivery and stationary depth stability.")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--last-frame", type=Path, help="Optional NPZ for the final aligned RGB/depth frame")
    parser.add_argument("--serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    if args.frames < 10:
        parser.error("--frames must be at least 10")

    config = RealSenseConfig(
        serial=args.serial,
        color_width=args.width,
        color_height=args.height,
        depth_width=args.width,
        depth_height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    depth_frames = []
    timestamps_ms = []
    frame_numbers = []
    last_frame = None
    with RealSenseD435f(config) as camera:
        device = camera.device_info
        for _ in range(args.frames):
            last_frame = camera.capture()
            depth_frames.append(last_frame.depth_m)
            timestamps_ms.append(last_frame.camera_timestamp_ms)
            frame_numbers.append(last_frame.frame_number)
    assert last_frame is not None

    report = analyze_depth_sequence(
        np.stack(depth_frames),
        timestamps_ms,
        frame_numbers,
        requested_fps=args.fps,
    )
    report["device"] = device.to_dict()
    report["intrinsics"] = {
        "fx": last_frame.intrinsics.fx,
        "fy": last_frame.intrinsics.fy,
        "cx": last_frame.intrinsics.cx,
        "cy": last_frame.intrinsics.cy,
    }
    report["stream"] = {
        "rgb_channel_order": "RGB",
        "depth_units": "m",
        "depth_aligned_to_rgb": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.last_frame:
        args.last_frame.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.last_frame, rgb=last_frame.rgb, depth_m=last_frame.depth_m)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

