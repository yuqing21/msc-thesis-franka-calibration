from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from msc_cali.ball_detection import BallDetectorConfig
from msc_cali.realsense import RealSenseConfig, RealSenseD435f
from msc_cali.robot_state import FrankaStateOnce, average_transforms, rotation_difference_rad
from msc_cali.sphere_detection import fit_known_radius_sphere


def read_franka_state(ssh_host: str, state_command: str, timeout_s: float) -> FrankaStateOnce:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_host, state_command],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one Franka state JSON object, received {len(lines)}")
    return FrankaStateOnce.from_dict(json.loads(lines[0]))


def validate_stationary(state: FrankaStateOnce, maximum_joint_speed_rad_s: float) -> None:
    if not state.current_errors_empty or not state.last_motion_errors_empty:
        raise RuntimeError("Franka reports a current or previous motion error")
    if state.max_abs_dq_rad_s > maximum_joint_speed_rad_s:
        raise RuntimeError(
            f"Franka is still moving: max |dq|={state.max_abs_dq_rad_s:.5f} rad/s exceeds "
            f"{maximum_joint_speed_rad_s:.5f} rad/s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one stationary Franka/ball calibration pose.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ssh-host", default="liuy@100.68.210.77")
    parser.add_argument("--state-command", default="/home/liuy/msc_thesis/bin/robot_state_once 172.16.0.2")
    parser.add_argument("--serial", default="242322072812")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--ball-radius-m", type=float, default=0.025)
    parser.add_argument("--ball-target-rgb", type=int, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--ball-chromaticity-tolerance", type=float, default=0.20)
    parser.add_argument("--ball-min-channel-dominance", type=float, default=0.08)
    parser.add_argument("--ball-min-brightness", type=float, default=0.12)
    parser.add_argument("--ball-min-area-px", type=int, default=80)
    parser.add_argument("--ball-roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--maximum-joint-speed-rad-s", type=float, default=0.01)
    parser.add_argument("--maximum-drift-m", type=float, default=0.002)
    parser.add_argument("--maximum-drift-deg", type=float, default=0.5)
    parser.add_argument("--temporal-group-frames", type=int, default=5)
    parser.add_argument("--maximum-group-p95-m", type=float, default=0.015)
    parser.add_argument("--maximum-sphere-fit-rmse-m", type=float, default=0.010)
    args = parser.parse_args()
    if args.frames < 5:
        parser.error("--frames must be at least 5")
    if args.ball_target_rgb is not None and any(not 0 <= value <= 255 for value in args.ball_target_rgb):
        parser.error("--ball-target-rgb values must lie in [0, 255]")

    detector_defaults = BallDetectorConfig()
    detector_config = BallDetectorConfig(
        target_rgb=tuple(args.ball_target_rgb) if args.ball_target_rgb else detector_defaults.target_rgb,
        chromaticity_tolerance=args.ball_chromaticity_tolerance,
        min_channel_dominance=args.ball_min_channel_dominance,
        min_brightness=args.ball_min_brightness,
        min_area_px=args.ball_min_area_px,
        roi_xyxy=tuple(args.ball_roi) if args.ball_roi else None,
    )

    output_dir = args.output_dir / args.sample_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"sample directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    state_before = read_franka_state(args.ssh_host, args.state_command, timeout_s=10.0)
    validate_stationary(state_before, args.maximum_joint_speed_rad_s)

    config = RealSenseConfig(
        serial=args.serial,
        color_width=args.width,
        color_height=args.height,
        depth_width=args.width,
        depth_height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    host_timestamps: list[float] = []
    camera_timestamps: list[float] = []
    frame_numbers: list[int] = []
    sphere_results = []
    with RealSenseD435f(config) as camera:
        device = camera.device_info
        for _ in range(args.frames):
            frame = camera.capture()
            sphere = fit_known_radius_sphere(
                frame.rgb,
                frame.depth_m,
                frame.intrinsics,
                args.ball_radius_m,
                detector_config,
            )
            rgb_frames.append(frame.rgb)
            depth_frames.append(frame.depth_m)
            host_timestamps.append(frame.host_timestamp_s)
            camera_timestamps.append(frame.camera_timestamp_ms)
            frame_numbers.append(frame.frame_number)
            sphere_results.append(sphere)
        intrinsics = frame.intrinsics

    state_after = read_franka_state(args.ssh_host, args.state_command, timeout_s=10.0)
    validate_stationary(state_after, args.maximum_joint_speed_rad_s)
    translation_drift_m = float(
        np.linalg.norm(state_after.transform_base_ee[:3, 3] - state_before.transform_base_ee[:3, 3])
    )
    rotation_drift_rad = rotation_difference_rad(
        state_before.transform_base_ee,
        state_after.transform_base_ee,
    )
    if translation_drift_m > args.maximum_drift_m:
        raise RuntimeError(f"Franka translated {translation_drift_m:.6f} m during capture")
    if np.degrees(rotation_drift_rad) > args.maximum_drift_deg:
        raise RuntimeError(f"Franka rotated {np.degrees(rotation_drift_rad):.4f} deg during capture")

    centers = np.asarray([result.center_camera_m for result in sphere_results])
    median_rgb = np.median(np.stack(rgb_frames), axis=0).astype(np.uint8)
    median_depth = np.median(np.stack(depth_frames), axis=0).astype(np.float32)
    aggregate_sphere = fit_known_radius_sphere(
        median_rgb,
        median_depth,
        intrinsics,
        args.ball_radius_m,
        detector_config,
    )
    if aggregate_sphere.fit_rmse_m > args.maximum_sphere_fit_rmse_m:
        raise RuntimeError(
            f"aggregate sphere fit RMSE {aggregate_sphere.fit_rmse_m:.6f} m exceeds "
            f"{args.maximum_sphere_fit_rmse_m:.6f} m"
        )

    group_centers = []
    for start in range(0, args.frames, args.temporal_group_frames):
        stop = min(start + args.temporal_group_frames, args.frames)
        if stop - start < 3:
            continue
        group_rgb = np.median(np.stack(rgb_frames[start:stop]), axis=0).astype(np.uint8)
        group_depth = np.median(np.stack(depth_frames[start:stop]), axis=0).astype(np.float32)
        group_centers.append(
            fit_known_radius_sphere(
                group_rgb,
                group_depth,
                intrinsics,
                args.ball_radius_m,
                detector_config,
            ).center_camera_m
        )
    group_centers_array = np.asarray(group_centers)
    group_deviations = np.linalg.norm(
        group_centers_array - aggregate_sphere.center_camera_m,
        axis=1,
    )
    group_p95_m = float(np.percentile(group_deviations, 95))
    if group_p95_m > args.maximum_group_p95_m:
        raise RuntimeError(
            f"grouped sphere-center p95 {group_p95_m:.6f} m exceeds "
            f"{args.maximum_group_p95_m:.6f} m"
        )
    transform_base_ee = average_transforms(
        state_before.transform_base_ee,
        state_after.transform_base_ee,
    )

    arrays_path = output_dir / "frames.npz"
    np.savez_compressed(
        arrays_path,
        rgb=np.stack(rgb_frames),
        depth_m=np.stack(depth_frames),
        host_timestamp_s=np.asarray(host_timestamps),
        camera_timestamp_ms=np.asarray(camera_timestamps),
        frame_number=np.asarray(frame_numbers),
        sphere_center_camera_m=centers,
        aggregate_rgb=median_rgb,
        aggregate_depth_m=median_depth,
        aggregate_sphere_center_camera_m=aggregate_sphere.center_camera_m,
        grouped_sphere_center_camera_m=group_centers_array,
    )
    Image.fromarray(median_rgb).save(output_dir / "preview_rgb.png")

    metadata = {
        "schema": "franka-ball-calibration-pose/v1",
        "sample_id": args.sample_id,
        "created_at_s": time.time(),
        "provisional_soft_ball_target": True,
        "camera": {
            "device": device.to_dict(),
            "resolution": [args.width, args.height],
            "fps": args.fps,
            "intrinsics": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
            },
            "depth_aligned_to_rgb": True,
            "depth_units": "m",
        },
        "sphere": {
            "known_radius_m": args.ball_radius_m,
            "detector": {
                "target_rgb": list(detector_config.target_rgb),
                "chromaticity_tolerance": detector_config.chromaticity_tolerance,
                "min_channel_dominance": detector_config.min_channel_dominance,
                "min_brightness": detector_config.min_brightness,
                "min_area_px": detector_config.min_area_px,
                "roi_xyxy": list(detector_config.roi_xyxy) if detector_config.roi_xyxy else None,
            },
            "aggregate": aggregate_sphere.to_dict(),
            "group_size_frames": args.temporal_group_frames,
            "group_centers_camera_m": group_centers_array.tolist(),
            "group_p95_deviation_m": group_p95_m,
            "per_frame": [result.to_dict() for result in sphere_results],
        },
        "robot": {
            "state_before": state_before.to_dict(),
            "state_after": state_after.to_dict(),
            "O_T_EE_column_major": transform_base_ee.reshape(16, order="F").tolist(),
            "translation_drift_m": translation_drift_m,
            "rotation_drift_deg": float(np.degrees(rotation_drift_rad)),
        },
        "arrays_file": arrays_path.name,
        "preview_file": "preview_rgb.png",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_id": args.sample_id,
        "camera_center_m": aggregate_sphere.center_camera_m.tolist(),
        "camera_group_p95_mm": group_p95_m * 1000.0,
        "sphere_fit_rmse_mm": aggregate_sphere.fit_rmse_m * 1000.0,
        "ee_translation_base_m": transform_base_ee[:3, 3].tolist(),
        "translation_drift_mm": translation_drift_m * 1000.0,
        "rotation_drift_deg": float(np.degrees(rotation_drift_rad)),
        "metadata": str(metadata_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
