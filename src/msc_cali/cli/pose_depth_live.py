from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from msc_cali.cli.realsense_live import depth_to_rgb
from msc_cali.pose_depth import (
    RIGHT_ARM_CONNECTIONS,
    RIGHT_ARM_JOINTS,
    JointObservation,
    PoseDepthTracker,
    joint_angle_deg,
)
from msc_cali.realsense import RealSenseConfig, RealSenseD435f


@dataclass(frozen=True)
class RenderedFrame:
    image: np.ndarray
    status: str


@dataclass(frozen=True)
class WorkerFailure:
    message: str


def _load_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Run: .venv\\Scripts\\python.exe -m pip install mediapipe"
        ) from exc
    return mp


def _draw_observations(
    rgb: np.ndarray,
    depth_visual: np.ndarray,
    observations: dict[str, JointObservation],
) -> tuple[np.ndarray, np.ndarray]:
    rgb_image = Image.fromarray(rgb, mode="RGB")
    depth_image = Image.fromarray(depth_visual, mode="RGB")
    rgb_draw = ImageDraw.Draw(rgb_image)
    depth_draw = ImageDraw.Draw(depth_image)
    for first, second in RIGHT_ARM_CONNECTIONS:
        a, b = observations[first], observations[second]
        if a.valid and b.valid:
            rgb_draw.line((a.pixel_uv, b.pixel_uv), fill=(0, 255, 80), width=5)
            depth_draw.line((a.pixel_uv, b.pixel_uv), fill=(255, 255, 255), width=4)
    for name, observation in observations.items():
        u, v = observation.pixel_uv
        colour = (0, 255, 80) if observation.valid else (255, 70, 50)
        radius = 9
        for draw in (rgb_draw, depth_draw):
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=colour, width=4)
        short_name = name.replace("left_", "L-").replace("right_", "R-")
        if observation.valid and observation.depth is not None:
            label = f"{short_name} {observation.depth.depth_m:.2f}m"
        else:
            label = f"{short_name} {observation.reason}"
        rgb_draw.text((u + 12, v - 16), label, fill=colour, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(rgb_image), np.asarray(depth_image)


class PoseWorker(threading.Thread):
    def __init__(
        self,
        config: RealSenseConfig,
        model_path: Path,
        minimum_depth_m: float,
        maximum_depth_m: float,
        status_file: Path,
    ) -> None:
        super().__init__(name="pose-depth-worker", daemon=True)
        self.config = config
        self.model_path = model_path
        self.minimum_depth_m = minimum_depth_m
        self.maximum_depth_m = maximum_depth_m
        self.status_file = status_file
        self.last_status_write_s = 0.0
        self.stop_event = threading.Event()
        self.frames: queue.Queue[RenderedFrame | WorkerFailure] = queue.Queue(maxsize=1)

    def publish(self, item: RenderedFrame | WorkerFailure) -> None:
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            pass

    def write_status(self, payload: dict[str, Any]) -> None:
        now_s = time.monotonic()
        if now_s - self.last_status_write_s < 0.5:
            return
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.status_file.with_suffix(self.status_file.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.status_file)
        self.last_status_write_s = now_s

    def run(self) -> None:
        try:
            mp = _load_mediapipe()
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path.resolve())),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.65,
                min_pose_presence_confidence=0.70,
                min_tracking_confidence=0.75,
                output_segmentation_masks=False,
            )
            tracker = PoseDepthTracker(
                minimum_visibility=0.70,
                minimum_presence=0.70,
                joint_indices=RIGHT_ARM_JOINTS,
            )
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                with RealSenseD435f(self.config) as camera:
                    previous_timestamp_ms = -1
                    while not self.stop_event.is_set():
                        frame = camera.capture()
                        timestamp_ms = max(previous_timestamp_ms + 1, int(time.monotonic() * 1000.0))
                        previous_timestamp_ms = timestamp_ms
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.rgb)
                        result = landmarker.detect_for_video(mp_image, timestamp_ms)
                        depth_visual = depth_to_rgb(
                            frame.depth_m,
                            self.minimum_depth_m,
                            self.maximum_depth_m,
                        )
                        if not result.pose_landmarks:
                            combined = np.concatenate((frame.rgb, depth_visual), axis=1)
                            self.publish(RenderedFrame(combined, "NO PERSON: move the upper body into RGB view"))
                            self.write_status({
                                "schema": "pose-depth-layout/v1",
                                "timestamp_s": time.time(),
                                "verdict": "NO_PERSON",
                                "valid_joint_count": 0,
                                "joints": {},
                            })
                            continue
                        observations = tracker.process(
                            result.pose_landmarks[0],
                            frame.depth_m,
                            frame.intrinsics,
                            frame.host_timestamp_s,
                        )
                        annotated_rgb, annotated_depth = _draw_observations(frame.rgb, depth_visual, observations)
                        combined = np.concatenate((annotated_rgb, annotated_depth), axis=1)
                        valid_count = sum(item.valid for item in observations.values())
                        right_wrist_valid = observations["right_wrist"].valid
                        elbow_angle_deg = None
                        if valid_count == 3:
                            elbow_angle_deg = joint_angle_deg(
                                observations["right_shoulder"].filtered_camera_xyz_m,
                                observations["right_elbow"].filtered_camera_xyz_m,
                                observations["right_wrist"].filtered_camera_xyz_m,
                            )
                        verdict = "LAYOUT PASS" if valid_count == 3 else "LAYOUT NOT READY"
                        angle_text = f" | elbow angle: {elbow_angle_deg:.1f} deg" if elbow_angle_deg is not None else ""
                        status = f"{verdict} | RIGHT shoulder/elbow/wrist: {valid_count}/3{angle_text}"
                        self.publish(RenderedFrame(combined, status))
                        self.write_status({
                            "schema": "pose-depth-layout/v1",
                            "timestamp_s": time.time(),
                            "verdict": "PASS" if valid_count == 3 else "NOT_READY",
                            "valid_joint_count": valid_count,
                            "right_wrist_valid": right_wrist_valid,
                            "right_elbow_angle_deg": elbow_angle_deg,
                            "joints": {
                                name: {
                                    "valid": item.valid,
                                    "reason": item.reason,
                                    "pixel_uv": list(item.pixel_uv),
                                    "visibility": item.visibility,
                                    "presence": item.presence,
                                    "depth_m": item.depth.depth_m if item.depth is not None else None,
                                    "depth_valid_ratio": item.depth.valid_ratio if item.depth is not None else 0.0,
                                    "filtered_camera_xyz_m": (
                                        item.filtered_camera_xyz_m.tolist()
                                        if item.filtered_camera_xyz_m is not None
                                        else None
                                    ),
                                }
                                for name, item in observations.items()
                            },
                        })
        except Exception as exc:
            self.publish(WorkerFailure(f"{type(exc).__name__}: {exc}"))

    def stop(self) -> None:
        self.stop_event.set()


class PoseDepthViewer:
    def __init__(self, root: tk.Tk, worker: PoseWorker) -> None:
        self.root = root
        self.worker = worker
        self.closed = False
        self.photo: ImageTk.PhotoImage | None = None
        root.title("D435f shoulder-elbow-wrist depth check (left RGB | right aligned depth)")
        root.configure(background="#181818")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _event: self.close())
        self.image_label = tk.Label(root, background="#181818")
        self.image_label.pack(padx=8, pady=(8, 2))
        self.status = tk.StringVar(value="Starting D435f and MediaPipe Pose Landmarker Full...")
        tk.Label(
            root,
            textvariable=self.status,
            foreground="white",
            background="#181818",
            font=("Segoe UI", 12, "bold"),
        ).pack(fill="x", padx=10, pady=(2, 8))
        self.worker.start()
        root.after(30, self.refresh)

    def refresh(self) -> None:
        if self.closed:
            return
        newest: RenderedFrame | WorkerFailure | None = None
        while True:
            try:
                newest = self.worker.frames.get_nowait()
            except queue.Empty:
                break
        if isinstance(newest, WorkerFailure):
            self.status.set(f"ERROR: {newest.message}")
        elif newest is not None:
            image = Image.fromarray(newest.image, mode="RGB")
            maximum_width = max(960, self.root.winfo_screenwidth() - 80)
            maximum_height = max(540, self.root.winfo_screenheight() - 160)
            scale = min(maximum_width / image.width, maximum_height / image.height, 1.0)
            if scale < 1.0:
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            self.status.set(newest.status)
        self.root.after(30, self.refresh)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.worker.stop()
        self.root.after(100, self._finish_close)

    def _finish_close(self) -> None:
        if self.worker.is_alive():
            self.root.after(100, self._finish_close)
            return
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable shoulder/elbow/wrist RGB-D layout check.")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--serial", default="242322072812")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--status-file", type=Path, default=Path("outputs/pose_depth_status.json"))
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"pose model not found: {args.model}")
    config = RealSenseConfig(
        serial=args.serial,
        color_width=args.width,
        color_height=args.height,
        depth_width=args.width,
        depth_height=args.height,
        fps=args.fps,
        warmup_frames=15,
    )
    root = tk.Tk()
    PoseDepthViewer(
        root,
        PoseWorker(config, args.model, args.min_depth_m, args.max_depth_m, args.status_file),
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
