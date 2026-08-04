from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageTk

from msc_cali.realsense import RealSenseConfig, RealSenseD435f, RealSenseFrame


def depth_to_rgb(depth_m: np.ndarray, minimum_m: float, maximum_m: float) -> np.ndarray:
    """Convert metric depth to a compact blue-to-red visualization."""
    if maximum_m <= minimum_m:
        raise ValueError("maximum depth must be greater than minimum depth")
    if depth_m.ndim != 2:
        raise ValueError("depth image must be two-dimensional")

    valid = np.isfinite(depth_m) & (depth_m >= minimum_m) & (depth_m <= maximum_m)
    normalized = np.clip((depth_m - minimum_m) / (maximum_m - minimum_m), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    image = (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    image[~valid] = 0
    return image


@dataclass(frozen=True)
class WorkerFailure:
    message: str


class CaptureWorker(threading.Thread):
    def __init__(self, config: RealSenseConfig) -> None:
        super().__init__(name="realsense-capture", daemon=True)
        self.config = config
        self.stop_event = threading.Event()
        self.frames: queue.Queue[RealSenseFrame | WorkerFailure] = queue.Queue(maxsize=1)

    def publish(self, item: RealSenseFrame | WorkerFailure) -> None:
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            with RealSenseD435f(self.config) as camera:
                while not self.stop_event.is_set():
                    self.publish(camera.capture())
        except Exception as exc:  # GUI must surface device/USB errors instead of disappearing.
            self.publish(WorkerFailure(f"{type(exc).__name__}: {exc}"))

    def stop(self) -> None:
        self.stop_event.set()


class LiveViewer:
    def __init__(
        self,
        root: tk.Tk,
        worker: CaptureWorker,
        *,
        minimum_depth_m: float,
        maximum_depth_m: float,
    ) -> None:
        self.root = root
        self.worker = worker
        self.minimum_depth_m = minimum_depth_m
        self.maximum_depth_m = maximum_depth_m
        self.closed = False

        root.title("D435f 实时画面（左：彩色｜右：对齐深度）")
        root.configure(background="#181818")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _event: self.close())

        self.image_label = tk.Label(root, background="#181818")
        self.image_label.pack(padx=8, pady=(8, 2))
        self.status = tk.StringVar(value="正在启动 D435f，请稍候……")
        tk.Label(
            root,
            textvariable=self.status,
            foreground="white",
            background="#181818",
            font=("Segoe UI", 11),
        ).pack(fill="x", padx=10, pady=(2, 1))
        tk.Label(
            root,
            text="调整到小球、机械臂工作区和黑色椅子位置都在画面内；按 Esc 关闭。",
            foreground="#f0c674",
            background="#181818",
            font=("Segoe UI", 11),
        ).pack(fill="x", padx=10, pady=(1, 8))

        self.photo: ImageTk.PhotoImage | None = None
        self.worker.start()
        self.root.after(30, self.refresh)

    def refresh(self) -> None:
        if self.closed:
            return
        newest: RealSenseFrame | WorkerFailure | None = None
        while True:
            try:
                newest = self.worker.frames.get_nowait()
            except queue.Empty:
                break

        if isinstance(newest, WorkerFailure):
            self.status.set(f"摄像头错误：{newest.message}")
        elif newest is not None:
            depth_rgb = depth_to_rgb(newest.depth_m, self.minimum_depth_m, self.maximum_depth_m)
            combined = np.concatenate((newest.rgb, depth_rgb), axis=1)
            self.photo = ImageTk.PhotoImage(Image.fromarray(combined, mode="RGB"))
            self.image_label.configure(image=self.photo)

            height, width = newest.depth_m.shape
            center_depth = float(newest.depth_m[height // 2, width // 2])
            center_text = f"{center_depth:.3f} m" if center_depth > 0.0 else "无有效深度"
            self.status.set(
                f"{newest.device.name}  S/N {newest.device.serial}  USB {newest.device.usb_type}  "
                f"帧 {newest.frame_number}  中心深度 {center_text}  "
                f"显示范围 {self.minimum_depth_m:.1f}–{self.maximum_depth_m:.1f} m"
            )

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
    parser = argparse.ArgumentParser(description="Display live aligned RGB/depth from an Intel RealSense D435f.")
    parser.add_argument("--serial", help="Optional camera serial number")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    args = parser.parse_args()

    if args.max_depth_m <= args.min_depth_m:
        parser.error("--max-depth-m must be greater than --min-depth-m")

    config = RealSenseConfig(
        serial=args.serial,
        color_width=args.width,
        color_height=args.height,
        depth_width=args.width,
        depth_height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
    )
    root = tk.Tk()
    worker = CaptureWorker(config)
    LiveViewer(
        root,
        worker,
        minimum_depth_m=args.min_depth_m,
        maximum_depth_m=args.max_depth_m,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
