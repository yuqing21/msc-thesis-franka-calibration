from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ball_detection import CameraIntrinsics


class RealSenseUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RealSenseConfig:
    serial: str | None = None
    color_width: int = 640
    color_height: int = 480
    depth_width: int = 640
    depth_height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    timeout_ms: int = 5000

    def __post_init__(self) -> None:
        dimensions = (self.color_width, self.color_height, self.depth_width, self.depth_height)
        if any(value <= 0 for value in dimensions):
            raise ValueError("stream dimensions must be positive")
        if self.fps <= 0 or self.warmup_frames < 0 or self.timeout_ms <= 0:
            raise ValueError("fps and timeout must be positive; warmup_frames cannot be negative")


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    name: str
    serial: str
    firmware_version: str
    usb_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "serial": self.serial,
            "firmware_version": self.firmware_version,
            "usb_type": self.usb_type,
        }


@dataclass(frozen=True)
class RealSenseFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    host_timestamp_s: float
    camera_timestamp_ms: float
    frame_number: int
    device: RealSenseDeviceInfo


def _load_pyrealsense2() -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except ImportError as exc:
        raise RealSenseUnavailableError(
            "pyrealsense2 is not installed. Install cali/requirements-realsense.txt "
            "with F:/bishe_ai_project/cali/.venv/Scripts/python.exe."
        ) from exc


def _safe_device_info(device: Any, rs: Any, key_name: str) -> str:
    key = getattr(rs.camera_info, key_name, None)
    if key is None:
        return "unknown"
    try:
        if hasattr(device, "supports") and not device.supports(key):
            return "unknown"
        return str(device.get_info(key))
    except RuntimeError:
        return "unknown"


def _device_info(device: Any, rs: Any) -> RealSenseDeviceInfo:
    return RealSenseDeviceInfo(
        name=_safe_device_info(device, rs, "name"),
        serial=_safe_device_info(device, rs, "serial_number"),
        firmware_version=_safe_device_info(device, rs, "firmware_version"),
        usb_type=_safe_device_info(device, rs, "usb_type_descriptor"),
    )


def list_realsense_devices(rs_module: Any | None = None) -> list[RealSenseDeviceInfo]:
    rs = rs_module or _load_pyrealsense2()
    context = rs.context()
    return [_device_info(device, rs) for device in context.query_devices()]


class RealSenseD435f:
    """D435f RGB/depth acquisition with depth registered to RGB pixels."""

    def __init__(self, config: RealSenseConfig | None = None, *, rs_module: Any | None = None) -> None:
        self.config = config or RealSenseConfig()
        self._rs = rs_module
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._depth_scale: float | None = None
        self._device: RealSenseDeviceInfo | None = None

    @property
    def device_info(self) -> RealSenseDeviceInfo:
        if self._device is None:
            raise RuntimeError("camera is not started")
        return self._device

    def start(self) -> RealSenseDeviceInfo:
        if self._pipeline is not None:
            return self.device_info
        rs = self._rs or _load_pyrealsense2()
        self._rs = rs
        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.serial:
            stream_config.enable_device(self.config.serial)
        stream_config.enable_stream(
            rs.stream.depth,
            self.config.depth_width,
            self.config.depth_height,
            rs.format.z16,
            self.config.fps,
        )
        stream_config.enable_stream(
            rs.stream.color,
            self.config.color_width,
            self.config.color_height,
            rs.format.bgr8,
            self.config.fps,
        )
        try:
            profile = pipeline.start(stream_config)
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            if not np.isfinite(depth_scale) or depth_scale <= 0:
                raise RuntimeError("camera returned an invalid depth scale")
            device_info = _device_info(device, rs)
            if "D435" not in device_info.name.upper():
                raise RuntimeError(
                    "connected RealSense device is {!r}, expected a D435/D435f".format(device_info.name)
                )
            self._pipeline = pipeline
            self._align = rs.align(rs.stream.color)
            self._depth_scale = depth_scale
            self._device = device_info
            for _ in range(self.config.warmup_frames):
                pipeline.wait_for_frames(self.config.timeout_ms)
            return device_info
        except Exception:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            raise

    def capture(self) -> RealSenseFrame:
        if self._pipeline is None or self._align is None or self._depth_scale is None:
            raise RuntimeError("camera must be started before capture")
        frames = self._pipeline.wait_for_frames(self.config.timeout_ms)
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("D435f returned an incomplete aligned frameset")

        bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise RuntimeError("unexpected D435f color-frame shape")
        if depth_raw.shape != bgr.shape[:2]:
            raise RuntimeError("aligned D435f depth shape does not match color shape")
        rgb = bgr[..., ::-1].copy()
        depth_m = depth_raw.astype(np.float32) * self._depth_scale

        video_profile = color_frame.profile.as_video_stream_profile()
        intrinsics = video_profile.intrinsics
        camera_intrinsics = CameraIntrinsics(
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.ppx),
            cy=float(intrinsics.ppy),
        )
        return RealSenseFrame(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=camera_intrinsics,
            host_timestamp_s=time.time(),
            camera_timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_frame_number()),
            device=self.device_info,
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._depth_scale = None
        self._device = None

    def __enter__(self) -> "RealSenseD435f":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
