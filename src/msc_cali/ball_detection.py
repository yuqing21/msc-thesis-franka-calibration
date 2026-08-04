from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


class BallNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if not np.all(np.isfinite(values)) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera intrinsics must be finite with positive focal lengths")


@dataclass(frozen=True)
class BallDetectorConfig:
    target_rgb: tuple[int, int, int] = (0, 160, 70)
    chromaticity_tolerance: float = 0.20
    dominant_channel: int = 1
    min_channel_dominance: float = 0.08
    min_brightness: float = 0.12
    min_area_px: int = 80
    min_aspect_ratio: float = 0.60
    max_aspect_ratio: float = 1.67
    min_depth_m: float = 0.15
    max_depth_m: float = 2.5
    roi_xyxy: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class BallDetection:
    center_uv: tuple[float, float]
    point_camera_m: np.ndarray
    depth_m: float
    area_px: int
    radius_px: float
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_uv": list(self.center_uv),
            "point_camera_m": self.point_camera_m.tolist(),
            "depth_m": self.depth_m,
            "area_px": self.area_px,
            "radius_px": self.radius_px,
            "bbox_xyxy": list(self.bbox_xyxy),
            "confidence": self.confidence,
        }


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_y = np.empty(0, dtype=np.int32)
    best_x = np.empty(0, dtype=np.int32)

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component_y: list[int] = []
        component_x: list[int] = []
        while queue:
            y, x = queue.popleft()
            component_y.append(y)
            component_x.append(x)
            for neighbor_y, neighbor_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    queue.append((neighbor_y, neighbor_x))
        if len(component_y) > len(best_y):
            best_y = np.asarray(component_y, dtype=np.int32)
            best_x = np.asarray(component_x, dtype=np.int32)
    return best_y, best_x


def find_ball_pixels(
    rgb: Any,
    config: BallDetectorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column indices for the largest ball-like colour component."""
    config = config or BallDetectorConfig()
    rgb_array = np.asarray(rgb)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if config.dominant_channel not in (0, 1, 2):
        raise ValueError("dominant_channel must be 0, 1, or 2")

    rgb_float = rgb_array.astype(np.float64) / 255.0
    intensity_sum = rgb_float.sum(axis=2)
    chromaticity = rgb_float / np.maximum(intensity_sum[..., None], 1e-12)
    target = np.asarray(config.target_rgb, dtype=np.float64)
    target /= max(float(target.sum()), 1.0)
    color_distance = np.linalg.norm(chromaticity - target, axis=2)
    dominant = rgb_float[..., config.dominant_channel]
    other_channels = np.delete(rgb_float, config.dominant_channel, axis=2)
    dominance = dominant - np.max(other_channels, axis=2)
    mask = (
        (color_distance <= config.chromaticity_tolerance)
        & (dominance >= config.min_channel_dominance)
        & ((intensity_sum / 3.0) >= config.min_brightness)
    )

    if config.roi_xyxy is not None:
        x0, y0, x1, y1 = config.roi_xyxy
        if not (0 <= x0 < x1 <= rgb_array.shape[1] and 0 <= y0 < y1 <= rgb_array.shape[0]):
            raise ValueError("roi_xyxy lies outside the image")
        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1, x0:x1] = True
        mask &= roi_mask

    component_y, component_x = _largest_component(mask)
    area = int(len(component_y))
    if area < config.min_area_px:
        raise BallNotFoundError(f"largest colour component has only {area} pixels")

    x0, x1 = int(component_x.min()), int(component_x.max()) + 1
    y0, y1 = int(component_y.min()), int(component_y.max()) + 1
    aspect_ratio = (x1 - x0) / max(y1 - y0, 1)
    if not config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
        raise BallNotFoundError(f"largest colour component aspect ratio {aspect_ratio:.3f} is not ball-like")
    return component_y, component_x


def detect_ball(
    rgb: Any,
    depth_m: Any,
    intrinsics: CameraIntrinsics,
    config: BallDetectorConfig | None = None,
) -> BallDetection:
    """Detect a coloured ball in RGB and aligned depth arrays.

    RGB must use channel order R,G,B. Depth must already be registered to RGB
    and expressed in metres. Camera SDK acquisition is intentionally separate.
    """

    config = config or BallDetectorConfig()
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth_m, dtype=np.float64)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if depth_array.shape != rgb_array.shape[:2]:
        raise ValueError("depth_m must be aligned with RGB and have shape (H, W)")
    if config.dominant_channel not in (0, 1, 2):
        raise ValueError("dominant_channel must be 0, 1, or 2")

    component_y, component_x = find_ball_pixels(rgb_array, config)
    area = int(len(component_y))

    x0, x1 = int(component_x.min()), int(component_x.max()) + 1
    y0, y1 = int(component_y.min()), int(component_y.max()) + 1

    valid_depth = depth_array[component_y, component_x]
    valid_depth = valid_depth[
        np.isfinite(valid_depth)
        & (valid_depth >= config.min_depth_m)
        & (valid_depth <= config.max_depth_m)
    ]
    if len(valid_depth) < max(10, area // 10):
        raise BallNotFoundError("not enough valid aligned-depth samples inside the ball mask")

    center_u = float(np.median(component_x))
    center_v = float(np.median(component_y))
    z_m = float(np.median(valid_depth))
    point_camera = np.asarray(
        [
            (center_u - intrinsics.cx) * z_m / intrinsics.fx,
            (center_v - intrinsics.cy) * z_m / intrinsics.fy,
            z_m,
        ],
        dtype=np.float64,
    )
    rgb_float = rgb_array.astype(np.float64) / 255.0
    chromaticity = rgb_float / np.maximum(rgb_float.sum(axis=2)[..., None], 1e-12)
    target = np.asarray(config.target_rgb, dtype=np.float64)
    target /= max(float(target.sum()), 1.0)
    color_distance = np.linalg.norm(chromaticity - target, axis=2)
    color_quality = float(np.clip(1.0 - np.median(color_distance[component_y, component_x]), 0.0, 1.0))
    fill_ratio = area / max((x1 - x0) * (y1 - y0), 1)
    confidence = float(np.clip(0.7 * color_quality + 0.3 * fill_ratio, 0.0, 1.0))
    return BallDetection(
        center_uv=(center_u, center_v),
        point_camera_m=point_camera,
        depth_m=z_m,
        area_px=area,
        radius_px=float(np.sqrt(area / np.pi)),
        bbox_xyxy=(x0, y0, x1, y1),
        confidence=confidence,
    )
