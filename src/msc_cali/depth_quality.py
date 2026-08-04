from __future__ import annotations

from typing import Any

import numpy as np


def analyze_depth_sequence(
    depth_frames_m: Any,
    camera_timestamps_ms: Any,
    frame_numbers: Any,
    *,
    requested_fps: float,
    min_valid_depth_m: float = 0.1,
    max_valid_depth_m: float = 5.0,
) -> dict[str, Any]:
    depths = np.asarray(depth_frames_m, dtype=np.float64)
    timestamps = np.asarray(camera_timestamps_ms, dtype=np.float64)
    numbers = np.asarray(frame_numbers, dtype=np.int64)
    if depths.ndim != 3:
        raise ValueError("depth_frames_m must have shape (N, H, W)")
    if len(depths) < 2:
        raise ValueError("at least two depth frames are required")
    if timestamps.shape != (len(depths),) or numbers.shape != (len(depths),):
        raise ValueError("timestamps and frame numbers must match the depth-frame count")
    if not np.all(np.isfinite(timestamps)) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("camera timestamps must be finite and strictly increasing")

    valid = (
        np.isfinite(depths)
        & (depths >= min_valid_depth_m)
        & (depths <= max_valid_depth_m)
    )
    height, width = depths.shape[1:]
    y0, y1 = height // 4, (3 * height) // 4
    x0, x1 = width // 4, (3 * width) // 4
    roi_depths = depths[:, y0:y1, x0:x1]
    roi_valid = valid[:, y0:y1, x0:x1]

    interval_ms = np.diff(timestamps)
    measured_fps = 1000.0 / float(np.median(interval_ms))
    frame_increments = np.diff(numbers)
    dropped_frames = int(np.maximum(frame_increments - 1, 0).sum())
    nonmonotonic_frame_numbers = int(np.count_nonzero(frame_increments <= 0))

    eligible_pixels = roi_valid.sum(axis=0) >= max(2, int(np.ceil(0.8 * len(depths))))
    if np.any(eligible_pixels):
        selected_depths = roi_depths[:, eligible_pixels]
        selected_valid = roi_valid[:, eligible_pixels]
        selected_depths = np.where(selected_valid, selected_depths, np.nan)
        temporal_std_mm = np.nanstd(selected_depths, axis=0) * 1000.0
        temporal_std_median_mm = float(np.nanmedian(temporal_std_mm))
        temporal_std_p95_mm = float(np.nanpercentile(temporal_std_mm, 95))
    else:
        temporal_std_median_mm = float("nan")
        temporal_std_p95_mm = float("nan")

    all_valid_depths = depths[valid]
    roi_valid_depths = roi_depths[roi_valid]
    if len(all_valid_depths) == 0 or len(roi_valid_depths) == 0:
        depth_percentiles_m = [float("nan")] * 3
        roi_median_depth_m = float("nan")
    else:
        depth_percentiles_m = [float(value) for value in np.percentile(all_valid_depths, [5, 50, 95])]
        roi_median_depth_m = float(np.median(roi_valid_depths))

    full_valid_ratios = valid.mean(axis=(1, 2))
    roi_valid_ratios = roi_valid.mean(axis=(1, 2))
    return {
        "schema": "realsense-depth-quality/v1",
        "frame_count": int(len(depths)),
        "resolution": [int(width), int(height)],
        "requested_fps": float(requested_fps),
        "measured_fps": measured_fps,
        "camera_interval_median_ms": float(np.median(interval_ms)),
        "camera_interval_p95_ms": float(np.percentile(interval_ms, 95)),
        "dropped_frames_from_frame_numbers": dropped_frames,
        "nonmonotonic_frame_numbers": nonmonotonic_frame_numbers,
        "valid_depth_ratio_mean": float(np.mean(full_valid_ratios)),
        "valid_depth_ratio_min": float(np.min(full_valid_ratios)),
        "center_roi_xyxy": [x0, y0, x1, y1],
        "center_valid_depth_ratio_mean": float(np.mean(roi_valid_ratios)),
        "center_valid_depth_ratio_min": float(np.min(roi_valid_ratios)),
        "center_median_depth_m": roi_median_depth_m,
        "depth_percentiles_m_p05_p50_p95": depth_percentiles_m,
        "center_temporal_std_median_mm": temporal_std_median_mm,
        "center_temporal_std_p95_mm": temporal_std_p95_mm,
        "temporal_noise_eligible_pixel_ratio": float(np.mean(eligible_pixels)),
        "notes": [
            "Temporal noise assumes the camera and observed scene remained stationary.",
            "These metrics do not measure absolute depth accuracy; a known-distance planar target is required.",
            "Metrics are computed from raw aligned depth without post-processing filters.",
        ],
    }

