# MSc Camera–Franka calibration and wrist-tracking project

> 项目状态：**进行中**。标定求解与腕部追踪仍在持续开发、验证中。

This repository contains the two-computer calibration and tracking stack used
for the MSc thesis experiment. It is intentionally split by responsibility:

| Side | Location in this repository | Responsibility |
|---|---|---|
| Windows | repository root (`src`, `configs`, `tests`, `docs`) | Intel RealSense D435f acquisition, RGB/depth processing, calibration, wrist tracking, and target generation |
| Herbert | [`herbert_ws`](herbert_ws) | ROS Noetic communication, read-only Franka state acquisition, target preview, and explicitly gated Franka Hand control |

The two computers communicate through Tailscale. Herbert is connected to the
Franka controller on the robot network. GPU-heavy or vision computation stays
on Windows; Herbert is kept as a robot-I/O computer.

## Safety boundary

- The current bridge does **not** move the Franka arm.
- Arm motion must not be enabled until calibration has passed independent
  validation, workspace limits and a supervised emergency-stop procedure are
  implemented.
- Gripper execution is separately gated and must only be used with an on-site
  operator, a supported object, clear hands, and an attended emergency stop.
- Never run autonomous robot motion over Tailscale as a real-time control loop.

## Current status — 2026-08-04

- D435f detected over USB 3.2; intrinsics and depth quality recorded.
- Laboratory camera/person/robot layout fixed and checked at 1280×720, 30 FPS.
- Green 50 mm soft ball detection uses RGB segmentation, aligned depth, known
  radius sphere fitting, and temporal grouping.
- Five calibration poses (`p001`–`p005`) passed the current quality gates.
- Raw 30-frame NPZ files remain local and are intentionally ignored by Git.
  Metadata and preview images are versioned.
- `p006` was rejected because temporal sphere-center p95 exceeded 15 mm.
- Camera-to-base extrinsics are **not solved yet**. The solver must jointly
  estimate the camera-to-base transform and the unknown ball-center offset in
  the end-effector frame.
- Wrist tracking is **not implemented yet**; it is the next Windows-side task.

See the [2026-08-04 experiment log](docs/logs/2026-08-04.md) and the
[2026-08-05 continuation checklist](docs/NEXT_SESSION_2026-08-05.md).

## Documentation

- [Windows-side setup and commands](docs/WINDOWS_SIDE_ZH.md)
- [Herbert-side setup and commands](docs/HERBERT_SIDE_ZH.md)
- [Calibration procedure](docs/CALIBRATION_PROCEDURE_ZH.md)
- [Herbert ROS workspace](herbert_ws/README.md)

## Windows tests

Run from `F:\bishe_ai_project\cali` with the calibration environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## RealSense quick check

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m msc_cali.cli.realsense_info
.\.venv\Scripts\python.exe -m msc_cali.cli.realsense_live `
  --serial 242322072812 --width 1280 --height 720 --fps 30
```

## Data policy

`data/calibration_session_*/p*/frames.npz`, ad-hoc camera captures, generated
outputs, virtual environments, caches, and build products are excluded from
Git. Do not delete the local raw NPZ files: they are the immutable source data
for reprocessing and should be backed up separately from GitHub.
