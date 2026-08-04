# Calibration session 2026-08-04

Valid samples: `p001`–`p005`.

Each committed sample directory contains:

- `metadata.json`: camera intrinsics, fitted sphere center, quality metrics,
  Franka end-effector state, and static-drift checks;
- `preview_rgb.png`: visual confirmation of the target and scene.

The corresponding `frames.npz` files contain the immutable 30-frame RGB and
aligned-depth arrays. They remain on the Windows workstation and its separate
data backup, and are intentionally ignored by Git because each file is about
74–78 MB.

Rejected `p006` attempts were not saved because grouped sphere-center p95
exceeded the 15 mm acceptance threshold.
