# Herbert ROS control workspace

This directory is deployed to `/home/liuy/msc_thesis` on Herbert.

The files in `src/franka_control_bridge` were hash-checked against the deployed
workspace on 2026-08-04. Build products, the isolated libfranka checkout, and
compiled binaries are intentionally not copied into Git.

Responsibilities:

- receive low-rate target previews from Windows over Tailscale;
- publish the preview as `/cali/preview_target` in `panda_link0`;
- provide a separately gated Franka Hand grasp client;
- never run camera, AI, or calibration computation.

The TCP bridge does **not** execute arm or gripper motion. Unknown commands are
rejected. The gripper client is dry-run by default and requires both
`--execute` and `--confirm ENABLE_GRIPPER` before it sends a ROS action.

## Build on Herbert

```bash
cd /home/liuy/msc_thesis
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Robot-state compatibility note

The installed ROS Noetic libfranka 0.9.2 uses an older server protocol than the
current robot system. Do not replace it globally. An isolated libfranka 0.13.3
installation is kept at:

```text
/home/liuy/msc_thesis/vendor/libfranka-0.13.3-install
```

`tools/robot_state_once.cpp` is compiled to
`/home/liuy/msc_thesis/bin/robot_state_once` with an RPATH to that isolated
library. It uses `readOnce()` only and does not issue motion commands.

## Start the preview bridge

```bash
roscore
```

In another terminal:

```bash
cd /home/liuy/msc_thesis
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch franka_control_bridge control_bridge.launch
```

## Gripper staging

Dry-run, no ROS command is sent:

```bash
rosrun franka_control_bridge gripper_grasp.py \
  --width-m 0.060 --speed-m-s 0.020 --force-n 10
```

Real execution must wait until the Hand is installed, the ball diameter and
safe force are measured, the action server is running, and the emergency stop
is supervised. At that point the local operator can explicitly add:

```text
--execute --confirm ENABLE_GRIPPER
```
