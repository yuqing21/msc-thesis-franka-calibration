# Herbert端说明

## 职责

Herbert是无GPU的机器人I/O电脑，负责：

- ROS Noetic节点与话题；
- 从Franka控制器读取机器人状态；
- 接收Windows发送的低频目标预览；
- 在显式安全门控下控制Franka Hand；
- 不运行相机、神经网络或外参求解。

部署目录为`/home/liuy/msc_thesis`，Windows中的源码备份位于
`cali/herbert_ws`。2026-08-04已核对：部署端与备份端的8个源码文件
SHA-256完全一致。

## 已知版本兼容性

- Ubuntu 20.04.6 LTS
- 实时内核：`5.9.1-rt20`
- ROS Noetic
- 系统`ros-noetic-libfranka 0.9.2`使用server protocol 5
- 当前Franka系统使用server protocol 7

因此系统libfranka不能读取该机器人。已在以下位置隔离安装
libfranka 0.13.3，且不覆盖系统0.9.2：

```text
/home/liuy/msc_thesis/vendor/libfranka-0.13.3
/home/liuy/msc_thesis/vendor/libfranka-0.13.3-install
```

只读状态工具：

```text
/home/liuy/msc_thesis/bin/robot_state_once
```

使用方式：

```bash
/home/liuy/msc_thesis/bin/robot_state_once 172.16.0.2
```

该工具只调用`readOnce()`，不会发送运动命令。Desk中必须启用FCI；若
FCI关闭，工具应明确失败而不是自动修改Desk状态。

## 构建ROS工作区

```bash
cd /home/liuy/msc_thesis
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## 启动顺序

终端1：

```bash
roscore
```

终端2：

```bash
cd /home/liuy/msc_thesis
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch franka_control_bridge control_bridge.launch
```

## Franka Hand

`gripper_grasp.py`默认dry-run。真实动作必须同时带有`--execute`和明确
确认字符串，并满足现场支撑物、手部离开、急停看守等条件。2026-08-04
绿色软球使用约2 N低力夹持；实验结束时夹爪已打开到约79.9 mm。

## 禁止事项

- 不通过Tailscale运行1 kHz控制环；
- 不在无人看守时触发机械臂或夹爪动作；
- 不用未验证的关节角替代Desk中已保存的任务；
- 不把系统libfranka 0.9.2全局替换为0.13.3；
- 不把`vendor`、`build`、`devel`或编译二进制提交到GitHub。
