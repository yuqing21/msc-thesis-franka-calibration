# Windows端说明

## 职责

Windows端负责所有相机与计算任务：

- Intel RealSense D435f的RGB、深度和内参采集；
- RGB对齐深度、绿色球检测和已知半径球面拟合；
- 相机—Franka外参求解与独立验证；
- 后续MediaPipe或同类方法的手腕检测；
- 把低频、有限幅、已验证的目标预览发送给Herbert。

Windows端不直接运行Franka实时控制环。

## 固定环境

- 工作目录：`F:\bishe_ai_project\cali`
- Python：`F:\bishe_ai_project\cali\.venv\Scripts\python.exe`
- 标定项目：`F:\bishe_ai_project\cali`
- D435f序列号：`242322072812`
- 正式采集流：1280×720，30 FPS，depth对齐到RGB

## 启动前检查

```powershell
tailscale status
tailscale ping 100.68.210.77
ssh liuy@100.68.210.77
```

相机检查：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m msc_cali.cli.realsense_info
.\.venv\Scripts\python.exe -m msc_cali.cli.realsense_live `
  --serial 242322072812 --width 1280 --height 720 --fps 30
```

## 采集一个标定位置

实时预览必须先关闭，以释放相机：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m msc_cali.cli.capture_calibration_pose `
  --sample-id p006 `
  --output-dir "F:\bishe_ai_project\cali\data\calibration_session_20260804"
```

程序会采集30帧、执行球面拟合和分组稳定性检查，并在采集前后读取一次
Franka状态。失败样本不应通过放宽阈值强行保留。

## 数据保存

- `frames.npz`：原始RGB/深度数组，仅保存在本地和独立备份中；
- `metadata.json`：相机内参、球心、拟合质量、Franka位姿和漂移；
- `preview_rgb.png`：人工核查检测目标与遮挡情况。

## 手腕跟踪待办

当前尚未实现手腕检测。下一步在Windows端加入：

1. MediaPipe Hand/Pose或等价关键点模型；
2. 使用对齐深度把腕部像素反投影为相机三维点；
3. 使用标定外参转换到Franka基座坐标；
4. 丢帧、置信度、深度空洞、速度和工作空间限制；
5. 先只显示/记录目标，人工验证后才允许进入机器人控制链。
