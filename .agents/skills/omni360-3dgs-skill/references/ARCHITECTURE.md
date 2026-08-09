# Architecture

## Canonical data model

```text
Camera-specific file
       |
       v
+------------------+
|  CameraAdapter   |
+------------------+
       |
       v
+------------------+
|  CaptureBundle   |
|------------------|
| lenses[]         |
| frame_clock      |
| telemetry        |
| calibration      |
| rig_hint         |
+------------------+
       |
       +--> Analyze / adaptive keyframes
       +--> Dynamic masks
       +--> Native-fisheye SfM
       +--> Trainer export
```

`CaptureBundle` 是資料鏈唯一 canonical boundary。後端不能依賴 DJI / Insta360 私有格式。

## Why native dual-fisheye first

360 stitch 通常需要：lens warp -> overlap selection -> seam placement -> exposure/color blend -> equirectangular/cubemap resample。

這些操作都可能製造：

- 接縫處 duplicated / missing geometry
- moving object ghosting
- interpolation blur
- 一個實體 ray 被轉成不完全一致的 stitched pixel

因此 SfM 直接處理兩顆已校正 fisheye 比「先 stitch 再猜 camera geometry」更乾淨。

若 trainer 只接受 pinhole，做法是：

```text
lens0 raw fisheye -> perspective tiles A/B/C/...
lens1 raw fisheye -> perspective tiles D/E/F/...
```

每個 tile 的 virtual camera pose 可由實體 lens pose + 固定 tile rotation 精確算出。不要先把兩顆鏡頭 blend 成 panorama。

## Adaptive keyframes

不要寫死「門 detector」。過道/門口真正需要加密的原因通常是：

- optical flow 快速上升
- visible feature set 快速變化
- 視角旋轉
- 新空間突然進入 FOV
- overlap 快速下降

因此密度函數應以一般化 transition score 控制：

```text
transition = w_flow * normalized_flow
           + w_scene * normalized_scene_delta
           + w_imu * normalized_angular_speed
```

由 `base_fps` 平滑增加到 `dense_fps`。

## Blur policy

不要只做 `variance(Laplacian) < fixed_threshold`。解析度、ISO、denoise、lens edge distortion 都會讓固定門檻失真。

採：

1. proxy 全片計算 sharpness distribution
2. dataset percentile rejection
3. absolute floor 防止整段都糊
4. 若被選 keyframe 模糊，在 +/- repair window 尋找最近且較清晰 frame

## Mask semantics

分兩層：

1. `lens-valid mask`：把 fisheye 圓外黑區排掉。
2. `dynamic-object mask`：person / car 等不要進 feature matching 或 training loss 的區域。

輸出統一：white = keep, black = ignore。
