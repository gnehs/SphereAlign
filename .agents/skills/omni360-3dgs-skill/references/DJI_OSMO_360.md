# DJI Osmo 360 notes

實測 Osmo 360 `.OSV` 可為雙 3840x3840 fisheye HEVC streams，加上 DJI timed metadata tracks。Osmo 360 II 的實際樣本已確認相同的雙串流布局，並可從 OQ102 envelope 取得 fused attitude 與 clip lens metadata。

第一代 Osmo 360 可用開源 `telemetry-parser` 的 `dvtm_oq101` protobuf：

- `FrameMetaOfCamera.camera_attitude`
- `FrameMetaOfCamera.camera_acc`
- `FrameMetaOfIMU.IMU_attitude_after_fusion`
- `ClipMeta.imu_sampling_rate`
- `ClipMeta.digital_focal_length`
- `ClipMeta.distortion_coefficients`
- `StreamMeta.pano_dewarp_params`

其中 `DeviceAttitude.attitude` 是一組 fused quaternions，對應 sensor vsync interval。

Osmo 360 II 的 OQ102 尚未假設成上游 `dvtm_oq101`；adapter 只以 DJI 共通 envelope 讀取已驗證的 frame timestamp、fused attitude，以及 clip-level lens 欄位。實機樣本的 factory profile 為 `OPENCV_FISHEYE`：3840×3840、`fx=fy=1143.9465332`、中心 `(1920,1920)`，以及 `k=[0.15513110, 0.13714090, -0.09386140, 0.00417040]`。只有尺寸相符且數值有限時才提供給 COLMAP。

OQ102 的 optical-occlusion curve 若全為零，表示素材沒有可用的固定遮擋邊界；流程回退到魚眼圓形遮罩，不會把零值當成有效曲線，也不會把 pano dewarp 欄位猜成通用投影模型。

Osmo 360 II 的 D-Log M 標記可辨識，但目前沒有獨立驗證的官方 II LUT，因此自動色彩處理保留原生像素。`cam_extri_q` 只包含旋轉提示，缺少足以建立 rig 的完整平移與座標語意；它不是 factory rig extrinsics，兩鏡外參仍需視覺 bootstrap。

## Coordinate systems

DJI quaternion 不可直接複製成 COLMAP `qvec`。

必須明確處理：

- DJI body/IMU frame
- physical lens camera frame
- rig frame
- COLMAP world-to-camera convention

`telemetry-parser` 自己也會對 DJI quaternion 乘固定 rotation，證明兩者 convention 不同。

第一版 skill 將 IMU 當：

- keyframe density signal
- rotation consistency check
- future VIO / custom BA input

而不是直接寫入 `images.txt` 當絕對姿態。

## IMU integration levels

### Level 1 - gravity prior (preferred)

若已驗證 DJI body/lens/COLMAP 座標轉換，從 fused attitude 推導每張實體 lens image 的 gravity vector，寫入 COLMAP `PosePrior.gravity`。新版 Global Mapper 可在 rotation averaging 使用 gravity prior。

這比直接把 quaternion 當 `images.txt` qvec 安全，因為它只約束「哪個方向是下」，不會假裝 IMU 已提供可靠 translation。

### Level 2 - full orientation constraint

完整 quaternion constraint 需要自訂 BA/VIO/pose graph backend。不要在 stock COLMAP 模式假裝已實作。
