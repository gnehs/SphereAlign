# DJI Osmo 360 notes

實測 Osmo 360 `.OSV` 可為雙 3840x3840 fisheye HEVC streams，加上 DJI timed metadata tracks。

開源 `telemetry-parser` 已包含 `dvtm_oq101` protobuf：

- `FrameMetaOfCamera.camera_attitude`
- `FrameMetaOfCamera.camera_acc`
- `FrameMetaOfIMU.IMU_attitude_after_fusion`
- `ClipMeta.imu_sampling_rate`
- `ClipMeta.digital_focal_length`
- `ClipMeta.distortion_coefficients`
- `StreamMeta.pano_dewarp_params`

其中 `DeviceAttitude.attitude` 是一組 fused quaternions，對應 sensor vsync interval。

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
