---
name: omni360-3dgs
summary: 一鍵把 360 相機原始素材轉成可供 COLMAP / LichtFeld / 3DGS 使用的資料集。保留原生雙魚眼、IMU、rig、mask 與自適應 keyframe，並以可插拔 CameraAdapter 支援 DJI Osmo 360、未來 Insta360 / GoPro MAX 等相機。
---

# Omni360 -> 3DGS Skill

這個 skill 的核心原則是：**不要太早 stitch**。

相機原始資料先標準化成 `CaptureBundle`，保留每顆實體鏡頭的原始投影、時間戳、校正與 IMU。SfM 優先直接使用原始 fisheye；equirectangular / cubemap 只做相容輸出。

## 何時使用

當使用者提供 360 相機素材並要求：

- 建立 3DGS / Gaussian Splatting 訓練資料集
- 自動抽幀、去除模糊影像
- 在走廊、門口、轉角等過渡區域提高 keyframe 密度
- 自動遮罩人、車或其他動態物件
- 使用 IMU / rig / 多鏡頭資訊改善 COLMAP 對齊
- 輸出給 LichtFeld Studio、COLMAP 或其他 trainer

## Pipeline

1. `probe`：辨識相機 adapter、影片 streams、FPS、解析度、metadata capabilities。
2. `telemetry`：抽取 IMU / quaternion / accelerometer / calibration，保留原始時間軸。
3. `analyze`：低解析度掃描所有實體 lens streams，計算 sharpness、optical flow、scene delta。
4. `select`：自適應 keyframe。普通場景使用 `base_fps`；視覺 motion / scene transition / IMU rotation 提高時漸進提高到 `dense_fps`。
5. `extract`：只解碼被選中的原始 fisheye full-resolution frames；同一 timestamp 的不同鏡頭保持 frame identity。
6. `mask`：建立 lens-valid mask，並可用 instance segmentation 排除 person / bicycle / car / motorcycle / bus / truck 等類別。
7. `sfm`：COLMAP fisheye + multi-camera rig。先以線性數量的 pair list 做 temporal / cross-lens matching；未知 rig extrinsics 時可兩階段估計再固定。若 adapter 能提供已驗證的 per-lens gravity，寫入 COLMAP pose priors 並啟用 gravity-aware global rotation averaging。
8. `validate`：檢查 registration ratio、reprojection、rig consistency、IMU rotation consistency、frame gaps。
9. `export`：輸出 `native_fisheye`、`lichtfeld` 或 `pinhole_tiles` profile。

## 不可違反的設計

- **禁止預設把兩顆 fisheye stitch 成 equirectangular 再做 SfM。** 這會引入 seam、blend、重採樣和接縫幾何錯誤。
- 原始 lens frame 是 canonical image。
- 兩顆鏡頭同一時間點必須共享 `frame_id`。
- 任何相機專屬解析都只能存在 `CameraAdapter`，核心 pipeline 不得檢查 `.OSV`、`.INSV` 等副檔名來決定幾何。
- accelerometer 不得直接雙積分當作可靠 translation prior。
- stock COLMAP 不得被描述成會直接吃 quaternion orientation prior。IMU 優先轉成每張影像的 `PosePrior.gravity`（僅在 adapter 已驗證 coordinate transform 時），供新版 COLMAP Global Mapper 的 gravity-aware rotation averaging 使用；完整 quaternion rotational prior 仍是可選 extension。
- mask 在 COLMAP feature extraction 中遵循：黑色 = 排除 feature，白色 = 保留。
- 若 trainer 不支援 native fisheye，先從每顆 fisheye **各自**轉成 perspective tiles / undistorted view，不跨鏡頭做 seam blending。

## CameraAdapter contract

每個 adapter 至少提供：

- `probe(path) -> CaptureDescriptor`
- `video_streams()`：實體鏡頭及 ffmpeg stream index
- `frame_clock()`：FPS / timestamp mapping
- `lens_model()`：projection、尺寸、有效成像區、intrinsics/distortion（若可得）
- `telemetry()`：IMU samples / fused attitude / gravity（若可得）
- `rig_hint()`：實體鏡頭間固定 extrinsics（若可得）
- `capabilities`：`imu`, `fused_attitude`, `factory_intrinsics`, `rig_extrinsics`, `native_fisheye`

核心程式只依賴以上 contract。

## DJI Osmo 360

第一版 adapter：`DjiOsmo360Adapter`。

已確認 `.OSV` 可包含：

- 兩路 3840x3840 fisheye HEVC video streams
- DJI `djmd` / `dbgi` timed metadata
- `camera_attitude`
- `camera_acc`
- `IMU_attitude_after_fusion`
- `imu_sampling_rate`
- frame timestamps
- lens metadata / focal / distortion（依素材 metadata 可用性）

優先使用 `telemetry-parser` 的 `dvtm_oq101` 支援，不自行猜 protobuf 欄位。

## Default command

```bash
python scripts/prepare_capture.py INPUT.OSV -o OUTPUT --config config.example.yaml
```

Windows：

```powershell
py scripts\prepare_capture.py D:\capture\CAM_xxx.OSV -o D:\3dgs\scene --config config.example.yaml
```

先測環境：

```bash
python scripts/doctor.py
```

## Output contract

```text
OUTPUT/
  images/
    lens0/
    lens1/
  masks/
    lens0/
    lens1/
  masks_colmap/
    lens0/
    lens1/
  metadata/
    capture.json
    keyframes.csv
    telemetry.json
    normalized_imu.json
    pipeline_report.json
  database.db
  rig_config.json
  sparse/
    0/
      cameras.bin|txt
      images.bin|txt
      points3D.bin|txt
      rigs.bin|txt
      frames.bin|txt
```

這個目錄本身就是 COLMAP dataset；LichtFeld profile 不應另外 stitch 一份影像。

## 執行策略

- 先跑 `doctor.py`。
- 讀 `config.example.yaml`，不要把參數 hard-code。
- 長素材先 `--no-sfm` 驗證抽幀與 mask，再正式跑 SfM。
- 若 initial SfM registration < `validation.min_registered_ratio`，先檢查 pair coverage、blur rejection 與 intrinsics，不要直接增加 3DGS steps。
- 若 fisheye intrinsics 無 factory metadata，使用 COLMAP fisheye model估計；不要改成 pinhole 假裝 distortion 不存在。

更多設計細節見 `references/`。
