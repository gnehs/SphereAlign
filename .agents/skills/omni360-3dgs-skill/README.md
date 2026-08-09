# Omni360 -> 3DGS Skill (v0.1)

把 360 相機原始素材整理成可直接進 COLMAP / LichtFeld Studio / 3DGS 的資料鏈。

## 目前實作

- DJI Osmo 360 `.OSV` 自動辨識
- 保留兩路原始 3840x3840 fisheye，不先 stitch
- 低解析度全片分析
- 模糊 / sharpness 過濾與鄰近 frame 修復
- adaptive keyframes：一般區域低密度、motion / scene transition 高時提高密度
- 同步抽出兩顆 fisheye full-resolution frame
- fisheye 圓外 valid mask
- 可選 Torchvision Mask R-CNN：排除 person / bicycle / car / motorcycle / bus / truck
- 線性規模 temporal + cross-lens pair list
- COLMAP `OPENCV_FISHEYE`
- dual-camera rig bootstrap -> infer rig -> second-pass fixed-rig reconstruction
- 保存 telemetry-parser 的原始 telemetry / normalized IMU
- CameraAdapter contract，未來可接 Insta360 / GoPro MAX

## 重要設計

**Canonical source 是兩顆原始 fisheye，不是 equirectangular，也不是 cubemap。**

COLMAP 官方原生支援 `OPENCV_FISHEYE` 等 fisheye camera models，也原生支援 multi-camera rigs。對只接受 pinhole 的 trainer，後續應該從每顆 fisheye 個別產生 perspective views，不要先跨兩顆鏡頭做 panorama seam blending。

近期直接 fisheye 3DGS 研究也指出，先 undistort 會有拉伸 / interpolation 與邊緣資訊損失；native fisheye projection 可避免其中一部分問題。因此資料鏈必須保留原始 lens frames，不能在 ingestion 階段不可逆地 stitch 掉。

## Quick start

Windows：

```powershell
cd omni360-3dgs-skill
.\scripts\install.ps1
py .\scripts\doctor.py

py .\scripts\prepare_capture.py `
  "D:\capture\CAM_0001.OSV" `
  -o "D:\3dgs\scene" `
  --config .\config.example.yaml
```

先不跑 SfM / AI mask：

```powershell
py .\scripts\prepare_capture.py `
  "D:\capture\CAM_0001.OSV" `
  -o "D:\3dgs\scene-test" `
  --config .\config.example.yaml `
  --no-sfm --no-mask
```

只看 keyframe 結果：

```powershell
py .\scripts\prepare_capture.py `
  "D:\capture\CAM_0001.OSV" `
  -o "D:\3dgs\scene-analysis" `
  --config .\config.example.yaml `
  --analysis-only
```

## Output

```text
scene/
  images/
    lens0/*.png
    lens1/*.png
  masks/                 # LichtFeld / training masks: same relative filename
    lens0/*.png
    lens1/*.png
  masks_colmap/          # COLMAP feature masks: image filename + .png
    lens0/*.png.png
    lens1/*.png.png
  metadata/
    capture.json
    keyframes.csv
    telemetry.json       # telemetry-parser available時
    normalized_imu.json
    pairs.txt
    pipeline_report.json
  database.db
  rig_config.json
  sparse/
    0/...
```

## IMU 現況

Osmo 360 的 DJI metadata (`dvtm_oq101`) 有 fused attitude、accelerometer、IMU sampling rate 等資料。第一版 pipeline 會保留下來。

下一層整合是把經過**座標系驗證**的 per-lens gravity 寫進 COLMAP `PosePrior.gravity`，再用 Global Mapper 的 gravity-aware rotation averaging。這比直接把 DJI quaternion 當 COLMAP qvec 安全。

完整 quaternion orientation constraint 仍需自訂 VIO / BA backend；不會假裝 stock COLMAP 已經支援。

## Smoke test

已用對話中的 DJI Osmo 360 樣本實跑：

- 22 source frames
- 2 raw fisheye streams
- adaptive selector 選出 3 個 timestamp
- 兩顆 lens 都成功同步輸出 3 張 full-resolution fisheye
- 成功產生 static fisheye masks 與 13 組 constrained pairs

此環境沒有 COLMAP binary / telemetry-parser Python package，因此沒有在這裡完成 SfM stage；pipeline 對這兩者會在 `doctor.py` 清楚標示。

## References

- COLMAP camera models: https://colmap.github.io/cameras.html
- COLMAP rig support: https://colmap.github.io/rigs.html
- COLMAP FAQ / masks / gravity: https://colmap.github.io/faq.html
- PyCOLMAP PosePrior: https://colmap.github.io/pycolmap/pycolmap.html
- telemetry-parser: https://github.com/AdrianEddy/telemetry-parser
- LichtFeld Studio: https://github.com/MrNeRF/LichtFeld-Studio
- DirectFisheye-GS: https://arxiv.org/abs/2604.00648
