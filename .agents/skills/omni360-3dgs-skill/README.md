# Omni360 -> 3DGS Skill (v0.1)

把 360 相機原始素材整理成可直接進 COLMAP / LichtFeld Studio / 3DGS 的資料鏈。

## 目前實作

- DJI Osmo 360 / Osmo 360 II `.OSV` 自動辨識
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
- 保存 telemetry-parser 的原始 telemetry / normalized IMU；Osmo 360 II 的 OQ102 fused attitude 由 DJI envelope fallback 解碼
- 已驗證的 Osmo 360 II clip factory `OPENCV_FISHEYE` intrinsics（`fx=fy`、中心、`k1..k4`）可供 COLMAP；全零 optical-occlusion 會回退魚眼圓形遮罩
- II D-Log M 可辨識但沒有獨立驗證的官方 II LUT，auto 保留原生像素；`cam_extri_q` 不宣稱為 factory rig extrinsics
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
  masks/                 # one canonical mask tree: same relative stem, .png
    lens0/*.png
    lens1/*.png
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

Mask contract: preserve the image's relative path under `images/`, replace
its suffix with `.png`, and write an 8-bit single-channel L8 PNG. For example,
`images/lens0/frame.jpg` maps to `masks/lens0/frame.png`. Pixel value `0`
means exclude and `255` means keep. The same canonical file is used by
COLMAP feature extraction and downstream 3DGS training; only this canonical
file is generated, with no second compatibility tree or `image.ext.png`
double-extension file.

## IMU 現況

第一代 Osmo 360 的 DJI metadata (`dvtm_oq101`) 有 fused attitude、accelerometer、IMU sampling rate 等資料；Osmo 360 II sample 則使用 OQ102 DJI envelope fallback。兩者的 pipeline 都會保留原始時間軸與 normalized telemetry。已通過檢查的 II clip lens metadata 會另外提供 `OPENCV_FISHEYE` factory profile；不完整的 `cam_extri_q` 不會被升格為 rig 外參。

下一層整合是把經過**座標系驗證**的 per-lens gravity 寫進 COLMAP `PosePrior.gravity`，再用 Global Mapper 的 gravity-aware rotation averaging。這比直接把 DJI quaternion 當 COLMAP qvec 安全。

完整 quaternion orientation constraint 仍需自訂 VIO / BA backend；不會假裝 stock COLMAP 已經支援。

## Smoke test

已用 DJI Osmo 360 II `.OSV` 樣本實跑並驗證：

- 2 路 3840×3840 原生 HEVC fisheye streams
- OQ102 fused attitude 可解析並通過時間軸／單調性檢查
- clip-level factory `OPENCV_FISHEYE` profile 可供 COLMAP 使用
- optical-occlusion 全零時正確回退到魚眼圓形遮罩
- D-Log M 可辨識且 auto 不套用未驗證的 II LUT

完整 SfM 品質仍需依實際場景與硬體另行 benchmark；上述項目是 metadata、抽幀與校正契約的實機驗證，不代表所有素材都能得到相同的重建品質。

## References

- COLMAP camera models: https://colmap.github.io/cameras.html
- COLMAP rig support: https://colmap.github.io/rigs.html
- COLMAP FAQ / masks / gravity: https://colmap.github.io/faq.html
- PyCOLMAP PosePrior: https://colmap.github.io/pycolmap/pycolmap.html
- telemetry-parser: https://github.com/AdrianEddy/telemetry-parser
- LichtFeld Studio: https://github.com/MrNeRF/LichtFeld-Studio
- DirectFisheye-GS: https://arxiv.org/abs/2604.00648
