# GS360 Studio

GS360 Studio 是一個本機執行的 Tauri 桌面工具，將 DJI Osmo 360 `.OSV` 或已處理一部分的資料夾，整理成可續作的雙魚眼 COLMAP 專案。

介面以任務為中心。Extract、Mask、Align 可以分開執行、取消與重試；同一個任務也能放入多個在同一空間拍攝的原始檔。

## 輸出格式

未指定輸出位置時，專案會建立在第一個來源旁邊：

```text
colmap-{filename}/
├── project.json                 # 可續作 manifest 與 stage checkpoints
├── images/
│   ├── lens0/
│   └── lens1/
├── masks/                       # 白色保留、黑色排除
├── masks_colmap/                # COLMAP 所需的 image-name.ext.png 命名
├── rig_config.json
├── database.db                  # Align 完成後
├── sparse/                      # COLMAP sparse reconstruction
├── capture/                     # 可續作的雙魚眼候選影格
└── metadata/
    ├── capture.json
    ├── pairs.txt
    ├── source*_selection.json
    ├── source*_streams.json
    ├── source*_telemetry.json   # 可解析時輸出融合姿態與 IMU 摘要
    └── source*_telemetry.bin    # OSV 中存在 data stream 時
```

## 三個階段

### Extract

- 透過系統 `ffmpeg` / `ffprobe` 找出前兩路 video stream，保留 native fisheye，不先轉 equirectangular。
- 依 `baseFps` 產生輸出；啟用「跳過模糊影格」時，以 `denseFps` 解碼高品質 JPEG 候選影格，避免 10-bit 來源被 FFmpeg 展開成極大的 48-bit PNG。
- Gaussian pre-blur、Laplacian variance 與 Tenengrad 只評估 fisheye 有效圓。
- 每個時間區間以 `min(lens0, lens1)` 選同一組影格，避免左右鏡頭不同步或單側模糊。
- 候選 checkpoint、partial file 與 selection metadata 讓中斷後可以驗證並續作。
- OSV data stream 以 FFmpeg stream copy 原樣保存；支援的 DJI metadata 另輸出標準化摘要與融合姿態。尚未驗證 sensor-to-camera 座標轉換的 quaternion 不會直接套用到 COLMAP。

### Mask

- 使用 ONNX Runtime 執行 YOLO11 segmentation 與可選的 skyseg。
- 支援 person、bicycle、car、motorcycle、bus、truck，以及天空遮罩。
- 物件與 fisheye 圓外為黑色，其餘區域為白色。
- 每張 mask 先寫 partial file，再原子替換；尺寸與解碼驗證通過才會在續作時略過。
- 未選任何物件且未遮天空時，不需要模型，只產生 fisheye 有效區 mask。

模型會依序從任務指定資料夾、`GS360_MODEL_DIR`、Tauri 應用程式資料目錄的 `models/`、工作目錄的 `models/` 與 `.models/` 尋找。缺少必要模型時會在首次執行 Mask 時自動下載至應用程式資料目錄；YOLO 一定按需下載，SkySeg 只在啟用天空遮罩時下載。下載會先寫入暫存檔，且須通過固定大小與 SHA-256 驗證後才會啟用。

- YOLO11 segmentation：沿用 `gs360masker` 已驗證的 `yolo11s-seg.onnx`，原始模型由 Ultralytics 提供。
- SkySeg：固定至 Hugging Face `JianyuanWang/skyseg` 的指定 revision。
- 若環境無法連線，可手動放置模型；支援的檔名與目錄可參考 `src-tauri/src/masking/models.rs`。

Ultralytics 模型權重預設採 AGPL-3.0，另有 Enterprise License；執行時下載不會免除其授權義務。專案公開原始碼前仍應選定相容的專案授權並保留模型來源與授權聲明。SkySeg 模型頁標示為 MIT。詳見 [Ultralytics License](https://www.ultralytics.com/license) 與 [JianyuanWang/skyseg](https://huggingface.co/JianyuanWang/skyseg)。

### Align

- 使用 `OPENCV_FISHEYE`，每個 lens 一台 camera。
- 同時間的 lens0 / lens1 使用相同檔名，並建立受限的跨鏡與時間鄰近 pairs。
- 未知 rig extrinsics 採兩階段流程：先以獨立相機 bootstrap，再用 `rig_configurator` 推算 rig，最後固定 sensor-from-rig 重新 mapper。
- Mask stage 已完成時才傳入 COLMAP mask path；沒有 mask 也能獨立 Align。
- 已驗證存在的 sparse model 會直接續用，不重算完成結果。

## 執行需求

- Node.js 與 `pnpm`
- Rust stable toolchain
- 系統 `ffmpeg`、`ffprobe`（必須在 `PATH`）
- Align 需要系統 `colmap`（必須在 `PATH`）
- 首次執行物件／天空 Mask 時需要網路下載相應 ONNX 模型，或預先放入支援的模型目錄

預設 build 在 Apple silicon 優先使用 CoreML、Windows 優先使用 DirectML，失敗時會退回 CPU。Cargo 另提供 `cuda`、`xnnpack` 等 opt-in features；發佈版本仍必須搭配相容的 ONNX Runtime provider 套件實機驗證。COLMAP GPU 是否可用取決於使用者安裝的 COLMAP build，後端在 doctor 未偵測到 CUDA 時會強制使用 CPU。

## 開發

```bash
pnpm install
pnpm dev
pnpm build
pnpm tauri dev
```

Rust 驗證：

```bash
cd src-tauri
cargo check
cargo test --lib
```

## 目前界線

- `denseFps` 目前是模糊過濾的候選密度，不宣稱已實作基於 motion / IMU 的 adaptive cadence。
- Telemetry 先無損保存；在沒有相機座標系、時間同步與尺度驗證前，不會把 raw IMU 當成 COLMAP pose prior。
- 專案不會自動下載大型模型或第三方執行檔，避免隱性網路存取；模型資料夾可在新增任務時指定。
