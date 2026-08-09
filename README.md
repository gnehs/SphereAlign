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
├── capture/                     # 輕量評分 checkpoint（不含候選圖片）
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
- 第一遍由 FFmpeg 將雙鏡候選縮成 512 px 灰階 rawvideo，直接透過 stdout 串流到 Rust 記憶體；不編碼、不保存候選圖片。一般擷取使用 `baseFps`，啟用「跳過模糊影格」時使用 `denseFps`，並以 Gaussian pre-blur、Laplacian variance 與 Tenengrad 評估 fisheye 有效圓。
- 選定時間點後，第二遍只重新解碼入選影格並以原始解析度寫入 `images/`。兩遍 FFmpeg 都會先嘗試自動硬體解碼，失敗時清理部分輸出並安全回退 CPU 軟體解碼。
- 每個時間區間以 `min(lens0, lens1)` 選同一組影格，避免左右鏡頭不同步或單側模糊。
- 候選階段只保存分數與序號的輕量 JSON checkpoint，不保存圖片；最終影格另使用 partial file、雙鏡配對回滾與原子 selection metadata，避免取消或失敗時留下單側或未提交結果。
- OSV data stream 以 FFmpeg stream copy 原樣保存；支援的 DJI metadata 另輸出標準化摘要與融合姿態。尚未驗證 sensor-to-camera 座標轉換的 quaternion 不會直接套用到 COLMAP。

### Mask

- 使用 ONNX Runtime 執行 YOLO11 segmentation 與可選的 skyseg。
- macOS 使用 Core ML（GPU／Neural Engine），Windows 使用 DirectML；模型無法完整交給硬體 provider 時會直接失敗，不會回退 CPU 推論。
- 支援 person、bicycle、car、motorcycle、bus、truck，以及天空遮罩。
- 物件、fisheye 圓外，以及 DJI OSV metadata 標記的固定光學遮擋區為黑色；校正曲線會依輸出解析度縮放，不額外內縮可用圓。手與自拍棒不由光學遮罩排除，仍交給物件 mask 流程處理。
- 原生鏡頭超過 180° 的雙鏡頭重疊區仍保留；目前的 optical mask 只表示「無法成像／固定遮擋」，不把邊緣畫質下降當成硬裁切。若要仿 DJI Studio 只取每顆鏡頭約 180° 的高品質區，應在鏡頭模型轉換後另產生 reconstruction-quality mask。
- 每張 mask 先寫 partial file，再原子替換。重跑 Mask 時，只有一般 mask 與 COLMAP mask 都能解碼且尺寸與來源影像一致才會自動略過；缺少、損壞或尺寸不符的輸出會重新產生。
- 未選任何物件且未遮天空時，不需要模型，只產生 fisheye 有效區 mask；OSV 缺少或無法解析校正資訊時會安全退回完整 fisheye 圓。

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
- Align 需要 COLMAP；可從系統 `PATH` 自動偵測，或在設定中指定啟動程式。Windows 官方免安裝版應選根目錄的 `COLMAP.bat`，讓它一併設定必要的 DLL 與 Qt plugin 路徑
- 首次執行物件／天空 Mask 時需要網路下載相應 ONNX 模型，或預先放入支援的模型目錄

預設 build 在 Apple silicon 使用 CoreML、Windows 使用 DirectML，並停用 ONNX Runtime 的 CPU execution-provider fallback。若模型含硬體 provider 不支援的節點，Mask 會回報錯誤而不是靜默改用 CPU。Cargo 另提供 `cuda`、`webgpu` 等 opt-in features；原生 WebGPU 在 macOS 與目前 YOLO／ONNX Runtime 組合的實機測試仍不穩定，因此不列為預設。COLMAP GPU 是否可用取決於使用者安裝的 COLMAP build，後端在 doctor 未偵測到 CUDA 時會強制使用 CPU。

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
