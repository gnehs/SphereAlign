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
    ├── align.checkpoint.json    # Align 輸入 fingerprint checkpoint
    ├── pairs.txt
    ├── source*_selection.json
    ├── source*_streams.json
    ├── source*_telemetry.json   # 可解析時輸出融合姿態與 IMU 摘要
    └── source*_telemetry.bin    # OSV 中存在 data stream 時
```

## 三個階段

### Extract

- 透過系統 `ffmpeg` / `ffprobe` 找出前兩路 video stream，保留 native fisheye，不先轉 equirectangular。
- 第一遍由 FFmpeg 將雙鏡候選縮成 512 px 灰階 rawvideo，直接透過 stdout 串流到 Rust 記憶體；不編碼、不保存候選圖片。一般擷取預設使用 3 FPS 的 `baseFps`；啟用「清晰度過濾」時，`denseFps` 可設定為截取影格率的 2–10 倍，並以 Gaussian pre-blur、Laplacian variance 與 Tenengrad 評估 fisheye 有效圓。
- 選定時間點後，第二遍只重新解碼入選影格並以原始解析度寫入 `images/`。兩遍 FFmpeg 都會先嘗試自動硬體解碼，失敗時清理部分輸出並安全回退 CPU 軟體解碼。
- 每個時間區間以 `min(lens0, lens1)` 選同一組影格，避免左右鏡頭不同步或單側模糊。
- 候選階段只保存分數與序號的輕量 JSON checkpoint，不保存圖片；最終影格另使用 partial file、雙鏡配對回滾與原子 selection metadata，避免取消或失敗時留下單側或未提交結果。
- OSV data stream 以 FFmpeg stream copy 原樣保存；支援的 DJI metadata 另輸出標準化摘要與融合姿態。尚未驗證 sensor-to-camera 座標轉換的 quaternion 不會直接套用到 COLMAP。

### Mask

- 使用 ONNX Runtime 執行 YOLO11 segmentation 與可選的 skyseg。
- macOS 使用 Core ML（GPU／Neural Engine），Windows 使用 DirectML；模型無法完整交給硬體 provider 時會直接失敗，不會回退 CPU 推論。
- YOLO 物件遮罩與 SkySeg 天空遮罩可獨立啟用；兩者皆關閉時會略過 Mask 階段並直接進入 Align。YOLO 支援 person、bicycle、car、motorcycle、bus、truck。
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

- 使用 SIFT 與 `OPENCV_FISHEYE`，每個 lens 一台 camera；無 EXIF 焦距時以適合 180–190° 等距魚眼的 `default_focal_length_factor=0.3` 初始化，而非 COLMAP 的一般鏡頭預設值 1.2。
- 同時間的 lens0 / lens1 使用相同檔名，並建立受限的跨鏡與時間鄰近 pairs。
- 預設把原生 `lens0`／`lens1` 視為共心、背對背且上下方向一致的 360 rig；`lens1` 相對 `lens0` 固定為繞相機 Y 軸 180°（WXYZ quaternion `[0, 0, 1, 0]`），並在 matching 前套用 `rig_configurator`，之後只跑一次 mapper，不再先做一遍無用的 bootstrap。舊版自動產生、未含外參的預設 config 會升級，其他自訂 config 保持不變。
- 自訂 config 若省略 rig extrinsics，仍採兩階段流程：先以獨立相機 bootstrap，再用 `rig_configurator` 推算 rig，最後固定 sensor-from-rig 重新 mapper。
- Mask stage 已完成時才傳入 COLMAP mask path；沒有 mask 也能獨立 Align。
- 啟用 `align.useGpu` 時，GPU 開關涵蓋 SIFT feature extraction、matching，以及 incremental mapper 的 Ceres bundle adjustment。`align.gpuIndex` 預設為 `-1`；feature extraction 與 matching 可傳入逗號分隔的多 GPU（例如 `0,1`），Ceres BA 使用清單中的第一張 GPU。
- SIFT 明確限制為每張影像最多 8192 個 features，matcher 則限制每個 image pair 最多 8192 個 matches，避免為不存在的 descriptor 配置過大的 GPU matching workspace。最終 mapper 會裁減 global BA 的冗餘 3D points，並套用 COLMAP 影片預設的 1.4 frames／points growth ratio，以降低大型場景反覆 global BA 的頻率；未知外參的 bootstrap 仍使用 COLMAP 保守預設，優先確保註冊與 rig 校正覆蓋率。
- GPU 能力只依設定中選定的 COLMAP 執行檔之 version banner 與 help 判斷，不以 FFmpeg 的 CUDA hwaccel 或 `nvidia-smi` 推論 COLMAP CUDA 能力。GPU stage 失敗時會以 CPU 重試；feature extraction 的 GPU fallback 會刪除可能半提交的資料庫後從乾淨資料庫重跑，matching 先還原 GPU 執行前的資料庫／WAL 備份，mapper 則先清理不完整的 sparse 輸出。
- 目前固定使用 Ceres backend，不啟用 Caspar；Caspar 與本流程的 `OPENCV_FISHEYE` 相機模型不相容。
- 已存在的自訂 `rig_config.json` 會保留；只有缺少時才建立預設雙鏡頭 rig，或在內容恰好等於舊版無外參預設時升級為固定背對背外參。
- bootstrap 完成後會把 sparse model 轉成官方文字格式，確認每顆設定鏡頭都有註冊影像，且每個未知外參鏡頭都至少有一組與參考鏡頭同名且同時註冊的影格；`rig_configurator` 後再核對 rig／sensor 數量並驗證所有 non-reference sensor 的 `HAS_POSE=1`，不讓缺鏡頭或未知 `sensor_from_rig` 進入 final mapper。
- Align 會唯讀檢查 SQLite database 的 `images`、`keypoints` 與 `descriptors`；每張影像都有成對特徵資料（即使 `rows=0`）時標記為完整，明確略過 `feature_extractor`。部分完成時保留資料庫，讓 COLMAP 依每張影像的既有特徵自動跳過並補齊缺項；影像集合不符、schema 或 feature blob 損壞時會記錄 warning、刪除資料庫後重跑。
- `metadata/align.checkpoint.json` 的完整 fingerprint 會納入 settings、COLMAP version、pipeline revision、`rig_config.json`、pairs、images／masks 的路徑、大小與修改時間；另存只涵蓋影像／遮罩、COLMAP version 與固定 SIFT／`OPENCV_FISHEYE`／`0.3` 語意的 feature fingerprint。非 retry 且 feature fingerprint 相符時才保留 `database.db` 及其 WAL／journal；若完整 fingerprint 已變更，會保留 features 但清除 `matches`／`two_view_geometries`，再重建 sparse、bootstrap 與配對結果。套用 rig 前另以原子目錄備份資料庫：未知外參的 final mapper 中止時可還原獨立鏡頭狀態；若偵測到已配置 rig 卻沒有可用備份，會捨棄受污染的資料庫而非用錯誤狀態 bootstrap。舊 checkpoint 尚未記錄 feature fingerprint 時，只有完整 fingerprint 相符才會安全遷移並沿用一次；feature fingerprint 不符時則完整清理。
- 只有 checkpoint 已標記完成、feature database 完整、database 具有有效 SQLite header／page size、sparse model 同時包含非空的 rigs、frames、cameras、images 與 points3D，且選定的 COLMAP 能重新轉換並通過 rig 驗證時才會續用完整結果。

## 執行需求

- Node.js 與 `pnpm`
- Rust stable toolchain
- 系統 `ffmpeg`、`ffprobe`（必須在 `PATH`）
- Align 最低支援 [COLMAP 4.1.1+](https://colmap.github.io/changelog.html)；不提供舊版參數相容層。可從系統 `PATH` 自動偵測，或在設定中指定啟動程式。Windows 官方免安裝版應選根目錄的 `COLMAP.bat`，讓它一併設定必要的 DLL 與 Qt plugin 路徑
- 首次執行物件／天空 Mask 時需要網路下載相應 ONNX 模型，或預先放入支援的模型目錄

預設 build 在 Apple silicon 使用 CoreML、Windows 使用 DirectML，並停用 ONNX Runtime 的 CPU execution-provider fallback。若模型含硬體 provider 不支援的節點，Mask 會回報錯誤而不是靜默改用 CPU。Cargo 另提供 `cuda`、`webgpu` 等 opt-in features；原生 WebGPU 在 macOS 與目前 YOLO／ONNX Runtime 組合的實機測試仍不穩定，因此不列為預設。COLMAP GPU 能力只由設定中選定的 COLMAP version/help banner 判斷，不使用 FFmpeg 或 `nvidia-smi` 代判；未確認 CUDA/Ceres 能力時，對應 Align stage 會使用 CPU。即使已確認 GPU，feature、matching 或 mapper 的 GPU 執行失敗仍會自動以 CPU 重試，mapper retry 會先清理不完整輸出。

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

- `denseFps` 目前是清晰度過濾的候選密度，不宣稱已實作基於 motion / IMU 的 adaptive cadence。
- Telemetry 先無損保存；在沒有相機座標系、時間同步與尺度驗證前，不會把 raw IMU 當成 COLMAP pose prior。
- Caspar bundle-adjustment backend 目前不啟用，因為它與本流程使用的 `OPENCV_FISHEYE` 不相容；Align 以 Ceres BA 為界線。
- Align checkpoint 只驗證目前輸入與設定的 fingerprint；settings、COLMAP version、rig、pairs、images 或使用中的 masks 改變時會失效並重建，不保證沿用舊 sparse 結果。
- COLMAP GPU 僅在指定 build 的 banner/help 確認能力後嘗試；GPU stage 失敗會回退 CPU，mapper 回退前會清理不完整 sparse output。
- 專案不會自動下載大型模型或第三方執行檔，避免隱性網路存取；模型資料夾可在新增任務時指定。
