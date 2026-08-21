# SphereAlign 開發文件

本文件收錄 SphereAlign 的開發環境、建置方式、處理管線、輸出結構與目前技術界線。產品定位與功能特色請回到 [README](../README.md)。

> [!WARNING]
> 專案仍在開發中，目前以 DJI Osmo 360 與 Insta360 作為正式支援與驗證範圍。程式雖能辨識部分其他影片容器，但不代表這些來源具有相同的雙影像串流、校正資訊或相容性保證。

## 技術棧

- Tauri 2 desktop shell
- React 19、TypeScript、Vite 7
- Rust stable
- FFmpeg / ffprobe
- ONNX Runtime、YOLO11 segmentation、SkySeg
- COLMAP 4.1.1+
- pnpm 11（專案的 `packageManager` 欄位固定為 `pnpm@11.21.0`）

## 開發環境需求

- Node.js 與 pnpm
- Rust stable toolchain
- 系統 `ffmpeg`、`ffprobe`，且必須可從 `PATH` 執行
- COLMAP 4.1.1 或更新版本，可從 `PATH` 偵測或在應用程式中指定
- Windows 使用官方免安裝版 COLMAP 時，應選擇根目錄的 `COLMAP.bat`，讓啟動腳本同時設定必要的 DLL 與 Qt plugin 路徑
- Linux 建置 Tauri 時需要 WebKitGTK 與相關原生套件；CI 目前安裝 `libwebkit2gtk-4.1-dev`、`libappindicator3-dev`、`librsvg2-dev`、`patchelf` 與 `xdg-utils`

## 常用指令

安裝相依套件：

```bash
pnpm install
```

僅啟動瀏覽器介面預覽：

```bash
pnpm dev
```

瀏覽器預覽可以檢視介面，但不能讀取本機專案或執行 Rust 後端管線。要執行完整桌面應用程式：

```bash
pnpm tauri dev
```

檢查前端建置：

```bash
pnpm build
```

Rust 驗證：

```bash
cd src-tauri
cargo check
cargo test --lib
```

版本同步檢查：

```bash
pnpm version:check
```

## 應用程式結構

前端以任務為中心，透過 Tauri commands 呼叫原生後端：

- `doctor`：檢查 FFmpeg、COLMAP、硬體能力與執行環境
- `inspect_paths`：檢查來源檔案或資料夾
- `create_project`：建立專案 manifest 與輸出目錄
- `load_project`：開啟既有或可復原的專案
- `start_stage`：執行 Extract、Mask 或 Align
- `cancel_job`：要求目前階段安全中止

後端同一時間只執行一個 stage job。前端可以依序自動執行 Extract → Mask → Align，也可以讓使用者單獨取消、繼續、重試或重跑各階段。

## 輸入契約

正式支援的來源是 DJI Osmo 360 `.OSV` 與 Insta360 `.INSV`。來源會先經由相機 adapter 正規化成兩顆實體鏡頭：單檔雙 track 的 INSV 取前兩路 video stream；較舊的 Insta360 雙檔素材則配對檔名中的 `_00_` 與 `_10_`，各自取唯一的 video stream。Extract 輸出分別作為 `lens0` 與 `lens1`，兩側鏡頭必須保持同名、同數量的同步影格。

檔案選擇器與來源檢查器也能辨識 `.mp4`、`.mov`、`.mkv`、`.avi`、`.webm`、`.m4v`、`.mts`、`.m2ts` 與 `.ts`。這些格式只代表容器可被檢查，仍不視為正式支援的相機來源；未提供相機 adapter 的一般容器必須自行確保雙鏡頭串流、同步與校正語意。

選擇資料夾時只掃描第一層檔案，避免意外把巢狀 proxy 或先前輸出重新加入來源。

## 處理階段

### Extract

- 透過相機 adapter 與系統 `ffmpeg` / `ffprobe` 找出兩顆實體鏡頭的 video stream，保留 native fisheye，不先轉為 equirectangular。
- 第一遍把雙鏡候選縮成 512 px 灰階影格，透過 stdout 串流進 Rust 記憶體，不編碼或保存候選圖片。
- 一般擷取預設使用 3 FPS。啟用清晰度過濾時，候選密度是基準的 2–10 倍。
- 清晰度評分結合 Gaussian pre-blur、Laplacian variance 與 Tenengrad，並只在魚眼有效圓內計算。
- 每個時間區間以兩側鏡頭評分的較低值挑選同一組影格，避免左右影像不同步或只有單側清晰。
- 第二遍只重新解碼入選時間點，並以原始解析度寫入 `images/lens0/` 與 `images/lens1/`。
- 最終檔名固定為 `sourceNNN_########.jpg`，兩側鏡頭使用完全相同的檔名。
- FFmpeg 會先嘗試自動硬體解碼；失敗時清理未完成輸出，再回退 CPU 軟體解碼。
- 候選 checkpoint 只保存分數與選擇結果，不保存候選影像。最終影格使用 partial file、雙鏡配對回滾與原子 metadata commit。
- 原始來源 data stream 以 stream copy 保存；來源 adapter 提供的 metadata 另輸出標準化摘要與融合姿態。DJI metadata 目前可產生較完整的 telemetry，Insta360 若素材未提供可驗證的相機校正或 IMU 欄位，會保留明確的 unavailable/unknown 狀態。

### Mask

- 使用 ONNX Runtime 執行 YOLO11 segmentation 與可選的 SkySeg。
- YOLO 可遮除 `person`、`bicycle`、`car`、`motorcycle`、`bus`、`truck`；天空遮罩可以獨立啟用。
- 物件與天空皆關閉時，pipeline 會直接略過 Mask stage，不產生遮罩輸出。
- 物件、天空、魚眼圓外，以及來源 adapter 提供且已驗證的固定光學遮擋區為黑色；其餘區域為白色。
- 推論工作解析度的最長邊限制為 640，再以 nearest-neighbor 放回來源尺寸；不要宣稱模型直接以原始 8K 解析度推論。
- 原生鏡頭超過 180° 的重疊區仍保留。optical mask 只表示無法成像或固定遮擋，不把邊緣畫質下降當成硬裁切。
- 每張 mask 先寫 partial file 再原子替換；既有 canonical mask 能解碼、且尺寸與來源一致時才會略過。
- Mask 只有一份 canonical 輸出：`masks/<relative-stem>.png`。保留 `images/` 下的相對目錄，並將來源副檔名替換成 `.png`（例如 `images/lens0/frame.jpg` → `masks/lens0/frame.png`）。
- Mask 固定為與來源同尺寸的 8-bit 單通道 L8 PNG；黑色（0）排除，白色（255）保留。這一份檔案同時供 COLMAP feature extraction 與 3DGS training 使用，只保留 canonical 檔案，不產生第二份相容檔或雙副檔名檔案。

模型尋找順序：

1. 任務指定的模型資料夾
2. `GS360_MODEL_DIR`
3. Tauri 應用程式資料目錄下的 `models/`
4. 工作目錄的 `models/`
5. 工作目錄的 `.models/`

缺少模型時，YOLO 會在首次執行 Mask 時下載；SkySeg 只在啟用天空遮罩時下載。下載先寫入暫存檔，通過固定大小與 SHA-256 驗證後才原子啟用。應用程式不會自動安裝 FFmpeg 或 COLMAP。

預設 ONNX Runtime provider：

- macOS：Core ML（目前 CI 與主要驗證目標為 Apple silicon）
- Windows：DirectML
- 其他 provider 由 Cargo features 選擇性啟用，包括 `cuda`、`xnnpack`、`tensorrt`、`nvrtx` 與 `webgpu`

Mask 明確停用 ONNX Runtime CPU execution-provider fallback。模型若無法完整交給選定的硬體 provider，stage 會回報失敗，不會靜默改用 CPU。

### Align

- 最低支援 COLMAP 4.1.1，不提供舊版參數相容層。
- 使用 SIFT 與 `OPENCV_FISHEYE`，每個 lens 對應一台 camera。
- 無 EXIF 焦距時，以 `default_focal_length_factor=0.3` 初始化。
- 每張影像最多 8192 個 features；每個 image pair 最多 8192 個 matches。
- 同時間的 `lens0` / `lens1` 使用相同檔名，並建立受限的跨鏡與時間鄰近 pairs；實體 rig 外參校正完成後，才加入固定、線性規模的 skip links 跨越短暫模糊或低紋理區段，避免未知外參 bootstrap 被長距配對改變，同時防止 final mapper 的局部註冊失敗永久切斷後續影格。
- 預設 rig 不假設兩顆實體鏡頭共心或精確相差 180°；兩鏡外參先由無 rig constraint 的 bootstrap reconstruction 估計。
- 有完整外參時，先執行 `rig_configurator` 再執行一次 mapper。
- config 缺少外參時，先以獨立相機 bootstrap；COLMAP 可建立多個子模型，pipeline 會先偏好至少 3 組共同註冊同名雙鏡影格的候選，再以共同影格數與已註冊影像數選擇可靠的 rig calibration seed。若自動候選皆不合格，最多再用 4 組已通過 two-view 幾何與 100-inlier 門檻的跨鏡 pair 作為初始 pair 重試。當 bootstrap 確實碎裂且主模型完整 rig coverage 低於 90%，程式只追加一次 continuation mapper：沿用已驗證模型、固定所有既有 frame pose、固定 rig 外參並嘗試註冊剩餘影格；候選必須提升 coverage／points，且通過 component、track、reprojection 與 rig quality gate 才以交易方式取代 seed，否則保留原模型。這避免從零第二次 mapper 的時間與軌跡漂移。
- 自訂 `rig_config.json` 會保留；只有缺少時才建立未知外參預設，或在內容精確等於舊版產生的共心 180° 預設時遷移。
- Mask stage 只有在每張來源影像都有可讀、同尺寸 PNG mask 時才完成並傳入 COLMAP mask path；沒有啟用 mask 仍能單獨執行 Align。
- 啟用 GPU 時，feature extraction、matching 與 Ceres bundle adjustment 都會使用對應 GPU 設定。
- GPU stage 失敗時會依階段清理或還原未完整的資料庫／sparse output，再以 CPU 重試。
- 目前固定使用 Ceres backend；Caspar 與本流程的 `OPENCV_FISHEYE` 相機模型不相容，因此不啟用。

## 專案輸出

未指定輸出位置時，專案會建立在第一個來源旁邊：

```text
colmap-{filename}/
├── project.json
├── capture/
│   └── sourceNNN/
│       └── selection.checkpoint.json
├── images/
│   ├── lens0/
│   └── lens1/
├── masks/
├── rig_config.json
├── database.db
├── sparse/
└── metadata/
    ├── capture.json
    ├── align.checkpoint.json
    ├── pairs.txt
    ├── sourceNNN_selection.json
    ├── sourceNNN_streams.json
    ├── sourceNNN_telemetry.json
    └── sourceNNN_stream{stream}_telemetry.bin
```

部分檔案只會在對應 stage 執行或來源包含相應資料時出現。`project.json` 會保存來源的絕對路徑；metadata 也可能包含相機 telemetry。分享專案資料夾前應先檢查與遮蔽本機路徑或拍攝中繼資料。

## Checkpoint 與可續作語意

- Extract 的候選選擇 checkpoint 會納入來源 identity、擷取設定與候選格式。
- Align checkpoint 的完整 fingerprint 會納入 settings、COLMAP version、pipeline revision、`rig_config.json`、pairs、images 與 masks。
- feature fingerprint 只涵蓋可安全沿用的影像／遮罩、COLMAP 版本與固定 SIFT／camera-model 語意。
- fingerprint 改變時可以保留仍相容的 features，但會清除 matches、two-view geometry、bootstrap 與 sparse 結果。
- 只有 checkpoint 完成、feature database 完整、SQLite header 有效、sparse model 內容齊全，而且目前 COLMAP 能重新轉換並通過 rig 驗證時，才沿用完整 Align 結果。

## Telemetry 原則

- 各來源 data stream 的原始副本是 source of truth。
- 可解析時另輸出 normalized IMU 與 fused attitude 摘要。
- 在 sensor-to-camera 座標轉換、時間同步與尺度尚未驗證前，quaternion 會標記為未套用，不會當作 COLMAP pose prior。
- `denseFps` 目前只控制清晰度候選密度，不宣稱已實作基於 motion 或 IMU 的 adaptive cadence。

## 模型授權

- YOLO11 segmentation 權重沿用 `gs360masker` 已驗證的 `yolo11s-seg.onnx`。Ultralytics 權重預設採 AGPL-3.0，另有 Enterprise License；執行時下載不會免除授權義務。
- SkySeg 固定至 Hugging Face `JianyuanWang/skyseg` 的指定 revision，模型頁標示為 MIT。

SphereAlign 自行開發的原始碼採用 `AGPL-3.0-only` 授權。第三方程式、函式庫、模型與其他資產仍適用各自的原始授權；完整說明與來源請見根目錄的 `THIRD_PARTY_LICENSES.md`。

## 目前界線

- 正式支援範圍是 DJI Osmo 360 `.OSV` 與 Insta360 `.INSV`（包含單檔雙 track，以及可依 `_00_`／`_10_` 配對的雙檔素材）；其他相機、鏡頭配置與一般雙串流影片尚未驗證。
- 不提供 equirectangular 預覽器或拼接輸出；核心輸出是原生雙魚眼 COLMAP 專案。
- 不會自動安裝 FFmpeg、ffprobe 或 COLMAP。
- 首次使用 Mask 時可能需要網路下載所需模型；素材本身不會為此上傳。
- macOS／Windows 的預設 Mask provider 不等於所有模型都能完整硬體執行，失敗時會明確回報。
- CI 能完成建置不代表所有平台、架構與硬體加速路徑都已完成實機驗證。
- checkpoint 只保證對目前輸入、設定與工具版本的一致性，不保證沿用任意舊版 sparse 結果。
