# Geometry：原生推論、草稿界面與品質驗證邊界

2026-09-07。已接上 Windows DirectML 原生推論、專案 Geometry 面板、取消／續跑及原始解析度預覽。依接續需求，先完成界面與推論，實際場景品質由使用者與開發者接著一起測試。**所有結果都是 `single_view_unverified` 草稿，不會啟用於訓練。**

原始候選 ONNX 在 Windows DirectML 的完整 GPU 分派檢查失敗，包含固定輸入尺寸的測試。進一步將 token budget 也變成常數後，找到獨立 hash 的靜態 graph 可行路徑；Rust 已能自行產生這個 artifact，且在一張合成圖上有 GPU parity 證據。但它仍是待認證的候選 profile。Spirula stock sampler 的嚴格 sentinel 語意不合格，修補版的完整 importer／loss／gradient 尚未驗收。因此不把這些結果包裝成可啟用的訓練整合。

## 已交付與未交付

| Phase | 實際狀態 |
| --- | --- |
| 0 | 模型 byte/hash、I/O、provider spike、Rust 靜態轉換、CPU golden、单張 GPU parity、Spirula help/check/短訓練、sampler 缺陷重現與 patch 已執行。normal sign、ray units 完整呼叫鏈、修補版逐像素 prior gradient、更多模型 fixtures 尚未認證。 |
| 1 | 獨立 optional Geometry job、共用 JobManager 排他鎖、跨程序 writer lock、設定／完整來源 hash 決定 run ID、逐影像原子提交與雜湊續跑。沿用舊三階段資料結構，未改成四階段排程。尚無分開的 inference／validation／export 三級 cache。 |
| 2 | Rust 讀取最終 COLMAP binary/text、PINHOLE／OPENCV_FISHEYE 含後半球投影、六個重疊 100° 視角、真實 MoGe GPU 推論、已知焦距 Z shift 恢復、法線旋回 source camera、原始解析度 ray range／有效性／原因圖與預覽。跨視角尺度尚未對齊，重疊不一致保守排除。 |
| 3 | **未實作** observed anchors、holdout、跨視角 screening、獨立 modality masks。 |
| 4 | **未實作** active profile 切換或正式 exporter。獨立 patch 未套到其他 repository／binary。 |
| 5 | **未執行** 使用者場景 A/B/C/D、完整 stage VRAM／disk 測量或品質提升驗收。 |

原始 dynamic graph 的 provider 停止條件曾觸發；現在只接入已能完整 GPU 執行的 pinned static profile，沒有 CPU fallback。後續訓練啟用仍須完成模型品質及 importer／loss 契約驗收，不能以草稿完成率或 sampler host 測試代替。

## 界面操作

1. 在已完成對齊的專案詳情中開啟 Geometry；預設關閉，不會加入 Extract → Mask → Align 自動排程。
2. 勾選「啟用 Geometry 草稿產生」，設定影像上限（預設 8；依檔名順序，0 為全部）。模型可選原始 pinned ONNX 或 Rust 靜態 ONNX；留空時，按產生才下載。
3. 「檢查輸入」確認最終相機、已註冊影像、原始／遮罩尺寸及預估輸出空間，再按「產生／續跑草稿」。
4. 執行中可取消，最晚等目前 ORT GPU 呼叫回傳後停止；已提交的完整影像可续用。下載、投影與 native gather 也會檢查取消。
5. 在草稿紀錄中選影像，檢查原始 RGB、相機座標法線、相對距離偽彩色、模型有效性、無效原因與六個透視 RGB 視角。「有效支援」不是品質通過率。

開關只控制明確執行；重新開專案仍預設關閉。上次執行設定保存在 run.json，開啟面板可讀回，無須遷移或重寫舊 project.json。品質 acceptance 一律顯示尚未測試，沒有訓練啟用按鈕。

## 原生 CLI

```powershell
cargo build --manifest-path src-tauri/Cargo.toml --bin spherealign-geometry-cli
src-tauri/target/debug/spherealign-geometry-cli.exe preflight "D:/data/project" --frames 8
src-tauri/target/debug/spherealign-geometry-cli.exe run "D:/data/project" --model ".work/geometry-spike/rust-static-768.onnx" --frames 8
src-tauri/target/debug/spherealign-geometry-cli.exe inspect "D:/data/project"
```

一般 `spherealign-cli geometry <preflight|run|inspect> <dataset>` 也呼叫相同實作，可使用 `--project <project-directory>`。CLI 是獨立程序；OS writer lock 防止同一資料集重複寫入，GUI 的共用 JobManager 防止 Geometry／Mask／Align 同時在該 app 執行。

## 草稿與續跑契約

輸出在 `<dataset>/geometry/runs/<hash>/`，不寫根目錄 `normals/`、`depths/`，不修改 images、masks 或 sparse。`run.json` 記錄 graph SHA、runtime、provider、完整最終模型 hash、設定、進度、座標／格式與 `activeForTraining=false`。

每張影像以 COLMAP image ID 存在 `frames/<id>/`，保留完整原始檔名及 camera ID；非連續 ID、双鏡頭相同 basename、UTF-8 路徑可用。`frame.json` 記錄 RGB／mask SHA 和所有輸出 SHA。續跑重新核對檔案內容，檔案存在不代表可沿用；變更 RGB、mask、相機／pose／points3D、runtime 或有效性門檻會形成新 run。新結果先寫 `.partial`，完整才 rename 發布。故障或取消的 partial 不會當作完成影像。關閉程序會釋放 OS lock，面板將殘留 running 標為 interrupted。

`native.f32` 是 little-endian float32 `[H,W,4]`，順序為 source-camera `(nx,ny,nz,ray_range)`，無效像素四值全零。range 為恢復後 face Z 除以 face unit ray Z，再乘模型 scale；**尚未對齊 COLMAP 尺度，也不是經驗證的公尺值**。法線採純 SO(3) 旋轉、不做逐像素翻面；模型 normal sign 尚待 trainer 認證。`reasons.u8` 每個 native 像素一 byte：0 有效支援，1 鏡頭外，2 原始 mask，3 模型／深度恢復無效，4 重疊角度 >30° 或相對 range 差 >20%。目前法線／range 共用草稿 validity，獨立 modality screening 留到品質階段。

PNG 預覽包括 rgb／normal／range／validity／reasons 及 face0–5。原生回填採最近有效面像素，不在深度邊界內插；重疊不同意直接排除。每張 range 圖使用自己的偽彩顯示尺度，不能跨圖比较顏色。一次保留一張來源及其六個面，順序推論，不將整個場景載入 RAM。相機最大 64 MP；預估磁碟空間是保守估算，尚未完成完整場景峰值資源基準。

## 執行已實作的原生工具

以下是保留的開發者 probe 路徑。一般 app 推論不需 Python，也不會因開專案而下載。

```powershell
cargo build --manifest-path src-tauri/Cargo.toml --bin spherealign-geometry-probe
python scripts/geometry_fetch_spike.py --model

# 原始候選：此 Windows 環境預期非零 exit code，原因見 JSON。
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx DirectML

# 完全原生 Rust graph 準備；output 必須是新檔案。
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx --prepare-static .work/geometry-spike/rust-static-768.onnx
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/rust-static-768.onnx DirectML --static-experiment
```

靜態 profile 只有 batch=1、RGB 768×768、1800 tokens；不提供任意 token／尺寸／ONNX 相容承諾。每次只建立一個 session，沿用 Mask 的 provider helper、停用 CPU fallback。原始模型、Python shape-inferred 實驗模型及 Rust 可重現模型有不同 ID/hash。準備程序保留未知 protobuf 欄位，拒絕截斷／overflow、非指定 hash 與既有輸出，驗證衍生 hash 後才在同一 filesystem 以 hard link 原子發布。此工具尚不提供 stage job 排他鎖或取消，因此不可當作正式 app stage 使用。

## Migration 與關閉狀態

Geometry 面板／job 獨立於既有 project/settings/stage 三階段欄位，關閉狀態不執行推論。既有 Extract → Mask → Align 與總進度維持原邏輯，舊 Align completed 不會因更新而變成 pending。

本次保留既有未提交修改，未 reset、commit、push 或建立 PR。資料集前處理、原始 RGB、RGB masks 與 COLMAP 模型均未由新工具修改。解析 fixture 的 31 份來源檔在實際 trainer smoke 後 hash 不變；這不等於完整既有 pipeline 的 off-mode 重跑驗收。

## 驗證與接續入口

無模型／GPU 的 Rust 測試已加入現有 build CI：

```powershell
cargo test --manifest-path src-tauri/Cargo.toml --lib geometry::
python scripts/geometry_spirula_sentinel.py
python scripts/geometry_make_fixture.py ".work/geometry-spike/new 中文 fixture"
```

後兩項為開發工具；sampler 測試需要 C++17 `g++` 與下載好的 pinned sources。它編譯原始函式與 proposed patch 的函式，不修改 upstream snapshot。

完整模型契約見 [GEOMETRY_MODEL_CONTRACT.md](GEOMETRY_MODEL_CONTRACT.md)，訓練限制與解除條件見 [GEOMETRY_COMPATIBILITY.md](GEOMETRY_COMPATIBILITY.md)，實測／命令與 skipped 見 [GEOMETRY_BENCHMARKS.md](GEOMETRY_BENCHMARKS.md)。
