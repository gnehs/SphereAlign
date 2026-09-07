# 法線：正式前處理流程

2026-09-07。正式流程為 **Extract → Mask → Align → Normals**。使用者完成 `disney_cruise_room` 的 20,000 步法線訓練並實際檢視場景後，確認漂浮物大幅減少，要求整合正式流程。此次啟用的是法線先驗；相對距離仍只供預覽，不輸出訓練用 `depths/`。

## 操作

- 新專案自動在對齊後處理所有已註冊影像，不受進階預覽的影像上限影響。
- 舊專案保留既有 Extract／Mask／Align 完成狀態，新增待執行的 Normals 階段；按繼續即可只執行法線。不會因開啟專案而下載模型或開始推論。
- 主流程的法線階段有進度、取消、錯誤與續跑。專案詳情的「法線貼圖」可檢查原圖、法線、有效性、原因圖及六個透視視角。進階的「預覽樣本」仍是獨立草稿，不匯出訓練法線。
- 處理設定可指定模型路徑及「保留中間檔案供除錯」。預設使用 pinned MoGe-2 ViT-B、Windows DirectML、每個透視面 768×768／1,800 tokens、有效性門檻 0.5。第一次需要時下載模型；沒有 CPU fallback。

## 輸出與訓練

正式階段使用最終 COLMAP 相機與 pose，生成原始解析度的相機座標法線：

```text
<dataset>/
  normals/<原影像子目錄>/<原檔名改為 .png>
  normals/.spherealign.json
  geometry/spirula-normals.json
  geometry/runs/<run-id>/run.json
  geometry/runs/<run-id>/frames/<image-id>/...
```

訓練 PNG 與已驗證實驗完全相同：RGB8 `round((n * 0.5 + 0.5) * 255)`，source-camera XYZ，不額外翻轉符號；無效像素為黑色。子目錄保留雙鏡頭身份。匯出檢查全部已註冊影像、RGB／mask／COLMAP identity、輸出 SHA-256、RGB8 格式、原始尺寸及副檔名轉換後的檔名碰撞。部分完成的 run 不能匯出。

`geometry/spirula-normals.json` 是本專案訓練啟動器使用的設定。它指定 `load_normals=true`、`normal_supervision_weight=0.01`、`load_depths=false`、`warp_to_pinhole=false`，使用 360-camera preset、20,000 步、SH1、3M 上限及 CPU 影像快取。原生魚眼法線不經另一套透視重投影。啟動器將 JSON 欄位轉為 Spirula CLI 參數，並在啟動前用正式法線流程重新核對來源與快取。

```powershell
cargo build --manifest-path src-tauri/Cargo.toml --bins

# 相同的正式生成／續跑／匯出／清理程序
src-tauri/target/debug/spherealign-geometry-cli.exe normals "D:/data/project"

# 使用自己的 Spirula executable；執行前核對／補齊法線
./scripts/run/spirula_train_normals.ps1 -DatasetRoot "D:/data/project" -Spirula "C:/tools/spirula.exe"
```

App 的第四階段完成到資料集輸出，不會自行啟動外部 Spirula。需要保留原始浮點資料時，可對 normals 命令加上 `--keep-intermediates`。`--model PATH` 支援指定模型。一般 `spherealign-cli geometry normals ...` 也使用相同實作。

## 續跑、更新與清理

- run ID 包含引擎、runtime、provider、模型 SHA、最終 COLMAP identity、有效性門檻、影像／mask SHA 及選取數量。保留中間檔政策不改變 run ID。
- 同一 app 共用 JobManager，資料集另有跨程序 writer lock。逐影像先寫 `.partial`，所有檔案及雜湊完整後才提交。取消後續用已提交影像。
- 完成全部推論後先核對並匯出訓練 PNG，然後自動清除 `native.f32` 和 `reasons.u8`。匯出失敗、推論失敗或在完成前取消，不會啟動新的清理。
- 清理前驗證保留 PNG，先將 `cleanupPending` 寫入逐影像紀錄，再刪除兩個固定名稱的中間檔；完成後記為 `removed`。中斷可安全續接，保留原始 hash 及移除 byte 數。PNG、RGB、masks、sparse、訓練輸出均保留。
- 已清理快取可直接續用，不需要模型或 GPU；切回保留原始浮點資料則重新推論已清理影像。預覽預檢顯示保留輸出與處理峰值估算，完整清理仍發生在整批完成之後，須預留整批中間檔空間及訓練 PNG 的副本空間。
- 匯出使用完整 staging 目錄。相同輸出直接續用；不同的既有 SphereAlign 法線會移至 `geometry/exports/previous/` 後再發布。既有未受管理的 normals 只有在全部檔名與內容完全相符時才接管，否則保留並報錯。
- 從 app 重跑 Extract／Mask／Align 前，舊的受管理 `normals/` 會移至上述備份位置，法線階段回到待執行，避免訓練載入舊來源先驗。外部修改來源後，請重新執行 normals 或使用上述訓練啟動器，它會重新檢查來源。

既有完整草稿也能只清理、不載入模型：

```powershell
src-tauri/target/debug/spherealign-geometry-cli.exe cleanup "D:/data/project" --run RUN_ID
```

## 幾何與品質範圍

原生投影支援 PINHOLE／OPENCV_FISHEYE，包括可逆的後半球區域。六個重疊 100° 面在已知焦距下恢復 additive Z shift，再將法線以 SO(3) 旋回來源相機；不在深度邊界內插。重疊法線差 >30° 或相對 range 差 >20% 時保守排除。魚眼有效區域是鏡頭範圍與畸變多項式中央可逆區間的交集。

保留的 `native.f32` 為 little-endian `[height,width,4]`：`nx,ny,nz,ray_range`，無效全零。`reasons.u8`：0 有效支援、1 鏡頭外、2 來源遮罩、3 模型／恢复無效、4 面間不一致。range 尚未對齊 COLMAP 尺度，也不是已驗證公尺值。range 偽彩每張各自縮放，不能跨圖比較顏色。

本次正式採用的依據是指定室內場景的完整訓練和使用者實際場景檢視，並非宣告所有場景或所有 trainer 的逐像素 validity／gradient 契約都已認證。既有 stock sampler 邊界缺陷及尚未完成的跨影像 screening 仍記錄在 [GEOMETRY_COMPATIBILITY.md](GEOMETRY_COMPATIBILITY.md)。新的場景訓練後仍需檢視品質；法線有效支援比例不等於品質通過率。

## 驗證

```powershell
cargo test --manifest-path src-tauri/Cargo.toml --lib --offline -- --test-threads=2
pnpm build
```

CPU 測試涵蓋完整訓練匯出、取消／續跑、快取損壞、來源改動、不同鏡頭同名檔、路徑限制、舊法線保留、上游失效、清理失敗恢復及舊專案遷移。模型／硬體測試需要顯式環境設定。實測資料另見 [GEOMETRY_BENCHMARKS.md](GEOMETRY_BENCHMARKS.md) 與 [GEOMETRY_MODEL_CONTRACT.md](GEOMETRY_MODEL_CONTRACT.md)。
