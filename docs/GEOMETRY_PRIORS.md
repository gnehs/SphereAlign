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
  normals/<原影像子目錄>/<原檔名改為 .jpg>
  normals/.spherealign.json
  geometry/spirula-normals.json
  geometry/runs/<run-id>/run.json
  geometry/runs/<run-id>/frames/<image-id>/...
```

法線先量化為 RGB8 `round((n * 0.5 + 0.5) * 255)`，source-camera XYZ，不額外翻轉符號，再以 **JPEG 品質 90、4:4:4** 儲存，維持原始尺寸。這是使用者接受弱法線先驗的壓縮誤差後採用的有損儲存設定；不再與先前 PNG 訓練實驗逐像素相同。無效像素在編碼前為黑色，JPEG 可能改動邊界的黑色值；精確的診斷有效性仍保留在 `validity.png`，不是額外傳給 trainer 的 validity 通道。模型、投影與法線 supervision weight 均未變更。

子目錄保留雙鏡頭身份。匯出檢查全部已註冊影像、RGB／mask／COLMAP identity、輸出 SHA-256、RGB8 格式、原始尺寸及副檔名轉換後的檔名碰撞。舊 PNG 紀錄仍可讀取與預覽；新的法線執行會將有效快取轉為 JPG。部分完成的 run 不能匯出。

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
- 舊 PNG 快取在 writer lock 下直接轉為 JPG，不載入模型或重新推論。先提交 JPG 和 `frame.json`，保存舊 PNG 的 SHA-256 後才刪除該快取 PNG；取消／中斷可續接。來源或輸出不符的快取不能直接轉換，既有 JPG 不重複壓縮。run ID 仍描述幾何與來源，儲存轉換不改 ID，各檔案 hash 與 normal encoding 記錄會更新。
- 完成全部推論後先核對並匯出訓練法線，然後自動清除 `native.f32` 和 `reasons.u8`。匯出失敗、推論失敗或在完成前取消，不會啟動新的原始浮點資料清理。
- 清理前驗證保留圖檔，先將 `cleanupPending` 寫入逐影像紀錄，再刪除兩個固定名稱的中間檔；完成後記為 `removed`。中斷可安全續接，保留原始 hash 及移除 byte 數。法線、RGB、masks、sparse、訓練輸出均保留。
- 已清理快取可直接續用，不需要模型或 GPU；切回保留原始浮點資料則重新推論已清理影像。預覽預檢顯示保留輸出與處理峰值估算，完整清理仍發生在整批完成之後，須預留整批中間檔空間及訓練法線的副本空間。壓縮比例因影像而異，預檢仍採保守容量估算。
- 匯出使用完整 staging 目錄。相同輸出直接續用；不同的既有 SphereAlign 法線會移至 `geometry/exports/previous/` 後再發布。既有未受管理的 normals 只有在全部檔名與內容完全相符時才接管，否則保留並報錯。
- 已匯出的 PNG 在轉為 JPG 時同樣保留上述舊匯出備份；壓縮不自動刪除歷史匯出。因此單張 JPG 的縮小比例不等於整個既有專案的空間釋放比例。
- 從 app 重跑 Extract／Mask／Align 前，舊的受管理 `normals/` 會移至上述備份位置，法線階段回到待執行，避免訓練載入舊來源先驗。外部修改來源後，請重新執行 normals 或使用上述訓練啟動器，它會重新檢查來源。

既有完整草稿也能只清理、不載入模型：

```powershell
src-tauri/target/debug/spherealign-geometry-cli.exe cleanup "D:/data/project" --run RUN_ID
```

## 幾何與品質範圍

原生投影支援 PINHOLE／OPENCV_FISHEYE，包括可逆的後半球區域。六個重疊 100° 面在已知焦距下恢復 additive Z shift，再將法線以 SO(3) 旋回來源相機；不在深度邊界內插。重疊法線差 >30° 或相對 range 差 >20% 時保守排除。魚眼有效區域是鏡頭範圍與畸變多項式中央可逆區間的交集。

保留的 `native.f32` 為 little-endian `[height,width,4]`：`nx,ny,nz,ray_range`，無效全零。`reasons.u8`：0 有效支援、1 鏡頭外、2 來源遮罩、3 模型／恢复無效、4 面間不一致。range 尚未對齊 COLMAP 尺度，也不是已驗證公尺值。range 偽彩每張各自縮放，不能跨圖比較顏色。

本次正式採用的依據是指定室內場景的完整訓練和使用者實際場景檢視，並非宣告所有場景或所有 trainer 的逐像素 validity／gradient 契約都已認證。既有 stock sampler 邊界缺陷及尚未完成的跨影像 screening 仍記錄在 [GEOMETRY_COMPATIBILITY.md](GEOMETRY_COMPATIBILITY.md)。新的場景訓練後仍需檢視品質；法線有效支援比例不等於品質通過率。

## 法線處理效能

原始解析度的回填使用每個工作專屬的 CPU 執行緒池（最多 8 條），相機射線則依模型、尺寸及完整內參快取精確的 `f64` 結果。同一標定的後續影像可直接重用；不同 mask 仍逐張套用。射線快取採 LRU、上限 768 MiB，3840×3840 每組約 337.5 MiB，雙鏡頭約 675 MiB。這是額外的 CPU RAM；超出快取預算或配置失敗時，改為平行即時計算射線。工作結束即釋放快取。

回填以 64 列為一個區塊並行計算，再依原始像素順序寫出浮點資料與原因標籤。模型、解析度、有效性／重疊門檻、法線量化及中間檔清理政策不變，也不改變 run ID。已完成快取可直接續用。

每張新生成影像的 `frame.json` 增加 `timings`：`inferenceMs`、`perspectiveMs`、`rayCacheMs`、`gatherMs`、`outputMs`、`totalMs`，並記錄快取命中、目前快取 bytes 及 worker 數。`gatherMs` 包含原始浮點／標籤寫入與同步；`outputMs` 包含 JPG／PNG 編碼及輸出雜湊；總時間另含影像解碼、幾何恢復等工作，因此不等於這些分項的總和。舊紀錄沒有 timings 時以預設值讀取；快取轉換不重算原始推論計時。

改用 JPG 之前的 3840×3840 release CPU 回填與 PNG 輸出測試：舊版 7.670 秒，新版首次 1.535 秒、快取命中 0.652 秒（約 5.0×／11.8×）。七個輸出檔案 SHA-256 與保留的舊版實作相同，證明回填數學不變；本次 JPEG 編碼是後續的獨立有損步驟。此測試使用合成模型輸出與真實魚眼標定，不包含 GPU 推論、透視擷取或整批匯出，不能解讀為整個法線階段加速倍率。

三張實際場景的 3840×3840 法線圖以 JPG 品質 90 儲存後縮小 62.5–69.6%，每張編碼約 218–228 ms；每 16 像素取一個非黑色樣本，與既有 PNG 比較的法線角度差中位數約 0.44–0.45°，P95 約 1.79–2.19°。實際 Spirula binary 以只有 JPG 法線的獨立解析 fixture 完成三步 smoke；未重跑完整場景的 JPG／PNG 品質 A/B。詳見 [實測紀錄](GEOMETRY_BENCHMARKS.md)。

## 驗證

```powershell
cargo test --manifest-path src-tauri/Cargo.toml --lib --offline -- --test-threads=2
pnpm build
```

CPU 測試涵蓋完整訓練匯出、取消／續跑、快取損壞、來源改動、不同鏡頭同名檔、路徑限制、舊法線保留、上游失效、清理失敗恢復及舊專案遷移。模型／硬體測試需要顯式環境設定。實測資料另見 [GEOMETRY_BENCHMARKS.md](GEOMETRY_BENCHMARKS.md) 與 [GEOMETRY_MODEL_CONTRACT.md](GEOMETRY_MODEL_CONTRACT.md)。
