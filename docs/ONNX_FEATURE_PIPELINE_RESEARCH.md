# ALIKED / LightGlue / LoMa ONNX 特徵管線研究

測試日期：2026-08-19 至 2026-08-20

## 結論摘要

- 現有 COLMAP 4.1.1 自訂版已內建 `ALIKED_N16ROT`、`ALIKED_N32` 與 `ALIKED_LIGHTGLUE`，不需更換 COLMAP；缺少的官方 ONNX 權重已下載並以 SHA-256 固定版本。
- 「ONNX 可在任何 GPU 執行」不完全正確。ONNX 是模型格式，實際硬體支援取決於 ONNX Runtime Execution Provider 以及模型內每個算子的相容性。現有 COLMAP 套件帶的是 CUDA provider，因此目前正式路線仍是 NVIDIA GPU。
- Windows DirectML 理論上涵蓋 NVIDIA、AMD、Intel DX12 GPU，但本機以 ONNX Runtime 1.24.4 實測 LoMa-B128 時，模型初始化／動態 `Reshape` 執行失敗；ALIKED DirectML session 可建立，但首次推論程序異常結束。因此不能把 DirectML 宣稱為已可用的跨廠牌方案。
- LoMa 的官方／作者結果很有吸引力，但 COLMAP 整合仍在未合併 PR，現階段適合研究分支，不適合直接放進正式流程。

## 測試環境

- CPU：Intel Core i5-13400F
- GPU：NVIDIA GeForce RTX 4070 Ti 12 GB
- NVIDIA driver：610.62
- COLMAP：4.1.1 custom CUDA build，commit `a0d785f`
- 輸入保留原生雙魚眼，不先縫合成 equirectangular。
- 完整重建固定使用目前建議的 Variant B：自適應 keyframe、雙鏡頭 rig、相同 pair graph 與 incremental mapper；只替換 feature extractor／matcher。

## 模型版本

| 模型 | SHA-256 |
|---|---|
| `aliked-n16rot.onnx` | `39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547` |
| `aliked-n32.onnx` | `a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c` |
| `aliked-lightglue.onnx` | `b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d` |
| `loma_detector.onnx` | `b6af99c5e730034ac9b675d1ebe05d0679af4569a3c26f10a6a50f91e02dc512` |
| `loma_descriptor_dedode_b.onnx` | `82660a364299013618fe649092ebc4f617559f6a77e1ab5a3412be62a47ddc2d` |
| `loma_matcher_B128.onnx` | `e71ad490d13713374433a7ef99a7b4f4877d09338e40f347b7e64cc90150ee16` |

## 先行功能測試

同一個室內鏡頭、相隔 0.5 秒的兩張原始 3840×3840 魚眼影像：

| 管線 | 尺寸 | Keypoints | Raw matches | 幾何 inliers | Inlier ratio | 備註 |
|---|---:|---:|---:|---:|---:|---|
| ALIKED-N16Rot + LightGlue / CUDA | 3840 | 2047 / 2048 | 1,234 | 841 | 68.15% | COLMAP 原生流程成功 |
| LoMa-B128 / CPU | 最長邊 1024 | 2048 / 2048 | 1,403 | 722 | 51.46% | 每組影像 9.55 秒，不含 1.00 秒 session init |
| LoMa-B128 / DirectML | 1024 與原尺寸 | — | — | — | — | dynamic `Reshape`／graph fusion 失敗 |

這個 pair smoke test 只驗證可執行性，解析度與幾何驗證實作不同，不能直接用來宣稱 LoMa 或 ALIKED 的整體精度優劣。完整重建結果才是主要判斷依據。

## 完整重建比較

完整結果由隔離輸出目錄中的 `abc-summary.json`、`align_timings.json` 與 benchmark JSON 產生。精度採用不需要 ground truth 的代理指標：完整 rig 註冊率、3D points、median track length、最大連通分量與 median reprojection error。重投影誤差較低但 coverage 大幅下降時，不視為更精確。

<!-- FULL_BENCHMARK_RESULTS -->

| 場景 | 特徵管線 | 完整 rig 註冊率 | 3D points | Median track | Median reprojection | Extract | Align | 總時間 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0807 房間（室內） | SIFT 基準 | 433 / 457（94.75%） | 132,821 | 3 | **0.852 px** | 18.96 分 | **76.03 分** | **95.00 分** |
| 0807 房間（室內） | ALIKED-N16Rot + LightGlue | **456 / 457（99.78%）** | **146,249** | 3 | 1.323 px | 17.85 分 | 178.21 分 | 196.07 分 |
| TPE toilet（室內） | SIFT 基準 | 257 / 283（90.81%） | 37,918 | 3 | **0.683 px** | 11.62 分 | **9.80 分** | **21.42 分** |
| TPE toilet（室內） | ALIKED-N16Rot + LightGlue | **283 / 283（100%）** | **82,360** | 3 | 1.251 px | 11.72 分 | 50.62 分 | 62.35 分 |
| TPE toilet（室內） | ALIKED-N32 + LightGlue | **283 / 283（100%）** | 71,002 | 3 | 1.176 px | 11.57 分 | 40.41 分 | 51.99 分 |
| 東京車站（戶外） | SIFT 基準 | 495 / 973（50.87%） | 175,497 | 4 | **0.751 px** | 47.15 分 | **178.17 分** | **225.32 分** |
| 東京車站（戶外） | ALIKED-N16Rot + LightGlue | **973 / 973（100%）** | **190,054** | 4 | 0.994 px | 47.11 分 | 548.64 分 | 595.76 分 |

時間是同一台電腦的實測 wall time。`Extract` 是 OSV 解碼／keyframe 輸出，兩條路線近似；`Align` 才包含特徵、配對、mapper 與 BA。東京 ALIKED 的 Align 分解為 feature 59.15 分、matching 22.35 分、mapping/BA 465.62 分；真正最大的代價是更多影像成功註冊後，mapper 與全域 BA 的問題規模變大，而不只是 ONNX inference。

相對 SIFT：

- 房間：完整覆蓋 +5.03 個百分點、點數 +10.1%，Align 2.34 倍；median reprojection error 增加 55.3%。
- Toilet：N16Rot 完整覆蓋 +9.19 個百分點、點數 +117.2%，Align 5.17 倍；median reprojection error 增加 83.2%。N32 同樣 100% 覆蓋，較 N16Rot 快 20.2%、誤差較低，但點數少 13.8%。
- 東京：完整覆蓋 +49.13 個百分點（幾乎翻倍）、點數 +8.3%，Align 3.08 倍；median reprojection error 增加 32.4%。這是 ALIKED 最有價值的場景，也是時間代價最大的場景。

這裡沒有 survey／LiDAR／已知相機軌跡 ground truth，因此只能評估 SfM 內部一致性，不能報告公尺或公分級「絕對精度」。ALIKED 的 reprojection error 較高，但它同時納入 SIFT 完全無法註冊的困難影格；這個數字不代表其整體場景一定比較不準。下一級嚴格驗證應固定 control points 或以已知軌跡計算 ATE/RPE，再以相同相機集合比較重投影誤差。

## 公開研究結果與成熟度

- [COLMAP feature 文件](https://colmap.github.io/features.html) 將 SIFT 列為預設且最成熟的選擇；ALIKED 適合低紋理、低重疊或明暗變化較大的情況，LightGlue 通常能提高大視角／光照變化 pair 的 inlier ratio。
- [LightGlue 論文](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html) 的核心優勢是可依影像難度自適應減少計算；[官方實作](https://github.com/cvg/lightglue) 報告 RTX 3080 約 150 FPS（1024 keypoints）或 50 FPS（4096 keypoints）。這是 matcher 單項速度，不是完整 SfM 時間。
- [ALIKED 論文](https://arxiv.org/abs/2304.03608) 主打可微分 keypoint detection 與高效 learned local feature。
- [LoMa 論文](https://arxiv.org/abs/2604.04931) 報告相對 ALIKED + LightGlue，在 HardMatch、WxBS、InLoc、RUBIK、IMC22 都有明顯提升；但這些資料集分數不能直接換算成雙魚眼 3DGS 的公尺級精度。
- [COLMAP LoMa PR #4524](https://github.com/colmap/colmap/pull/4524) 在作者提供的 H200、1024×768、8,128 pairs 測試中，ALIKED-N16Rot + LightGlue 為 163.5 秒、97.2% verified pairs、平均 233.7 inliers；LoMa-B128 fp32 為 153.0 秒、99.95%、317.3 inliers，matcher bf16 後為 134.0 秒。PR 尚未合併，討論中也記錄了高解析度、resize、DINO extraction 成本與修正過的整合問題。
- [ONNX Runtime provider 文件](https://onnxruntime.ai/docs/execution-providers/) 說明硬體是由 Execution Provider 抽象；[DirectML provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) 可面向 DX12 GPU，但有算子與維護狀態限制；[CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 則依賴 NVIDIA CUDA/cuDNN。

## 實務建議

<!-- RECOMMENDATION -->

1. **保留 SIFT 當預設快速路線。** 兩個室內小場景已達 90.8–94.7% 完整覆蓋，SIFT 的時間與殘差都明顯較好；一般室內案沒有必要固定支付 2–5 倍 Align 成本。
2. **加入 ALIKED-N32 + LightGlue「高覆蓋／救援模式」。** 室內 N32 與 N16Rot 同為 100%，但 N32 更快且 reprojection error 較低。東京目前只完整測 N16Rot；在取得東京 N32 數據前，不應假定兩者排序完全相同。
3. **大型戶外或 SIFT 註冊率低於門檻時自動升級。** 東京 SIFT 只有 50.87%，ALIKED-N16Rot 達 100%，這是明確的品質勝利。建議先跑 SIFT；若完整 rig 覆蓋低於約 90%，再提示或自動重跑 ALIKED，避免所有場景一律耗時。
4. **下一步優先做 hybrid，而非直接全面換掉 SIFT。** 可保留 SIFT 已註冊的核心模型，只對未註冊區段、斷裂 component 與 bridge pairs 執行 ALIKED/LightGlue，再重新 triangulate/register。這有機會保留東京的覆蓋增益，同時避免 1,946 張 learned features 造成 7.8 小時 mapper/BA。
5. **LoMa 暫列研究選項。** CPU smoke test 可跑，但約 9.55 秒／pair；DirectML 在實機失敗，現有 COLMAP 又沒有 LoMa。應等 COLMAP PR 穩定，或另建隔離的 LoMa COLMAP 分支後，再以同一組完整場景做 A/B。
6. **產品文案不要寫「任何 GPU」。** 正確說法是「ONNX 可透過相容 Execution Provider 支援多種硬體」；目前已驗證的正式路線是 NVIDIA CUDA。AMD／Intel 需另做 WinML/DirectML provider、算子相容性與整場壓力測試。

CLI 已加入實驗參數：`--feature-pipeline sift|aliked-n16rot-lightglue|aliked-n32-lightglue` 與 `--model-dir <path>`。Feature fingerprint 會包含管線與模型 SHA-256，避免切換模型後誤用舊 database。正式 UI 尚未暴露這些開關。

## 封存與重現

- 機器可讀統計：[ONNX_FEATURE_PIPELINE_BENCHMARKS.csv](./ONNX_FEATURE_PIPELINE_BENCHMARKS.csv) 與 [ONNX_FEATURE_PIPELINE_BENCHMARKS.json](./ONNX_FEATURE_PIPELINE_BENCHMARKS.json)。
- 原始 OSV、COLMAP 安裝和正式專案資料不屬於研究暫存，未納入清理。
- 2026-08-20 已清除隔離的完整重建輸出、ONNX 虛擬環境、下載模型、第三方 repository 副本、pair smoke-test 中間檔及一次性研究 runner，約 32.4 GB。模型檔名與 SHA-256 已保存在本文件及 JSON，必要時可重新下載並核驗。

重跑完整場景時，使用 release CLI 的 `abc` command、Variant B，並維持原生雙魚眼：

```powershell
src-tauri\target\release\spherealign-cli.exe abc `
  --input D:\3dgs\scene.osv `
  --output-root D:\temporary-benchmark `
  --colmap D:\COLMAP-4.1.1-windows-2022-CUDA-cuDSS-Caspar-GUI\COLMAP.bat `
  --gpu-index 0 --variants B `
  --feature-pipeline aliked-n32-lightglue `
  --model-dir D:\onnx-models
```
