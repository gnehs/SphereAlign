# COLMAP A/B/C schema v2 single-pass 實測

素材：`CAM_20260503151923_0031_D.OSV`（約 116.5 秒、2.11 GB）

COLMAP：4.1.1 CUDA/cuDSS Caspar GUI build；GPU：RTX 4070 Ti；總 wall time：約 55.4 分鐘。
三組 COLMAP profile：`baseline`

輸入 SHA-256：`35ef6ccb0813dbbb0858fa51650da4f8e2c274935599fee52046a0b20475e092`

本次使用全新 output root，摘要 schema v2，主 coverage 指標為 `completeRegisteredRigFrameCount`。未知 rig 外參只執行一次 mapper，再由 `rig_configurator` 將已驗證 bootstrap 模型轉成 rig model；不再清空成果重跑第二次 mapper。

## Single-pass A/B/C

| Variant | 主要差異 | Selected rigs | Complete rigs | Any-sensor rigs | Pairs | 3D points | Median track | Median reprojection | Components | Largest component | Extract | Align |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 無 pruning、無 retrieval | 350 | 159/350 (45.4%) | 214 | 7,128 | 31,152 | 3 | 0.726 px | 2 | 323 images | 399.6 s | 1,138.2 s |
| B | pruning + retrieval | 257 | 114/257 (44.4%) | 155 | 5,281 | 20,129 | 3 | 0.747 px | 1 | 269 images | 396.1 s | 521.5 s |
| C | B + telemetry/focal/FOV/mapper auto | 257 | 70/257 (27.2%) | 113 | 5,281 | 15,347 | 3 | 0.682 px | 1 | 183 images | 401.1 s | 464.6 s |

## 現行判讀

- **B 是目前整體最佳預設**：complete coverage 只比 A 少 1.1 個百分點，但所有 269 張已註冊影像位於同一 component，Align 比 A 快 54%。
- A 的絕對 complete rigs、points 與最大 component 較高，但模型分成 2 個 components，且保留更多近似影格造成 mapper 明顯變慢。
- C 的 reprojection error 最低，但 coverage 明顯退化。此次 focal round-trip 驗證失敗；fresh project 在 mapper 前尚無完整 rig 外參，因此 auto 退回 incremental mapper，gravity prior 未套用。低 reprojection 不能抵銷 coverage 損失。
- 修正前的 schema v2 雙 mapper run，A/B/C 都只剩 3 個 complete rigs；當時第一輪其實已有 185/65/99 個共同雙鏡 frames，證明第二次 mapper 覆蓋了可用 bootstrap 模型。

## 第一版 CLI 歷史數據（不可作 release evidence）

第一版 CLI 沒有拒絕非空 output root、摘要缺少獨立 provenance，並把 `registeredRigFrameCount` 標成完整 rigs、從初始化 capability 推導 IMU。以下只保留作歷史參考。

### 歷史 A/B/C

| Variant | 主要差異 | Selected rigs | Pairs | 舊版 reported rigs | 3D points | Median track | Median reprojection | Components | Largest component | Extract | Align |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 無 pruning、無 retrieval、incremental | 350 | 9,156 | 3/350 (0.9%) | 202 | 2 | 0.690 px | 1 | 6 images | 391.0 s | 1,015.5 s |
| B | pruning + retrieval、incremental | 257 | 6,751 | 170/257 (66.1%) | 20,542 | 3 | 0.749 px | 6 | 297 images | 372.2 s | 1,011.4 s |
| C | B + telemetry/focal/FOV/mapper auto | 257 | 6,751 | 171/257 (66.5%) | 19,920 | 3 | 0.770 px | 8 | 313 images | 366.2 s | 1,030.7 s |

### 歷史判讀

- B 明確勝 A：註冊 rigs 約增加 57 倍，3D points 約增加 102 倍，median track 由 2 提升至 3；總 stage 時間還略短。A 較低的 reprojection error 來自只剩 3 個成功 rigs，不能視為較高品質。
- B 較 C 穩健：C 只多 1 個 registered rig、最大 component 多 16 images，但 points 少 3.0%、reprojection error 高 2.8%、components 由 6 增至 8，Align 多 19.2 秒。
- C 的 focal round-trip 驗證失敗後已安全還原；但舊版沒有可靠記錄最終 effective mapper／gravity prior 是否真正套用，因此不能由 `imuApplied=false` 下結論。
- 舊數據傾向 B；現已由上方 schema v2 fresh-output single-pass 結果取代。

## 受控 COLMAP profile smoke test（歷史）

下列測試固定同一段 15 秒素材、21 rig frames、42 images、193 pairs，只切換 COLMAP profile：

| Profile | Registered rigs | 3D points | Median track | Median reprojection | Align |
|---|---:|---:|---:|---:|---:|
| baseline | 9/21 (42.9%) | 398 | 3 | 0.474 px | 137.7 s |
| tuned v1（含 3.5 px mapper gates） | 2/21 (9.5%) | 31 | 2 | 0.299 px | 194.1 s |
| tuned v2（只增加 features/matches） | 2/21 (9.5%) | 119 | 2 | 227.024 px | 167.0 s |

兩個 tuned 版本均有反效果，已從產品預設撤回。`tuned` 只保留為 CLI 顯式實驗 override。

> 以上是 sparse reconstruction 的實測。最終 3DGS 視覺品質仍需使用相同 training/render 設定比較 floaters、接縫、牆面變形與 coverage holes。
