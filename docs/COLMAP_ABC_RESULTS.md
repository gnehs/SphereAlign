# COLMAP A/B/C 歷史實測（待 schema v2 重跑確認）

素材：`CAM_20260503151923_0031_D.OSV`（約 116.5 秒、2.11 GB）

COLMAP：4.1.1 CUDA/cuDSS Caspar GUI build；GPU：RTX 4070 Ti；總 wall time：約 69.8 分鐘。
三組 COLMAP profile：`baseline`

> 這份結果由第一版 CLI 產生。該版本沒有拒絕非空 output root、摘要沒有獨立 provenance，並把 `registeredRigFrameCount` 標成完整 rigs、從初始化 capability 推導 IMU。數值保留作歷史參考，但不再作為 release acceptance evidence；須以修正版 CLI 的 fresh output、`completeRegisteredRigFrameCount`、`effectiveMapper` 與 gravity prior report 重跑確認。

## 歷史 A/B/C

| Variant | 主要差異 | Selected rigs | Pairs | 舊版 reported rigs | 3D points | Median track | Median reprojection | Components | Largest component | Extract | Align |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 無 pruning、無 retrieval、incremental | 350 | 9,156 | 3/350 (0.9%) | 202 | 2 | 0.690 px | 1 | 6 images | 391.0 s | 1,015.5 s |
| B | pruning + retrieval、incremental | 257 | 6,751 | 170/257 (66.1%) | 20,542 | 3 | 0.749 px | 6 | 297 images | 372.2 s | 1,011.4 s |
| C | B + telemetry/focal/FOV/mapper auto | 257 | 6,751 | 171/257 (66.5%) | 19,920 | 3 | 0.770 px | 8 | 313 images | 366.2 s | 1,030.7 s |

## 判讀

- B 明確勝 A：註冊 rigs 約增加 57 倍，3D points 約增加 102 倍，median track 由 2 提升至 3；總 stage 時間還略短。A 較低的 reprojection error 來自只剩 3 個成功 rigs，不能視為較高品質。
- B 較 C 穩健：C 只多 1 個 registered rig、最大 component 多 16 images，但 points 少 3.0%、reprojection error 高 2.8%、components 由 6 增至 8，Align 多 19.2 秒。
- C 的 focal round-trip 驗證失敗後已安全還原；但舊版沒有可靠記錄最終 effective mapper／gravity prior 是否真正套用，因此不能由 `imuApplied=false` 下結論。
- 舊數據傾向 B，但在 schema v2 fresh-output 重跑前不下正式推薦結論。

## 受控 COLMAP profile smoke test

下列測試固定同一段 15 秒素材、21 rig frames、42 images、193 pairs，只切換 COLMAP profile：

| Profile | Registered rigs | 3D points | Median track | Median reprojection | Align |
|---|---:|---:|---:|---:|---:|
| baseline | 9/21 (42.9%) | 398 | 3 | 0.474 px | 137.7 s |
| tuned v1（含 3.5 px mapper gates） | 2/21 (9.5%) | 31 | 2 | 0.299 px | 194.1 s |
| tuned v2（只增加 features/matches） | 2/21 (9.5%) | 119 | 2 | 227.024 px | 167.0 s |

兩個 tuned 版本均有反效果，已從產品預設撤回。`tuned` 只保留為 CLI 顯式實驗 override。

> 以上是 sparse reconstruction 的實測。最終 3DGS 視覺品質仍需使用相同 training/render 設定比較 floaters、接縫、牆面變形與 coverage holes。
