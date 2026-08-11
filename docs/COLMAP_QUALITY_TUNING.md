# COLMAP 品質優先調校

目前 Align pipeline 針對 Osmo 360 雙魚眼素材預設採用實測較穩定的 COLMAP baseline。品質優先的實驗 profile 仍可由 CLI 顯式指定，但不再作為產品預設。

## 主要調整

- Baseline SIFT/matching：`8192` features、COLMAP 預設 peak threshold、`8192` matches 與預設 RANSAC 設定。
- 實驗 tuned SIFT/matching：`10240` features、`0.006` peak threshold、`10240` matches、`15000` RANSAC trials；受控 smoke test 已證明不適合目前素材，因此必須以 `--profile-override tuned` 顯式啟用。
- Incremental mapper：保留 COLMAP 預設 BA window、iterations 與 reprojection gates。15 秒受控測試顯示，3.5 px gate 會把註冊率由 baseline 的 9/21 降至 2/21，因此已撤回。
- Global mapper：global positioning 從 100 小幅提高至 120 iterations；track completion/merge 從預設 15 px 收緊至 8 px，normalized reprojection gate 收緊至 `0.008`。
- 不再對最終 incremental model 使用 `1.4` BA growth ratio 或 redundant landmark pruning。Bootstrap 仍保留較寬鬆的 COLMAP 預設，以免犧牲初始 rig calibration coverage。

## 建議應用程式設定

品質測試優先使用：

- Keyframe pruning：開啟
- min rotation：5°
- min gap：200 ms
- max gap：600 ms
- visual novelty：0.08
- Visual retrieval：開啟
- Mapper mode：`auto`
- Gravity prior、Auto calibrate telemetry、Focal calibration、Calibrated FOV pairs：素材有足夠多軸旋轉時開啟
- Fixed rotation BA：先關閉
- Rolling-shutter trajectory：開啟

這些參數會增加 Align 時間、database 大小、RAM 和 VRAM 使用量。實際品質仍應以相同 OSV 和相同 3DGS 設定比較 registered frames、3D points、median track length、median reprojection error、floaters、牆面變形和 coverage holes。

最初曾測試 16384 features、10-image local BA、100 次 global BA 與更嚴格的 3 px gate；在 15 秒 smoke capture 上，tuned final mapper 超過 10 分鐘仍未完成，而 baseline 整個 Align 約 101 秒。第二輪 12288 features、8-image local BA、70 次 global BA 仍在 final mapper 超過 5 分鐘，因此兩組激進設定均已回退，不應作為預設值。

第三輪受控比較固定相同的 21 個 rig frames、42 張影像與 193 組 pairs，只切換 COLMAP profile。使用 3.5 px incremental gates 的 tuned v1 僅註冊 2/21 rig frames、31 points、median track 2、Align 194.1 秒；baseline 註冊 9/21、398 points、median track 3、Align 137.7 秒。雖然 tuned v1 的 median reprojection error 較低（0.299 vs 0.474 px），但它只保留極少量容易點，屬於 coverage 大幅退化下的倖存者偏差，因此 tuned v1 已淘汰。

移除 3.5 px gates 後的 tuned v2 仍只註冊 2/21 rig frames、119 points，median reprojection error 惡化至 227.0 px，Align 167.0 秒。因此正式 A/B/C 與未指定 profile 的專案均使用 baseline；tuned 僅留作可重現歷史實驗。

## 原始 OSV 正式 A/B/C 結果

以 `CAM_20260503151923_0031_D.OSV`（約 116.5 秒）實跑，三組都固定使用 baseline profile：

- A（無 pruning/retrieval）：3/350 rigs、202 points、median track 2、0.690 px，Align 1015.5 秒。
- B（pruning + retrieval）：170/257 rigs、20542 points、median track 3、0.749 px，Align 1011.4 秒。
- C（B + telemetry/focal/FOV/auto）：171/257 rigs、19920 points、median track 3、0.770 px、8 components，Align 1030.7 秒；focal 驗證回退且 `imuApplied=false`。

因此目前推薦 B：它相較 A 大幅改善 coverage 與 track support；相較 C 則有更多 points、較低 reprojection error、較少 components 與較短 Align。完整報告位於 `docs/COLMAP_ABC_RESULTS.md`。
