<div align="center">
  <img src="src-tauri/icons/icon.png" width="96" alt="GS360 Studio 標誌">
  <h1>GS360 Studio</h1>
  <p>把 DJI Osmo 360 原始素材，整理成可續作的雙魚眼 COLMAP 專案。</p>
</div>

> [!WARNING]
> **目前仍在開發中。** 功能、輸出格式與操作流程仍可能調整；現階段只支援 **DJI Osmo 360**，其他相機與影片來源尚未驗證，也不在支援範圍內。

![GS360 Studio 將 Osmo 360 素材依序完成雙魚眼影格擷取、清晰影格挑選、動態物件與天空遮罩、相機對齊及稀疏點雲重建](assets/readme/workflow-hero.png)

## 從原始素材到可續作的重建專案

GS360 Studio 把原本需要在多個工具間往返、手動整理檔案與反覆確認狀態的工作，收進同一個本機桌面流程。加入一個或多個在相同場景拍攝的 Osmo 360 素材後，即可依序完成影格擷取、遮罩與相機對齊。

| 影格擷取 | 場景遮罩 | 相機對齊 |
| --- | --- | --- |
| 從兩側魚眼影像同步挑選影格，保留原生魚眼畫面，並優先留下較清晰的配對。 | 自動辨識人、腳踏車與常見車輛，也能選擇遮除天空及鏡頭無效區域。 | 以雙鏡頭 rig 建立受限配對，交由 COLMAP 產生相機姿態與稀疏重建結果。 |

## 特色

- **一個任務處理多段素材**：同一空間拍攝的多個 Osmo 360 檔案可整理進同一個重建專案。
- **保留原生雙魚眼資料**：直接處理正反兩側魚眼影像，不先轉換為等距柱狀投影，避免多一次不必要的重採樣。
- **IMU＋畫面變化挑選影格**：以 fused attitude 相對角、低解析 visual novelty 與最大時間間隔挑選 keyframe，再只解碼入選的完整解析度雙鏡配對。
- **感應器輔助可接近兩倍加速**：在同一組實測素材中，IMU＋畫面變化抽幀讓 Align 從約 52.2 分鐘降至 28.0 分鐘，約快 1.86 倍（節省 46.3%）；實際效果會依素材的運動、紋理與來源是否存在視覺重疊而變動，這不是固定保證值。
- **減少動態干擾**：可遮除人、腳踏車、汽車、機車、公車、卡車與天空，讓重建更聚焦於穩定場景。
- **針對 360 rig 對齊**：來源內使用 IMU-aware temporal graph，跨來源使用 bounded 視覺 retrieval；校正完成後可依魚眼 FOV overlap 再縮減配對。
- **安全使用拍攝中繼資料**：先以視覺模型估時間偏移與 rotational hand-eye；通過 residual、coverage、rig 與 focal gate 後才寫入每鏡頭 gravity prior。完整 DJI quaternion 永遠不會直接冒充 COLMAP qvec。
- **自動扶正訓練座標**：Align 完成後以校正過的 per-image gravity 旋轉整個 sparse model，使 LichtFeld 的 `+Y` 固定朝上；不估地面高度、不額外指定 yaw，也不依賴 Global Mapper 候選成功。
- **Incremental／Global 自動切換**：首次可用 incremental 建立校正種子；COLMAP 4.1.1 前提通過後，以候選 global model 驗證成功才取代原結果。
- **本機優先**：影片、影格、遮罩與重建結果都在本機處理；首次使用遮罩功能時可能需要下載對應模型。
- **環境能力檢查**：集中顯示 FFmpeg、COLMAP、硬體加速與儲存空間等執行條件。

## 可以中止，也可以接著做

![GS360 Studio 的影格擷取、遮罩與對齊階段可獨立取消、重試，並從本機檢查點繼續](assets/readme/resumable-workflow.png)

影格擷取、遮罩與對齊都能分開執行、取消、重試或重跑。已完成的產物會保留在專案中；再次開啟任務時，GS360 Studio 會檢查既有結果，盡可能從可安全沿用的進度繼續，而不是每次全部重來。

IMU／global mapper 的校正 gate、產物與 benchmark 方法請參閱 [IMU 重建流程](docs/IMU_RECONSTRUCTION.md)。開發環境、建置方式、架構、完整輸出結構與實作界線請參閱 [開發文件](docs/DEVELOPMENT.md)。

上述效能數字來自 [2026-08-13 A/B/C benchmark](docs/evidence/2026-08-13-two-osv-abc/README.md)，比較的是相同輸入下的 Align 階段；兩個來源若不是同一場景或沒有視覺重疊，註冊率不應直接用來判斷 IMU 抽幀的加速效果。
