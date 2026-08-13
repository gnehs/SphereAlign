# IMU 重建流程

GS360 Studio 不會把 DJI quaternion 直接寫成 COLMAP `qvec`。流程先使用不依賴座標校正的相對旋轉縮小問題，再用成功的視覺模型估時間偏移與 rotational hand-eye；只有通過驗證的結果才可建立 gravity prior 或進入 global mapper。

## 執行順序

1. Extract 以 FFmpeg 真實 PTS 對齊 fused attitude，使用相對角度、32×32 雙鏡 gradient novelty、最小／最大時間間隔挑選 keyframe。telemetry 缺失時保留 visual novelty 與 max-gap fallback。
2. Align 建立同時間 stereo、同鏡頭 `+1/+2`、必要 cross-lens temporal links。較長連結依時間／旋轉篩選；多來源使用 bounded visual retrieval，任何 descriptor 失敗都回退舊 anchor graph。
3. Matching 後以 COLMAP `view_graph_calibrator` 更新 focal。程式會從 database round-trip 驗證所有 perspective cameras 的 `prior_focal_length` 與參數，並拒絕未變動的 `0.3 × max(width,height)` 預設猜值。
4. 首次 `auto` 若沒有 calibration marker，先保留一個 incremental seed。程式會逐一評估各 bootstrap component，從每來源有效的 camera-from-world rotations 與 fused attitude 估 angular-speed time offset，再解 `A X = X B`；低樣本、低激發、單軸退化、低相關或高 residual 都會拒絕。每個候選與最後選擇都保存在稽核檔，不會混合不同 component 的座標系。
5. 通過的來源會輸出 calibrated rig orientation manifest。gravity 由世界 down 經各鏡頭 `camera_from_world` 轉換後，寫入 COLMAP 4.1.1 `pose_priors`；位置與 covariance 缺值使用 NaN，既有有效位置 prior 會保留。
6. 若 gravity coverage 至少 80%、rig 外參完整、focal prior 有效且 CLI options 可用，`auto` 會建立 global candidate。候選模型包含 rigs/frames、驗證成功，而且 complete-rig coverage、最大 component coverage、points/track support、reprojection error 與 component count 都通過相對 seed 的防退步閘才提交；失敗或退步時保留 seed。calibrated pair refresh 若無法完整 rollback database 與 `pairs.txt`，流程會 fail-closed，不會在不一致的 matching graph 上執行 global mapper。
7. 實驗性的 fixed-rotation BA 只在上述 gate 與對應 COLMAP option 都通過時啟用。完整 SO(3) constraint 只會交給完成 capability handshake 的外部 orientation-aware BA executable，stock COLMAP 不會收到 quaternion prior。

未知 rig 會先由 incremental bootstrap 估外參，再從 COLMAP database round-trip 回寫 `rig_config.json`，因此同一次執行即可繼續 calibration/global candidate。首次執行為了取得視覺 calibration seed，時間不會比直接 incremental 更短；後續有效 checkpoint 可直接使用 global path。

## 產品設定

一般 GS360 Studio 任務固定採用實測的 B 流程：keyframe pruning（5°／200 ms／600 ms／0.08）、多來源 visual retrieval 與 incremental mapper。產品介面只保留影格率、清晰度過濾、遮罩及 GPU 選項；不再顯示或保存下列實驗設定。

A／B／C benchmark CLI 仍可在獨立測試專案中明確傳入：

- `mapperMode`: `auto`、`incremental`、`global`。產品預設為 `incremental`；`global` 是 fail-closed，必須已有有效 marker。
- `useGravityPrior`: global rotation averaging 是否使用 gravity。
- `autoCalibrateTelemetry`: 是否從 incremental seed 自動估時間與座標轉換。
- `calibrateFocalPrior`: 是否執行 view graph focal calibration。
- `useVisualRetrieval`: 多來源是否使用低解析 visual retrieval。
- `useCalibratedFovPairs`: 有 calibration 時是否用 FOV overlap 篩 optional cross-lens pairs。
- `fixedRotationBa`: 略過 joint rotation optimization 的實驗模式。
- `exportRollingShutterTrajectory`: 輸出逐列 SLERP sidecar；`pixelsModified` 固定為 `false`，目前不修改影像。
- `orientationPriorExecutable`: 選用的外部完整 quaternion BA 工具。工具必須宣告 `gs360.orientation-ba/v1` capability；不可填 stock `colmap`。

## 校正與診斷產物

- `metadata/sourceNNN_telemetry.json`: 標準化 fused attitude 與 diagnostics。
- `metadata/sourceNNN_frame_motion.json`: 每個候選的 PTS、旋轉、novelty 與 selection reason。
- `metadata/cross_source_retrieval.json`: bounded retrieval 結果與 fallback 狀態。
- `metadata/imu_calibration.json`: 每來源 offset、convention、hand-eye quaternion、correlation、excitation、coverage 與 residual。
- `metadata/orientation_priors_sourceNNN.json`: calibrated `rig_from_world` orientation interchange；不是 stock COLMAP pose prior。
- `metadata/orientation_priors.json`: 單來源時是可供外部 BA 使用的 manifest；多來源時是 index，保留各來源不同 offset。
- `metadata/global_mapper_priors.json`: database injection、focal/gravity coverage、calibration version 與代表性 offset marker。
- `metadata/global_mapper_candidate.json`: requested/attempted 狀態、seed 與 candidate complete-rig 數，以及最後實際 mapper。
- `run-provenance.json`: run ID、輸入/COLMAP/CLI binary hash、Git commit、dirty 狀態與 align pipeline revision；路徑只保存 basename。
- `metadata/rolling_shutter_sourceNNN.json`: 選用的 calibrated row trajectory sidecar。
- `metadata/align_timings.json`: pair graph、feature extraction、matching、mapping 與總時間。
- `metadata/intra_source_loop_retrieval.json`: 單一錄影依時間分段後的 bounded visual retrieval 診斷；候選只補強長距離連通性，仍須通過 COLMAP 幾何驗證。
- `metadata/benchmark_*.json`: A／B／C metrics 與人工 3DGS quality checklist。

上述 calibration、orientation、prior marker、frame motion、pairs、telemetry hash 與設定都會影響 align fingerprint。若實際採用的 mapper 不符合 checkpoint 預期，`auto` 不會錯誤沿用結果。

## A／B／C benchmark

用同一批 OSV、同一版 COLMAP 與相同 3DGS 設定建立三個專案：

| Variant | Extract | Pair graph | Mapper |
| --- | --- | --- | --- |
| A `a_current` | 關閉 keyframe pruning | 關閉 visual retrieval／calibrated FOV | incremental |
| B `b_imu_pruning` | 開啟 IMU＋visual pruning | 開啟 visual retrieval | incremental |
| C `c_global` | 同 B | 同 B＋calibrated FOV | auto/global＋gravity |

Align 完成會自動輸出對應 report；也可呼叫 Tauri command `generate_benchmark_report` 重新收集。缺少輸入或 binary-only model 時 report 會標記 `partial`，不以 0 假裝測量值。除註冊率、3D points、track length、reprojection error 與 connected components 外，仍應填寫 report 的 3DGS checklist，檢查遠處細節、floaters、接縫、牆面、路徑抖動與 coverage holes。

加速度計不會積分成 XYZ prior。它只適合後續做靜止／晃動／motion-blur diagnostics。

## 實作邊界

- 目前 focal prior 的自動可信來源是 COLMAP `view_graph_calibrator`。DJI dewarp protobuf 只在已驗證欄位中提供中心、尺寸與 optical-occlusion curve；在沒有真實樣本證明 focal／distortion 欄位語意前，程式不會猜欄位或把 `0.3` 初始值標成 metadata calibration。
- calibrated FOV pair pruning 目前使用每鏡頭姿態與保守的 95° 球面 half-FOV。DJI optical-occlusion curve 已用於遮罩，但尚未被誤當成完整的角度投影模型。
- rolling-shutter 階段只輸出帶校正座標與時間的 trajectory sidecar；像素 dewarp 仍需已知曝光／掃描模型與獨立影像重採樣器。
- 完整 quaternion BA 已定義可驗證的外部工具協定、候選模型驗證與 rollback；repository 不會假裝 stock COLMAP 支援這種 residual。沒有相容 external executable 時，這個步驟會明確略過。
- 真實效能與品質結論必須以實際 OSV 執行 A／B／C benchmark。單元測試只驗證數學、schema、fallback 與交易性，不能替代真實 capture、GPU 與 3DGS 品質檢查。

## COLMAP 依據

實作以 COLMAP 4.1.1 的官方 [CLI](https://colmap.github.io/cli.html)、[database format](https://colmap.github.io/database.html)、[output format](https://colmap.github.io/format.html) 與 [PyCOLMAP PosePrior API](https://colmap.github.io/pycolmap/pycolmap.html) 為準。CLI capability 會對使用者指定的同一個 COLMAP binary 做精確 probe，不只檢查 command 名稱。
