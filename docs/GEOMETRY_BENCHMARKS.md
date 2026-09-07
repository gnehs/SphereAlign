# Native normal-map CPU acceleration — 2026-09-07

Release-mode comparison against the frozen serial implementation from commit `424e059`, on this Windows machine with 8 native workers. The fixture uses the real 3840×3840 fisheye calibration and synthetic six-face inference heads, RGB and mask, including invalid support and overlap disagreement. This is one run per variant, not a repeated statistical benchmark or an end-to-end model benchmark.

| Variant | Native gather + seven output files | Relative to serial |
| --- | --- | --- |
| Previous serial implementation | 7,670 ms | 1× |
| Parallel, cold ray cache | 1,535 ms | 5.00× |
| Parallel, warm ray cache | 652 ms | 11.76× |

The cold run spent 911 ms preparing 353,894,400 bytes of exact `f64` camera rays. Both new runs matched all seven reference file SHA-256 hashes (`native.f32`, `reasons.u8`, normal/range/validity/reasons/RGB PNGs) and all five reason counts. Hash comparisons run outside the timed interval. Timings include raw file flush/sync and PNG encoding, but exclude model inference, perspective extraction, shift recovery, production output hashing and final training export. The test-only `timings.totalMs` and `outputMs` remain zero because the production `predict_frame` wrapper is not called. [Machine-readable result](evidence/geometry-2026-09-07/native-gather-performance.json).

Regular parity tests additionally cover PINHOLE and folded OPENCV_FISHEYE, changing masks, partial final stripes, and cancellation before output. Cache tests verify bit-exact rays, calibration-based identity, eviction and oversize fallback. The full current library suite passed: **363 passed, 0 failed, 18 ignored**; the explicit release benchmark also passed. Ignored tests include hardware-dependent checks and this opt-in benchmark. No new GPU inference or full-dataset elapsed-time measurement was performed for this optimization.

Frontend build and the release NSIS package completed successfully. The release executable contains the new native worker and timing fields; executable/installer SHA-256 hashes are recorded in [package verification](evidence/geometry-2026-09-07/native-speed-package.json). The installer was built but not installed over the running application.

```powershell
cargo test --manifest-path src-tauri/Cargo.toml --release --lib geometry::draft::tests::native_gather_benchmark --offline -- --ignored --nocapture
```

# Production normal-map integration — 2026-09-07

- `disney_cruise_room`: 202 normal maps, 20,000 training steps, about 39m50s, exported PLY 2,801,390 splats. Source COLMAP hashes and exported normal hashes verified; finite PLY attributes checked.
- User inspection of the trained scene reported a substantial reduction in floaters and explicitly requested production integration. Six fixed novel-view A/B renders had shown mixed results; that limited static review is not a substitute for the user's scene inspection. No new claim of quantitative or universally improved quality is made.
- Completed cleanup removed 50,636,390,400 bytes in 404 reproducible intermediate files. All normal PNGs, previews, source data and training outputs were retained.
- Production integration CPU suite: 360 passed, 16 environment-dependent tests ignored. This includes all-frame export, no-GPU cache reuse, intermediate cleanup, foreign-output preservation and read-only polling/legacy-stage migration. Frontend production build passed.
- Browser component check passed in Traditional Chinese: fourth-stage description, original-resolution normal preview, advanced sample controls, default automatic cleanup and preflight disk estimates. Tauri IPC was mocked; this was not a complete desktop automation run.
- Production CLI export on the real 202-frame dataset completed with exit code 0. Every export-manifest hash matches the previously trained normal PNGs; all emitted training configuration fields match that verified run. Cached inference was reused and raw intermediates remained absent. [Verification record](evidence/geometry-2026-09-07/production-normal-export.json).
- The current workflow is documented in [GEOMETRY_PRIORS.md](GEOMETRY_PRIORS.md). The measurements below describe the preceding inference prototype.

---

# Geometry Phase 0：實測、命令與未完成驗收

## 2026-09-07 界面與原生草稿接線

- 真實 DirectML 在解析魚眼 fixture 的 `lens0/frame000.png` 完成六面流程（五面有輸入支援、一面跳過），输出 64×64 native maps。有效支援 2379／4096 = 58.1%，鏡頭外 868、模型／恢復無效 539、面重疊不一致 310。**不是品質驗收數字**，也不是實景品質提升證據。
- 額外的 Rust `directml_cancel_resume_native_end_to_end` 使用兩個非連續 camera/image IDs、相同 basename 的雙鏡頭 32×32 fixture，真實建立 GPU session。第一張提交後取消，續跑不重寫第一張、完成第二張；再執行所有輸出 hash 核對通過且不建立 GPU session。輸出全部 finite、native 尺寸正確、source SHA 不變、未產生 training normals/depths。此測試實際通過，65.50 秒（含重載 session 與檔案寫入，不是純 kernel 時間）。
- 整體 Rust：350 passed、16 ignored；其中 Geometry GPU 測試另以上述命令顯式執行通過，其餘 ignored 是既有外部素材／硬體測試。CPU Geometry 包括 PINHOLE／fisheye 含 rear-ray／畸變 round-trip、SO(3)、平面已知焦距恢復、binary／text noncontinuous IDs 與 UTF-8／重複空白路徑、mask 命名、writer lock、輸入／快取變更拒用。
- `pnpm build` 通過，僅有既有的大 chunk 提示。前端測試使用實際 `GeometryDrafts` component、Tauri 官方 IPC mocks 與真實模型 PNG，已在本機瀏覽器確認繁中設定／按鈕 gating／running 鎖定／取消／續跑／原因圖 legend。這是 component 互動測試，**不是完整桌面 Tauri IPC 的自動操作驗收**（桌面自動化啟動等待逾時）。
- 實景品質、跨面尺度對齊、anchor／holdout／multi-view validation、獨立 normal/depth acceptance、正式 exporter、完整場景 VRAM／disk 峰值與使用者 A/B/C/D 留待接續測試。

```powershell
$env:GEOMETRY_TEST_MODEL=(Resolve-Path .work/geometry-spike/rust-static-768.onnx).Path
cargo test --manifest-path src-tauri/Cargo.toml --lib geometry::draft::tests::directml_cancel_resume_native_end_to_end --offline -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml --lib --offline -- --test-threads=2
pnpm build
```

以下保留 2026-09-06 的 Phase 0 原始量測。

2026-09-06，Windows x86_64，RTX 4070 Ti，總顯存 12282 MiB，driver 610.62。沒有中止其他 GPU 工作。**本報告不是整個 Geometry stage 或使用者場景品質報告。**

## 實際量測

| 測項 | 實測結果 |
| --- | --- |
| 原始 ONNX 下載 | 419411850 bytes；SHA-256 與規格完全相同 |
| 原始 ONNX / DirectML | 動態、32²、96×64、768² 的 session 初始化均因 CPU EP 節點失敗；CPU fallback 保持關閉 |
| 靜態 768² / 1800 tokens / DirectML | Python shape-inferred 與 Rust 原生準備的獨立 artifacts 均成功執行 |
| Rust / Python 準備一致性 | 無 shape inference 的靜態 ONNX byte/hash 完全相同：`e3a6e5…8ec4ac` |
| 原始 CPU ORT reference | 32²/4 tokens：0.0200 s；96×64/24：0.0536 s；768²/1800：3.4215 s，均 finite |
| Rust 靜態 GPU probe | 含 hash／session／推論／統計約 14.64 s；run 包含輸入建立、GPU 推論、host readback、dump 與統計約 5.185 s。未作純 GPU kernel 計時 |
| 768² raw points parity | relative L2 `2.084757e-6`；max abs `7.748604e-6` |
| normal parity | median `0.000574°`、P99 `0.003527°`、max `0.006792°` |
| mask parity | threshold disagreement 0；reference 的 threshold±0.01 範圍內 **0 pixels**，故該邊界尚未測到 |
| known-focal numeric diagnostic | positive-z relative L2 `6.107657e-6`；不是可宣稱正確幾何的場景 |
| 原生 probe 資源單次採樣 | 14.33 s；process peak working set 2613882880 bytes；max sampled private bytes 3760082944 |
| GPU memory | 採樣期間 device-wide 9445–10801 MiB／12282 MiB，包含其他工作。**不可當作模型單獨 VRAM 峰值或 12GB stage 保證** |
| Spirula `geometry --check` | binary 實際通過 camera round-trip 與 split checks |
| 解析魚眼 actual train smoke | baseline 與 normals+depth 各 3 steps 完成；416 seed points、8 cameras；console 有 RGB loss，但沒有逐像素 prior gradient |
| 追加 actual smoke 矩陣 | normals-only、depth-only、priors/divisor=2、priors/warp/divisor=2 均 exit 0；warp 案例有 40 split cameras。4 案例各約 4.2–4.7 s，來源 hashes 不變；[exact argv/report](evidence/geometry-2026-09-06/spirula-matrix/report.json) |
| stock sampler sentinel | CPU `[0,1000]` → `[0,250,750,1000]`；GPU weight renormalization 會延展 validity；嚴格契約不合格 |
| proposed patch host test | `[0,0,0,1000]`，all-invalid／island／normal／zero-weight taps 通過；RGB interpolation 不變 |
| fixture 不可變性 | trainer smoke 後 31 個來源檔 hash 不變 |

數值來自 [evidence 目錄](evidence/geometry-2026-09-06/)。兩個静態 artifact 在同一合成 RGB pattern 上得到相同 parity 報告；不可把這一張當成真實室內素材、完整 provider 認證或模型幾何正確率。短訓練的 RGB 指標不列為 A/B 品質提升，因為沒有控制 seed、正式 capture-group holdout 或重複實驗。

## 已執行的基本驗證

```powershell
pnpm build
pnpm version:check
pnpm version:check 0.1.0
cargo check --offline --manifest-path src-tauri/Cargo.toml
cargo test --lib --offline --manifest-path src-tauri/Cargo.toml -- --test-threads=2
cargo build --offline --manifest-path src-tauri/Cargo.toml --bin spherealign-geometry-probe
```

- `pnpm build` 通過；既有 Vite chunk-size warning 保留。
- 原樣 `pnpm version:check` 回傳 exit 1，既有 script 要求版本參數；用 `pnpm version:check 0.1.0` 通過。沒有為此修改版本 script。
- `cargo check` 通過；保留既有 `telemetry::parse_and_write` dead-code warning。
- 最終 `cargo test --lib`：**342 passed、0 failed、15 ignored**。其中 7 個新增 Geometry CPU 測試不需 ONNX/GPU。完整輸出包含原有 ignored 測試的各別原因；沒有把它們改成通過。
- Windows 的新 CLI build 通過。新增 CPU tests 已接入既有 CI；**遠端 CI 未執行**。

## 模型與 parity 重現命令

本機實際使用 `.work/.venv/Scripts/python.exe`，額外參考依賴安裝在獨立的 `.work/geometry-spike/pydeps`，沒有改變 app runtime：

```powershell
.work/.venv/Scripts/python.exe -m pip install --target .work/geometry-spike/pydeps onnx==1.19.1 onnxruntime==1.23.2
.work/.venv/Scripts/python.exe scripts/geometry_fetch_spike.py --model
.work/.venv/Scripts/python.exe scripts/geometry_inspect_onnx.py .work/geometry-spike/model.onnx --dependencies .work/geometry-spike/pydeps --output docs/evidence/geometry-2026-09-06/onnx-inspection.json

src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx DirectML
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx DirectML 32 32 4
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx DirectML 96 64 24
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx DirectML 768 768 1800

.work/.venv/Scripts/python.exe scripts/geometry_specialize_onnx.py .work/geometry-spike/model.onnx --dependencies .work/geometry-spike/pydeps --output .work/geometry-spike/static-768-t1800.onnx
.work/.venv/Scripts/python.exe scripts/geometry_specialize_onnx.py .work/geometry-spike/model.onnx --dependencies .work/geometry-spike/pydeps --output .work/geometry-spike/static-768-noinfer.onnx --skip-shape-inference
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/model.onnx --prepare-static .work/geometry-spike/rust-static-768.onnx
src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/rust-static-768.onnx DirectML --static-experiment .work/geometry-spike/gpu-native-768

.work/.venv/Scripts/python.exe scripts/geometry_reference_ort.py .work/geometry-spike/model.onnx --dependencies .work/geometry-spike/pydeps --output docs/evidence/geometry-2026-09-06/reference-32
.work/.venv/Scripts/python.exe scripts/geometry_reference_ort.py .work/geometry-spike/model.onnx --dependencies .work/geometry-spike/pydeps --output docs/evidence/geometry-2026-09-06/reference-96x64 --width 96 --height 64 --tokens 24
.work/.venv/Scripts/python.exe scripts/geometry_reference_ort.py .work/geometry-spike/model.onnx --dependencies .work/geometry-spike/pydeps --output .work/geometry-spike/reference-768 --width 768 --height 768 --tokens 1800
.work/.venv/Scripts/python.exe scripts/geometry_compare_heads.py .work/geometry-spike/reference-768 .work/geometry-spike/gpu-native-768 --dependencies .work/geometry-spike/pydeps --output docs/evidence/geometry-2026-09-06/parity-native-768.json
.work/.venv/Scripts/python.exe scripts/geometry_measure_probe.py --output .work/geometry-spike/resources-native -- src-tauri/target/debug/spherealign-geometry-probe.exe .work/geometry-spike/rust-static-768.onnx DirectML --static-experiment
```

模型／dump／fixture／resource output 路徑要求新檔案或新目錄；重跑時使用新名稱。原始 graph 四個 probe 的非零 exit code 是被記錄的真實失敗。靜態 profile 不會在它失敗後自動接替。

## Spirula 與 patch 重現命令

```powershell
& 'C:/Users/gnehs/Downloads/spirula.exe' --help --lang en
& 'C:/Users/gnehs/Downloads/spirula.exe' train --help-all --lang en
& 'C:/Users/gnehs/Downloads/spirula.exe' geometry --check --lang en
.work/.venv/Scripts/python.exe scripts/geometry_make_fixture.py '.work/geometry-spike/中文 check dataset'
.work/.venv/Scripts/python.exe scripts/geometry_spirula_sentinel.py
.work/.venv/Scripts/python.exe scripts/geometry_spirula_smoke.py --spirula C:/Users/gnehs/Downloads/spirula.exe --fixture '.work/geometry-spike/中文 check dataset' --output .work/geometry-spike/spirula-matrix
```

矩陣 harness 保存 exact argv、binary hash、stdout/stderr 與來源 hash 檢查，依序測 normals-only、depth-only、priors/divisor=2、priors/warp/divisor=2。它最多允許每個 child 90 秒，逾時會結束該 child，並明記 timeout。這個 smoke harness 不是完整 A/B/C/D 品質 harness。

最初兩個獨立 3-step 命令使用下列已由 help 核對的參數：

```powershell
& 'C:/Users/gnehs/Downloads/spirula.exe' train --data 'D:/Repos/gs360studio/.work/geometry-spike/中文 check dataset' --data-format colmap --output-dir-prefix 'D:/Repos/gs360studio/.work/geometry-spike/train-smoke' --output-dir-name baseline --num-iterations 3 --cap-max 512 --min-init-fraction 0 --disable-viewer 1 --train-resolution-divisor 1 --eval-mode all --load-depths 0 --load-normals 0 --normal-supervision-weight 0 --depth-supervision-weight 0 --sh-degree 0 --lang en
& 'C:/Users/gnehs/Downloads/spirula.exe' train --data 'D:/Repos/gs360studio/.work/geometry-spike/中文 check dataset' --data-format colmap --output-dir-prefix 'D:/Repos/gs360studio/.work/geometry-spike/train-smoke' --output-dir-name normals-depth --num-iterations 3 --cap-max 512 --min-init-fraction 0 --disable-viewer 1 --train-resolution-divisor 1 --eval-mode all --load-depths 1 --load-normals 1 --normal-supervision-weight 0.0025 --depth-supervision-weight 0.01 --median-normal-supervision-weight 0 --supervision-warmup 0 --input-depth-is-ray-depth 1 --depth-unit-scale-factor 0.00024414435034714275 --sh-degree 0 --lang en
```

上述 depth multiplier 是解析 fixture 的 `16/65535`；不代表實測米制或已認證的正式 exporter 設定。沒有因極短 smoke 通過而聲稱 posterior geometry 更好。

## Skipped／阻擋條件

| 驗收項目 | 狀態與原因／接續所需 |
| --- | --- |
| 原始 graph GPU inference/parity | **blocked**：DirectML session 拒絕 CPU nodes；不能靜默 CPU fallback |
| 靜態 profile 更多影像／threshold 邊界 | **not run**：目前只完成單張數值 fixture，待更多合法影像與邊界案例；不是完整認證 |
| macOS CoreML / CUDA | **unverified**：無 macOS；未建置 CUDA feature/runtime 測試 |
| 修補版 actual trainer sign／units／prior-gradient | **blocked**：本機 binary 不同於 pinned source；未提供逐像素 prior telemetry，也未安裝／建置獨立 patch。需 instrumented pinned/patched build |
| 全 invalid、island、slanted edge 的實際 loss backward、多尺度與 warp | **unverified**：host sampler 測試不可代替。即使 trainer smoke 能跑也不升級狀態 |
| SphereAlign stage／projection／anchors／cross-view／active/export | **not implemented**，見 Phase 表；沒有用 mock 宣稱完成 |
| 舊 project migration／optional stage 進度／GUI／run 取消續跑／磁碟滿及 run 崩潰復原 | **not implemented/tested**：沒有接入新 stage；模型 no-clobber 測試不能代替 |
| 真實使用者場景 A/B/C/D、PSNR/SSIM/LPIPS／幾何穩定性 | **not run**：沒有完成的 preprocessing/exporter，不能產生可信的 C/D；沒有品質改善結論 |
| 全 stage 峰值與 12GB 上限 | **not measured**：目前只有單一 probe 的 process memory 與 device-wide GPU 採樣 |

這些未完成項是實際交付界線；本次不能標記整合完成，亦沒有 production-supported active profile。
