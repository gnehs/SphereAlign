# Geometry compatibility — measured, not certified

Update 2026-09-07: the user reviewed the complete 20,000-step `disney_cruise_room` scene and confirmed a substantial reduction in floaters. Camera-space RGB8 normals are now integrated into the production pipeline with supervision weight 0.01, native fisheye cameras and depth loading disabled. See [the current workflow](GEOMETRY_PRIORS.md). The tested installed trainer SHA-256 is `263a45b3e4e9b91550f5d6d2ed94cf2fb9d73ff9869233ba7e257475138bae77`.

This is an empirical normals-only rollout. It does not certify all sampler boundary/gradient semantics or enable metric depth. The dated findings below remain the record of the earlier compatibility investigation; their original “no activation” status has been superseded for this normal-map path by the user's scene review.

Storage update 2026-09-07: at the user's request, new normals use lossy JPEG quality 90 with full chroma resolution. PNG cache conversion is resumable and avoids inference; previous training exports remain archived. The same installed trainer completed a JPEG-only normal-input three-step smoke. Measured compression and angular differences are recorded in [GEOMETRY_BENCHMARKS.md](GEOMETRY_BENCHMARKS.md). The earlier complete scene review used PNG, not JPEG; neither full-scene JPEG/PNG A/B quality nor exact invalid-pixel behavior after JPEG is newly certified.

## Actual versions

| Component | Actual identity |
| --- | --- |
| SphereAlign working-tree HEAD at start | `d5f3287a2f9feb416f52b0d03208f41488ceb16b` |
| Git origin | `https://github.com/gnehs/gs360studio.git`; project/package is SphereAlign, no remote changed |
| Existing changes | Cargo.toml already marked modified; many untracked reconstruction/training scripts and evidence preserved |
| Design Spirula source snapshot | `bddda193ee09f03ea1b0aad44b5c6b96f81e249a` |
| Installed executable | `C:\Users\gnehs\Downloads\spirula.exe`, 130512384 bytes |
| Installed reported version | `Spirula Studio 2026.9.2 (4361a41)` — **different from design snapshot** |
| Installed binary SHA-256 | `263a45b3e4e9b91550f5d6d2ed94cf2fb9d73ff9869233ba7e257475138bae77` |
| MoGe inspected source | `74fbce054ebed49800de42d0ad0e83495065719a` |
| LichtFeld design reference | `1466bb107f317a4a3222261333392dd2c3fbcd93`; no runtime integration or loss port |

Source acquisition receipts retain URL, hash and byte size in [acquisition.json](evidence/geometry-2026-09-06/acquisition.json). GitHub tree API hit an unauthenticated rate limit; exact known raw sources were retrieved. The design's MoGe `docs/normal.md` path returned 404 at the inspected current commit; no normal-head convention is inferred from that missing document.

## Matrix

| Target / path | Result | Limitation |
| --- | --- | --- |
| Windows DirectML, original graph | **failed** | CPU-assigned nodes; fallback stays disabled |
| Windows DirectML, static 768×768/1800 | **draft inference implemented, not training-certified** | Native dataset run and cancel/resume workflow; actual-scene quality and diverse image parity suite pending |
| CUDA | **unverified** | Optional feature not built/tested here |
| macOS CoreML | **unverified** | No macOS hardware in this environment |
| Installed Spirula `geometry --check` | **passed** | Its own math suite; not SphereAlign projection or trainer gradients |
| Installed Spirula analytic native-fisheye import + 3 steps | **smoke passed** | Actual binary, baseline and priors; different version, no pixel-level prior telemetry |
| Installed Spirula normals-only / depth-only / divisor 2 / pinhole warp | **smoke passed** | Four actual 3-step runs, unchanged source hashes; no gradient certificate |
| Design snapshot CPU depth/normal resize | **strict sentinel failed** | Ordinary bilinear mixes invalid values |
| Design snapshot GPU revalidated sampler | **strict sentinel failed** | Renormalizing remaining taps extends support into rejected regions |
| Proposed patch, extracted host functions | **passed** | Not compiled/installed as complete trainer; no CUDA/Slang backward execution |
| Source-camera normal sign, range-to-render depth, resize/warp/multiscale zero gradients | **unverified** | Require instrumented pinned binary/build |

## Trace findings and limits

[DataManager.cpp](https://github.com/harry7557558/spirula-studio/blob/bddda193ee09f03ea1b0aad44b5c6b96f81e249a/src/data/DataManager.cpp) requires UINT16 depth in its stb path and reads three RGB8 channels for normals. Its CPU mismatch-size branches call ordinary bilinear resizers. Exact-size memcpy avoids that branch but does not prove downstream warp/multiscale safety.

[Interpolation.cuh](https://github.com/harry7557558/spirula-studio/blob/bddda193ee09f03ea1b0aad44b5c6b96f81e249a/src/core/Interpolation.cuh) excludes invalid tap values and renormalizes valid weights. This avoids averaging a sentinel as a numeric sample but still extends valid support across a hole. The stricter SphereAlign requirement must reject sampling support containing a positive-weight invalid tap, or carry explicit validity throughout the trainer.

[per_pixel_losses.slang](https://github.com/harry7557558/spirula-studio/blob/bddda193ee09f03ea1b0aad44b5c6b96f81e249a/src/shaders/per_pixel_losses.slang) checks depth and normal validity independently of RGB. Both rendered-normal/reference and depth-derived-normal/reference terms exist; normal supervision can affect position. The Pearson reduction includes floors and a numerical `+1.0` term: do not claim exact scale invariance at all numerical magnitudes. This source alone does not resolve upstream depth transformation. The comment mentioning log is not an instruction for the exporter to take log. Full loader/rasterizer call-chain and derivative unit tests remain unverified; no production `depth_unit_scale_factor` is chosen.

[TrainerCore.cpp](https://github.com/harry7557558/spirula-studio/blob/bddda193ee09f03ea1b0aad44b5c6b96f81e249a/src/app/TrainerCore.cpp) gates supervision with `step > supervision_warmup`; it is not a linear ramp. Per-image scale adjustment is for diagnostics/reprojection and does not add LichtFeld-style absolute anchoring. SfM units and model metric estimates are not measured metres.

## Reproduction fixture and patch

`scripts/geometry_make_fixture.py` creates 8 native 64×64 OPENCV_FISHEYE images with two independently translated lens centers, opposite lens orientations, sparse non-contiguous IDs, tilted inward-facing walls, relative ray-range UINT16 PNGs, RGB8 normals, duplicate basenames in lens directories, missing maps, all-invalid maps and a slanted invalid region. RGB masks remain independent. Its sparse tracks intentionally have one observation per point: usable for importer testing, **insufficient for anchor certification**. Use `--camera PINHOLE` for the alternate analytic case.

`scripts/geometry_spirula_sentinel.py` verifies pinned source hashes, extracts and compiles original CPU resize and GPU weight-revalidation logic as host C++17 functions, then tests the patch. Measured stock depth `[0,1000]` resizes to `[0,250,750,1000]`; strict patched result is `[0,0,0,1000]`. Tests cover normal sentinel, all-invalid, isolated valid island, exact zero-weight taps and unchanged RGB interpolation. GPU support logic is executed on the CPU; this is explicitly **not** GPU/gradient certification.

The proposed [minimal patch](../patches/spirula/geometry-sentinel.patch) is separate from upstream. [Patch scope and license](../patches/spirula/README.md). It has not been applied to another repository or installed binary.

Before active profiles can be enabled, an actual patched/pinned build must execute native/warp and divisors/multiscales tests and expose prior loss or gradients showing: rejected geometry gives zero corresponding prior contribution, valid geometry contributes with correct oriented source-camera normals and ray semantics, and RGB gradients remain present. The installed console only reports aggregate RGB metrics, so successful 3-step training cannot establish these properties. This is a concrete missing validation interface, not a passed compatibility check.

## Actual CLI capability probing

The installed `train --help-all --lang en` says booleans use `0/1`, and hyphens/underscores are interchangeable. Captured help is [here](evidence/geometry-2026-09-06/spirula-train-help-all.txt). The smoke command uses only advertised flags and passes an argument vector. Its synthetic depth decoder input multiplier `16/65535` belongs solely to this fixture, not a release/exporter unit contract. No seed flag was invented; reproducible quality A/B and controlled random initialization were not established. No production argv, `train.sh`, `train.ps1` or active profile is generated.
