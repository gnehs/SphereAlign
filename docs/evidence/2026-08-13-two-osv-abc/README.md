# Two-OSV A/B/C release benchmark (2026-08-13)

This directory preserves the raw artifacts from one fresh A/B/C invocation. All variants used the same inputs, source revision, release CLI binary, COLMAP executable, GPU, and baseline profile.

## Provenance

- Run ID: `abc-1786549692603525700-24056`
- Source commit: `29972618e04527bf8f153622817c161cea8466f2`
- Git dirty at run start: `false`
- Align pipeline revision: `21`
- CLI SHA-256: `2809e94b4bfcf1a10c1d7e95800088b08dfa34ad03eb3e78c9052c2b2301f82e`
- COLMAP SHA-256: `044874e23516a84e2a09510070d1413133394aabd605db02abbb1251fe992a9b`
- `CAM_20260224122138_0020_D.OSV`: `63b97b1c49da2c635a22db87f18f992dac9f832b18b3ead08f7d93ea49369422`
- `CAM_20260503151520_0030_D.OSV`: `bc1643d73966d85ffe15a28c30cbf57811f624336ea807e061d7ddf99eb71c3e`

`run-provenance.json` contains the corresponding sizes and run settings. Committed artifacts contain basenames or placeholders, not user-specific absolute paths.

## Final results

| Requested | Effective benchmark | Effective mapper | Complete rigs | Coverage | Registered images | Largest component | Points | Median track | Reprojection | Align |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | `a_current` | `bootstrap_mapper` | 295 / 796 | 37.1% | 598 | 598 / 598 | 150,119 | 4.0 | 0.745 px | 3,130.9 s |
| B | `b_imu_pruning` | `bootstrap_mapper` | 164 / 531 | 30.9% | 332 | 332 / 332 | 58,875 | 4.0 | 0.703 px | 1,681.3 s |
| C | `b_imu_pruning` | `bootstrap_mapper` | 166 / 531 | 31.3% | 334 | 334 / 334 | 64,043 | 4.0 | 0.708 px | 2,066.5 s |

B is the best default tradeoff for this capture pair: its Align is 46.3% faster than A, although complete coverage is 6.2 percentage points lower. C preserves two more complete rigs than B but takes 22.9% longer, so the effective fallback does not justify the extra work here.

## C global candidate decision

C reached post-rig focal calibration, IMU calibration, gravity injection, calibrated-FOV rematching, and an actual global mapper candidate. Focal and gravity coverage were both 100%.

The global candidate increased complete rigs from 166 to 363 and retained reasonable point, track, and reprojection support, but it fragmented the reconstruction:

- seed: 1 component, largest component 334 / 334 images (100%)
- candidate: 6 components, largest component 311 / 726 images (42.8%)

The quality gate therefore rejected it and retained the connected bootstrap seed. `C/global_mapper_candidate.json` contains the full seed/candidate metrics and rejection reasons. This is expected fail-safe behavior, not a successful C/global result.

## Verification scope

The run exercises two sources and records four mutual cross-source retrieval matches with `fallbackToLegacy=false`. It validates sparse reconstruction behavior; it does not execute the optional external quaternion BA backend or replace an equal-settings visual 3DGS comparison.

`audit-manifest.json` records SHA-256 and byte size for every artifact in this directory except itself.
