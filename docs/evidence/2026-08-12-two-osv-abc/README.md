# Two-OSV COLMAP A/B/C evidence (2026-08-12)

This directory preserves the raw, machine-generated artifacts used for the comparison. `audit-manifest.json` records the SHA-256 and byte size of every copied artifact.

## Inputs

- `CAM_20260224122138_0020_D.OSV`: `63b97b1c49da2c635a22db87f18f992dac9f832b18b3ead08f7d93ea49369422`
- `CAM_20260503151520_0030_D.OSV`: `bc1643d73966d85ffe15a28c30cbf57811f624336ea807e061d7ddf99eb71c3e`

The exact paths, sizes, timestamps, COLMAP executable, GPU index, and requested variants are in `run-provenance-ab.json` and `run-provenance-c.json`.

## Results

| Requested | Effective benchmark | Effective mapper | Complete rigs | Coverage | Registered images | Largest component | 3D points | Median track | Reprojection | Align |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | `a_current` | `bootstrap_mapper` | 299 / 796 | 37.6% | 602 | 602 / 602 | 153,270 | 4.0 | 0.744 px | 3,337.6 s |
| B | `b_imu_pruning` | `bootstrap_mapper` | 165 / 531 | 31.1% | 333 | 333 / 333 | 62,976 | 4.0 | 0.709 px | 1,583.4 s |
| C | `c_global` | `global_mapper` | 363 / 531 | 68.4% | 726 | 686 / 726 | 52,115 | 4.0 | 0.705 px | 2,280.5 s |

C is the best result for reconstruction coverage in this run. Compared with A it registers 64 more complete rigs and is about 31.7% faster in Align. Compared with B it registers 198 more complete rigs, but takes about 44.0% longer in Align. A and B each produce one connected component; C produces three, while 686 of its 726 registered images (94.5%) remain in the largest component.

The accepted C global candidate increased complete-rig coverage from 157 seed rigs to 363 candidate rigs. `C/global_mapper_candidate.json` contains that promotion gate. `C/global_mapper_priors.json` proves 100% focal coverage, 100% gravity coverage, 1,062 injected image priors, a calibration version, and the representative time offset. `C/imu_calibration.json` preserves every per-source component candidate and the selected valid hand-eye calibration.

Cross-source retrieval was exercised for B and C. Both raw reports record one evaluated source pair, four mutual visual matches, no descriptor failures, and `fallbackToLegacy=false`.

## Run separation

A and B come from the first fresh `A,B,C` run (`abc-summary-ab.json`, `cli-events-ab.jsonl`). That run exposed the post-bootstrap focal/IMU defects. C comes from the final fresh C-only run after those defects were fixed (`abc-summary-c.json`, `cli-events-c.jsonl`). The inputs, CLI settings, COLMAP binary, GPU index, quality profile, and extraction settings are identical; the raw files are kept separate so the implementation revision is not misrepresented as a single invocation.

These are sparse reconstruction metrics, not a substitute for training and visually comparing 3DGS outputs with identical settings.
