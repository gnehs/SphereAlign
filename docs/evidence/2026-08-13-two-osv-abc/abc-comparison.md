# COLMAP A/B/C comparison

Registration coverage and track support take priority over a low reprojection error from a tiny surviving model.

## Inputs

- `CAM_20260224122138_0020_D.OSV`
- `CAM_20260503151520_0030_D.OSV`

## Metrics

| Requested | Effective benchmark | Profile | Complete rigs | Any-sensor rigs | Pairs | 3D points | Median track | Median reprojection | Components | Largest component | Extract | Align | Effective mapper | Gravity applied |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| A | a_current | baseline | 295/796 (37.1%) | 303 | 17320 | 150119 | 4.0 | 0.745 px | 1 | 598 images | 829.5 s | 3130.9 s | bootstrap_mapper | no |
| B | b_imu_pruning | baseline | 164/531 (30.9%) | 168 | 10741 | 58875 | 4.0 | 0.703 px | 1 | 332 images | 829.4 s | 1681.3 s | bootstrap_mapper | no |
| C | b_imu_pruning | baseline | 166/531 (31.3%) | 168 | 16825 | 64043 | 4.0 | 0.708 px | 1 | 334 images | 813.1 s | 2066.5 s | bootstrap_mapper | no |

The table is an automatic sparse-reconstruction comparison. Visual 3DGS quality still requires identical training and render settings.
