<div align="center">
  <img src="src-tauri/icons/icon.png" width="96" alt="SphereAlign logo">
  <h1>SphereAlign</h1>
  <p>Turn raw DJI Osmo 360 footage into a resumable dual-fisheye COLMAP project.</p>
  <p><a href="README.zh.md">繁體中文</a></p>
</div>

> [!WARNING]
> **This project is still under development.** Features, output formats, and workflows may change. SphereAlign currently supports **DJI Osmo 360** only; other cameras and video sources have not been validated and are not supported.

![SphereAlign processes Osmo 360 footage through dual-fisheye frame extraction, sharp-frame selection, dynamic-object and sky masking, camera alignment, and sparse point-cloud reconstruction](assets/readme/workflow-hero.png)

## From raw footage to a resumable reconstruction project

SphereAlign brings a workflow that would otherwise require switching between multiple tools, organizing files manually, and repeatedly checking progress into a single local desktop app. Add one or more Osmo 360 recordings captured in the same scene, then run frame extraction, masking, and camera alignment in sequence.

| Frame extraction | Scene masking | Camera alignment |
| --- | --- | --- |
| Select synchronized frames from both fisheye views, preserve the native fisheye images, and prioritize sharper pairs. | Automatically identify people, bicycles, and common vehicles, with optional masks for the sky and invalid lens regions. | Build constrained pairs for the dual-camera rig, then use COLMAP to generate camera poses and a sparse reconstruction. |

## Features

- **Process multiple recordings in one task**: Combine multiple Osmo 360 files captured in the same space into a single reconstruction project.
- **Preserve native dual-fisheye data**: Process the front and rear fisheye images directly instead of converting them to an equirectangular projection first, avoiding an unnecessary resampling step.
- **Select frames using IMU data and visual change**: Choose keyframes based on fused-attitude relative angle, low-resolution visual novelty, and a maximum time interval, then decode only the selected full-resolution dual-camera pairs.
- **Nearly 2× faster with sensor assistance**: On the same real-world test footage, IMU- and visual-change-based frame selection reduced Align time from about 52.2 minutes to 28.0 minutes—a 1.86× speedup (46.3% less time). Actual results vary with motion, texture, and visual overlap between sources; this is not a guaranteed fixed improvement.
- **Reduce dynamic interference**: Mask people, bicycles, cars, motorcycles, buses, trucks, and the sky so reconstruction can focus on the static scene.
- **Alignment designed for 360 rigs**: Use an IMU-aware temporal graph within each source and bounded visual retrieval across sources. After calibration, further reduce pairings based on fisheye FOV overlap.
- **Use capture metadata safely**: First estimate time offset and rotational hand-eye calibration from the visual model. Write per-camera gravity priors only after the residual, coverage, rig, and focal gates pass. A complete DJI quaternion is never passed off as a COLMAP qvec.
- **Automatically level training coordinates**: After Align finishes, rotate the entire sparse model using calibrated per-image gravity so LichtFeld's `+Y` axis always points upward. SphereAlign does not estimate ground height, impose an additional yaw, or depend on a successful Global Mapper candidate.
- **Automatically switch between Incremental and Global**: The first run can use incremental mapping to establish a calibration seed. Once the COLMAP 4.1.1 prerequisites pass, a candidate global model replaces the original result only after successful validation.
- **Local-first processing**: Videos, frames, masks, and reconstruction results are processed locally. The required model may be downloaded the first time you use masking.
- **Environment capability checks**: View runtime requirements such as FFmpeg, COLMAP, hardware acceleration, and storage capacity in one place.

## Stop when needed, resume when ready

![SphereAlign lets you cancel or retry frame extraction, masking, and alignment independently, then resume from local checkpoints](assets/readme/resumable-workflow.png)

Frame extraction, masking, and alignment can each be run, canceled, retried, or rerun independently. Completed artifacts remain in the project. When you reopen a task, SphereAlign checks existing results and resumes from any safely reusable progress whenever possible instead of starting over every time.

For IMU and Global Mapper calibration gates, artifacts, and benchmark methodology, see the [IMU reconstruction workflow](docs/IMU_RECONSTRUCTION.md). For the development environment, build instructions, architecture, complete output structure, and implementation boundaries, see the [development documentation](docs/DEVELOPMENT.md).

The performance figures above come from the [2026-08-13 A/B/C benchmark](docs/evidence/2026-08-13-two-osv-abc/README.md), which compares the Align stage using identical inputs. If two sources were not captured in the same scene or have no visual overlap, their registration rate should not be used to assess the speedup from IMU-assisted frame selection.
