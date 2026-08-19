<div align="center">
  <img src="src-tauri/icons/icon.png" width="96" alt="SphereAlign logo">
  <h1>SphereAlign</h1>
  <p>Convert raw panoramic camera footage into a COLMAP dataset.</p>
  <p><a href="README.zh.md">繁體中文</a></p>
</div>

> [!WARNING]
> **This project is still under development.** SphereAlign currently supports some **DJI Osmo 360** (`.OSV`) and **Insta360** (`.INSV`) models. Other models or formats may work, but I do not have the corresponding devices or files, so they have not been thoroughly tested.

![SphereAlign selects frames from dual-fisheye panoramic video, masks distractions, aligns cameras, and builds a sparse 3D reconstruction](assets/readme/workflow-hero.png)

## Turn raw footage directly into a COLMAP dataset

This workflow used to require switching between several applications and CLI tools. You had to organize files manually, remember a long list of commands, parameters, and their execution order, and dig through logs yourself whenever something went wrong. The entire process was cumbersome and easy to get stuck on.

SphereAlign brings all these steps into a single interface. Simply add one or more panoramic videos captured in the same scene, and it automatically completes the following stages in order:

| Select frames | Generate masks | 3D reconstruction |
| --- | --- | --- |
| Automatically discard blurry frames and select footage suitable for training. | Automatically detect and mask people, bicycles, and common vehicles to reduce interference from moving objects in the training results. | Pair the two lenses as a rig and let COLMAP calculate and infer the camera positions. |

## How to use and download

SphereAlign is currently in **beta**. If you encounter any problems, please open an issue and include the complete error message and execution logs whenever possible.

> [!TIP]
> **An NVIDIA GPU with CUDA support is strongly recommended.**
>
> COLMAP's bundle adjustment can be extremely slow without CUDA acceleration.
>
> In testing, the same reconstruction that took about half an hour on a CUDA-enabled machine could take several days without CUDA.

<details>
<summary>macOS</summary>

1. Install COLMAP and FFmpeg with Homebrew.
2. Download SphereAlign from Releases.
3. Open the DMG and drag SphereAlign into Applications.
4. If macOS prevents the app from running, use `xattr -c` to remove the quarantine attribute.

</details>

<details>
<summary>Windows</summary>

1. Install FFmpeg.
2. Download COLMAP. The precompiled, CUDA-enabled [COLMAP Windows Build](https://github.com/lyehe/build_gpu_colmap) is recommended.
3. Download and install SphereAlign from Releases.
4. Open SphereAlign's settings and select the COLMAP executable you just downloaded, for example:

   `bin\colmap.exe`

</details>

<details>
<summary>Linux</summary>

1. Install COLMAP and FFmpeg.
2. Download and install SphereAlign from Releases.
3. Run SphereAlign.

> [!WARNING]
> The Linux version has not been thoroughly tested, and I cannot guarantee that it will work in every environment.
> If you encounter a problem, please open an issue and include details about your system environment, the complete error message, and execution logs. Thank you!

</details>

## Key features

* **Process multiple panoramic videos at once**
  Combine multiple dual-fisheye videos captured in the same scene directly into a single reconstruction project.

* **Use the original dual-fisheye footage without panorama stitching or conversion**
  There is no need to convert dual-fisheye footage into a panorama, cubemap, or another projection first. This saves conversion and resampling time while avoiding misalignment, ghosting, and detail loss caused by panorama stitching seams.

* **Automatically select only the frames you actually need**
  Combine camera sensor data, viewing angles, and visual changes to discard unnecessary frames and process only footage suitable for reconstruction.

* **Use camera sensors to greatly accelerate alignment**
  Skip panorama conversion, then use camera sensor data and visual changes to select frames and narrow the pairing range. In testing, alignment time dropped from about 52 minutes to 28 minutes—a speedup of approximately 1.86×.

* **Optimize reconstruction for panoramic cameras**
  Treat the two lenses as a fixed rig and use their known relationship to build pairs, helping COLMAP calculate camera positions more efficiently.

* **Automatically mask people, vehicles, and other distractions**
  Mask people, bicycles, cars, motorcycles, buses, trucks, and the sky to reduce the effect of moving objects on feature matching and reconstruction results.

* **Automatically level the model**
  Use camera sensor data to determine the direction of gravity and automatically level the model, eliminating the need to rotate it manually and repeatedly align the ground and horizon.

## Resume where you left off

![SphereAlign lets you cancel or retry frame extraction, masking, and alignment independently, then resume from local checkpoints](assets/readme/resumable-workflow.png)

Frame selection, masking, and alignment can each be run, canceled, retried, or rerun independently. Completed results remain in the project. When you reopen a task, the app checks and reuses existing progress instead of starting over.
