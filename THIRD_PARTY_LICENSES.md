# Third-Party Licenses

SphereAlign-developed source code is licensed under `AGPL-3.0-only`, as described in [LICENSE](LICENSE). Third-party programs, libraries, models, and other assets are not relicensed by SphereAlign and remain subject to their respective original license terms. You are responsible for complying with the terms that apply to the particular versions and builds you use or distribute.

Open-source license identifiers below are SPDX identifiers. The Ultralytics Enterprise License is a separate commercial license.

## FFmpeg

- **Use in SphereAlign:** SphereAlign invokes the separately installed `ffmpeg` and `ffprobe` executables as external command-line tools. It does not link against FFmpeg libraries.
- **License:** `LGPL-2.1-or-later` by default, or `GPL-2.0-or-later` when the FFmpeg build includes GPL-covered optional components. The license of the actual executable therefore depends on how that executable was built.
- **Official project:** [ffmpeg.org](https://ffmpeg.org/)
- **Official licensing information:** [FFmpeg License and Legal Considerations](https://ffmpeg.org/legal.html)

## COLMAP

- **Use in SphereAlign:** SphereAlign invokes a separately installed, user-selected COLMAP executable as an external command-line tool. It does not link against COLMAP libraries.
- **License:** `BSD-3-Clause` for COLMAP itself. COLMAP's official notice states that its dependencies are separately licensed and may affect the licensing obligations of a particular binary build.
- **Official project:** [github.com/colmap/colmap](https://github.com/colmap/colmap)
- **Official licensing information:** [COLMAP COPYING.txt](https://github.com/colmap/colmap/blob/main/COPYING.txt)

## Ultralytics YOLO11

- **Use in SphereAlign:** An optional YOLO11 segmentation model is used for object-mask generation. The model is not part of the SphereAlign source tree and is downloaded separately when first needed.
- **License:** `AGPL-3.0` by default, or the Ultralytics Enterprise License when separately obtained from Ultralytics. The model remains independently licensed by Ultralytics; SphereAlign's license does not replace its terms.
- **Official project:** [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Official model documentation:** [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- **Official licensing information:** [Ultralytics Licensing](https://www.ultralytics.com/license)
- **Current downloaded artifact:** [`yolo11s-seg.onnx` in gs360masker](https://github.com/gnehs/gs360masker/blob/5f26a7c1d9de98fff6ee6ffef51701e1d288a27d/src-tauri/resources/models/yolo11s-seg.onnx)

## SkySeg

- **Use in SphereAlign:** The SkySeg ONNX model provides optional sky segmentation and is downloaded separately when first needed.
- **License:** `MIT`
- **Current model source:** [JianyuanWang/skyseg pinned revision](https://huggingface.co/JianyuanWang/skyseg/tree/3ba8c6df1d9ba9ff26f637c7ba9568ac11a9aa7f)
- **Original project:** [github.com/xiongzhu666/Sky-Segmentation-and-Post-processing](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing)

## ONNX Runtime

- **Use in SphereAlign:** ONNX Runtime is used to run YOLO11 and SkySeg ONNX model inference through the Rust `ort` integration.
- **License:** `MIT`
- **Official project:** [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **Official licensing information:** [ONNX Runtime LICENSE](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)

## ort

- **Use in SphereAlign:** The Rust `ort` crate provides the bindings used to integrate ONNX Runtime and obtain its runtime binaries.
- **License:** `MIT OR Apache-2.0`
- **Official project:** [github.com/pykeio/ort](https://github.com/pykeio/ort)
- **Official licensing information:** [ort licensing files](https://github.com/pykeio/ort#license)

## Model and dependency changes

### Geometry feasibility tools (2026-09-06)

- **MoGe candidate:** [Ruicheng/moge-2-vitb-normal-onnx](https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/tree/2d247a122ada42ce700fe59273cd62e076da1f32), downloaded only by an explicit developer command. No model weights are committed or automatically downloaded by the app. The [original model card](https://huggingface.co/Ruicheng/moge-2-vitb-normal/blob/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/README.md) declares MIT, while the pinned ONNX repository contains no separate model card/license. Artifact-specific provenance/notice remains a release check; see [model contract](docs/GEOMETRY_MODEL_CONTRACT.md).
- **Spirula reference/patch:** Exact sampler functions are downloaded and compiled by a developer reproduction tool. The source is covered by [Spirula's pinned GNU GPL v3 LICENSE](https://github.com/harry7557558/spirula-studio/blob/bddda193ee09f03ea1b0aad44b5c6b96f81e249a/LICENSE). The separate proposed patch retains upstream context and has [its own scope/notice](patches/spirula/README.md). It is not linked into SphereAlign and is not installed into another repository.
- **Analytic and RGB-pattern fixtures:** Generated in this repository; they contain no third-party photos or user capture data. ONNX-generated small raw-head reference outputs are test evidence, not redistributed model weights or approved training priors.

Replacing a model, or adding another third-party model, library, program, or asset, does not change the license of SphereAlign-developed source code. Update this document with the applicable third-party source and license information whenever such a component is replaced or added. SphereAlign remains licensed under `AGPL-3.0-only` unless its copyright holders explicitly change that license.
