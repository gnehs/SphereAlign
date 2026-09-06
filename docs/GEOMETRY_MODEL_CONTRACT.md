# Geometry model contract — candidate v1

Status: **Phase 0 evidence, no production-certified geometry model**. Measured 2026-09-06; ordinary model preparation and GPU probing are Rust + the existing `ort` runtime. Python is used only by development reference/inspection tools.

## Immutable identities

Source: [Ruicheng/moge-2-vitb-normal-onnx at revision 2d247a1](https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/tree/2d247a122ada42ce700fe59273cd62e076da1f32).

| Artifact/profile | Bytes | SHA-256 |
| --- | ---: | --- |
| Original `moge2-vitb-normal` | 419411850 | `bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21` |
| `static768-t1800-experiment-v1`, Python ONNX shape inference | 419586451 | `0b33aa5249fdea9338b94276adbd700476cfde2aa0e5af24a8f26f1e39876ecf` |
| `static768-t1800-native-v1`, Rust lossless protobuf specialization | 419411835 | `e3a6e567a98cf8dfbbdc113fa59e38f9b31c1cd1aff1fcbd316a1438388ec4ac` |

Both derivatives fix image shape to `[1,3,768,768]`, remove the `num_tokens` graph input and add scalar INT64 initializer 1800. They preserve weights and computation. Rust output is byte-identical to Python ONNX 1.19.1 specialization **without** shape inference. The original artifact is never overwritten, and the candidate model ID changes. Derivatives have no published download URL and are prepared locally from the fixed source; no alternative arbitrary ONNX is accepted.

## Inspected graph

Actual ONNX checker result and graph evidence: [onnx-inspection.json](evidence/geometry-2026-09-06/onnx-inspection.json). IR version 7, default-domain opset 14, producer PyTorch 2.7.0; no external tensor files. The original has dynamic image batch/height/width and scalar dynamic token input. There is an `If` node and extensive dynamic shape/token arithmetic; dynamic dimensions are not a provider support certificate.

| Name | dtype | Actual layout / shape |
| --- | --- | --- |
| `image` input | FLOAT | NCHW `[batch_size,3,height,width]`, RGB values in 0..1 |
| `num_tokens` input | INT64 | scalar `[]`; omitted from static profile interface |
| `points` output | FLOAT | NHWC `[N,H,W,3]`, raw affine points after remapping |
| `normal` output | FLOAT | NHWC `[N,H,W,3]`, normalized vector head |
| `mask` output | FLOAT | `[N,H,W]`, sigmoid already applied |
| `metric_scale` output | FLOAT | `[N]`, exponential already applied |

The graph resizes internally according to token budget and resizes heads back to input dimensions. It includes ImageNet subtraction/division with mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`. Do **not** apply that normalization a second time. The graph's points remap is sinh(xy) / exp(z). It has no known-focal recovery, camera K input or application of the scale head to recovered points. Metric scale is a model estimate, never a measured unit certificate. `mask >= 0.5` is a proposed validity threshold, not calibrated probability of geometric correctness.

The source graph and [pinned MoGe v2 implementation](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v2.py) separate `forward` raw heads from `infer` recovery. Raw `points.z` must not be exported as ray range. The Rust draft engine now uses known face K with a robust shift estimate/refinement, rejects nonpositive/nonfinite recovered z, then converts to Euclidean ray range and applies the estimated scale head. The probe still exports **no trainer maps**; draft results are also isolated from training.

## Real execution evidence

Original graph fails DirectML session initialization with CPU fallback disabled: dynamic shapes, 32×32/4 tokens, 96×64/24 tokens and 768×768/1800 tokens. Free-dimension overrides alone do not fix dynamic token arithmetic. This is not an OOM result.

The separate fixed-token artifacts execute on DirectML with fallback disabled. Rust uses existing provider registration and one session, sequential execution, disabled DirectML memory-pattern optimization. Runtime build is ORT `rel-1.28.0`, commit `da9b5e3`, via `ort = 2.0.0-rc.13`. Python CPU golden uses ORT 1.23.2; the version difference is recorded, not hidden.

Original CPU fixtures exercised 32×32, 96×64 and 768×768 with different token counts. Native static GPU was compared against the original CPU graph on the same 768×768 synthetic RGB pattern: `channel-major byte[i] = (17*i+31) mod 256`, divided by 255 once. [Native parity report](evidence/geometry-2026-09-06/parity-native-768.json): points relative L2 `2.0848e-6`, normal P99 angle `0.00353°`, max `0.00679°`, validity threshold disagreement 0. **No pixels were near the validity threshold**; that part of parity coverage is still absent. This is a single-fixture observation, not a calibrated release tolerance or geometric accuracy result.

`geometry_compare_heads.py` also compares an independent known-focal diagnostic shift at fx=fy=320, in f64, on matched heads. The pattern does not depict a meaningful calibrated scene; matching recovered values only checks numeric consistency. Runtime native-fisheye recovery is implemented with an analytic plane test; model normal sign and actual-scene accuracy remain unverified.

Raw development dumps use explicit `.f32le` row-major float32 bytes; shapes are in probe JSON. They are not `.npy`, PNGs, native fisheye output or a committed geometry run. Small CPU golden heads are saved in standard NumPy `.npz`, with no pickle.

## License/provenance limits

The original [PyTorch model card](https://huggingface.co/Ruicheng/moge-2-vitb-normal/blob/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/README.md) declares MIT; [MoGe LICENSE](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/LICENSE) contains its notices. The ONNX repository at the pinned revision contains only `.gitattributes` and `model.onnx`, with **no independent model card/LICENSE**. Upstream Spirula describes it as the same weights; exact equivalence to the PyTorch checkpoint was not verified here. Record this missing artifact-specific notice before publishing a model registry/download promise; do not invent one. No weights or full ONNX graph are added to Git by this change.
