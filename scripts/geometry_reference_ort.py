"""Explicit CPU golden reference, developer-only; never a GPU fallback.

Synthetic RGB pattern is identical to the native probe and contains no user data.
Raw heads are diagnostic affine points, NOT recovered geometry or trainer input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--dependencies", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--tokens", type=int, default=4)
    args = p.parse_args()
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies.resolve()))
    import numpy as np
    import onnxruntime as ort
    from geometry_fetch_spike import MODEL_SHA256
    with args.model.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("pinned model hash mismatch")
    if not (14 <= args.width <= 1064 and 14 <= args.height <= 1064 and 1 <= args.tokens <= 3600):
        raise ValueError("shape/token budget exceeds bounded spike limits")
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    image = (((np.arange(3 * args.height * args.width) * 17 + 31) % 256).astype(np.float32) / 255).reshape(1, 3, args.height, args.width)
    inputs = {"image": image, "num_tokens": np.array(args.tokens, dtype=np.int64)}
    start = time.perf_counter()
    heads = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
    report = {"purpose": "explicit_cpu_reference_not_gpu_parity", "modelSha256": digest,
              "ortVersion": ort.__version__, "provider": "CPUExecutionProvider", "width": args.width,
              "height": args.height, "tokens": args.tokens, "inferenceSeconds": time.perf_counter() - start,
              "inputSha256": hashlib.sha256(image.tobytes()).hexdigest(), "outputs": {}}
    for name, value in heads.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} has nonfinite values")
        report["outputs"][name] = {"shape": list(value.shape), "min": float(value.min()), "max": float(value.max()),
                                   "mean": float(value.mean()), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    report["normalLengthMaxError"] = float(np.max(np.abs(np.linalg.norm(heads["normal"], axis=-1) - 1)))
    report["validityAboveHalfFraction"] = float((heads["mask"] >= .5).mean())
    np.savez_compressed(args.output / "raw-heads.npz", **heads)
    (args.output / "reference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
