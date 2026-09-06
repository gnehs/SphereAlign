"""Diagnostic static graph experiment; produces a NEW candidate, never overwrites stock.

This is not an approved runtime model. Hashes and profiles cannot be substituted
for the pinned stock identity. Useful to isolate dynamic shape/token EP blockers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--dependencies", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--tokens", type=int, default=1800)
    p.add_argument("--skip-shape-inference", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.dependencies.resolve()))
    import onnx
    import numpy as np
    from geometry_fetch_spike import MODEL_SHA256
    with args.model.open("rb") as f:
        source_hash = hashlib.file_digest(f, "sha256").hexdigest()
    if source_hash != MODEL_SHA256:
        raise ValueError("source hash mismatch")
    if args.output.exists() or args.output.resolve() == args.model.resolve():
        raise ValueError("output must be new; never replace an artifact")
    if not (14 <= args.width <= 1064 and 14 <= args.height <= 1064 and 1 <= args.tokens <= 3600):
        raise ValueError("out of bounded shape/token limits")
    model = onnx.load(args.model)
    image = next(v for v in model.graph.input if v.name == "image")
    for dimension, value in zip(image.type.tensor_type.shape.dim, [1,3,args.height,args.width]):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    token = next(v for v in model.graph.input if v.name == "num_tokens")
    model.graph.input.remove(token)
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array(args.tokens, dtype=np.int64), "num_tokens"))
    if not args.skip_shape_inference:
        model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    with args.output.open("rb") as f:
        derived_hash = hashlib.file_digest(f,"sha256").hexdigest()
    report = {"status":"diagnostic_not_registered", "sourceSha256":source_hash,"derivedSha256":derived_hash,
              "width":args.width,"height":args.height,"tokens":args.tokens,"onnxVersion":onnx.__version__,"bytes":args.output.stat().st_size,"shapeInference":not args.skip_shape_inference}
    args.output.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
