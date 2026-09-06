"""Developer-only graph inspection. Install onnx separately; never used by app."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--dependencies", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies.resolve()))
    import onnx
    from onnx import numpy_helper
    model = onnx.load(args.model, load_external_data=False)
    onnx.checker.check_model(model)

    def tensor(v):
        t = v.type.tensor_type
        return {"name": v.name, "dtype": onnx.TensorProto.DataType.Name(t.elem_type),
                "shape": [d.dim_param if d.dim_param else d.dim_value for d in t.shape.dim]}

    constants = {}
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                t = attribute.t
                if len(t.raw_data) < 256:
                    constants[node.name] = numpy_helper.to_array(t).tolist()
    report = {
        "sha256": hashlib.file_digest(args.model.open("rb"), "sha256").hexdigest(),
        "bytes": args.model.stat().st_size,
        "irVersion": model.ir_version,
        "producer": [model.producer_name, model.producer_version],
        "opsets": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "inputs": [tensor(v) for v in model.graph.input],
        "outputs": [tensor(v) for v in model.graph.output],
        "externalTensors": [t.name for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL],
        "normalization": {t.name: numpy_helper.to_array(t).tolist() for t in model.graph.initializer if t.name in ["encoder.image_mean", "encoder.image_std"]},
        "operators": dict(Counter(n.op_type for n in model.graph.node)),
        "firstNodes": [{"name": n.name, "op": n.op_type, "in": list(n.input), "out": list(n.output)} for n in model.graph.node[:45]],
        "lastNodes": [{"name": n.name, "op": n.op_type, "in": list(n.input), "out": list(n.output)} for n in model.graph.node[-45:]],
        "smallConstants": constants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["smallConstants", "firstNodes", "lastNodes"]}, indent=2))


if __name__ == "__main__":
    main()
