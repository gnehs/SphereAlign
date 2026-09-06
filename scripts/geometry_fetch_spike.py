"""Developer-only, pinned source/artifact acquisition for Geometry Phase 0.

Uses only the standard library. No dataset or existing model is modified.
Downloaded third-party sources remain under the ignored .work directory.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

SPIRULA = "bddda193ee09f03ea1b0aad44b5c6b96f81e249a"
MODEL_REVISION = "2d247a122ada42ce700fe59273cd62e076da1f32"
MODEL_SHA256 = "bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21"
SPIRULA_FILES = [
    "LICENSE", "src/app/GeometryWarp.h", "src/app/TrainerCore.cpp",
    "src/moge/README.md", "src/moge/model/Fetch.cpp",
    "src/config/TrainConfig.h", "src/data/DataManager.cpp",
    "src/shaders/per_pixel_losses.slang", "src/app/GeometryWarp.cpp",
    "src/core/Interpolation.cuh", "src/moge/model/Recover.cpp",
    "src/data/DatasetParser.h",
]


def fetch(url, path, expected_hash=None, expected_size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    sha = digest.hexdigest()
    if expected_hash and (sha != expected_hash or size != expected_size):
        raise ValueError(f"artifact verification failed: {path}: {size}, {sha}")
    partial.replace(path)
    return {"url": url, "path": str(path), "bytes": size, "sha256": sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".work/geometry-spike"))
    parser.add_argument("--model", action="store_true", help="Download the 419 MB candidate; NOT a runtime registry approval")
    args = parser.parse_args()
    jobs = [(f"https://raw.githubusercontent.com/harry7557558/spirula-studio/{SPIRULA}/{f}", args.output / "spirula" / f) for f in SPIRULA_FILES]
    jobs += [("https://huggingface.co/api/models/Ruicheng/moge-2-vitb-normal-onnx", args.output / "model-repository.json")]
    moge = "74fbce054ebed49800de42d0ad0e83495065719a"
    jobs += [(f"https://raw.githubusercontent.com/microsoft/MoGe/{moge}/{f}", args.output / "moge" / f) for f in ["LICENSE", "moge/model/v2.py", "moge/utils/geometry_torch.py"]]
    jobs += [("https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/README.md", args.output / "original-model-card.md")]
    if args.model:
        jobs.append((f"https://huggingface.co/Ruicheng/moge-2-vitb-normal-onnx/resolve/{MODEL_REVISION}/model.onnx", args.output / "model.onnx", MODEL_SHA256, 419411850))
    def acquire(job):
        try:
            return fetch(*job)
        except Exception as error:
            return {"url": job[0], "path": str(job[1]), "error": str(error)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(acquire, jobs))
    receipt_path = args.output / "acquisition.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else []
    urls = {r["url"] for r in receipts}
    receipts += [r for r in previous if r["url"] not in urls]
    (args.output / "acquisition.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    print(json.dumps(receipts, indent=2))
    if any("error" in r for r in receipts if r["url"] in urls):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
