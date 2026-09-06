# Geometry Phase 0 evidence

See [measured results and commands](../../GEOMETRY_BENCHMARKS.md) and
[compatibility limits](../../GEOMETRY_COMPATIBILITY.md).

- `acquisition.json`: pinned source/model URLs, file sizes and SHA-256 receipts.
- `onnx-inspection.json`: real graph I/O, normalization, constants and operators.
- `directml*.json` / `*-stderr.txt`: native ORT results; original graph failures
  are retained alongside explicitly selected static experiments.
- `reference-*`: explicit CPU golden outputs and statistics. Small `.npz`
  fixtures contain raw heads; they are not native camera maps or trainer data.
- `parity*.json`: one synthetic pattern, no model correctness claim.
- `resources-native.json`: single-probe process memory; GPU readings include
  unrelated work. No full-stage VRAM bound.
- `spirula-*.txt`: actual installed binary help and geometry checks.
- `train-*.txt`: actual 3-step analytic fixture training, not a quality A/B result.
- `spirula-matrix/`: four more actual smoke runs and exact argv, including
  independent normal/depth switches and divisor=2/pinhole warp.
- `sentinel.json`: exact source sampler host reproduction and proposed patch
  checks; not a CUDA/Slang gradient test.
- `fixture.json`: analytic source-camera fixture provenance and source hashes.
- `fixture-preservation.json`: unchanged fixture hashes after the first two smoke
  runs; not full existing-pipeline off-mode regression.
- `cargo-test-lib.txt`, `cargo-check.txt`, `pnpm-build.txt`, `version-check.txt`:
  local build/regression evidence. Remote CI was not run.

Full downloaded models/sources, the 768² raw dumps and training checkpoints stay
under ignored `.work/geometry-spike`; no user's capture data is included here.
