#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass
class LensStream:
    lens_id: str
    stream_index: int
    width: int
    height: int
    fps: float
    projection: str = "fisheye"


@dataclass
class CaptureDescriptor:
    vendor: str
    model: str
    source_path: str
    duration: float
    fps: float
    lens_count: int
    lenses: list[LensStream]
    capabilities: dict[str, bool]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def run_capture(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout


def frac(v: str | None, default: float = 0.0) -> float:
    if not v or v in {"0/0", "N/A"}:
        return default
    try:
        return float(Fraction(v))
    except Exception:
        try:
            return float(v)
        except Exception:
            return default


def ffprobe(path: Path) -> dict[str, Any]:
    return json.loads(run_capture([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]))


class DjiOsmo360Adapter:
    def __init__(self, path: Path, probe: dict[str, Any]):
        self.path = path
        self.probe = probe
        vids = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        if len(vids) < 2:
            raise RuntimeError("DJI Osmo 360 adapter expected at least two video streams")
        self.vids = vids[:2]

    @classmethod
    def can_open(cls, path: Path, probe: dict[str, Any]) -> bool:
        vids = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        data_tags = " ".join(str(s.get("codec_tag_string", "")) for s in probe.get("streams", []))
        return path.suffix.lower() == ".osv" and len(vids) >= 2 and ("djmd" in data_tags or "dbgi" in data_tags or True)

    def descriptor(self) -> CaptureDescriptor:
        duration = float(self.probe.get("format", {}).get("duration") or 0.0)
        lenses: list[LensStream] = []
        for i, s in enumerate(self.vids):
            fps = frac(s.get("avg_frame_rate"), frac(s.get("r_frame_rate"), 30.0))
            lenses.append(LensStream(
                lens_id=f"lens{i}", stream_index=int(s["index"]), width=int(s["width"]),
                height=int(s["height"]), fps=fps, projection="fisheye"))
        return CaptureDescriptor(
            vendor="DJI", model="Osmo 360", source_path=str(self.path), duration=duration,
            fps=lenses[0].fps, lens_count=len(lenses), lenses=lenses,
            capabilities={
                "native_fisheye": True,
                "imu": True,
                "fused_attitude": True,
                "factory_intrinsics": True,
                "rig_extrinsics": False,
            })

    def telemetry(self) -> tuple[Any | None, Any | None, str | None]:
        try:
            import telemetry_parser  # type: ignore
            tp = telemetry_parser.Parser(str(self.path))
            return tp.telemetry(), tp.normalized_imu(), None
        except Exception as e:
            return None, None, f"telemetry_parser unavailable or failed: {e}"


def choose_adapter(path: Path, probe: dict[str, Any], requested: str):
    if requested in {"auto", "dji_osmo360"} and DjiOsmo360Adapter.can_open(path, probe):
        return DjiOsmo360Adapter(path, probe)
    if requested in {"insta360", "gopro_max"}:
        raise NotImplementedError(
            f"Adapter {requested} is intentionally reserved by the contract but not implemented in v0.1")
    raise RuntimeError("No compatible CameraAdapter found")


def decode_proxy_metrics(path: Path, lens: LensStream, proxy_size: int) -> dict[str, list[float]]:
    # Preserve the square fisheye layout. Use a low-resolution grayscale proxy only for analysis.
    w = h = proxy_size
    vf = f"scale={w}:{h}:flags=area,format=gray"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-map", f"0:{lens.stream_index}", "-vf", vf,
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout is not None

    sharp: list[float] = []
    flow: list[float] = []
    scene: list[float] = []
    prev: np.ndarray | None = None
    prev_pts: np.ndarray | None = None
    frame_bytes = w * h

    while True:
        buf = p.stdout.read(frame_bytes)
        if len(buf) != frame_bytes:
            break
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        sharp.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))

        if prev is None:
            flow.append(0.0)
            scene.append(0.0)
            prev = img.copy()
            prev_pts = cv2.goodFeaturesToTrack(prev, maxCorners=350, qualityLevel=0.01, minDistance=6)
            continue

        # Histogram distance reacts to entering a new room / doorway without requiring a door detector.
        h1 = cv2.calcHist([prev], [0], None, [64], [0, 256])
        h2 = cv2.calcHist([img], [0], None, [64], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        scene.append(float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)))

        fmag = 0.0
        if prev_pts is not None and len(prev_pts) >= 12:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, img, prev_pts, None)
            if nxt is not None and status is not None:
                good_old = prev_pts[status.ravel() == 1].reshape(-1, 2)
                good_new = nxt[status.ravel() == 1].reshape(-1, 2)
                if len(good_new) >= 8:
                    d = np.linalg.norm(good_new - good_old, axis=1)
                    # robust median pixel motion, normalized to proxy diagonal
                    fmag = float(np.median(d) / math.sqrt(w * w + h * h))
        flow.append(fmag)
        prev = img.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev, maxCorners=350, qualityLevel=0.01, minDistance=6)

    p.stdout.close()
    rc = p.wait()
    if rc != 0:
        err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
        raise RuntimeError(f"ffmpeg proxy decode failed for {lens.lens_id}: {err}")
    return {"sharpness": sharp, "flow": flow, "scene": scene}


def robust_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.percentile(values, 25))
    hi = float(np.percentile(values, 90))
    if hi <= lo + 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def select_keyframes(metrics: list[dict[str, list[float]]], fps: float, cfg: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    n = min(len(m["sharpness"]) for m in metrics)
    if n == 0:
        raise RuntimeError("No video frames decoded")
    sharp = np.stack([np.asarray(m["sharpness"][:n], dtype=np.float64) for m in metrics])
    flow = np.max(np.stack([np.asarray(m["flow"][:n], dtype=np.float64) for m in metrics]), axis=0)
    scene = np.max(np.stack([np.asarray(m["scene"][:n], dtype=np.float64) for m in metrics]), axis=0)

    an = cfg["analysis"]
    blur = cfg["blur"]
    transition = (
        float(an.get("optical_flow_weight", 0.55)) * robust_norm(flow)
        + float(an.get("scene_change_weight", 0.30)) * robust_norm(scene)
    )
    # IMU contribution is added by an OrientationProvider in a future backend. Keep weights normalized for now.
    denom = max(1e-9, float(an.get("optical_flow_weight", 0.55)) + float(an.get("scene_change_weight", 0.30)))
    transition = np.clip(transition / denom, 0.0, 1.0)

    combined_sharp = np.max(sharp, axis=0)
    abs_floor = float(blur.get("absolute_floor", 12.0))
    pct = float(blur.get("reject_percentile", 12.0))
    sharp_floor = max(abs_floor, float(np.percentile(combined_sharp, pct))) if blur.get("enabled", True) else -1.0

    base_fps = max(0.1, float(an.get("base_fps", 2.0)))
    dense_fps = max(base_fps, float(an.get("dense_fps", 8.0)))
    max_gap = max(1, int(round(float(an.get("max_gap_seconds", 1.0)) * fps)))
    repair = int(an.get("repair_window_frames", 3))

    selected: list[int] = [0]
    i = 0
    while i < n - 1:
        score = float(transition[i])
        desired_fps = base_fps + (dense_fps - base_fps) * score
        step = max(1, int(round(fps / max(desired_fps, 0.1))))
        step = min(step, max_gap)
        target = min(n - 1, i + step)

        if blur.get("enabled", True) and combined_sharp[target] < sharp_floor:
            lo = max(i + 1, target - repair)
            hi = min(n - 1, target + repair)
            candidates = list(range(lo, hi + 1))
            if candidates:
                target = max(candidates, key=lambda k: combined_sharp[k])
        if target <= i:
            target = min(n - 1, i + 1)
        selected.append(target)
        i = target

    selected = sorted(set(selected))
    info = {
        "frame_count": n,
        "sharpness_floor": sharp_floor,
        "transition": transition.tolist(),
        "combined_sharpness": combined_sharp.tolist(),
        "flow": flow.tolist(),
        "scene": scene.tolist(),
        "per_lens_sharpness": sharp.tolist(),
    }
    return selected, info


def extract_selected_frames(path: Path, lens: LensStream, selected: list[int], out_dir: Path) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(selected)
    w, h = lens.width, lens.height
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-map", f"0:{lens.stream_index}", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout is not None
    frame_bytes = w * h * 3
    idx = 0
    written: dict[int, Path] = {}
    last_wanted = max(selected)
    while idx <= last_wanted:
        buf = p.stdout.read(frame_bytes)
        if len(buf) != frame_bytes:
            break
        if idx in wanted:
            # Keep all physical lenses for a selected timestamp. A textureless wall can have a low
            # Laplacian score without being motion-blurred, so never drop an individual lens solely
            # from sharpness. The keyframe selector repairs timestamps only when the multi-lens
            # sharpness signal is poor.
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            dst = out_dir / f"{idx:08d}.png"
            if not cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise RuntimeError(f"Failed to write {dst}")
            written[idx] = dst
        idx += 1
    p.stdout.close()
    p.terminate()
    p.wait(timeout=10)
    return written


def circle_valid_mask(h: int, w: int, ratio: float) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    r = int(min(w, h) * ratio)
    cv2.circle(m, (w // 2, h // 2), r, 255, -1, lineType=cv2.LINE_AA)
    return m


class TorchvisionMasker:
    def __init__(self, cfg: dict[str, Any]):
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
        self.torch = torch
        self.weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn_v2(weights=self.weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.categories = self.weights.meta["categories"]
        self.ignore = set(cfg.get("ignore_classes", []))
        self.conf = float(cfg.get("confidence", 0.55))
        self.max_size = int(cfg.get("inference_max_size", 1536))
        self.dilate_4k = int(cfg.get("dilate_px_at_4k", 18))

    def dynamic_ignore(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_size / max(h, w))
        small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ten = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        with self.torch.inference_mode():
            pred = self.model([ten])[0]
        ignore = np.zeros(rgb.shape[:2], dtype=np.uint8)
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        masks = pred["masks"].detach().cpu().numpy()[:, 0]
        for score, label, mask in zip(scores, labels, masks):
            if float(score) < self.conf:
                continue
            name = self.categories[int(label)] if int(label) < len(self.categories) else str(label)
            if name in self.ignore:
                ignore[mask >= 0.5] = 255
        if scale < 1:
            ignore = cv2.resize(ignore, (w, h), interpolation=cv2.INTER_NEAREST)
        dilate = max(1, round(self.dilate_4k * max(h, w) / 4096))
        k = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        return cv2.dilate(ignore, k)


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """Write one canonical 8-bit grayscale (L8) training/SfM mask.

    The mask contract is independent of the source image suffix: callers map
    ``images/lens0/frame.jpg`` to ``masks/lens0/frame.png``. ``0`` excludes a
    pixel and ``255`` keeps it.
    """
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional uint8 array")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("mask pixels must be black (0) or white (255)")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise RuntimeError(f"Failed to write mask {path}")


def build_masks(images: dict[str, dict[int, Path]], out_masks: Path,
                cfg: dict[str, Any], radius_ratio: float):
    mask_cfg = cfg["mask"]
    masker = None
    if mask_cfg.get("enabled", True) and mask_cfg.get("backend") == "torchvision_maskrcnn":
        print("Loading torchvision Mask R-CNN...")
        masker = TorchvisionMasker(mask_cfg)
    elif mask_cfg.get("backend") not in {"none", None, "external"}:
        raise ValueError(f"Unknown mask backend: {mask_cfg.get('backend')}")

    for lens_id, frames in images.items():
        for idx, path in frames.items():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Cannot load {path}")
            h, w = img.shape[:2]
            keep = circle_valid_mask(h, w, radius_ratio)
            if masker is not None:
                dyn = masker.dynamic_ignore(img)
                keep[dyn > 0] = 0

            # Canonical mask tree: preserve the image's relative stem and
            # always replace its suffix with .png (e.g. frame.jpg -> frame.png).
            # The same file is consumed by training and COLMAP; do not emit a
            # second compatibility tree or a double-extension variant.
            dst = out_masks / lens_id / f"{path.stem}.png"
            write_mask_png(dst, keep)


def generate_pairs(images: dict[str, dict[int, Path]], cfg: dict[str, Any], out_path: Path) -> int:
    mc = cfg["matching"]
    t = int(mc.get("temporal_neighbors", 5))
    c = int(mc.get("cross_lens_neighbors", 2))
    loop_every = int(mc.get("long_loop_every", 45))
    loop_radius = int(mc.get("long_loop_radius", 2))
    lenses = sorted(images)
    pairs: set[tuple[str, str]] = set()

    def rel(lid: str, idx: int) -> str:
        return f"{lid}/{idx:08d}.png"

    for lid in lenses:
        ids = sorted(images[lid])
        pos = {v: i for i, v in enumerate(ids)}
        for i, idx in enumerate(ids):
            for j in range(i + 1, min(len(ids), i + 1 + t)):
                a, b = rel(lid, idx), rel(lid, ids[j])
                pairs.add(tuple(sorted((a, b))))
        if loop_every > 0:
            for i in range(0, len(ids), loop_every):
                j0 = min(len(ids) - 1, i + loop_every)
                for dj in range(-loop_radius, loop_radius + 1):
                    j = j0 + dj
                    if 0 <= j < len(ids) and j != i:
                        pairs.add(tuple(sorted((rel(lid, ids[i]), rel(lid, ids[j])))))

    if len(lenses) >= 2:
        a, b = lenses[0], lenses[1]
        ids_a, ids_b = sorted(images[a]), sorted(images[b])
        set_b = set(ids_b)
        for idx in ids_a:
            # same timestamp and nearby selected source frames; useful around the fisheye overlap belt.
            cand = [x for x in ids_b if abs(x - idx) <= max(1, c * 8)]
            cand = sorted(cand, key=lambda x: abs(x - idx))[: 1 + 2 * c]
            for j in cand:
                pairs.add(tuple(sorted((rel(a, idx), rel(b, j)))))

    with out_path.open("w", encoding="utf-8") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    return len(pairs)


def write_rig_config(out_path: Path, lens_ids: list[str]):
    cams = []
    for i, lid in enumerate(lens_ids):
        entry: dict[str, Any] = {"image_prefix": f"{lid}/"}
        if i == 0:
            entry["ref_sensor"] = True
        cams.append(entry)
    out_path.write_text(json.dumps([{"cameras": cams}], indent=2), encoding="utf-8")


def run_colmap(out: Path, cfg: dict[str, Any], lens_ids: list[str]):
    if not shutil.which("colmap"):
        raise RuntimeError("COLMAP not found. Re-run with --no-sfm or install COLMAP.")
    db = out / "database.db"
    if db.exists():
        db.unlink()
    images = out / "images"
    masks = out / "masks"
    model = cfg["fisheye"].get("model", "OPENCV_FISHEYE")

    run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(images),
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.camera_model", str(model),
        "--ImageReader.mask_path", str(masks),
    ])

    pairs = out / "metadata" / "pairs.txt"
    run([
        "colmap", "matches_importer",
        "--database_path", str(db),
        "--match_list_path", str(pairs),
        "--match_type", "pairs",
    ])

    bootstrap = out / "sparse_bootstrap"
    shutil.rmtree(bootstrap, ignore_errors=True)
    bootstrap.mkdir(parents=True)
    run([
        "colmap", "mapper", "--database_path", str(db), "--image_path", str(images),
        "--output_path", str(bootstrap)
    ])
    boot0 = bootstrap / "0"
    if not boot0.exists():
        raise RuntimeError("COLMAP bootstrap produced no sparse/0 model")

    sfm_cfg = cfg["sfm"]
    final_sparse = out / "sparse"
    shutil.rmtree(final_sparse, ignore_errors=True)
    if sfm_cfg.get("two_pass_rig", True) and len(lens_ids) > 1:
        rig_cfg = out / "rig_config.json"
        rig_seed = out / "sparse_rig_seed"
        shutil.rmtree(rig_seed, ignore_errors=True)
        run([
            "colmap", "rig_configurator", "--database_path", str(db),
            "--input_path", str(boot0), "--rig_config_path", str(rig_cfg),
            "--output_path", str(rig_seed)
        ])
        final_sparse.mkdir(parents=True)
        cmd = [
            "colmap", "mapper", "--database_path", str(db), "--image_path", str(images),
            "--output_path", str(final_sparse),
        ]
        if sfm_cfg.get("fix_rig_after_bootstrap", True):
            cmd += ["--Mapper.ba_refine_sensor_from_rig", "0"]
        run(cmd)
    else:
        shutil.copytree(boot0, final_sparse / "0")


def write_keyframes_csv(path: Path, selected: list[int], fps: float, analysis: dict[str, Any]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_frame", "time_s", "sharpness", "flow", "scene_delta", "transition_score"])
        for i in selected:
            w.writerow([
                i, f"{i / fps:.6f}", f"{analysis['combined_sharpness'][i]:.6f}",
                f"{analysis['flow'][i]:.8f}", f"{analysis['scene'][i]:.8f}",
                f"{analysis['transition'][i]:.6f}",
            ])


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Camera-agnostic 360 capture -> COLMAP/3DGS dataset")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--no-sfm", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--analysis-only", action="store_true")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("FFmpeg/ffprobe are required")
    cfg = load_config(args.config)
    if args.no_mask:
        cfg["mask"]["enabled"] = False
        cfg["mask"]["backend"] = "none"
    if args.no_sfm:
        cfg["sfm"]["enabled"] = False

    inp = args.input.resolve()
    out = args.output.resolve()
    meta = out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)

    probe = ffprobe(inp)
    adapter = choose_adapter(inp, probe, cfg.get("input", {}).get("adapter", "auto"))
    desc = adapter.descriptor()
    (meta / "capture.json").write_text(json.dumps(asdict(desc), indent=2), encoding="utf-8")
    print(f"Detected: {desc.vendor} {desc.model}, {desc.lens_count} lenses @ {desc.fps:.3f} fps")

    tel, nimu, tel_err = adapter.telemetry()
    if tel is not None:
        (meta / "telemetry.json").write_text(json.dumps(tel, indent=2, default=str), encoding="utf-8")
    if nimu is not None:
        (meta / "normalized_imu.json").write_text(json.dumps(nimu, indent=2, default=str), encoding="utf-8")
    if tel_err:
        print("[WARN]", tel_err)

    metrics: list[dict[str, list[float]]] = []
    for lens in desc.lenses:
        print(f"Analyzing {lens.lens_id}...")
        metrics.append(decode_proxy_metrics(inp, lens, int(cfg["analysis"].get("proxy_size", 512))))
    selected, an = select_keyframes(metrics, desc.fps, cfg)
    write_keyframes_csv(meta / "keyframes.csv", selected, desc.fps, an)
    summary = {
        "source_frames": an["frame_count"], "selected_frames": len(selected),
        "selected_ratio": len(selected) / max(1, an["frame_count"]),
        "sharpness_floor": an["sharpness_floor"],
    }
    print("Keyframes:", json.dumps(summary, indent=2))
    if args.analysis_only:
        return

    images: dict[str, dict[int, Path]] = {}
    for li, lens in enumerate(desc.lenses):
        images[lens.lens_id] = extract_selected_frames(
            inp, lens, selected, out / "images" / lens.lens_id)
        print(f"{lens.lens_id}: wrote {len(images[lens.lens_id])} frames")

    build_masks(images, out / "masks", cfg,
                float(cfg["fisheye"].get("valid_radius_ratio", 0.497)))

    pair_count = generate_pairs(images, cfg, meta / "pairs.txt")
    print(f"Generated {pair_count} constrained image pairs")
    write_rig_config(out / "rig_config.json", sorted(images))

    if cfg["sfm"].get("enabled", True):
        run_colmap(out, cfg, sorted(images))

    report = {
        **summary,
        "pair_count": pair_count,
        "output_profile": cfg["pipeline"].get("output_profile", "lichtfeld"),
        "sfm_requested": bool(cfg["sfm"].get("enabled", True)),
        "telemetry_available": tel is not None,
        "normalized_imu_available": nimu is not None,
        "notes": [
            "Canonical images are raw physical fisheye lens frames; no panorama stitching was performed.",
            "IMU is preserved but stock COLMAP is not given a fake full quaternion prior.",
        ],
    }
    (meta / "pipeline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
