#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import shutil
import subprocess
import sys


def version(cmd):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return (p.stdout or '').splitlines()[0][:160]
    except Exception as e:
        return str(e)

checks = [
    ("ffmpeg", shutil.which("ffmpeg"), ["ffmpeg", "-version"]),
    ("ffprobe", shutil.which("ffprobe"), ["ffprobe", "-version"]),
    ("COLMAP", shutil.which("colmap"), ["colmap", "-h"]),
]

print("Omni360 -> 3DGS doctor\n")
ok = True
for name, path, cmd in checks:
    if path:
        print(f"[OK] {name}: {path}")
        print(f"     {version(cmd)}")
    else:
        level = "WARN" if name == "COLMAP" else "FAIL"
        print(f"[{level}] {name}: not found")
        if name != "COLMAP": ok = False

for mod in ["numpy", "cv2", "yaml"]:
    found = importlib.util.find_spec(mod) is not None
    print(f"[{'OK' if found else 'FAIL'}] Python {mod}")
    ok &= found

for mod in ["telemetry_parser", "torch", "torchvision"]:
    found = importlib.util.find_spec(mod) is not None
    print(f"[{'OK' if found else 'WARN'}] optional {mod}")

print("\ntelemetry_parser: 建議安裝，用於 DJI/Insta360 IMU metadata。")
print("torch/torchvision: 只有 mask.backend=torchvision_maskrcnn 時需要。")
print("COLMAP: --no-sfm 可先只做資料整理。")
sys.exit(0 if ok else 2)
