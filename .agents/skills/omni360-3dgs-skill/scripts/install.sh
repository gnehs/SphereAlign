#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install numpy opencv-python pyyaml telemetry-parser
printf '%s\n' 'Core Python dependencies installed.'
printf '%s\n' 'Optional mask backend: python -m pip install torch torchvision'
printf '%s\n' 'FFmpeg and COLMAP must also be installed and on PATH.'
