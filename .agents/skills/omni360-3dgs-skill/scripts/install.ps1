$ErrorActionPreference = "Stop"
py -m pip install --upgrade pip
py -m pip install numpy opencv-python pyyaml telemetry-parser
Write-Host "Core Python dependencies installed."
Write-Host "For automatic person/car segmentation:"
Write-Host "  py -m pip install torch torchvision"
Write-Host "Also install FFmpeg and COLMAP and add them to PATH."
