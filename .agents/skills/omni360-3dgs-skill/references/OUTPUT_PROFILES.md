# Output profiles

## native_fisheye

- 原始每顆 lens frame
- COLMAP fisheye camera model
- rig / frame relationship
- masks
- sparse reconstruction

適合能理解 distorted/fisheye camera model 的後端。

## lichtfeld

同樣保留 native lens images + COLMAP fisheye model。

LichtFeld Studio 可讀 COLMAP fisheye camera model，並有自己的 undistortion load path，因此 skill 不需要先 stitch panorama。

mask 放在與 `images/` 相同 relative hierarchy 的 `masks/`，但將來源副檔名
統一替換成 `.png`：`images/lens0/frame.jpg` → `masks/lens0/frame.png`。
Mask 固定為與來源同尺寸的 8-bit 單通道 L8 PNG；黑色（0）排除，白色（255）保留。
同一份 `masks/` 檔案供 COLMAP feature extraction 與 training 使用。

## pinhole_tiles

給只支援 pinhole 的 trainer：

- 每顆實體 fisheye 獨立產生 virtual perspective tiles
- 不跨 lens blend
- tile extrinsics = physical lens extrinsics * tile rotation
- tile mask 同步 reproject

這個模式仍有 resampling，但沒有雙鏡頭 stitching seam。

## equirectangular / cubemap

只作 legacy compatibility / visualization，不是 default SfM input。
