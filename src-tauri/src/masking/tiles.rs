//! Calibrated perspective inference, gathered back into canonical fisheye pixels.
//! Pixel coordinates follow COLMAP's corner-based convention. Tile borders are
//! rectangular; a fisheye aperture must never be applied to a perspective tile.

use super::{CancelToken, MaskEngine, MaskError, MaskRequest, MaskResult, SegmentationMask};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const TILE_SIZE: u32 = 640;
pub const TILE_FOV_DEGREES: f64 = 100.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FisheyeCamera {
    pub width: u32,
    pub height: u32,
    /// OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4.
    pub params: [f64; 8],
}

impl FisheyeCamera {
    pub fn validate(&self) -> MaskResult<()> {
        if self.width == 0 || self.height == 0 || self.params.iter().any(|v| !v.is_finite())
            || self.params[0] <= 0.0 || self.params[1] <= 0.0
        {
            return Err(MaskError::invalid_input("invalid OPENCV_FISHEYE calibration"));
        }
        // Reject a folded radial mapping instead of choosing an arbitrary root.
        for step in 0..=256 {
            let theta = std::f64::consts::FRAC_PI_2 * step as f64 / 256.0;
            let t2 = theta * theta;
            let derivative = 1.0 + t2 * (3.0 * self.params[4] + t2 *
                (5.0 * self.params[5] + t2 * (7.0 * self.params[6] + t2 * 9.0 * self.params[7])));
            if !derivative.is_finite() || derivative <= 0.0 {
                return Err(MaskError::invalid_input("fisheye calibration folds within the forward hemisphere"));
            }
        }
        Ok(())
    }

    fn distorted_theta(&self, theta: f64) -> f64 {
        let t2 = theta * theta;
        theta * (1.0 + t2 * (self.params[4] + t2 * (self.params[5] + t2 *
            (self.params[6] + t2 * self.params[7]))))
    }

    pub fn project(&self, ray: [f64; 3]) -> Option<[f64; 2]> {
        if ray.iter().any(|v| !v.is_finite()) || ray[2] <= 0.0 { return None; }
        let radius = ray[0].hypot(ray[1]);
        let factor = if radius < 1e-12 { 1.0 / ray[2] }
            else { self.distorted_theta(radius.atan2(ray[2])) / radius };
        let uv = [self.params[0] * ray[0] * factor + self.params[2],
            self.params[1] * ray[1] * factor + self.params[3]];
        uv.iter().all(|v| v.is_finite()).then_some(uv)
    }

    pub fn unproject(&self, uv: [f64; 2]) -> Option<[f64; 3]> {
        let x = (uv[0] - self.params[2]) / self.params[0];
        let y = (uv[1] - self.params[3]) / self.params[1];
        let radius = x.hypot(y);
        if !radius.is_finite() { return None; }
        if radius < 1e-12 { return Some([0.0, 0.0, 1.0]); }
        let mut lower = 0.0;
        let mut upper = std::f64::consts::FRAC_PI_2 - 1e-8;
        if radius > self.distorted_theta(upper) { return None; }
        let mut theta = radius.min(upper);
        for _ in 0..40 {
            let residual = self.distorted_theta(theta) - radius;
            if residual.abs() < 1e-11 { break; }
            if residual < 0.0 { lower = theta; } else { upper = theta; }
            let t2 = theta * theta;
            let derivative = 1.0 + t2 * (3.0 * self.params[4] + t2 *
                (5.0 * self.params[5] + t2 * (7.0 * self.params[6] + t2 * 9.0 * self.params[7])));
            let next = theta - residual / derivative;
            theta = if next > lower && next < upper { next } else { (lower + upper) * 0.5 };
        }
        let scale = theta.sin() / radius;
        Some([x * scale, y * scale, theta.cos()])
    }
}

/// Calibration is bound to the physical lens AND source. Camera ids are not
/// assumed to be 1/2, and a different recording cannot inherit a camera by name.
pub fn camera_group(name: &str) -> Option<String> {
    let (lens, file) = name.rsplit_once('/')?;
    let (source, _) = file.rsplit_once('_')?;
    if !matches!(lens, "lens0" | "lens1") || !source.starts_with("source") { return None; }
    Some(format!("{lens}/{source}"))
}

pub fn read_calibration(model: &Path) -> MaskResult<BTreeMap<String, FisheyeCamera>> {
    let mut cameras = BTreeMap::new();
    for line in BufReader::new(File::open(model.join("cameras.txt"))?).lines() {
        let line = line?;
        if line.trim().is_empty() || line.trim_start().starts_with('#') { continue; }
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 12 || parts[1] != "OPENCV_FISHEYE" {
            return Err(MaskError::invalid_input("calibrated masks require OPENCV_FISHEYE cameras.txt"));
        }
        let parse_error = |_| MaskError::invalid_input("malformed cameras.txt calibration");
        let id: u64 = parts[0].parse().map_err(parse_error)?;
        let mut params = [0.0; 8];
        for (value, text) in params.iter_mut().zip(&parts[4..]) {
            *value = text.parse().map_err(|_| MaskError::invalid_input("malformed camera parameter"))?;
        }
        let camera = FisheyeCamera {
            width: parts[2].parse().map_err(parse_error)?,
            height: parts[3].parse().map_err(parse_error)?, params,
        };
        camera.validate()?;
        if cameras.insert(id, camera).is_some() {
            return Err(MaskError::invalid_input("duplicate calibration camera id"));
        }
    }
    let mut result = BTreeMap::new();
    let mut header = true;
    for line in BufReader::new(File::open(model.join("images.txt"))?).lines() {
        let line = line?;
        if line.trim_start().starts_with('#') { continue; }
        if !header { header = true; continue; }
        if line.trim().is_empty() { continue; }
        header = false;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() != 10 { return Err(MaskError::invalid_input("malformed calibration images.txt")); }
        let id: u64 = parts[8].parse().map_err(|_| MaskError::invalid_input("invalid calibration camera id"))?;
        let name = parts[9].replace('\\', "/");
        let group = camera_group(&name).ok_or_else(|| MaskError::invalid_input(format!("unbound calibration image: {name}")))?;
        let camera = cameras.get(&id).ok_or_else(|| MaskError::invalid_input("calibration image references missing camera"))?;
        if result.get(&group).is_some_and(|previous| previous != camera) {
            return Err(MaskError::invalid_input(format!("multiple calibrations for {group}")));
        }
        result.insert(group, camera.clone());
    }
    if result.is_empty() { return Err(MaskError::invalid_input("calibration model contains no camera bindings")); }
    Ok(result)
}

fn rotate(ray: [f64; 3], view: usize, inverse: bool) -> [f64; 3] {
    if view == 0 { return ray; }
    let angle = (if view % 2 == 1 { -60.0_f64 } else { 60.0 }).to_radians()
        * if inverse { -1.0 } else { 1.0 };
    let (s, c) = angle.sin_cos();
    if view <= 2 { [ray[0], c * ray[1] - s * ray[2], s * ray[1] + c * ray[2]] }
    else { [c * ray[0] + s * ray[2], ray[1], -s * ray[0] + c * ray[2]] }
}

fn focal(size: u32) -> f64 { size as f64 / (2.0 * (TILE_FOV_DEGREES.to_radians() * 0.5).tan()) }

fn tile_index(ray: [f64; 3], view: usize, size: u32) -> Option<usize> {
    let local = rotate(ray, view, true);
    if local[2] <= 0.0 { return None; }
    let x = local[0] / local[2] * focal(size) + size as f64 / 2.0;
    let y = local[1] / local[2] * focal(size) + size as f64 / 2.0;
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x >= size as f64 || y >= size as f64 { return None; }
    Some(y.floor() as usize * size as usize + x.floor() as usize)
}

fn render_view(rgb: &RgbImage, camera: &FisheyeCamera, view: usize) -> DynamicImage {
    let mut output = RgbImage::new(TILE_SIZE, TILE_SIZE);
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let ray = rotate([(x as f64 + 0.5 - TILE_SIZE as f64 / 2.0) / focal(TILE_SIZE),
            (y as f64 + 0.5 - TILE_SIZE as f64 / 2.0) / focal(TILE_SIZE), 1.0], view, false);
        let Some([u, v]) = camera.project(ray) else { continue; };
        let (u, v) = (u - 0.5, v - 0.5);
        if u < 0.0 || v < 0.0 || u > (rgb.width() - 1) as f64 || v > (rgb.height() - 1) as f64 { continue; }
        let (x0, y0) = (u.floor() as u32, v.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(rgb.width() - 1), (y0 + 1).min(rgb.height() - 1));
        let (wx, wy) = (u - x0 as f64, v - y0 as f64);
        *pixel = Rgb(std::array::from_fn(|c| ((1.0 - wy) * ((1.0 - wx) * rgb.get_pixel(x0, y0)[c] as f64
            + wx * rgb.get_pixel(x1, y0)[c] as f64) + wy * ((1.0 - wx) * rgb.get_pixel(x0, y1)[c] as f64
            + wx * rgb.get_pixel(x1, y1)[c] as f64)).round() as u8));
    }
    DynamicImage::ImageRgb8(output)
}

/// Gather each native pixel from all overlapping tiles. Uncovered pixels keep
/// their original optical/manual mask; no sparse forward splatting or holes.
#[cfg(test)]
fn gather(camera: &FisheyeCamera, masks: &[SegmentationMask], cancel: &CancelToken) -> MaskResult<SegmentationMask> {
    let mut data = vec![0; camera.width as usize * camera.height as usize];
    for y in 0..camera.height {
        if cancel.is_cancelled() { return Err(MaskError::Cancelled); }
        for x in 0..camera.width {
            let Some(ray) = camera.unproject([x as f64 + 0.5, y as f64 + 0.5]) else { continue; };
            if masks.iter().enumerate().any(|(view, mask)|
                tile_index(ray, view, mask.width).is_some_and(|i| mask.data[i] != 0)) {
                data[y as usize * camera.width as usize + x as usize] = 255;
            }
        }
    }
    SegmentationMask::new(camera.width, camera.height, data)
}

/// Reuse the expensive ray inversion across a batch. Retain at most two camera
/// maps (about 560 MiB for two 3840² lenses); equal calibrations share a map
/// across capture sources. Native images themselves are never cached here.
type NativeMap = Vec<[u32; 5]>;
#[derive(Default)]
pub(super) struct ProjectionCache(Mutex<Vec<(FisheyeCamera, Arc<NativeMap>)>>);

impl ProjectionCache {
    fn get(&self, camera: &FisheyeCamera, cancel: &CancelToken) -> MaskResult<Arc<NativeMap>> {
        let mut cache = self.0.lock().map_err(|_| MaskError::inference("projection cache lock failed"))?;
        if let Some((_, map)) = cache.iter().find(|(key, _)| key == camera) { return Ok(map.clone()); }
        let count = (camera.width as usize).checked_mul(camera.height as usize)
            .ok_or_else(|| MaskError::invalid_input("camera dimensions overflow"))?;
        let mut map = Vec::new();
        map.try_reserve_exact(count).map_err(|_| MaskError::inference("not enough memory for calibrated native mask lookup"))?;
        for y in 0..camera.height {
            if cancel.is_cancelled() { return Err(MaskError::Cancelled); }
            for x in 0..camera.width {
                let indices = camera.unproject([x as f64 + 0.5, y as f64 + 0.5]).map_or([u32::MAX; 5], |ray|
                    std::array::from_fn(|view| tile_index(ray, view, TILE_SIZE).map_or(u32::MAX, |i| i as u32)));
                map.push(indices);
            }
        }
        let map = Arc::new(map);
        if cache.len() >= 2 { cache.remove(0); }
        cache.push((camera.clone(), map.clone()));
        Ok(map)
    }
}

pub(super) fn infer<E: MaskEngine + ?Sized>(image: &DynamicImage, camera: &FisheyeCamera,
    request: &MaskRequest, engine: &E, cancel: &CancelToken, cache: &ProjectionCache) -> MaskResult<SegmentationMask> {
    if image.dimensions() != (camera.width, camera.height) {
        return Err(MaskError::invalid_input("calibration dimensions differ from the native image; do not rescale or guess intrinsics"));
    }
    let rgb = image.to_rgb8();
    let mut masks = Vec::with_capacity(5);
    for view in 0..5 {
        if cancel.is_cancelled() { return Err(MaskError::Cancelled); }
        let tile = render_view(&rgb, camera, view);
        let mut mask = super::infer_rotations(engine, &tile, request, false, cancel)?;
        super::dilate_exclusions(&mut mask, request.dilation);
        masks.push(mask);
    }
    let map = cache.get(camera, cancel)?;
    let mut data = vec![0; map.len()];
    for (row, indices) in map.chunks(camera.width as usize).enumerate() {
        if cancel.is_cancelled() { return Err(MaskError::Cancelled); }
        for (column, views) in indices.iter().enumerate() {
            if views.iter().enumerate().any(|(view, index)| *index != u32::MAX && masks[view].data[*index as usize] != 0) {
                data[row * camera.width as usize + column] = 255;
            }
        }
    }
    let mut result = SegmentationMask::new(camera.width, camera.height, data)?;
    // SkySeg has an upright-image prior; evaluate it once in the native view.
    if request.mask_sky {
        let working = super::resize_for_mask_working_resolution(image.clone());
        let sky = engine.generate_exclusion_mask(&working, &[], request.confidence, true, cancel)?;
        super::validate_segmentation(&sky, working.dimensions())?;
        let sky = super::resize_binary_mask(sky.data, sky.width, sky.height, camera.width, camera.height)?;
        for (pixel, sky) in result.data.iter_mut().zip(sky) { *pixel |= sky; }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> FisheyeCamera { FisheyeCamera { width: 32, height: 32, params: [10.0, 10.0, 16.0, 16.0, 0.01, -0.002, 0.0, 0.0] } }

    #[test]
    fn calibrated_pixels_round_trip_and_reject_behind_camera() {
        let camera = camera();
        camera.validate().unwrap();
        for uv in [[16.0, 16.0], [8.5, 12.5], [27.5, 16.5]] {
            let actual = camera.project(camera.unproject(uv).unwrap()).unwrap();
            assert!((actual[0] - uv[0]).abs() < 1e-7 && (actual[1] - uv[1]).abs() < 1e-7);
        }
        assert!(camera.project([0.0, 0.0, -1.0]).is_none());
    }

    #[test]
    fn inverse_gather_preserves_blank_masks_and_fills_every_mapped_pixel() {
        let camera = camera();
        let mut masks = vec![SegmentationMask::new(16, 16, vec![0; 256]).unwrap(); 5];
        let cancel = CancelToken::new();
        assert!(gather(&camera, &masks, &cancel).unwrap().data.iter().all(|v| *v == 0));
        masks[0].data[8 * 16 + 8] = 255;
        let result = gather(&camera, &masks, &cancel).unwrap();
        let mut excluded = 0;
        for y in 0..camera.height { for x in 0..camera.width {
            let expected = camera.unproject([x as f64 + 0.5, y as f64 + 0.5])
                .and_then(|ray| tile_index(ray, 0, 16)) == Some(8 * 16 + 8);
            assert_eq!(result.data[(y * camera.width + x) as usize] != 0, expected);
            excluded += usize::from(expected);
        }}
        assert!(excluded > 0);
        // The square corners are valid tile pixels, not a circular aperture.
        assert!(tile_index([-1.0, -1.0, 1.0], 0, 16).is_some());
    }

    #[test]
    fn cached_mapping_matches_reference_for_all_views() {
        let camera = camera();
        let cancel = CancelToken::new();
        let cache = ProjectionCache::default();
        let map = cache.get(&camera, &cancel).unwrap();
        assert!(Arc::ptr_eq(&map, &cache.get(&camera, &cancel).unwrap()));
        let masks: Vec<_> = (0..5).map(|view| SegmentationMask::new(TILE_SIZE, TILE_SIZE,
            (0..TILE_SIZE*TILE_SIZE).map(|i| if (i + view * 73) % 127 < 32 { 255 } else { 0 }).collect()).unwrap()).collect();
        let reference = gather(&camera, &masks, &cancel).unwrap();
        for (indices, expected) in map.iter().zip(reference.data) {
            let excluded = indices.iter().enumerate().any(|(view,index)| *index != u32::MAX && masks[view].data[*index as usize] != 0);
            assert_eq!(excluded, expected != 0);
        }
    }

    #[test]
    fn calibration_groups_do_not_cross_sources_or_lenses() {
        assert_eq!(camera_group("lens1/source004_00000123.jpg"), Some("lens1/source004".into()));
        assert!(camera_group("source004_00000123.jpg").is_none());
        let mut folded = camera(); folded.params[4] = -1.0;
        assert!(folded.validate().is_err());
    }
}
