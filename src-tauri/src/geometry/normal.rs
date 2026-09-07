//! Lossy storage for a weak normal prior; geometry and validity maps stay unchanged.
use super::{
    dataset,
    draft::{check, guard_output, save, FrameReport},
    CancelToken,
};
use image::{codecs::jpeg::JpegEncoder, RgbImage};
use std::{fs, io::Write, path::Path};

pub(super) const JPEG_QUALITY: u8 = 90;
pub(super) const JPEG_FILE: &str = "normal.jpg";
pub(super) const PNG_FILE: &str = "normal.png";

pub(super) fn filename(frame: &FrameReport) -> Result<&'static str, String> {
    match (frame.files.contains_key(PNG_FILE), frame.files.contains_key(JPEG_FILE)) {
        (true, false) => Ok(PNG_FILE),
        (false, true) => Ok(JPEG_FILE),
        _ => Err("Normal manifest must contain exactly one PNG or JPEG".into()),
    }
}

pub(super) fn encode(image: &RgbImage) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
        .encode_image(image)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}

pub(super) fn write(path: &Path, image: &RgbImage) -> Result<(), String> {
    let bytes = encode(image)?;
    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}

/// Called only for a verified, source-matched cache under the dataset writer lock.
/// Commit the new file and frame manifest before deleting the verified PNG. The old
/// hash survives in the manifest so an interrupted deletion can be finished later.
pub(super) fn compress_cached(
    root: &Path,
    dir: &Path,
    frame: &mut FrameReport,
    cancel: &CancelToken,
) -> Result<(), String> {
    check(cancel)?;
    let png = dir.join(PNG_FILE);
    guard_output(root, &png)?;
    if filename(frame)? == PNG_FILE {
        let expected = frame.files[PNG_FILE].clone();
        if dataset::hash(&png)? != expected {
            return Err("Normal PNG changed before compression".into());
        }
        let image = image::open(&png).map_err(|e| e.to_string())?.to_rgb8();
        if image.dimensions() != (frame.width, frame.height) {
            return Err("Normal dimensions changed before compression".into());
        }
        let bytes = encode(&image)?;
        check(cancel)?;
        let jpg = dir.join(JPEG_FILE);
        let partial = dir.join("normal.jpg.partial");
        for path in [&jpg, &partial, &dir.join("frame.json"), &dir.join("frame.json.partial")] {
            guard_output(root, path)?;
        }
        // A matching JPEG may have been committed just before a previous crash.
        if jpg.exists() && fs::read(&jpg).map_err(|e| e.to_string())? != bytes {
            return Err("Unexpected existing normal.jpg; refusing to replace it".into());
        }
        let mut file = fs::File::create(&partial).map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        check(cancel)?;
        fs::rename(&partial, &jpg).map_err(|e| e.to_string())?;
        let mut next = frame.clone();
        next.files.remove(PNG_FILE);
        next.files.insert(JPEG_FILE.into(), dataset::hash(&jpg)?);
        next.normal_png_sha256 = Some(expected);
        save(&dir.join("frame.json"), &next)?;
        *frame = next;
    }
    if let Some(expected) = &frame.normal_png_sha256 {
        if png.exists() {
            check(cancel)?;
            if dataset::hash(&png)? != *expected {
                return Err("Obsolete normal PNG changed; preserving it".into());
            }
            fs::remove_file(png).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_444(bytes: &[u8]) {
        let mut i = 2;
        while i + 4 < bytes.len() {
            assert_eq!(bytes[i], 0xff);
            let marker = bytes[i + 1];
            let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
            if marker == 0xc0 {
                assert_eq!(bytes[i + 9], 3);
                for channel in 0..3 {
                    assert_eq!(bytes[i + 11 + channel * 3], 0x11, "JPEG must retain full chroma resolution");
                }
                return;
            }
            i += 2 + len;
        }
        panic!("Missing baseline JPEG frame header");
    }

    #[test]
    fn jpeg_keeps_resolution_and_full_chroma_sampling() {
        let source = RgbImage::from_fn(97, 131, |x, y| image::Rgb([x as u8, y as u8, 240]));
        let bytes = encode(&source).unwrap();
        assert_444(&bytes);
        let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), source.dimensions());
        let worst = source.as_raw().iter().zip(decoded.as_raw())
            .map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(worst <= 8, "Unexpected JPEG error on smooth normal channels: {worst}");
    }

    #[test]
    #[ignore = "explicit real-normal JPEG size/error benchmark; requires GEOMETRY_JPEG_BENCH_INPUTS"]
    fn real_normal_compression_benchmark() {
        let inputs = std::env::var("GEOMETRY_JPEG_BENCH_INPUTS").expect("semicolon-separated input PNGs");
        let mut results = Vec::new();
        for input in inputs.split(';') {
            let path = Path::new(input);
            let source_hash = dataset::hash(path).unwrap();
            let source = image::open(path).unwrap().to_rgb8();
            let png_bytes = fs::metadata(path).unwrap().len();
            for quality in [80, JPEG_QUALITY, 95] {
                let start = std::time::Instant::now();
                let mut bytes = Vec::new();
                JpegEncoder::new_with_quality(&mut bytes, quality).encode_image(&source).unwrap();
                let elapsed = start.elapsed().as_millis();
                assert_444(&bytes);
                let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
                let mut angles = Vec::new();
                let mut invalid = 0u64;
                let mut invalid_changed = 0u64;
                let mut invalid_above_16 = 0u64;
                for (i, (a, b)) in source.pixels().zip(decoded.pixels()).enumerate() {
                    if a.0 == [0; 3] {
                        invalid += 1;
                        invalid_changed += u64::from(b.0 != [0; 3]);
                        invalid_above_16 += u64::from(b.0.iter().any(|v| *v > 16));
                    } else if i % 16 == 0 {
                        let a = super::super::camera::unit(a.0.map(|v| 2. * v as f64 / 255. - 1.)).unwrap();
                        let b = super::super::camera::unit(b.0.map(|v| 2. * v as f64 / 255. - 1.)).unwrap();
                        angles.push(super::super::camera::dot(a, b).clamp(-1., 1.).acos().to_degrees());
                    }
                }
                angles.sort_by(f64::total_cmp);
                let percentile = |p: f64| angles[((angles.len() - 1) as f64 * p) as usize];
                assert!(!angles.is_empty());
                results.push(serde_json::json!({
                    "input":input,"sourceSha256":source_hash,"width":source.width(),"height":source.height(),
                    "pngBytes":png_bytes,"jpegQuality":quality,"jpegBytes":bytes.len(),
                    "reductionPercent":100. * (1. - bytes.len() as f64 / png_bytes as f64),
                    "encodeMs":elapsed,"chromaSampling":"4:4:4",
                    "angleSampleStride":16,"angleSamples":angles.len(),
                    "angleMedianDegrees":percentile(0.5),"angleP95Degrees":percentile(0.95),"angleP99Degrees":percentile(0.99),
                    "invalidBlackPixels":invalid,"invalidBlackPixelsChanged":invalid_changed,"invalidPixelsWithChannelAbove16":invalid_above_16
                }));
            }
            assert_eq!(dataset::hash(path).unwrap(), source_hash);
        }
        println!("JPEG_BENCH {}", serde_json::json!({"results":results,"scope":"image encoding only; angle errors relative to existing quantized normal PNGs, not geometry truth or training quality"}));
    }
}
