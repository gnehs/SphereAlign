//! Lightweight source thumbnails for the task picker.

use crate::doctor;
use crate::process::silent_command;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const PREVIEW_FILTER: &str =
    "scale=w=480:h=270:force_original_aspect_ratio=decrease:force_divisible_by=2";
static PREVIEW_LOCK: Mutex<()> = Mutex::new(());

fn preview_args(path: &Path) -> Vec<String> {
    vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-i".to_owned(),
        path.to_string_lossy().into_owned(),
        "-map".to_owned(),
        "0:v:0".to_owned(),
        "-frames:v".to_owned(),
        "1".to_owned(),
        "-vf".to_owned(),
        PREVIEW_FILTER.to_owned(),
        "-an".to_owned(),
        "-c:v".to_owned(),
        "mjpeg".to_owned(),
        "-f".to_owned(),
        "image2pipe".to_owned(),
        "pipe:1".to_owned(),
    ]
}

fn validate_source(path: &Path) -> Result<PathBuf, String> {
    if !path.is_file() {
        return Err("找不到來源檔案".to_owned());
    }
    let supported = matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("osv")
            | Some("mp4")
            | Some("mov")
            | Some("mkv")
            | Some("avi")
            | Some("webm")
            | Some("m4v")
            | Some("mts")
            | Some("m2ts")
            | Some("ts")
    );
    if !supported {
        return Err("不支援此來源格式".to_owned());
    }
    Ok(path.to_path_buf())
}

pub fn extract_first_frame(path: String) -> Result<Vec<u8>, String> {
    let path = validate_source(Path::new(&path))?;
    // Source pickers can add many large captures at once. Decode previews one
    // at a time so thumbnail generation cannot saturate the machine.
    let _guard = PREVIEW_LOCK
        .lock()
        .map_err(|_| "預覽服務暫時無法使用".to_owned())?;
    let ffmpeg = doctor::find_executable("ffmpeg").ok_or_else(|| "尚未安裝 FFmpeg".to_owned())?;
    let output = silent_command(ffmpeg)
        .args(preview_args(&path))
        .output()
        .map_err(|_| "無法啟動 FFmpeg".to_owned())?;

    if !output.status.success() || output.stdout.is_empty() {
        // FFmpeg diagnostics can repeat the user's full local path. Keep IPC
        // errors generic so private source locations never reach the webview.
        return Err("無法讀取影片第一幀".to_owned());
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_selects_only_the_first_video_stream_and_one_frame() {
        let args = preview_args(Path::new("capture.OSV"));
        assert!(args.windows(2).any(|pair| pair == ["-map", "0:v:0"]));
        assert!(args.windows(2).any(|pair| pair == ["-frames:v", "1"]));
        assert!(args.windows(2).any(|pair| pair == ["-f", "image2pipe"]));
        assert_eq!(args.last().map(String::as_str), Some("pipe:1"));
    }

    #[test]
    fn preview_bounds_both_dimensions_and_keeps_aspect_ratio() {
        assert!(PREVIEW_FILTER.contains("w=480:h=270"));
        assert!(PREVIEW_FILTER.contains("force_original_aspect_ratio=decrease"));
        assert!(PREVIEW_FILTER.contains("force_divisible_by=2"));
    }

    #[test]
    fn validation_rejects_missing_files_without_exposing_the_path() {
        let error = validate_source(Path::new("private/missing.OSV")).unwrap_err();
        assert_eq!(error, "找不到來源檔案");
        assert!(!error.contains("private"));
    }

    #[test]
    fn extracts_a_bounded_jpeg_from_a_real_video_container() {
        let Some(ffmpeg) = doctor::find_executable("ffmpeg") else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("sample.OSV");
        let generated = silent_command(ffmpeg)
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=640x480:d=0.1",
                "-frames:v",
                "1",
                "-c:v",
                "mpeg4",
                "-f",
                "mp4",
                "-y",
            ])
            .arg(&source)
            .status()
            .unwrap();
        assert!(generated.success());

        let jpeg = extract_first_frame(source.to_string_lossy().into_owned()).unwrap();
        assert!(jpeg.starts_with(&[0xff, 0xd8]));
        assert!(jpeg.ends_with(&[0xff, 0xd9]));
        let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).unwrap();
        assert!(decoded.width() <= 480);
        assert!(decoded.height() <= 270);
    }
}
