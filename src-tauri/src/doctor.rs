//! Runtime capability discovery.
//!
//! The application deliberately probes the executables that are available on the
//! host instead of bundling a second copy of FFmpeg/COLMAP.  Probe results are
//! informational and are safe to display in the UI; no paths or command output
//! containing user media are persisted.

use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolInfo {
    pub name: String,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceleratorInfo {
    pub kind: String,
    pub name: String,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DoctorCapabilities {
    pub extract: bool,
    pub mask: bool,
    pub align: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DoctorReport {
    pub platform: String,
    pub arch: String,
    pub tools: Vec<ToolInfo>,
    pub accelerators: Vec<AcceleratorInfo>,
    pub capabilities: DoctorCapabilities,
    pub warnings: Vec<String>,
}

/// Locate an executable without invoking a shell.  This keeps paths with
/// spaces safe and works on Windows where `which` is not normally installed.
pub fn find_executable(name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    let names: Vec<String> = if cfg!(windows) && Path::new(name).extension().is_none() {
        vec![
            name.to_owned(),
            format!("{name}.exe"),
            format!("{name}.cmd"),
        ]
    } else {
        vec![name.to_owned()]
    };

    for dir in env::split_paths(&path_var) {
        for candidate_name in &names {
            let candidate = dir.join(candidate_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    // A caller may pass an absolute or relative executable path.  It is useful
    // to support that in tests and for custom installations.
    let direct = PathBuf::from(name);
    direct.is_file().then_some(direct)
}

pub fn command_version(path: &Path) -> Option<String> {
    let output = Command::new(path).arg("--version").output().ok()?;
    let text = if output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stderr).into_owned()
    } else {
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn tool(name: &str) -> ToolInfo {
    match find_executable(name) {
        Some(path) => ToolInfo {
            name: name.to_owned(),
            available: true,
            version: command_version(&path),
            path: Some(path.to_string_lossy().into_owned()),
            note: None,
        },
        None => ToolInfo {
            name: name.to_owned(),
            available: false,
            version: None,
            path: None,
            note: Some(format!("{name} was not found on PATH")),
        },
    }
}

fn ffmpeg_hwaccels(ffmpeg: &ToolInfo) -> String {
    let Some(path) = ffmpeg.path.as_deref() else {
        return String::new();
    };
    let Ok(output) = Command::new(path)
        .args(["-hide_banner", "-hwaccels"])
        .output()
    else {
        return String::new();
    };
    format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn cuda_accelerator(ffmpeg: &ToolInfo) -> AcceleratorInfo {
    let nvidia = find_executable("nvidia-smi").and_then(|path| {
        Command::new(path)
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
            .filter(|s| !s.is_empty())
    });
    let hw = ffmpeg_hwaccels(ffmpeg).to_ascii_lowercase();
    let available = nvidia.is_some() || hw.contains("cuda") || hw.contains("cuvid");
    let note = if let Some(gpu) = nvidia {
        Some(format!(
            "Detected NVIDIA GPU: {}",
            gpu.lines().next().unwrap_or(gpu.as_str())
        ))
    } else if available {
        Some("FFmpeg advertises CUDA acceleration; GPU details were not reported".to_owned())
    } else {
        Some("CUDA is optional; COLMAP can run with its CPU solver".to_owned())
    };
    AcceleratorInfo {
        kind: "cuda".to_owned(),
        name: "CUDA".to_owned(),
        available,
        note,
    }
}

fn videotoolbox_accelerator(ffmpeg: &ToolInfo) -> AcceleratorInfo {
    let hw = ffmpeg_hwaccels(ffmpeg).to_ascii_lowercase();
    let available = cfg!(target_os = "macos") && hw.contains("videotoolbox");
    AcceleratorInfo {
        kind: "videoToolbox".to_owned(),
        name: "Apple VideoToolbox".to_owned(),
        available,
        note: if cfg!(target_os = "macos") {
            Some("FFmpeg VideoToolbox support is used for decode/encode only; COLMAP alignment remains CPU unless a compatible CUDA build is installed".to_owned())
        } else {
            Some("VideoToolbox is only available on macOS".to_owned())
        },
    }
}

/// Probe host tools and accelerators.  This function is pure from the point of
/// view of the app: it does not install, modify, or download anything.
pub fn report() -> DoctorReport {
    let ffmpeg = tool("ffmpeg");
    let ffprobe = tool("ffprobe");
    let colmap = tool("colmap");
    let tools = vec![ffmpeg.clone(), ffprobe.clone(), colmap.clone()];
    let accelerators = vec![cuda_accelerator(&ffmpeg), videotoolbox_accelerator(&ffmpeg)];

    let mut warnings = Vec::new();
    if !ffmpeg.available || !ffprobe.available {
        warnings.push("影格擷取需要系統已安裝 FFmpeg 與 ffprobe".to_owned());
    }
    if !colmap.available {
        warnings.push("找不到 COLMAP；對齊階段會維持可繼續的待執行狀態".to_owned());
    } else if let Some(version) = &colmap.version {
        if !version.contains("4.") {
            warnings.push(
                "COLMAP 3.x 可用於增量對齊；若未安裝 COLMAP 4.x，將不啟用重力與全域對齊功能"
                    .to_owned(),
            );
        }
    }
    if cfg!(target_os = "macos") && !accelerators[1].available {
        warnings.push("FFmpeg 不支援 VideoToolbox；影格擷取將使用 CPU 解碼".to_owned());
    }

    DoctorReport {
        platform: env::consts::OS.to_owned(),
        arch: env::consts::ARCH.to_owned(),
        capabilities: DoctorCapabilities {
            extract: ffmpeg.available && ffprobe.available,
            // The built-in validity mask is always available.  Neural masks
            // are an optional adapter and are reported by the mask stage.
            mask: true,
            align: colmap.available,
        },
        tools,
        accelerators,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_serializable_with_camel_case_fields() {
        let value = serde_json::to_value(report()).expect("doctor report should serialize");
        assert!(value.get("platform").is_some());
        assert!(value
            .get("capabilities")
            .and_then(|v| v.get("extract"))
            .is_some());
        assert!(value.get("accelerators").is_some());
    }

    #[test]
    fn find_executable_accepts_absolute_paths() {
        let path = std::env::current_exe().expect("test executable");
        assert_eq!(find_executable(path.to_string_lossy().as_ref()), Some(path));
    }
}
