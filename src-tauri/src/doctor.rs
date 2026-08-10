//! Runtime capability discovery.
//!
//! The application deliberately probes the executables that are available on the
//! host instead of bundling a second copy of FFmpeg/COLMAP.  Probe results are
//! informational and are safe to display in the UI; no paths or command output
//! containing user media are persisted.

use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};

use crate::process::silent_command;

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

/// Capabilities reported by the selected COLMAP executable itself.
///
/// These values deliberately describe build/CLI capabilities, not whether a
/// GPU happens to be present on the host.  In particular, an NVIDIA device or
/// FFmpeg's CUDA hwaccel cannot prove that this COLMAP binary was built with
/// CUDA (or that its Ceres solver has GPU support).
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ColmapCapabilities {
    pub cuda_build: bool,
    /// The selected CUDA build exposes the Ceres GPU request option. Ceres
    /// CUDA/cuDSS linkage is still confirmed from mapper runtime diagnostics.
    pub ceres_gpu: bool,
    pub global_mapper: bool,
    pub caspar: bool,
    pub rig_configurator: bool,
    pub matches_importer: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DoctorReport {
    pub platform: String,
    pub arch: String,
    pub tools: Vec<ToolInfo>,
    pub accelerators: Vec<AcceleratorInfo>,
    pub capabilities: DoctorCapabilities,
    pub colmap_capabilities: ColmapCapabilities,
    pub warnings: Vec<String>,
}

/// Locate an executable without invoking a shell.  This keeps paths with
/// spaces safe and works on Windows where `which` is not normally installed.
pub fn find_executable(name: &str) -> Option<PathBuf> {
    let names: Vec<String> = if cfg!(windows) && Path::new(name).extension().is_none() {
        vec![
            name.to_owned(),
            format!("{name}.exe"),
            format!("{name}.bat"),
            format!("{name}.cmd"),
        ]
    } else {
        vec![name.to_owned()]
    };

    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            for candidate_name in &names {
                let candidate = dir.join(candidate_name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    // A caller may pass an absolute or relative executable path.  It is useful
    // to support that in tests and for custom installations.
    let direct = PathBuf::from(name);
    direct.is_file().then_some(direct)
}

/// Resolve COLMAP from an explicit local preference or, when it is empty, the
/// host PATH. Windows pre-built releases are expected to use `COLMAP.bat`
/// because that launcher prepares the required library search path.
pub fn resolve_colmap(custom_path: Option<&str>) -> Result<PathBuf, String> {
    match custom_path.map(str::trim).filter(|path| !path.is_empty()) {
        Some(path) => {
            let path = PathBuf::from(path);
            path.is_file()
                .then_some(path)
                .ok_or_else(|| "指定的 COLMAP 路徑不存在或不是檔案".to_owned())
        }
        None => find_executable("colmap").ok_or_else(|| "在系統 PATH 中找不到 COLMAP".to_owned()),
    }
}

fn command_output(path: &Path, args: &[&str]) -> Option<String> {
    let output = silent_command(path).args(args).output().ok()?;
    // COLMAP normally writes its banner/help to stdout, but launchers and
    // older builds may write diagnostics to stderr.  Parsing both streams is
    // intentional and keeps an error-only help invocation useful.
    Some(format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

pub fn command_version(path: &Path) -> Option<String> {
    let text = command_output(path, &["--version"])?;
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}

fn has_token(text: &str, token: &str) -> bool {
    text.split(|character: char| {
        !character.is_ascii_alphanumeric() && character != '_' && character != '.'
    })
    .any(|part| part == token)
}

fn has_token_prefix(text: &str, prefix: &str) -> bool {
    text.split(|character: char| {
        !character.is_ascii_alphanumeric() && character != '_' && character != '.'
    })
    .any(|part| {
        part == prefix
            || part
                .strip_prefix(prefix)
                .is_some_and(|suffix| suffix.starts_with('.'))
    })
}

fn has_help_command(text: &str, command: &str) -> bool {
    text.lines().any(|line| {
        let normalized = line.to_ascii_lowercase();
        !normalized.contains("unknown")
            && !normalized.contains("not found")
            && !normalized.contains("unrecognized")
            && has_token(line, command)
    })
}

fn has_cuda_build_marker(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    // A negative marker always wins.  This matters when a launcher prints a
    // generic CUDA note alongside its actual "without CUDA" build banner.
    if normalized.contains("without cuda")
        || normalized.contains("without gpu support")
        || normalized.contains("no cuda")
        || normalized.contains("cuda disabled")
        || normalized.contains("cuda support disabled")
        || normalized.contains("cuda unavailable")
        || normalized.contains("cuda is not available")
        || normalized.contains("cuda support unavailable")
        || normalized.contains("cuda support not available")
        || normalized.contains("cuda enabled: false")
        || normalized.contains("cuda_enabled=false")
        || normalized.contains("cuda_enabled=off")
        || normalized.contains("cuda_enabled=0")
    {
        return false;
    }
    normalized.contains("with cuda")
        || normalized.contains("cuda enabled")
        || normalized.contains("cuda-enabled")
        || normalized.contains("cuda_enabled")
}

fn has_ceres_gpu_marker(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    if normalized.contains("ceres was compiled without cuda")
        || normalized.contains("ceres compiled without cuda")
        || normalized.contains("ceres without cuda")
        || normalized.contains("ceres cuda support disabled")
        || normalized.contains("ceres gpu disabled")
        || normalized.contains("ceres gpu support unavailable")
        || normalized.contains("ceres gpu support not available")
        || normalized.contains("ceres cuda unavailable")
        || normalized.contains("ceres cuda not available")
        || normalized.contains("ceres_no_cuda")
        || normalized.contains("ceres_no_gpu")
        || normalized.contains("ceres no cuda")
        || normalized.contains("ceres no gpu")
        || normalized.contains("ceres_gpu=false")
        || normalized.contains("ceres_gpu=off")
    {
        return false;
    }
    has_token(text, "BundleAdjustmentCeres.use_gpu")
        || has_token(text, "GlobalMapper.ba_ceres_use_gpu")
        || normalized.contains("ceres cuda")
        || normalized.contains("ceres gpu")
        || normalized.contains("ceres[cuda]")
}

fn has_caspar_marker(text: &str) -> bool {
    let normalized = text.to_ascii_lowercase();
    if normalized.contains("without caspar")
        || normalized.contains("caspar disabled")
        || normalized.contains("caspar unavailable")
        || normalized.contains("caspar not available")
        || normalized.contains("caspar support unavailable")
        || normalized.contains("caspar support not available")
        || normalized.contains("caspar_enabled=false")
        || normalized.contains("caspar_enabled=off")
        || normalized.contains("caspar_enabled=0")
        || normalized.contains("caspar backend disabled")
    {
        return false;
    }
    has_token_prefix(text, "BundleAdjustmentCaspar")
        || normalized.contains("caspar_enabled")
        || normalized.contains("caspar backend")
}

/// Parse COLMAP's version banner and command help without requiring COLMAP to
/// be installed on the machine running the tests.
///
/// The argument order mirrors the probe calls: `--version`, main `--help`,
/// then the relevant subcommand help text.  Unknown or incomplete output is
/// deliberately reported as `false`.
fn parse_colmap_capabilities(
    version: &str,
    main_help: &str,
    global_mapper_help: &str,
    mapper_help: &str,
    bundle_adjuster_help: &str,
    rig_configurator_help: &str,
    matches_importer_help: &str,
) -> ColmapCapabilities {
    let build_text = format!(
        "{version}\n{main_help}\n{global_mapper_help}\n{mapper_help}\n{bundle_adjuster_help}\n{rig_configurator_help}\n{matches_importer_help}"
    );
    let cuda_build = has_cuda_build_marker(&build_text);
    let ceres_text = format!(
        "{version}\n{main_help}\n{global_mapper_help}\n{mapper_help}\n{bundle_adjuster_help}"
    );
    let command_text = format!(
        "{main_help}\n{global_mapper_help}\n{mapper_help}\n{bundle_adjuster_help}\n{rig_configurator_help}\n{matches_importer_help}"
    );

    ColmapCapabilities {
        cuda_build,
        ceres_gpu: cuda_build && has_ceres_gpu_marker(&ceres_text),
        global_mapper: has_help_command(&command_text, "global_mapper"),
        caspar: cuda_build && has_caspar_marker(&ceres_text),
        rig_configurator: has_help_command(&command_text, "rig_configurator"),
        matches_importer: has_help_command(&command_text, "matches_importer"),
    }
}

/// Probe capabilities of the exact COLMAP executable selected by the caller.
///
/// This is intentionally `pub(crate)` so the pipeline can make decisions from
/// the same executable path that the user selected in settings.  It never
/// consults FFmpeg, `nvidia-smi`, or a different `colmap` found on `PATH`.
pub(crate) fn probe_colmap_capabilities(path: &Path) -> ColmapCapabilities {
    let version = command_output(path, &["--version"]).unwrap_or_default();
    let main_help = command_output(path, &["--help"]).unwrap_or_default();
    let global_mapper_help = command_output(path, &["global_mapper", "--help"]).unwrap_or_default();
    let mapper_help = command_output(path, &["mapper", "--help"]).unwrap_or_default();
    let bundle_adjuster_help =
        command_output(path, &["bundle_adjuster", "--help"]).unwrap_or_default();
    let rig_configurator_help =
        command_output(path, &["rig_configurator", "--help"]).unwrap_or_default();
    let matches_importer_help =
        command_output(path, &["matches_importer", "--help"]).unwrap_or_default();

    parse_colmap_capabilities(
        &version,
        &main_help,
        &global_mapper_help,
        &mapper_help,
        &bundle_adjuster_help,
        &rig_configurator_help,
        &matches_importer_help,
    )
}

fn tool(name: &str, custom_path: Option<&str>) -> ToolInfo {
    let resolved = if name == "colmap" {
        resolve_colmap(custom_path).ok()
    } else {
        find_executable(name)
    };
    match resolved {
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
            note: Some(
                if name == "colmap"
                    && custom_path
                        .map(str::trim)
                        .is_some_and(|path| !path.is_empty())
                {
                    "指定的 COLMAP 路徑不存在或不是檔案".to_owned()
                } else {
                    format!("{name} was not found on PATH")
                },
            ),
        },
    }
}

fn ffmpeg_hwaccels(ffmpeg: &ToolInfo) -> String {
    let Some(path) = ffmpeg.path.as_deref() else {
        return String::new();
    };
    let Ok(output) = silent_command(path)
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

fn cuda_accelerator(colmap: &ToolInfo, capabilities: &ColmapCapabilities) -> AcceleratorInfo {
    let note = if !colmap.available {
        Some("尚未選取可執行的 COLMAP；無法判定 COLMAP CUDA 建置能力".to_owned())
    } else if !capabilities.cuda_build {
        Some(
            "指定的 COLMAP 未在 version/help banner 標示 CUDA；將使用 CPU 特徵擷取與配對"
                .to_owned(),
        )
    } else if capabilities.ceres_gpu {
        Some(
            "指定的 COLMAP 建置支援 CUDA，且可請求 Ceres GPU；實際 CUDA/cuDSS 支援會在 mapper 執行期確認"
                .to_owned(),
        )
    } else {
        Some(
            "指定的 COLMAP 建置支援 CUDA，但未確認 Ceres GPU 求解器；Bundle Adjustment 會使用 CPU"
                .to_owned(),
        )
    };
    AcceleratorInfo {
        kind: "cuda".to_owned(),
        name: "COLMAP CUDA".to_owned(),
        available: capabilities.cuda_build,
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
pub fn report(custom_colmap_path: Option<&str>) -> DoctorReport {
    let ffmpeg = tool("ffmpeg", None);
    let ffprobe = tool("ffprobe", None);
    let colmap = tool("colmap", custom_colmap_path);
    let colmap_capabilities = colmap
        .path
        .as_deref()
        .map(Path::new)
        .map(probe_colmap_capabilities)
        .unwrap_or_default();
    let tools = vec![ffmpeg.clone(), ffprobe.clone(), colmap.clone()];
    let accelerators = vec![
        cuda_accelerator(&colmap, &colmap_capabilities),
        videotoolbox_accelerator(&ffmpeg),
    ];

    let mut warnings = Vec::new();
    if !ffmpeg.available || !ffprobe.available {
        warnings.push("影格擷取需要系統已安裝 FFmpeg 與 ffprobe".to_owned());
    }
    if !colmap.available {
        if custom_colmap_path
            .map(str::trim)
            .is_some_and(|path| !path.is_empty())
        {
            warnings.push("指定的 COLMAP 路徑不存在或不是檔案".to_owned());
        } else {
            warnings.push("找不到 COLMAP；對齊階段會維持可繼續的待執行狀態".to_owned());
        }
    } else {
        if let Some(version) = &colmap.version {
            if !version.contains("4.") {
                warnings.push(
                    "COLMAP 3.x 可用於增量對齊；若未安裝 COLMAP 4.x，將不啟用重力與全域對齊功能"
                        .to_owned(),
                );
            }
        } else {
            warnings.push(
                "無法讀取指定 COLMAP 的 version banner；能力判定採保守的未知/不可用".to_owned(),
            );
        }
        if !colmap_capabilities.cuda_build {
            warnings.push(
                "指定的 COLMAP 未確認 CUDA 建置；特徵擷取與配對會使用 CPU，且無法啟用 Ceres GPU 求解器"
                    .to_owned(),
            );
        } else if !colmap_capabilities.ceres_gpu {
            warnings.push(
                "指定的 COLMAP 支援 CUDA，但未確認 Ceres GPU 求解器；Bundle Adjustment 會使用 CPU"
                    .to_owned(),
            );
        }
        if !colmap_capabilities.global_mapper {
            warnings.push("指定的 COLMAP 不提供 global_mapper；只能使用增量對齊".to_owned());
        }
        if !colmap_capabilities.rig_configurator || !colmap_capabilities.matches_importer {
            warnings.push(
                "指定的 COLMAP 缺少 rig_configurator 或 matches_importer；雙鏡頭對齊流程可能無法完成"
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
        colmap_capabilities,
        tools,
        accelerators,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn report_is_serializable_with_camel_case_fields() {
        let value = serde_json::to_value(report(None)).expect("doctor report should serialize");
        assert!(value.get("platform").is_some());
        assert!(value
            .get("capabilities")
            .and_then(|v| v.get("extract"))
            .is_some());
        assert!(value.get("colmapCapabilities").is_some());
        assert!(value.get("accelerators").is_some());
    }

    #[test]
    fn parser_reads_cuda_and_colmap_4_commands_from_help() {
        let capabilities = parse_colmap_capabilities(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc with CUDA)",
            "Available commands:\n  global_mapper\n  rig_configurator\n  matches_importer",
            "Usage: colmap global_mapper [options]\n  --GlobalMapper.ba_ceres_use_gpu",
            "Usage: colmap mapper [options]",
            "Usage: colmap bundle_adjuster [options]\n  --BundleAdjustmentCeres.use_gpu\n  --BundleAdjustmentCaspar.gpu_index",
            "Usage: colmap rig_configurator [options]",
            "Usage: colmap matches_importer [options]",
        );

        assert!(capabilities.cuda_build);
        assert!(capabilities.ceres_gpu);
        assert!(capabilities.global_mapper);
        assert!(capabilities.caspar);
        assert!(capabilities.rig_configurator);
        assert!(capabilities.matches_importer);
    }

    #[test]
    fn parser_rejects_without_cuda_even_when_help_mentions_gpu_options() {
        let capabilities = parse_colmap_capabilities(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc without CUDA)",
            "Available commands:\n  global_mapper\n  rig_configurator\n  matches_importer",
            "Usage: colmap global_mapper [options]\n  --GlobalMapper.ba_ceres_use_gpu",
            "Usage: colmap mapper [options]",
            "Usage: colmap bundle_adjuster [options]\n  --BundleAdjustmentCeres.use_gpu\n  --BundleAdjustmentCaspar.gpu_index",
            "Usage: colmap rig_configurator [options]",
            "Usage: colmap matches_importer [options]",
        );

        assert!(!capabilities.cuda_build);
        assert!(!capabilities.ceres_gpu);
        assert!(!capabilities.caspar);
        assert!(capabilities.global_mapper);
        assert!(capabilities.rig_configurator);
        assert!(capabilities.matches_importer);
    }

    #[test]
    fn parser_reports_false_for_unknown_or_error_only_output() {
        let capabilities = parse_colmap_capabilities(
            "COLMAP 4.1.1",
            "unknown command global_mapper",
            "error: command not found",
            "",
            "",
            "unrecognized command rig_configurator",
            "",
        );

        assert!(!capabilities.cuda_build);
        assert!(!capabilities.ceres_gpu);
        assert!(!capabilities.global_mapper);
        assert!(!capabilities.caspar);
        assert!(!capabilities.rig_configurator);
        assert!(!capabilities.matches_importer);
    }

    #[test]
    fn find_executable_accepts_absolute_paths() {
        let path = std::env::current_exe().expect("test executable");
        assert_eq!(find_executable(path.to_string_lossy().as_ref()), Some(path));
    }

    #[test]
    fn resolve_colmap_prefers_an_explicit_path() {
        let path = std::env::current_exe().expect("test executable");
        assert_eq!(
            resolve_colmap(Some(path.to_string_lossy().as_ref())),
            Ok(path)
        );
    }

    #[test]
    fn resolve_colmap_rejects_an_invalid_explicit_path() {
        assert_eq!(
            resolve_colmap(Some("/definitely/not/a/colmap/executable")),
            Err("指定的 COLMAP 路徑不存在或不是檔案".to_owned())
        );
    }

    #[test]
    fn resolve_colmap_accepts_a_path_containing_spaces() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let directory = temp.path().join("COLMAP portable");
        fs::create_dir_all(&directory).expect("portable directory");
        let path = directory.join(if cfg!(windows) {
            "COLMAP.bat"
        } else {
            "colmap"
        });
        fs::write(&path, b"test launcher").expect("test launcher");

        assert_eq!(
            resolve_colmap(Some(path.to_string_lossy().as_ref())),
            Ok(path)
        );
    }
}
