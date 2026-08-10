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
    pub feature_extractor: bool,
    pub mapper: bool,
    pub model_converter: bool,
    /// The selected CUDA build exposes the Ceres GPU request option used by
    /// the mapper. Ceres CUDA/cuDSS linkage is still confirmed from mapper
    /// runtime diagnostics.
    pub ceres_gpu: bool,
    /// The selected executable accepts the GPU option pair used by
    /// `feature_extractor`.
    pub feature_extraction_gpu: bool,
    /// The selected executable accepts the GPU option pair used by
    /// `matches_importer` (and the other feature matchers).
    pub feature_matching_gpu: bool,
    /// The selected executable accepts the GPU option pair used by `mapper`.
    /// This is kept separate from `ceres_gpu` so callers can avoid passing
    /// options that a particular command does not understand.
    pub mapper_ba_gpu: bool,
    pub global_mapper: bool,
    /// `global_mapper` accepts gravity-aware rotation averaging options.
    pub global_mapper_gravity: bool,
    /// `global_mapper` accepts GPU global positioning options.
    pub global_mapper_gp_gpu: bool,
    /// `global_mapper` accepts GPU Ceres bundle-adjustment options.
    pub global_mapper_ba_gpu: bool,
    /// `global_mapper` accepts the fixed-rotation BA stage controls.
    pub global_mapper_fixed_rotation_ba: bool,
    /// The binary exposes COLMAP's focal-length view-graph calibration pass.
    pub view_graph_calibrator: bool,
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

#[derive(Debug)]
struct CommandOutput {
    text: String,
    success: bool,
}

fn command_output(path: &Path, args: &[&str]) -> Option<CommandOutput> {
    let output = silent_command(path).args(args).output().ok()?;
    // COLMAP normally writes its banner/help to stdout, but launchers and
    // older builds may write diagnostics to stderr.  Parsing both streams is
    // intentional and keeps an error-only help invocation useful.
    Some(CommandOutput {
        text: format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        success: output.status.success(),
    })
}

fn command_help(path: &Path, args: &[&str]) -> String {
    let Some(output) = command_output(path, args).filter(|output| output.success) else {
        return String::new();
    };
    let has_error_marker = output.text.lines().any(|line| {
        let normalized = line.trim().to_ascii_lowercase();
        normalized.starts_with("error:")
            || normalized.contains("unknown command")
            || normalized.contains("unrecognized command")
            || normalized.contains("command not found")
    });
    if has_error_marker {
        String::new()
    } else {
        output.text
    }
}

pub fn command_version(path: &Path) -> Option<String> {
    let output = command_output(path, &["--version"])?;
    if !output.success {
        return None;
    }
    let lines = output
        .text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let mut first = None;
    for line in lines {
        if first.is_none() {
            first = Some(line.to_owned());
        }
        if line.to_ascii_lowercase().contains("colmap") {
            return Some(line.to_owned());
        }
    }
    first
}

/// Return whether a COLMAP version banner meets the application's supported
/// minimum. COLMAP 4.1.1 is the first supported release for the current rig
/// and global-mapper workflow; an unparseable banner is conservatively false.
pub(crate) fn colmap_version_at_least_4_1_1(version: &str) -> bool {
    let lower = version.to_ascii_lowercase();
    let Some(colmap_offset) = lower.find("colmap") else {
        return false;
    };
    let version_tail = &version[colmap_offset + "colmap".len()..];
    let Some(start) = version_tail.find(|character: char| character.is_ascii_digit()) else {
        return false;
    };
    let mut components = version_tail[start..]
        .split(|character: char| !character.is_ascii_digit())
        .filter(|component| !component.is_empty())
        .take(3)
        .map(|component| component.parse::<u32>().ok());
    let major = components.next().flatten();
    let minor = components.next().flatten();
    let patch = components.next().flatten();
    let (Some(major), Some(minor), Some(patch)) = (major, minor, patch) else {
        return false;
    };
    (major, minor, patch) >= (4, 1, 1)
}

fn has_cuda_build_marker(text: &str) -> bool {
    // COLMAP 4.x emits this marker from `GetBuildInfo()` in the version
    // banner.  Restricting the positive match to a line that identifies
    // COLMAP prevents a generic help paragraph (or FFmpeg's CUDA output) from
    // turning a CPU-only COLMAP build into a false positive.
    let mut saw_colmap_banner = false;
    let mut saw_cuda_marker = false;
    for line in text.lines() {
        let normalized = line.to_ascii_lowercase();
        if !normalized.contains("colmap") {
            continue;
        }
        saw_colmap_banner = true;
        // A negative marker always wins. This matters when a launcher prints
        // a generic CUDA note alongside its actual "without CUDA" banner.
        if normalized.contains("without cuda")
            || normalized.contains("without gpu support")
            || normalized.contains("built without cuda")
            || normalized.contains("compiled without cuda")
            || normalized.contains("not built with cuda")
            || normalized.contains("no cuda")
            || normalized.contains("cuda disabled")
            || normalized.contains("cuda support disabled")
            || normalized.contains("cuda unavailable")
            || normalized.contains("cuda is not available")
            || normalized.contains("cuda support unavailable")
            || normalized.contains("cuda support not available")
            || normalized.contains("cuda support is not available")
            || normalized.contains("cuda enabled: false")
            || normalized.contains("cuda_enabled=false")
            || normalized.contains("cuda_enabled=off")
            || normalized.contains("cuda_enabled=0")
        {
            return false;
        }
        saw_cuda_marker |= normalized.contains("with cuda")
            || normalized.contains("cuda support enabled")
            || normalized.contains("cuda enabled: true")
            || normalized.contains("cuda_enabled=true")
            || normalized.contains("cuda_enabled=on")
            || normalized.contains("cuda_enabled=1");
    }
    saw_colmap_banner && saw_cuda_marker
}

fn help_option_token(line: &str) -> Option<&str> {
    let token = line.split_whitespace().next()?;
    let token = token.strip_prefix("--")?;
    let token = token.split_once('=').map_or(token, |(name, _)| name);
    (!token.is_empty()
        && token.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '.'
        }))
    .then_some(token)
}

/// Check an option from the generated command help, not from arbitrary prose.
/// COLMAP's option manager prints each option as the first token on an
/// indented line (`--Foo.bar arg (=default)`). Requiring that shape avoids
/// accepting an option merely because a paragraph mentions it.
fn has_help_option(text: &str, option: &str) -> bool {
    text.lines()
        .filter_map(help_option_token)
        .any(|token| token == option)
}

fn has_help_option_pair(text: &str, use_gpu: &str, gpu_index: &str) -> bool {
    has_help_option(text, use_gpu) && has_help_option(text, gpu_index)
}

fn has_help_command(text: &str, command: &str) -> bool {
    let mut in_command_list = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("available commands:") {
            in_command_list = true;
            continue;
        }
        if in_command_list {
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == command {
                return true;
            }
            // The generated command list contains one exact command per
            // line. A non-indented heading marks the end of that section.
            if line
                .chars()
                .next()
                .is_some_and(|character| !character.is_whitespace())
            {
                in_command_list = false;
            }
        }
        // Keep a strict fallback for compact/fake help output, while still
        // rejecting prose such as `global_mapper is unavailable`.
        if trimmed == command {
            return true;
        }
    }
    false
}

fn has_help_usage(text: &str, command: &str) -> bool {
    let expected = format!("colmap {command}");
    let mut usage_header = false;
    for line in text.lines() {
        let normalized = line.trim().to_ascii_lowercase();
        if normalized.starts_with("usage:") {
            if normalized.contains(&expected) {
                return true;
            }
            usage_header = true;
            continue;
        }
        if usage_header && normalized.starts_with(&expected) {
            return true;
        }
        if usage_header
            && !normalized.is_empty()
            && line
                .chars()
                .next()
                .is_some_and(|character| !character.is_whitespace())
        {
            usage_header = false;
        }
    }
    false
}

fn has_caspar_marker(text: &str) -> bool {
    has_help_option(text, "BundleAdjustmentCaspar.gpu_index")
}

/// Parse COLMAP's version banner and command help without requiring COLMAP to
/// be installed on the machine running the tests.
///
/// The argument order mirrors the probe calls: `--version`, main `--help`,
/// then the relevant subcommand help text.  Unknown or incomplete output is
/// deliberately reported as `false`.
#[cfg(test)]
fn parse_colmap_capabilities(
    version: &str,
    main_help: &str,
    global_mapper_help: &str,
    mapper_help: &str,
    bundle_adjuster_help: &str,
    rig_configurator_help: &str,
    matches_importer_help: &str,
) -> ColmapCapabilities {
    parse_colmap_capabilities_with_feature_help(
        version,
        main_help,
        "",
        global_mapper_help,
        mapper_help,
        bundle_adjuster_help,
        rig_configurator_help,
        matches_importer_help,
    )
}

#[allow(clippy::too_many_arguments)]
fn parse_colmap_capabilities_with_feature_help(
    version: &str,
    main_help: &str,
    feature_extractor_help: &str,
    global_mapper_help: &str,
    mapper_help: &str,
    bundle_adjuster_help: &str,
    rig_configurator_help: &str,
    matches_importer_help: &str,
) -> ColmapCapabilities {
    // CUDA support is a build property and is therefore parsed only from the
    // version banner. Help text may describe CUDA options even in a CPU-only
    // build and must not be allowed to change this result.
    let cuda_build = has_cuda_build_marker(version);
    let feature_extraction_gpu = cuda_build
        && has_help_option_pair(
            feature_extractor_help,
            "FeatureExtraction.use_gpu",
            "FeatureExtraction.gpu_index",
        );
    let feature_matching_gpu = cuda_build
        && has_help_option_pair(
            matches_importer_help,
            "FeatureMatching.use_gpu",
            "FeatureMatching.gpu_index",
        );
    let mapper_ba_gpu =
        cuda_build && has_help_option_pair(mapper_help, "Mapper.ba_use_gpu", "Mapper.ba_gpu_index");
    let global_mapper_gravity = has_help_option(global_mapper_help, "GlobalMapper.ra_use_gravity")
        && has_help_option(global_mapper_help, "GlobalMapper.ra_use_stratified");
    let global_mapper_gp_gpu = has_help_option_pair(
        global_mapper_help,
        "GlobalMapper.gp_use_gpu",
        "GlobalMapper.gp_gpu_index",
    );
    let global_mapper_ba_gpu = has_help_option_pair(
        global_mapper_help,
        "GlobalMapper.ba_ceres_use_gpu",
        "GlobalMapper.ba_ceres_gpu_index",
    );
    let global_mapper_fixed_rotation_ba = has_help_option(
        global_mapper_help,
        "GlobalMapper.ba_skip_fixed_rotation_stage",
    ) && has_help_option(
        global_mapper_help,
        "GlobalMapper.ba_skip_joint_optimization_stage",
    );
    ColmapCapabilities {
        cuda_build,
        feature_extractor: has_help_command(main_help, "feature_extractor")
            || has_help_usage(feature_extractor_help, "feature_extractor"),
        mapper: has_help_command(main_help, "mapper") || has_help_usage(mapper_help, "mapper"),
        model_converter: has_help_command(main_help, "model_converter"),
        // COLMAP's mapper forwards `Mapper.ba_use_gpu` to Ceres. Checking the
        // exact mapper option pair keeps the result aligned with the command
        // that this application launches. This still cannot prove that Ceres
        // was linked with CUDA/cuDSS; mapper diagnostics remain authoritative
        // at runtime.
        ceres_gpu: mapper_ba_gpu,
        feature_extraction_gpu,
        feature_matching_gpu,
        mapper_ba_gpu,
        global_mapper: has_help_command(main_help, "global_mapper")
            || has_help_usage(global_mapper_help, "global_mapper"),
        global_mapper_gravity,
        global_mapper_gp_gpu,
        global_mapper_ba_gpu,
        global_mapper_fixed_rotation_ba,
        view_graph_calibrator: has_help_command(main_help, "view_graph_calibrator"),
        caspar: cuda_build && has_caspar_marker(bundle_adjuster_help),
        rig_configurator: has_help_command(main_help, "rig_configurator")
            || has_help_usage(rig_configurator_help, "rig_configurator"),
        matches_importer: has_help_command(main_help, "matches_importer")
            || has_help_usage(matches_importer_help, "matches_importer"),
    }
}

/// Probe capabilities of the exact COLMAP executable selected by the caller.
///
/// This is intentionally `pub(crate)` so the pipeline can make decisions from
/// the same executable path that the user selected in settings.  It never
/// consults FFmpeg, `nvidia-smi`, or a different `colmap` found on `PATH`.
pub(crate) fn probe_colmap_capabilities(path: &Path) -> ColmapCapabilities {
    let version = command_output(path, &["--version"])
        .map(|output| output.text)
        .unwrap_or_default();
    let main_help = command_help(path, &["--help"]);
    let feature_extractor_help = command_help(path, &["feature_extractor", "--help"]);
    let global_mapper_help = command_help(path, &["global_mapper", "--help"]);
    let mapper_help = command_help(path, &["mapper", "--help"]);
    let bundle_adjuster_help = command_help(path, &["bundle_adjuster", "--help"]);
    let rig_configurator_help = command_help(path, &["rig_configurator", "--help"]);
    let matches_importer_help = command_help(path, &["matches_importer", "--help"]);

    parse_colmap_capabilities_with_feature_help(
        &version,
        &main_help,
        &feature_extractor_help,
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
    let colmap_supported = colmap.available
        && colmap
            .version
            .as_deref()
            .is_some_and(colmap_version_at_least_4_1_1);
    let colmap_workflow_supported = colmap_supported
        && colmap_capabilities.feature_extractor
        && colmap_capabilities.mapper
        && colmap_capabilities.model_converter
        && colmap_capabilities.rig_configurator
        && colmap_capabilities.matches_importer;
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
            if !colmap_version_at_least_4_1_1(version) {
                warnings.push(
                    "目前流程需要 COLMAP 4.1.1 或更新版本；較舊版本不支援完整的 rig 與全域對齊流程"
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
        if colmap_capabilities.cuda_build
            && (!colmap_capabilities.feature_extraction_gpu
                || !colmap_capabilities.feature_matching_gpu)
        {
            warnings.push(
                "指定的 COLMAP 未同時提供 FeatureExtraction/FeatureMatching GPU 選項；特徵階段會使用 CPU"
                    .to_owned(),
            );
        }
        if !colmap_capabilities.global_mapper {
            warnings.push("指定的 COLMAP 不提供 global_mapper；只能使用增量對齊".to_owned());
        } else {
            if !colmap_capabilities.global_mapper_gravity {
                warnings.push(
                    "指定的 global_mapper 未提供 gravity rotation averaging 選項；global gravity 模式不可用"
                        .to_owned(),
                );
            }
            if !colmap_capabilities.global_mapper_gp_gpu {
                warnings.push(
                    "指定的 global_mapper 未提供 GPU global positioning 選項；global positioning 會使用 CPU"
                        .to_owned(),
                );
            }
            if !colmap_capabilities.global_mapper_ba_gpu {
                warnings.push(
                    "指定的 global_mapper 未提供 GPU Ceres BA 選項；global Bundle Adjustment 會使用 CPU"
                        .to_owned(),
                );
            }
            if !colmap_capabilities.global_mapper_fixed_rotation_ba {
                warnings.push(
                    "指定的 global_mapper 未提供 fixed-rotation/joint BA stage 選項；無法使用固定旋轉實驗模式"
                        .to_owned(),
                );
            }
        }
        if !colmap_capabilities.view_graph_calibrator {
            warnings.push(
                "指定的 COLMAP 不提供 view_graph_calibrator；global mapper 的 focal prior 必須由外部校正提供"
                    .to_owned(),
            );
        }
        if !colmap_capabilities.feature_extractor
            || !colmap_capabilities.mapper
            || !colmap_capabilities.model_converter
            || !colmap_capabilities.rig_configurator
            || !colmap_capabilities.matches_importer
        {
            warnings.push(
                "指定的 COLMAP 缺少 feature_extractor、matches_importer、mapper、model_converter 或 rig_configurator；雙鏡頭對齊流程無法完成"
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
            align: colmap_workflow_supported,
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
    fn supported_colmap_version_requires_4_1_1_or_newer() {
        assert!(colmap_version_at_least_4_1_1(
            "COLMAP 4.1.1 (Commit abc with CUDA)"
        ));
        assert!(colmap_version_at_least_4_1_1("COLMAP 4.2.0"));
        assert!(colmap_version_at_least_4_1_1("COLMAP 5.0.0"));
        assert!(!colmap_version_at_least_4_1_1("COLMAP 4.1.0"));
        assert!(!colmap_version_at_least_4_1_1("COLMAP 4.0.9"));
        assert!(!colmap_version_at_least_4_1_1("COLMAP 3.13.0"));
        assert!(!colmap_version_at_least_4_1_1("unknown version"));
    }

    #[test]
    fn parser_reads_cuda_and_colmap_4_commands_from_help() {
        let capabilities = parse_colmap_capabilities(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc with CUDA)",
            "Available commands:\n  global_mapper\n  view_graph_calibrator\n  rig_configurator\n  matches_importer",
            "Usage: colmap global_mapper [options]\n  --GlobalMapper.ra_use_gravity\n  --GlobalMapper.ra_use_stratified\n  --GlobalMapper.gp_use_gpu\n  --GlobalMapper.gp_gpu_index\n  --GlobalMapper.ba_ceres_use_gpu\n  --GlobalMapper.ba_ceres_gpu_index\n  --GlobalMapper.ba_skip_fixed_rotation_stage\n  --GlobalMapper.ba_skip_joint_optimization_stage",
            "Usage: colmap mapper [options]\n  --Mapper.ba_use_gpu\n  --Mapper.ba_gpu_index",
            "Usage: colmap bundle_adjuster [options]\n  --BundleAdjustmentCeres.use_gpu\n  --BundleAdjustmentCeres.gpu_index\n  --BundleAdjustmentCaspar.gpu_index",
            "Usage: colmap rig_configurator [options]",
            "Usage: colmap matches_importer [options]",
        );

        assert!(capabilities.cuda_build);
        assert!(capabilities.ceres_gpu);
        assert!(capabilities.global_mapper);
        assert!(capabilities.global_mapper_gravity);
        assert!(capabilities.global_mapper_gp_gpu);
        assert!(capabilities.global_mapper_ba_gpu);
        assert!(capabilities.global_mapper_fixed_rotation_ba);
        assert!(capabilities.view_graph_calibrator);
        assert!(capabilities.caspar);
        assert!(capabilities.rig_configurator);
        assert!(capabilities.matches_importer);
    }

    #[test]
    fn parser_rejects_without_cuda_even_when_help_mentions_gpu_options() {
        let capabilities = parse_colmap_capabilities(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc without CUDA)",
            "Available commands:\n  global_mapper\n  rig_configurator\n  matches_importer",
            "Usage: colmap global_mapper [options]\n  --GlobalMapper.ba_ceres_use_gpu\n  --GlobalMapper.ba_ceres_gpu_index",
            "Usage: colmap mapper [options]\n  --Mapper.ba_use_gpu\n  --Mapper.ba_gpu_index",
            "Usage: colmap bundle_adjuster [options]\n  --BundleAdjustmentCeres.use_gpu\n  --BundleAdjustmentCeres.gpu_index\n  --BundleAdjustmentCaspar.gpu_index",
            "Usage: colmap rig_configurator [options]",
            "Usage: colmap matches_importer [options]",
        );

        assert!(!capabilities.cuda_build);
        assert!(!capabilities.ceres_gpu);
        assert!(!capabilities.caspar);
        assert!(capabilities.global_mapper);
        assert!(!capabilities.global_mapper_gravity);
        assert!(!capabilities.global_mapper_gp_gpu);
        assert!(capabilities.global_mapper_ba_gpu);
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
    fn parser_uses_exact_command_options_for_each_gpu_stage() {
        let capabilities = parse_colmap_capabilities_with_feature_help(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc with CUDA)",
            "Available commands:\n  feature_extractor\n  matches_importer\n  mapper",
            "Usage: colmap feature_extractor [options]\n  --FeatureExtraction.use_gpu arg (=1)\n  --FeatureExtraction.gpu_index arg (=-1)",
            "Usage: colmap global_mapper [options]",
            "Usage: colmap mapper [options]\n  --Mapper.ba_use_gpu arg (=0)\n  --Mapper.ba_gpu_index arg (=-1)",
            "Usage: colmap bundle_adjuster [options]",
            "Usage: colmap rig_configurator [options]",
            "Usage: colmap matches_importer [options]\n  --FeatureMatching.use_gpu arg (=1)\n  --FeatureMatching.gpu_index arg (=-1)",
        );

        assert!(capabilities.cuda_build);
        assert!(capabilities.feature_extraction_gpu);
        assert!(capabilities.feature_matching_gpu);
        assert!(capabilities.mapper_ba_gpu);
        assert!(capabilities.ceres_gpu);
    }

    #[test]
    fn parser_does_not_promote_help_mentions_or_ffmpeg_to_cuda_build() {
        let capabilities = parse_colmap_capabilities_with_feature_help(
            "ffmpeg 7.0 (with CUDA)\nCOLMAP 4.1.1",
            "Available commands:\n  feature_extractor",
            "Usage: colmap feature_extractor [options]\n  --FeatureExtraction.use_gpu\n  --FeatureExtraction.gpu_index\nNotes: CUDA enabled builds can use these options.",
            "",
            "",
            "",
            "",
            "",
        );

        assert!(!capabilities.cuda_build);
        assert!(!capabilities.feature_extraction_gpu);
        assert!(!capabilities.feature_matching_gpu);
        assert!(!capabilities.mapper_ba_gpu);
        assert!(!capabilities.ceres_gpu);
    }

    #[test]
    fn parser_rejects_negative_cuda_build_markers() {
        for version in [
            "COLMAP 4.1.1 (not built with CUDA)",
            "COLMAP 4.1.1 (compiled without CUDA)",
            "COLMAP 4.1.1 (CUDA support is not available)",
        ] {
            assert!(
                !has_cuda_build_marker(version),
                "unexpected CUDA marker: {version}"
            );
        }
    }

    #[test]
    fn parser_requires_option_lines_instead_of_prose_tokens() {
        let capabilities = parse_colmap_capabilities_with_feature_help(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc with CUDA)",
            "Available commands:\n  feature_extractor\n  mapper",
            "Usage: colmap feature_extractor [options]\nThis build accepts --FeatureExtraction.use_gpu and --FeatureExtraction.gpu_index.",
            "",
            "Usage: colmap mapper [options]\nThe mapper accepts --Mapper.ba_use_gpu and --Mapper.ba_gpu_index.",
            "",
            "",
            "",
        );

        assert!(capabilities.cuda_build);
        assert!(!capabilities.feature_extraction_gpu);
        assert!(!capabilities.mapper_ba_gpu);
        assert!(!capabilities.ceres_gpu);
    }

    #[test]
    fn parser_does_not_use_standalone_ceres_help_for_mapper_capability() {
        let capabilities = parse_colmap_capabilities_with_feature_help(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc with CUDA)",
            "Available commands:\n  mapper\n  bundle_adjuster",
            "",
            "",
            "Usage: colmap mapper [options]",
            "Usage: colmap bundle_adjuster [options]\n  --BundleAdjustmentCeres.use_gpu\n  --BundleAdjustmentCeres.gpu_index",
            "",
            "",
        );

        assert!(capabilities.cuda_build);
        assert!(!capabilities.mapper_ba_gpu);
        assert!(!capabilities.ceres_gpu);
    }

    #[test]
    fn parser_does_not_find_commands_in_error_text_or_other_help() {
        let capabilities = parse_colmap_capabilities(
            "COLMAP 4.1.1 -- Structure-from-Motion (Commit abc without CUDA)",
            "Usage: colmap mapper [options]\nglobal_mapper is unavailable",
            "unknown command global_mapper",
            "Usage: colmap mapper [options]",
            "",
            "unrecognized command rig_configurator",
            "error: command matches_importer not found",
        );

        assert!(!capabilities.global_mapper);
        assert!(!capabilities.rig_configurator);
        assert!(!capabilities.matches_importer);
    }

    #[cfg(unix)]
    #[test]
    fn probe_reads_capabilities_from_the_selected_executable() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("temporary directory");
        let path = temp.path().join("custom-colmap");
        fs::write(
            &path,
            r##"#!/bin/sh
case "$1 $2" in
  "--version ") printf '%s\n' 'launcher: preparing CUDA runtime' 'COLMAP 4.1.0 (Commit test with CUDA)' ;;
  "--help ") printf '%s\n' 'Available commands:' '  feature_extractor' '  matches_importer' '  mapper' '  model_converter' '  global_mapper' ;;
  "feature_extractor --help") printf '%s\n' '--FeatureExtraction.use_gpu arg (=1)' '--FeatureExtraction.gpu_index arg (=-1)' ;;
  "matches_importer --help") printf '%s\n' '--FeatureMatching.use_gpu arg (=1)' '--FeatureMatching.gpu_index arg (=-1)' ;;
  "mapper --help") printf '%s\n' '--Mapper.ba_use_gpu arg (=0)' '--Mapper.ba_gpu_index arg (=-1)' ;;
  "bundle_adjuster --help") printf '%s\n' '--BundleAdjustmentCeres.use_gpu arg (=0)' '--BundleAdjustmentCeres.gpu_index arg (=-1)' ;;
  *) exit 1 ;;
esac
"##,
        )
        .expect("fake COLMAP executable");
        let mut permissions = fs::metadata(&path)
            .expect("fake COLMAP metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).expect("make fake COLMAP executable");

        assert_eq!(
            command_version(&path).as_deref(),
            Some("COLMAP 4.1.0 (Commit test with CUDA)")
        );
        let capabilities = probe_colmap_capabilities(&path);
        assert!(capabilities.cuda_build);
        assert!(capabilities.feature_extraction_gpu);
        assert!(capabilities.feature_matching_gpu);
        assert!(capabilities.mapper_ba_gpu);
        assert!(capabilities.ceres_gpu);
        assert!(capabilities.feature_extractor);
        assert!(capabilities.mapper);
        assert!(capabilities.model_converter);
        assert!(capabilities.global_mapper);
        assert!(capabilities.matches_importer);
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
