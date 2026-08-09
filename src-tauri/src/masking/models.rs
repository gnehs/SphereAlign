//! Model discovery and verified first-use downloads for the mask pipeline.

use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::{CancelToken, MaskError, MaskResult};

const YOLO_CACHE_PATH: &str = "ultralytics/yolo11s-seg-onnx/onnx/model.onnx";
const SKYSEG_CACHE_PATH: &str = "JianyuanWang/skyseg/skyseg.onnx";

// Keep the exact YOLO artifact already validated by gs360masker. The immutable
// commit URL avoids silently accepting a replaced GitHub release asset.
const YOLO_DOWNLOAD_URL: &str = "https://raw.githubusercontent.com/gnehs/gs360masker/5f26a7c1d9de98fff6ee6ffef51701e1d288a27d/src-tauri/resources/models/yolo11s-seg.onnx";
const YOLO_SHA256: &str = "5d10f1c4f80b32e30ccaad031203cfa5c94275dbe2b7cd60c03fb212c3e14e54";
const YOLO_SIZE: u64 = 40_680_531;

// Pin the Hugging Face revision so a future repository update cannot change the
// downloaded bytes without an intentional application update.
const SKYSEG_DOWNLOAD_URL: &str = "https://huggingface.co/JianyuanWang/skyseg/resolve/3ba8c6df1d9ba9ff26f637c7ba9568ac11a9aa7f/skyseg.onnx";
const SKYSEG_SHA256: &str = "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39";
const SKYSEG_SIZE: u64 = 175_997_079;

const YOLO_SPEC: DownloadSpec<'static> = DownloadSpec {
    label: "YOLO11 segmentation",
    relative_path: YOLO_CACHE_PATH,
    url: YOLO_DOWNLOAD_URL,
    sha256: YOLO_SHA256,
    size: YOLO_SIZE,
};

const SKYSEG_SPEC: DownloadSpec<'static> = DownloadSpec {
    label: "SkySeg",
    relative_path: SKYSEG_CACHE_PATH,
    url: SKYSEG_DOWNLOAD_URL,
    sha256: SKYSEG_SHA256,
    size: SKYSEG_SIZE,
};

/// Byte-level progress for a model being downloaded on first use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadProgress {
    pub label: &'static str,
    pub downloaded: u64,
    pub total: u64,
}

/// Resolved ONNX model files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    /// YOLO11 segmentation model (`[1,116,8400]` + prototype output).
    pub yolo: PathBuf,
    /// Optional sky segmentation model. It is required when `mask_sky` is true.
    pub skyseg: Option<PathBuf>,
}

impl ModelPaths {
    /// Discover explicit/local model files, then download missing required models
    /// into the application data model directory with SHA-256 verification.
    pub fn resolve(
        model_dir: Option<&Path>,
        cache_dir: Option<&Path>,
        yolo_model: Option<&Path>,
        skyseg_model: Option<&Path>,
        require_skyseg: bool,
        cancel: &CancelToken,
        on_download: &dyn Fn(ModelDownloadProgress),
    ) -> MaskResult<Self> {
        let roots = model_roots(model_dir, cache_dir);
        let yolo = match yolo_model {
            Some(path) => validate_model_path(path, "YOLO11 segmentation")?,
            None => resolve_required_model(
                &roots,
                cache_dir,
                YOLO_CANDIDATES,
                &YOLO_SPEC,
                cancel,
                on_download,
            )?,
        };

        let skyseg = match skyseg_model {
            Some(path) => Some(validate_model_path(path, "SkySeg")?),
            None => resolve_optional_model(
                &roots,
                cache_dir,
                SKYSEG_CANDIDATES,
                &SKYSEG_SPEC,
                require_skyseg,
                cancel,
                on_download,
            )?,
        };

        Ok(Self { yolo, skyseg })
    }
}

// Older gs360masker builds use a Hugging Face-style repository path while
// portable Studio installs may use a flat models directory.
const YOLO_CANDIDATES: &[&str] = &[
    YOLO_CACHE_PATH,
    "ultralytics/yolo11s-seg-onnx/yolo11s-seg.onnx",
    "yolo11s-seg-onnx/onnx/model.onnx",
    "onnx/model.onnx",
    "yolo11s-seg.onnx",
    "yolo11n-seg.onnx",
    "yolo11m-seg.onnx",
];

const SKYSEG_CANDIDATES: &[&str] = &[SKYSEG_CACHE_PATH, "skyseg/skyseg.onnx", "skyseg.onnx"];

#[derive(Clone, Copy)]
struct DownloadSpec<'a> {
    label: &'static str,
    relative_path: &'a str,
    url: &'a str,
    sha256: &'a str,
    size: u64,
}

fn resolve_required_model(
    roots: &[PathBuf],
    cache_dir: Option<&Path>,
    candidates: &[&str],
    spec: &DownloadSpec<'_>,
    cancel: &CancelToken,
    on_download: &dyn Fn(ModelDownloadProgress),
) -> MaskResult<PathBuf> {
    if let Some(path) = find_usable_model(roots, candidates, cache_dir, spec)? {
        return Ok(path);
    }

    let Some(cache_dir) = cache_dir else {
        return Err(model_not_found(spec.label, roots));
    };
    download_model(cache_dir, spec, cancel, on_download)
}

fn resolve_optional_model(
    roots: &[PathBuf],
    cache_dir: Option<&Path>,
    candidates: &[&str],
    spec: &DownloadSpec<'_>,
    required: bool,
    cancel: &CancelToken,
    on_download: &dyn Fn(ModelDownloadProgress),
) -> MaskResult<Option<PathBuf>> {
    if let Some(path) = find_usable_model(roots, candidates, cache_dir, spec)? {
        return Ok(Some(path));
    }

    if !required {
        return Ok(None);
    }
    let Some(cache_dir) = cache_dir else {
        return Err(model_not_found(spec.label, roots));
    };
    download_model(cache_dir, spec, cancel, on_download).map(Some)
}

fn model_roots(explicit: Option<&Path>, cache_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for path in explicit
        .map(Path::to_path_buf)
        .into_iter()
        .chain(env::var_os("GS360_MODEL_DIR").map(PathBuf::from))
        .chain(cache_dir.map(Path::to_path_buf))
        .chain([PathBuf::from("models"), PathBuf::from(".models")])
    {
        if !roots.iter().any(|root| root == &path) {
            roots.push(path);
        }
    }
    roots
}

fn find_usable_model(
    roots: &[PathBuf],
    candidates: &[&str],
    cache_dir: Option<&Path>,
    spec: &DownloadSpec<'_>,
) -> MaskResult<Option<PathBuf>> {
    for path in roots
        .iter()
        .flat_map(|root| candidates.iter().map(move |relative| root.join(relative)))
        .filter(|path| path.is_file())
    {
        if !is_managed_cache_path(&path, cache_dir, spec) || verify_sha256(&path, spec.sha256)? {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn validate_model_path(path: &Path, label: &str) -> MaskResult<PathBuf> {
    if !path.is_file() {
        return Err(MaskError::model(format!(
            "{label} model path is not a file: {}",
            path.display()
        )));
    }
    Ok(path.to_path_buf())
}

fn model_not_found(label: &str, roots: &[PathBuf]) -> MaskError {
    MaskError::model(format!(
        "{label} model not found and no application model cache is available; searched {}",
        format_roots(roots)
    ))
}

fn is_managed_cache_path(path: &Path, cache_dir: Option<&Path>, spec: &DownloadSpec<'_>) -> bool {
    cache_dir
        .map(|root| path == root.join(spec.relative_path))
        .unwrap_or(false)
}

fn download_model(
    cache_dir: &Path,
    spec: &DownloadSpec<'_>,
    cancel: &CancelToken,
    on_download: &dyn Fn(ModelDownloadProgress),
) -> MaskResult<PathBuf> {
    let destination = cache_dir.join(spec.relative_path);
    if destination.is_file() && verify_sha256(&destination, spec.sha256)? {
        return Ok(destination);
    }
    if cancel.is_cancelled() {
        return Err(MaskError::Cancelled);
    }

    let parent = destination.parent().ok_or_else(|| {
        MaskError::model(format!(
            "invalid model cache path: {}",
            destination.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        MaskError::model(format!(
            "unable to create model cache {}: {error}",
            parent.display()
        ))
    })?;

    let partial = partial_path(&destination);
    let result = download_to_partial(&partial, spec, cancel, on_download).and_then(|()| {
        // Another process may have completed the same download while this request
        // was in flight. Preserve its verified file instead of replacing it.
        if destination.is_file() && verify_sha256(&destination, spec.sha256)? {
            return Ok(destination.clone());
        }
        if destination.exists() {
            fs::remove_file(&destination).map_err(|error| {
                MaskError::model(format!(
                    "unable to replace invalid cached model {}: {error}",
                    destination.display()
                ))
            })?;
        }
        fs::rename(&partial, &destination).map_err(|error| {
            MaskError::model(format!(
                "unable to commit downloaded model {}: {error}",
                destination.display()
            ))
        })?;
        Ok(destination.clone())
    });

    if partial.exists() {
        let _ = fs::remove_file(&partial);
    }
    result
}

fn download_to_partial(
    partial: &Path,
    spec: &DownloadSpec<'_>,
    cancel: &CancelToken,
    on_download: &dyn Fn(ModelDownloadProgress),
) -> MaskResult<()> {
    let config = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30 * 60)))
        .build();
    let agent: ureq::Agent = config.into();
    let mut response = agent
        .get(spec.url)
        .header(
            "User-Agent",
            concat!("gs360studio/", env!("CARGO_PKG_VERSION")),
        )
        .call()
        .map_err(|error| {
            MaskError::model(format!(
                "unable to download {} model from {}: {error}",
                spec.label, spec.url
            ))
        })?;
    let mut reader = response.body_mut().as_reader();
    let mut output = File::create(partial).map_err(|error| {
        MaskError::model(format!(
            "unable to create model download {}: {error}",
            partial.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    let mut last_reported_percent = 0_u64;
    let mut buffer = [0_u8; 128 * 1024];

    on_download(ModelDownloadProgress {
        label: spec.label,
        downloaded,
        total: spec.size,
    });
    loop {
        if cancel.is_cancelled() {
            return Err(MaskError::Cancelled);
        }
        let count = reader.read(&mut buffer).map_err(|error| {
            MaskError::model(format!("failed while downloading {}: {error}", spec.label))
        })?;
        if count == 0 {
            break;
        }
        downloaded = downloaded.saturating_add(count as u64);
        if downloaded > spec.size {
            return Err(MaskError::model(format!(
                "{} download exceeded the expected size of {} bytes",
                spec.label, spec.size
            )));
        }
        output.write_all(&buffer[..count]).map_err(|error| {
            MaskError::model(format!("unable to write {} model: {error}", spec.label))
        })?;
        hasher.update(&buffer[..count]);
        let percent = downloaded.saturating_mul(100) / spec.size;
        if percent > last_reported_percent {
            last_reported_percent = percent;
            on_download(ModelDownloadProgress {
                label: spec.label,
                downloaded,
                total: spec.size,
            });
        }
    }

    output.flush().map_err(|error| {
        MaskError::model(format!("unable to flush {} model: {error}", spec.label))
    })?;
    output.sync_all().map_err(|error| {
        MaskError::model(format!("unable to sync {} model: {error}", spec.label))
    })?;

    if downloaded != spec.size {
        return Err(MaskError::model(format!(
            "{} download size mismatch: expected {}, got {} bytes",
            spec.label, spec.size, downloaded
        )));
    }
    let actual = format!("{:x}", hasher.finalize());
    if !actual.eq_ignore_ascii_case(spec.sha256) {
        return Err(MaskError::model(format!(
            "{} download checksum mismatch: expected {}, got {}",
            spec.label, spec.sha256, actual
        )));
    }
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> MaskResult<bool> {
    let mut file = File::open(path).map_err(|error| {
        MaskError::model(format!(
            "unable to inspect cached model {}: {error}",
            path.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            MaskError::model(format!(
                "unable to verify model {}: {error}",
                path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()).eq_ignore_ascii_case(expected))
}

fn partial_path(destination: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    destination.with_extension(format!("onnx.part-{}-{stamp}", std::process::id()))
}

fn format_roots(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        "(no model roots)".to_string()
    } else {
        roots
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::thread;
    use tempfile::TempDir;

    #[test]
    fn downloads_verifies_and_reuses_cached_model() -> MaskResult<()> {
        let body = b"verified model bytes".to_vec();
        let expected = format!("{:x}", Sha256::digest(&body));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        });
        let url = format!("http://{address}/model.onnx");
        let spec = DownloadSpec {
            label: "test model",
            relative_path: "test/model.onnx",
            url: &url,
            sha256: &expected,
            size: 20,
        };
        let dir = TempDir::new().unwrap();
        let token = CancelToken::new();
        let events = RefCell::new(Vec::new());
        let path = download_model(dir.path(), &spec, &token, &|event| {
            events.borrow_mut().push(event)
        })?;
        server.join().unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"verified model bytes");
        assert_eq!(events.borrow().last().unwrap().downloaded, 20);
        assert_eq!(download_model(dir.path(), &spec, &token, &|_| {})?, path);
        Ok(())
    }

    #[test]
    fn rejects_a_download_with_the_wrong_checksum() {
        let body = b"bad model bytes".to_vec();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        });
        let url = format!("http://{address}/model.onnx");
        let spec = DownloadSpec {
            label: "test model",
            relative_path: "test/model.onnx",
            url: &url,
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            size: 15,
        };
        let dir = TempDir::new().unwrap();
        let error = download_model(dir.path(), &spec, &CancelToken::new(), &|_| {}).unwrap_err();
        server.join().unwrap();

        assert!(error.to_string().contains("checksum mismatch"));
        assert!(!dir.path().join("test/model.onnx").exists());
    }

    #[test]
    fn skips_a_corrupted_managed_cache_when_a_manual_model_exists() -> MaskResult<()> {
        let dir = TempDir::new().unwrap();
        let cache = dir.path().join("cache");
        let manual = dir.path().join("manual");
        fs::create_dir_all(cache.join("test"))?;
        fs::create_dir_all(&manual)?;
        fs::write(cache.join("test/model.onnx"), b"corrupted")?;
        fs::write(manual.join("model.onnx"), b"custom model")?;
        let expected = format!("{:x}", Sha256::digest(b"expected managed model"));
        let spec = DownloadSpec {
            label: "test model",
            relative_path: "test/model.onnx",
            url: "http://unused.invalid/model.onnx",
            sha256: &expected,
            size: 22,
        };

        let resolved = find_usable_model(
            &[cache.clone(), manual.clone()],
            &["test/model.onnx", "model.onnx"],
            Some(&cache),
            &spec,
        )?;

        assert_eq!(resolved, Some(manual.join("model.onnx")));
        Ok(())
    }
}
