//! A pinned candidate registry and real ORT provider probe (no fallback).
use ort::{session::Session, value::Tensor};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path, time::Instant};

pub const MODEL_ID: &str = "moge2-vitb-normal";
pub const REVISION: &str = "2d247a122ada42ce700fe59273cd62e076da1f32";
pub const SHA256: &str = "bbf14e07a30f11e69d36ab861590123f5598ababcbc8946a063eb4a966f35a21";
pub const BYTES: u64 = 419_411_850;
// Developer-only deterministic ONNX 1.19.1 specialization. This is a distinct
// artifact/profile, not a replacement hash for the official model.
const STATIC_SHA256: &str = "0b33aa5249fdea9338b94276adbd700476cfde2aa0e5af24a8f26f1e39876ecf";
const STATIC_BYTES: u64 = 419_586_451;

/// One GPU session, sequential faces, pinned native graph only.
pub struct GeometryModel(Session);
pub struct Heads {
    pub points: Vec<f32>,
    pub normal: Vec<f32>,
    pub mask: Vec<f32>,
    pub metric_scale: f32,
}
impl GeometryModel {
    pub fn load(path: &Path, provider: &str) -> Result<Self, String> {
        if provider != "DirectML" || !cfg!(target_os = "windows") {
            return Err("This experimental profile currently requires Windows DirectML; CPU fallback is disabled".into());
        }
        verify_hash(
            path,
            super::specialize::DERIVED_HASH,
            super::specialize::DERIVED_BYTES,
        )?;
        let mut builder =
            crate::masking::session_builder_for_provider(provider).map_err(|e| e.to_string())?;
        crate::masking::register_execution_provider_with_cache(&mut builder, provider, None)
            .map_err(|e| e.to_string())?;
        Ok(Self(builder.commit_from_file(path).map_err(|e| {
            format!("DirectML session failed (no CPU fallback): {e}")
        })?))
    }
    pub fn infer(&mut self, pixels: Vec<f32>) -> Result<Heads, String> {
        const N: usize = 768;
        if pixels.len() != 3 * N * N {
            return Err("Invalid image tensor size".into());
        }
        let input = Tensor::from_array(([1, 3, N, N], pixels)).map_err(|e| e.to_string())?;
        let outputs = self
            .0
            .run(ort::inputs!["image"=>input])
            .map_err(|e| format!("DirectML inference failed: {e}"))?;
        let get = |name: &str, expected: &[i64]| -> Result<Vec<f32>, String> {
            let (shape, data) = outputs[name]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            if &shape[..] != expected {
                return Err(format!("Unexpected {name} shape: {shape:?}"));
            }
            Ok(data.to_vec())
        };
        let metric_scale = get("metric_scale", &[1])?[0];
        if !metric_scale.is_finite() || metric_scale <= 0. {
            return Err("Invalid metric scale head".into());
        }
        Ok(Heads {
            points: get("points", &[1, N as i64, N as i64, 3])?,
            normal: get("normal", &[1, N as i64, N as i64, 3])?,
            mask: get("mask", &[1, N as i64, N as i64])?,
            metric_scale,
        })
    }
}

pub fn verify(path: &Path) -> Result<String, String> {
    verify_hash(path, SHA256, BYTES)
}

pub(crate) fn verify_hash(path: &Path, expected: &str, size: u64) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() != size {
        return Err(format!("{MODEL_ID}: expected {size} bytes"));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let hash = format!("{:x}", hasher.finalize());
    if hash != expected {
        return Err(format!("{MODEL_ID}: SHA-256 mismatch: {hash}"));
    }
    Ok(hash)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Probe {
    pub model_id: &'static str,
    pub model_revision: &'static str,
    pub model_sha256: &'static str,
    pub provider: String,
    pub cpu_fallback: bool,
    pub status: String,
    pub error: Option<String>,
    pub elapsed_ms: u128,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub runs: Vec<serde_json::Value>,
    pub runtime: Option<String>,
    pub shape_override: Option<[usize; 3]>,
}

/// Explicit developer probe; CPU is never accepted as a geometry provider.
/// Success is session/shape feasibility only, never quality or parity certification.
pub fn probe(path: &Path, provider: &str) -> Probe {
    probe_shape(path, provider, None)
}

pub fn probe_shape(path: &Path, provider: &str, shape: Option<[usize; 3]>) -> Probe {
    probe_candidate(path, provider, shape, false, None)
}

/// Explicit static experiment is never selected automatically after a failure.
pub fn probe_static_experiment(path: &Path, provider: &str, dump: Option<&Path>) -> Probe {
    probe_candidate(path, provider, Some([768, 768, 1800]), true, dump)
}

fn probe_candidate(
    path: &Path,
    provider: &str,
    shape: Option<[usize; 3]>,
    static_experiment: bool,
    dump: Option<&Path>,
) -> Probe {
    let native_static = static_experiment
        && std::fs::metadata(path).is_ok_and(|m| m.len() == super::specialize::DERIVED_BYTES);
    let start = Instant::now();
    let mut report = Probe {
        model_id: if native_static {
            "moge2-vitb-normal-static768-t1800-native-v1"
        } else if static_experiment {
            "moge2-vitb-normal-static768-t1800-experiment-v1"
        } else {
            MODEL_ID
        },
        model_revision: REVISION,
        model_sha256: if native_static {
            super::specialize::DERIVED_HASH
        } else if static_experiment {
            STATIC_SHA256
        } else {
            SHA256
        },
        provider: provider.to_owned(),
        cpu_fallback: false,
        status: "failed".into(),
        error: None,
        elapsed_ms: 0,
        inputs: vec![],
        outputs: vec![],
        runs: vec![],
        runtime: None,
        shape_override: shape,
    };
    let result = (|| -> Result<(), String> {
        if !["DirectML", "CoreML", "CUDA"].contains(&provider) {
            return Err("Geometry probe requires explicit DirectML, CoreML or CUDA; CPU fallback is forbidden".into());
        }
        if native_static {
            verify_hash(
                path,
                super::specialize::DERIVED_HASH,
                super::specialize::DERIVED_BYTES,
            )?;
        } else if static_experiment {
            verify_hash(path, STATIC_SHA256, STATIC_BYTES)?;
        } else {
            verify(path)?;
        }
        if let Some(dir) = dump {
            std::fs::create_dir(dir).map_err(|e| format!("dump directory must be new: {e}"))?;
        }
        if shape.is_some_and(|[w, h, t]| {
            !(14..=1064).contains(&w) || !(14..=1064).contains(&h) || !(1..=3600).contains(&t)
        }) {
            return Err("probe shape outside bounded 14..1064 / 1..3600 token limits".into());
        }
        report.runtime = Some(ort::info().to_owned());
        let mut builder =
            crate::masking::session_builder_for_provider(provider).map_err(|e| e.to_string())?;
        builder = builder
            .with_log_level(ort::logging::LogLevel::Verbose)
            .map_err(|e| e.to_string())?
            .with_logger(std::sync::Arc::new(
                |level, category, _, location, message| {
                    eprintln!("{level:?} {category} {location}: {message}");
                },
            ))
            .map_err(|e| e.to_string())?;
        if let Some([w, h, _]) = shape {
            builder = builder
                .with_dimension_override("batch_size", 1)
                .map_err(|e| e.to_string())?
                .with_dimension_override("width", w as i64)
                .map_err(|e| e.to_string())?
                .with_dimension_override("height", h as i64)
                .map_err(|e| e.to_string())?;
        }
        crate::masking::register_execution_provider_with_cache(&mut builder, provider, None)
            .map_err(|e| e.to_string())?;
        let mut session: Session = builder.commit_from_file(path).map_err(|e| e.to_string())?;
        report.inputs = session
            .inputs()
            .iter()
            .map(|v| format!("{}: {:?}", v.name(), v.dtype()))
            .collect();
        report.outputs = session
            .outputs()
            .iter()
            .map(|v| format!("{}: {:?}", v.name(), v.dtype()))
            .collect();
        // The pinned artifact's contract is verified separately by ONNX inspection.
        // Refuse arbitrary models rather than guessing a single image input.
        if session.inputs().len() != if static_experiment { 1 } else { 2 }
            || session.inputs()[0].name() != "image"
            || (!static_experiment && session.inputs()[1].name() != "num_tokens")
        {
            return Err("candidate graph input contract mismatch; inspect before inference".into());
        }
        let shapes = shape
            .map(|[w, h, t]| vec![(w, h, t as i64)])
            .unwrap_or_else(|| vec![(32, 32, 4), (96, 64, 24), (768, 768, 1800)]);
        for (width, height, tokens) in shapes {
            let t = Instant::now();
            let pixels: Vec<f32> = (0..3 * height * width)
                .map(|i| ((i * 17 + 31) % 256) as f32 / 255.)
                .collect();
            let input =
                Tensor::from_array(([1, 3, height, width], pixels)).map_err(|e| e.to_string())?;
            let tokens_input = Tensor::from_array((Vec::<usize>::new(), vec![tokens]))
                .map_err(|e| e.to_string())?;
            let outputs = if static_experiment {
                session.run(ort::inputs!["image" => input])
            } else {
                session.run(ort::inputs!["image" => input, "num_tokens" => tokens_input])
            }
            .map_err(|e| e.to_string())?;
            let mut stats = serde_json::Map::new();
            for (name, output) in outputs.iter() {
                let (shape, data) = output
                    .try_extract_tensor::<f32>()
                    .map_err(|e| e.to_string())?;
                if data.is_empty() || data.iter().any(|x| !x.is_finite()) {
                    return Err(format!("{name}: nonfinite or empty output"));
                }
                if let Some(dir) = dump {
                    if !["points", "normal", "mask", "metric_scale"].contains(&name) {
                        return Err("unexpected head name".into());
                    }
                    use std::io::Write;
                    let mut file = std::io::BufWriter::new(
                        File::create(dir.join(format!("{name}.f32le")))
                            .map_err(|e| e.to_string())?,
                    );
                    for value in data {
                        file.write_all(&value.to_le_bytes())
                            .map_err(|e| e.to_string())?;
                    }
                    file.flush().map_err(|e| e.to_string())?;
                }
                stats.insert(name.to_owned(), serde_json::json!({"shape": shape.to_vec(), "min": data.iter().copied().fold(f32::INFINITY,f32::min), "max": data.iter().copied().fold(f32::NEG_INFINITY,f32::max)}));
            }
            report.runs.push(serde_json::json!({"width":width,"height":height,"tokens":tokens,"elapsedMs":t.elapsed().as_millis(),"outputs":stats}));
        }
        Ok(())
    })();
    match result {
        Ok(()) => report.status = "feasible_not_certified".into(),
        Err(e) => report.error = Some(e),
    }
    report.elapsed_ms = start.elapsed().as_millis();
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_artifact_before_ort() {
        let f = tempfile::NamedTempFile::new().unwrap();
        assert!(verify(f.path()).unwrap_err().contains("bytes"));
    }
    #[test]
    fn cpu_is_never_a_geometry_provider() {
        let result = probe(Path::new("missing.onnx"), "CPU");
        assert_eq!(result.status, "failed");
        assert!(!result.cpu_fallback);
        assert!(result.error.unwrap().contains("forbidden"));
    }
    #[test]
    fn rejects_hash_mismatch_even_when_size_matches() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"ab").unwrap();
        assert!(verify_hash(file.path(), &"0".repeat(64), 2)
            .unwrap_err()
            .contains("SHA-256 mismatch"));
    }
}
