//! Strict COLMAP input reader. IDs are keys, never array indices.
use super::camera::Camera;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub id: u32,
    pub camera_id: u32,
    pub name: String,
    pub pose: [f64; 7],
}
pub struct Dataset {
    pub cameras: BTreeMap<u32, Camera>,
    pub frames: Vec<Frame>,
    pub identity: String,
}
pub fn hash(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut b = [0u8; 65536];
    loop {
        let n = f.read(&mut b).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn safe_join(root: &Path, name: &str) -> Result<PathBuf, String> {
    if name.contains('\\')
        || name.contains(':')
        || name.is_empty()
        || Path::new(name)
            .components()
            .any(|v| !matches!(v, Component::Normal(_)))
    {
        return Err(format!("Unsafe image/artifact name: {name}"));
    }
    let p = root.join(name);
    let canonical = p
        .canonicalize()
        .map_err(|e| format!("{}: {e}", p.display()))?;
    if !canonical.starts_with(root.canonicalize().map_err(|e| e.to_string())?) {
        return Err("Path escapes dataset".into());
    }
    Ok(canonical)
}
fn scalar<const N: usize>(r: &mut impl Read) -> Result<[u8; N], String> {
    let mut b = [0; N];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b)
}
fn u32v(r: &mut impl Read) -> Result<u32, String> {
    Ok(u32::from_le_bytes(scalar(r)?))
}
fn u64v(r: &mut impl Read) -> Result<u64, String> {
    Ok(u64::from_le_bytes(scalar(r)?))
}
fn f64v(r: &mut impl Read) -> Result<f64, String> {
    Ok(f64::from_le_bytes(scalar(r)?))
}
fn count(r: &mut impl Read, max: u64) -> Result<usize, String> {
    let n = u64v(r)?;
    if n > max {
        return Err("COLMAP count exceeds bounds".into());
    }
    Ok(n as usize)
}
fn parse<T: std::str::FromStr>(v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("Invalid COLMAP value: {v}"))
}
pub fn read(root: &Path) -> Result<Dataset, String> {
    let mut model = root.join("sparse/0");
    if !model.is_dir() {
        model = root.join("sparse");
    }
    let binary = model.join("cameras.bin").exists();
    let ext = if binary { "bin" } else { "txt" };
    let mut h = Sha256::new();
    for name in ["cameras", "images", "points3D", "rigs", "frames"] {
        let p = model.join(format!("{name}.{ext}"));
        if p.exists() {
            h.update(name.as_bytes());
            h.update(hash(&p)?.as_bytes());
        } else if ["cameras", "images", "points3D"].contains(&name) {
            return Err(format!("Missing final model: {}", p.display()));
        }
    }
    let mut cameras = BTreeMap::new();
    let mut frames = vec![];
    if binary {
        let mut r =
            BufReader::new(File::open(model.join("cameras.bin")).map_err(|e| e.to_string())?);
        for _ in 0..count(&mut r, 100_000)? {
            let id = u32v(&mut r)?;
            let code = u32v(&mut r)?;
            let (name, n) = match code {
                1 => ("PINHOLE", 4),
                5 => ("OPENCV_FISHEYE", 8),
                _ => return Err(format!("Unsupported COLMAP camera model ID {code}")),
            };
            let width = u32::try_from(u64v(&mut r)?).map_err(|_| "Invalid width")?;
            let height = u32::try_from(u64v(&mut r)?).map_err(|_| "Invalid height")?;
            let params = (0..n)
                .map(|_| f64v(&mut r))
                .collect::<Result<Vec<_>, _>>()?;
            let mut c = Camera {
                id,
                model: name.into(),
                width,
                height,
                params,
                theta_max: 0.,
            };
            c.validate()?;
            if cameras.insert(id, c).is_some() {
                return Err("Duplicate camera ID".into());
            }
        }
        let mut r =
            BufReader::new(File::open(model.join("images.bin")).map_err(|e| e.to_string())?);
        let length = r.get_ref().metadata().map_err(|e| e.to_string())?.len();
        for _ in 0..count(&mut r, 1_000_000)? {
            let id = u32v(&mut r)?;
            let mut pose = [0.; 7];
            for v in &mut pose {
                *v = f64v(&mut r)?;
            }
            let camera_id = u32v(&mut r)?;
            let mut bytes = vec![];
            loop {
                let b = scalar::<1>(&mut r)?[0];
                if b == 0 {
                    break;
                }
                bytes.push(b);
                if bytes.len() > 4096 {
                    return Err("COLMAP image name too long".into());
                }
            }
            let name = String::from_utf8(bytes).map_err(|_| "Image names must be UTF-8")?;
            let obs = count(&mut r, 100_000_000)? as u64;
            let end = r
                .stream_position()
                .map_err(|e| e.to_string())?
                .checked_add(obs * 24)
                .ok_or("Observation overflow")?;
            if end > length {
                return Err("Truncated COLMAP observations".into());
            }
            r.seek(SeekFrom::Start(end)).map_err(|e| e.to_string())?;
            frames.push(Frame {
                id,
                camera_id,
                name,
                pose,
            });
        }
    } else {
        let txt = fs::read_to_string(model.join("cameras.txt")).map_err(|e| e.to_string())?;
        for line in txt
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let p: Vec<_> = line.split_whitespace().collect();
            if p.len() < 8 {
                return Err("Invalid cameras.txt row".into());
            }
            let id = parse(p[0])?;
            let mut c = Camera {
                id,
                model: p[1].into(),
                width: parse(p[2])?,
                height: parse(p[3])?,
                params: p[4..].iter().map(|v| parse(v)).collect::<Result<_, _>>()?,
                theta_max: 0.,
            };
            c.validate()?;
            if cameras.insert(id, c).is_some() {
                return Err("Duplicate camera ID".into());
            }
        }
        let txt = fs::read_to_string(model.join("images.txt")).map_err(|e| e.to_string())?;
        let mut lines = txt.lines().filter(|l| !l.trim_start().starts_with('#'));
        while let Some(line) = lines.next() {
            if line.trim().is_empty() {
                continue;
            }
            let p: Vec<_> = line.split_whitespace().collect();
            if p.len() < 10 {
                return Err("Invalid images.txt pose row".into());
            }
            let mut pose = [0.; 7];
            for i in 0..7 {
                pose[i] = parse(p[i + 1])?;
            }
            let obs = lines.next().ok_or("Missing images.txt observation row")?;
            let fields: Vec<_> = obs.split_whitespace().collect();
            if fields.len() % 3 != 0 {
                return Err("Invalid observations row".into());
            }
            for o in fields.chunks_exact(3) {
                let _: f64 = parse(o[0])?;
                let _: f64 = parse(o[1])?;
                let _: i64 = parse(o[2])?;
            }
            frames.push(Frame {
                id: parse(p[0])?,
                camera_id: parse(p[8])?,
                name: {
                    // Preserve repeated spaces inside a valid image filename.
                    let mut rest = line.trim_start();
                    for _ in 0..9 {
                        let end = rest.find(char::is_whitespace).ok_or("Missing image name")?;
                        rest = rest[end..].trim_start();
                    }
                    rest.trim_end().to_owned()
                },
                pose,
            });
        }
    }
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    for f in &frames {
        if !ids.insert(f.id)
            || !names.insert(&f.name)
            || !cameras.contains_key(&f.camera_id)
            || f.pose.iter().any(|v| !v.is_finite())
            || (f.pose[..4].iter().map(|v| v * v).sum::<f64>() - 1.).abs() > 1e-3
        {
            return Err("Invalid/duplicate registered frame or pose".into());
        }
        // Validate names before hashing or writing any frame cache.
        safe_join(&root.join("images"), &f.name)?;
    }
    if frames.is_empty() {
        return Err("Final COLMAP model has no registered images".into());
    }
    frames.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Dataset {
        cameras,
        frames,
        identity: format!("{:x}", h.finalize()),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traversal_rejected() {
        let d = tempfile::tempdir().unwrap();
        for s in ["../x", "a/../../b", "C:/x", "a\\b", "/tmp/x"] {
            assert!(safe_join(d.path(), s).is_err());
        }
    }
    #[test]
    fn binary_ids_and_utf8_names_are_preserved_and_truncation_fails() {
        let d = tempfile::tempdir().unwrap();
        let model = d.path().join("sparse/0");
        fs::create_dir_all(&model).unwrap();
        fs::create_dir_all(d.path().join("images/lens1")).unwrap();
        let name = "lens1/中文  frame.png";
        fs::write(d.path().join("images").join(name), b"rgb").unwrap();
        let mut c = vec![];
        c.extend(1u64.to_le_bytes());
        c.extend(91u32.to_le_bytes());
        c.extend(5u32.to_le_bytes());
        c.extend(64u64.to_le_bytes());
        c.extend(64u64.to_le_bytes());
        for v in [17f64, 17., 32., 32., 0., 0., 0., 0.] {
            c.extend(v.to_le_bytes());
        }
        fs::write(model.join("cameras.bin"), c).unwrap();
        let mut i = vec![];
        i.extend(1u64.to_le_bytes());
        i.extend(901u32.to_le_bytes());
        for v in [1f64, 0., 0., 0., 1., 2., 3.] {
            i.extend(v.to_le_bytes());
        }
        i.extend(91u32.to_le_bytes());
        i.extend(name.as_bytes());
        i.push(0);
        i.extend(0u64.to_le_bytes());
        fs::write(model.join("images.bin"), &i).unwrap();
        fs::write(model.join("points3D.bin"), 0u64.to_le_bytes()).unwrap();
        let dataset = read(d.path()).unwrap();
        assert_eq!(dataset.frames[0].id, 901);
        assert_eq!(dataset.frames[0].name, name);
        assert_eq!(dataset.cameras[&91].params[0], 17.);
        let n = i.len();
        i[n - 8..].copy_from_slice(&1u64.to_le_bytes());
        fs::write(model.join("images.bin"), &i).unwrap();
        assert!(read(d.path()).err().unwrap().contains("Truncated"));
    }
    #[test]
    fn text_names_keep_repeated_spaces() {
        let d = tempfile::tempdir().unwrap();
        fs::create_dir_all(d.path().join("sparse/0")).unwrap();
        fs::create_dir_all(d.path().join("images/lens0")).unwrap();
        fs::write(d.path().join("images/lens0/中文  frame.png"), b"rgb").unwrap();
        fs::write(
            d.path().join("sparse/0/cameras.txt"),
            "91 PINHOLE 64 64 32 32 32 32\n",
        )
        .unwrap();
        fs::write(
            d.path().join("sparse/0/images.txt"),
            "901 1 0 0 0 0 0 0 91 lens0/中文  frame.png\n\n",
        )
        .unwrap();
        fs::write(d.path().join("sparse/0/points3D.txt"), "").unwrap();
        assert_eq!(
            read(d.path()).unwrap().frames[0].name,
            "lens0/中文  frame.png"
        );
    }
}
