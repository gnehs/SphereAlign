//! Lossless bounded protobuf edit for ONE pinned ONNX/profile. Unknown fields
//! are preserved byte-for-byte. Does not decode or copy tensor values piecemeal.
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

pub const DERIVED_HASH: &str = "e3a6e567a98cf8dfbbdc113fa59e38f9b31c1cd1aff1fcbd316a1438388ec4ac";
pub const DERIVED_BYTES: u64 = 419_411_835;

struct Field<'a> {
    tag: u64,
    wire: u8,
    raw: &'a [u8],
    data: &'a [u8],
}
fn varint(bytes: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let b = *bytes.get(*pos).ok_or("truncated protobuf varint")?;
        *pos += 1;
        if shift == 63 && b > 1 {
            return Err("protobuf varint overflow".into());
        }
        value |= u64::from(b & 127) << shift;
        if b < 128 {
            return Ok(value);
        }
    }
    Err("protobuf varint overflow".into())
}
fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let start = pos;
        let key = varint(bytes, &mut pos)?;
        let tag = key >> 3;
        let wire = (key & 7) as u8;
        if tag == 0 || tag >= (1 << 29) {
            return Err("invalid protobuf field number".into());
        }
        let data_start;
        match wire {
            0 => {
                data_start = pos;
                varint(bytes, &mut pos)?;
            }
            1 => {
                data_start = pos;
                pos = pos.checked_add(8).ok_or("length overflow")?;
            }
            2 => {
                let n = usize::try_from(varint(bytes, &mut pos)?).map_err(|_| "length overflow")?;
                data_start = pos;
                pos = pos.checked_add(n).ok_or("length overflow")?;
            }
            5 => {
                data_start = pos;
                pos = pos.checked_add(4).ok_or("length overflow")?;
            }
            _ => return Err("unsupported protobuf wire type".into()),
        }
        if pos > bytes.len() {
            return Err("truncated protobuf field".into());
        }
        result.push(Field {
            tag,
            wire,
            raw: &bytes[start..pos],
            data: &bytes[data_start..pos],
        });
        if result.len() > 100_000 {
            return Err("protobuf field limit".into());
        }
    }
    Ok(result)
}
fn push_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 128 {
        out.push((v as u8 & 127) | 128);
        v >>= 7;
    }
    out.push(v as u8);
}
fn message(tag: u64, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 12);
    push_varint(&mut out, tag << 3 | 2);
    push_varint(&mut out, data.len() as u64);
    out.extend(data);
    out
}
fn scalar(tag: u64, value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    push_varint(&mut out, tag << 3);
    push_varint(&mut out, value);
    out
}
fn field_name(bytes: &[u8]) -> Result<&[u8], String> {
    fields(bytes)?
        .into_iter()
        .find(|f| f.tag == 1 && f.wire == 2)
        .map(|f| f.data)
        .ok_or("missing ValueInfo name".into())
}
fn rewrite_graph(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut tensor = scalar(2, 7); // TensorProto INT64 scalar; empty dimensions.
    tensor.extend(message(8, b"num_tokens"));
    tensor.extend(message(9, &1800i64.to_le_bytes()));
    let mut shape = Vec::new();
    for d in [1, 3, 768, 768] {
        shape.extend(message(1, &scalar(1, d)));
    }
    let mut tensor_type = scalar(1, 1);
    tensor_type.extend(message(2, &shape));
    let mut image = message(1, b"image");
    image.extend(message(2, &message(1, &tensor_type)));
    let mut out = Vec::with_capacity(bytes.len());
    let mut inserted = false;
    let mut image_count = 0;
    let mut token_count = 0;
    for f in fields(bytes)? {
        if f.tag > 5 && !inserted {
            out.extend(message(5, &tensor));
            inserted = true;
        }
        if f.tag == 11 && f.wire == 2 {
            match field_name(f.data)? {
                b"image" => {
                    out.extend(message(11, &image));
                    image_count += 1;
                }
                b"num_tokens" => {
                    token_count += 1;
                }
                _ => return Err("unexpected pinned ONNX input".into()),
            }
        } else {
            out.extend(f.raw);
        }
    }
    if !inserted || image_count != 1 || token_count != 1 {
        return Err("pinned graph contract mismatch".into());
    }
    Ok(out)
}
fn rewrite(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut graphs = 0;
    for f in fields(bytes)? {
        if f.tag == 7 && f.wire == 2 {
            out.extend(message(7, &rewrite_graph(f.data)?));
            graphs += 1;
        } else {
            out.extend(f.raw);
        }
    }
    if graphs != 1 {
        return Err("ONNX requires exactly one graph".into());
    }
    Ok(out)
}

/// Explicit model preparation, not inference/fallback. CPU performs protobuf
/// editing only; graph execution remains restricted to the selected GPU EP.
pub fn prepare(source: &Path, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        return Err("derived artifact must be a new file".into());
    }
    super::model::verify(source)?;
    let bytes = fs::read(source).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != super::model::SHA256 {
        return Err("source changed during preparation".into());
    }
    let derived = rewrite(&bytes)?;
    if derived.len() as u64 != DERIVED_BYTES
        || format!("{:x}", Sha256::digest(&derived)) != DERIVED_HASH
    {
        return Err("derived graph hash mismatch; cannot publish".into());
    }
    let parent = destination.parent().ok_or("destination requires parent")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let partial = destination.with_extension(format!("partial-{}", std::process::id()));
    let mut created = false;
    let result = (|| -> Result<(), String> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(&derived).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        // Hard-link publication is atomic and no-clobber on the same filesystem.
        fs::hard_link(&partial, destination).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if created {
        let _ = fs::remove_file(partial);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_truncated_and_overflow_wire_data() {
        for bytes in [
            &[0x3a, 9, 1][..],
            &[0x80][..],
            &[0u8][..],
            &[0xffu8; 11][..],
        ] {
            assert!(fields(bytes).is_err());
        }
    }
    #[test]
    fn preserves_unknown_model_fields() {
        let mut graph = message(11, &message(1, b"image"));
        graph.extend(message(11, &message(1, b"num_tokens")));
        let unknown = message(100, b"preserve");
        let mut model = unknown.clone();
        model.extend(message(7, &graph));
        let result = rewrite(&model).unwrap();
        assert!(result.starts_with(&unknown));
        let graph = fields(&result)
            .unwrap()
            .into_iter()
            .find(|f| f.tag == 7)
            .unwrap();
        assert_eq!(
            fields(graph.data)
                .unwrap()
                .iter()
                .filter(|f| f.tag == 11)
                .count(),
            1
        );
    }
    #[test]
    fn refuses_missing_graph_or_input_contract() {
        assert!(rewrite(&[]).is_err());
        assert!(rewrite(&message(7, &[])).is_err());
    }
    #[test]
    fn refuses_to_overwrite_existing_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("already.onnx");
        fs::write(&destination, b"user artifact").unwrap();
        assert!(prepare(Path::new("missing"), &destination)
            .unwrap_err()
            .contains("new file"));
        assert_eq!(fs::read(destination).unwrap(), b"user artifact");
    }
}
