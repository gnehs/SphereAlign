//! Read-only validation of the per-image feature cache in a COLMAP database.
//!
//! COLMAP considers an image's features present when both its `keypoints` and
//! `descriptors` rows exist.  The rows may legitimately be zero for images in
//! which no features were detected, so this module validates row presence and
//! blob shape instead of requiring a positive feature count.

use rusqlite::{Connection, OpenFlags, Row, TransactionBehavior};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const SUPPORTED_EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];
const SIFT_DESCRIPTOR_COLS: i64 = 128;
const KEYPOINT_SCALAR_BYTES: u64 = 4;
const DESCRIPTOR_SCALAR_BYTES: u64 = 1;

/// Whether all expected images have a valid keypoint/descriptor pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FeatureCacheStatus {
    Complete,
    Incomplete,
}

/// Result of checking a COLMAP feature cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FeatureCacheReport {
    pub(crate) status: FeatureCacheStatus,
    pub(crate) expected: usize,
    pub(crate) completed: usize,
}

impl FeatureCacheReport {
    pub(crate) fn is_complete(&self) -> bool {
        self.status == FeatureCacheStatus::Complete
    }
}

/// Errors that make a feature cache unsafe to reuse.
#[derive(Debug)]
pub(crate) enum FeatureCacheError {
    ImageRoot {
        path: PathBuf,
        message: String,
    },
    Database {
        path: PathBuf,
        source: rusqlite::Error,
    },
    Schema {
        path: PathBuf,
        message: String,
    },
    IncompatibleImages {
        missing: Vec<String>,
        extra: Vec<String>,
    },
    CorruptFeature {
        image: String,
        table: &'static str,
        message: String,
    },
}

impl fmt::Display for FeatureCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageRoot { path, message } => {
                write!(
                    formatter,
                    "invalid COLMAP image root {}: {message}",
                    path.display()
                )
            }
            Self::Database { path, source } => {
                write!(
                    formatter,
                    "cannot read COLMAP database {}: {source}",
                    path.display()
                )
            }
            Self::Schema { path, message } => {
                write!(
                    formatter,
                    "invalid COLMAP database schema {}: {message}",
                    path.display()
                )
            }
            Self::IncompatibleImages { missing, extra } => write!(
                formatter,
                "COLMAP database image set is incompatible (missing: {}; extra: {})",
                format_image_names(missing),
                format_image_names(extra)
            ),
            Self::CorruptFeature {
                image,
                table,
                message,
            } => write!(
                formatter,
                "invalid {table} feature row for {image}: {message}"
            ),
        }
    }
}

impl std::error::Error for FeatureCacheError {}

/// Inspect the supported images under `images_root` against a COLMAP database.
///
/// The database is opened with read-only SQLite flags.  Image names are
/// compared as normalized paths using `/`, matching COLMAP's `images.name`
/// convention.  A missing database image or feature row yields
/// [`FeatureCacheStatus::Incomplete`] so an interrupted extraction can resume;
/// an extra database image or malformed feature blob is an error because
/// reusing that database could silently combine stale data with the current
/// images.
pub(crate) fn inspect_feature_cache(
    images_root: &Path,
    database: &Path,
) -> Result<FeatureCacheReport, FeatureCacheError> {
    let expected_names = collect_image_names(images_root)?;
    let connection = Connection::open_with_flags(
        database,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|source| FeatureCacheError::Database {
        path: database.to_path_buf(),
        source,
    })?;

    validate_schema(&connection, database)?;
    let database_names = read_database_image_names(&connection, database)?;
    let extra = database_names
        .difference(&expected_names)
        .cloned()
        .collect::<Vec<_>>();
    if !extra.is_empty() {
        return Err(FeatureCacheError::IncompatibleImages {
            missing: Vec::new(),
            extra,
        });
    }

    let mut statement = connection
        .prepare(
            "SELECT images.name, \
                    keypoints.rows, keypoints.cols, length(keypoints.data), \
                    descriptors.rows, descriptors.cols, length(descriptors.data) \
             FROM images \
             LEFT JOIN keypoints ON keypoints.image_id = images.image_id \
             LEFT JOIN descriptors ON descriptors.image_id = images.image_id \
             ORDER BY images.name",
        )
        .map_err(|source| schema_error(database, source, "feature tables are unreadable"))?;
    let rows = statement
        .query_map([], read_feature_row)
        .map_err(|source| database_error(database, source))?;

    let mut completed = 0usize;
    for row in rows {
        let feature_row = row.map_err(|source| database_error(database, source))?;
        if validate_feature_row(&feature_row)? {
            completed = completed.saturating_add(1);
        }
    }

    let status = if !expected_names.is_empty() && completed == expected_names.len() {
        FeatureCacheStatus::Complete
    } else {
        FeatureCacheStatus::Incomplete
    };
    Ok(FeatureCacheReport {
        status,
        expected: expected_names.len(),
        completed,
    })
}

/// Remove stale image-pair matching results while retaining the image feature cache.
///
/// Matching output is derived from descriptors and can be rebuilt independently when
/// alignment inputs change.  Both deletes run in one immediate transaction so a
/// failure cannot leave only one matching table cleared.  The database is opened
/// read/write without `SQLITE_OPEN_CREATE`, so a missing database is reported rather
/// than silently created.
pub(crate) fn clear_matching_cache(database: &Path) -> Result<(), FeatureCacheError> {
    let mut connection = Connection::open_with_flags(
        database,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|source| FeatureCacheError::Database {
        path: database.to_path_buf(),
        source,
    })?;

    validate_tables(&connection, database, &["matches", "two_view_geometries"])?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|source| database_error(database, source))?;
    transaction
        .execute("DELETE FROM matches", [])
        .map_err(|source| database_error(database, source))?;
    transaction
        .execute("DELETE FROM two_view_geometries", [])
        .map_err(|source| database_error(database, source))?;
    transaction
        .commit()
        .map_err(|source| database_error(database, source))?;
    Ok(())
}

/// Return whether the database contains a configured multi-sensor rig.
///
/// COLMAP stores only non-reference sensors in `rig_sensors`, so any row means
/// the mapper would no longer see the independent-camera state required to
/// bootstrap unknown rig extrinsics.
pub(crate) fn database_has_nontrivial_rig(database: &Path) -> Result<bool, FeatureCacheError> {
    let connection = Connection::open_with_flags(
        database,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|source| FeatureCacheError::Database {
        path: database.to_path_buf(),
        source,
    })?;
    validate_tables(&connection, database, &["rig_sensors"])?;
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM rig_sensors LIMIT 1)",
            [],
            |row| row.get::<_, bool>(0),
        )
        .map_err(|source| database_error(database, source))
}

#[derive(Debug)]
struct FeatureRow {
    name: String,
    keypoints: Option<FeatureBlob>,
    descriptors: Option<FeatureBlob>,
}

#[derive(Debug)]
struct FeatureBlob {
    rows: i64,
    cols: i64,
    data_len: i64,
}

fn read_feature_row(row: &Row<'_>) -> rusqlite::Result<FeatureRow> {
    let name = row.get(0)?;
    let keypoints = read_blob(row, 1, 2, 3)?;
    let descriptors = read_blob(row, 4, 5, 6)?;
    Ok(FeatureRow {
        name,
        keypoints,
        descriptors,
    })
}

fn read_blob(
    row: &Row<'_>,
    rows_column: usize,
    cols_column: usize,
    data_column: usize,
) -> rusqlite::Result<Option<FeatureBlob>> {
    let rows = row.get::<_, Option<i64>>(rows_column)?;
    let cols = row.get::<_, Option<i64>>(cols_column)?;
    let data_len = row.get::<_, Option<i64>>(data_column)?.unwrap_or_default();
    Ok(match (rows, cols) {
        (Some(rows), Some(cols)) => Some(FeatureBlob {
            rows,
            cols,
            data_len,
        }),
        (None, None) => None,
        _ => Some(FeatureBlob {
            rows: -1,
            cols: -1,
            data_len,
        }),
    })
}

fn validate_feature_row(row: &FeatureRow) -> Result<bool, FeatureCacheError> {
    let (Some(keypoints), Some(descriptors)) = (&row.keypoints, &row.descriptors) else {
        return Ok(false);
    };
    if keypoints.rows < 0 || keypoints.cols < 0 {
        return Err(corrupt_error(
            &row.name,
            "keypoints",
            "rows and cols must be non-negative",
        ));
    }
    if descriptors.rows < 0 || descriptors.cols < 0 {
        return Err(corrupt_error(
            &row.name,
            "descriptors",
            "rows and cols must be non-negative",
        ));
    }
    if !matches!(keypoints.cols, 2 | 4 | 6) {
        return Err(corrupt_error(
            &row.name,
            "keypoints",
            "cols must be 2, 4, or 6",
        ));
    }
    if descriptors.cols != SIFT_DESCRIPTOR_COLS {
        return Err(corrupt_error(
            &row.name,
            "descriptors",
            "SIFT descriptor cols must be 128",
        ));
    }
    if keypoints.rows != descriptors.rows {
        return Err(corrupt_error(
            &row.name,
            "features",
            "keypoint and descriptor row counts differ",
        ));
    }
    let keypoint_bytes = checked_blob_bytes(
        keypoints.rows,
        keypoints.cols,
        KEYPOINT_SCALAR_BYTES,
        &row.name,
        "keypoints",
    )?;
    let descriptor_bytes = checked_blob_bytes(
        descriptors.rows,
        descriptors.cols,
        DESCRIPTOR_SCALAR_BYTES,
        &row.name,
        "descriptors",
    )?;
    let keypoint_data_len = u64::try_from(keypoints.data_len)
        .map_err(|_| corrupt_error(&row.name, "keypoints", "blob length must be non-negative"))?;
    let descriptor_data_len = u64::try_from(descriptors.data_len)
        .map_err(|_| corrupt_error(&row.name, "descriptors", "blob length must be non-negative"))?;
    if keypoint_data_len != keypoint_bytes {
        return Err(corrupt_error(
            &row.name,
            "keypoints",
            format!(
                "blob length is {}, expected {keypoint_bytes}",
                keypoint_data_len
            ),
        ));
    }
    if descriptor_data_len != descriptor_bytes {
        return Err(corrupt_error(
            &row.name,
            "descriptors",
            format!(
                "blob length is {}, expected {descriptor_bytes}",
                descriptor_data_len
            ),
        ));
    }
    Ok(true)
}

fn checked_blob_bytes(
    rows: i64,
    cols: i64,
    scalar_bytes: u64,
    image: &str,
    table: &'static str,
) -> Result<u64, FeatureCacheError> {
    let rows = u64::try_from(rows).map_err(|_| {
        corrupt_error(
            image,
            table,
            "rows cannot be represented as an unsigned value",
        )
    })?;
    let cols = u64::try_from(cols).map_err(|_| {
        corrupt_error(
            image,
            table,
            "cols cannot be represented as an unsigned value",
        )
    })?;
    rows.checked_mul(cols)
        .and_then(|bytes| bytes.checked_mul(scalar_bytes))
        .ok_or_else(|| corrupt_error(image, table, "rows, cols, and scalar size overflow"))
}

fn collect_image_names(images_root: &Path) -> Result<BTreeSet<String>, FeatureCacheError> {
    let metadata = fs::metadata(images_root).map_err(|error| FeatureCacheError::ImageRoot {
        path: images_root.to_path_buf(),
        message: error.to_string(),
    })?;
    if !metadata.is_dir() {
        return Err(FeatureCacheError::ImageRoot {
            path: images_root.to_path_buf(),
            message: "path is not a directory".to_owned(),
        });
    }
    let mut names = BTreeSet::new();
    collect_image_names_recursive(images_root, images_root, &mut names)?;
    Ok(names)
}

fn collect_image_names_recursive(
    root: &Path,
    directory: &Path,
    names: &mut BTreeSet<String>,
) -> Result<(), FeatureCacheError> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| image_root_error(root, error.to_string()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| image_root_error(root, error.to_string()))?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| image_root_error(root, error.to_string()))?;
        if file_type.is_dir() {
            collect_image_names_recursive(root, &path, names)?;
        } else if file_type.is_file() && is_supported_image(&path) {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| image_root_error(root, error.to_string()))?
                .to_string_lossy()
                .replace('\\', "/");
            names.insert(relative);
        }
    }
    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            SUPPORTED_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        })
}

fn validate_schema(connection: &Connection, database: &Path) -> Result<(), FeatureCacheError> {
    validate_tables(
        connection,
        database,
        &["images", "keypoints", "descriptors"],
    )
}

fn validate_tables(
    connection: &Connection,
    database: &Path,
    tables: &[&str],
) -> Result<(), FeatureCacheError> {
    for table in tables {
        let exists = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
                [table],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|source| database_error(database, source))?;
        if exists == 0 {
            return Err(FeatureCacheError::Schema {
                path: database.to_path_buf(),
                message: format!("missing {table} table"),
            });
        }
    }
    Ok(())
}

fn read_database_image_names(
    connection: &Connection,
    database: &Path,
) -> Result<BTreeSet<String>, FeatureCacheError> {
    let mut statement = connection
        .prepare("SELECT name FROM images ORDER BY name")
        .map_err(|source| schema_error(database, source, "images table is unreadable"))?;
    let names = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|source| database_error(database, source))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| database_error(database, source))?;
    let mut unique_names = BTreeSet::new();
    for name in names {
        if !unique_names.insert(name.clone()) {
            return Err(FeatureCacheError::IncompatibleImages {
                missing: Vec::new(),
                extra: vec![name],
            });
        }
    }
    Ok(unique_names)
}

fn image_root_error(path: &Path, message: String) -> FeatureCacheError {
    FeatureCacheError::ImageRoot {
        path: path.to_path_buf(),
        message,
    }
}

fn database_error(path: &Path, source: rusqlite::Error) -> FeatureCacheError {
    FeatureCacheError::Database {
        path: path.to_path_buf(),
        source,
    }
}

fn schema_error(path: &Path, source: rusqlite::Error, message: &str) -> FeatureCacheError {
    FeatureCacheError::Schema {
        path: path.to_path_buf(),
        message: format!("{message}: {source}"),
    }
}

fn corrupt_error(
    image: &str,
    table: &'static str,
    message: impl Into<String>,
) -> FeatureCacheError {
    FeatureCacheError::CorruptFeature {
        image: image.to_owned(),
        table,
        message: message.into(),
    }
}

fn format_image_names(names: &[String]) -> String {
    if names.is_empty() {
        return "none".to_owned();
    }
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::{
        clear_matching_cache, database_has_nontrivial_rig, inspect_feature_cache,
        FeatureCacheError, FeatureCacheStatus, SIFT_DESCRIPTOR_COLS,
    };
    use rusqlite::{params, Connection};
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    fn fixture() -> (TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let images = directory.path().join("images");
        fs::create_dir_all(images.join("lens0")).unwrap();
        fs::create_dir_all(images.join("nested/lens1")).unwrap();
        fs::write(images.join("lens0/frame.png"), b"png").unwrap();
        fs::write(images.join("nested/lens1/frame.JpG"), b"jpg").unwrap();
        fs::write(images.join("ignored.txt"), b"not an image").unwrap();
        let database = directory.path().join("database.db");
        create_database(&database);
        (directory, database)
    }

    fn create_database(path: &Path) {
        let connection = Connection::open(path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE images (
                    image_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    camera_id INTEGER NOT NULL
                 );
                 CREATE TABLE keypoints (
                    image_id INTEGER PRIMARY KEY,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                 );
                 CREATE TABLE descriptors (
                    image_id INTEGER PRIMARY KEY,
                    type INTEGER NOT NULL DEFAULT 0,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                 );
                 CREATE TABLE matches (
                    pair_id INTEGER PRIMARY KEY,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                 );
                 CREATE TABLE two_view_geometries (
                    pair_id INTEGER PRIMARY KEY,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                 );
                 CREATE TABLE rig_sensors (
                    rig_id INTEGER NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    sensor_type INTEGER NOT NULL,
                    sensor_from_rig BLOB
                 );",
            )
            .unwrap();
    }

    #[test]
    fn detects_only_configured_non_reference_rig_sensors() {
        let (_directory, database) = fixture();
        assert!(!database_has_nontrivial_rig(&database).unwrap());

        let connection = open_fixture_database(&database);
        connection
            .execute(
                "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type) VALUES (1, 2, 0)",
                [],
            )
            .unwrap();
        drop(connection);

        assert!(database_has_nontrivial_rig(&database).unwrap());
    }

    fn insert_image(connection: &Connection, id: i64, name: &str) {
        connection
            .execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?1, ?2, 1)",
                params![id, name],
            )
            .unwrap();
    }

    fn insert_features(
        connection: &Connection,
        id: i64,
        rows: i64,
        keypoint_cols: i64,
        keypoint_data_len: usize,
        descriptor_rows: i64,
        descriptor_data_len: usize,
    ) {
        insert_features_with_data(
            connection,
            id,
            rows,
            keypoint_cols,
            vec![0; keypoint_data_len],
            descriptor_rows,
            vec![0; descriptor_data_len],
        );
    }

    fn insert_features_with_data(
        connection: &Connection,
        id: i64,
        rows: i64,
        keypoint_cols: i64,
        keypoint_data: Vec<u8>,
        descriptor_rows: i64,
        descriptor_data: Vec<u8>,
    ) {
        connection
            .execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?1, ?2, ?3, ?4)",
                params![id, rows, keypoint_cols, keypoint_data],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO descriptors(image_id, type, rows, cols, data) VALUES (?1, 0, ?2, ?3, ?4)",
                params![id, descriptor_rows, SIFT_DESCRIPTOR_COLS, descriptor_data],
            )
            .unwrap();
    }

    fn open_fixture_database(path: &Path) -> Connection {
        Connection::open(path).unwrap()
    }

    #[test]
    fn complete_cache_accepts_zero_feature_rows() {
        let (directory, database) = fixture();
        let connection = open_fixture_database(&database);
        insert_image(&connection, 1, "lens0/frame.png");
        insert_image(&connection, 2, "nested/lens1/frame.JpG");
        insert_features(&connection, 1, 0, 6, 0, 0, 0);
        insert_features(&connection, 2, 0, 4, 0, 0, 0);
        drop(connection);

        let report = inspect_feature_cache(&directory.path().join("images"), &database).unwrap();
        assert_eq!(report.status, FeatureCacheStatus::Complete);
        assert_eq!(report.expected, 2);
        assert_eq!(report.completed, 2);
        assert!(report.is_complete());
    }

    #[test]
    fn partial_cache_reports_completed_images_without_error() {
        let (directory, database) = fixture();
        let connection = open_fixture_database(&database);
        insert_image(&connection, 1, "lens0/frame.png");
        insert_image(&connection, 2, "nested/lens1/frame.JpG");
        insert_features(&connection, 1, 1, 4, 16, 1, 128);
        drop(connection);

        let report = inspect_feature_cache(&directory.path().join("images"), &database).unwrap();
        assert_eq!(report.status, FeatureCacheStatus::Incomplete);
        assert_eq!(report.expected, 2);
        assert_eq!(report.completed, 1);
    }

    #[test]
    fn corrupt_cache_rejects_wrong_blob_length() {
        let (directory, database) = fixture();
        let connection = open_fixture_database(&database);
        insert_image(&connection, 1, "lens0/frame.png");
        insert_image(&connection, 2, "nested/lens1/frame.JpG");
        insert_features(&connection, 1, 1, 4, 4, 1, 128);
        insert_features(&connection, 2, 1, 4, 16, 1, 128);
        drop(connection);

        let error = inspect_feature_cache(&directory.path().join("images"), &database).unwrap_err();
        assert!(matches!(
            error,
            FeatureCacheError::CorruptFeature {
                table: "keypoints",
                ..
            }
        ));
    }

    #[test]
    fn extra_database_image_is_incompatible() {
        let (directory, database) = fixture();
        let connection = open_fixture_database(&database);
        insert_image(&connection, 1, "lens0/frame.png");
        insert_image(&connection, 2, "nested/lens1/frame.JpG");
        insert_image(&connection, 3, "lens0/stale.jpg");
        insert_features(&connection, 1, 0, 6, 0, 0, 0);
        insert_features(&connection, 2, 0, 6, 0, 0, 0);
        insert_features(&connection, 3, 0, 6, 0, 0, 0);
        drop(connection);

        let error = inspect_feature_cache(&directory.path().join("images"), &database).unwrap_err();
        match error {
            FeatureCacheError::IncompatibleImages { missing, extra } => {
                assert!(missing.is_empty());
                assert_eq!(extra, vec!["lens0/stale.jpg"]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn clear_matching_cache_preserves_feature_tables() {
        let (_directory, database) = fixture();
        let connection = open_fixture_database(&database);
        insert_image(&connection, 1, "lens0/frame.png");
        insert_features(&connection, 1, 0, 6, 0, 0, 0);
        connection
            .execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?1, 1, 2, ?2)",
                params![1_i64, vec![0_u8; 2]],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO two_view_geometries(pair_id, rows, cols, data) VALUES (?1, 1, 3, ?2)",
                params![1_i64, vec![0_u8; 3]],
            )
            .unwrap();
        drop(connection);

        clear_matching_cache(&database).unwrap();

        let connection = open_fixture_database(&database);
        for table in ["matches", "two_view_geometries"] {
            let count: i64 = connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 0, "{table} should be empty");
        }
        for table in ["images", "keypoints", "descriptors"] {
            let count: i64 = connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 1, "{table} should retain its row");
        }
    }

    #[test]
    fn clear_matching_cache_rolls_back_if_second_delete_fails() {
        let (_directory, database) = fixture();
        let connection = open_fixture_database(&database);
        connection
            .execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?1, 1, 2, ?2)",
                params![1_i64, vec![0_u8; 2]],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO two_view_geometries(pair_id, rows, cols, data) VALUES (?1, 1, 3, ?2)",
                params![1_i64, vec![0_u8; 3]],
            )
            .unwrap();
        connection
            .execute_batch(
                "CREATE TRIGGER reject_two_view_delete
                 BEFORE DELETE ON two_view_geometries
                 BEGIN
                    SELECT RAISE(ABORT, 'test failure');
                 END;",
            )
            .unwrap();
        drop(connection);

        assert!(clear_matching_cache(&database).is_err());

        let connection = open_fixture_database(&database);
        for table in ["matches", "two_view_geometries"] {
            let count: i64 = connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 1, "{table} should be restored by rollback");
        }
    }

    #[test]
    fn missing_database_image_is_incomplete() {
        let (directory, database) = fixture();
        let connection = open_fixture_database(&database);
        insert_image(&connection, 1, "lens0/frame.png");
        insert_features(&connection, 1, 0, 6, 0, 0, 0);
        drop(connection);

        let report = inspect_feature_cache(&directory.path().join("images"), &database).unwrap();
        assert_eq!(report.status, FeatureCacheStatus::Incomplete);
        assert_eq!(report.expected, 2);
        assert_eq!(report.completed, 1);
    }

    #[test]
    fn empty_image_root_is_not_complete() {
        let (directory, database) = fixture();
        let empty_images = directory.path().join("empty-images");
        fs::create_dir_all(&empty_images).unwrap();
        let report = inspect_feature_cache(&empty_images, &database).unwrap();
        assert_eq!(report.status, FeatureCacheStatus::Incomplete);
        assert_eq!(report.expected, 0);
        assert_eq!(report.completed, 0);
    }
}
