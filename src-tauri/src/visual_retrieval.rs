//! Deterministic, low-cost visual retrieval helpers.
//!
//! The retrieval implementation deliberately has no learned-model dependency.
//! It uses a small grayscale image and its first-order gradients as a global
//! descriptor.  This is intended to filter cross-recording anchor candidates;
//! it is not a replacement for COLMAP feature matching.
//!
//! The second half of this module contains spherical-cap overlap helpers for
//! calibrated fisheye cameras.  The quaternion convention is scalar-first
//! `(w, x, y, z)`, and `q_a_from_b` rotates a vector expressed in frame `b`
//! into frame `a`.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use image::{imageops::FilterType, DynamicImage, GenericImageView};
use serde::Serialize;

/// Scalar-first quaternion `(w, x, y, z)`.
pub type Quaternion = [f64; 4];

/// Three-dimensional vector used by the camera-orientation helpers.
pub type Vec3 = [f64; 3];

const MAX_DESCRIPTOR_IMAGE_PIXELS: u64 = 100_000_000;
const VECTOR_EPSILON: f64 = 1.0e-12;

/// Descriptor extraction parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DescriptorConfig {
    /// Width of the resized grayscale image.
    pub width: u32,
    /// Height of the resized grayscale image.
    pub height: u32,
    /// Append horizontal and vertical first-order gradients.
    pub include_gradients: bool,
}

impl Default for DescriptorConfig {
    fn default() -> Self {
        Self {
            width: 32,
            height: 32,
            include_gradients: true,
        }
    }
}

impl DescriptorConfig {
    fn validate(self) -> Result<Self, DescriptorError> {
        if self.width == 0 || self.height == 0 || self.width > 128 || self.height > 128 {
            return Err(DescriptorError::InvalidConfig {
                width: self.width,
                height: self.height,
            });
        }
        Ok(self)
    }
}

/// Errors raised while decoding or building one descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorError {
    /// The descriptor dimensions are outside the intentionally small limit.
    InvalidConfig { width: u32, height: u32 },
    /// The source image is too large to decode safely for this use case.
    ImageTooLarge { width: u32, height: u32 },
    /// The image has no usable visual variation.
    Degenerate,
    /// The image decoder reported an error.
    Decode(String),
}

impl fmt::Display for DescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { width, height } => {
                write!(formatter, "invalid descriptor dimensions {width}x{height}")
            }
            Self::ImageTooLarge { width, height } => {
                write!(
                    formatter,
                    "image dimensions {width}x{height} exceed the retrieval limit"
                )
            }
            Self::Degenerate => formatter.write_str("image has no usable visual variation"),
            Self::Decode(message) => write!(formatter, "image decode failed: {message}"),
        }
    }
}

impl Error for DescriptorError {}

/// A unit-normalized global descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalDescriptor {
    values: Vec<f32>,
}

impl GlobalDescriptor {
    /// Return the normalized descriptor values.
    #[allow(dead_code)]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Return the cosine distance in `[0, 2]`, or infinity for incompatible
    /// descriptors.  Unit normalization makes this a cheap dot product.
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        if self.values.len() != other.values.len() {
            return f32::INFINITY;
        }
        let dot = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(left, right)| f64::from(*left) * f64::from(*right))
            .sum::<f64>();
        (1.0 - dot.clamp(-1.0, 1.0)) as f32
    }
}

/// Build a descriptor from an already decoded image.
pub fn descriptor_from_image(
    image: &DynamicImage,
    config: DescriptorConfig,
) -> Result<GlobalDescriptor, DescriptorError> {
    let config = config.validate()?;
    let (width, height) = image.dimensions();
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or(DescriptorError::ImageTooLarge { width, height })?;
    if pixels > MAX_DESCRIPTOR_IMAGE_PIXELS {
        return Err(DescriptorError::ImageTooLarge { width, height });
    }

    // `resize_exact` is intentional: every source contributes to the same
    // fixed descriptor layout, independent of its aspect ratio.  The image
    // crate documents this as the API that does not preserve aspect ratio.
    let gray = image
        .grayscale()
        .resize_exact(config.width, config.height, FilterType::Triangle)
        .to_luma8();
    let sample_count = (config.width as usize) * (config.height as usize);
    if sample_count == 0 {
        return Err(DescriptorError::Degenerate);
    }

    let pixels = gray
        .pixels()
        .map(|pixel| f64::from(pixel[0]) / 255.0)
        .collect::<Vec<_>>();
    let mean = pixels.iter().sum::<f64>() / pixels.len() as f64;
    let variance = pixels
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / pixels.len() as f64;
    let standard_deviation = variance.sqrt();
    if !standard_deviation.is_finite() || standard_deviation <= VECTOR_EPSILON {
        return Err(DescriptorError::Degenerate);
    }

    let feature_count = if config.include_gradients {
        sample_count * 3
    } else {
        sample_count
    };
    let mut values = Vec::with_capacity(feature_count);

    // Centered intensity preserves coarse layout while removing exposure
    // offsets.  Gradients add useful structure for otherwise similar frames.
    for value in &pixels {
        values.push(((*value - mean) / standard_deviation) as f32);
    }
    if config.include_gradients {
        let width = config.width as usize;
        let height = config.height as usize;
        for y in 0..height {
            for x in 0..width {
                let left = pixels[y * width + x.saturating_sub(1)];
                let right = pixels[y * width + (x + 1).min(width - 1)];
                values.push(((right - left) / standard_deviation) as f32);
            }
        }
        for y in 0..height {
            for x in 0..width {
                let top = pixels[y.saturating_sub(1) * width + x];
                let bottom = pixels[(y + 1).min(height - 1) * width + x];
                values.push(((bottom - top) / standard_deviation) as f32);
            }
        }
    }

    let norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= VECTOR_EPSILON {
        return Err(DescriptorError::Degenerate);
    }
    for value in &mut values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(GlobalDescriptor { values })
}

/// Decode one path and build its descriptor.
pub fn descriptor_from_path(
    path: impl AsRef<Path>,
    config: DescriptorConfig,
) -> Result<GlobalDescriptor, DescriptorError> {
    let path = path.as_ref();
    let image = image::open(path).map_err(|error| DescriptorError::Decode(error.to_string()))?;
    descriptor_from_image(&image, config)
}

/// One anchor frame considered for cross-recording retrieval.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalAnchor {
    /// Stable frame identifier (usually a sequence number or filename).
    pub frame_id: String,
    /// Full-resolution or low-resolution image path.
    pub path: PathBuf,
    /// Optional timestamp retained for the caller's frame-level graph.
    pub timestamp_ms: Option<f64>,
}

impl RetrievalAnchor {
    /// Construct an anchor whose ID is the filename when available.
    #[allow(dead_code)]
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let frame_id = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        Self {
            frame_id,
            path,
            timestamp_ms: None,
        }
    }
}

/// The anchors belonging to one recording/source.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalSource {
    pub source_id: String,
    pub anchors: Vec<RetrievalAnchor>,
}

/// Retrieval policy.  All limits are hard caps, even when the input contains
/// more recordings or anchors than usual.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalConfig {
    pub descriptor: DescriptorConfig,
    /// Number of anchors retained from each source before matching.
    pub max_anchors_per_source: usize,
    /// Maximum number of source pairs whose anchor sets are expanded.
    pub max_source_pairs: usize,
    /// Maximum frame-level candidates emitted for each source pair.
    pub max_frame_pairs_per_source_pair: usize,
    /// Number of nearest candidates retained when applying ratio tests.
    pub top_k_per_anchor: usize,
    /// Lowe-style ratio threshold.  A value below one is required to enable it.
    pub ratio_threshold: f32,
    /// Absolute descriptor-distance threshold.
    pub distance_threshold: f32,
    /// Source-summary distance threshold used before expanding anchor sets.
    pub source_distance_threshold: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            descriptor: DescriptorConfig::default(),
            max_anchors_per_source: 20,
            max_source_pairs: 128,
            max_frame_pairs_per_source_pair: 4,
            top_k_per_anchor: 2,
            ratio_threshold: 0.85,
            distance_threshold: 0.65,
            source_distance_threshold: 0.9,
        }
    }
}

impl RetrievalConfig {
    fn normalized(&self) -> Option<Self> {
        let descriptor = self.descriptor.validate().ok()?;
        if self.max_anchors_per_source == 0
            || self.max_source_pairs == 0
            || self.max_frame_pairs_per_source_pair == 0
            || self.top_k_per_anchor == 0
            || !self.ratio_threshold.is_finite()
            || self.ratio_threshold <= 0.0
            || self.ratio_threshold >= 1.0
            || !self.distance_threshold.is_finite()
            || self.distance_threshold < 0.0
            || self.distance_threshold > 2.0
            || !self.source_distance_threshold.is_finite()
            || self.source_distance_threshold < 0.0
            || self.source_distance_threshold > 2.0
        {
            return None;
        }
        Some(Self {
            descriptor,
            max_anchors_per_source: self.max_anchors_per_source.min(256),
            max_source_pairs: self.max_source_pairs.min(4096),
            max_frame_pairs_per_source_pair: self.max_frame_pairs_per_source_pair.min(256),
            top_k_per_anchor: self.top_k_per_anchor.min(16),
            ratio_threshold: self.ratio_threshold,
            distance_threshold: self.distance_threshold,
            source_distance_threshold: self.source_distance_threshold,
        })
    }
}

/// One descriptor failure.  Retrieval keeps going and reports failures so the
/// caller can fall back to the legacy cross-source anchor graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DescriptorFailure {
    pub source_id: String,
    pub frame_id: String,
    pub path: PathBuf,
    pub reason: String,
}

/// One retained frame-level retrieval candidate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameRetrievalCandidate {
    pub source_a: String,
    pub frame_a: String,
    pub path_a: PathBuf,
    pub source_b: String,
    pub frame_b: String,
    pub path_b: PathBuf,
    pub timestamp_a_ms: Option<f64>,
    pub timestamp_b_ms: Option<f64>,
    pub distance: f32,
    pub ratio_a: Option<f32>,
    pub ratio_b: Option<f32>,
    pub mutual: bool,
}

/// Expanded match group for one source pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourcePairRetrieval {
    pub source_a: String,
    pub source_b: String,
    pub source_distance: f32,
    pub matches: Vec<FrameRetrievalCandidate>,
}

/// Result of bounded cross-source retrieval.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RetrievalReport {
    pub source_pairs: Vec<SourcePairRetrieval>,
    pub failed_descriptors: Vec<DescriptorFailure>,
    /// Number of source pairs expanded to anchor-level matching after the
    /// configured hard cap and source-summary threshold were applied.
    pub evaluated_source_pair_count: usize,
    /// Set when the caller should use its legacy cross-source graph as a safe
    /// fallback (for example, a descriptor failure, no descriptors, or no
    /// accepted candidates).
    pub fallback_to_legacy: bool,
}

impl RetrievalReport {
    pub fn requires_fallback(&self) -> bool {
        self.fallback_to_legacy
    }

    pub fn frame_candidates(&self) -> impl Iterator<Item = &FrameRetrievalCandidate> {
        self.source_pairs
            .iter()
            .flat_map(|pair| pair.matches.iter())
    }

    pub fn make_paths_relative_to(&mut self, root: &Path) {
        fn relative(path: &Path, root: &Path) -> PathBuf {
            path.strip_prefix(root)
                .map(Path::to_path_buf)
                .or_else(|_| path.file_name().map(PathBuf::from).ok_or(()))
                .unwrap_or_else(|_| PathBuf::from("redacted"))
        }

        for pair in &mut self.source_pairs {
            for candidate in &mut pair.matches {
                candidate.path_a = relative(&candidate.path_a, root);
                candidate.path_b = relative(&candidate.path_b, root);
            }
        }
        for failure in &mut self.failed_descriptors {
            failure.path = relative(&failure.path, root);
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedAnchor {
    anchor: RetrievalAnchor,
    descriptor: GlobalDescriptor,
}

#[derive(Debug, Clone)]
struct PreparedSource {
    source: RetrievalSource,
    anchors: Vec<PreparedAnchor>,
    summary: GlobalDescriptor,
}

/// Run deterministic bounded visual retrieval over all source pairs.
pub fn retrieve_cross_source_candidates(
    sources: &[RetrievalSource],
    config: &RetrievalConfig,
) -> RetrievalReport {
    let Some(config) = config.normalized() else {
        return RetrievalReport {
            fallback_to_legacy: true,
            ..RetrievalReport::default()
        };
    };

    let mut failures = Vec::new();
    let mut prepared = Vec::new();
    for source in sources {
        let mut anchors = Vec::new();
        for anchor in source.anchors.iter().take(config.max_anchors_per_source) {
            match descriptor_from_path(&anchor.path, config.descriptor) {
                Ok(descriptor) => anchors.push(PreparedAnchor {
                    anchor: anchor.clone(),
                    descriptor,
                }),
                Err(error) => failures.push(DescriptorFailure {
                    source_id: source.source_id.clone(),
                    frame_id: anchor.frame_id.clone(),
                    path: anchor.path.clone(),
                    reason: error.to_string(),
                }),
            }
        }
        let Some(summary) = average_descriptors(&anchors) else {
            continue;
        };
        prepared.push(PreparedSource {
            source: source.clone(),
            anchors,
            summary,
        });
    }

    let mut source_candidates = Vec::new();
    for left in 0..prepared.len() {
        for right in (left + 1)..prepared.len() {
            let distance = prepared[left]
                .summary
                .cosine_distance(&prepared[right].summary);
            if distance.is_finite() && distance <= config.source_distance_threshold {
                source_candidates.push((
                    distance,
                    left,
                    right,
                    prepared[left].source.source_id.clone(),
                    prepared[right].source.source_id.clone(),
                ));
            }
        }
    }
    source_candidates.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.3.cmp(&right.3))
            .then_with(|| left.4.cmp(&right.4))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    source_candidates.truncate(config.max_source_pairs);
    let evaluated_source_pair_count = source_candidates.len();

    let mut source_pairs = Vec::new();
    for (source_distance, left, right, _, _) in source_candidates {
        let matches = match_source_anchors(&prepared[left], &prepared[right], &config);
        if matches.is_empty() {
            continue;
        }
        source_pairs.push(SourcePairRetrieval {
            source_a: prepared[left].source.source_id.clone(),
            source_b: prepared[right].source.source_id.clone(),
            source_distance,
            matches,
        });
    }

    RetrievalReport {
        evaluated_source_pair_count,
        // A partial descriptor failure can silently disconnect one recording
        // if the caller uses only the filtered graph.  Prefer the complete
        // legacy anchor grid whenever any input was unreadable.
        fallback_to_legacy: source_pairs.is_empty() || !failures.is_empty(),
        source_pairs,
        failed_descriptors: failures,
    }
}

fn average_descriptors(anchors: &[PreparedAnchor]) -> Option<GlobalDescriptor> {
    let first = anchors.first()?;
    let length = first.descriptor.values.len();
    if length == 0
        || anchors
            .iter()
            .any(|anchor| anchor.descriptor.values.len() != length)
    {
        return None;
    }
    let mut values = vec![0.0_f32; length];
    for anchor in anchors {
        for (sum, value) in values.iter_mut().zip(&anchor.descriptor.values) {
            *sum += *value;
        }
    }
    let norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= VECTOR_EPSILON {
        return None;
    }
    for value in &mut values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Some(GlobalDescriptor { values })
}

fn match_source_anchors(
    left: &PreparedSource,
    right: &PreparedSource,
    config: &RetrievalConfig,
) -> Vec<FrameRetrievalCandidate> {
    if left.anchors.is_empty() || right.anchors.is_empty() {
        return Vec::new();
    }

    let mut left_to_right = Vec::with_capacity(left.anchors.len());
    for left_anchor in &left.anchors {
        let mut scores = right
            .anchors
            .iter()
            .enumerate()
            .map(|(index, right_anchor)| {
                (
                    left_anchor
                        .descriptor
                        .cosine_distance(&right_anchor.descriptor),
                    index,
                )
            })
            .collect::<Vec<_>>();
        scores.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        left_to_right.push(scores);
    }
    let mut right_to_left = Vec::with_capacity(right.anchors.len());
    for right_anchor in &right.anchors {
        let mut scores = left
            .anchors
            .iter()
            .enumerate()
            .map(|(index, left_anchor)| {
                (
                    right_anchor
                        .descriptor
                        .cosine_distance(&left_anchor.descriptor),
                    index,
                )
            })
            .collect::<Vec<_>>();
        scores.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        right_to_left.push(scores);
    }

    let mut candidates = Vec::new();
    for (left_index, scores) in left_to_right.iter().enumerate() {
        for &(distance, right_index) in scores.iter().take(config.top_k_per_anchor) {
            if !distance.is_finite() || distance > config.distance_threshold {
                continue;
            }
            let ratio_a = ratio_from_scores(scores);
            let right_scores = &right_to_left[right_index];
            let ratio_b = ratio_from_scores(right_scores);
            let mutual = right_scores
                .first()
                .is_some_and(|(_, best_left)| *best_left == left_index);
            // Requiring both directional ratios avoids a one-sided false
            // positive.  Mutual nearest-neighbour matches remain valid even
            // when an anchor's second neighbour is tied.
            let ratio_match = ratio_a.is_some_and(|ratio| ratio <= config.ratio_threshold)
                && ratio_b.is_some_and(|ratio| ratio <= config.ratio_threshold);
            if !(mutual || ratio_match) {
                continue;
            }
            candidates.push(FrameRetrievalCandidate {
                source_a: left.source.source_id.clone(),
                frame_a: left.anchors[left_index].anchor.frame_id.clone(),
                path_a: left.anchors[left_index].anchor.path.clone(),
                source_b: right.source.source_id.clone(),
                frame_b: right.anchors[right_index].anchor.frame_id.clone(),
                path_b: right.anchors[right_index].anchor.path.clone(),
                timestamp_a_ms: left.anchors[left_index].anchor.timestamp_ms,
                timestamp_b_ms: right.anchors[right_index].anchor.timestamp_ms,
                distance,
                ratio_a,
                ratio_b,
                mutual,
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.frame_a.cmp(&right.frame_a))
            .then_with(|| left.frame_b.cmp(&right.frame_b))
            .then_with(|| left.path_a.cmp(&right.path_a))
            .then_with(|| left.path_b.cmp(&right.path_b))
    });
    candidates.dedup_by(|left, right| {
        left.frame_a == right.frame_a
            && left.frame_b == right.frame_b
            && left.path_a == right.path_a
            && left.path_b == right.path_b
    });
    candidates.truncate(config.max_frame_pairs_per_source_pair);
    candidates
}

fn ratio_from_scores(scores: &[(f32, usize)]) -> Option<f32> {
    let best = scores.first()?.0;
    let second = scores.get(1)?.0;
    if !best.is_finite() || !second.is_finite() || second <= f32::EPSILON {
        return None;
    }
    Some((best / second).clamp(0.0, 1.0))
}

/// Normalize a scalar-first quaternion.
pub fn normalize_quaternion(quaternion: Quaternion) -> Option<Quaternion> {
    if quaternion.iter().any(|component| !component.is_finite()) {
        return None;
    }
    let norm = quaternion
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= VECTOR_EPSILON {
        return None;
    }
    Some(quaternion.map(|component| component / norm))
}

/// Compose `q_a_from_b * q_b_from_c`.
#[allow(dead_code)]
pub fn multiply_quaternions(a_from_b: Quaternion, b_from_c: Quaternion) -> Option<Quaternion> {
    let a = normalize_quaternion(a_from_b)?;
    let b = normalize_quaternion(b_from_c)?;
    normalize_quaternion([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])
}

/// Rotate one vector by a normalized or non-normalized quaternion.
pub fn rotate_vector(quaternion: Quaternion, vector: Vec3) -> Option<Vec3> {
    let q = normalize_quaternion(quaternion)?;
    if vector.iter().any(|component| !component.is_finite()) {
        return None;
    }
    let qv = [q[1], q[2], q[3]];
    let uv = cross(qv, vector);
    let uuv = cross(qv, uv);
    Some([
        vector[0] + 2.0 * (q[0] * uv[0] + uuv[0]),
        vector[1] + 2.0 * (q[0] * uv[1] + uuv[1]),
        vector[2] + 2.0 * (q[0] * uv[2] + uuv[2]),
    ])
}

/// Return the angle between two unit-vector directions in degrees.
pub fn vector_angle_deg(left: Vec3, right: Vec3) -> Option<f64> {
    if left.iter().any(|component| !component.is_finite())
        || right.iter().any(|component| !component.is_finite())
    {
        return None;
    }
    let left_norm = dot(left, left).sqrt();
    let right_norm = dot(right, right).sqrt();
    if left_norm <= VECTOR_EPSILON || right_norm <= VECTOR_EPSILON {
        return None;
    }
    Some(
        (dot(left, right) / (left_norm * right_norm))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees(),
    )
}

/// Return the camera optical axis (`+Z`) in world coordinates.
#[allow(dead_code)]
pub fn camera_forward_world(
    world_from_rig: Quaternion,
    rig_from_camera: Quaternion,
) -> Option<Vec3> {
    let world_from_camera = multiply_quaternions(world_from_rig, rig_from_camera)?;
    rotate_vector(world_from_camera, [0.0, 0.0, 1.0])
}

/// Test whether two calibrated fisheye spherical caps overlap.
#[allow(dead_code)]
pub fn fisheye_views_overlap(
    world_from_rig: Quaternion,
    rig_from_camera_a: Quaternion,
    rig_from_camera_b: Quaternion,
    half_fov_a_deg: f64,
    half_fov_b_deg: f64,
) -> Option<bool> {
    let left_half = validate_half_fov(half_fov_a_deg)?;
    let right_half = validate_half_fov(half_fov_b_deg)?;
    let left = camera_forward_world(world_from_rig, rig_from_camera_a)?;
    let right = camera_forward_world(world_from_rig, rig_from_camera_b)?;
    let axis_angle = vector_angle_deg(left, right)?;
    Some(axis_angle <= left_half + right_half + 1.0e-9)
}

/// Test overlap when the calibrated chain is supplied as
/// `world_from_imu * imu_from_rig * rig_from_camera`.
#[allow(dead_code)]
pub fn fisheye_views_overlap_from_imu(
    world_from_imu: Quaternion,
    imu_from_rig: Quaternion,
    rig_from_camera_a: Quaternion,
    rig_from_camera_b: Quaternion,
    half_fov_a_deg: f64,
    half_fov_b_deg: f64,
) -> Option<bool> {
    let world_from_rig = multiply_quaternions(world_from_imu, imu_from_rig)?;
    fisheye_views_overlap(
        world_from_rig,
        rig_from_camera_a,
        rig_from_camera_b,
        half_fov_a_deg,
        half_fov_b_deg,
    )
}

/// Convenience form for equal 180-degree fisheyes (`half_fov = 90°`).
#[allow(dead_code)]
pub fn fisheye_views_overlap_equal_fov(
    world_from_rig: Quaternion,
    rig_from_camera_a: Quaternion,
    rig_from_camera_b: Quaternion,
    half_fov_deg: f64,
) -> Option<bool> {
    fisheye_views_overlap(
        world_from_rig,
        rig_from_camera_a,
        rig_from_camera_b,
        half_fov_deg,
        half_fov_deg,
    )
}

/// Test overlap when callers already have world-from-camera quaternions.
pub fn camera_views_overlap(
    world_from_camera_a: Quaternion,
    world_from_camera_b: Quaternion,
    half_fov_a_deg: f64,
    half_fov_b_deg: f64,
) -> Option<bool> {
    let left_half = validate_half_fov(half_fov_a_deg)?;
    let right_half = validate_half_fov(half_fov_b_deg)?;
    let left = rotate_vector(world_from_camera_a, [0.0, 0.0, 1.0])?;
    let right = rotate_vector(world_from_camera_b, [0.0, 0.0, 1.0])?;
    let axis_angle = vector_angle_deg(left, right)?;
    Some(axis_angle <= left_half + right_half + 1.0e-9)
}

fn validate_half_fov(value: f64) -> Option<f64> {
    (value.is_finite() && value > 0.0 && value <= 180.0).then_some(value)
}

fn dot(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use tempfile::tempdir;

    fn gradient_image(seed: u8) -> GrayImage {
        GrayImage::from_fn(64, 48, |x, y| {
            let value = (x.wrapping_mul(7) as u8)
                .wrapping_add(y.wrapping_mul(11) as u8)
                .wrapping_add(seed);
            Luma([value])
        })
    }

    fn save_anchor(directory: &Path, name: &str, seed: u8) -> RetrievalAnchor {
        let path = directory.join(name);
        gradient_image(seed).save(&path).unwrap();
        RetrievalAnchor {
            frame_id: name.to_owned(),
            path,
            timestamp_ms: None,
        }
    }

    fn identity() -> Quaternion {
        [1.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn descriptor_is_deterministic_and_unit_normalized() {
        let image = DynamicImage::ImageLuma8(gradient_image(3));
        let first = descriptor_from_image(&image, DescriptorConfig::default()).unwrap();
        let second = descriptor_from_image(&image, DescriptorConfig::default()).unwrap();
        assert_eq!(first, second);
        let norm = first
            .values()
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1.0e-6);
        assert!(first.cosine_distance(&second).abs() < 1.0e-6);
    }

    #[test]
    fn descriptor_rejects_uniform_images_and_invalid_config() {
        let image = DynamicImage::ImageLuma8(GrayImage::from_pixel(16, 16, Luma([42])));
        assert_eq!(
            descriptor_from_image(&image, DescriptorConfig::default()),
            Err(DescriptorError::Degenerate)
        );
        assert!(matches!(
            descriptor_from_image(
                &image,
                DescriptorConfig {
                    width: 0,
                    ..DescriptorConfig::default()
                }
            ),
            Err(DescriptorError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn retrieval_is_bounded_deterministic_and_reports_missing_inputs() {
        let directory = tempdir().unwrap();
        let mut source_a = RetrievalSource {
            source_id: "source-a".to_owned(),
            anchors: vec![save_anchor(directory.path(), "a0.png", 3)],
        };
        source_a.anchors.push(RetrievalAnchor {
            frame_id: "missing".to_owned(),
            path: directory.path().join("missing.png"),
            timestamp_ms: Some(20.0),
        });
        let source_b = RetrievalSource {
            source_id: "source-b".to_owned(),
            anchors: vec![save_anchor(directory.path(), "b0.png", 3)],
        };
        let source_c = RetrievalSource {
            source_id: "source-c".to_owned(),
            anchors: vec![save_anchor(directory.path(), "c0.png", 101)],
        };
        let config = RetrievalConfig {
            max_source_pairs: 1,
            max_frame_pairs_per_source_pair: 1,
            ..RetrievalConfig::default()
        };
        let first = retrieve_cross_source_candidates(
            &[source_a.clone(), source_b.clone(), source_c.clone()],
            &config,
        );
        let second = retrieve_cross_source_candidates(&[source_a, source_b, source_c], &config);
        assert_eq!(first, second);
        assert_eq!(first.evaluated_source_pair_count, 1);
        assert_eq!(first.source_pairs.len(), 1);
        assert!(first.source_pairs[0].matches.len() <= 1);
        assert_eq!(first.failed_descriptors.len(), 1);
        assert!(first.fallback_to_legacy);
    }

    #[test]
    fn retrieval_falls_back_when_every_descriptor_fails() {
        let directory = tempdir().unwrap();
        let absolute = directory.path().join("missing.jpg");
        let mut report = retrieve_cross_source_candidates(
            &[RetrievalSource {
                source_id: "bad".to_owned(),
                anchors: vec![RetrievalAnchor {
                    frame_id: "bad".to_owned(),
                    path: absolute,
                    timestamp_ms: None,
                }],
            }],
            &RetrievalConfig::default(),
        );
        assert!(report.fallback_to_legacy);
        assert!(report.source_pairs.is_empty());
        assert_eq!(report.failed_descriptors.len(), 1);
        report.make_paths_relative_to(directory.path());
        assert_eq!(report.failed_descriptors[0].path, PathBuf::from("missing.jpg"));
    }

    #[test]
    fn quaternion_composition_and_fisheye_overlap_are_deterministic() {
        let quarter_turn_z = [
            (std::f64::consts::FRAC_PI_4).cos(),
            0.0,
            0.0,
            (std::f64::consts::FRAC_PI_4).sin(),
        ];
        let forward = camera_forward_world(identity(), identity()).unwrap();
        assert_eq!(forward, [0.0, 0.0, 1.0]);
        assert!(
            fisheye_views_overlap_equal_fov(identity(), identity(), quarter_turn_z, 90.0).unwrap()
        );
        assert!(fisheye_views_overlap_from_imu(
            identity(),
            identity(),
            identity(),
            quarter_turn_z,
            90.0,
            90.0
        )
        .unwrap());
        assert!(
            !fisheye_views_overlap(identity(), identity(), [0.0, 1.0, 0.0, 0.0], 80.0, 80.0)
                .unwrap()
        );
        assert!(camera_views_overlap(identity(), quarter_turn_z, 45.0, 45.0).unwrap());
        assert!(fisheye_views_overlap_equal_fov(identity(), identity(), identity(), 90.0).unwrap());
        assert!(
            fisheye_views_overlap_equal_fov(identity(), identity(), identity(), 90.0).is_some()
        );
        assert!(fisheye_views_overlap_equal_fov(identity(), identity(), identity(), 0.0).is_none());
    }
}
