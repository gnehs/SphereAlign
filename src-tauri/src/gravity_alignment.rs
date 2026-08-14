//! Estimate a global, gravity-only rotation for a COLMAP reconstruction.
//!
//! The calibrated gravity observations are expressed in each camera frame.
//! COLMAP image quaternions map world coordinates into that camera frame, so
//! applying the inverse image rotation recovers one gravity observation in the
//! arbitrary reconstructed world.  The final shortest-arc rotation aligns
//! that direction with LichtFeld Studio's `-Y` down axis without introducing
//! an additional twist around gravity (yaw).

use crate::colmap_priors::GravityPriorInput;
use crate::reconstruction_benchmark::ColmapTextModel;
use serde::Serialize;
use std::collections::BTreeMap;

const MIN_MATCHED_OBSERVATIONS: usize = 8;
const MIN_MATCHED_COVERAGE_RATIO: f64 = 0.8;
const MIN_INLIER_RATIO: f64 = 0.7;
const MAX_MEDIAN_RESIDUAL_DEG: f64 = 10.0;
const MIN_INLIER_THRESHOLD_DEG: f64 = 5.0;
const MAX_INLIER_THRESHOLD_DEG: f64 = 20.0;
const ALIGNMENT_EPSILON_DEG: f64 = 0.05;

pub const TARGET_UP_AXIS: [f64; 3] = [0.0, 1.0, 0.0];
pub const TARGET_GRAVITY_AXIS: [f64; 3] = [0.0, -1.0, 0.0];

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GravityAlignmentEstimate {
    pub registered_image_count: usize,
    pub matched_observation_count: usize,
    pub inlier_observation_count: usize,
    pub matched_coverage_ratio: f64,
    pub inlier_ratio: f64,
    pub median_residual_deg: f64,
    pub inlier_threshold_deg: f64,
    pub gravity_world_before: [f64; 3],
    pub gravity_world_after: [f64; 3],
    pub target_up_axis: [f64; 3],
    pub rotation_wxyz: [f64; 4],
    pub rotation_angle_deg: f64,
    pub already_aligned: bool,
}

pub fn estimate_gravity_alignment(
    model: &ColmapTextModel,
    priors: &[GravityPriorInput],
) -> Result<GravityAlignmentEstimate, String> {
    let images = model
        .images
        .iter()
        .map(|image| (image.name.as_str(), image.qvec_camera_from_world))
        .collect::<BTreeMap<_, _>>();
    if images.is_empty() {
        return Err("COLMAP 模型沒有已註冊影像，無法估計重力方向".to_owned());
    }

    let mut observations = Vec::new();
    for prior in priors {
        let Some(camera_from_world) = images.get(prior.image_name.as_str()) else {
            continue;
        };
        let Some(gravity_camera) = normalize(prior.gravity) else {
            continue;
        };
        let Some(world_from_camera) = conjugate(*camera_from_world) else {
            continue;
        };
        if let Some(gravity_world) = rotate_vector(world_from_camera, gravity_camera) {
            observations.push(gravity_world);
        }
    }
    if observations.len() < MIN_MATCHED_OBSERVATIONS {
        return Err(format!(
            "可用的已註冊影像重力樣本不足：{} < {}",
            observations.len(),
            MIN_MATCHED_OBSERVATIONS
        ));
    }
    let matched_coverage_ratio = observations.len() as f64 / images.len() as f64;
    if matched_coverage_ratio < MIN_MATCHED_COVERAGE_RATIO {
        return Err(format!(
            "已註冊影像的重力 coverage 不足：{:.1}% < {:.1}%",
            matched_coverage_ratio * 100.0,
            MIN_MATCHED_COVERAGE_RATIO * 100.0
        ));
    }

    // Component-wise medians provide a deterministic seed that remains stable
    // when a minority of telemetry samples use a bad timestamp or convention.
    let seed = normalize([
        median(observations.iter().map(|value| value[0]).collect()),
        median(observations.iter().map(|value| value[1]).collect()),
        median(observations.iter().map(|value| value[2]).collect()),
    ])
    .ok_or_else(|| "重力樣本互相抵消，無法得到穩定方向".to_owned())?;
    let residuals = observations
        .iter()
        .map(|value| angle_deg(*value, seed))
        .collect::<Vec<_>>();
    let median_residual_deg = median(residuals.clone());
    if !median_residual_deg.is_finite() || median_residual_deg > MAX_MEDIAN_RESIDUAL_DEG {
        return Err(format!(
            "重力方向不一致：median residual {:.2}° > {:.2}°",
            median_residual_deg, MAX_MEDIAN_RESIDUAL_DEG
        ));
    }
    let mad = median(
        residuals
            .iter()
            .map(|value| (value - median_residual_deg).abs())
            .collect(),
    );
    let inlier_threshold_deg = (median_residual_deg + 3.0 * mad.max(0.5))
        .clamp(MIN_INLIER_THRESHOLD_DEG, MAX_INLIER_THRESHOLD_DEG);
    let inliers = observations
        .iter()
        .zip(&residuals)
        .filter_map(|(value, residual)| (*residual <= inlier_threshold_deg).then_some(*value))
        .collect::<Vec<_>>();
    let inlier_ratio = inliers.len() as f64 / observations.len() as f64;
    if inliers.len() < MIN_MATCHED_OBSERVATIONS || inlier_ratio < MIN_INLIER_RATIO {
        return Err(format!(
            "重力 inlier 不足：{} / {} ({:.1}%)",
            inliers.len(),
            observations.len(),
            inlier_ratio * 100.0
        ));
    }
    let gravity_world_before = normalized_sum(&inliers)
        .ok_or_else(|| "重力 inlier 互相抵消，無法得到穩定方向".to_owned())?;
    let rotation_wxyz = shortest_arc_quaternion(gravity_world_before, TARGET_GRAVITY_AXIS)
        .ok_or_else(|| "無法建立重力扶正旋轉".to_owned())?;
    let gravity_world_after = rotate_vector(rotation_wxyz, gravity_world_before)
        .ok_or_else(|| "無法驗證重力扶正旋轉".to_owned())?;
    let rotation_angle_deg = 2.0 * rotation_wxyz[0].abs().clamp(-1.0, 1.0).acos().to_degrees();

    Ok(GravityAlignmentEstimate {
        registered_image_count: images.len(),
        matched_observation_count: observations.len(),
        inlier_observation_count: inliers.len(),
        matched_coverage_ratio,
        inlier_ratio,
        median_residual_deg,
        inlier_threshold_deg,
        gravity_world_before,
        gravity_world_after,
        target_up_axis: TARGET_UP_AXIS,
        rotation_wxyz,
        rotation_angle_deg,
        already_aligned: rotation_angle_deg <= ALIGNMENT_EPSILON_DEG,
    })
}

pub fn sim3_file_contents(rotation_wxyz: [f64; 4]) -> Result<String, String> {
    let rotation =
        normalize_quaternion(rotation_wxyz).ok_or_else(|| "重力扶正 quaternion 無效".to_owned())?;
    Ok(format!(
        "1 {:.17} {:.17} {:.17} {:.17} 0 0 0\n",
        rotation[0], rotation[1], rotation[2], rotation[3]
    ))
}

fn normalized_sum(values: &[[f64; 3]]) -> Option<[f64; 3]> {
    normalize(values.iter().fold([0.0; 3], |mut sum, value| {
        for index in 0..3 {
            sum[index] += value[index];
        }
        sum
    }))
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    if !value.iter().all(|component| component.is_finite()) {
        return None;
    }
    let norm = value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    (norm > 1e-12).then(|| [value[0] / norm, value[1] / norm, value[2] / norm])
}

fn normalize_quaternion(value: [f64; 4]) -> Option<[f64; 4]> {
    if !value.iter().all(|component| component.is_finite()) {
        return None;
    }
    let norm = value
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    (norm > 1e-12).then(|| {
        let mut normalized = [
            value[0] / norm,
            value[1] / norm,
            value[2] / norm,
            value[3] / norm,
        ];
        // q and -q are the same rotation; keep reports deterministic.
        if normalized[0] < 0.0 {
            normalized
                .iter_mut()
                .for_each(|component| *component = -*component);
        }
        normalized
    })
}

fn conjugate(value: [f64; 4]) -> Option<[f64; 4]> {
    let value = normalize_quaternion(value)?;
    Some([value[0], -value[1], -value[2], -value[3]])
}

fn rotate_vector(quaternion: [f64; 4], vector: [f64; 3]) -> Option<[f64; 3]> {
    let q = normalize_quaternion(quaternion)?;
    let v = normalize(vector)?;
    let u = [q[1], q[2], q[3]];
    let uv = cross(u, v);
    let uuv = cross(u, uv);
    normalize([
        v[0] + 2.0 * (q[0] * uv[0] + uuv[0]),
        v[1] + 2.0 * (q[0] * uv[1] + uuv[1]),
        v[2] + 2.0 * (q[0] * uv[2] + uuv[2]),
    ])
}

fn shortest_arc_quaternion(from: [f64; 3], to: [f64; 3]) -> Option<[f64; 4]> {
    let from = normalize(from)?;
    let to = normalize(to)?;
    let dot = dot(from, to).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-12 {
        return Some([1.0, 0.0, 0.0, 0.0]);
    }
    if dot < -1.0 + 1e-8 {
        // At exactly 180 degrees the tilt axis is undefined, so any choice
        // would also make an arbitrary yaw decision. Fail closed instead.
        return None;
    }
    let axis = cross(from, to);
    normalize_quaternion([1.0 + dot, axis[0], axis[1], axis[2]])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn angle_deg(left: [f64; 3], right: [f64; 3]) -> f64 {
    dot(left, right).clamp(-1.0, 1.0).acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction_benchmark::ColmapImageRecord;

    fn model(quaternions: &[[f64; 4]]) -> ColmapTextModel {
        ColmapTextModel {
            images: quaternions
                .iter()
                .enumerate()
                .map(|(index, qvec)| ColmapImageRecord {
                    image_id: index as u64 + 1,
                    name: format!("lens0/frame{index:03}.jpg"),
                    camera_id: 1,
                    qvec_camera_from_world: *qvec,
                    tvec_camera_from_world: [0.0; 3],
                    observed_point_count: 0,
                    frame_id: Some(index as u64 + 1),
                    registered: true,
                })
                .collect(),
            ..Default::default()
        }
    }

    fn priors(count: usize, gravity: [f64; 3]) -> Vec<GravityPriorInput> {
        (0..count)
            .map(|index| GravityPriorInput {
                image_name: format!("lens0/frame{index:03}.jpg"),
                gravity,
            })
            .collect()
    }

    #[test]
    fn leaves_y_up_model_unchanged() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 8]);
        let estimate = estimate_gravity_alignment(&model, &priors(8, TARGET_GRAVITY_AXIS)).unwrap();
        assert!(estimate.already_aligned);
        assert!(estimate.rotation_angle_deg < 1e-8);
        assert_eq!(estimate.rotation_wxyz, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn aligns_world_gravity_to_negative_y() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 8]);
        let estimate = estimate_gravity_alignment(&model, &priors(8, [1.0, 0.0, 0.0])).unwrap();
        assert!((estimate.rotation_angle_deg - 90.0).abs() < 1e-8);
        assert!(angle_deg(estimate.gravity_world_after, TARGET_GRAVITY_AXIS) < 1e-8);
    }

    #[test]
    fn converts_camera_gravity_back_into_model_world() {
        // +90 degrees around Y maps world +X to camera -Z.
        let q = [
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ];
        let model = model(&vec![q; 8]);
        let estimate = estimate_gravity_alignment(&model, &priors(8, [0.0, 0.0, -1.0])).unwrap();
        assert!(angle_deg(estimate.gravity_world_before, [1.0, 0.0, 0.0]) < 1e-8);
    }

    #[test]
    fn quaternion_sign_and_scale_do_not_change_alignment() {
        let quaternions = (0..8)
            .map(|index| {
                if index % 2 == 0 {
                    [2.0, 0.0, 0.0, 0.0]
                } else {
                    [-3.0, 0.0, 0.0, 0.0]
                }
            })
            .collect::<Vec<_>>();
        let estimate =
            estimate_gravity_alignment(&model(&quaternions), &priors(8, TARGET_GRAVITY_AXIS))
                .unwrap();
        assert!(estimate.already_aligned);
    }

    #[test]
    fn rejects_a_minority_of_direction_outliers() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 10]);
        let mut observations = priors(8, TARGET_GRAVITY_AXIS);
        observations.extend((8..10).map(|index| GravityPriorInput {
            image_name: format!("lens0/frame{index:03}.jpg"),
            gravity: [0.0, 1.0, 0.0],
        }));
        let estimate = estimate_gravity_alignment(&model, &observations).unwrap();
        assert_eq!(estimate.inlier_observation_count, 8);
        assert!(estimate.already_aligned);
    }

    #[test]
    fn refuses_insufficient_registered_gravity() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 7]);
        let error =
            estimate_gravity_alignment(&model, &priors(7, TARGET_GRAVITY_AXIS)).unwrap_err();
        assert!(error.contains("不足"));
    }

    #[test]
    fn refuses_ambiguous_upside_down_yaw() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 8]);
        let error = estimate_gravity_alignment(&model, &priors(8, TARGET_UP_AXIS)).unwrap_err();
        assert!(error.contains("無法建立重力扶正旋轉"));
    }

    #[test]
    fn refuses_tiny_coverage_in_a_large_model() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 200]);
        let error =
            estimate_gravity_alignment(&model, &priors(8, TARGET_GRAVITY_AXIS)).unwrap_err();
        assert!(error.contains("coverage"));
    }

    #[test]
    fn refuses_mixed_opposite_gravity_directions() {
        let model = model(&vec![[1.0, 0.0, 0.0, 0.0]; 8]);
        let observations = (0..8)
            .map(|index| GravityPriorInput {
                image_name: format!("lens0/frame{index:03}.jpg"),
                gravity: if index % 2 == 0 {
                    TARGET_GRAVITY_AXIS
                } else {
                    TARGET_UP_AXIS
                },
            })
            .collect::<Vec<_>>();
        let error = estimate_gravity_alignment(&model, &observations).unwrap_err();
        assert!(error.contains("互相抵消"));
    }

    #[test]
    fn emits_colmap_sim3_format() {
        assert_eq!(
            sim3_file_contents([1.0, 0.0, 0.0, 0.0]).unwrap(),
            "1 1.00000000000000000 0.00000000000000000 0.00000000000000000 0.00000000000000000 0 0 0\n"
        );
    }
}
