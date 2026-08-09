//! Shared native-fisheye valid-region geometry.
//!
//! The optical circle keeps the maximum sensor area. DJI OSV metadata can add
//! a calibrated lower occlusion curve that follows the fixed lens/body border;
//! scene content such as hands and selfie sticks remains untouched above that
//! curve for the semantic mask stage to handle separately.

/// Maximum radius of the optical circle as a ratio of the shorter dimension.
///
/// There is deliberately no safety inset: the DJI per-lens calibration handles
/// the fixed lower occlusion, while the circle only rejects pixels outside the
/// physical image disc.
pub const DJI_VALID_RADIUS_RATIO: f64 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct OpticalOcclusion {
    center_x_ratio: f64,
    center_y_ratio: f64,
    points: Vec<(f64, f64)>,
}

impl OpticalOcclusion {
    /// Build the fixed optical/body boundary embedded in DJI's native profile.
    ///
    /// This is intentionally independent of scene contents and does not impose
    /// a 180-degree quality crop on the still-image-bearing overlap region.
    pub fn from_source_pixels(
        source_width: f32,
        source_height: f32,
        center_x: f32,
        center_y: f32,
        point_x: &[f32],
        point_y: &[f32],
    ) -> Option<Self> {
        if !source_width.is_finite()
            || !source_height.is_finite()
            || source_width <= 0.0
            || source_height <= 0.0
            || !center_x.is_finite()
            || !center_y.is_finite()
            || !(0.0..=source_width).contains(&center_x)
            || !(0.0..=source_height).contains(&center_y)
            || point_x.len() != point_y.len()
            || point_x.len() < 2
        {
            return None;
        }

        let mut points = point_x
            .iter()
            .zip(point_y)
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(x, y)| {
                (
                    f64::from(*x) / f64::from(source_width),
                    f64::from(*y) / f64::from(source_height),
                )
            })
            .filter(|(x, y)| (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y))
            .collect::<Vec<_>>();
        points.sort_by(|left, right| left.0.total_cmp(&right.0));

        // DJI repeats the bottom-center point in its curve. Merge duplicate x
        // coordinates by keeping the lower boundary to maximize valid pixels.
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(points.len());
        for point in points {
            if let Some(previous) = merged
                .last_mut()
                .filter(|previous| (previous.0 - point.0).abs() <= f64::EPSILON * 16.0)
            {
                previous.1 = previous.1.max(point.1);
            } else {
                merged.push(point);
            }
        }
        if merged.len() < 2 {
            return None;
        }

        Some(Self {
            center_x_ratio: f64::from(center_x) / f64::from(source_width),
            center_y_ratio: f64::from(center_y) / f64::from(source_height),
            points: merged,
        })
    }

    fn boundary_y_ratio(&self, x_ratio: f64) -> Option<f64> {
        let first = *self.points.first()?;
        let last = *self.points.last()?;
        if x_ratio < first.0 || x_ratio > last.0 {
            return None;
        }
        match self
            .points
            .binary_search_by(|point| point.0.total_cmp(&x_ratio))
        {
            Ok(index) => Some(self.points[index].1),
            Err(upper) if upper > 0 && upper < self.points.len() => {
                let left = self.points[upper - 1];
                let right = self.points[upper];
                let fraction = (x_ratio - left.0) / (right.0 - left.0);
                Some(left.1 + (right.1 - left.1) * fraction)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LensOpticalOcclusions {
    pub lens0: OpticalOcclusion,
    pub lens1: OpticalOcclusion,
}

#[derive(Debug, Clone)]
pub struct ValidRegion {
    center_x: f64,
    center_y: f64,
    radius_squared: f64,
    maximum_y_by_x: Vec<Option<f64>>,
}

impl ValidRegion {
    pub fn new(
        width: u32,
        height: u32,
        radius_ratio: f64,
        optical_occlusion: Option<&OpticalOcclusion>,
    ) -> Self {
        let radius = f64::from(width.min(height)) * radius_ratio;
        let (center_x_ratio, center_y_ratio) = optical_occlusion
            .map(|occlusion| (occlusion.center_x_ratio, occlusion.center_y_ratio))
            .unwrap_or((0.5, 0.5));
        let maximum_y_by_x = (0..width)
            .map(|x| {
                optical_occlusion.and_then(|occlusion| {
                    let x_ratio = (f64::from(x) + 0.5) / f64::from(width);
                    occlusion
                        .boundary_y_ratio(x_ratio)
                        .map(|ratio| ratio * f64::from(height))
                })
            })
            .collect();
        Self {
            center_x: f64::from(width) * center_x_ratio,
            center_y: f64::from(height) * center_y_ratio,
            radius_squared: radius * radius,
            maximum_y_by_x,
        }
    }

    /// Return the squared vertical offset for a row of pixel centers.
    ///
    /// COLMAP uses a corner-based coordinate convention, so the center of
    /// pixel `(x, y)` is `(x + 0.5, y + 0.5)`.
    #[inline]
    pub fn row_offset_squared(&self, y: u32) -> f64 {
        let offset = f64::from(y) + 0.5 - self.center_y;
        offset * offset
    }

    #[inline]
    pub fn contains_x(&self, x: u32, y: u32, row_offset_squared: f64) -> bool {
        let offset = f64::from(x) + 0.5 - self.center_x;
        if offset * offset + row_offset_squared > self.radius_squared {
            return false;
        }
        self.maximum_y_by_x[x as usize].is_none_or(|maximum_y| f64::from(y) + 0.5 <= maximum_y)
    }

    #[cfg(test)]
    fn contains(&self, x: u32, y: u32) -> bool {
        self.contains_x(x, y, self.row_offset_squared(y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_occlusion() -> OpticalOcclusion {
        OpticalOcclusion::from_source_pixels(
            100.0,
            100.0,
            50.0,
            50.0,
            &[20.0, 50.0, 50.0, 80.0],
            &[70.0, 90.0, 90.0, 70.0],
        )
        .unwrap()
    }

    #[test]
    fn preserves_the_full_circle_and_only_applies_the_calibrated_lower_curve() {
        let occlusion = sample_occlusion();
        let region = ValidRegion::new(100, 100, DJI_VALID_RADIUS_RATIO, Some(&occlusion));

        assert!(region.contains(49, 1));
        assert!(region.contains(99, 49));
        assert!(region.contains(49, 88));
        assert!(region.contains(19, 70));

        assert!(!region.contains(49, 90));
        assert!(!region.contains(49, 99));
    }
}
