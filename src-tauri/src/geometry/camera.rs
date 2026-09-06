//! Pixel-centre projection in the immutable source camera, including rear rays.
use serde::{Deserialize, Serialize};
pub type V3 = [f64; 3];
pub fn dot(a: V3, b: V3) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn unit(a: V3) -> Option<V3> {
    let n = dot(a, a).sqrt();
    (n.is_finite() && n > 1e-12).then(|| a.map(|v| v / n))
}

// Isolate polynomial roots using derivative roots as monotonic partitions.
// A fixed angle scan can skip a narrow fold or a tangent (double) root.
fn polynomial_roots(coefficients: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let mut coefficients = coefficients;
    while coefficients.len() > 1 && coefficients.last() == Some(&0.) {
        coefficients = &coefficients[..coefficients.len() - 1];
    }
    if coefficients.len() <= 1 {
        return vec![];
    }
    if coefficients.len() == 2 {
        let root = -coefficients[0] / coefficients[1];
        return if root.is_finite() && root >= lo && root <= hi {
            vec![root]
        } else {
            vec![]
        };
    }
    let evaluate = |x: f64| coefficients.iter().rev().fold(0., |y, a| y * x + a);
    let near_zero = |x: f64| {
        let scale = coefficients
            .iter()
            .rev()
            .fold(0., |y, a| y * x.abs() + a.abs());
        evaluate(x).abs() <= 1e-13 * scale
    };
    let derivative: Vec<f64> = coefficients
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, a)| *a * i as f64)
        .collect();
    let mut partitions = vec![lo];
    partitions.extend(polynomial_roots(&derivative, lo, hi));
    partitions.push(hi);
    partitions.sort_by(f64::total_cmp);
    let mut roots: Vec<f64> = partitions
        .iter()
        .copied()
        .filter(|x| near_zero(*x))
        .collect();
    for interval in partitions.windows(2) {
        let (mut a, mut b) = (interval[0], interval[1]);
        if near_zero(a)
            || near_zero(b)
            || evaluate(a).is_sign_positive() == evaluate(b).is_sign_positive()
        {
            continue;
        }
        let positive = evaluate(a).is_sign_positive();
        for _ in 0..64 {
            let mid = (a + b) / 2.;
            if evaluate(mid).is_sign_positive() == positive {
                a = mid;
            } else {
                b = mid;
            }
        }
        roots.push((a + b) / 2.);
    }
    roots.sort_by(f64::total_cmp);
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    roots
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    pub id: u32,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub params: Vec<f64>,
    #[serde(skip)]
    pub theta_max: f64,
}
impl Camera {
    pub fn validate(&mut self) -> Result<(), String> {
        let n = match self.model.as_str() {
            "PINHOLE" => 4,
            "OPENCV_FISHEYE" => 8,
            _ => return Err(format!("Unsupported camera {}: {}", self.id, self.model)),
        };
        if self.params.len() != n
            || self.params.iter().any(|v| !v.is_finite())
            || self.params[0] <= 0.
            || self.params[1] <= 0.
            || self.width == 0
            || self.height == 0
            || u64::from(self.width) * u64::from(self.height) > 64_000_000
        {
            return Err("Invalid camera dimensions/intrinsics (maximum 64 MP)".into());
        }
        if self.model == "OPENCV_FISHEYE" {
            self.theta_max = self.theta_limit()?;
        }
        Ok(())
    }
    fn radial(&self, t: f64) -> f64 {
        let s = t * t;
        t * (1.
            + s * (self.params[4]
                + s * (self.params[5] + s * (self.params[6] + s * self.params[7]))))
    }
    fn theta_limit(&self) -> Result<f64, String> {
        // dr/dtheta is a quartic in theta². Only the central, increasing
        // branch is invertible; an unreachable outer rim must not reject the
        // usable camera or be clamped onto this branch by ray().
        let derivative = [
            1.,
            3. * self.params[4],
            5. * self.params[5],
            7. * self.params[6],
            9. * self.params[7],
        ];
        if derivative.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite fisheye derivative".into());
        }
        let angular_max = std::f64::consts::PI - 1e-5;
        let branch_end = polynomial_roots(&derivative, 0., angular_max * angular_max)
            .into_iter()
            .find(|s| *s > 0.)
            .map(f64::sqrt)
            .unwrap_or(angular_max);
        let branch_radius = self.radial(branch_end);
        if !branch_radius.is_finite() || branch_radius <= 0. {
            return Err("Fisheye camera has no finite invertible domain".into());
        }
        let max_r = (self.width.min(self.height) as f64 / 2.) / self.params[0].min(self.params[1]);
        if branch_radius <= max_r {
            return Ok(branch_end);
        }
        let (mut lo, mut hi) = (0., branch_end);
        for _ in 0..64 {
            let t = (lo + hi) / 2.;
            if self.radial(t) < max_r {
                lo = t;
            } else {
                hi = t;
            }
        }
        Ok((lo + hi) / 2.)
    }
    pub fn ray(&self, x: f64, y: f64) -> Option<V3> {
        let [fx, fy, cx, cy] = [
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
        ];
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.
            || y < 0.
            || x >= self.width as f64
            || y >= self.height as f64
        {
            return None;
        }
        let u = (x - cx) / fx;
        let v = (y - cy) / fy;
        if self.model == "PINHOLE" {
            return unit([u, v, 1.]);
        }
        if (x - cx).hypot(y - cy) > self.width.min(self.height) as f64 / 2. {
            return None;
        }
        let r = u.hypot(v);
        if r >= self.radial(self.theta_max) {
            return None;
        }
        if r < 1e-12 {
            return Some([0., 0., 1.]);
        }
        // Solve only inside the validated central monotonic interval.
        let mut lo = 0.;
        let mut hi = self.theta_max;
        for _ in 0..35 {
            let t = (lo + hi) / 2.;
            if self.radial(t) < r {
                lo = t;
            } else {
                hi = t;
            }
        }
        let t = (lo + hi) / 2.;
        Some([u / r * t.sin(), v / r * t.sin(), t.cos()])
    }
    pub fn pixel(&self, r: V3) -> Option<[f64; 2]> {
        let r = unit(r)?;
        let d = r[0].hypot(r[1]);
        if self.model == "OPENCV_FISHEYE" && d.atan2(r[2]) >= self.theta_max {
            return None;
        }
        let k = if self.model == "PINHOLE" {
            if r[2] <= 0. {
                return None;
            }
            1. / r[2]
        } else if d < 1e-12 {
            if r[2] < 0. {
                return None;
            }
            1.
        } else {
            self.radial(d.atan2(r[2])) / d
        };
        let x = self.params[0] * r[0] * k + self.params[2];
        let y = self.params[1] * r[1] * k + self.params[3];
        if self.model == "OPENCV_FISHEYE"
            && (x - self.params[2]).hypot(y - self.params[3])
                > self.width.min(self.height) as f64 / 2.
        {
            return None;
        }
        (x >= 0. && y >= 0. && x < self.width as f64 && y < self.height as f64).then_some([x, y])
    }
}
pub const SIZE: usize = 768;
pub fn focal() -> f64 {
    SIZE as f64 / (2. * 50f64.to_radians().tan())
}
// Columns of R_source_from_face; every basis is right handed.
pub const FACES: [[V3; 3]; 6] = [
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]],
    [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
    [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
    [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
    [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]],
];
pub fn rotate(face: usize, r: V3) -> V3 {
    std::array::from_fn(|i| (0..3).map(|j| FACES[face][j][i] * r[j]).sum())
}
pub fn face_ray(x: usize, y: usize) -> V3 {
    unit([
        (x as f64 + 0.5 - SIZE as f64 / 2.) / focal(),
        (y as f64 + 0.5 - SIZE as f64 / 2.) / focal(),
        1.,
    ])
    .unwrap()
}
pub fn face_pixel(face: usize, r: V3) -> Option<(usize, f64)> {
    let q = FACES[face].map(|a| dot(a, r));
    if q[2] <= 0. {
        return None;
    }
    let x = focal() * q[0] / q[2] + SIZE as f64 / 2.;
    let y = focal() * q[1] / q[2] + SIZE as f64 / 2.;
    // Leave one-pixel support border for safe source sampling.
    (x >= 1. && y >= 1. && x < (SIZE - 1) as f64 && y < (SIZE - 1) as f64)
        .then(|| (y.floor() as usize * SIZE + x.floor() as usize, q[2]))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rear_rays_round_trip_and_rotation() {
        let mut c = Camera {
            id: 91,
            model: "OPENCV_FISHEYE".into(),
            width: 64,
            height: 64,
            params: vec![17., 17., 32., 32., 0., 0., 0., 0.],
            theta_max: 0.,
        };
        c.validate().unwrap();
        let mut rear = 0;
        for y in 0..64 {
            for x in 0..64 {
                if let Some(r) = c.ray(x as f64 + 0.5, y as f64 + 0.5) {
                    let p = c.pixel(r).unwrap();
                    assert!((p[0] - x as f64 - 0.5).abs() < 1e-6);
                    assert!((p[1] - y as f64 - 0.5).abs() < 1e-6);
                    assert!((0..6).any(|f| face_pixel(f, r).is_some()));
                    if r[2] < 0. {
                        rear += 1;
                    }
                }
            }
        }
        assert!(rear > 100);
        for b in FACES {
            let cross = [
                b[0][1] * b[1][2] - b[0][2] * b[1][1],
                b[0][2] * b[1][0] - b[0][0] * b[1][2],
                b[0][0] * b[1][1] - b[0][1] * b[1][0],
            ];
            assert_eq!(cross, b[2]);
        }
        c.params[4..].copy_from_slice(&[0.04, -0.005, 0.001, 0.0001]);
        c.validate().unwrap();
        for y in 0..64 {
            for x in 0..64 {
                if let Some(r) = c.ray(x as f64 + 0.5, y as f64 + 0.5) {
                    let p = c.pixel(r).unwrap();
                    assert!((p[0] - x as f64 - 0.5).hypot(p[1] - y as f64 - 0.5) < 0.1);
                }
            }
        }
        c.params[4] = -1.;
        c.validate().unwrap();
        assert!(c.ray(32., 32.).is_some());
        assert!(c.ray(48., 32.).is_none());
    }

    #[test]
    fn real_calibration_clips_unreachable_rim_without_rejecting_camera() {
        // Calibrated 3840² dataset: first branch misses the assumed circle
        // by less than one pixel on one axis for each physical lens.
        for params in [
            vec![
                1052.40428525708,
                1053.83941556301,
                1920.,
                1920.,
                0.058281605659520064,
                0.003136722719621677,
                -0.004200184664444925,
                -0.0006035981682356756,
            ],
            vec![
                1048.3157799019473,
                1047.6682856355524,
                1920.,
                1920.,
                0.059284058108084764,
                0.00024617085149156013,
                -0.0031916201073061683,
                -0.0006390165595437247,
            ],
        ] {
            let mut c = Camera {
                id: 1,
                model: "OPENCV_FISHEYE".into(),
                width: 3840,
                height: 3840,
                params,
                theta_max: 0.,
            };
            c.validate().unwrap();
            assert!((104. ..106.).contains(&c.theta_max.to_degrees()));
            let max_radius = c.radial(c.theta_max) * c.params[0].min(c.params[1]);
            assert!((1919. ..1920.).contains(&max_radius));
            let axis = if c.params[0] < c.params[1] { 0 } else { 1 };
            let mut p = [1920., 1920.];
            p[axis] += max_radius + 0.05;
            assert!(
                c.ray(p[0], p[1]).is_none(),
                "Unreachable rim must not clamp to the last valid ray"
            );
            p[axis] -= 0.1;
            let r = c.ray(p[0], p[1]).unwrap();
            let back = c.pixel(r).unwrap();
            assert!((back[0] - p[0]).hypot(back[1] - p[1]) < 0.1);
            let folded = c.theta_max + 0.05;
            assert!(c.pixel([folded.sin(), 0., folded.cos()]).is_none());
            for y in (0..3840).step_by(61) {
                for x in (0..3840).step_by(61) {
                    if let Some(r) = c.ray(x as f64 + 0.5, y as f64 + 0.5) {
                        let back = c.pixel(r).unwrap();
                        assert!((back[0] - x as f64 - 0.5).hypot(back[1] - y as f64 - 0.5) < 0.1);
                    }
                }
            }
        }
    }

    #[test]
    fn derivative_roots_include_narrow_folds_and_tangencies() {
        let a = 1.;
        let b = 1.0001;
        let roots = polynomial_roots(&[a * b, -a - b, 1.], 0., 10.);
        assert_eq!(roots.len(), 2);
        assert!((roots[0] - a).abs() < 1e-9);
        assert!((roots[1] - b).abs() < 1e-9);
        let roots = polynomial_roots(&[1., -2., 1.], 0., 10.);
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - 1.).abs() < 1e-9);
    }
}
