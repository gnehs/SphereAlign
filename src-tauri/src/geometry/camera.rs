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
        // Stop at the first fold; the optical circle must fit entirely before it.
        let max_r = (self.width.min(self.height) as f64 / 2.) / self.params[0].min(self.params[1]);
        let mut previous = 0.;
        for i in 1..=4096 {
            let t = (std::f64::consts::PI - 1e-5) * i as f64 / 4096.;
            let r = self.radial(t);
            if !r.is_finite() || r <= previous {
                return Err("Fisheye polynomial folds inside the optical circle".into());
            }
            if r >= max_r {
                return Ok(t);
            }
            previous = r;
        }
        Err("Fisheye optical circle exceeds invertible angular domain".into())
    }
    pub fn ray(&self, x: f64, y: f64) -> Option<V3> {
        let [fx, fy, cx, cy] = [
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
        ];
        if x < 0. || y < 0. || x >= self.width as f64 || y >= self.height as f64 {
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
        if r < 1e-12 {
            return Some([0., 0., 1.]);
        }
        // Validated monotonic interval; Newton remains bounded by bisection.
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
        if self.model == "OPENCV_FISHEYE" && d.atan2(r[2]) > self.theta_max {
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
        assert!(c.validate().is_err());
    }
}
