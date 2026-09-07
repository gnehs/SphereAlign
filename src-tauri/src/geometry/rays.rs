//! Per-job, bounded camera-ray LRU. Preserve the exact f64 camera.ray result.
use super::{
    camera::{Camera, V3},
    draft::check,
    CancelToken,
};
use rayon::prelude::*;
use std::{collections::VecDeque, sync::Arc};

const CACHE_BYTES: usize = 768 * 1024 * 1024;
pub(super) type Rays = Arc<Vec<V3>>;

#[derive(PartialEq, Eq)]
struct Key {
    model: String,
    width: u32,
    height: u32,
    params: Vec<u64>,
    theta_max: u64,
}
impl From<&Camera> for Key {
    fn from(c: &Camera) -> Self {
        Self {
            model: c.model.clone(),
            width: c.width,
            height: c.height,
            params: c.params.iter().map(|v| v.to_bits()).collect(),
            theta_max: c.theta_max.to_bits(),
        }
    }
}

pub(super) struct NativeProcessor {
    pub pool: rayon::ThreadPool,
    entries: VecDeque<(Key, Rays)>,
    bytes: usize,
    budget: usize,
}
impl NativeProcessor {
    pub fn new() -> Result<Self, String> {
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .saturating_sub(1)
            .clamp(1, 8);
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|i| format!("geometry-native-{i}"))
                .build()
                .map_err(|e| e.to_string())?,
            entries: VecDeque::new(),
            bytes: 0,
            budget: CACHE_BYTES,
        })
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn prepare(
        &mut self,
        camera: &Camera,
        cancel: &CancelToken,
    ) -> Result<(Option<Rays>, bool), String> {
        check(cancel)?;
        let key = Key::from(camera);
        if let Some(i) = self.entries.iter().position(|(k, _)| *k == key) {
            let entry = self.entries.remove(i).unwrap();
            let rays = entry.1.clone();
            self.entries.push_back(entry);
            return Ok((Some(rays), true));
        }
        let count = camera.width as usize * camera.height as usize;
        let required = count * std::mem::size_of::<V3>();
        // Large images still use parallel, exact projection without storing a full ray map.
        if required > self.budget {
            return Ok((None, false));
        }
        while self.bytes + required > self.budget {
            if let Some((_, rays)) = self.entries.pop_front() {
                self.bytes -= rays.len() * std::mem::size_of::<V3>();
            } else {
                break;
            }
        }
        let mut rays = Vec::new();
        if rays.try_reserve_exact(count).is_err() {
            return Ok((None, false));
        }
        rays.resize(count, [0.; 3]);
        self.pool.install(|| {
            rays.par_chunks_mut(camera.width as usize)
                .enumerate()
                .try_for_each(|(y, row)| {
                    check(cancel)?;
                    for (x, ray) in row.iter_mut().enumerate() {
                        *ray = camera
                            .ray(x as f64 + 0.5, y as f64 + 0.5)
                            .unwrap_or([0.; 3]);
                    }
                    Ok::<_, String>(())
                })
        })?;
        check(cancel)?;
        let rays = Arc::new(rays);
        self.bytes += required;
        self.entries.push_back((key, rays.clone()));
        Ok((Some(rays), false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn camera() -> Camera {
        let mut c = Camera {
            id: 1,
            model: "OPENCV_FISHEYE".into(),
            width: 32,
            height: 32,
            params: vec![8.8, 8.8, 16., 16., 0.058, 0.003, -0.0042, -0.0006],
            theta_max: 0.,
        };
        c.validate().unwrap();
        c
    }
    #[test]
    fn cache_is_exact_bounded_and_keyed_by_calibration_not_id() {
        let mut worker = NativeProcessor::new().unwrap();
        worker.budget = 32 * 32 * 24;
        let mut c = camera();
        let cancel = CancelToken::new();
        let (first, hit) = worker.prepare(&c, &cancel).unwrap();
        assert!(!hit);
        let first = first.unwrap();
        for y in 0..c.height {
            for x in 0..c.width {
                assert_eq!(
                    first[(y * c.width + x) as usize].map(f64::to_bits),
                    c.ray(x as f64 + 0.5, y as f64 + 0.5)
                        .unwrap_or([0.; 3])
                        .map(f64::to_bits)
                );
            }
        }
        c.id = 999;
        let (same, hit) = worker.prepare(&c, &cancel).unwrap();
        assert!(hit && Arc::ptr_eq(&first, &same.unwrap()));
        c.params[0] += 0.1;
        c.validate().unwrap();
        assert!(!worker.prepare(&c, &cancel).unwrap().1);
        assert_eq!(worker.entries.len(), 1);
        assert!(worker.bytes <= worker.budget);
        c.width = 64;
        assert!(worker.prepare(&c, &cancel).unwrap().0.is_none());
        cancel.cancel();
        assert!(worker.prepare(&camera(), &cancel).is_err());
        assert_eq!(worker.entries.len(), 1);
    }
}
