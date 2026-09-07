// Frozen serial native-gather oracle from commit 424e059. Test/benchmark only.
fn reference_gather(
    out: &Path,
    camera: &Camera,
    mask: &Option<GrayImage>,
    faces: &[Option<Face>],
    src: RgbImage,
    settings: &Settings,
    cancel: &CancelToken,
) -> Result<[u64; 5], String> {
    let mut normal = RgbImage::new(camera.width, camera.height);
    let mut depth = RgbImage::new(camera.width, camera.height);
    let mut validity = GrayImage::new(camera.width, camera.height);
    let mut reasons = RgbImage::new(camera.width, camera.height);
    let mut map = BufWriter::new(io(File::create(out.join("native.f32")))?);
    let mut reason_file = BufWriter::new(io(File::create(out.join("reasons.u8")))?);
    let mut counts = [0u64; 5];
    let mut scale_samples: Vec<f64> = faces
        .iter()
        .flatten()
        .flat_map(|f| {
            f.heads
                .points
                .chunks_exact(3)
                .step_by(1024)
                .map(move |p| (p[2] as f64 + f.shift) * f.heads.metric_scale as f64)
        })
        .filter(|z| z.is_finite() && *z > 0.)
        .collect();
    scale_samples.sort_by(f64::total_cmp);
    let display_max = scale_samples
        .get(scale_samples.len() * 9 / 10)
        .copied()
        .unwrap_or(1.)
        * 2.;
    for y in 0..camera.height {
        if y % 16 == 0 {
            check(cancel)?;
        }
        for x in 0..camera.width {
            let mut reason = 1u8;
            let mut value = [0f32; 4];
            if let Some(ray) = camera.ray(x as f64 + 0.5, y as f64 + 0.5) {
                reason = 2;
                if mask.as_ref().is_none_or(|m| m.get_pixel(x, y)[0] >= 128) {
                    reason = 3;
                    let mut candidates = [(0f64, [0f64; 3], 0f64); 6];
                    let mut candidate_count = 0;
                    for (f, face) in faces.iter().enumerate() {
                        let Some(face) = face else {
                            continue;
                        };
                        let Some((i, zray)) = camera::face_pixel(f, ray) else {
                            continue;
                        };
                        let p = &face.heads;
                        let z = (p.points[3 * i + 2] as f64 + face.shift) * p.metric_scale as f64;
                        if !face.support[i]
                            || !p.mask[i].is_finite()
                            || p.mask[i] < settings.validity_threshold
                            || !z.is_finite()
                            || z <= 0.
                            || z / zray > f32::MAX as f64
                        {
                            continue;
                        }
                        let Some(n) = camera::unit([
                            p.normal[3 * i] as f64,
                            p.normal[3 * i + 1] as f64,
                            p.normal[3 * i + 2] as f64,
                        ]) else {
                            continue;
                        };
                        candidates[candidate_count] = (zray, camera::rotate(f, n), z / zray);
                        candidate_count += 1;
                    }
                    let candidates = &mut candidates[..candidate_count];
                    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
                    if let Some(&(_, n, r)) = candidates.first() {
                        // Never blend depths/normals across disagreement at cube overlaps.
                        if candidates.iter().skip(1).any(|&(_, nn, rr)| {
                            camera::dot(n, nn) < 30f64.to_radians().cos()
                                || (r - rr).abs() / r.max(rr) > 0.20
                        }) {
                            reason = 4;
                        } else {
                            reason = 0;
                            value = [n[0] as f32, n[1] as f32, n[2] as f32, r as f32];
                            normal.put_pixel(
                                x,
                                y,
                                Rgb(n.map(|a| ((a * 0.5 + 0.5) * 255.).round() as u8)),
                            );
                            let t = (r / display_max).clamp(0., 1.);
                            depth.put_pixel(
                                x,
                                y,
                                Rgb([
                                    (255. * t) as u8,
                                    (255. * (1. - (2. * t - 1.).abs())) as u8,
                                    (255. * (1. - t)) as u8,
                                ]),
                            );
                            validity.put_pixel(x, y, Luma([255]));
                        }
                    }
                }
            }
            counts[reason as usize] += 1;
            reasons.put_pixel(
                x,
                y,
                Rgb([
                    [30, 160, 90],
                    [0, 0, 0],
                    [210, 140, 20],
                    [100, 100, 100],
                    [210, 50, 170],
                ][reason as usize]),
            );
            for v in value {
                io(map.write_all(&v.to_le_bytes()))?;
            }
            io(reason_file.write_all(&[reason]))?;
        }
    }
    io(map.flush())?;
    io(map.get_ref().sync_all())?;
    io(reason_file.flush())?;
    io(reason_file.get_ref().sync_all())?;
    for (name, img) in [
        ("rgb", src),
        ("normal", normal),
        ("range", depth),
        ("reasons", reasons),
    ] {
        img.save(out.join(format!("{name}.png")))
            .map_err(|e| e.to_string())?;
    }
    validity
        .save(out.join("validity.png"))
        .map_err(|e| e.to_string())?;
    Ok(counts)
}

fn gather_fixture_faces() -> Vec<Option<Face>> {
    (0..6)
        .map(|face| {
            let mut heads = Heads {
                points: vec![0.; SIZE * SIZE * 3],
                normal: vec![0.; SIZE * SIZE * 3],
                mask: vec![1.; SIZE * SIZE],
                metric_scale: 1.,
            };
            let mut support = vec![true; SIZE * SIZE];
            // Common camera-space normal, transformed back to each face. Checkerboard
            // corruptions deliberately exercise overlap disagreement and invalid heads.
            let n = camera::FACES[face].map(|axis| camera::dot(axis, [0., 0., -1.]));
            for i in 0..SIZE * SIZE {
                heads.points[i * 3 + 2] =
                    (3. * camera::face_ray(i % SIZE, i / SIZE)[2] - 0.2) as f32;
                let sign = if face % 2 == 0 && (i / SIZE / 60) % 3 == 0 {
                    -1.
                } else {
                    1.
                };
                for c in 0..3 {
                    heads.normal[i * 3 + c] = (n[c] * sign) as f32;
                }
                if i % 47 < 7 {
                    heads.mask[i] = f32::NAN;
                }
                if i % 53 < 9 {
                    support[i] = false;
                }
                if i % 61 == 0 {
                    heads.points[i * 3 + 2] = -1.;
                }
            }
            Some(Face {
                heads,
                support,
                shift: 0.2,
            })
        })
        .collect()
}

fn gather_fixture_camera(width: u32, height: u32, pinhole: bool) -> Camera {
    let scale = width as f64 / 3840.;
    let mut c = Camera {
        id: 91,
        model: if pinhole { "PINHOLE" } else { "OPENCV_FISHEYE" }.into(),
        width,
        height,
        params: vec![
            1052.40428525708 * scale,
            1053.83941556301 * scale,
            width as f64 / 2.,
            height as f64 / 2.,
        ],
        theta_max: 0.,
    };
    if !pinhole {
        c.params.extend([
            0.058281605659520064,
            0.003136722719621677,
            -0.004200184664444925,
            -0.0006035981682356756,
        ]);
    }
    c.validate().unwrap();
    c
}
fn save_native_maps(out: &Path, maps: NativeMaps, src: RgbImage) -> [u64; 5] {
    for (name, img) in [
        ("normal", maps.normal),
        ("range", maps.range),
        ("reasons", maps.reasons),
        ("rgb", src),
    ] {
        img.save(out.join(format!("{name}.png"))).unwrap();
    }
    maps.validity.save(out.join("validity.png")).unwrap();
    maps.counts
}
fn assert_native_files_equal(a: &Path, b: &Path) {
    for name in [
        "native.f32",
        "reasons.u8",
        "normal.png",
        "range.png",
        "validity.png",
        "reasons.png",
        "rgb.png",
    ] {
        assert_eq!(
            dataset::hash(&a.join(name)).unwrap(),
            dataset::hash(&b.join(name)).unwrap(),
            "{name}"
        );
    }
}
#[test]
fn cached_parallel_gather_matches_serial_bytes_across_masks_and_camera_models() {
    let faces = gather_fixture_faces();
    let d = tempfile::tempdir().unwrap();
    let mut processor = NativeProcessor::new().unwrap();
    let cancel = CancelToken::new();
    for pinhole in [false, true] {
        let camera = gather_fixture_camera(97, 131, pinhole);
        let src = RgbImage::from_fn(97, 131, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 80]));
        for masked in [false, true] {
            let mask = masked.then(|| {
                GrayImage::from_fn(97, 131, |x, y| {
                    Luma([if (x * 7 + y * 3) % 43 < 6 { 0 } else { 255 }])
                })
            });
            let reference = d.path().join(format!("reference-{pinhole}-{masked}"));
            let actual = d.path().join(format!("parallel-{pinhole}-{masked}"));
            fs::create_dir(&reference).unwrap();
            fs::create_dir(&actual).unwrap();
            let counts = reference_gather(
                &reference,
                &camera,
                &mask,
                &faces,
                src.clone(),
                &Settings::default(),
                &cancel,
            )
            .unwrap();
            let mut timings = FrameTimings::default();
            let maps = gather_native(
                &actual,
                &camera,
                &mask,
                &faces,
                &Settings::default(),
                &mut processor,
                &cancel,
                &mut timings,
            )
            .unwrap();
            assert_eq!(save_native_maps(&actual, maps, src.clone()), counts);
            assert_native_files_equal(&reference, &actual);
            if !pinhole && masked {
                assert!(
                    counts.iter().all(|n| *n > 0),
                    "Missing rejection branch: {counts:?}"
                );
            }
        }
    }
    cancel.cancel();
    let path = d.path().join("cancelled");
    fs::create_dir(&path).unwrap();
    assert!(gather_native(
        &path,
        &gather_fixture_camera(97, 131, false),
        &None,
        &faces,
        &Settings::default(),
        &mut processor,
        &cancel,
        &mut FrameTimings::default()
    )
    .is_err());
    assert!(!path.join("native.f32").exists());
}

#[test]
#[ignore = "explicit release-mode native-gather performance benchmark; no GPU required"]
fn native_gather_benchmark() {
    let width = std::env::var("GEOMETRY_BENCH_WIDTH")
        .ok()
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(3840);
    assert!((256..=3840).contains(&width));
    let camera = gather_fixture_camera(width, width, false);
    let faces = gather_fixture_faces();
    let src = RgbImage::from_fn(width, width, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, 80])
    });
    let mask = Some(GrayImage::from_fn(width, width, |x, y| {
        Luma([if (x * 7 + y * 3) % 43 < 6 { 0 } else { 255 }])
    }));
    let d = tempfile::tempdir().unwrap();
    let reference = d.path().join("serial");
    fs::create_dir(&reference).unwrap();
    let cancel = CancelToken::new();
    let settings = Settings::default();
    let started = std::time::Instant::now();
    let counts = reference_gather(
        &reference,
        &camera,
        &mask,
        &faces,
        src.clone(),
        &settings,
        &cancel,
    )
    .unwrap();
    let serial_ms = started.elapsed().as_millis();
    let mut processor = NativeProcessor::new().unwrap();
    let mut results = Vec::new();
    for name in ["cold", "warm"] {
        let out = d.path().join(name);
        fs::create_dir(&out).unwrap();
        let mut timings = FrameTimings::default();
        let started = std::time::Instant::now();
        let maps = gather_native(
            &out,
            &camera,
            &mask,
            &faces,
            &settings,
            &mut processor,
            &cancel,
            &mut timings,
        )
        .unwrap();
        assert_eq!(save_native_maps(&out, maps, src.clone()), counts);
        let elapsed = started.elapsed().as_millis();
        assert_native_files_equal(&reference, &out);
        results.push(serde_json::json!({"cache":name,"elapsedMs":elapsed,"timings":timings,"speedupVsSerial":serial_ms as f64/elapsed.max(1) as f64,"allSevenFileHashesEqual":true}));
    }
    println!(
        "NATIVE_BENCH {}",
        serde_json::json!({"width":width,"height":width,"serialMs":serial_ms,"runs":results,"scope":"native gather and output writing with synthetic inference heads and real fisheye calibration; excludes GPU inference and perspective extraction"})
    );
}
