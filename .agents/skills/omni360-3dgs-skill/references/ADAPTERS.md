# Camera Adapter Contract

建議 Python interface：

```python
class CameraAdapter(Protocol):
    @classmethod
    def can_open(cls, path: Path, probe: dict) -> bool: ...
    def descriptor(self) -> CaptureDescriptor: ...
    def video_streams(self) -> list[LensStream]: ...
    def telemetry(self) -> TelemetryBundle | None: ...
    def lens_models(self) -> list[LensModel]: ...
    def rig_hint(self) -> RigHint | None: ...
```

## CaptureDescriptor

- vendor
- model
- source_path
- duration
- frame_rate
- time_base
- lens_count
- capabilities

## LensStream

- lens_id
- ffmpeg_stream_index
- width / height
- pixel_format
- projection (`fisheye`, `pinhole`, `equirectangular`...)
- valid_region

## TelemetryBundle

時間必須明確，不允許只有「第 N 筆」。

- `attitude[] = {t, qw, qx, qy, qz, coordinate_frame}`
- `gyro[] = {t, x, y, z, unit}`
- `accel[] = {t, x, y, z, unit}`
- `gravity[]`
- `source_clock`
- `video_time_mapping`

## RigHint

- `sensor_from_rig` per lens
- covariance / confidence
- source = `factory_metadata | calibrated | inferred`

若 extrinsics 不可靠，必須標成 unknown，不能硬寫一個 180-degree rotation 當真值。`factory_intrinsics` capability 也只有在 profile 的投影模型、影像尺寸與數值欄位通過驗證，且核心 pipeline 實際傳給 COLMAP 時才能標為可用。

## Planned adapters

### DJI Osmo 360 / Osmo 360 II

- dual raw fisheye streams
- DJI timed protobuf metadata (`dvtm_oq101` for first generation; OQ102 envelope fallback for Osmo 360 II)
- fused attitude
- accelerometer
- factory lens metadata when present and independently validated
- Osmo 360 II clip profile: `OPENCV_FISHEYE`, `fx=fy`, image center, `k1..k4`
- all-zero optical-occlusion metadata falls back to the circular fisheye mask
- `cam_extri_q` is an incomplete rotation hint, not factory rig extrinsics; visual bootstrap estimates the rig

### Insta360

預留：

- `.insv` / vendor container pairing
- lens streams
- gyro telemetry
- calibration metadata

不要在 core 直接依賴 `.insv`；只在 adapter 裡處理。

### GoPro MAX

預留 GPMF telemetry + dual-lens extraction。
