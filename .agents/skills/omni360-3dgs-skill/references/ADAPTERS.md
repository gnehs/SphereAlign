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

若 extrinsics 不可靠，必須標成 unknown，不能硬寫一個 180-degree rotation 當真值。

## Planned adapters

### DJI Osmo 360

- dual raw fisheye streams
- DJI timed protobuf metadata (`dvtm_oq101`)
- fused attitude
- accelerometer
- factory lens metadata when present

### Insta360

預留：

- `.insv` / vendor container pairing
- lens streams
- gyro telemetry
- calibration metadata

不要在 core 直接依賴 `.insv`；只在 adapter 裡處理。

### GoPro MAX

預留 GPMF telemetry + dual-lens extraction。
