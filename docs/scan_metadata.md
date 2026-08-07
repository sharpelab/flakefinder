# Scan Metadata Format (`scan_meta.json`)

Each scan directory contains a `scan_meta.json` file with complete metadata about the scan, including camera settings, optics, timing, and per-frame position data.

Type definitions: `src/flakefinder/types.py` (`ScanMeta`, `FrameMeta`, `ScanLineMeta`, etc.).
Loader: `flakefinder.data_utils.load_scan_meta()`.

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `git_version` | string | Git commit hash (e.g. `"b36777c"` or `"b36777c-dirty"`; `"unknown"` if git unavailable) |
| `timestamp` | string | ISO 8601 timestamp |
| `x_min_um` | float | Scan area X minimum (µm) |
| `x_max_um` | float | Scan area X maximum (µm) |
| `y_min_um` | float | Scan area Y minimum (µm) |
| `y_max_um` | float | Scan area Y maximum (µm) |
| `y_step_um` | float | Y distance between row centers (µm) |
| `y_overlap_percent` | float | Row overlap as percentage of frame height |
| `downsample` | int | Downsample factor applied to saved frames |
| `scan_duration_s` | float | Total scan time (seconds) |
| `frame_count` | int | Total number of frames captured |
| `position_sample_count` | int | Total position samples recorded |

## `scan_params` Object

| Field | Type | Description |
|-------|------|-------------|
| `scan_speed_mm_s` | float | Actual stage speed used (mm/s) |
| `move_speed_mm_s` | float | Stage repositioning speed (mm/s) |
| `lead_in_um` | float | Lead-in distance (chip_scan only) |
| `lead_out_um` | float | Lead-out distance (chip_scan only) |
| `area_rect` | string\|null | Area rect argument (overview scan only) |
| `margin_um` | float\|null | Margin from stage edges (overview scan only) |

## `camera` Object

Camera hardware settings (`CameraMeta`):

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Camera model (e.g., "K5C") |
| `exposure_s` | float | Exposure time (seconds) |
| `binning` | int | Binning factor (1, 2, or 3 for 1×1, 2×2, 3×3) |
| `readout_time_s` | float | Sensor readout time (seconds) |
| `frame_width_px` | int | Output frame width (pixels) |
| `frame_height_px` | int | Output frame height (pixels) |
| `pixel_size_x_um` | float | SDK-reported pixel size X (µm) |
| `pixel_size_y_um` | float | SDK-reported pixel size Y (µm) |
| `sensor_width_px` | int | Full sensor width (pixels) |
| `sensor_height_px` | int | Full sensor height (pixels) |
| `physical_pixel_x_um` | float | Physical pixel pitch X (µm) |
| `physical_pixel_y_um` | float | Physical pixel pitch Y (µm) |
| `white_balance_bgr` | [B,G,R] | White balance gains (blue, green, red) |
| `colour_temperature` | int | CCM selector index (0=identity, 2=5800K); pinned to 2 at connection unless overridden |
| `gamma` | float | Gamma correction value |

## `optics` Object

Optical configuration (`OpticsMeta`):

| Field | Type | Description |
|-------|------|-------------|
| `objective_mag` | float | Objective magnification (e.g., 5, 10, 20) |
| `objective_idx` | int | Nosepiece position (1-indexed) |
| `sample_pixel_x_um` | float | Sample-plane pixel size X (µm/px) |
| `sample_pixel_y_um` | float | Sample-plane pixel size Y (µm/px) |
| `frame_width_um` | float | Frame width at sample plane (µm) |
| `frame_height_um` | float | Frame height at sample plane (µm) |

**Sample pixel calculation:** `sample_pixel = physical_pixel × binning / magnification`

## `lighting` Object

Illumination settings (`LightingMeta`):

| Field | Type | Description |
|-------|------|-------------|
| `lamp_name` | string | Lamp unit name |
| `lamp_intensity` | int | Lamp intensity (0-255) |
| `lamp_max_intensity` | int | Maximum lamp intensity |
| `shutter_name` | string | Shutter unit name |
| `shutter_open` | bool | Shutter state during scan |

## `lines` Array

Per-line (row) metadata for the snake scan pattern (`ScanLineMeta`):

| Field | Type | Description |
|-------|------|-------------|
| `line_idx` | int | Line index (0-based) |
| `y_um` | float | Row Y position (µm) |
| `direction` | int | Scan direction: 1 = +X, -1 = -X |
| `frame_start` | int | First frame index for this line |
| `frame_end` | int | Last frame index + 1 for this line |
| `x_min_um` | float | Row X minimum (chip_scan only) |
| `x_max_um` | float | Row X maximum (chip_scan only) |
| `duration_s` | float | Row scan duration (seconds) |
| `position_samples` | int | Position samples recorded for this row |

## `frames` Array

Per-frame metadata with interpolated positions (`FrameMeta`):

| Field | Type | Description |
|-------|------|-------------|
| `n` | int | Frame index |
| `line` | int | Line (row) index |
| `t_capture` | float | Capture start time (seconds from scan start) |
| `capture_duration_s` | float | Capture duration (seconds) |
| `x_um` | float | Interpolated X position (µm) |
| `y_um` | float | Row Y position (µm) |
| `x_vel_um_s` | float | X velocity (µm/s) |
| `y_vel_um_s` | float | Y velocity (µm/s) |
| `phase` | string | `"lead_in"`, `"capture"`, or `"lead_out"` |
| `z_um` | float\|null | Z position (chip_scan only) |
| `z_vel_um_s` | float | Z velocity (chip_scan only) |
| `z_plane_um` | float\|null | Ideal Z from focus plane (chip_scan only) |
| `z_error_um` | float\|null | Z tracking error (chip_scan only) |

## `position_stream` Array

Raw position samples used for interpolation (`PositionStreamSample`):

| Field | Type | Description |
|-------|------|-------------|
| `t_before` | float | Timestamp before SDK read (seconds) |
| `t_after` | float | Timestamp after SDK read (seconds) |
| `x_um` | float | X position (µm) |
| `line` | int | Line (row) index |

Position interpolation uses the midpoint of `t_before` and `t_after` as the effective sample time.

## `focus_plane` Object (chip_scan only)

| Field | Type | Description |
|-------|------|-------------|
| `a` | float | Plane coefficient dZ/dX (µm/µm) |
| `b` | float | Plane coefficient dZ/dY (µm/µm) |
| `c` | float | Plane intercept (µm) |
| `equation` | string | Human-readable plane equation |
| `z_range_um` | [min, max] | Validated Z range |

## Example Usage

```python
from flakefinder.data_utils import load_scan_meta

meta = load_scan_meta(Path("scans/my_scan"))

# Get sample pixel size for coordinate conversion
um_per_px = meta["optics"]["sample_pixel_x_um"]

# Get frame position
frame = meta["frames"][100]
print(f"Frame 100: ({frame['x_um']:.1f}, {frame['y_um']:.1f}) µm")

# Iterate lines
for line in meta["lines"]:
    n_frames = line["frame_end"] - line["frame_start"]
    print(f"Line {line['line_idx']}: {n_frames} frames, Y={line['y_um']:.0f} µm")
```

## Coordinate System

- **Stage coordinates**: X and Y in micrometers (µm), matching SDK/hardware
- **Frame positions**: `x_um` is interpolated stage X at capture time
- **Y positions**: Each line has a constant Y value; frames within a line share `y_um`
- **Time**: All timestamps relative to scan start (seconds)
