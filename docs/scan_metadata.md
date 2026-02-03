# Scan Metadata Format (`scan_meta.json`)

Each scan directory contains a `scan_meta.json` file with complete metadata about the scan, including camera settings, optics, timing, and per-frame position data.

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
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

Parameters passed to scan_area_v1.py:

| Field | Type | Description |
|-------|------|-------------|
| `speed_mm_s` | float | Actual stage speed used (mm/s) |
| `area_rect` | string\|null | Area rect argument if provided |
| `margin_um` | float\|null | Margin from stage edges if area_rect not used |
| `auto_focus_pos_um` | [x,y]\|null | Autofocus position if specified |
| `objective_requested` | string\|null | Objective argument if provided |

## `camera` Object

Camera hardware settings:

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
| `gamma` | float | Gamma correction value |

## `optics` Object

Optical configuration:

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

Illumination settings:

| Field | Type | Description |
|-------|------|-------------|
| `lamp_name` | string | Lamp unit name |
| `lamp_intensity` | int | Lamp intensity (0-255) |
| `lamp_max_intensity` | int | Maximum lamp intensity |
| `shutter_name` | string | Shutter unit name |
| `shutter_open` | bool | Shutter state during scan |

## `rows` Array

Per-row metadata for the snake scan pattern:

| Field | Type | Description |
|-------|------|-------------|
| `row_idx` | int | Row index (0-based) |
| `y_um` | float | Row Y position (µm) |
| `direction` | int | Scan direction: 1 = +X, -1 = -X |
| `frame_start` | int | First frame index for this row |
| `frame_end` | int | Last frame index + 1 for this row |
| `duration_s` | float | Row scan duration (seconds) |
| `position_samples` | int | Position samples recorded for this row |

## `frames` Array

Per-frame metadata with interpolated positions:

| Field | Type | Description |
|-------|------|-------------|
| `n` | int | Frame index |
| `row` | int | Row index |
| `t_start` | float | Exposure start time (seconds from scan start) |
| `t_end` | float | Exposure end time (seconds from scan start) |
| `x_start` | float | Interpolated X position at exposure start (µm) |
| `x_end` | float | Interpolated X position at exposure end (µm) |
| `x_vel` | float | X velocity during exposure (µm/s) |
| `y_um` | float | Row Y position (µm) |

## `position_stream` Array

Raw position samples used for interpolation:

| Field | Type | Description |
|-------|------|-------------|
| `t_before` | float | Timestamp before SDK read (seconds) |
| `t_after` | float | Timestamp after SDK read (seconds) |
| `x_um` | float | X position (µm) |
| `row` | int | Row index |

Position interpolation uses the midpoint of `t_before` and `t_after` as the effective sample time.

## Example Usage

```python
import json

with open("scans/my_scan/scan_meta.json") as f:
    meta = json.load(f)

# Get sample pixel size for coordinate conversion
um_per_px = meta["optics"]["sample_pixel_x_um"]

# Get frame position
frame = meta["frames"][100]
x_center = (frame["x_start"] + frame["x_end"]) / 2
y_center = frame["y_um"]
print(f"Frame 100 center: ({x_center:.1f}, {y_center:.1f}) µm")

# Iterate rows
for row in meta["rows"]:
    n_frames = row["frame_end"] - row["frame_start"]
    print(f"Row {row['row_idx']}: {n_frames} frames, Y={row['y_um']:.0f} µm")
```

## Coordinate System

- **Stage coordinates**: X and Y in micrometers (µm), matching SDK/hardware
- **Frame positions**: `x_start`/`x_end` are stage X coordinates at frame edges
- **Y positions**: Each row has a constant Y value; frames within a row share `y_um`
- **Time**: All timestamps relative to scan start (seconds)
