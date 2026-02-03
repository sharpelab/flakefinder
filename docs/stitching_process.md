# Stitching Process

## Overview

The stitching script (`stitch_position.py`) combines microscope scan frames into a single panoramic image using position data from the stage controller and a linear fit to smooth out position jitter.

## Process

### 1. Load Calibration from Metadata

Read `scan_meta.json` which contains camera/optics info from the microscope:
- `sample_pixel_x_um` (1.44 µm/px native) × `downsample` (2) = **2.88 µm/px**
- FOV: 2629 × 1753 µm

### 2. Identify Constant-Velocity Frames

The scan has three phases:
| Phase | Frames | Distance | Time |
|-------|--------|----------|------|
| Acceleration | 0-26 | 5.5 mm | 432 ms |
| Constant velocity | 27-165 | 80.5 mm | 2070 ms |
| Deceleration | 166-199 | 4.8 mm | 495 ms |

To identify the constant-velocity region:
- Compute frame-to-frame position deltas from raw `x_start` values
- Find median delta (robust to outliers from accel/decel phases)
- Mark frames as "constant velocity" if their delta is within 20% of median
- Find first and last CV frame
- Discard acceleration/deceleration frames at start and end

### 3. Fit Linear Model to Position Data

The raw position samples have ~217 µm RMS jitter due to SDK latency issues (see `docs/position_sampling_issue.md`). However, the stage moves at constant velocity during the CV phase.

For CV frames:
- Extract midpoint time `(t_start + t_end) / 2` and raw `x_start`
- Fit linear regression: `x = velocity × t + intercept`
- Each frame gets a **smoothed position** based on its timestamp

This recovers true positions from the accurate timestamps, bypassing the jittery position samples.

Computed constant velocity: **39.74 mm/s**

### 4. Render Stitched Image

- Calculate canvas width from smoothed position range + one frame width
- For each CV frame:
  - Load JPEG, convert to RGBA
  - Set **50% alpha** (to visualize overlap regions)
  - Compute X offset: `(smoothed_position - x_min) / um_per_px`
  - Composite onto canvas at that offset
  - Draw frame number in center (white text, black outline)
- Flatten to RGB with white background
- Save as PNG

### Output

- Canvas size: ~29,635 × 608 px
- Frame overlap: ~77% (2033 µm of 2629 µm FOV)
- Frame spacing: 595.7 µm

## Usage

```bash
cd ~/sharpelab/flakefinder
uv run python stitch_position.py
```

Output: `position_stitch.png`

## Future Improvements

- Rolling shutter deskew correction (frames are skewed due to stage motion during readout)
- Feature-based alignment refinement for remaining position errors
- Blending/feathering at seams instead of alpha overlay
- Support for multi-row scans (2D stitching)
