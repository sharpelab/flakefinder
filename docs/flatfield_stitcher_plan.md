# Flatfield Correction in Stitcher - Implementation Plan

## Overview

Add flatfield/vignetting correction to `stitch_area.py` to eliminate brightness banding in stitched images.

## Correction Formula

```python
corrected = (raw / flatfield) * flatfield_mean
```

This normalizes each pixel by its vignette factor, then scales back to original brightness.

## Implementation Steps

### 1. Add CLI Arguments

In `stitch_area.py` argument parser:

```python
parser.add_argument("--flatfield", type=Path, default=None,
                    help="Path to flatfield .npy file (default: auto-load from calibration/)")
parser.add_argument("--no-flatfield", action="store_true",
                    help="Disable flatfield correction")
```

### 2. Load Flatfield at Startup

After loading scan metadata, before processing rows:

```python
# Determine expected flatfield path
if args.no_flatfield:
    flatfield = None
elif args.flatfield:
    flatfield_path = args.flatfield
else:
    # Auto-load based on objective and binning
    obj_mag = optics["objective_mag"]
    binning = meta["camera"]["binning"]
    flatfield_path = Path("calibration") / f"flatfield_{obj_mag}x_bin{binning}.npy"

# Load and validate
if flatfield_path:
    if not flatfield_path.exists():
        print(f"Error: Flatfield not found: {flatfield_path}")
        return 1  # FAIL HARD

    flatfield = np.load(flatfield_path).astype(np.float32)
    flatfield_mean = np.mean(flatfield)
    print(f"Loaded flatfield: {flatfield_path}")
```

### 3. Resize Flatfield to Match Frame Dimensions

The flatfield is stored at full resolution. Resize to match the actual frame size (accounting for capture downsample AND stitch --downsample):

```python
# frame_w, frame_h are the working frame dimensions after all downsampling
if flatfield.shape[0] != frame_h or flatfield.shape[1] != frame_w:
    from PIL import Image as PILImage
    ff_img = PILImage.fromarray(flatfield.astype(np.uint8))
    ff_img = ff_img.resize((frame_w, frame_h), PILImage.LANCZOS)
    flatfield = np.array(ff_img, dtype=np.float32)
    flatfield_mean = np.mean(flatfield)
    print(f"  Resized flatfield to {frame_w}x{frame_h}")
```

### 4. Apply Correction Per-Frame

In `stitch_row_to_global()`, pass flatfield as parameter. After loading and downsampling frame, before blending:

```python
img = Image.open(scan_dir / f"frame_{frame_idx:04d}.jpg").convert("RGBA")

if output_downsample > 1:
    img = img.resize((frame_w, frame_h), Image.LANCZOS)

# Apply flatfield correction HERE (after downsample)
if flatfield is not None:
    img_arr = np.array(img, dtype=np.float32)
    # Correct RGB channels (not alpha)
    for c in range(3):
        img_arr[:, :, c] = (img_arr[:, :, c] / flatfield[:, :, c]) * flatfield_mean
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr, mode="RGBA")
```

Note: Need to handle the alpha channel - only correct RGB, preserve alpha.

### 5. Update Function Signature

```python
def stitch_row_to_global(meta, row, frame_w, frame_h, um_per_px, output_downsample,
                         scan_dir, global_x_min, global_x_max, blend=True, deskew=True,
                         hysteresis_um=0, flatfield=None, flatfield_mean=None):
```

### 6. Add to Output Metadata

In the `stitch_meta` dict saved at the end:

```python
stitch_meta = {
    # ... existing fields ...
    "flatfield_correction": {
        "applied": flatfield is not None,
        "file": str(flatfield_path) if flatfield is not None else None,
    }
}
```

### 7. Logging

Print at startup:
```
Flatfield: calibration/flatfield_5x_bin3.npy (resized 1824x1216 -> 912x608)
```

Or if disabled:
```
Flatfield: disabled (--no-flatfield)
```

Or if not found (then exit):
```
Error: Flatfield not found: calibration/flatfield_5x_bin3.npy
  Run capture_flatfield.py or specify --no-flatfield to skip correction.
```

## Validation Rules (Fail Hard)

1. If `--flatfield PATH` specified but file doesn't exist → error
2. If auto-loading and file doesn't exist (and not `--no-flatfield`) → error
3. If flatfield channel count doesn't match frame (e.g., grayscale vs RGB) → error

## Testing

After implementation, re-stitch `chips_5x` scan:

```bash
python stitch_area.py scans/chips_5x
```

Compare `chips_5x_stitch.png` before/after to verify banding is reduced.

## Files to Modify

- `stitch_area.py` - main changes

## Not In Scope (Punted)

- White balance correction during stitch
- Per-channel WB normalization
- Comparing scan WB to calibration WB
