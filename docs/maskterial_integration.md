# MaskTerial Integration with FlakeFinder

## Overview

MaskTerial is a deep learning-based 2D material flake detector that can replace or complement the GMM-based detection in 2DMatGMM-System. This document captures findings from evaluating MaskTerial for use with FlakeFinder.

## Key Repositories

| Repo | Path | Purpose |
|------|------|---------|
| **FlakeFinder** | `~/sharpelab/flakefinder` | New scanning system for Leica DM6M |
| **MaskTerial** | `~/code/MaskTerial` | Deep learning flake detector |
| **2DMatGMM-System** | `~/code/2DMatGMM-System` | Original scanning system (reference) |

### Key Files for Context Recovery

**MaskTerial:**
- `~/code/MaskTerial/server.py` - FastAPI inference server with `/predict` endpoint
- `~/code/MaskTerial/maskterial/maskterial.py` - Main `MaskTerial` class, `predict()` method
- `~/code/MaskTerial/maskterial/FlakeClass.py` - `Flake` output object definition
- `~/code/MaskTerial/DATASETS.md` - Dataset structure documentation
- `~/code/MaskTerial/TRAINING.md` - Training requirements (RLE masks, COCO format)

**2DMatGMM-System:**
- `~/code/2DMatGMM-System/Utils/raster_functions.py:479` - `search_scan_area_map()` detection loop
- `~/code/2DMatGMM-System/Utils/preprocessor_functions.py` - `remove_vignette_fast()` flatfield correction
- `~/code/2DMatGMM-System/Drivers/Camera_Driver/camera_class.py:16` - binning setting
- `~/code/2DMatGMM-System/Drivers/Full_Microscope_Driver/enums/BINNING_MODE.py` - binning enum (X1=0, X2=1, X3=2)

**FlakeFinder:**
- `~/sharpelab/flakefinder/src/flakefinder/leica/camera.py:262` - `sensor_size_px` property
- `~/sharpelab/flakefinder/scans/test_area/scan_meta.json` - example scan with camera specs

## Camera Compatibility

**Leica K5C specs** (from FlakeFinder scan metadata):

```json
{
  "sensor_width_px": 5472,
  "sensor_height_px": 3648,
  "binning": 3,
  "frame_width_px": 1824,
  "frame_height_px": 1216,
  "physical_pixel_x_um": 2.40234375,
  "pixel_size_x_um": 4.8046875
}
```

**Scale at each magnification** (with 3×3 binning):

| Mag | µm/pixel | Frame Size |
|-----|----------|------------|
| 5×  | 1.44     | 1824×1216  |
| 10× | 0.72     | 1824×1216  |
| 20× | 0.36     | 1824×1216  |
| 50× | 0.144    | 1824×1216  |

**MaskTerial was trained on 20× images at ~0.36 µm/px** - exact match with our camera at 3×3 binning.

## Data Collection Requirements

### For MaskTerial Inference

| Requirement | Value |
|-------------|-------|
| Magnification | 20× |
| Binning | 3×3 (binning_level = 2) |
| Resolution | 1824 × 1216 |
| Format | PNG preferred (lossless) |
| Color | RGB/BGR, 8-bit |
| Background | Must have visible substrate regions |

### No Tiling Required

MaskTerial processes individual frames independently. No stitching or tile maps needed for detection - just feed single microscope images.

### Flatfield/Background Handling

**2DMatGMM approach:** Explicit flatfield correction
```python
corrected = (raw / flatfield) * flatfield_mean
```

**MaskTerial approach:** Auto-estimates background from histogram mode
```python
background_color = mode(image, per_channel=True)
contrast = (image / background_color) - 1
```

MaskTerial handles this internally - just ensure images have visible substrate (not 100% flake coverage).

## Integration Architecture

### Option 1: MaskTerial FastAPI Server (Recommended)

MaskTerial includes a ready-to-use inference server:

```bash
# Start server
cd ~/code/MaskTerial
uvicorn server:app --host 0.0.0.0 --port 8000

# POST image, get flakes
curl -X POST http://localhost:8000/predict \
  -F "files=@frame.png" \
  -F "segmentation_model=M2F/Synthetic_Data" \
  -F "size_threshold=200"
```

Response: List of flake dicts with `mask`, `center`, `size`, `thickness`, etc.

### Option 2: Direct Python Integration

```python
from maskterial import MaskTerial
import cv2

model = MaskTerial.from_pretrained("path/to/model")

image = cv2.imread("frame.png")
flakes = model.predict(image)

for flake in flakes:
    print(f"Flake at {flake.center}, size={flake.size}px, layers={flake.thickness}")
```

## Pipeline Architecture for Continuous Scanning

```
[Leica DM6M @ 20×]
        │
        ▼ (continuous capture, ~10Hz target)
[Image Queue / Watch Directory]
        │
        ▼
[MaskTerial Worker] ← GPU (3060Ti)
        │
        ▼
[Candidates List]
        │
        ▼
[50× Revisit Positions]
```

### Throughput Considerations

- **Capture rate target:** 10 Hz (100ms per frame)
- **MaskTerial inference:** ~150-300ms per frame on 3060Ti
- **Result:** Detection lags behind capture; queue drains after scan completes
- **Candidate list:** Ready within seconds to minutes after scan ends

### Potential Optimizations (if real-time needed)

1. **TensorRT/ONNX export** - 2-3× speedup possible
2. **Batched inference** - Server currently processes one image; could batch
3. **Multiple GPU instances** - 3060Ti (8GB) might fit 2 instances

## Detection Output Format

MaskTerial returns `Flake` objects:

```python
class Flake:
    mask: np.ndarray              # Binary mask (H×W)
    thickness: str                # Layer count ("1", "2", etc.)
    size: int                     # Pixel area
    mean_contrast: np.ndarray     # BGR contrast values
    center: tuple[float, float]   # (x, y) centroid in pixels
    max_sidelength: float
    min_sidelength: float
    aspect_ratio: float
    false_positive_probability: float
    entropy: float
```

To get stage coordinates for 50× revisit:
```python
stage_x = frame_stage_x + (flake.center[0] - frame_width/2) * um_per_pixel
stage_y = frame_stage_y + (flake.center[1] - frame_height/2) * um_per_pixel
```

## Comparison: MaskTerial vs 2DMatGMM Detection

| Aspect | 2DMatGMM (GMM) | MaskTerial (Mask2Former) |
|--------|----------------|--------------------------|
| Method | Statistical color model | Deep learning segmentation |
| Speed | ~10-20ms/frame | ~150-300ms/frame |
| Training | Calibration parameters per material | Pretrained + optional fine-tune |
| Accuracy | Good with calibration | Generally better, especially edge cases |
| GPU required | No | Yes |

## Next Steps

1. [ ] Test MaskTerial server with captured 20× frames
2. [ ] Benchmark inference speed on microscope PC (3060Ti)
3. [ ] Implement watch directory / queue architecture
4. [ ] Add 50× revisit coordinate calculation
5. [ ] Compare detection quality vs GMM on same samples

## References

- MaskTerial paper/repo: https://github.com/dgglab/MaskTerial (assumed)
- 2DMatGMM-System: https://github.com/dgglab/2DMatGMM-System
- Leica SDK documentation: (on microscope PC)
