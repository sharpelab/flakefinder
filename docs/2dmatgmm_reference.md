# 2DMatGMM-System Reference

Reference for the original flake detection system at `~/code/2DMatGMM-System/`. FlakeFinder is a rewrite/successor focused on continuous-motion scanning.

## Codebase Location

`~/code/2DMatGMM-System/` — not deployed, used as reference only.

## Directory Structure

```
2DMatGMM-System/
├── Auto_Detect_Flakes_GUI_V3.py    # Main entry point (Tkinter GUI)
├── Drivers/
│   ├── Full_Microscope_Driver/     # Low-level Leica SDK wrapper
│   │   ├── microscope.py           # Microscope class — SDK init, axes, camera
│   │   ├── focus.py                # find_focus() — autofocus algorithm
│   │   ├── helpers.py              # find_unit_by_type() — SDK tree traversal
│   │   ├── interfaces.py           # Type stubs for .NET SDK interfaces
│   │   └── enums/                  # IID, TID, UCAPI constants
│   ├── Camera_Driver/camera_class.py    # CameraDriver (exposure, gain, capture)
│   ├── Motor_Driver/motor_class.py      # MotorDriver (XY stage, abs/rel move)
│   ├── Microscope_Driver/microscope_class.py  # MicroscopeDriver (high-level)
│   └── Interfaces/                 # ABCs for driver swapping
├── Utils/
│   ├── raster_functions.py         # Scanning, focus map, flake revisit
│   ├── stitcher_functions.py       # Image stitching, overview creation
│   ├── conversion_functions.py     # Pixel ↔ µm conversion, mag mapping
│   ├── maskterial_functions.py     # MaskTerial ML detection client
│   ├── preprocessor_functions.py   # Vignette removal (Numba JIT)
│   ├── marker_functions.py         # Flake annotation overlays
│   ├── upload_functions.py         # Server upload
│   ├── etc_functions.py            # Config loading
│   └── structures.py              # Flake dataclass
├── GUI/
│   └── parameter_picker_V3.py      # Tkinter parameter input
├── Parameters/                     # Runtime config (not in git, on microscope PC)
│   └── Materials/                  # Per-material camera/microscope/GMM configs + flatfields
│       ├── Graphene_90nm_10x/      # camera_parameters.json, microscope_parameters.json, etc.
│       ├── hBN_Goldilocks_90nm_10x/
│       └── ...                     # 7 profiles total
└── other scripts (live_viewer.py, stitch.py, raster_area.py, etc.)
```

## End-to-End Scan Workflow

1. **Low-mag overview** (2.5x): `raster_plate_low_magnification()` — snake scan of whole wafer
2. **Stitch** → overview image + binary mask + scan area map
3. **Focus map** (10x): `create_z_map()` — autofocus at chip edge points, RBF interpolation
4. **High-mag search** (20x): `search_scan_area_map()` — stop-and-shoot with MaskTerial detection
5. **Multi-mag revisit**: recapture each flake at 20x, 50x, 100x, 5x, 10x
6. **Upload** to server

All scanning is **stop-and-shoot** (move → settle → capture). No continuous motion.

## Focus Map System

**File:** `Utils/raster_functions.py`, `create_z_map()` (line 279)

### Point Selection
- Uses `cv2.findContours(scan_area_map, CHAIN_APPROX_SIMPLE)` to find chip outlines
- Focus points = every contour polygon vertex + centroid per chip
- scan_area_map is at 10x grid resolution (~1.3×0.9 mm per cell, ~61×65 cells for full wafer)
- Chip boundaries from thresholding are irregular, so CHAIN_APPROX_SIMPLE still produces many vertices per chip perimeter — likely **50-100+ points total** across all chips
- Dense sampling means the RBF can tolerate noisy individual points

### Autofocus Per Point
**File:** `Drivers/Full_Microscope_Driver/focus.py`, `find_focus()`

4-iteration hierarchical hillclimb:
- Step sizes: `[20, 4, 1, 0.3]` µm, 11 samples per iteration
- Metric: Tenengrad (Sobel gradient magnitude mean) — same as FlakeFinder
- Hardcoded start: 24600 µm, upper limit: 24700 µm
- Returns best Z position, no quality/confidence metric
- 44 total stop-and-capture moves per point

### Surface Model
```python
rbf = RBFInterpolator(z_positions[:, :2], z_positions[:, 2], kernel="linear")
z_map = rbf(grid_points).reshape(height, width)
z_map = np.where(scan_area_map > 0, z_map, np.nan)
```
- **RBF interpolation** with linear kernel (effectively thin-plate spline)
- Input: sparse (x_idx, y_idx, z_um) tuples — grid index coordinates, not µm
- Output: dense 2D array matching scan_area_map dimensions
- No point rejection — all measured points used regardless of quality
- Saved as `z_map.npy`

### Application During Scanning
```python
# In image_generator() / search_scan_area_map()
z_pos = z_map[y_idx, x_idx]
motor_driver.microscope.zDrive.move_abs(z_pos)
```
Simple lookup — move Z to interpolated value before each frame capture.

## Key Constants

### Pixel Sizes
```python
# Utils/conversion_functions.py
MICROMETER_PER_PIXEL = {
    1: 1.4408 / 1000,       # 5x:  1.4408 µm/px (÷1000 → mm/px)
    2: 1.4408 / 1000 / 2,   # 10x: 0.7204 µm/px
    3: 1.4408 / 1000 / 4,   # 20x: 0.3602 µm/px
    4: 1.4408 / 1000 / 10,  # 50x: 0.14408 µm/px
}
```
Note: values are in mm/px, not µm/px (multiply by 1000 for µm).

### Magnification Indices
```
5x → index 1, 10x → index 2, 20x → index 3, 50x → index 4
```

## Camera & White Balance Settings

**Source:** `Parameters/Materials/` on microscope PC (`GGG-Leica-DM6M`), copied to `~/code/2DMatGMM-System/downloads/2DMatGMM-Parameters/Materials/`.

**Config loading:** `Utils/etc_functions.py` — `load_all_detection_parameters()` reads `camera_parameters.json` and `microscope_parameters.json` per material profile.

### White Balance Tuple Order: BGR

Despite docstrings claiming "rgb", the driver (`Drivers/Camera_Driver/camera_class.py:43-46`) maps:
```python
camera.gain_blue  = white_balance[0]  # index 0 = Blue
camera.gain_green = white_balance[1]  # index 1 = Green
camera.gain_red   = white_balance[2]  # index 2 = Red
```

### Two WB Profiles (all 90nm SiO2)

**Graphene / WSe2 profile** — `(B=1.7, G=1.0, R=1.4)`:
- Used by: `Graphene_90nm_10x`, `GrapheneThick_90nm_10x`, `WSe2_90nm_10x`
- Same WB at all magnifications
- Exposure varies: 0.01s (5x/10x), 0.02s (20x), 0.03s (50x), 0.01s (100x)

**hBN profile** — `(B=2.51, G=1.02, R=1.41)` (5x uses G=1.05):
- Used by: `hBN_Goldilocks_90nm_10x`, `hBN_ThickOnly_90nm_10x`, `hBN_Thick_90nm_10x`, `hBN_Thin_scuffed_90nm_10x`
- Exposure varies: 0.005s (5x/10x), 0.01s (20x/50x/100x)
- Derived from `calibrate_whitebalance.ipynb` calibration

### Common Settings (all profiles)
- **Gain:** 1 (unity), **Gamma:** 1 (unity)
- **Light voltage:** 200, **Aperture:** 11

### FlakeFinder Comparison

FlakeFinder uses the hBN WB values `(R=1.41, G=1.02, B=2.51)` stored in RGB order in `calibration/flatfield_*_5800K_bin3.json`. This produces a brown/warm substrate appearance on 90nm SiO2. The graphene/WSe2 profile's lower blue gain `(B=1.7)` would produce a more neutral/blue-shifted substrate tone.

### Material Profiles on Microscope PC

Seven profiles exist, all 90nm SiO2, all 10x detection mag:
- `Graphene_90nm_10x`, `GrapheneThick_90nm_10x`
- `hBN_Goldilocks_90nm_10x`, `hBN_Thick_90nm_10x`, `hBN_ThickOnly_90nm_10x`, `hBN_Thin_scuffed_90nm_10x`
- `WSe2_90nm_10x`

Each contains: `camera_parameters.json`, `microscope_parameters.json`, `GMM_parameters.json`, `flatfield.png`

## Detection Pipeline

**MaskTerial** (`Utils/maskterial_functions.py`):
- Remote REST API client → `http://134.61.8.242:8000/api`
- Sends JPEG frame, receives segmentation masks + thickness classification
- Thresholds: score ≥ 0.2, min_class_occupancy ≥ 0.3, size ≥ 400 px

**Flake dataclass** (`Utils/structures.py`):
- mask, thickness label, size (px), mean contrast (BGR), center, aspect ratio, false positive prob

## Comparison with FlakeFinder

| Aspect | 2DMatGMM | FlakeFinder |
|--------|---------|-------------|
| Scanning | Stop-and-shoot | Continuous motion |
| Focus model | RBF interpolation (dense array) | Plane equation (3 coefficients) |
| Focus points | All chip contour vertices (50-100+ total) | Contour + grid (25-33/chip) |
| Point rejection | None | Sharpness/DR/CF thresholds |
| Z application | Per-frame lookup | Constant Z velocity per row |
| Detection | MaskTerial (remote API) | TBD |
| Speed | ~15 min for overview | <1 min for overview |

## Useful Files for Specific Questions

| Topic | File | Key function/class |
|-------|------|--------------------|
| How autofocus works | `Drivers/Full_Microscope_Driver/focus.py` | `find_focus()` |
| Focus map creation | `Utils/raster_functions.py:279` | `create_z_map()` |
| Focus map application | `Utils/raster_functions.py:233` | `image_generator()` z_map lookup |
| Pixel↔µm conversion | `Utils/conversion_functions.py` | `MICROMETER_PER_PIXEL` dict |
| Low-level SDK access | `Drivers/Full_Microscope_Driver/microscope.py` | `Microscope` class |
| Vignette correction | `Utils/preprocessor_functions.py` | `remove_vignette_fast()` |
| Image stitching | `Utils/stitcher_functions.py` | `create_overview_image_and_map()` |
| Flake data model | `Utils/structures.py` | `Flake` class |
| MaskTerial detection | `Utils/maskterial_functions.py` | `MaskTerial_Model` class |
| Scan output format | `Utils/upload_functions.py` | directory structure conventions |
