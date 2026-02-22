# 2DMatGMM Website Upload Format

Reference for producing scan output compatible with the [2DMatGMM-Website](~/code/2DMatGMM-Website). Based on analysis of both the website's upload handler (`Backend/functions.py`) and the existing scan system (`~/code/2DMatGMM-System`).

## Directory Structure

The website accepts a ZIP file via `POST /upload`. The ZIP contains a single top-level directory named after the scan:

```
<scan_name>/
├── meta.json                          # Scan metadata
├── flatfield.png                      # Flatfield correction reference
├── overview.png                       # Full-res stitched overview
├── overview_compressed.jpg            # 2000x2000 preview
├── mask.png                           # Binary chip detection mask
├── scan_area_map.png                  # Labeled chip regions
├── Chip_1/
│   ├── Flake_1/
│   │   ├── meta.json                  # Flake metadata + camera settings
│   │   ├── overview_marked.jpg        # Overview with flake circled
│   │   ├── eval_img.jpg               # Close-up with flake outlined
│   │   ├── raw_img.png                # Uncorrected capture
│   │   ├── flake_mask.png             # Binary flake silhouette
│   │   ├── 5x.png                     # Optional multi-mag images
│   │   ├── 20x.png
│   │   └── 50x.png
│   └── Flake_2/
│       └── ...
└── Chip_2/
    └── ...
```

**Naming**: Chip/Flake directories use 1-indexed numbering (`Chip_1`, `Flake_1`). The upload handler discovers them by listing subdirectories (alphanumeric sort), so the exact prefix doesn't strictly matter, but the existing system and frontend both expect this convention.

**ZIP creation**: `shutil.make_archive(scan_dir, "zip", scan_dir)` — the scan directory itself is the ZIP root.

---

## Scan-Level meta.json

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `scan_user` | `str` | Yes | Who ran the scan |
| `scan_time` | `float` | Yes | Unix timestamp (`time.time()`) |
| `chip_thickness` | `str` | Yes | Substrate thickness, e.g. `"90nm"`, `"285nm"`. Stored as `Chip.wafer` for ALL chips in the scan |
| `scan_exfoliated_material` | `str` | No | Material name, e.g. `"graphene"`, `"hBN_Thick"`. Applied to ALL chips. Defaults to `None` |
| `comment` | `str` | No | Free-text comment. If absent, falls back to `scan_exfoliation_method` |
| `scan_exfoliation_method` | `str` | No | Legacy field, used as comment fallback only |

**Additional fields** written by the existing system but **not consumed** by the upload handler (informational only):

| Field | Type | Description |
|-------|------|-------------|
| `scan_magnification` | `int` | Detection magnification (e.g. 10, 20) |
| `standard_deviation_threshold` | `float` | GMM detection parameter |
| `size_threshold` | `float` | Minimum flake size in µm² |
| `server_url` | `str` | Upload endpoint URL |
| `image_directory` | `str` | Local image path on scan machine |
| `use_auto_AF` | `bool` | Whether hardware autofocus was used |

**Scan name** is derived from the directory name, not from any JSON field.

### FlakeFinder Extension (`flakefinder` object)

Both scan-level and flake-level meta.json include a `flakefinder` object with FlakeFinder pipeline metadata. These fields are archived with the upload ZIP but not stored in the website's database (the DB schema is fixed — see below). The `flakefinder` namespace keeps our data cleanly separated from the website's expected fields.

At scan level: operator, scan name, notes, and run provenance. At flake level: detection traceability (chip/frame/det_id), tier/score, and segmentation features (contrast, morphology metrics). See `build_scan_meta()` and `build_flake_meta()` in `upload.py` for the full field list.

### Example

```json
{
  "scan_user": "Sandesh",
  "scan_time": 1738800000.0,
  "chip_thickness": "285nm",
  "scan_exfoliated_material": "hBN",
  "comment": "SF119 A-H | preset=5_20 | scan_mag=20x",
  "flakefinder": {
    "operator": "Sandesh",
    "name": "SF119 A-H",
    "notes": null,
    "run_dir": "run_20260221_1651",
    "scan_name": "SF119_A-H_20260221_1651"
  }
}
```

---

## Flake-Level meta.json

Two top-level keys: `flake` (properties) and `images` (per-magnification camera settings).

### Flake Properties

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `position_x` | `float` | Yes | mm | Stage X coordinate of flake center |
| `position_y` | `float` | Yes | mm | Stage Y coordinate of flake center |
| `size` | `float` | Yes | µm² | Flake area |
| `thickness` | `str` | Yes | — | Layer classification: `"1L"`, `"2L"`, `"thick"`, `"bulk"`, etc. |
| `entropy` | `float` | Yes | — | Shannon entropy of flake region |
| `max_sidelength` | `float` | Yes | nm | Longer side of rotated bounding box |
| `min_sidelength` | `float` | Yes | nm | Shorter side of rotated bounding box |
| `false_positive_probability` | `float` | No | 0–1 | Classifier confidence. Defaults to `0.0` |

Flake-level meta.json also includes a `flakefinder` object — see scan-level note above.

### Images Object

One entry per magnification. Key is the magnification string (e.g. `"5x"`, `"20x"`).

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `aperture` | `float` | Yes | — | Aperture stop setting |
| `light` | `float` | Yes | V | Lamp voltage |
| `nosepiece` | `float` | Yes | — | Objective magnification (e.g. `20.0`) |
| `gamma` | `int` | Yes | — | Gamma × 100 (e.g. `120` = gamma 1.2) |
| `gain` | `float` | Yes | — | Camera gain |
| `exposure` | `float` | Yes | s | Exposure time |
| `white_balance` | `[int, int, int]` | Yes | — | `[R, G, B]` white balance values |

**At least one magnification entry is required** — each one creates an `Image` row in the DB.

### Position Calculation

The existing system computes flake position from the stage motor position at capture time plus the pixel offset of the flake center from the image center:

```
flake_x_mm = motor_x + (flake_center_px_x - image_width/2) * µm_per_pixel / 1000
flake_y_mm = motor_y + (flake_center_px_y - image_height/2) * µm_per_pixel / 1000
```

### Example

```json
{
  "flake": {
    "position_x": 47.234,
    "position_y": 31.567,
    "size": 1234.5,
    "thickness": "2L",
    "entropy": 2.34,
    "max_sidelength": 45600.0,
    "min_sidelength": 19800.0,
    "false_positive_probability": 0.15
  },
  "images": {
    "20x": {
      "aperture": 6,
      "light": 6.2,
      "nosepiece": 20.0,
      "gamma": 120,
      "gain": 4.0,
      "exposure": 0.001,
      "white_balance": [64, 64, 64]
    }
  }
}
```

---

## Image Files

### Scan-Level Images

| File | Format | Typical Size | Description |
|------|--------|-------------|-------------|
| **flatfield.png** | PNG | 1824×1216 | Flatfield correction reference at the detection magnification. Used for vignette removal: `corrected = (raw / flatfield) * flatfield_mean`, clipped to 241. The existing system loads this from pre-calibrated parameter files per material/thickness/magnification combo. |
| **overview.png** | PNG | ~7600×6100 | Full stitched overview from low-mag (5x) raster. Individual 1824×1216 tiles downsampled 8× during stitching → ~228×152 per tile, then assembled into the mosaic. Pixel (0,0) aligned to stage origin (0,0). |
| **overview_compressed.jpg** | JPEG 80% | 2000×2000 | The overview resized to exactly 2000×2000 (aspect ratio NOT preserved). This is what the frontend displays, and what `overview_marked.jpg` is derived from. |
| **mask.png** | PNG | Same as overview | Binary (0/255) chip detection mask. Generated from the overview via: grayscale → Gaussian blur (11×11) → Otsu threshold → contour filter (min 1000px²) → fill. White = chip, black = background. |
| **scan_area_map.png** | PNG | ~118×136 | Tiny labeled image. Each pixel represents one high-mag field of view. Values: 0 = background, 1 = chip 1, 2 = chip 2, etc. Generated by sampling the mask at high-mag FOV grid spacing, eroding edges by 1px, then connected-component labeling. |

### Flake-Level Images

All flake images are at the camera's native resolution: **1824×1216 pixels**.

| File | Format | Description |
|------|--------|-------------|
| **eval_img.jpg** | JPEG | Vignette-corrected image with: red outline tracing the flake edge (morphological gradient of mask), green rotated bounding box around the flake. This is the primary flake view in the frontend. |
| **raw_img.png** | PNG | Original uncorrected camera capture at detection magnification. Not displayed by frontend, but preserved for reprocessing. |
| **flake_mask.png** | PNG | Binary mask (0 = background, 255 = flake). Same dimensions as the capture. |
| **overview_marked.jpg** | JPEG | Copy of `overview_compressed.jpg` (2000×2000) with a green circle (r=20px) and red flake number at the flake's location. Coordinate mapping: `px = (motor_mm / 84.2) * 2000`. |
| **{mag}x.png** | PNG | Multi-magnification revisit images: `2.5x.png`, `5x.png`, `10x.png`, `20x.png`, `50x.png`, `100x.png`. Captured by moving to flake position, switching objective, autofocusing, and capturing. Optional — frontend tries to load each and silently handles 404s. |

### Frontend Image Paths

Images served by Apache at `IMAGE_URL` (not through Flask):

```
{IMAGE_URL}/{scan_name}/Chip_1/Flake_1/eval_img.jpg
{IMAGE_URL}/{scan_name}/Chip_1/Flake_1/20x.png
{IMAGE_URL}/{scan_name}/Chip_1/Flake_1/overview_marked.jpg
```

The magnification image extension is configurable via `REACT_APP_MAGNIFICATION_SUFFIX` (default: `png`).

---

## Database Schema

Four tables with cascade deletes:

```
Scan (1) ──→ (N) Chip (1) ──→ (N) Flake (1) ──→ (N) Image
```

| Table | Key Columns | Notes |
|-------|------------|-------|
| **Scan** | `name` (from dir name), `user`, `time` (BigInt), `comment` (nullable) | |
| **Chip** | `wafer` (= chip_thickness), `material` (= scan_exfoliated_material) | Same wafer/material for all chips in a scan |
| **Flake** | position_x/y, size, thickness, entropy, max/min_sidelength, false_positive_probability, `path` (reconstructed), `used` (bool, default false), `favorite` (bool, default false) | `used` and `favorite` are UI toggle states |
| **Image** | aperture, light_voltage, magnification, gain, gamma, exposure_time, white_balance_r/g/b | One row per magnification per flake |

All string columns are `VARCHAR(255)`.

---

## Upload Processing

1. ZIP received, extracted to `IMAGE_DIRECTORY/<scan_name>/`
2. Read `meta.json` → create `Scan` row
3. List subdirectories (sorted alphanumeric) → each becomes a `Chip` row
4. Within each chip dir, list subdirectories → each becomes a `Flake` row
5. Read each flake's `meta.json` → create `Flake` row + `Image` rows per magnification
6. Path stored as `"scan_name/chip_dir/flake_dir"` (forward slashes, reconstructed from filesystem)

**No file existence validation** — missing images just 404 in the frontend. Missing `meta.json` files cause a hard error.

**Size limit**: 32 GB max upload.

---

## Thoughts on Implementation

### What FlakeFinder Needs to Produce

At minimum for a working upload:

1. **Scan `meta.json`** — straightforward, just 3 required fields
2. **At least one `Chip_N/Flake_M/meta.json`** — needs all required flake fields + at least one image entry
3. **`overview_compressed.jpg`** — the frontend's main scan overview (2000×2000)
4. **`eval_img.jpg` per flake** — the primary flake view shown in the UI
5. **`overview_marked.jpg` per flake** — overview with flake location marked

Everything else is nice-to-have. In particular:
- `flatfield.png` — not consumed by the website, just archived. Include it for reproducibility
- `overview.png` — only used to generate the compressed version and mask; the website doesn't serve it
- `mask.png`, `scan_area_map.png` — not used by the website at all, artifacts of the existing detection pipeline
- `raw_img.png`, `flake_mask.png` — not displayed by frontend, but useful for reprocessing
- Multi-mag images — loaded on demand, gracefully absent

### Suggested Simplifications

**Drop `mask.png` and `scan_area_map.png`**: These are intermediate artifacts of the existing chip-detection pipeline. FlakeFinder already knows where chips are. Include them only if the website actually renders them (it doesn't in any path I found — they're only used server-side to verify upload success).

**`overview_marked.jpg` coordinate mapping is fragile**: The existing system hardcodes `84.2mm` as the stage range divisor. FlakeFinder's Leica stage has a different range (95.2 × 85.1 mm). Either:
- Adopt the same formula but with corrected stage dimensions, or
- Generate `overview_marked.jpg` by mapping flake position to overview pixel coordinates using the actual stitch geometry (more robust)

**Sidelengths are in nanometers**: Easy to miss — the field names don't indicate units but values are in nm (µm × 1000).

**`chip_thickness` names a substrate, not a chip**: Confusing but harmless. Just set it to your substrate type.

**`overview_compressed.jpg` squashes aspect ratio**: The existing system resizes to exactly 2000×2000 regardless of scan shape. This means non-square scans get distorted. The overview_marked coordinate mapping depends on this. If your scans are roughly square this is fine; if not, consider padding to square before resizing, and adjusting the marker coordinate math to match.

**`images` metadata is verbose for continuous scans**: During continuous-motion scanning, camera settings don't change between frames. You could store the settings once in the scan-level meta and reference them, but the per-flake format is what the DB expects. Just duplicate the same camera settings dict into every flake meta.

**Thickness classification**: The existing system uses a GMM classifier to assign layer labels. If FlakeFinder doesn't have a classifier yet, use `"unknown"` or `"unclassified"` — the field is a free-text string, not an enum.

**`false_positive_probability`**: If you don't have a confidence model, just set to `0.0` (the default). The frontend uses it for filtering/sorting but works fine without it.

### Minimal Implementation Checklist

- [ ] Generate stitched overview → resize to 2000×2000 JPEG 80% quality
- [ ] Run flake detection → for each flake, capture/crop the eval image and draw outlines
- [ ] Generate overview_marked per flake (circle on compressed overview)
- [ ] Write scan `meta.json` with user, time, substrate
- [ ] Write flake `meta.json` with position, size, thickness label, sidelengths, camera settings
- [ ] Organize into `Chip_N/Flake_M/` directory tree
- [ ] ZIP and POST to `/upload`

### Things That May Need Website Changes

- **Magnification suffix**: If FlakeFinder captures TIFF or different formats, set `REACT_APP_MAGNIFICATION_SUFFIX` accordingly
- **Scale bars**: The frontend overlays static scale bar images from `scalebars/{mag}.png`. These are calibrated for the existing system's camera/optics. If your pixel scale differs, you'll need to provide updated scale bar overlays or disable them
- **Stage range in overview marker**: The `84.2mm` constant in marker_functions.py would need updating for a different microscope
