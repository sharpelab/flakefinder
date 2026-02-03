# Chip Detection Plan

## Overview

New script `find_chips.py` that:
1. Loads stitched image + metadata from stitcher
2. Detects chips using Otsu's thresholding on luminance
3. Outputs chip bounding boxes in stage coordinates
4. Optionally outputs debug visualization

---

## Part 1: Stitcher Changes (`stitch_area.py`)

**New output file:** `stitch_meta.json` alongside the stitched PNG

**Contents:**
```json
{
  "image_file": "test_area_stitch.png",
  "image_size_px": [4200, 3800],

  "stage_bounds_um": {
    "x_min": 1000.0,
    "x_max": 74000.0,
    "y_min": 1000.0,
    "y_max": 74000.0
  },

  "scale_um_per_px": 17.4,

  "source_scan": "test_area/scan_meta.json",
  "objective_mag": 2.5,
  "downsample": 1
}
```

**Key fields for coordinate conversion:**
- `stage_bounds_um`: Bounding box of stitched image in stage coords
- `scale_um_per_px`: Microns per pixel (accounts for objective, binning, downsample)

**Coordinate conversion:**
```python
# Pixel (px_x, px_y) -> Stage (stage_x, stage_y)
stage_x = stage_bounds_um["x_min"] + px_x * scale_um_per_px
stage_y = stage_bounds_um["y_min"] + px_y * scale_um_per_px
```

---

## Part 2: Chip Detection Script (`find_chips.py`)

**Inputs:**
- Stitched image (PNG)
- Stitch metadata (JSON) - auto-discovered from image path
- `--min-area-um2`: Minimum chip area in µm² (default: 1e6 = 1mm²)
- `--debug`: Output visualization image

**Algorithm:**

```
1. Load image, convert to grayscale (luminance)
   - Use cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   - Or extract L from LAB for better perceptual uniformity

2. Apply Otsu's threshold
   - threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   - Automatically finds optimal threshold between dark/bright

3. Morphological cleanup
   - cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes
   - cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove small noise
   - Kernel size proportional to scale (e.g., 50µm)

4. Find contours
   - contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

5. Filter by area
   - For each contour:
     - area_px = cv2.contourArea(contour)
     - area_um2 = area_px * (scale_um_per_px ** 2)
     - Keep if area_um2 >= min_area_um2

6. Check edge contact
   - Reject chips where contour touches image border
   - Flag these as "edge_clipped" in output for upstream debugging

7. Extract geometry
   - For each valid contour:
     - Bounding box: x, y, w, h = cv2.boundingRect(contour)
     - Convex hull: cv2.convexHull(contour)
     - Centroid from moments
     - Convert all to stage coordinates

8. Output results
```

**Output file:** `<stitch_name>_chips.json`
```json
{
  "source_stitch": "test_area_stitch.png",
  "source_meta": "test_area_stitch_meta.json",
  "detection_params": {
    "otsu_threshold": 127,
    "min_area_um2": 1000000,
    "morph_kernel_um": 50
  },
  "chips": [
    {
      "id": 0,
      "bbox_stage_um": {
        "x_min": 5000.0,
        "y_min": 8000.0,
        "x_max": 25000.0,
        "y_max": 18000.0
      },
      "convex_hull_stage_um": [[5000, 8000], [25000, 8500], ...],
      "centroid_stage_um": [15000.0, 13000.0],
      "area_um2": 200000000,
      "bbox_px": [230, 400, 1150, 575]
    }
  ],
  "rejected": [
    {
      "reason": "edge_clipped",
      "bbox_px": [0, 100, 500, 400],
      "edge": "left"
    }
  ]
}
```

**Debug output (`--debug`):**
- Original image with detected chip contours drawn (green)
- Rejected/clipped chips drawn in red
- Bounding boxes overlaid
- Chip IDs labeled at centroids

---

## Part 3: Usage Flow

```bash
# 1. Scan area at low mag (2.5x)
python scan_area_v1.py -o scans/wafer1 --clean

# 2. Stitch the scan
python stitch_area.py scans/wafer1 -o scans/wafer1_stitch.png

# 3. Find chips
python find_chips.py scans/wafer1_stitch.png --min-area-um2 500000 --debug

# 4. Output: scans/wafer1_stitch_chips.json + debug image
```

---

## Part 4: Future Integration

The `chips.json` output enables:
1. **High-res rescan**: Loop through chips, move stage to each centroid, scan at 20x/50x
2. **Flake detection**: Run 2DMatGMM or similar on each chip region
3. **Reporting**: Generate summary of chips found per wafer

---

## Design Decisions

1. **Edge handling**: Reject chips touching image border and flag them in `rejected` array. This indicates a failure in the scan coverage - the scan area should fully contain all chips.

2. **Geometry output**: Store both bounding box (simple, axis-aligned) and convex hull (tighter fit for angled/irregular chips).

3. **Orientation**: Not estimated. Chips may be placed at arbitrary angles; we don't assume or require alignment.

4. **Color independence**: Using luminance-based Otsu thresholding rather than color segmentation. This handles varying chip colors (pink, orange, etc.) and white balance settings - only assumes chips are brighter than the dark substrate holder.
