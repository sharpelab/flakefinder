# Multi-Row Snake Scan Specification

## Overview

For stitching multiple rows in a snake raster pattern, the scan metadata needs to identify which frames belong to which row and the scan direction of each row.

## Data Model

Add a `rows` array to `scan_meta.json`:

```json
{
  "rows": [
    {
      "row_idx": 0,
      "y_um": 42551.5,
      "direction": 1,
      "frame_start": 0,
      "frame_end": 200
    },
    {
      "row_idx": 1,
      "y_um": 40800.0,
      "direction": -1,
      "frame_start": 200,
      "frame_end": 400
    }
  ],
  "frames": [
    {"n": 0, "t_start": ..., "x_start": ..., ...}
  ]
}
```

### Row Fields

| Field | Description |
|-------|-------------|
| `row_idx` | Row index (0 = first row) |
| `y_um` | Y position of this row in µm |
| `direction` | Scan direction: `1` for +X (left-to-right), `-1` for -X (right-to-left) |
| `frame_start` | First frame index for this row (inclusive) |
| `frame_end` | Last frame index for this row (exclusive) |

### Frame Position Convention

- `x_start` is the X position at frame capture start (center of frame)
- For +X rows: `x_start` increases through the row
- For -X rows: `x_start` decreases through the row
- Keep this natural/as-captured; stitcher will handle the direction

## Scan Parameters

- **Y overlap**: ~200 µm out of ~1750 µm frame height (~12%)
- **X overlap**: ~77% (same as current single-row)
- **Snake pattern**: Row 0 is +X, Row 1 is -X, Row 2 is +X, etc.

## Notes

- No frames should be recorded during Y-axis stepping between rows
- Each row will have its own accel/decel phase at the ends
- Stitcher will detect constant-velocity region per row, then apply a global trim to make all rows equal length (rectangular output)
- Deskew direction flips with scan direction (-X rows get opposite shear)
