# cli_utils refactor plan

Move CLI/argparse helpers and metadata builders from `scan_utils.py` into `cli_utils.py`.

## What moves

From `scan_utils.py` → `cli_utils.py`:

- `parse_position` (str → Point2F)
- `parse_white_balance` + `DEFAULT_WB` (str → GainRGB)
- `parse_area_rect` (str → AreaRect)
- `parse_area_rect_i` (str → AreaRectI)
- `validate_area_rect` (AreaRect + StageBounds → validation)
- `build_camera_meta`, `build_optics_meta`, `build_lighting_meta`, `build_microscope_meta`
- `_BINNING_FACTOR` (used by build_camera_meta/build_optics_meta)

## What stays in scan_utils

Pure scan/geometry logic with no CLI or metadata concerns:

- `apply_flatfield`
- `interpolate_position`
- `intersect_polygon_with_y`, `compute_plane_z`
- `compute_planar_scan_plan`

## Import updates needed

Mechanical `scan_utils` → `cli_utils` for the moved symbols:

- `commands/scan.py`
- `commands/chip_scan.py`
- `commands/stitch.py`
- `commands/focus_map.py`
- `find_flakes.py`
- `capture_util.py`
- `autofocus_demo.py`
- `scripts/mosaic_util.py`
- `scripts/build_flatfield.py`
- `archive/scan_area_with_focus.py`
