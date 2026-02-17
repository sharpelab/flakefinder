"""Data loading utilities for flakefinder."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from flakefinder.types import (
    AxisDescription,
    BBox,
    BinningLevel,
    CameraDescription,
    ChipGeometry,
    ChipScanMeta,
    MicroscopeDescription,
    ObjectiveDescription,
    Point2F,
    ScanMeta,
    StageDescription,
)

_DESCRIPTION_PATH = Path(str(files("flakefinder"))) / "microscope_description.json"


def require_microscope_description() -> MicroscopeDescription:
    """Load and parse the bundled microscope hardware description.

    Returns:
        MicroscopeDescription.

    Raises:
        FileNotFoundError: If the bundled file is missing.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(_DESCRIPTION_PATH) as f:
        raw = json.load(f)

    raw_camera = raw.get("camera", {})
    raw_binning = raw_camera.get("binning_levels", {})
    binning_levels: dict[int, BinningLevel] = {}
    for idx_str, bl in raw_binning.items():
        binning_levels[int(idx_str)] = BinningLevel(
            name=bl["name"],
            factor=bl["factor"],
            frame_width_px=bl["frame_width_px"],
            frame_height_px=bl["frame_height_px"],
        )

    camera = CameraDescription(
        name=raw_camera.get("name", ""),
        sensor_width_px=raw_camera.get("sensor_width_px", 0),
        sensor_height_px=raw_camera.get("sensor_height_px", 0),
        physical_pixel_x_um=raw_camera.get("physical_pixel_x_um", 0.0),
        physical_pixel_y_um=raw_camera.get("physical_pixel_y_um", 0.0),
        binning_levels=binning_levels,
    )

    objectives: dict[int, ObjectiveDescription] = {}
    for pos_str, obj in raw.get("objectives", {}).items():
        objectives[int(pos_str)] = ObjectiveDescription(
            position=int(pos_str),
            magnification=obj["magnification"],
            name=obj["name"],
        )

    raw_stage = raw.get("stage", {})
    stage = StageDescription(
        x=_parse_axis(raw_stage.get("x", {})),
        y=_parse_axis(raw_stage.get("y", {})),
        z=_parse_axis(raw_stage.get("z", {})),
    )

    return MicroscopeDescription(camera=camera, objectives=objectives, stage=stage)


def _parse_axis(raw: dict) -> AxisDescription:
    return AxisDescription(
        min_um=raw.get("min_um", 0.0),
        max_um=raw.get("max_um", 0.0),
        max_speed_mm_s=raw.get("max_speed_mm_s", 0.0),
    )


def compute_frame_size_um(
    camera: CameraDescription,
    objective_mag: float,
    binning_idx: int = 2,
) -> Point2F | None:
    """Compute frame size in µm from camera description and objective.

    Args:
        camera: Camera hardware description.
        objective_mag: Objective magnification (e.g., 5, 10, 20).
        binning_idx: Binning index (0=1x1, 1=2x2, 2=3x3).

    Returns:
        (frame_width_um, frame_height_um) or None if binning level not found.
    """
    bl = camera.binning_levels.get(binning_idx)
    if bl is None:
        return None

    sample_pixel_x = camera.physical_pixel_x_um * bl.factor / objective_mag
    sample_pixel_y = camera.physical_pixel_y_um * bl.factor / objective_mag

    return Point2F(bl.frame_width_px * sample_pixel_x, bl.frame_height_px * sample_pixel_y)


def load_scan_meta(scan_dir: Path) -> ScanMeta:
    """Load typed scan_meta.json from a scan directory.

    Args:
        scan_dir: Directory containing scan_meta.json.

    Returns:
        ScanMeta typed dict.

    Raises:
        FileNotFoundError: If scan_meta.json not found.
    """
    meta_path = scan_dir / "scan_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"scan_meta.json not found in {scan_dir}")
    with open(meta_path) as f:
        return json.load(f)


def load_chip_scan_meta(scan_dir: Path) -> ChipScanMeta:
    """Load typed scan_meta.json for a chip scan directory.

    Same as load_scan_meta but returns ChipScanMeta with Z tracking
    fields, focus_plane, chip_info, etc.

    Args:
        scan_dir: Directory containing scan_meta.json from chip_scan.py.

    Returns:
        ChipScanMeta typed dict.

    Raises:
        FileNotFoundError: If scan_meta.json not found.
    """
    meta_path = scan_dir / "scan_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"scan_meta.json not found in {scan_dir}")
    with open(meta_path) as f:
        return json.load(f)


def load_chip_geometry(chips_path: Path, chip_index: int) -> ChipGeometry:
    """Load chip geometry from find_chips.py output.

    Args:
        chips_path: Path to chips JSON file.
        chip_index: Index of the chip to load.

    Returns:
        ChipGeometry with typed bbox, polygon, centroid, and area.

    Raises:
        FileNotFoundError: If chips_path doesn't exist.
        ValueError: If chip_index is out of range.
    """
    with open(chips_path) as f:
        chips_data = json.load(f)

    chips = chips_data.get("chips", [])
    if chip_index < 0 or chip_index >= len(chips):
        raise ValueError(f"Chip index {chip_index} out of range (0-{len(chips) - 1})")

    chip = chips[chip_index]
    raw_bbox = chip["bbox_stage_um"]
    raw_polygon = chip["convex_hull_stage_um"]
    raw_centroid = chip["centroid_stage_um"]

    return ChipGeometry(
        chip_index=chip_index,
        bbox=BBox(
            x_min=raw_bbox["x_min"],
            x_max=raw_bbox["x_max"],
            y_min=raw_bbox["y_min"],
            y_max=raw_bbox["y_max"],
        ),
        polygon=[Point2F(v[0], v[1]) for v in raw_polygon],
        centroid=Point2F(raw_centroid[0], raw_centroid[1]),
        area_um2=chip["area_um2"],
    )
