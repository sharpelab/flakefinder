"""Data loading utilities for flakefinder."""

from __future__ import annotations

import json
import os

from flakefinder.types import BBox, ChipGeometry


def load_microscope_description(path: str) -> dict | None:
    """Load microscope hardware description JSON.

    Args:
        path: Path to microscope_description.json.

    Returns:
        Dict with hardware specs, or None if file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def compute_frame_size_um(
    desc: dict, objective_mag: float, binning_idx: int = 2,
) -> tuple[float, float] | None:
    """Compute frame size in µm from microscope description.

    Args:
        desc: Loaded microscope description.
        objective_mag: Objective magnification (e.g., 5, 10, 20).
        binning_idx: Binning index (0=1x1, 1=2x2, 2=3x3).

    Returns:
        (frame_width_um, frame_height_um) or None if can't compute.
    """
    camera = desc.get("camera", {})
    binning_info = camera.get("binning_levels", {}).get(str(binning_idx))
    if not binning_info:
        return None

    physical_pixel_x: float | None = camera.get("physical_pixel_x_um")
    physical_pixel_y: float | None = camera.get("physical_pixel_y_um")
    if not physical_pixel_x or not physical_pixel_y:
        return None

    frame_width_px: int | None = binning_info.get("frame_width_px")
    frame_height_px: int | None = binning_info.get("frame_height_px")
    binning_factor: int = binning_info.get("factor", 1)

    if not frame_width_px or not frame_height_px:
        return None

    # sample_pixel = physical_pixel × binning / magnification
    sample_pixel_x = physical_pixel_x * binning_factor / objective_mag
    sample_pixel_y = physical_pixel_y * binning_factor / objective_mag

    return (frame_width_px * sample_pixel_x, frame_height_px * sample_pixel_y)


def load_chip_geometry(chips_path: str, chip_index: int) -> ChipGeometry:
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
        polygon=[(v[0], v[1]) for v in raw_polygon],
        centroid=(raw_centroid[0], raw_centroid[1]),
        area_um2=chip["area_um2"],
    )
