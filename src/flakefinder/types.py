"""Shared type definitions for flakefinder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class BBox(TypedDict):
    """Bounding box from chip detection JSON."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class ChipGeometry:
    """Chip geometry from find_chips.py output."""
    chip_index: int
    bbox: BBox
    polygon: list[tuple[float, float]]
    centroid: tuple[float, float]
    area_um2: float


@dataclass
class PlanarScanPlan:
    """Computed scan plan from chip geometry and focus plane."""
    rows: list[tuple[float, float, float]]  # [(y_um, x_min_um, x_max_um), ...]
    target_advance_um: float
    y_step_um: float
    validated_z_min_um: float
    validated_z_max_um: float
    plane_a: float
    plane_b: float
    plane_c: float
    frame_width_um: float
    frame_height_um: float
    inputs: dict[str, Any] = field(default_factory=dict)


class CameraMeta(TypedDict):
    """Camera metadata block for scan output."""
    name: str
    exposure_s: float | None
    gain: float | None
    binning: int
    readout_time_s: float | None
    frame_width_px: int | None
    frame_height_px: int | None
    pixel_size_x_um: float | None
    pixel_size_y_um: float | None
    sensor_width_px: int | None
    sensor_height_px: int | None
    physical_pixel_x_um: float | None
    physical_pixel_y_um: float | None
    white_balance_bgr: list[float]
    gamma: float


class OpticsMeta(TypedDict):
    """Optics metadata block for scan output."""
    objective_mag: float | None
    objective_idx: int | None
    sample_pixel_x_um: float | None
    sample_pixel_y_um: float | None
    frame_width_um: float | None
    frame_height_um: float | None


class LightingMeta(TypedDict):
    """Lighting metadata block for scan output."""
    lamp_name: str | None
    lamp_intensity: float | None
    lamp_max_intensity: float | None
    shutter_name: str | None
    shutter_open: bool | None
