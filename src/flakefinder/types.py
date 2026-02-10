"""Shared type definitions for flakefinder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypedDict

# Geometric points


class Point2F(NamedTuple):
    """2D point (x, y)."""

    x: float
    y: float


class Point3F(NamedTuple):
    """3D point (x, y, z)."""

    x: float
    y: float
    z: float


class GainRGB(NamedTuple):
    """Per-channel camera gain (red, green, blue)."""

    red: float
    green: float
    blue: float


class ScanRow(NamedTuple):
    """A single row in a scan plan: (y_um, x_min_um, x_max_um)."""

    y_um: float
    x_min_um: float
    x_max_um: float


class PositionSample(NamedTuple):
    """A single polled position sample: (t_before, t_after, x_um)."""

    t_before: float
    t_after: float
    x_um: float


# ============================================================================
# Microscope description
# ============================================================================


@dataclass
class BinningLevel:
    """Camera binning level from microscope description."""

    name: str
    factor: int
    frame_width_px: int
    frame_height_px: int


@dataclass
class CameraDescription:
    """Camera hardware description."""

    name: str
    sensor_width_px: int
    sensor_height_px: int
    physical_pixel_x_um: float
    physical_pixel_y_um: float
    binning_levels: dict[int, BinningLevel]


@dataclass
class ObjectiveDescription:
    """Objective lens description."""

    position: int
    magnification: float
    name: str


@dataclass
class AxisDescription:
    """Stage axis description."""

    min_um: float
    max_um: float
    max_speed_mm_s: float


@dataclass
class StageDescription:
    """Stage hardware description."""

    x: AxisDescription
    y: AxisDescription
    z: AxisDescription


@dataclass
class MicroscopeDescription:
    """Parsed microscope hardware description."""

    camera: CameraDescription
    objectives: dict[int, ObjectiveDescription]
    stage: StageDescription


# ============================================================================
# Chip geometry
# ============================================================================


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
    polygon: list[Point2F]
    centroid: Point2F
    area_um2: float


# ============================================================================
# Scan planning
# ============================================================================


@dataclass
class PlanarScanPlan:
    """Computed scan plan from chip geometry and focus plane."""

    rows: list[ScanRow]
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


# ============================================================================
# Scan output metadata
# ============================================================================


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


class MicroscopeMeta(TypedDict):
    """Combined microscope metadata from live hardware."""

    camera: CameraMeta
    optics: OpticsMeta
    lighting: LightingMeta
