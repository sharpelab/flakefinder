"""Shared type definitions for flakefinder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypedDict

import numpy as np

# Image type alias — all images in flakefinder are RGB uint8 numpy arrays.
# The SDK returns BGR; sdk_image_to_numpy() converts at the boundary.
type RGBImage = np.ndarray  # (H, W, 3) uint8, RGB channel order

# Geometric points


class Point2F(NamedTuple):
    """2D point (x, y)."""

    x: float
    y: float


class Point2I(NamedTuple):
    """2D integer point (x, y) in pixels."""

    x: int
    y: int


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


class AreaRect(NamedTuple):
    """Stage-coordinate rectangle in µm."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class AreaRectI(NamedTuple):
    """Integer pixel rectangle."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int


class ScanRow(NamedTuple):
    """A single row in a scan plan: (y_um, x_min_um, x_max_um)."""

    y_um: float
    x_min_um: float
    x_max_um: float


class PositionSample(NamedTuple):
    """A single polled position sample: (t_before, t_after, axis_um)."""

    t_before: float
    t_after: float
    axis_um: float


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


class AxisBounds(Protocol):
    """Anything with min/max position in µm (Axis, AxisDescription, etc.)."""

    @property
    def min_um(self) -> float: ...
    @property
    def max_um(self) -> float: ...


class StageBounds(Protocol):
    """Anything with x/y axes that have min/max bounds."""

    @property
    def x(self) -> AxisBounds: ...
    @property
    def y(self) -> AxisBounds: ...


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


class PlanInputs(TypedDict):
    """Provenance record of inputs to compute_planar_scan_plan."""

    bbox: BBox
    polygon: list[Point2F]
    x_overlap_pct: float
    y_overlap_pct: float
    padding: float
    row_limit: int | None
    speed_mm: float
    z_max: float
    plane_a: float
    plane_b: float
    plane_c: float


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
    inputs: PlanInputs | None = None


# ============================================================================
# Scan output metadata
# ============================================================================


class ScanLineMeta(TypedDict):
    """Line metadata from scan_meta.json lines array."""

    line_idx: int
    direction: int
    y_um: float
    frame_start: int
    frame_end: int


class ChipScanLineMeta(ScanLineMeta):
    """Line metadata with row extents (chip_scan only)."""

    x_min_um: float
    x_max_um: float


type Phase = str  # "lead_in" | "capture" | "lead_out"


class FrameMeta(TypedDict):
    """Per-frame metadata from scan_meta.json frames array."""

    n: int
    line: int
    t_capture: float
    capture_duration_s: float
    x_um: float
    y_um: float
    x_vel_um_s: float
    y_vel_um_s: float
    phase: Phase


class ChipScanFrameMeta(FrameMeta):
    """Per-frame metadata with Z tracking (chip_scan only)."""

    z_um: float
    z_vel_um_s: float
    z_plane_um: float
    z_error_um: float


class PositionStreamSample(TypedDict):
    """Raw position poll sample from scan_meta.json position_stream."""

    t_before: float
    t_after: float
    x_um: float
    line: int


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
    aperture_value: int | None
    aperture_max_value: int | None


class MicroscopeMeta(TypedDict):
    """Combined microscope metadata from live hardware."""

    camera: CameraMeta
    optics: OpticsMeta
    lighting: LightingMeta


# ============================================================================
# Top-level scan_meta.json
# ============================================================================


class ScanParamsMeta(TypedDict):
    """Common scan parameters (present in both overview and chip scans)."""

    scan_speed_mm_s: float
    move_speed_mm_s: float


class ChipScanParamsMeta(ScanParamsMeta):
    """Chip scan parameters (extends common with lead-in/out)."""

    lead_in_um: float
    lead_out_um: float


class FocusPlaneMeta(TypedDict):
    """Focus plane metadata (chip_scan only)."""

    a: float
    b: float
    c: float
    equation: str


class ScanMeta(TypedDict):
    """Top-level scan metadata — common fields for all scan types."""

    timestamp: str
    downsample: int
    y_step_um: float
    scan_duration_s: float
    frame_count: int
    camera: CameraMeta
    optics: OpticsMeta
    lighting: LightingMeta
    lines: list[ScanLineMeta]
    frames: list[FrameMeta]
    position_stream: list[PositionStreamSample]


class ChipInfoMeta(TypedDict):
    """Chip geometry info embedded in chip scan metadata."""

    chips_meta: str
    chip_index: int
    bbox_stage_um: BBox
    centroid_stage_um: Point2F
    area_um2: float
    hull_vertices: int


class ChipScanMeta(TypedDict):
    """Top-level chip scan metadata — all fields required.

    Cannot inherit ScanMeta because lines/frames have narrower element
    types (ChipScanLineMeta, ChipScanFrameMeta).
    """

    timestamp: str
    downsample: int
    y_step_um: float
    scan_duration_s: float
    frame_count: int
    camera: CameraMeta
    optics: OpticsMeta
    lighting: LightingMeta
    lines: list[ChipScanLineMeta]
    frames: list[ChipScanFrameMeta]
    position_stream: list[PositionStreamSample]
    scan_params: ChipScanParamsMeta
    focus_plane: FocusPlaneMeta
    chip_info: ChipInfoMeta
