"""Shared scan utilities for scan_chip.py and scan_area_v1.py.

Geometry helpers, scan planning, position interpolation, microscope
description loading, and metadata type definitions.
"""

from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence, TypedDict


# ============================================================================
# Microscope description
# ============================================================================

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


# ============================================================================
# Position interpolation
# ============================================================================

def interpolate_position(
    t: float, samples: list[tuple[float, float, float]],
) -> float | None:
    """Interpolate position at time t from (t_before, t_after, x_um) samples.

    Uses midpoint of t_before/t_after as the effective sample time.
    """
    if not samples:
        return None

    times = [(s[0] + s[1]) / 2 for s in samples]
    idx = bisect.bisect_left(times, t)

    if idx == 0:
        return samples[0][2]
    if idx >= len(samples):
        return samples[-1][2]

    t0, x0 = times[idx - 1], samples[idx - 1][2]
    t1, x1 = times[idx], samples[idx][2]

    if t1 == t0:
        return x0

    alpha = (t - t0) / (t1 - t0)
    return x0 + alpha * (x1 - x0)


# ============================================================================
# Geometry helpers
# ============================================================================

def intersect_polygon_with_y(
    polygon: Sequence[Sequence[float]], y: float,
) -> tuple[float, float] | None:
    """Find X extent where horizontal line y intersects a convex polygon.

    Args:
        polygon: List of [x, y] vertices.
        y: Y coordinate of the horizontal line.

    Returns:
        (x_min, x_max) or None if no intersection.
    """
    intersections: list[float] = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i][0], polygon[i][1]
        x2, y2 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]

        # Check if edge crosses this Y (half-open interval to avoid double-counting vertices)
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            intersections.append(x)

    if len(intersections) < 2:
        return None

    return (min(intersections), max(intersections))


def compute_plane_z(a: float, b: float, c: float, x_um: float, y_um: float) -> float:
    """Compute Z from plane coefficients. Z_um = a * X_um + b * Y_um + c."""
    return a * x_um + b * y_um + c


# ============================================================================
# Scan planning
# ============================================================================

class BBox(TypedDict):
    """Bounding box from chip detection JSON."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

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


def compute_planar_scan_plan(
    bbox: BBox, polygon: Sequence[Sequence[float]], *,
    plane_a: float, plane_b: float, plane_c: float,
    frame_width_um: float, frame_height_um: float,
    x_overlap_pct: float, y_overlap_pct: float,
    padding: float, row_limit: int | None, speed_mm: float, z_max: float,
) -> PlanarScanPlan:
    """Compute row plan from chip geometry and focus plane.

    Raises ValueError if plan is invalid (no rows, Z exceeds limit).
    """
    target_advance = frame_width_um * (1 - x_overlap_pct / 100)
    y_step = frame_height_um * (1 - y_overlap_pct / 100)

    rows: list[tuple[float, float, float]] = []
    y = bbox["y_min"]
    while y <= bbox["y_max"]:
        extent = intersect_polygon_with_y(polygon, y)
        if extent is not None:
            x_min, x_max = extent
            # Skip rows narrower than one frame (polygon tips)
            if (x_max - x_min) < frame_width_um:
                y += y_step
                continue
            # Apply padding — extends X range beyond hull intersection
            x_min -= padding
            x_max += padding
            rows.append((y, x_min, x_max))
        y += y_step

    if row_limit:
        rows = rows[:row_limit]

    if not rows:
        raise ValueError("No rows intersect the chip contour")

    # Z range validation
    all_z: list[float] = []
    for row_y, row_x_min, row_x_max in rows:
        all_z.append(compute_plane_z(plane_a, plane_b, plane_c, row_x_min, row_y))
        all_z.append(compute_plane_z(plane_a, plane_b, plane_c, row_x_max, row_y))
    z_min = min(all_z)
    z_max_val = max(all_z)

    if z_max_val > z_max:
        raise ValueError(f"Scan Z max ({z_max_val:.0f}) exceeds limit ({z_max:.0f})")

    return PlanarScanPlan(
        rows=rows,
        target_advance_um=target_advance,
        y_step_um=y_step,
        validated_z_min_um=z_min,
        validated_z_max_um=z_max_val,
        plane_a=plane_a,
        plane_b=plane_b,
        plane_c=plane_c,
        frame_width_um=frame_width_um,
        frame_height_um=frame_height_um,
        inputs={
            "bbox": bbox,
            "polygon": polygon,
            "x_overlap_pct": x_overlap_pct,
            "y_overlap_pct": y_overlap_pct,
            "padding": padding,
            "row_limit": row_limit,
            "speed_mm": speed_mm,
            "z_max": z_max,
            "plane_a": plane_a,
            "plane_b": plane_b,
            "plane_c": plane_c,
        },
    )


# ============================================================================
# Metadata types
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



def build_lighting_meta(
    *, lamp: Any = None, shutter: Any = None,
) -> LightingMeta:
    """Build lighting metadata dict from hardware objects.

    lamp/shutter are Leica SDK objects (Lamp, Shutter) — typed as Any to avoid
    coupling this module to the leica package.
    """
    return LightingMeta(
        lamp_name=lamp.name if lamp else None,
        lamp_intensity=lamp.intensity if lamp else None,
        lamp_max_intensity=lamp.max_intensity if lamp else None,
        shutter_name=shutter.name if shutter else None,
        shutter_open=shutter.is_open if shutter else None,
    )
