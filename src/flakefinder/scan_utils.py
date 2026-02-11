"""Scan planning, geometry helpers, and position interpolation."""

from __future__ import annotations

import bisect
from argparse import ArgumentTypeError
from collections.abc import Sequence
from typing import Any

import numpy as np

from flakefinder.types import (
    BBox,
    CameraMeta,
    GainRGB,
    LightingMeta,
    MicroscopeMeta,
    OpticsMeta,
    PlanarScanPlan,
    Point2F,
    PositionSample,
    ScanRow,
)

# Binning index (SDK) -> binning factor (NxN)
_BINNING_FACTOR = {0: 1, 1: 2, 2: 3}


def parse_position(s: str) -> Point2F:
    """Parse 'X,Y' position string into Point2F.

    Intended as an argparse type= callback:
        parser.add_argument("--position", type=parse_position, ...)

    Args:
        s: Comma-separated string (e.g., "50000,35000") in µm.

    Returns:
        Point2F(x, y).
    """
    parts = s.split(",")
    if len(parts) != 2:
        raise ArgumentTypeError(f"expected 2 comma-separated values (X,Y), got: {s}")
    try:
        return Point2F(float(parts[0]), float(parts[1]))
    except ValueError as e:
        raise ArgumentTypeError(f"values must be numbers, got: {s}") from e


def parse_white_balance(s: str) -> GainRGB:
    """Parse 'B,G,R' white balance string into GainRGB.

    Intended as an argparse type= callback:
        parser.add_argument("--white-balance", type=parse_white_balance, ...)

    Args:
        s: Comma-separated string in B,G,R order (e.g., "2.51,1.02,1.41").

    Returns:
        GainRGB(red, green, blue) suitable for camera.gain_rgb.
    """
    parts = s.split(",")
    if len(parts) != 3:
        raise ArgumentTypeError(f"expected 3 comma-separated values (B,G,R), got: {s}")
    try:
        b, g, r = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as e:
        raise ArgumentTypeError(f"values must be numbers, got: {s}") from e
    return GainRGB(red=r, green=g, blue=b)


# ============================================================================
# Flatfield correction
# ============================================================================


def apply_flatfield(image: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    """Apply flatfield correction using per-channel means (pure vignetting removal).

    Corrects spatial non-uniformity (vignetting) without shifting image color.
    Each channel is independently normalized by its own mean, so the output
    preserves the original color balance.

    Formula per channel c:
        corrected[:,:,c] = image[:,:,c] / flatfield[:,:,c] * mean(flatfield[:,:,c])

    Args:
        image: Input image array (H, W, 3), any dtype.
        flatfield: Flatfield reference array (H, W, 3), float32.

    Returns:
        Corrected image as uint8, clipped to [0, 255].
    """
    ff = flatfield.astype(np.float32)
    ff[ff < 1] = 1
    ch_means = ff.mean(axis=(0, 1))  # shape (3,)
    corrected = image.astype(np.float32) / ff * ch_means
    return np.clip(corrected, 0, 255).astype(np.uint8)


# ============================================================================
# Position interpolation
# ============================================================================


def interpolate_position(
    t: float,
    samples: list[PositionSample],
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
    polygon: Sequence[Point2F],
    y: float,
) -> Point2F | None:
    """Find X extent where horizontal line y intersects a convex polygon.

    Args:
        polygon: List of (x, y) vertices.
        y: Y coordinate of the horizontal line.

    Returns:
        (x_min, x_max) or None if no intersection.
    """
    intersections: list[float] = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if edge crosses this Y (half-open interval to avoid double-counting vertices)
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            intersections.append(x)

    if len(intersections) < 2:
        return None

    return Point2F(min(intersections), max(intersections))


def compute_plane_z(a: float, b: float, c: float, x_um: float, y_um: float) -> float:
    """Compute Z from plane coefficients. Z_um = a * X_um + b * Y_um + c."""
    return a * x_um + b * y_um + c


# ============================================================================
# Scan planning
# ============================================================================


def compute_planar_scan_plan(
    bbox: BBox,
    polygon: Sequence[Point2F],
    *,
    plane_a: float,
    plane_b: float,
    plane_c: float,
    frame_width_um: float,
    frame_height_um: float,
    x_overlap_pct: float,
    y_overlap_pct: float,
    padding: float,
    row_limit: int | None,
    speed_mm: float,
    z_max: float,
) -> PlanarScanPlan:
    """Compute row plan from chip geometry and focus plane.

    Raises ValueError if plan is invalid (no rows, Z exceeds limit).
    """
    target_advance = frame_width_um * (1 - x_overlap_pct / 100)
    y_step = frame_height_um * (1 - y_overlap_pct / 100)

    rows: list[ScanRow] = []
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
            rows.append(ScanRow(y, x_min, x_max))
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
# Metadata helpers
# ============================================================================


def build_lighting_meta(
    *,
    lamp: Any = None,
    shutter: Any = None,
    aperture: Any = None,
) -> LightingMeta:
    """Build lighting metadata dict from hardware objects.

    lamp/shutter/aperture are Leica SDK objects — typed as Any to avoid
    coupling this module to the leica package.
    """
    return LightingMeta(
        lamp_name=lamp.name if lamp else None,
        lamp_intensity=lamp.intensity if lamp else None,
        lamp_max_intensity=lamp.max_intensity if lamp else None,
        shutter_name=shutter.name if shutter else None,
        shutter_open=shutter.is_open if shutter else None,
        aperture_value=aperture.value if aperture else None,
        aperture_max_value=aperture.max_value if aperture else None,
    )


def build_camera_meta(camera: Any) -> CameraMeta:
    """Build camera metadata entirely from live hardware state.

    Reads all values (gain, white balance, gamma, exposure, binning,
    frame size, sensor size, pixel sizes) from the camera object.
    """
    frame_width_px, frame_height_px = camera.frame_size_px
    sensor_width_px, sensor_height_px = camera.sensor_size_px
    pixel_size_x_um, pixel_size_y_um = camera.pixel_size_um
    physical_pixel_x_um, physical_pixel_y_um = camera.physical_pixel_size_um
    r, g, b = camera.gain_rgb
    return CameraMeta(
        name=camera.name,
        exposure_s=camera.exposure_time,
        gain=camera.gain,
        binning=_BINNING_FACTOR.get(camera.binning, camera.binning),
        readout_time_s=camera.readout_time_s,
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        pixel_size_x_um=pixel_size_x_um,
        pixel_size_y_um=pixel_size_y_um,
        sensor_width_px=sensor_width_px,
        sensor_height_px=sensor_height_px,
        physical_pixel_x_um=physical_pixel_x_um,
        physical_pixel_y_um=physical_pixel_y_um,
        white_balance_bgr=[b, g, r],
        gamma=camera.gamma,
    )


def build_optics_meta(
    *,
    nosepiece: Any,
    camera: Any,
) -> OpticsMeta:
    """Build optics metadata from nosepiece and camera.

    Computes frame FOV in µm from physical pixel size, binning, and
    objective magnification. All values read from live hardware.
    """
    mag = nosepiece.magnification
    binning_factor = _BINNING_FACTOR.get(camera.binning, camera.binning)
    physical_pixel_x_um, physical_pixel_y_um = camera.physical_pixel_size_um
    frame_width_px, frame_height_px = camera.frame_size_px

    if mag:
        sample_pixel_x_um = physical_pixel_x_um * binning_factor / mag
        sample_pixel_y_um = physical_pixel_y_um * binning_factor / mag
    else:
        sample_pixel_x_um = None
        sample_pixel_y_um = None

    frame_width_um = sample_pixel_x_um * frame_width_px if sample_pixel_x_um and frame_width_px else None
    frame_height_um = sample_pixel_y_um * frame_height_px if sample_pixel_y_um and frame_height_px else None

    return OpticsMeta(
        objective_mag=mag,
        objective_idx=nosepiece.position,
        sample_pixel_x_um=sample_pixel_x_um,
        sample_pixel_y_um=sample_pixel_y_um,
        frame_width_um=frame_width_um,
        frame_height_um=frame_height_um,
    )


def build_microscope_meta(scope: Any) -> MicroscopeMeta:
    """Build combined microscope metadata from live hardware.

    Reads camera, optics, and lighting state from the Microscope facade.
    Frame FOV is computed from camera pixel size + objective magnification.

    Args:
        scope: Microscope facade instance.

    Returns:
        MicroscopeMeta with camera, optics, and lighting sub-dicts.
    """
    return MicroscopeMeta(
        camera=build_camera_meta(scope.camera),
        optics=build_optics_meta(nosepiece=scope.nosepiece, camera=scope.camera),
        lighting=build_lighting_meta(lamp=scope.lamp, shutter=scope.shutter, aperture=scope.aperture),
    )
