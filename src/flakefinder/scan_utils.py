"""Scan planning, geometry helpers, and position interpolation."""

from __future__ import annotations

import bisect
import subprocess
from argparse import ArgumentTypeError
from collections.abc import Sequence
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

from flakefinder.leica.camera import Camera
from flakefinder.leica.microscope import Microscope
from flakefinder.leica.units import Aperture, Lamp, Nosepiece, Shutter
from flakefinder.types import (
    AreaRect,
    AreaRectI,
    BBox,
    CameraMeta,
    GainRGB,
    LightingMeta,
    MicroscopeMeta,
    OpticsMeta,
    PlanarScanPlan,
    PlanInputs,
    Point2F,
    PositionSample,
    ScanMeta,
    ScanRow,
    StageBounds,
)


@cache
def _repo_root() -> Path:
    """Walk up from the package directory to find the repo root (pyproject.toml)."""
    p = Path(str(files("flakefinder"))).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root (no pyproject.toml found)")


CALIBRATION_DIR = _repo_root() / "calibration"


@cache
def get_git_version() -> str:
    """Return short git commit hash with optional '-dirty' suffix.

    Returns ``"unknown"`` if git is unavailable or the repo has no commits.
    """
    try:
        root = _repo_root()
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if rev.returncode != 0:
            return "unknown"
        sha = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=root,
            capture_output=True,
        )
        if dirty.returncode != 0:
            sha += "-dirty"
        return sha
    except Exception:
        return "unknown"


# Binning index (SDK) -> binning factor (NxN)
_BINNING_FACTOR = {0: 1, 1: 2, 2: 3}

# Parfocal reference Z positions (µm) per objective magnification.
# Measured 2026-02-14 at chip 2 rank 18 (Elijah's Chip Rejects).
PARFOCAL_Z_UM: dict[float, float] = {
    2.5: 24584.9,
    5: 24671.9,
    10: 24669.3,
    20: 24682.5,
    50: 24685.1,
}

# Parcentric XY offsets (µm) per objective magnification, relative to 10x.
# Offset = optical_center(this_obj) - optical_center(10x).
# Measured 2026-02-21 on SF119 A-H (chip 6, rank01_frame_0327_d0).
# Accuracy ~±5 µm. +X = stage right, +Y = stage down (toward higher Y).
PARCENTRIC_XY_UM: dict[float, Point2F] = {
    2.5: Point2F(-25, 0),
    5: Point2F(5, 20),
    10: Point2F(0, 0),
    20: Point2F(30, 25),
    50: Point2F(17, 30),
}


def build_revisit_json(
    base_points: list[dict],
    scan_mag: float,
    target_mag: float,
) -> dict:
    """Build a revisit JSON dict with parfocal Z + parcentric XY offsets applied.

    Args:
        base_points: Points at scan magnification [{x, y, z, label}, ...].
        scan_mag: Magnification the points were detected at.
        target_mag: Magnification for revisit capture.

    Returns:
        Dict ready for JSON serialization with adjusted points and metadata.
    """
    z_delta = PARFOCAL_Z_UM[target_mag] - PARFOCAL_Z_UM[scan_mag]
    xy_scan = PARCENTRIC_XY_UM.get(scan_mag, Point2F(0, 0))
    xy_target = PARCENTRIC_XY_UM.get(target_mag, Point2F(0, 0))
    dx = xy_target.x - xy_scan.x
    dy = xy_target.y - xy_scan.y

    adjusted = [
        {
            **pt,
            "x": round(pt["x"] + dx, 2),
            "y": round(pt["y"] + dy, 2),
            "z": round(pt["z"] + z_delta, 2),
        }
        for pt in base_points
    ]
    return {
        "objective_mag": target_mag,
        "scan_mag": scan_mag,
        "applied_parfocal_delta_um": round(z_delta, 1),
        "applied_parcentric_delta_xy_um": [round(dx, 1), round(dy, 1)],
        "points": adjusted,
    }


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


# Default white balance for hBN on SiO2 (calibrated on Sharpe Lab DM6M)
DEFAULT_WB = GainRGB(red=1.41, green=1.02, blue=2.51)


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


def validate_area_rect(area: AreaRect, stage: StageBounds) -> None:
    """Validate area rect against stage limits.

    Args:
        area: Scan area rectangle in µm.
        stage: Anything with x/y axes that have min/max bounds
            (StageDescription for dry runs, live Stage for connected runs).

    Raises:
        ValueError: If area exceeds stage limits.
    """
    x_min, x_max, y_min, y_max = area
    if x_min < stage.x.min_um:
        raise ValueError(f"x_min ({x_min:.0f}) is below stage minimum ({stage.x.min_um:.0f})")
    if x_max > stage.x.max_um:
        raise ValueError(f"x_max ({x_max:.0f}) exceeds stage maximum ({stage.x.max_um:.0f})")
    if y_min < stage.y.min_um:
        raise ValueError(f"y_min ({y_min:.0f}) is below stage minimum ({stage.y.min_um:.0f})")
    if y_max > stage.y.max_um:
        raise ValueError(f"y_max ({y_max:.0f}) exceeds stage maximum ({stage.y.max_um:.0f})")


def parse_area_rect(s: str) -> AreaRect:
    """Parse 'x_min,x_max,y_min,y_max' string into AreaRect.

    Intended as an argparse type= callback:
        parser.add_argument("--area-rect", type=parse_area_rect, ...)

    Sorts min/max if swapped.  Validates non-zero extent.

    Args:
        s: Comma-separated string (e.g., "8000,95000,0,78000") in µm.

    Returns:
        AreaRect(x_min, x_max, y_min, y_max).
    """
    parts = s.split(",")
    if len(parts) != 4:
        raise ArgumentTypeError(f"expected 4 comma-separated values (x_min,x_max,y_min,y_max), got: {s}")
    try:
        x1, x2, y1, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError as e:
        raise ArgumentTypeError(f"values must be numbers, got: {s}") from e
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    if x_min == x_max:
        raise ArgumentTypeError(f"x_min and x_max cannot be equal ({x_min})")
    if y_min == y_max:
        raise ArgumentTypeError(f"y_min and y_max cannot be equal ({y_min})")
    return AreaRect(x_min, x_max, y_min, y_max)


def parse_area_rect_i(s: str) -> AreaRectI:
    """Parse 'x_min,x_max,y_min,y_max' string into AreaRectI (integer pixels).

    Delegates to :func:`parse_area_rect` and truncates to int.
    """
    r = parse_area_rect(s)
    return AreaRectI(int(r.x_min), int(r.x_max), int(r.y_min), int(r.y_max))


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
    """Interpolate position at time t from position samples.

    Uses t_after as the effective sample time — the SDK position read
    completes at the end of the call, so t_after best represents when
    the encoder value was latched.
    """
    if not samples:
        return None

    times = [s.t_after for s in samples]
    idx = bisect.bisect_left(times, t)

    if idx == 0:
        return samples[0].axis_um
    if idx >= len(samples):
        return samples[-1].axis_um

    t0, x0 = times[idx - 1], samples[idx - 1].axis_um
    t1, x1 = times[idx], samples[idx].axis_um

    if t1 == t0:
        return x0

    alpha = (t - t0) / (t1 - t0)
    return x0 + alpha * (x1 - x0)


def smooth_frame_positions(meta: ScanMeta, *, quiet: bool = False) -> None:
    """Smooth frame x_um and x_vel_um_s using the raw position sample stream.

    Uses t_after as the timing base for position samples — the SDK read
    completes at the end of the call, so t_after best represents when
    the encoder was latched.  Fits a smoothing spline (UnivariateSpline)
    which handles non-uniform time spacing correctly, unlike savgol which
    assumes uniform sample intervals.

    Applies a velocity-proportional timing correction to align position
    samples (read at t_after) with frame exposures (at t_start).  The
    correction is largest at full scan velocity and zero at row edges
    during accel/decel.

    Drops sparse boundary samples (from adaptive 10→100 Hz polling ramp).
    Frames outside the dense region keep their raw interpolated positions.

    Modifies meta["frames"] in place.  No-op if position_stream is absent.
    """
    # Empirical offset between position read (t_after) and frame exposure
    # (t_start).  Calibrated from USB (D2XX) scans by minimizing the
    # directional offset between +X and -X rows.
    _TIMING_OFFSET_S = 0.000825

    if "position_stream" not in meta:
        return

    ps = meta["position_stream"]
    frames = meta["frames"]
    lines = meta["lines"]

    count = 0
    for line in lines:
        line_idx = line["line_idx"]
        line_ps = [s for s in ps if s["line"] == line_idx]
        if len(line_ps) < 5:
            continue

        t_ps = np.array([s["t_after"] for s in line_ps])
        x_ps = np.array([s["x_um"] for s in line_ps])

        # Estimate position noise from second-differences, corrected for
        # non-uniform time spacing.  Raw d²x = v*d²t + noise; subtract
        # the velocity contribution so only noise remains.
        d2x = np.diff(x_ps, 2)
        d2t = np.diff(t_ps, 2)
        v_triplet = (x_ps[2:] - x_ps[:-2]) / (t_ps[2:] - t_ps[:-2])
        d2x_corrected = d2x - v_triplet * d2t
        noise_um = float(np.median(np.abs(d2x_corrected)) / 0.6745 / np.sqrt(6))
        s_factor = len(t_ps) * noise_um**2

        spline = UnivariateSpline(t_ps, x_ps, k=3, s=s_factor)
        spline_deriv = spline.derivative()

        for fi in range(line["frame_start"], line["frame_end"]):
            f = frames[fi]
            t = f["t_capture"]
            if t_ps[0] <= t <= t_ps[-1]:
                vel = float(spline_deriv(t))
                f["x_um"] = float(spline(t)) - vel * _TIMING_OFFSET_S
                f["x_vel_um_s"] = vel
                count += 1

    if not quiet:
        print(f"Position smoothing: updated {count} frames from {len(ps)} raw samples")


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


def intersect_polygon_with_y_band(
    polygon: Sequence[Point2F],
    y_center: float,
    frame_height: float,
) -> Point2F | None:
    """Find widest X extent where a horizontal band intersects a polygon.

    The band covers [y_center - frame_height/2, y_center + frame_height/2].
    For a convex polygon with straight edges, the X extent is piecewise-linear
    in Y, so the extremes occur at band edges or polygon vertex Y values.

    Returns:
        (x_min, x_max) envelope or None if no Y in the band intersects.
    """
    y_top = y_center - frame_height / 2
    y_bot = y_center + frame_height / 2

    # Candidate Y values: band edges + polygon vertices within the band
    candidates = [y_top, y_bot]
    for _, vy in polygon:
        if y_top <= vy <= y_bot:
            candidates.append(vy)

    x_min_all = float("inf")
    x_max_all = float("-inf")
    any_hit = False

    for y in candidates:
        extent = intersect_polygon_with_y(polygon, y)
        if extent is not None:
            x_min_all = min(x_min_all, extent[0])
            x_max_all = max(x_max_all, extent[1])
            any_hit = True

    if not any_hit:
        return None

    return Point2F(x_min_all, x_max_all)


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
        extent = intersect_polygon_with_y_band(polygon, y, frame_height_um)
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
        inputs=PlanInputs(
            bbox=bbox,
            polygon=list(polygon),
            x_overlap_pct=x_overlap_pct,
            y_overlap_pct=y_overlap_pct,
            padding=padding,
            row_limit=row_limit,
            speed_mm=speed_mm,
            z_max=z_max,
            plane_a=plane_a,
            plane_b=plane_b,
            plane_c=plane_c,
        ),
    )


# ============================================================================
# Metadata helpers
# ============================================================================


def build_lighting_meta(
    *,
    lamp: Lamp | None = None,
    shutter: Shutter | None = None,
    aperture: Aperture | None = None,
) -> LightingMeta:
    """Build lighting metadata dict from hardware objects."""
    return LightingMeta(
        lamp_name=lamp.name if lamp else None,
        lamp_intensity=lamp.intensity if lamp else None,
        lamp_max_intensity=lamp.max_intensity if lamp else None,
        shutter_name=shutter.name if shutter else None,
        shutter_open=shutter.is_open if shutter else None,
        aperture_value=aperture.value if aperture else None,
        aperture_max_value=aperture.max_value if aperture else None,
    )


def build_camera_meta(camera: Camera) -> CameraMeta:
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
    nosepiece: Nosepiece,
    camera: Camera,
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


def build_microscope_meta(scope: Microscope) -> MicroscopeMeta:
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
