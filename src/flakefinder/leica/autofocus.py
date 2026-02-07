"""Continuous Z-scan autofocus.

Performs autofocus by scanning Z while capturing frames, computing sharpness
for each frame, and returning the Z position with maximum sharpness.

The scan direction is always downward (decreasing Z = away from sample)
for safety. Z range is auto-calculated from the current objective's
working distance if not specified.
"""

from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .core import get_interface_required, find_unit
from .enums import TID, UCAPI_IID
from .units import Axis, Nosepiece, ZDrive

if TYPE_CHECKING:
    from .core import LeicaConnection
    from .camera import Camera


# Working distances in µm by objective position (from stage_util.py)
# Position 1-indexed as used by the Nosepiece class.
WORKING_DISTANCES_UM: dict[int, float] = {
    1: 12700,   # 5x
    2: 11000,   # 10x
    3: 1900,    # 20x
    4: 380,     # 50x
    5: 210,     # 150x
    6: 15000,   # 2.5x
}


def sharpness_tenengrad(image: np.ndarray) -> float:
    """Compute Tenengrad sharpness (Sobel gradient magnitude mean).

    Args:
        image: RGB or grayscale image as numpy array.

    Returns:
        Sharpness value (higher = sharper).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.mean(cv2.magnitude(sobel_x, sobel_y))[0]


def sharpness_laplacian(image: np.ndarray) -> float:
    """Compute Laplacian variance sharpness.

    More reliable than Tenengrad for detecting actual focus quality.
    Less susceptible to being fooled by bright blurry blobs.

    Args:
        image: RGB or grayscale image as numpy array.

    Returns:
        Sharpness value (higher = sharper).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def sharpness_brenner(image: np.ndarray) -> float:
    """Brenner gradient sharpness."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    g = gray.astype(np.float64)
    diff = g[:, 2:] - g[:, :-2]
    return np.mean(diff**2)


def sharpness_normalized_variance(image: np.ndarray) -> float:
    """Normalized variance sharpness."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    mu = gray.mean()
    return gray.var() / mu if mu > 0 else 0


def sharpness_vollath_f4(image: np.ndarray) -> float:
    """Vollath F4 autocorrelation sharpness."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    g = gray.astype(np.float64)
    auto1 = np.mean(g[:, :-1] * g[:, 1:])
    auto2 = np.mean(g[:, :-2] * g[:, 2:])
    return auto1 - auto2


ALL_SHARPNESS_METRICS = {
    "tenengrad": sharpness_tenengrad,
    "laplacian": sharpness_laplacian,
    "brenner": sharpness_brenner,
    "normalized_variance": sharpness_normalized_variance,
    "vollath_f4": sharpness_vollath_f4,
}


# Default sharpness function (kept for backwards compatibility)
def sharpness(image: np.ndarray, method: str = "tenengrad") -> float:
    """Compute sharpness using specified method.

    Args:
        image: RGB or grayscale image as numpy array.
        method: "tenengrad" or "laplacian"

    Returns:
        Sharpness value (higher = sharper).
    """
    if method == "laplacian":
        return sharpness_laplacian(image)
    return sharpness_tenengrad(image)


def interpolate_position(t: float, samples: list[tuple[float, float, float]]) -> float | None:
    """Interpolate position at time t from polled position samples.

    Uses midpoint of t_before/t_after as the effective sample time
    and performs linear interpolation between adjacent samples.

    Args:
        t: Time to interpolate at (from time.perf_counter()).
        samples: List of (t_before, t_after, z_um) tuples from position polling.

    Returns:
        Interpolated Z position in µm, or None if samples is empty.
    """
    if not samples:
        return None

    # Use midpoint of before/after as effective time
    times = [(s[0] + s[1]) / 2 for s in samples]

    # Find insertion point
    idx = bisect.bisect_left(times, t)

    if idx == 0:
        return samples[0][2]  # Before first sample
    if idx >= len(samples):
        return samples[-1][2]  # After last sample

    # Linear interpolate between samples[idx-1] and samples[idx]
    t0, z0 = times[idx - 1], samples[idx - 1][2]
    t1, z1 = times[idx], samples[idx][2]

    if t1 == t0:
        return z0

    alpha = (t - t0) / (t1 - t0)
    return z0 + alpha * (z1 - z0)


@dataclass
class AutofocusFrame:
    """Single frame from autofocus scan."""

    z_um: float
    sharpness: float
    image: np.ndarray | None = None  # Only populated if store_frames=True


@dataclass
class AutofocusResult:
    """Result from autofocus operation."""

    best_z_um: float
    best_sharpness: float
    initial_z_um: float
    initial_sharpness: float
    final_sharpness: float
    z_range_um: float              # Actual range used
    objective_position: int | None  # Queried from microscope
    scan_duration_s: float
    frame_count: int
    z_sample_count: int
    # Diagnostic fields for debugging autofocus issues
    coarse_z_start_um: float = 0.0
    coarse_z_end_um: float = 0.0
    coarse_best_z_um: float = 0.0
    coarse_best_sharpness: float = 0.0
    fine_z_start_um: float | None = None  # None if no fine pass
    fine_z_end_um: float | None = None
    stayed_at_initial: bool = False  # True if scan found nothing better than initial
    sharpness_curve: list[dict] = field(default_factory=list)  # [{z_um, sharpness}, ...] coarse only
    frames: list[AutofocusFrame] | None = None  # Coarse frames only (if store_frames=True)
    fine_sharpness_curve: list[dict] = field(default_factory=list)  # Fine pass only
    fine_frames: list[AutofocusFrame] | None = None  # Fine frames only (if store_frames=True)


def _get_safe_range(conn: "LeicaConnection", z_range_um: float | None) -> tuple[float, int | None]:
    """Query microscope and calculate safe Z range.

    Args:
        conn: Active LeicaConnection for querying objective.
        z_range_um: Explicit range if provided, or None for auto-calculation.

    Returns:
        (safe_range_um, objective_position) tuple.

    Raises:
        ValueError: If explicit z_range_um exceeds safe limit for objective.
    """
    # Query current objective from microscope
    objective_position = None
    working_distance = None
    try:
        nosepiece = Nosepiece.from_connection(conn)
        objective_position = nosepiece.position
        working_distance = WORKING_DISTANCES_UM.get(objective_position)
    except LookupError:
        pass  # No nosepiece available

    if z_range_um is not None:
        # Explicit range provided - validate against objective
        if working_distance and z_range_um > working_distance / 2:
            raise ValueError(
                f"Z range {z_range_um}µm exceeds safe limit for "
                f"objective position {objective_position} (working distance: {working_distance}µm)"
            )
        return z_range_um, objective_position

    # Auto-calculate from objective
    if working_distance:
        return min(working_distance / 3, 500), objective_position

    # Unknown objective - use conservative default
    return 200, objective_position  # Safe for all objectives


def _validate_z_limits(
    z_axis: Axis,
    z_start: float,
    z_end: float,
    z_max_safe: float | None,
) -> None:
    """Validate Z positions against axis and safety limits.

    Args:
        z_axis: Z axis for checking hardware limits.
        z_start: Starting Z position in µm.
        z_end: Ending Z position in µm.
        z_max_safe: Optional hard upper limit in µm.

    Raises:
        ValueError: If any position exceeds limits.
    """
    if z_start > z_axis.max_um:
        raise ValueError(f"Z start {z_start:.1f}µm exceeds axis max {z_axis.max_um:.1f}µm")
    if z_end < z_axis.min_um:
        raise ValueError(f"Z end {z_end:.1f}µm below axis min {z_axis.min_um:.1f}µm")
    if z_max_safe is not None and z_start > z_max_safe:
        raise ValueError(f"Z start {z_start:.1f}µm exceeds safe max {z_max_safe:.1f}µm")


def _run_z_scan(
    z_axis: Axis,
    z_start: float,
    z_end: float,
    acquisition,
    context,
    camera_class,
    store_frames: bool,
    sharpness_method: str = "tenengrad",
    compute_all_metrics: bool = False,
) -> tuple[list[dict], list[AutofocusFrame] | None, float, int, int]:
    """Execute Z scan and capture frames.

    Args:
        z_axis: Z axis for movement.
        z_start: Starting Z position in µm.
        z_end: Ending Z position in µm.
        acquisition: SDK acquisition interface.
        context: SDK acquisition context.
        camera_class: Camera class for image conversion.
        store_frames: Whether to store images in result.
        sharpness_method: "tenengrad" or "laplacian".
        compute_all_metrics: If True, compute all 5 sharpness metrics per frame.

    Returns:
        (sharpness_curve, frames, duration, frame_count, z_sample_count) tuple.
    """
    # Data collection
    z_samples: list[tuple[float, float, float]] = []  # (t_before, t_after, z_um)
    frame_data: list[tuple[float, np.ndarray]] = []  # (t_capture, image)
    stop_polling = threading.Event()

    # Get fast position reading interfaces
    z_bcv = z_axis.bcv
    z_converter = z_axis.converter

    current_image = [None]

    def on_image(image):
        current_image[0] = image

    # Get delegate class from SDK (available after ExUCAPI.Register())
    from LeicaMicrosystems.HardwareModel import Extensions
    context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

    def z_poll_thread():
        """Poll Z position continuously during scan."""
        while not stop_polling.is_set():
            t_before = time.perf_counter()
            z_native = z_bcv.GetControlValue()
            t_after = time.perf_counter()
            z_um = z_converter.GetMetricsValue(z_native)
            z_samples.append((t_before, t_after, z_um))

    # Move to scan start position
    z_axis.move_to(z_start)
    time.sleep(0.1)  # Brief settle

    # Start Z polling
    z_thread = threading.Thread(target=z_poll_thread, daemon=True)
    z_thread.start()

    scan_start_time = time.perf_counter()

    # Start async Z move (downward)
    z_handle = z_axis.move_to_async(z_end)

    # Capture frames during move
    while not z_handle.is_complete:
        t_capture = time.perf_counter()
        current_image[0] = None
        acquisition.Acquire(context, None)

        if current_image[0] is not None:
            # Convert immediately and dispose .NET object
            img_arr = camera_class._image_to_numpy(current_image[0])
            current_image[0].Dispose()
            frame_data.append((t_capture, img_arr))

    scan_end_time = time.perf_counter()
    z_handle.dispose()

    # Stop polling
    stop_polling.set()
    z_thread.join(timeout=1.0)

    scan_duration = scan_end_time - scan_start_time

    # Compute sharpness for each frame
    sharpness_curve = []
    frames = [] if store_frames else None

    for i, (t_capture, img) in enumerate(frame_data):
        z_interp = interpolate_position(t_capture, z_samples)
        s = sharpness(img, method=sharpness_method)
        entry = {
            "frame": i,
            "z_um": z_interp,
            "sharpness": s,
        }
        if compute_all_metrics:
            entry["metrics"] = {
                name: fn(img) for name, fn in ALL_SHARPNESS_METRICS.items()
            }
        sharpness_curve.append(entry)
        if store_frames:
            frames.append(AutofocusFrame(z_um=z_interp, sharpness=s, image=img))

    return sharpness_curve, frames, scan_duration, len(frame_data), len(z_samples)


def continuous_autofocus(
    conn: "LeicaConnection",
    camera: "Camera",
    acquisition,
    context,  # TODO: create context internally instead of requiring caller to pass it
    z_range_um: float | None = None,
    z_start_um: float | None = None,
    z_max_safe_um: float | None = None,
    z_speed_um_s: float | None = None,
    fine_pass: bool = False,
    fine_range_um: float = 50.0,
    fine_speed_factor: float = 0.25,
    sharpness_method: str = "tenengrad",
    store_frames: bool = False,
    compute_all_metrics: bool = False,
    backlash_overshoot_um: float = 100.0,
    settle_time_s: float = 0.2,
) -> AutofocusResult:
    """Perform continuous Z-scan autofocus.

    Scans the Z axis downward while capturing frames, computes sharpness
    for each frame, and moves to the Z position with maximum sharpness.

    Safety behavior:
    - Queries nosepiece directly to get current objective position
    - Auto-calculates safe range from objective's working distance
    - Raises ValueError if z_range_um exceeds safe range for objective
    - Raises ValueError if z_start_um would exceed limits
    - If scan finds nothing better than initial sharpness, stays at initial Z

    Args:
        conn: Active LeicaConnection (used to query objective for safety).
        camera: Camera instance for image capture.
        acquisition: SDK acquisition interface (from get_interface_required).
        context: SDK CancellableImageAcquisitionContext.
        z_range_um: Z scan range in µm. None = auto from objective
            (working_distance / 3, max 500µm).
        z_start_um: Starting Z position in µm. None = current position.
        z_max_safe_um: Hard upper limit for Z. Raises if z_start > this.
        z_speed_um_s: Z axis speed in µm/s. None = use current speed.
            Slower speeds capture more frames for better precision.
        fine_pass: If True, do a second pass with fine_range_um around best Z.
        fine_range_um: Range for fine pass (default 50µm).
        fine_speed_factor: Speed multiplier for fine pass (default 0.25 = 1/4 speed).
            Slower fine pass improves precision in the critical region.
        sharpness_method: "tenengrad" (default) or "laplacian". Laplacian is more
            reliable for low-contrast areas and less fooled by bright blurry blobs.
        store_frames: If True, store images in result.frames for debugging.
        compute_all_metrics: If True, compute all 5 sharpness metrics per frame
            (tenengrad, laplacian, brenner, normalized_variance, vollath_f4).
            Results stored in each sharpness_curve entry's "metrics" dict.
        backlash_overshoot_um: Overshoot distance above best Z before final
            approach (default 100µm). Ensures final move approaches from above,
            matching the scan direction, to avoid ~45µm backlash error.
            Set to 0 to disable.
        settle_time_s: Settle time in seconds after final Z move, before
            capturing final_sharpness (default 0.2s).

    Returns:
        AutofocusResult with best Z, sharpness curve, and scan statistics.
        If stayed_at_initial is True, the scan found nothing better than
        the initial position and did not move.

    Raises:
        ValueError: If Z range/position exceeds safety limits.
        LookupError: If Z drive not found.
    """
    from .camera import Camera

    # Get Z axis
    z_axis = ZDrive.from_connection(conn)
    current_z = z_axis.position_um

    # Save original speed (for fine pass and restoration)
    original_speed = None
    if z_axis.supports_velocity:
        original_speed = z_axis.velocity_um_s

    # Set Z speed if specified
    if z_speed_um_s is not None and z_axis.supports_velocity:
        z_axis.set_velocity_um_s(z_speed_um_s)

    # Get safe range based on objective
    safe_range, objective_position = _get_safe_range(conn, z_range_um)

    # Calculate scan bounds (centered on current/specified position, scan downward)
    initial_z = z_start_um if z_start_um is not None else current_z
    z_start = initial_z + safe_range / 2  # Start high
    z_end = initial_z - safe_range / 2    # End low (away from sample)

    # Validate against limits
    _validate_z_limits(z_axis, z_start, z_end, z_max_safe_um)

    # Capture initial sharpness at current position
    z_axis.move_to(initial_z)
    time.sleep(0.05)
    initial_image = camera.capture()
    initial_sharpness = sharpness(initial_image, method=sharpness_method) if initial_image is not None else 0.0

    # Run main scan
    sharpness_curve, frames, scan_duration, frame_count, z_sample_count = _run_z_scan(
        z_axis=z_axis,
        z_start=z_start,
        z_end=z_end,
        acquisition=acquisition,
        context=context,
        camera_class=Camera,
        store_frames=store_frames,
        sharpness_method=sharpness_method,
        compute_all_metrics=compute_all_metrics,
    )

    if not sharpness_curve:
        raise ValueError("No frames captured during autofocus scan")

    # Find best frame from coarse pass
    best = max(sharpness_curve, key=lambda r: r["sharpness"])
    best_z = best["z_um"]
    best_sharpness = best["sharpness"]

    # Track coarse results for diagnostics
    coarse_best_z = best_z
    coarse_best_sharpness = best_sharpness
    actual_fine_z_start = None
    actual_fine_z_end = None

    # Optional fine pass
    if fine_pass:
        fine_z_start = best_z + fine_range_um / 2
        fine_z_end = best_z - fine_range_um / 2

        # Clamp to axis limits
        fine_z_start = min(fine_z_start, z_axis.max_um)
        fine_z_end = max(fine_z_end, z_axis.min_um)

        # Record actual fine pass bounds for diagnostics
        actual_fine_z_start = fine_z_start
        actual_fine_z_end = fine_z_end

        # Use slower speed for fine pass (better precision)
        if z_axis.supports_velocity and fine_speed_factor < 1.0:
            coarse_speed = z_speed_um_s if z_speed_um_s is not None else original_speed
            if coarse_speed is not None:
                z_axis.set_velocity_um_s(coarse_speed * fine_speed_factor)

        fine_curve, fine_frames, fine_duration, fine_frame_count, fine_z_count = _run_z_scan(
            z_axis=z_axis,
            z_start=fine_z_start,
            z_end=fine_z_end,
            acquisition=acquisition,
            context=context,
            camera_class=Camera,
            store_frames=store_frames,
            sharpness_method=sharpness_method,
            compute_all_metrics=compute_all_metrics,
        )

        # Restore speed after fine pass (before final move)
        if z_axis.supports_velocity and original_speed is not None:
            z_axis.set_velocity_um_s(z_speed_um_s if z_speed_um_s is not None else original_speed)

        if fine_curve:
            fine_best = max(fine_curve, key=lambda r: r["sharpness"])
            if fine_best["sharpness"] > best_sharpness:
                best_z = fine_best["z_um"]
                best_sharpness = fine_best["sharpness"]

            scan_duration += fine_duration
            frame_count += fine_frame_count
            z_sample_count += fine_z_count

    # Accept scan best if it's within margin of initial reading.
    # On low-contrast substrates, frame-to-frame noise can cause
    # the initial reading to randomly exceed the true peak.
    initial_margin = 0.95  # 5% margin
    stayed_at_initial = best_sharpness < initial_sharpness * initial_margin

    if stayed_at_initial:
        # Scan found nothing better - stay at initial position
        best_z = initial_z
        best_sharpness = initial_sharpness

    # Move to best Z position (or back to initial if stayed_at_initial)
    # Backlash compensation: overshoot above, then approach from above
    # (matching the downward scan direction) to avoid ~45µm hysteresis error.
    if backlash_overshoot_um > 0:
        z_axis.move_to(best_z + backlash_overshoot_um)
        time.sleep(0.05)
    z_axis.move_to_corrected(best_z)
    time.sleep(settle_time_s)

    # Restore original Z speed if we changed it
    if original_speed is not None:
        z_axis.set_velocity_um_s(original_speed)

    # Capture final sharpness
    final_image = camera.capture()
    final_sharpness = sharpness(final_image, method=sharpness_method) if final_image is not None else 0.0

    return AutofocusResult(
        best_z_um=best_z,
        best_sharpness=best_sharpness,
        initial_z_um=initial_z,
        initial_sharpness=initial_sharpness,
        final_sharpness=final_sharpness,
        z_range_um=safe_range,
        objective_position=objective_position,
        scan_duration_s=scan_duration,
        frame_count=frame_count,
        z_sample_count=z_sample_count,
        coarse_z_start_um=z_start,
        coarse_z_end_um=z_end,
        coarse_best_z_um=coarse_best_z,
        coarse_best_sharpness=coarse_best_sharpness,
        fine_z_start_um=actual_fine_z_start,
        fine_z_end_um=actual_fine_z_end,
        stayed_at_initial=stayed_at_initial,
        sharpness_curve=sharpness_curve,
        frames=frames if store_frames else None,
        fine_sharpness_curve=fine_curve if fine_pass and fine_curve else [],
        fine_frames=fine_frames if store_frames and fine_pass else None,
    )
