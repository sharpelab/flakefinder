"""Continuous Z-scan autofocus.

Performs autofocus by scanning Z while capturing frames, computing sharpness
for each frame, and returning the Z position with maximum sharpness.

The scan direction is always downward (decreasing Z = away from sample)
for safety. Z range is auto-calculated from the current objective's
working distance if not specified.
"""

import bisect
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..image_utils import sdk_image_to_numpy
from .units import Axis, Nosepiece, ZDrive

if TYPE_CHECKING:
    from .microscope import Microscope

# Working distances in µm by objective position (from stage_util.py)
# Position 1-indexed as used by the Nosepiece class.
WORKING_DISTANCES_UM: dict[int, float] = {
    1: 12700,  # 5x
    2: 11000,  # 10x
    3: 1900,  # 20x
    4: 380,  # 50x
    5: 210,  # 150x
    6: 15000,  # 2.5x
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


def sharpness(image: np.ndarray, method: str = "tenengrad") -> float:
    """Compute sharpness using specified method.

    Args:
        image: RGB or grayscale image as numpy array.
        method: One of ALL_SHARPNESS_METRICS keys.

    Returns:
        Sharpness value (higher = sharper).
    """
    fn = ALL_SHARPNESS_METRICS.get(method)
    if fn is None:
        raise ValueError(f"Unknown sharpness method: {method}")
    return float(fn(image))


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

    selected_z_um: float  # Selected Z position (may be initial if stayed_at_initial)
    selected_sharpness: float  # Sharpness at the Z we actually went to
    scan_best_z_um: float  # Best Z found during scanning (before stayed_at_initial override)
    scan_best_sharpness: float  # Best sharpness from scanning (before stayed_at_initial override)
    initial_z_um: float
    initial_sharpness: float
    final_sharpness: float
    dynamic_range: float  # (s_max - s_min) / s_mean of coarse curve
    z_range_um: float  # Actual range used
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
    super_fine_z_start_um: float | None = None  # None if no super fine pass
    super_fine_z_end_um: float | None = None
    stayed_at_initial: bool = False  # True if scan found nothing better than initial
    sharpness_curve: list[dict] = field(default_factory=list)  # [{z_um, sharpness}, ...] coarse only
    frames: list[AutofocusFrame] | None = None  # Coarse frames only (if store_frames=True)
    fine_sharpness_curve: list[dict] = field(default_factory=list)  # Fine pass only
    fine_frames: list[AutofocusFrame] | None = None  # Fine frames only (if store_frames=True)
    super_fine_sharpness_curve: list[dict] = field(default_factory=list)  # Super fine pass only
    super_fine_frames: list[AutofocusFrame] | None = None  # Super fine frames only (if store_frames=True)
    initial_image: np.ndarray | None = None  # Image at initial Z (if store_frames=True)
    final_image: np.ndarray | None = None  # Image at selected Z after move (if store_frames=True)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (excludes frames and images)."""
        return {
            "selected": {"z_um": self.selected_z_um, "sharpness": self.selected_sharpness},
            "scan_best": {"z_um": self.scan_best_z_um, "sharpness": self.scan_best_sharpness},
            "initial": {"z_um": self.initial_z_um, "sharpness": self.initial_sharpness},
            "final_sharpness": self.final_sharpness,
            "dynamic_range": self.dynamic_range,
            "stayed_at_initial": self.stayed_at_initial,
            "scan": {
                "z_range_um": self.z_range_um,
                "duration_s": self.scan_duration_s,
                "frame_count": self.frame_count,
                "z_sample_count": self.z_sample_count,
                "objective_position": self.objective_position,
            },
            "coarse": {
                "z_start_um": self.coarse_z_start_um,
                "z_end_um": self.coarse_z_end_um,
                "best_z_um": self.coarse_best_z_um,
                "best_sharpness": self.coarse_best_sharpness,
            },
            "fine": {
                "z_start_um": self.fine_z_start_um,
                "z_end_um": self.fine_z_end_um,
            },
            "super_fine": {
                "z_start_um": self.super_fine_z_start_um,
                "z_end_um": self.super_fine_z_end_um,
            },
            "sharpness_curve": self.sharpness_curve,
            "fine_sharpness_curve": self.fine_sharpness_curve,
            "super_fine_sharpness_curve": self.super_fine_sharpness_curve,
        }


def _get_safe_range(nosepiece: Nosepiece, z_range_um: float | None) -> tuple[float, int | None]:
    """Calculate safe Z range from current objective.

    Args:
        nosepiece: Nosepiece instance for querying objective position.
        z_range_um: Explicit range if provided, or None for auto-calculation.

    Returns:
        (safe_range_um, objective_position) tuple.

    Raises:
        ValueError: If explicit z_range_um exceeds safe limit for objective.
    """
    objective_position = nosepiece.position
    working_distance = WORKING_DISTANCES_UM[objective_position]

    if z_range_um is not None:
        if z_range_um > working_distance / 2:
            raise ValueError(
                f"Z range {z_range_um}µm exceeds safe limit for "
                f"objective position {objective_position} (working distance: {working_distance}µm)"
            )
        return z_range_um, objective_position

    return min(working_distance / 3, 500), objective_position


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
    z_axis: ZDrive,
    z_start: float,
    z_end: float,
    acquisition,
    context,
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

    # Get fast position reading interfaces (prefer hysteresis-corrected for accurate Z during motion)
    z_bcv = getattr(z_axis, "bcv_hysteresis", None) or z_axis.bcv
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
    z_axis.move_to_corrected(z_start)
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
            img_arr = sdk_image_to_numpy(current_image[0])
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
        if z_interp is None:
            raise RuntimeError(f"Frame {i}: Z interpolation failed at t={t_capture:.6f}s")
        s = sharpness(img, method=sharpness_method)
        entry = {
            "frame": i,
            "z_um": z_interp,
            "sharpness": s,
        }
        if compute_all_metrics:
            entry["metrics"] = {name: float(fn(img)) for name, fn in ALL_SHARPNESS_METRICS.items()}
        sharpness_curve.append(entry)
        if store_frames:
            assert frames is not None
            frames.append(AutofocusFrame(z_um=z_interp, sharpness=s, image=img))

    return sharpness_curve, frames, scan_duration, len(frame_data), len(z_samples)


def continuous_autofocus(
    scope: "Microscope",
    *,
    z_range_um: float | None = None,
    z_start_um: float | None = None,
    z_max_safe_um: float | None = None,
    z_speed_um_s: float | None = None,
    fine_pass: bool = False,
    fine_range_um: float = 50.0,
    fine_speed_factor: float = 0.25,
    super_fine_pass: bool = False,
    super_fine_range_um: float = 10.0,
    super_fine_speed_um_s: float = 20.0,
    sharpness_method: str = "tenengrad",
    store_frames: bool = False,
    compute_all_metrics: bool = False,
    settle_time_s: float = 0.2,
    min_dynamic_range: float = 0.20,
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
        scope: Microscope facade instance.
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
        super_fine_pass: If True, do a third pass with super_fine_range_um around
            best Z at super_fine_speed_um_s. Requires fine_pass or runs after coarse.
        super_fine_range_um: Range for super fine pass (default 10µm).
        super_fine_speed_um_s: Absolute Z speed for super fine pass (default 20µm/s).
        sharpness_method: "tenengrad" (default) or "laplacian". Laplacian is more
            reliable for low-contrast areas and less fooled by bright blurry blobs.
        store_frames: If True, store images in result.frames for debugging.
        compute_all_metrics: If True, compute all 5 sharpness metrics per frame
            (tenengrad, laplacian, brenner, normalized_variance, vollath_f4).
            Results stored in each sharpness_curve entry's "metrics" dict.
        settle_time_s: Settle time in seconds after final Z move, before
            capturing final_sharpness (default 0.2s).
        min_dynamic_range: Minimum (s_max - s_min) / s_mean for the coarse
            sharpness curve to be considered valid (default 0.05 = 5%).
            Below this threshold the curve is flat noise and no real focus
            exists, so stayed_at_initial is set and fine pass is skipped.

    Returns:
        AutofocusResult with best Z, sharpness curve, and scan statistics.
        If stayed_at_initial is True, the scan found nothing better than
        the initial position and did not move.

    Raises:
        ValueError: If Z range/position exceeds safety limits.
    """
    # Extract subsystems from facade
    z_axis = scope.z
    camera = scope.camera
    acquisition = scope.acquisition
    context = scope.context

    current_z = z_axis.position_um

    # Save original speed for restoration after scan
    original_speed = z_axis.velocity_um_s

    # Set Z speed if specified
    if z_speed_um_s is not None:
        z_axis.set_velocity_um_s(z_speed_um_s)

    # Get safe range based on objective
    safe_range, objective_position = _get_safe_range(scope.nosepiece, z_range_um)

    # Calculate scan bounds (centered on current/specified position, scan downward)
    initial_z = z_start_um if z_start_um is not None else current_z
    z_start = initial_z + safe_range / 2  # Start high
    z_end = initial_z - safe_range / 2  # End low (away from sample)

    # Validate against limits
    _validate_z_limits(z_axis, z_start, z_end, z_max_safe_um)

    # Capture initial sharpness at current position (flush stale sensor buffer first)
    z_axis.move_to_corrected(initial_z)
    time.sleep(0.05)
    camera.capture()
    initial_image = camera.capture()
    initial_sharpness = sharpness(initial_image, method=sharpness_method)
    stored_initial_image = initial_image if store_frames else None

    # Run main scan
    sharpness_curve, frames, scan_duration, frame_count, z_sample_count = _run_z_scan(
        z_axis=z_axis,
        z_start=z_start,
        z_end=z_end,
        acquisition=acquisition,
        context=context,
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

    # Check if sharpness curve has meaningful variation
    sharpness_values = [r["sharpness"] for r in sharpness_curve]
    s_min, s_max = min(sharpness_values), max(sharpness_values)
    s_mean = sum(sharpness_values) / len(sharpness_values)
    dynamic_range = (s_max - s_min) / s_mean if s_mean > 0 else 0

    if dynamic_range < min_dynamic_range:
        # Curve is flat noise — no real focus found
        stayed_at_initial = True
    else:
        stayed_at_initial = False

    # Track coarse results for diagnostics
    coarse_best_z = best_z
    coarse_best_sharpness = best_sharpness
    actual_fine_z_start = None
    actual_fine_z_end = None

    # Optional fine pass (skip if dynamic range too low)
    if fine_pass and not stayed_at_initial:
        fine_z_start = best_z + fine_range_um / 2
        fine_z_end = best_z - fine_range_um / 2

        # Clamp to axis limits
        fine_z_start = min(fine_z_start, z_axis.max_um)
        fine_z_end = max(fine_z_end, z_axis.min_um)

        # Record actual fine pass bounds for diagnostics
        actual_fine_z_start = fine_z_start
        actual_fine_z_end = fine_z_end

        # Position at full speed, then set slow scan speed
        z_axis.move_to_corrected(fine_z_start)
        if fine_speed_factor < 1.0:
            coarse_speed = z_speed_um_s if z_speed_um_s is not None else original_speed
            z_axis.set_velocity_um_s(coarse_speed * fine_speed_factor)

        fine_curve, fine_frames, fine_duration, fine_frame_count, fine_z_count = _run_z_scan(
            z_axis=z_axis,
            z_start=fine_z_start,
            z_end=fine_z_end,
            acquisition=acquisition,
            context=context,
            store_frames=store_frames,
            sharpness_method=sharpness_method,
            compute_all_metrics=compute_all_metrics,
        )

        # Restore speed after fine pass (before final move)
        z_axis.set_velocity_um_s(z_speed_um_s if z_speed_um_s is not None else original_speed)

        if fine_curve:
            fine_best = max(fine_curve, key=lambda r: r["sharpness"])
            if fine_best["sharpness"] > best_sharpness:
                best_z = fine_best["z_um"]
                best_sharpness = fine_best["sharpness"]

            scan_duration += fine_duration
            frame_count += fine_frame_count
            z_sample_count += fine_z_count

    # Optional super fine pass (skip if stayed_at_initial)
    super_fine_curve = []
    super_fine_frames_result = None
    actual_super_fine_z_start = None
    actual_super_fine_z_end = None

    if super_fine_pass and not stayed_at_initial:
        sf_z_start = best_z + super_fine_range_um / 2
        sf_z_end = best_z - super_fine_range_um / 2

        # Clamp to axis limits
        sf_z_start = min(sf_z_start, z_axis.max_um)
        sf_z_end = max(sf_z_end, z_axis.min_um)

        actual_super_fine_z_start = sf_z_start
        actual_super_fine_z_end = sf_z_end

        # Position at full speed, then set slow scan speed
        z_axis.move_to_corrected(sf_z_start)
        z_axis.set_velocity_um_s(super_fine_speed_um_s)

        sf_curve, sf_frames, sf_duration, sf_frame_count, sf_z_count = _run_z_scan(
            z_axis=z_axis,
            z_start=sf_z_start,
            z_end=sf_z_end,
            acquisition=acquisition,
            context=context,
            store_frames=store_frames,
            sharpness_method=sharpness_method,
            compute_all_metrics=compute_all_metrics,
        )

        # Restore speed after super fine pass
        z_axis.set_velocity_um_s(z_speed_um_s if z_speed_um_s is not None else original_speed)

        if sf_curve:
            super_fine_curve = sf_curve
            super_fine_frames_result = sf_frames
            sf_best = max(sf_curve, key=lambda r: r["sharpness"])
            if sf_best["sharpness"] > best_sharpness:
                best_z = sf_best["z_um"]
                best_sharpness = sf_best["sharpness"]

            scan_duration += sf_duration
            frame_count += sf_frame_count
            z_sample_count += sf_z_count

    # Accept scan best if it's within margin of initial reading.
    # On low-contrast substrates, frame-to-frame noise can cause
    # the initial reading to randomly exceed the true peak.
    if not stayed_at_initial:
        initial_margin = 0.95  # 5% margin
        stayed_at_initial = best_sharpness < initial_sharpness * initial_margin

    # Save raw scan best before any stayed_at_initial override
    scan_best_z = best_z
    scan_best_sharpness = best_sharpness

    if stayed_at_initial:
        # Scan found nothing better - stay at initial position
        best_z = initial_z
        best_sharpness = initial_sharpness

    # Move to best Z position (or back to initial if stayed_at_initial)
    z_axis.move_to_corrected(best_z)
    time.sleep(settle_time_s)

    # Restore original Z speed
    z_axis.set_velocity_um_s(original_speed)

    # Capture final sharpness (flush stale sensor buffer first)
    camera.capture()
    final_image = camera.capture()
    final_sharpness = sharpness(final_image, method=sharpness_method)
    stored_final_image = final_image if store_frames else None

    return AutofocusResult(
        selected_z_um=best_z,
        selected_sharpness=best_sharpness,
        scan_best_z_um=scan_best_z,
        scan_best_sharpness=scan_best_sharpness,
        initial_z_um=initial_z,
        initial_sharpness=initial_sharpness,
        final_sharpness=final_sharpness,
        dynamic_range=dynamic_range,
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
        super_fine_z_start_um=actual_super_fine_z_start,
        super_fine_z_end_um=actual_super_fine_z_end,
        stayed_at_initial=stayed_at_initial,
        sharpness_curve=sharpness_curve,
        frames=frames if store_frames else None,
        fine_sharpness_curve=fine_curve if fine_pass and not stayed_at_initial and fine_curve else [],
        fine_frames=fine_frames if store_frames and fine_pass and not stayed_at_initial else None,
        super_fine_sharpness_curve=super_fine_curve,
        super_fine_frames=super_fine_frames_result if store_frames else None,
        initial_image=stored_initial_image,
        final_image=stored_final_image,
    )
