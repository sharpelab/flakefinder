"""Continuous Z-scan autofocus.

Performs autofocus by scanning Z while capturing frames, computing sharpness
for each frame, and returning the Z position with maximum sharpness.

The scan direction is always downward (decreasing Z = away from sample)
for safety. Z range is auto-calculated from the current objective's
working distance if not specified.
"""

import bisect
import enum
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Self, TypedDict

import cv2
import numpy as np

from ..image_utils import sdk_image_to_numpy
from ..types import Point2F, PositionSample, RGBImage
from .microscope import Microscope
from .polling import start_motion_polling
from .units import Axis, Nosepiece, ZDrive

# Working distances in µm by objective position (from commands/stage.py)
# Position 1-indexed as used by the Nosepiece class.
WORKING_DISTANCES_UM: dict[int, float] = {
    1: 12700,  # 5x
    2: 11000,  # 10x
    3: 1900,  # 20x
    4: 380,  # 50x
    5: 210,  # 150x
    6: 15000,  # 2.5x
}


class FCDefaults(NamedTuple):
    """Validated focus-and-capture defaults per objective."""

    z_range_um: float
    z_speed_um_s: float
    exposure_ms: float
    gain: float


# Validated 2026-02-14 on silicon substrate. See docs/microscope_reference.md.
FC_DEFAULTS: dict[int, FCDefaults] = {
    6: FCDefaults(z_range_um=200, z_speed_um_s=1000, exposure_ms=1, gain=1.0),  # 2.5x
    1: FCDefaults(z_range_um=100, z_speed_um_s=500, exposure_ms=1, gain=1.0),  # 5x
    2: FCDefaults(z_range_um=50, z_speed_um_s=250, exposure_ms=1, gain=1.0),  # 10x
    3: FCDefaults(z_range_um=40, z_speed_um_s=75, exposure_ms=1, gain=1.0),  # 20x
    4: FCDefaults(z_range_um=30, z_speed_um_s=25, exposure_ms=2, gain=1.0),  # 50x
}


class AFDefaults(NamedTuple):
    """Per-objective multi-pass autofocus defaults.

    Speeds are max safe coarse speeds from systematic AF speed sweep (2026-02-26,
    90nm substrate). Pass ranges sized so each refinement captures ≥3 points from
    the coarser pass (at ~60 fps). Final pass gives 2-3 points within DOF.
    """

    coarse_range_um: float
    coarse_speed_um_s: float
    fine_range_um: float
    fine_speed_um_s: float
    super_fine_range_um: float | None  # None = skip super-fine pass
    super_fine_speed_um_s: float | None

    def with_overrides(self, **kwargs) -> Self:
        """Return a copy with specified fields replaced."""
        return self._replace(**kwargs)


# Keyed by nosepiece position (1-indexed).
AF_DEFAULTS: dict[int, AFDefaults] = {
    6: AFDefaults(  # 2.5x — two-pass (coarse → fine)
        coarse_range_um=2000,
        coarse_speed_um_s=4000,
        fine_range_um=200,
        fine_speed_um_s=1000,
        super_fine_range_um=None,
        super_fine_speed_um_s=None,
    ),
    1: AFDefaults(  # 5x — two-pass
        coarse_range_um=1000,
        coarse_speed_um_s=2000,
        fine_range_um=100,
        fine_speed_um_s=500,
        super_fine_range_um=None,
        super_fine_speed_um_s=None,
    ),
    2: AFDefaults(  # 10x — three-pass
        coarse_range_um=1000,
        coarse_speed_um_s=1500,
        fine_range_um=75,
        fine_speed_um_s=500,
        super_fine_range_um=50,
        super_fine_speed_um_s=250,
    ),
    3: AFDefaults(  # 20x — three-pass
        coarse_range_um=500,
        coarse_speed_um_s=1250,
        fine_range_um=65,
        fine_speed_um_s=250,
        super_fine_range_um=40,
        super_fine_speed_um_s=75,
    ),
    4: AFDefaults(  # 50x — three-pass
        coarse_range_um=100,
        coarse_speed_um_s=400,
        fine_range_um=20,
        fine_speed_um_s=100,
        super_fine_range_um=30,
        super_fine_speed_um_s=25,
    ),
}


def sharpness_tenengrad(image: RGBImage) -> float:
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


def _sharpness_tenengrad_into(
    image: RGBImage,
    *,
    gray: np.ndarray,
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    mag: np.ndarray,
) -> float:
    """Tenengrad sharpness with pre-allocated buffers (CV_32F, zero alloc per call)."""
    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dst=gray)
    cv2.Sobel(gray, cv2.CV_32F, 1, 0, dst=sobel_x, ksize=5)
    cv2.Sobel(gray, cv2.CV_32F, 0, 1, dst=sobel_y, ksize=5)
    cv2.magnitude(sobel_x, sobel_y, mag)
    return cv2.mean(mag)[0]


def sharpness_laplacian(image: RGBImage) -> float:
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


def sharpness_brenner(image: RGBImage) -> float:
    """Brenner gradient sharpness."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    g = gray.astype(np.float64)
    diff = g[:, 2:] - g[:, :-2]
    return np.mean(diff**2)


def sharpness_normalized_variance(image: RGBImage) -> float:
    """Normalized variance sharpness."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    mu = gray.mean()
    return gray.var() / mu if mu > 0 else 0


def sharpness_vollath_f4(image: RGBImage) -> float:
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


def sharpness(image: RGBImage, method: str = "tenengrad") -> float:
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


class FocusQuality(enum.StrEnum):
    """Three-tier focus quality classification."""

    OK = "ok"
    BORDERLINE = "borderline"
    BAD = "bad"


def af_curve_monotonicity(sharpness_values: list[float]) -> float:
    """Fraction of adjacent diffs that are negative after 5-frame rolling mean smoothing.

    A well-shaped AF curve (rising then falling) has low monotonicity (~0.4-0.5).
    A monotonically decreasing curve (missed focus) has monotonicity ~1.0.

    Returns 0.0 if fewer than 5 frames.
    """
    if len(sharpness_values) < 5:
        return 0.0
    smoothed = np.convolve(sharpness_values, np.ones(5) / 5, mode="valid")
    return float((np.diff(smoothed) < 0).mean())


def classify_focus_quality(best_sharpness_laplacian: float, monotonicity: float) -> FocusQuality:
    """Classify focus quality from image sharpness (Laplacian variance) and AF curve monotonicity.

    Thresholds validated on 608 50x revisit captures across 3 independent runs.
    """
    if best_sharpness_laplacian <= 2.4 and monotonicity > 0.89:
        return FocusQuality.BAD
    if best_sharpness_laplacian <= 2.7 and monotonicity > 0.85:
        return FocusQuality.BORDERLINE
    return FocusQuality.OK


def interpolate_position(t: float, samples: list[PositionSample]) -> float | None:
    """Interpolate position at time t from polled position samples.

    Uses midpoint of t_before/t_after as the effective sample time
    and performs linear interpolation between adjacent samples.

    Args:
        t: Time to interpolate at (from time.perf_counter()).
        samples: List of PositionSample from position polling.

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
    image: np.ndarray


class _SharpnessSampleBase(TypedDict):
    """Required fields for a sharpness measurement."""

    frame: int
    z_um: float
    sharpness: float
    mean_intensity: float


class SharpnessSample(_SharpnessSampleBase, total=False):
    """Single sharpness measurement from a Z scan.

    Optional: metrics (present when store_all_metrics=True).
    """

    metrics: dict[str, float]


class ZScanTiming(NamedTuple):
    """Timing breakdown from _run_z_scan."""

    pre_scan_s: float  # Move to z_start + settle (0 if pre_positioned)
    scan_s: float  # Async Z motion + frame capture
    sharpness_tail_s: float  # Sharpness worker drain after scan ends
    total_s: float  # Wall-clock for entire _run_z_scan call


class ZScanResult(NamedTuple):
    """Result from _run_z_scan."""

    sharpness_curve: list[SharpnessSample]
    frames: list[AutofocusFrame] | None
    timing: ZScanTiming
    frame_count: int
    z_sample_count: int
    best_frame: AutofocusFrame | None = None


class FocusCaptureTiming(NamedTuple):
    position_s: float  # Move to z_start at full speed
    set_speed_s: float  # Set scan velocity
    pre_scan_s: float  # _run_z_scan: move + settle (0 when pre_positioned)
    scan_s: float  # Actual Z motion + frame capture
    sharpness_tail_s: float  # Sharpness worker drain after scan ends
    restore_speed_s: float  # Restore original velocity
    total_s: float  # End-to-end


@dataclass
class FocusCaptureResult:
    """Result from focus_and_capture: best-focus image from a single Z scan."""

    image: np.ndarray
    z_um: float
    sharpness: float
    frame_count: int
    scan_duration_s: float
    z_range_um: float
    objective_position: int
    sharpness_curve: list[SharpnessSample]
    timing: FocusCaptureTiming
    monotonicity: float = 0.0
    best_sharpness_laplacian: float = 0.0
    focus_quality: FocusQuality = FocusQuality.OK


@dataclass
class AutofocusResult:
    """Result from autofocus operation."""

    selected_z_um: float  # Best Z position found across all passes
    selected_sharpness: float  # Sharpness at the selected Z
    scan_best_z_um: float  # Best Z found during scanning (same as selected_z_um)
    scan_best_sharpness: float  # Best sharpness from scanning (same as selected_sharpness)
    initial_z_um: float
    initial_sharpness: float
    final_sharpness: float
    dynamic_range: float  # (s_max - s_min) / s_mean of coarse curve
    mean_intensity: float  # Median of per-frame mean pixel brightness (coarse curve)
    z_range_um: float  # Actual range used
    objective_position: int | None  # Queried from microscope
    scan_duration_s: float
    frame_count: int
    z_sample_count: int
    position_um: Point2F  # Stage XY position where autofocus was performed
    # Diagnostic fields for debugging autofocus issues
    coarse_z_start_um: float = 0.0
    coarse_z_end_um: float = 0.0
    coarse_best_z_um: float = 0.0
    coarse_best_sharpness: float = 0.0
    fine_z_start_um: float | None = None  # None if no fine pass
    fine_z_end_um: float | None = None
    super_fine_z_start_um: float | None = None  # None if no super fine pass
    super_fine_z_end_um: float | None = None
    peak_near_edge: bool = False  # True if coarse best Z is within 10% of scan boundary
    monotonicity: float = 0.0  # AF curve monotonicity (coarse pass)
    best_sharpness_laplacian: float = 0.0  # Laplacian variance of final image
    sharpness_curve: list[SharpnessSample] = field(default_factory=list)  # [{z_um, sharpness}, ...] coarse only
    best_frame: AutofocusFrame | None = None  # Best frame across all passes (if save_best_frame=True)
    frames: list[AutofocusFrame] | None = None  # Coarse frames only (if store_frames=True)
    fine_sharpness_curve: list[SharpnessSample] = field(default_factory=list)  # Fine pass only
    fine_frames: list[AutofocusFrame] | None = None  # Fine frames only (if store_frames=True)
    super_fine_sharpness_curve: list[SharpnessSample] = field(default_factory=list)  # Super fine pass only
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
            "mean_intensity": self.mean_intensity,
            "monotonicity": self.monotonicity,
            "best_sharpness_laplacian": self.best_sharpness_laplacian,
            "peak_near_edge": self.peak_near_edge,
            "position_um": list(self.position_um),
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


def _get_safe_range(nosepiece: Nosepiece, z_range_um: float | None) -> tuple[float, int]:
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
    pre_positioned: bool = False,
    save_best_frame: bool = False,
) -> ZScanResult:
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
        pre_positioned: If True, skip move to z_start and settle (caller already there).
        save_best_frame: If True, populate best_frame with the highest-sharpness frame.

    Returns:
        ZScanResult with sharpness curve, frames, timing, and counts.
    """
    t_func_start = time.perf_counter()
    # Data collection
    frame_data: list[tuple[float, np.ndarray]] = []  # (t_capture, image)

    # Get fast position reading interfaces (prefer hysteresis-corrected for accurate Z during motion)
    z_bcv = getattr(z_axis, "bcv_hysteresis", None) or z_axis.bcv
    z_converter = z_axis.converter

    current_image = [None]

    def on_image(image):
        current_image[0] = image

    # Get delegate class from SDK (available after ExUCAPI.Register())
    from LeicaMicrosystems.HardwareModel import Extensions

    context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

    # Sharpness worker: compute sharpness in background as frames arrive.
    # OpenCV releases the GIL, so this runs in true parallel with SDK Acquire.
    sharpness_q: queue.Queue[np.ndarray | None] = queue.Queue()
    # (sharpness_value, mean_intensity, metrics_dict_or_None) per frame, in capture order
    sharpness_out: list[tuple[float, float, dict | None]] = []
    use_fast_tenengrad = sharpness_method == "tenengrad"

    def sharpness_worker():
        bufs: dict[str, np.ndarray] | None = None
        while True:
            img = sharpness_q.get()
            if img is None:
                break
            # Fast path: tenengrad with pre-allocated CV_32F buffers
            if use_fast_tenengrad:
                if bufs is None:
                    h, w = img.shape[:2]
                    bufs = {
                        "gray": np.empty((h, w), dtype=np.uint8),
                        "sobel_x": np.empty((h, w), dtype=np.float32),
                        "sobel_y": np.empty((h, w), dtype=np.float32),
                        "mag": np.empty((h, w), dtype=np.float32),
                    }
                s = _sharpness_tenengrad_into(img, **bufs)
            else:
                s = sharpness(img, method=sharpness_method)
            mean_int = float(img.mean())
            metrics = (
                {name: float(fn(img)) for name, fn in ALL_SHARPNESS_METRICS.items()} if compute_all_metrics else None
            )
            sharpness_out.append((s, mean_int, metrics))

    worker = threading.Thread(target=sharpness_worker, daemon=True)
    worker.start()

    # Move to scan start position (skip if caller already positioned us)
    if not pre_positioned:
        z_axis.move_to_corrected(z_start)
        time.sleep(0.1)  # Brief settle
    # Dummy acquire to flush any stale camera buffer before scanning
    current_image[0] = None
    acquisition.Acquire(context, None)
    if current_image[0] is not None:
        current_image[0].Dispose()
        current_image[0] = None

    t_pre_scan_done = time.perf_counter()

    # Start Z polling (adaptive Hz: 10 Hz startup, ramps to 100 Hz on motion)
    z_polling = start_motion_polling(z_bcv, z_converter)

    scan_start_time = time.perf_counter()

    # Start async Z move (downward: z_start > z_end)
    z_handle = z_axis.move_to_async(z_end)

    # Capture frames during move, using position to detect arrival
    # (avoids calling GetState on the same axis as polling — see
    # docs/sdk_threading_investigation.md for starvation background)
    _STOP_MARGIN_UM = 0.1
    scan_range_um = z_start - z_end
    scan_speed = z_axis.velocity_um_s or 1000
    scan_timeout_s = max(30.0, (scan_range_um / scan_speed) * 5)

    while True:
        t_capture = time.perf_counter()
        current_image[0] = None
        acquisition.Acquire(context, None)

        if current_image[0] is not None:
            # Convert immediately and dispose .NET object
            img_arr = sdk_image_to_numpy(current_image[0])
            current_image[0].Dispose()
            frame_data.append((t_capture, img_arr))
            sharpness_q.put(img_arr)

            # Stop when Z reaches target (scan is always downward)
            z_now = z_polling.samples[-1].axis_um
            if z_now <= z_end + _STOP_MARGIN_UM:
                break

        # Safety timeout
        if time.perf_counter() - scan_start_time > scan_timeout_s:
            break

    scan_end_time = time.perf_counter()

    # Stop polling to free bus, then wait for SDK move completion
    z_polling.join()
    if not z_handle.is_complete:
        z_handle.wait(timeout=2.0)
    z_handle.dispose()

    scan_duration = scan_end_time - scan_start_time

    # Wait for sharpness worker to drain remaining frames
    sharpness_q.put(None)
    worker.join()
    t_sharpness_done = time.perf_counter()

    # Assemble results (Z interpolation is cheap, sharpness already computed)
    z_samples = z_polling.samples
    sharpness_curve = []
    frames = [] if store_frames else None
    best_frame: AutofocusFrame | None = None
    best_sharpness = -1.0

    for i, (t_capture, img) in enumerate(frame_data):
        z_interp = interpolate_position(t_capture, z_samples)
        if z_interp is None:
            raise RuntimeError(f"Frame {i}: Z interpolation failed at t={t_capture:.6f}s")
        s, mean_int, metrics = sharpness_out[i]
        entry = SharpnessSample(
            frame=i,
            z_um=z_interp,
            sharpness=s,
            mean_intensity=mean_int,
        )
        if metrics is not None:
            entry["metrics"] = metrics
        sharpness_curve.append(entry)
        if store_frames:
            assert frames is not None
            frames.append(AutofocusFrame(z_um=z_interp, sharpness=s, image=img))
        if save_best_frame and s > best_sharpness:
            best_sharpness = s
            best_frame = AutofocusFrame(z_um=z_interp, sharpness=s, image=img)

    t_func_end = time.perf_counter()

    return ZScanResult(
        sharpness_curve=sharpness_curve,
        frames=frames,
        timing=ZScanTiming(
            pre_scan_s=t_pre_scan_done - t_func_start,
            scan_s=scan_duration,
            sharpness_tail_s=t_sharpness_done - scan_end_time,
            total_s=t_func_end - t_func_start,
        ),
        frame_count=len(frame_data),
        z_sample_count=len(z_samples),
        best_frame=best_frame,
    )


def focus_and_capture(
    scope: "Microscope",
    *,
    z_center_um: float | None = None,
    z_range_um: float | None = None,
    z_speed_um_s: float | None = None,
    exposure_ms: float | None = None,
    gain: float | None = None,
    sharpness_method: str = "tenengrad",
) -> FocusCaptureResult:
    """Scan Z range and return the sharpest frame.

    Wrapper around _focus_and_capture_impl that applies per-objective defaults
    from FC_DEFAULTS for any parameter left as None. Also sets camera exposure
    and gain when resolved.

    Args:
        scope: Microscope facade instance.
        z_center_um: Center of Z scan range in µm. None = current position.
        z_range_um: Z scan range in µm. None = use objective default.
        z_speed_um_s: Z scan speed in µm/s. None = use objective default.
        exposure_ms: Camera exposure in ms. None = use objective default.
        gain: Camera gain. None = use objective default.
        sharpness_method: Sharpness metric to use.

    Returns:
        FocusCaptureResult with the sharpest frame's image and metadata.
    """
    defaults = FC_DEFAULTS.get(scope.nosepiece.position)
    if defaults is not None:
        if z_range_um is None:
            z_range_um = defaults.z_range_um
        if z_speed_um_s is None:
            z_speed_um_s = defaults.z_speed_um_s
        if exposure_ms is None:
            exposure_ms = defaults.exposure_ms
        if gain is None:
            gain = defaults.gain

    if exposure_ms is not None:
        scope.camera.exposure_time = exposure_ms / 1000.0
    if gain is not None:
        scope.camera.gain = gain

    return _focus_and_capture_impl(
        scope,
        z_center_um=z_center_um,
        z_range_um=z_range_um,
        z_speed_um_s=z_speed_um_s,
        sharpness_method=sharpness_method,
    )


def _focus_and_capture_impl(
    scope: "Microscope",
    *,
    z_center_um: float | None = None,
    z_range_um: float | None = None,
    z_speed_um_s: float | None = None,
    sharpness_method: str = "tenengrad",
) -> FocusCaptureResult:
    """Single-pass Z scan returning the sharpest frame.

    Low-level implementation — no default resolution. Callers should use
    focus_and_capture() which applies per-objective defaults.
    """
    t_start = time.perf_counter()

    z_axis = scope.z
    original_speed = z_axis.velocity_um_s

    safe_range, objective_position = _get_safe_range(scope.nosepiece, z_range_um)

    center_z = z_center_um if z_center_um is not None else z_axis.position_um
    z_start = center_z + safe_range / 2
    z_end = center_z - safe_range / 2

    _validate_z_limits(z_axis, z_start, z_end, None)

    # Position at z_start at full speed, then set scan speed
    z_axis.move_to_corrected(z_start)
    t_positioned = time.perf_counter()

    if z_speed_um_s is not None:
        z_axis.set_velocity_um_s(z_speed_um_s)
    t_speed_set = time.perf_counter()

    zsr = _run_z_scan(
        z_axis=z_axis,
        z_start=z_start,
        z_end=z_end,
        acquisition=scope.acquisition,
        context=scope.context,
        store_frames=False,
        save_best_frame=True,
        sharpness_method=sharpness_method,
        pre_positioned=True,
    )

    t_restore_start = time.perf_counter()
    z_axis.set_velocity_um_s(original_speed)
    t_end = time.perf_counter()

    assert zsr.best_frame is not None, "save_best_frame=True but no frames captured"
    best = zsr.best_frame

    # Focus quality metrics
    mono = af_curve_monotonicity([s["sharpness"] for s in zsr.sharpness_curve])
    img_sharp_lap = sharpness_laplacian(best.image)
    fq = classify_focus_quality(img_sharp_lap, mono)

    return FocusCaptureResult(
        image=best.image,
        z_um=best.z_um,
        sharpness=best.sharpness,
        frame_count=zsr.frame_count,
        scan_duration_s=zsr.timing.scan_s,
        z_range_um=safe_range,
        objective_position=objective_position,
        sharpness_curve=zsr.sharpness_curve,
        timing=FocusCaptureTiming(
            position_s=t_positioned - t_start,
            set_speed_s=t_speed_set - t_positioned,
            pre_scan_s=zsr.timing.pre_scan_s,
            scan_s=zsr.timing.scan_s,
            sharpness_tail_s=zsr.timing.sharpness_tail_s,
            restore_speed_s=t_end - t_restore_start,
            total_s=t_end - t_start,
        ),
        monotonicity=mono,
        best_sharpness_laplacian=img_sharp_lap,
        focus_quality=fq,
    )


def continuous_autofocus(
    scope: "Microscope",
    *,
    af_defaults: AFDefaults | None = None,
    z_start_um: float | None = None,
    z_max_safe_um: float | None = None,
    sharpness_method: str = "tenengrad",
    store_frames: bool = False,
    save_best_frame: bool = False,
    compute_all_metrics: bool = False,
    settle_time_s: float = 0.2,
    move_to_best_z: bool = True,
) -> AutofocusResult:
    """Perform continuous Z-scan autofocus.

    Multi-pass Z scan: coarse pass finds approximate focus, fine pass refines,
    optional super-fine pass (when defined in defaults) achieves DOF-level
    precision. Pass structure and speeds are controlled by af_defaults.

    Args:
        scope: Microscope facade instance.
        af_defaults: Pass parameters (speeds, ranges). None = look up from
            AF_DEFAULTS for the current objective. Use
            ``AF_DEFAULTS[pos].with_overrides(coarse_range_um=...)`` to tweak.
        z_start_um: Center of coarse scan in µm. None = current Z position.
        z_max_safe_um: Hard upper Z limit. Raises if z_start exceeds this.
        sharpness_method: "tenengrad" (default) or "laplacian".
        store_frames: If True, store images in result.frames for debugging.
        save_best_frame: If True, populate best_frame with the highest-sharpness
            frame across all passes.
        compute_all_metrics: If True, compute all 5 sharpness metrics per frame.
        settle_time_s: Settle time after final Z move (default 0.2s).
        move_to_best_z: Move to best Z after scan (default True).

    Returns:
        AutofocusResult with best Z, sharpness curve, and scan statistics.

    Raises:
        ValueError: If Z range/position exceeds safety limits.
    """
    z_axis = scope.z
    camera = scope.camera
    acquisition = scope.acquisition
    context = scope.context

    current_z = z_axis.position_um
    original_speed = z_axis.velocity_um_s
    objective_position = scope.nosepiece.position

    # Resolve AF defaults: explicit > per-objective table > safety fallback
    af = af_defaults or AF_DEFAULTS.get(objective_position)
    if af is None:
        working_dist = WORKING_DISTANCES_UM.get(objective_position, 1000)
        safe_range = min(working_dist / 3, 500)
        af = AFDefaults(
            coarse_range_um=safe_range,
            coarse_speed_um_s=original_speed,
            fine_range_um=50.0,
            fine_speed_um_s=original_speed * 0.25,
            super_fine_range_um=None,
            super_fine_speed_um_s=None,
        )

    # Validate coarse range against working distance
    working_dist = WORKING_DISTANCES_UM.get(objective_position, 1000)
    if af.coarse_range_um > working_dist / 2:
        raise ValueError(
            f"Coarse range {af.coarse_range_um}µm exceeds safe limit for "
            f"objective position {objective_position} (working distance: {working_dist}µm)"
        )

    coarse_speed = af.coarse_speed_um_s
    do_fine = True  # always do fine pass
    do_super_fine = af.super_fine_speed_um_s is not None

    z_axis.set_velocity_um_s(coarse_speed)

    # Calculate scan bounds (centered on current/specified position, scan downward)
    initial_z = z_start_um if z_start_um is not None else current_z
    z_start = initial_z + af.coarse_range_um / 2  # Start high
    z_end = initial_z - af.coarse_range_um / 2  # End low (away from sample)

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
    zsr = _run_z_scan(
        z_axis=z_axis,
        z_start=z_start,
        z_end=z_end,
        acquisition=acquisition,
        context=context,
        store_frames=store_frames,
        sharpness_method=sharpness_method,
        compute_all_metrics=compute_all_metrics,
        save_best_frame=save_best_frame,
    )
    sharpness_curve = zsr.sharpness_curve
    frames = zsr.frames
    overall_best_frame = zsr.best_frame
    scan_duration = zsr.timing.scan_s
    frame_count = zsr.frame_count
    z_sample_count = zsr.z_sample_count

    if not sharpness_curve:
        raise ValueError("No frames captured during autofocus scan")

    # Find best frame from coarse pass
    best = max(sharpness_curve, key=lambda r: r["sharpness"])
    best_z = best["z_um"]
    best_sharpness = best["sharpness"]

    # Compute quality metrics from coarse curve
    sharpness_values = [r["sharpness"] for r in sharpness_curve]
    s_min, s_max = min(sharpness_values), max(sharpness_values)
    s_mean = sum(sharpness_values) / len(sharpness_values)
    dynamic_range = (s_max - s_min) / s_mean if s_mean > 0 else 0
    mean_intensity = float(np.median([r["mean_intensity"] for r in sharpness_curve]))

    # Track coarse results for diagnostics
    coarse_best_z = best_z
    coarse_best_sharpness = best_sharpness
    actual_fine_z_start = None
    actual_fine_z_end = None

    # Detect if coarse peak is near scan boundary (truncated curve)
    edge_margin = af.coarse_range_um * 0.10
    peak_near_edge = (best_z >= z_start - edge_margin) or (best_z <= z_end + edge_margin)

    # Fine pass
    fine_curve: list[SharpnessSample] = []
    fine_frames = None

    if do_fine:
        fine_z_start = best_z + af.fine_range_um / 2
        fine_z_end = best_z - af.fine_range_um / 2

        # Clamp to axis limits
        fine_z_start = min(fine_z_start, z_axis.max_um)
        fine_z_end = max(fine_z_end, z_axis.min_um)

        actual_fine_z_start = fine_z_start
        actual_fine_z_end = fine_z_end

        # Position at coarse speed, then set fine scan speed
        z_axis.move_to_corrected(fine_z_start)
        z_axis.set_velocity_um_s(af.fine_speed_um_s)

        fine_zsr = _run_z_scan(
            z_axis=z_axis,
            z_start=fine_z_start,
            z_end=fine_z_end,
            acquisition=acquisition,
            context=context,
            store_frames=store_frames,
            sharpness_method=sharpness_method,
            compute_all_metrics=compute_all_metrics,
            save_best_frame=save_best_frame,
        )
        fine_curve = fine_zsr.sharpness_curve
        fine_frames = fine_zsr.frames

        # Restore coarse speed for positioning
        z_axis.set_velocity_um_s(coarse_speed)

        if fine_curve:
            fine_best = max(fine_curve, key=lambda r: r["sharpness"])
            if fine_best["sharpness"] > best_sharpness:
                best_z = fine_best["z_um"]
                best_sharpness = fine_best["sharpness"]
                overall_best_frame = fine_zsr.best_frame

            scan_duration += fine_zsr.timing.scan_s
            frame_count += fine_zsr.frame_count
            z_sample_count += fine_zsr.z_sample_count

    # Super-fine pass (only for three-pass objectives)
    super_fine_curve: list[SharpnessSample] = []
    super_fine_frames_result = None
    actual_super_fine_z_start = None
    actual_super_fine_z_end = None

    if do_super_fine:
        assert af.super_fine_range_um is not None
        assert af.super_fine_speed_um_s is not None
        sf_z_start = best_z + af.super_fine_range_um / 2
        sf_z_end = best_z - af.super_fine_range_um / 2

        # Clamp to axis limits
        sf_z_start = min(sf_z_start, z_axis.max_um)
        sf_z_end = max(sf_z_end, z_axis.min_um)

        actual_super_fine_z_start = sf_z_start
        actual_super_fine_z_end = sf_z_end

        # Position at coarse speed, then set super-fine scan speed
        z_axis.move_to_corrected(sf_z_start)
        z_axis.set_velocity_um_s(af.super_fine_speed_um_s)

        sf_zsr = _run_z_scan(
            z_axis=z_axis,
            z_start=sf_z_start,
            z_end=sf_z_end,
            acquisition=acquisition,
            context=context,
            store_frames=store_frames,
            sharpness_method=sharpness_method,
            compute_all_metrics=compute_all_metrics,
            save_best_frame=save_best_frame,
        )

        if sf_zsr.sharpness_curve:
            super_fine_curve = sf_zsr.sharpness_curve
            super_fine_frames_result = sf_zsr.frames
            sf_best = max(sf_zsr.sharpness_curve, key=lambda r: r["sharpness"])
            if sf_best["sharpness"] > best_sharpness:
                best_z = sf_best["z_um"]
                best_sharpness = sf_best["sharpness"]
                overall_best_frame = sf_zsr.best_frame

            scan_duration += sf_zsr.timing.scan_s
            frame_count += sf_zsr.frame_count
            z_sample_count += sf_zsr.z_sample_count

    # Restore original Z speed
    z_axis.set_velocity_um_s(original_speed)

    # Focus quality metrics (monotonicity from coarse curve only)
    mono = af_curve_monotonicity(sharpness_values)

    if move_to_best_z:
        # Move to best Z position
        z_axis.move_to_corrected(best_z)
        time.sleep(settle_time_s)

        # Capture final sharpness (flush stale sensor buffer first)
        camera.capture()
        final_image = camera.capture()
        final_sharpness = sharpness(final_image, method=sharpness_method)
        img_sharp_lap = sharpness_laplacian(final_image)
        stored_final_image = final_image if store_frames else None
    else:
        final_sharpness = best_sharpness
        img_sharp_lap = 0.0
        stored_final_image = None

    return AutofocusResult(
        selected_z_um=best_z,
        selected_sharpness=best_sharpness,
        scan_best_z_um=best_z,
        scan_best_sharpness=best_sharpness,
        initial_z_um=initial_z,
        initial_sharpness=initial_sharpness,
        final_sharpness=final_sharpness,
        dynamic_range=dynamic_range,
        mean_intensity=mean_intensity,
        z_range_um=af.coarse_range_um,
        objective_position=objective_position,
        scan_duration_s=scan_duration,
        frame_count=frame_count,
        z_sample_count=z_sample_count,
        position_um=Point2F(*scope.stage.position_um),
        coarse_z_start_um=z_start,
        coarse_z_end_um=z_end,
        coarse_best_z_um=coarse_best_z,
        coarse_best_sharpness=coarse_best_sharpness,
        fine_z_start_um=actual_fine_z_start,
        fine_z_end_um=actual_fine_z_end,
        super_fine_z_start_um=actual_super_fine_z_start,
        super_fine_z_end_um=actual_super_fine_z_end,
        peak_near_edge=peak_near_edge,
        monotonicity=mono,
        best_sharpness_laplacian=img_sharp_lap,
        sharpness_curve=sharpness_curve,
        best_frame=overall_best_frame,
        frames=frames if store_frames else None,
        fine_sharpness_curve=fine_curve,
        fine_frames=fine_frames if store_frames else None,
        super_fine_sharpness_curve=super_fine_curve,
        super_fine_frames=super_fine_frames_result if store_frames else None,
        initial_image=stored_initial_image,
        final_image=stored_final_image,
    )
