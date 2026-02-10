"""Single-row scan with active Z focus tracking.

NOTE: Unmaintained. Superseded by scan_area_v1.py + focus_map.py workflow.
Kept for reference but not updated with new conventions (e.g. parse_white_balance,
Aperture, cli_utils).

Combines:
- Camera capture loop from scan_area_v1.py
- State feedback + feedforward Z controller from test_z_curve_tracking_v2.py

The controller tracks a surface defined by plane fit + residual spline from
autofocus measurements, adjusting Z in real-time during X motion.

SAFETY: Pre-flight Z range validation, runtime Z limit check, emergency halt.

Run on microscope PC.

Usage:
    python scan_area_with_focus.py -o test_scan --area-rect 12000,35000,8175,8175 --speed-mm 5
    python scan_area_with_focus.py -o test_scan --area-rect 12000,35000,8175,8175 --speed-mm 10 --objective-mag 20x
"""

import argparse
import bisect
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime

import numpy as np
from scipy.interpolate import CubicSpline

# ============================================================================
# Surface Data (from autofocus measurements)
# ============================================================================

# Residual points from autofocus grid - deviation from plane fit
SURFACE_DATA = {
    # Top row (Y=8175 µm) - sharpest features
    "top": {
        "y_um": 8175,
        "points": [
            (12112, -4.6),
            (16612, +6.0),
            (21112, +2.0),
            (25612, -4.6),
            (30112, +4.3),
            (34612, -2.4),
        ],
    },
    # Middle row (Y=12675 µm) - gentler
    "middle": {
        "y_um": 12675,
        "points": [
            (12112, -3.5),
            (16612, -0.7),
            (21112, +4.7),
            (25612, -2.0),
            (30112, +1.7),
            (34612, -4.9),
        ],
    },
}

# Plane fit from autofocus data: Z = a*X_mm + b*Y_mm + c
PLANE_FIT = {
    "a_um_per_mm": 1.4722,  # X slope
    "b_um_per_mm": -0.6527,  # Y slope
    "c_um": 24666.85,  # Intercept
}


def build_z_profile(surface_name: str, x_start_um: float, x_end_um: float):
    """Build Z profile function from surface data.

    Returns:
        (z_func, dzdx_func, info_dict) where:
        - z_func(x_um) returns desired Z in µm
        - dzdx_func(x_um) returns dZ/dX in µm/µm
        - info_dict contains metadata about the profile
    """
    surface = SURFACE_DATA[surface_name]
    y_um = surface["y_um"]
    points = surface["points"]

    # Build cubic spline for residuals
    x_points = np.array([p[0] for p in points])
    residuals = np.array([p[1] for p in points])
    spline = CubicSpline(x_points, residuals, extrapolate=True)
    spline_deriv = spline.derivative()

    # Plane contribution
    a = PLANE_FIT["a_um_per_mm"]
    b = PLANE_FIT["b_um_per_mm"]
    c = PLANE_FIT["c_um"]
    y_mm = y_um / 1000

    def z_total(x_um: float) -> float:
        """Total Z = plane + residual."""
        x_mm = x_um / 1000
        z_plane = a * x_mm + b * y_mm + c
        z_residual = float(spline(x_um))
        return z_plane + z_residual

    def dzdx_total(x_um: float) -> float:
        """Total dZ/dX = plane_slope + residual_slope (in µm/µm)."""
        plane_slope = a / 1000  # Convert µm/mm to µm/µm
        return plane_slope + float(spline_deriv(x_um))

    # Calculate expected stats
    x_test = np.linspace(x_start_um, x_end_um, 1000)
    z_test = np.array([z_total(x) for x in x_test])
    dzdx_test = np.array([dzdx_total(x) for x in x_test])

    info = {
        "surface_name": surface_name,
        "y_um": y_um,
        "x_range_um": (x_start_um, x_end_um),
        "z_range_um": (float(z_test.min()), float(z_test.max())),
        "z_delta_um": float(z_test[-1] - z_test[0]),
        "max_gradient_um_per_mm": float(np.max(np.abs(dzdx_test)) * 1000),
        "plane_slope_um_per_mm": a,
    }

    return z_total, dzdx_total, info


# ============================================================================
# Helpers from scan_area_v1.py
# ============================================================================


def interpolate_position(t, samples):
    """Interpolate position at time t from (t_before, t_after, x_um) samples."""
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


def interpolate_z_position(t, z_samples):
    """Interpolate Z position at time t from (t, z_um) samples."""
    if not z_samples:
        return None

    times = [s[0] for s in z_samples]
    idx = bisect.bisect_left(times, t)

    if idx == 0:
        return z_samples[0][1]
    if idx >= len(z_samples):
        return z_samples[-1][1]

    t0, z0 = times[idx - 1], z_samples[idx - 1][1]
    t1, z1 = times[idx], z_samples[idx][1]

    if t1 == t0:
        return z0

    alpha = (t - t0) / (t1 - t0)
    return z0 + alpha * (z1 - z0)


def parse_area_rect(value: str) -> tuple[float, float, float, float]:
    """Parse area rectangle from comma-separated string."""
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("--area-rect must be x_min,x_max,y_min,y_max (4 values)")
    try:
        x1, x2, y1, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError as e:
        raise ValueError("--area-rect values must be numbers") from e

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    if x_min == x_max:
        raise ValueError(f"--area-rect: x_min and x_max cannot be equal ({x_min})")

    return (x_min, x_max, y_min, y_max)


# ============================================================================
# Main Scan
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Single-row scan with active Z focus tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan top row at 5 mm/s with focus tracking
  python scan_area_with_focus.py -o scan_focus --area-rect 12000,35000,8175,8175 --speed-mm 5

  # 20x scan with faster speed
  python scan_area_with_focus.py -o scan_20x --area-rect 12000,35000,8175,8175 --objective-mag 20x --speed-mm 10

  # Objective by turret position
  python scan_area_with_focus.py -o scan_20x --area-rect 12000,35000,8175,8175 --objective-pos 3 --speed-mm 10
""",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    # Scan area
    area_group = parser.add_argument_group("Scan area")
    area_group.add_argument(
        "--area-rect",
        type=str,
        required=True,
        metavar="X1,X2,Y1,Y2",
        help="Scan area as x_min,x_max,y_min,y_max in µm (Y values should match for single row)",
    )
    area_group.add_argument(
        "--surface", choices=["top", "middle"], default="top", help="Surface profile to track (default: top)"
    )

    # Optics
    optics_group = parser.add_argument_group("Optics")
    optics_group.add_argument(
        "--objective-mag",
        type=str,
        metavar="MAG",
        help="Objective by magnification (e.g., 5, 5x, 20, 2.5) - switches before scan",
    )
    optics_group.add_argument(
        "--objective-pos", type=int, metavar="POS", help="Objective by turret position (1-6) - switches before scan"
    )

    # Motion
    motion_group = parser.add_argument_group("Motion")
    motion_group.add_argument("--speed-mm", type=float, default=5.0, help="Scan speed in mm/s (default: 5)")
    motion_group.add_argument(
        "--move-speed-mm", type=float, default=40, help="Move speed for positioning (default: 40)"
    )

    # Camera
    frame_group = parser.add_argument_group("Camera")
    frame_group.add_argument("--exposure-ms", type=float, default=0.25, help="Exposure time in ms (default: 0.25)")
    frame_group.add_argument("--gain", type=float, default=4.0, help="Camera gain (default: 4.0)")
    frame_group.add_argument(
        "--binning", type=int, default=3, choices=[1, 2, 3], help="Camera binning NxN (default: 3)"
    )
    frame_group.add_argument("--white-balance", type=str, default="2.51,1.02,1.41", help="White balance as B,G,R gains")
    frame_group.add_argument("--gamma", type=float, default=1.0, help="Gamma (default: 1.0)")
    frame_group.add_argument("--downsample", type=int, default=1, help="Downsample factor")
    frame_group.add_argument("--warmup-frames", type=int, default=3, help="Warmup captures before scan (default: 3)")

    # Z Controller
    ctrl_group = parser.add_argument_group("Z Controller")
    ctrl_group.add_argument("--kp", type=float, default=3.0, help="Position error gain (default: 3.0)")
    ctrl_group.add_argument("--kv", type=float, default=0.3, help="Velocity error gain (default: 0.3)")
    ctrl_group.add_argument("--kff", type=float, default=1.0, help="Feedforward gain (default: 1.0)")
    ctrl_group.add_argument(
        "--lookahead-ms", type=float, default=50.0, help="Feedforward lookahead in ms (default: 50)"
    )
    ctrl_group.add_argument(
        "--control-rate", type=float, default=50.0, help="Z controller update rate in Hz (default: 50)"
    )

    # Safety
    safety_group = parser.add_argument_group("Safety")
    safety_group.add_argument("--z-max", type=float, default=26000.0, help="Hard Z limit in µm (default: 26000)")
    safety_group.add_argument("--z-margin", type=float, default=500.0, help="Safety margin below z-max (default: 500)")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--clean", action="store_true", help="Wipe output directory if exists")
    output_group.add_argument("--write-threads", type=int, default=2, help="Image writer threads")
    output_group.add_argument("--compress", action="store_true", help="Create .zip of output")

    args = parser.parse_args()

    # Validate objective args are mutually exclusive
    if args.objective_mag is not None and args.objective_pos is not None:
        print("Error: --objective-mag and --objective-pos are mutually exclusive")
        return 1

    # Parse white balance
    wb_parts = args.white_balance.split(",")
    if len(wb_parts) != 3:
        print("Error: --white-balance must be 3 comma-separated values (B,G,R)")
        return 1
    try:
        wb_blue, wb_green, wb_red = float(wb_parts[0]), float(wb_parts[1]), float(wb_parts[2])
    except ValueError:
        print("Error: --white-balance values must be numbers")
        return 1

    # Parse area
    try:
        x_min, x_max, y_min, y_max = parse_area_rect(args.area_rect)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # For single-row scan, use the Y center
    row_y = (y_min + y_max) / 2

    # Find closest surface profile by Y
    surface_name = args.surface
    surface = SURFACE_DATA[surface_name]
    surface_y = surface["y_um"]

    # Warn if Y doesn't match surface
    if abs(row_y - surface_y) > 1000:
        print(f"Warning: Row Y ({row_y:.0f}) doesn't match surface '{surface_name}' Y ({surface_y:.0f})")
        print("  Consider using a surface with closer Y, or adjust --area-rect")

    # Build Z profile
    z_func, dzdx_func, profile_info = build_z_profile(surface_name, x_min, x_max)

    # Pre-flight Z validation
    x_test = np.linspace(x_min, x_max, 100)
    z_test = np.array([z_func(x) for x in x_test])
    z_min_expected = float(z_test.min())
    z_max_expected = float(z_test.max())

    z_limit = args.z_max - args.z_margin
    if z_max_expected > z_limit:
        print(f"ABORT: Expected max Z ({z_max_expected:.0f}) exceeds limit ({z_limit:.0f})")
        return 1

    # Binning index
    binning_idx = args.binning - 1

    # Check/create output directory
    if os.path.exists(args.output):
        if args.clean:
            shutil.rmtree(args.output)
        else:
            print(f"Error: Output directory '{args.output}' exists. Use --clean to wipe.")
            return 1
    os.makedirs(args.output)

    # Convert speed
    x_speed_um_s = args.speed_mm * 1000
    x_distance_um = x_max - x_min
    expected_duration_s = x_distance_um / x_speed_um_s
    max_z_vel_needed = profile_info["max_gradient_um_per_mm"] * args.speed_mm

    print("Scan with Focus Tracking")
    print("=" * 60)
    print(f"X range: {x_min:.0f} -> {x_max:.0f} µm ({x_distance_um / 1000:.1f} mm)")
    print(f"Row Y: {row_y:.0f} µm (surface '{surface_name}' Y={surface_y:.0f})")
    print(f"X speed: {args.speed_mm:.1f} mm/s")
    print(f"Expected duration: {expected_duration_s:.2f} s")
    print()
    print("Z profile:")
    print(f"  Expected Z range: {z_min_expected:.0f} - {z_max_expected:.0f} µm")
    print(f"  Max gradient: {profile_info['max_gradient_um_per_mm']:.2f} µm/mm")
    print(f"  Max Z velocity: {max_z_vel_needed:.1f} µm/s")
    print(f"  Z limit: {z_limit:.0f} µm (margin {args.z_margin:.0f})")
    print()
    print(f"Controller: Kp={args.kp}, Kv={args.kv}, Kff={args.kff}, lookahead={args.lookahead_ms}ms")
    print()

    from PIL import Image as PILImage

    from flakefinder.image_utils import sdk_image_to_numpy
    from flakefinder.leica import Microscope

    with Microscope() as scope:
        stage = scope.stage
        z_drive = scope.z

        # Fast position readers
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter

        # Check directed velocity support
        if not stage.x.supports_directed_velocity:
            print("Error: X axis does not support directed velocity")
            return 1
        if not z_drive.supports_directed_velocity:
            print("Error: Z axis does not support directed velocity")
            return 1

        # Switch objective if requested
        if args.objective_mag is not None:
            if scope.switch_objective_mag(args.objective_mag):
                print(f"Switched objective to {scope.objective_mag}x")
            else:
                print(f"Objective: already at {scope.objective_mag}x")
        elif args.objective_pos is not None:
            if scope.switch_objective_pos(args.objective_pos):
                print(f"Switched objective to position {args.objective_pos} ({scope.objective_mag}x)")
            else:
                print(f"Objective: already at {scope.objective_mag}x")

        objective_mag = scope.objective_mag
        objective_idx = scope.nosepiece.position

        # Lighting and camera
        scope.light_on()
        camera = scope.camera
        acquisition = scope.acquisition

        # Configure camera
        camera.trigger_mode = 0  # CONTINUOUS
        camera.binning = binning_idx
        camera.exposure_time = args.exposure_ms / 1000.0
        camera.gain = args.gain
        camera.gain_rgb = (wb_red, wb_green, wb_blue)
        camera.gamma = args.gamma

        # Read camera properties
        frame_width_px, frame_height_px = camera.frame_size_px
        sensor_width_px, sensor_height_px = camera.sensor_size_px
        pixel_size_x_um, pixel_size_y_um = camera.pixel_size_um
        physical_pixel_x_um, physical_pixel_y_um = camera.physical_pixel_size_um
        readout_time = camera.readout_time_s
        actual_exposure = camera.exposure_time
        actual_binning_idx = camera.binning
        binning_map = {0: 1, 1: 2, 2: 3}
        actual_binning = binning_map.get(actual_binning_idx, actual_binning_idx)

        # Compute frame size in µm
        from flakefinder.data_utils import compute_frame_size_um, require_microscope_description

        desc = require_microscope_description()
        scope.validate_description(desc)
        actual_frame_size = compute_frame_size_um(desc.camera, objective_mag, actual_binning_idx)
        if actual_frame_size is None:
            print("Error: Could not determine frame size. Check objective/camera.")
            return 1
        frame_width_um, frame_height_um = actual_frame_size
        sample_pixel_x_um = frame_width_um / frame_width_px if frame_width_px else None
        sample_pixel_y_um = frame_height_um / frame_height_px if frame_height_px else None

        print(f"Camera: {camera.name}")
        exp_str = f"{actual_exposure * 1000:.2f}ms" if actual_exposure else "?"
        print(f"  Binning: {actual_binning}x{actual_binning}, Exposure: {exp_str}, Gain: {args.gain}")
        print(f"  Frame: {frame_width_px}x{frame_height_px} px")
        print(f"  FOV: {frame_width_um:.1f} x {frame_height_um:.1f} µm")
        print(f"  Objective: {objective_mag}x")
        print()

        # Set up image acquisition context
        from LeicaMicrosystems.HardwareModel import Extensions

        context = scope.context
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # Set up saver threads
        save_queue = queue.Queue()
        saved_frames_meta = []

        def saver_thread():
            while True:
                item = save_queue.get()
                if item is None:
                    break

                frame_idx, t_start, t_end, image, x_samples, z_samples, t0 = item

                # Convert to numpy
                arr = sdk_image_to_numpy(image)
                image.Dispose()

                img = PILImage.fromarray(arr)
                if args.downsample > 1:
                    new_size = (img.width // args.downsample, img.height // args.downsample)
                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                path = os.path.join(args.output, f"frame_{frame_idx:04d}.jpg")
                img.save(path, quality=95)

                # Compute metadata
                x_start_interp = interpolate_position(t_start, x_samples)
                x_end_interp = interpolate_position(t_end, x_samples)
                z_interp = interpolate_z_position(t_start, z_samples)

                # Compute ideal Z and error
                z_ideal = z_func(x_start_interp) if x_start_interp else None
                z_error = (z_interp - z_ideal) if (z_interp and z_ideal) else None

                dt = t_end - t_start
                x_vel = (x_end_interp - x_start_interp) / dt if dt > 0 and x_start_interp and x_end_interp else 0

                saved_frames_meta.append(
                    {
                        "n": frame_idx,
                        "row": 0,  # Single row scan
                        "t_start": t_start - t0,
                        "t_end": t_end - t0,
                        "x_start": x_start_interp,
                        "x_end": x_end_interp,
                        "x_vel": x_vel,
                        "y_um": row_y,
                        "z_actual": z_interp,
                        "z_ideal": z_ideal,
                        "z_error": z_error,
                    }
                )

                save_queue.task_done()

        savers = []
        for _ in range(args.write_threads):
            t = threading.Thread(target=saver_thread, daemon=True)
            t.start()
            savers.append(t)

        # Save initial positions
        initial_x = stage.x.position_um
        initial_y = stage.y.position_um
        initial_z = z_drive.position_um

        print(f"Initial position: X={initial_x:.0f}, Y={initial_y:.0f}, Z={initial_z:.0f} µm")

        # Threading state
        emergency_stop = threading.Event()
        emergency_reason = [None]
        stop_control = threading.Event()
        stop_polling = threading.Event()

        # Sampling storage
        x_samples = []  # [(t_before, t_after, x_um), ...]
        z_samples = []  # [(t, z_um), ...]
        control_log = []  # Control loop debug info

        try:
            # Move to start position
            z_start = z_func(x_min)
            print(f"\nMoving to start: X={x_min:.0f}, Y={row_y:.0f}, Z={z_start:.0f} µm...")

            # Set move speed
            stage.x.set_velocity_um_s(args.move_speed_mm * 1000)
            stage.y.set_velocity_um_s(args.move_speed_mm * 1000)

            # Move Y first, then X, then Z
            stage.y.move_to(row_y)
            stage.x.move_to(x_min)
            z_drive.move_to(z_start)
            time.sleep(0.2)

            actual_x_start = stage.x.position_um
            actual_z_start = z_drive.position_um
            print(f"At start: X={actual_x_start:.0f}, Z={actual_z_start:.0f} µm")

            # Define polling threads
            def poll_x():
                """Poll X position continuously."""
                while not stop_polling.is_set():
                    t_before = time.perf_counter()
                    x_native = x_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    x_um = x_converter.GetMetricsValue(x_native)
                    x_samples.append((t_before, t_after, x_um))

            def poll_z():
                """Poll Z position continuously."""
                while not stop_polling.is_set():
                    t = time.perf_counter()
                    z_native = z_bcv.GetControlValue()
                    z_um = z_converter.GetMetricsValue(z_native)
                    z_samples.append((t, z_um))

            def get_latest_x() -> tuple[float, float]:
                """Get latest X position and timestamp from polling thread."""
                if x_samples:
                    s = x_samples[-1]
                    return ((s[0] + s[1]) / 2, s[2])  # (t_mid, x_um)
                return (time.perf_counter(), x_min)

            def get_latest_z() -> tuple[float, float]:
                """Get latest Z position and timestamp from polling thread."""
                if z_samples:
                    return z_samples[-1]  # (t, z_um)
                return (time.perf_counter(), z_start)

            # Z control thread (reads from polling threads, no direct SDK calls)
            def control_z():
                """State feedback + feedforward Z velocity control."""
                control_interval = 1.0 / args.control_rate

                last_z = z_start
                last_t_z = time.perf_counter()
                last_z_vel = 0.0
                last_cmd_vel = 0.0

                while not stop_control.is_set() and not emergency_stop.is_set():
                    t_now = time.perf_counter()

                    # Get positions from polling threads (no SDK calls here)
                    _, current_x = get_latest_x()
                    t_z, current_z = get_latest_z()

                    # Safety: runtime Z limit check
                    if current_z > args.z_max:
                        emergency_reason[0] = f"Z exceeded limit: {current_z:.0f} > {args.z_max:.0f}"
                        emergency_stop.set()
                        try:
                            z_drive.halt()
                            stage.x.halt()
                        except Exception:
                            pass
                        break

                    # Compute actual Z velocity from sample timestamps
                    dt_z = t_z - last_t_z
                    if dt_z > 0.001:
                        actual_z_vel = (current_z - last_z) / dt_z
                    else:
                        actual_z_vel = last_z_vel  # Keep previous if no new sample

                    # Clamp X to data range
                    x_clamped = max(x_min, min(x_max, current_x))

                    # Ideal state from curve
                    ideal_x_vel = x_speed_um_s
                    ideal_z = z_func(x_clamped)
                    ideal_z_vel = dzdx_func(x_clamped) * ideal_x_vel

                    # Feedforward with lookahead
                    x_lookahead = current_x + ideal_x_vel * (args.lookahead_ms / 1000.0)
                    x_lookahead_clamped = max(x_min, min(x_max, x_lookahead))
                    z_vel_feedforward = dzdx_func(x_lookahead_clamped) * ideal_x_vel

                    # Errors
                    error_pos = ideal_z - current_z
                    error_vel = ideal_z_vel - actual_z_vel

                    # Control law
                    vel_from_pos = args.kp * error_pos
                    vel_from_vel = args.kv * error_vel
                    vel_from_ff = args.kff * z_vel_feedforward
                    commanded_vel = vel_from_pos + vel_from_vel + vel_from_ff

                    # Update Z state only if new sample
                    if t_z != last_t_z:
                        last_z = current_z
                        last_t_z = t_z
                        last_z_vel = actual_z_vel

                    # Safety: limit velocity near Z max
                    z_headroom = args.z_max - current_z
                    if z_headroom < 100 and commanded_vel > 0:
                        scale = max(0, z_headroom / 100)
                        commanded_vel = commanded_vel * scale

                    # Command Z velocity
                    if abs(commanded_vel - last_cmd_vel) > 0.5:
                        if commanded_vel > 0:
                            z_drive.start_towards_max(abs(commanded_vel))
                        elif commanded_vel < 0:
                            z_drive.start_towards_min(abs(commanded_vel))
                        else:
                            z_drive.halt()

                        control_log.append(
                            {
                                "t": t_now,
                                "x_um": current_x,
                                "z_actual_um": current_z,
                                "z_target_um": ideal_z,
                                "error_pos_um": error_pos,
                                "commanded_vel_um_s": commanded_vel,
                            }
                        )
                        last_cmd_vel = commanded_vel

                    # Sleep
                    elapsed = time.perf_counter() - t_now
                    sleep_time = max(0, control_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            # Start position polling threads
            print("\nStarting position polling...")
            x_thread = threading.Thread(target=poll_x, daemon=True)
            z_thread = threading.Thread(target=poll_z, daemon=True)
            x_thread.start()
            z_thread.start()
            time.sleep(0.1)

            # Warmup camera
            print(f"Camera warmup ({args.warmup_frames} frames)...")
            for _ in range(args.warmup_frames):
                current_image[0] = None
                acquisition.Acquire(context, None)
                if current_image[0] is not None:
                    current_image[0].Dispose()

            # Start control thread
            print("Starting Z controller...")
            control_thread = threading.Thread(target=control_z, daemon=True)
            control_thread.start()
            time.sleep(0.05)  # Brief settling

            # Start X motion
            print(f"\nStarting scan: X {x_min:.0f} -> {x_max:.0f} µm at {args.speed_mm} mm/s")
            t_scan_start = time.perf_counter()
            stage.x.start_towards_max(x_speed_um_s)

            # Capture loop
            frame_idx = 0
            while not emergency_stop.is_set():
                current_x = stage.x.position_um
                if current_x >= x_max:
                    break

                t_start = time.perf_counter()
                current_image[0] = None
                acquisition.Acquire(context, None)
                t_end = time.perf_counter()

                if current_image[0] is not None:
                    # Make copies of current sample lists for the saver
                    save_queue.put(
                        (frame_idx, t_start, t_end, current_image[0], list(x_samples), list(z_samples), t_scan_start)
                    )
                    frame_idx += 1

            t_scan_end = time.perf_counter()

            # Stop everything
            if emergency_stop.is_set():
                print(f"\n*** EMERGENCY STOP: {emergency_reason[0]} ***")

            stop_control.set()
            stage.x.halt()
            z_drive.halt()

            scan_duration = t_scan_end - t_scan_start
            print("\nScan complete!")
            print(f"  Duration: {scan_duration * 1000:.0f} ms (expected: {expected_duration_s * 1000:.0f} ms)")
            print(f"  Frames captured: {frame_idx}")

            # Stop polling
            time.sleep(0.1)
            stop_polling.set()
            x_thread.join(timeout=0.5)
            z_thread.join(timeout=0.5)
            control_thread.join(timeout=0.5)

            final_x = stage.x.position_um
            final_z = z_drive.position_um
            print(f"  Final X: {final_x:.0f} µm (target: {x_max:.0f})")
            print(f"  Final Z: {final_z:.0f} µm")
            print(f"  Position samples: X={len(x_samples)}, Z={len(z_samples)}")
            print(f"  Control updates: {len(control_log)}")

        finally:
            # Restore position
            print("\nRestoring position...")
            z_drive.move_to(initial_z)
            stage.x.move_to(initial_x)
            print(f"  Restored to: X={stage.x.position_um:.0f}, Z={z_drive.position_um:.0f} µm")

        # Wait for savers
        print(f"\nWaiting for savers ({save_queue.qsize()} queued)...")
        for _ in savers:
            save_queue.put(None)
        for t in savers:
            t.join()
        print(f"Saved {len(saved_frames_meta)} frames")

        # Sort frames by index
        saved_frames_meta.sort(key=lambda f: f["n"])

        # Analyze Z tracking error
        z_errors = [f["z_error"] for f in saved_frames_meta if f["z_error"] is not None]
        if z_errors:
            z_error_mean = float(np.mean(z_errors))
            z_error_std = float(np.std(z_errors))
            z_error_max = float(np.max(np.abs(z_errors)))
            z_error_p95 = float(np.percentile(np.abs(z_errors), 95))

            dof_20x = 1.7  # µm (20x DOF is ~1.7µm)
            within_dof = z_error_max < dof_20x

            print("\nZ Tracking Error:")
            print(f"  Mean: {z_error_mean:+.2f} µm")
            print(f"  Std: {z_error_std:.2f} µm")
            print(f"  Max: {z_error_max:.2f} µm")
            print(f"  95th percentile: {z_error_p95:.2f} µm")
            print(
                f"  {'PASS' if within_dof else 'FAIL'}: {'within' if within_dof else 'exceeds'} 20x DOF ({dof_20x} µm)"  # noqa: E501
            )
        else:
            z_error_mean = z_error_std = z_error_max = z_error_p95 = None
            within_dof = None

        # Build metadata (compatible with scan_area_v1.py for stitching)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "x_min_um": x_min,
            "x_max_um": x_max,
            "y_min_um": y_min,
            "y_max_um": y_max,
            "y_step_um": frame_height_um,  # Single row, so step = frame height
            "y_overlap_percent": 0,  # Single row
            "downsample": args.downsample,
            "scan_duration_s": scan_duration,
            "frame_count": frame_idx,
            "position_sample_count": len(x_samples),
            "scan_params": {
                "scan_speed_mm_s": args.speed_mm,
                "move_speed_mm_s": args.move_speed_mm,
                "area_rect": args.area_rect,
            },
            "camera": {
                "name": camera.name,
                "exposure_s": actual_exposure,
                "gain": args.gain,
                "binning": actual_binning,
                "readout_time_s": readout_time,
                "frame_width_px": frame_width_px,
                "frame_height_px": frame_height_px,
                "pixel_size_x_um": pixel_size_x_um,
                "pixel_size_y_um": pixel_size_y_um,
                "sensor_width_px": sensor_width_px,
                "sensor_height_px": sensor_height_px,
                "physical_pixel_x_um": physical_pixel_x_um,
                "physical_pixel_y_um": physical_pixel_y_um,
                "white_balance_bgr": [wb_blue, wb_green, wb_red],
                "gamma": args.gamma,
            },
            "optics": {
                "objective_mag": objective_mag,
                "objective_idx": objective_idx,
                "sample_pixel_x_um": sample_pixel_x_um,
                "sample_pixel_y_um": sample_pixel_y_um,
                "frame_width_um": frame_width_um,
                "frame_height_um": frame_height_um,
            },
            "lighting": {
                "lamp_name": scope.lamp.name,
                "lamp_intensity": scope.lamp.intensity,
                "lamp_max_intensity": scope.lamp.max_intensity,
                "shutter_name": scope.shutter.name,
                "shutter_open": scope.shutter.is_open,
            },
            # Rows array for stitch_area.py compatibility
            "rows": [
                {
                    "row_idx": 0,
                    "y_um": row_y,
                    "direction": 1,  # +X direction
                    "frame_start": 0,
                    "frame_end": frame_idx,
                    "duration_s": scan_duration,
                    "position_samples": len(x_samples),
                }
            ],
            # Position stream for stitching (scan_area_v1 format)
            "position_stream": [
                {"t_before": s[0] - t_scan_start, "t_after": s[1] - t_scan_start, "x_um": s[2], "row": 0}
                for s in x_samples
            ],
            "focus_tracking": {
                "enabled": True,
                "method": "state_feedback_feedforward",
                "params": {
                    "kp": args.kp,
                    "kv": args.kv,
                    "kff": args.kff,
                    "lookahead_ms": args.lookahead_ms,
                    "control_rate_hz": args.control_rate,
                },
                "surface": surface_name,
                "plane": PLANE_FIT,
                "profile_info": profile_info,
                "row_y_um": row_y,
                "z_safety": {
                    "z_max_um": args.z_max,
                    "z_margin_um": args.z_margin,
                    "z_expected_max_um": z_max_expected,
                    "emergency_stop": emergency_stop.is_set(),
                    "emergency_reason": emergency_reason[0],
                },
                "tracking_error": {
                    "mean_um": z_error_mean,
                    "std_um": z_error_std,
                    "max_um": z_error_max,
                    "p95_um": z_error_p95,
                    "within_20x_dof": within_dof,
                },
            },
            "timing": {
                "expected_duration_s": expected_duration_s,
            },
            "control_update_count": len(control_log),
            "frames": saved_frames_meta,
            "control_samples": control_log,
            # Position streams for detailed analysis (z_samples from control thread)
            "z_samples": [{"t": s[0] - t_scan_start, "z_um": s[1]} for s in z_samples],
        }

        # Save metadata
        meta_path = os.path.join(args.output, "scan_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nSaved metadata to {meta_path}")

        # Compress if requested
        if args.compress:
            print(f"Creating {args.output}.zip...")
            shutil.make_archive(args.output, "zip", args.output)
            print(f"Created {args.output}.zip")

        print("\nDone.")
        return 0


if __name__ == "__main__":
    exit(main())
