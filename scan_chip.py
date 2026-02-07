"""Chip-aware multi-row scan with focus plane Z tracking.

Scans only within a chip's convex hull polygon, tracking Z with a plane fit.
Each row's X extent is clipped to the polygon intersection, and Z velocity
is set to match the plane slope during X motion.

Run on microscope PC.

Usage:
    python scan_chip.py -o scans/chip0_20x \
        --chips-meta scans/working_overview_5x_stitch_chips.json \
        --chip 0 \
        --plane scans/focus_map_chip0_v10_no_overshoot_plane.json \
        --objective 20x --speed-mm 5

    # Test with just 3 rows
    python scan_chip.py -o scans/chip0_test \
        --chips-meta scans/chips.json --chip 0 --plane scans/plane.json \
        --objective 20x --row-limit 3
"""

import argparse
import bisect
from datetime import datetime
import json
import os
import queue
import re
import shutil
import threading
import time
from pathlib import Path


# ============================================================================
# Geometry helpers
# ============================================================================

def intersect_polygon_with_y(polygon, y):
    """Find X extent where horizontal line y intersects a convex polygon.

    Args:
        polygon: List of [x, y] vertices.
        y: Y coordinate of the horizontal line.

    Returns:
        (x_min, x_max) or None if no intersection.
    """
    intersections = []
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

    return (min(intersections), max(intersections))


def compute_plane_z(a, b, c, x_um, y_um):
    """Compute Z from plane coefficients. Z_um = a * X_um + b * Y_um + c."""
    return a * x_um + b * y_um + c


# ============================================================================
# Helpers (from scan_area_v1.py)
# ============================================================================

MICROSCOPE_DESCRIPTION = os.path.join(os.path.dirname(__file__), "microscope_description.json")


def load_microscope_description() -> dict | None:
    """Load microscope hardware description for pre-connection validation."""
    if not os.path.exists(MICROSCOPE_DESCRIPTION):
        return None
    try:
        with open(MICROSCOPE_DESCRIPTION) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


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


def parse_objective_arg(value: str, nosepiece) -> int:
    """Parse objective argument to position number."""
    try:
        pos = int(value)
        if nosepiece.min_position <= pos <= nosepiece.max_position:
            return pos
    except ValueError:
        pass

    match = re.match(r"^(\d+(?:\.\d+)?)[xX]?$", value.strip())
    if match:
        mag = float(match.group(1))
        for pos, obj_mag in nosepiece.magnifications.items():
            if obj_mag == mag:
                return pos

    valid = []
    for pos in range(nosepiece.min_position, nosepiece.max_position + 1):
        mag = nosepiece.magnifications.get(pos)
        if mag:
            valid.append(f"{pos} ({mag}x)")
        else:
            valid.append(str(pos))

    raise ValueError(f"Invalid objective '{value}'. Valid options: {', '.join(valid)}")


# ============================================================================
# Main Scan
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chip-aware scan with focus plane Z tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan chip 0 at 20x with focus plane
  python scan_chip.py -o scans/chip0_20x \\
      --chips-meta scans/chips.json --chip 0 \\
      --plane scans/plane.json --objective 20x

  # Test with 3 rows
  python scan_chip.py -o scans/chip0_test \\
      --chips-meta scans/chips.json --chip 0 \\
      --plane scans/plane.json --objective 20x --row-limit 3
"""
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    # Chip and plane
    chip_group = parser.add_argument_group("Chip and focus")
    chip_group.add_argument("--chips-meta", type=str, required=True,
                            help="Path to chips JSON from find_chips.py")
    chip_group.add_argument("--chip", type=int, required=True,
                            help="Chip index (from chips JSON)")
    chip_group.add_argument("--plane", type=str, required=True,
                            help="Path to plane JSON from analyze_focus_map.py")
    chip_group.add_argument("--padding", type=float, default=0,
                            help="Padding around contour per row in µm (default: 0, negative = shrink)")

    # Scan control
    scan_group = parser.add_argument_group("Scan control")
    scan_group.add_argument("--x-overlap-percent", type=float, default=30,
                            help="Target X overlap between saved frames, %% (default: 30). "
                                 "Frames captured before advancing enough are discarded.")
    scan_group.add_argument("--row-limit", type=int, default=None,
                            help="Scan only N rows then stop (for testing)")

    # Optics
    optics_group = parser.add_argument_group("Optics")
    optics_group.add_argument("--objective", type=str, metavar="MAG",
                              help="Objective magnification (e.g., 5, 10, 20x)")

    # Motion
    motion_group = parser.add_argument_group("Motion")
    motion_group.add_argument("--speed-mm", type=float, default=5.0,
                              help="Scan speed in mm/s (default: 5)")
    motion_group.add_argument("--move-speed-mm", type=float, default=40,
                              help="Move speed for positioning (default: 40)")

    # Camera
    frame_group = parser.add_argument_group("Camera")
    frame_group.add_argument("--exposure-ms", type=float, default=0.25,
                             help="Exposure time in ms (default: 0.25)")
    frame_group.add_argument("--gain", type=float, default=4.0,
                             help="Camera gain (default: 4.0)")
    frame_group.add_argument("--binning", type=int, default=3, choices=[1, 2, 3],
                             help="Camera binning NxN (default: 3)")
    frame_group.add_argument("--white-balance", type=str, default="2.51,1.02,1.41",
                             help="White balance as B,G,R gains")
    frame_group.add_argument("--gamma", type=float, default=1.0, help="Gamma (default: 1.0)")
    frame_group.add_argument("--downsample", type=int, default=1, help="Downsample factor")
    frame_group.add_argument("--warmup-frames", type=int, default=3,
                             help="Warmup captures before each row (default: 3)")
    frame_group.add_argument("--y-overlap-percent", type=float, default=12,
                             help="Y overlap between rows as %% of frame height (default: 12)")

    # Safety
    safety_group = parser.add_argument_group("Safety")
    safety_group.add_argument("--z-max", type=float, default=26000.0,
                              help="Hard Z limit in µm (default: 26000)")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--clean", action="store_true", help="Wipe output directory if exists")
    output_group.add_argument("--write-threads", type=int, default=2, help="Image writer threads")
    output_group.add_argument("--compress", action="store_true", help="Create .zip of output")

    args = parser.parse_args()

    # ---- Load chip data ----
    chips_path = Path(args.chips_meta)
    if not chips_path.exists():
        print(f"Error: Chips file not found: {chips_path}")
        return 1

    with open(chips_path) as f:
        chips_data = json.load(f)

    chips = chips_data.get("chips", [])
    if args.chip < 0 or args.chip >= len(chips):
        print(f"Error: Chip index {args.chip} out of range (0-{len(chips)-1})")
        return 1

    chip = chips[args.chip]
    polygon = chip["convex_hull_stage_um"]
    bbox = chip["bbox_stage_um"]
    centroid = chip["centroid_stage_um"]

    # ---- Load plane data ----
    plane_path = Path(args.plane)
    if not plane_path.exists():
        print(f"Error: Plane file not found: {plane_path}")
        return 1

    with open(plane_path) as f:
        plane_data = json.load(f)

    plane = plane_data["plane"]
    plane_a = plane["a"]  # µm/µm
    plane_b = plane["b"]  # µm/µm
    plane_c = plane["c"]  # µm

    # ---- Parse white balance ----
    wb_parts = args.white_balance.split(",")
    if len(wb_parts) != 3:
        print("Error: --white-balance must be 3 comma-separated values (B,G,R)")
        return 1
    try:
        wb_blue, wb_green, wb_red = float(wb_parts[0]), float(wb_parts[1]), float(wb_parts[2])
    except ValueError:
        print("Error: --white-balance values must be numbers")
        return 1

    binning_idx = args.binning - 1

    # ---- Pre-flight Z validation ----
    # Check Z at all polygon vertices
    z_at_vertices = [compute_plane_z(plane_a, plane_b, plane_c, v[0], v[1]) for v in polygon]
    z_min_expected = min(z_at_vertices)
    z_max_expected = max(z_at_vertices)

    if z_max_expected > args.z_max:
        print(f"ABORT: Expected max Z ({z_max_expected:.0f}) exceeds limit ({args.z_max:.0f})")
        return 1
    if z_min_expected < 0:
        print(f"ABORT: Expected min Z ({z_min_expected:.0f}) is negative")
        return 1

    # ---- Output directory ----
    if os.path.exists(args.output):
        if args.clean:
            shutil.rmtree(args.output)
        else:
            print(f"Error: Output directory '{args.output}' exists. Use --clean to wipe.")
            return 1
    os.makedirs(args.output)

    # ---- Print scan plan ----
    print("Chip Scan with Focus Plane")
    print("=" * 60)
    print(f"Chip: #{args.chip}")
    print(f"  BBox: X=[{bbox['x_min']:.0f}, {bbox['x_max']:.0f}], Y=[{bbox['y_min']:.0f}, {bbox['y_max']:.0f}] µm")
    print(f"  Centroid: ({centroid[0]:.0f}, {centroid[1]:.0f}) µm")
    print(f"  Hull vertices: {len(polygon)}")
    print(f"  Area: {chip['area_um2']/1e6:.1f} mm²")
    print()
    print(f"Focus plane: Z = {plane_a*1000:.4f}*X_mm + {plane_b*1000:.4f}*Y_mm + {plane_c:.2f}")
    print(f"  Expected Z range: {z_min_expected:.0f} - {z_max_expected:.0f} µm")
    print(f"  Z limit: {args.z_max:.0f} µm")
    print()
    print(f"Scan speed: {args.speed_mm:.1f} mm/s")
    print(f"Padding: {args.padding:.0f} µm")
    if args.row_limit:
        print(f"Row limit: {args.row_limit}")
    print()

    # ---- Connect to hardware ----
    from PIL import Image as PILImage
    from flakefinder.leica import LeicaConnection, Stage, ZDrive, Lamp, Shutter, Nosepiece
    from flakefinder.leica.camera import Camera
    from flakefinder.leica.enums import UCAPI_IID
    from flakefinder.leica.core import get_interface_required

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up hardware
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        # Fast position readers
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter
        z_bcv_hysteresis = getattr(z_drive, 'bcv_hysteresis', None) or z_drive.bcv

        # Nosepiece
        nosepiece = None
        objective_mag = None
        objective_idx = None
        try:
            nosepiece = Nosepiece.from_connection(conn)
            objective_idx = nosepiece.position
            objective_mag = nosepiece.magnification
        except LookupError:
            pass

        # Switch objective if requested
        if args.objective is not None:
            if nosepiece is None:
                print("Error: Nosepiece not available")
                return 1
            try:
                target_pos = parse_objective_arg(args.objective, nosepiece)
                if target_pos != objective_idx:
                    target_mag = nosepiece.magnifications.get(target_pos)
                    print(f"Switching objective: {objective_mag}x -> {target_mag}x...")
                    nosepiece.position = target_pos
                    objective_idx = nosepiece.position
                    objective_mag = nosepiece.magnification
                    print(f"Objective: now at {objective_mag}x")
                else:
                    print(f"Objective: already at {objective_mag}x")
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        # Lighting
        shutter = None
        lamp = None
        try:
            shutter = Shutter.from_connection(conn)
            shutter.open()
        except LookupError:
            pass
        try:
            lamp = Lamp.from_connection(conn)
            lamp.full()
        except LookupError:
            pass

        # Camera
        try:
            camera = Camera.from_connection(conn)
        except LookupError:
            print("Camera not found!")
            return 1

        acquisition = get_interface_required(camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)

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
        frame_width_um = None
        frame_height_um = None
        sample_pixel_x_um = None
        sample_pixel_y_um = None
        if physical_pixel_x_um and actual_binning and objective_mag:
            sample_pixel_x_um = physical_pixel_x_um * actual_binning / objective_mag
            if frame_width_px:
                frame_width_um = frame_width_px * sample_pixel_x_um
        if physical_pixel_y_um and actual_binning and objective_mag:
            sample_pixel_y_um = physical_pixel_y_um * actual_binning / objective_mag
            if frame_height_px:
                frame_height_um = frame_height_px * sample_pixel_y_um

        if not frame_height_um:
            print("Error: Could not determine frame height. Check objective/camera.")
            return 1

        exp_str = f"{actual_exposure*1000:.2f}ms" if actual_exposure else "?"
        print(f"Camera: {camera.name}")
        print(f"  Binning: {actual_binning}x{actual_binning}, Exposure: {exp_str}, Gain: {args.gain}")
        if frame_width_px and frame_height_px:
            print(f"  Frame: {frame_width_px}x{frame_height_px} px")
        if frame_width_um and frame_height_um:
            print(f"  FOV: {frame_width_um:.1f} x {frame_height_um:.1f} µm")
        if objective_mag:
            print(f"  Objective: {objective_mag}x")
        print()

        # ---- Frame skip target ----
        target_advance = frame_width_um * (1 - args.x_overlap_percent / 100)
        print(f"Frame skip: target advance {target_advance:.0f} µm ({args.x_overlap_percent:.0f}% X overlap)")
        print()

        # ---- Compute row plan ----
        y_step = frame_height_um * (1 - args.y_overlap_percent / 100)

        rows_plan = []  # List of (y, x_min, x_max)
        y = bbox["y_min"]
        while y <= bbox["y_max"]:
            extent = intersect_polygon_with_y(polygon, y)
            if extent is not None:
                x_min, x_max = extent
                # Apply padding (negative = shrink inward)
                x_min -= args.padding
                x_max += args.padding
                # Skip rows narrower than one frame (polygon tips or large negative padding)
                if (x_max - x_min) < (frame_width_um or 0):
                    y += y_step
                    continue
                rows_plan.append((y, x_min, x_max))
            y += y_step

        if args.row_limit:
            rows_plan = rows_plan[:args.row_limit]

        if not rows_plan:
            print("Error: No rows intersect the chip contour")
            return 1

        # Compute Z range across all row endpoints
        all_z = []
        for row_y, row_x_min, row_x_max in rows_plan:
            all_z.append(compute_plane_z(plane_a, plane_b, plane_c, row_x_min, row_y))
            all_z.append(compute_plane_z(plane_a, plane_b, plane_c, row_x_max, row_y))

        z_scan_min = min(all_z)
        z_scan_max = max(all_z)

        if z_scan_max > args.z_max:
            print(f"ABORT: Scan Z max ({z_scan_max:.0f}) exceeds limit ({args.z_max:.0f})")
            return 1

        # Print row plan summary
        row_widths = [(x_max - x_min) / 1000 for _, x_min, x_max in rows_plan]
        total_distance_mm = sum(row_widths)
        est_scan_time_s = total_distance_mm / args.speed_mm
        # Add repositioning time estimate (Y jog + X return per row)
        est_total_time_s = est_scan_time_s + len(rows_plan) * 0.5

        print(f"Row plan: {len(rows_plan)} rows")
        print(f"  Y step: {y_step:.1f} µm ({args.y_overlap_percent:.0f}% overlap)")
        print(f"  Row widths: {min(row_widths):.1f} - {max(row_widths):.1f} mm")
        print(f"  Total scan distance: {total_distance_mm:.1f} mm")
        print(f"  Z range: {z_scan_min:.0f} - {z_scan_max:.0f} µm")
        z_vel = abs(plane_a * args.speed_mm * 1000)
        print(f"  Z velocity: {z_vel:.1f} µm/s (from plane slope)")
        print(f"  Estimated time: ~{est_total_time_s:.0f}s")
        print()

        # ---- Set up acquisition context ----
        context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # ---- Set up saver threads ----
        save_queue = queue.Queue()
        saved_frames_meta = []

        def saver_thread():
            while True:
                item = save_queue.get()
                if item is None:
                    break

                frame_idx, row_idx, t_start, t_end, image, row_y, row_x_samples, row_z_samples, t0 = item

                arr = Camera._image_to_numpy(image)
                image.Dispose()

                img = PILImage.fromarray(arr)
                if args.downsample > 1:
                    new_size = (img.width // args.downsample, img.height // args.downsample)
                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                path = os.path.join(args.output, f"frame_{frame_idx:04d}.jpg")
                img.save(path, quality=95)

                # Compute metadata
                x_start_interp = interpolate_position(t_start, row_x_samples)
                x_end_interp = interpolate_position(t_end, row_x_samples)
                z_interp = interpolate_z_position(t_start, row_z_samples)

                # Compute ideal Z and error
                z_ideal = compute_plane_z(plane_a, plane_b, plane_c, x_start_interp, row_y) if x_start_interp else None
                z_error = (z_interp - z_ideal) if (z_interp is not None and z_ideal is not None) else None

                dt = t_end - t_start
                x_vel = (x_end_interp - x_start_interp) / dt if dt > 0 and x_start_interp and x_end_interp else 0

                saved_frames_meta.append({
                    "n": frame_idx,
                    "row": row_idx,
                    "t_start": t_start - t0,
                    "t_end": t_end - t0,
                    "x_start": x_start_interp,
                    "x_end": x_end_interp,
                    "x_vel": x_vel,
                    "y_um": row_y,
                    "z_actual": z_interp,
                    "z_ideal": z_ideal,
                    "z_error": z_error,
                })

                save_queue.task_done()

        savers = []
        for _ in range(args.write_threads):
            t = threading.Thread(target=saver_thread, daemon=True)
            t.start()
            savers.append(t)

        # ---- Save initial position ----
        initial_x = stage.x.position_um
        initial_y = stage.y.position_um
        initial_z = z_drive.position_um

        print(f"Initial position: X={initial_x:.0f}, Y={initial_y:.0f}, Z={initial_z:.0f} µm")

        # Set move speed for positioning
        stage.x.set_velocity_um_s(args.move_speed_mm * 1000)
        stage.y.set_velocity_um_s(args.move_speed_mm * 1000)

        # Scan speed
        x_speed_um_s = args.speed_mm * 1000

        # ---- Scan loop ----
        total_scan_start = time.perf_counter()
        all_position_samples = []
        global_frame_idx = 0

        try:
            for row_idx, (row_y, row_x_min, row_x_max) in enumerate(rows_plan):
                # Snake pattern: even rows +X, odd rows -X
                direction = 1 if row_idx % 2 == 0 else -1
                if direction == 1:
                    x_start_pos, x_end_pos = row_x_min, row_x_max
                    dir_str = "+X"
                else:
                    x_start_pos, x_end_pos = row_x_max, row_x_min
                    dir_str = "-X"

                # Compute Z for start and end of this row
                z_start = compute_plane_z(plane_a, plane_b, plane_c, x_start_pos, row_y)
                z_end = compute_plane_z(plane_a, plane_b, plane_c, x_end_pos, row_y)

                # Z velocity during this row: dZ/dt = (plane_a * direction) * x_speed
                z_vel_um_s = plane_a * direction * x_speed_um_s

                row_width_mm = (row_x_max - row_x_min) / 1000
                print(f"Row {row_idx}/{len(rows_plan)-1}: Y={row_y:.0f}µm, {dir_str}, "
                      f"X=[{row_x_min:.0f},{row_x_max:.0f}] ({row_width_mm:.1f}mm), "
                      f"Z={z_start:.0f}->{z_end:.0f}")

                # Runtime Z safety check
                if max(z_start, z_end) > args.z_max:
                    print(f"  SKIP: Z would exceed limit ({max(z_start, z_end):.0f} > {args.z_max:.0f})")
                    continue

                # Move to row start position
                stage.y.move_to(row_y)
                stage.x.move_to(x_start_pos)
                z_drive.move_to_corrected(z_start)
                time.sleep(0.1)

                # Set up position polling for this row
                x_samples = []
                z_samples = []
                stop_polling = threading.Event()

                def x_poll_thread():
                    while not stop_polling.is_set():
                        t_before = time.perf_counter()
                        x_native = x_bcv.GetControlValue()
                        t_after = time.perf_counter()
                        x_um = x_converter.GetMetricsValue(x_native)
                        x_samples.append((t_before, t_after, x_um))

                def z_poll_thread():
                    while not stop_polling.is_set():
                        t = time.perf_counter()
                        z_native = z_bcv_hysteresis.GetControlValue()
                        z_um = z_converter.GetMetricsValue(z_native)
                        z_samples.append((t, z_um))

                # Start polling
                x_thread = threading.Thread(target=x_poll_thread, daemon=True)
                z_thread = threading.Thread(target=z_poll_thread, daemon=True)
                x_thread.start()
                z_thread.start()

                # Warmup camera
                for _ in range(args.warmup_frames):
                    current_image[0] = None
                    acquisition.Acquire(context, None)
                    if current_image[0] is not None:
                        current_image[0].Dispose()

                row_start = time.perf_counter()
                row_frame_start = global_frame_idx
                row_frame_count = 0
                row_skip_count = 0
                last_saved_x = None

                # Start Z velocity tracking
                if abs(z_vel_um_s) > 0.1:
                    if z_vel_um_s > 0:
                        z_drive.start_towards_max(abs(z_vel_um_s))
                    else:
                        z_drive.start_towards_min(abs(z_vel_um_s))

                # Start X motion
                stage.x.set_velocity_um_s(x_speed_um_s)
                handle = stage.x.move_to_async(x_end_pos)

                # Capture frames during move (with position-based skip)
                while not handle.is_complete:
                    t_start = time.perf_counter()
                    current_image[0] = None
                    acquisition.Acquire(context, None)
                    t_end = time.perf_counter()

                    if current_image[0] is not None:
                        # Check if we've advanced enough to save this frame
                        x_now = x_samples[-1][2] if x_samples else None
                        if last_saved_x is not None and x_now is not None and abs(x_now - last_saved_x) < target_advance:
                            current_image[0].Dispose()
                            row_skip_count += 1
                        else:
                            save_queue.put((
                                global_frame_idx, row_idx, t_start, t_end, current_image[0],
                                row_y, list(x_samples), list(z_samples), total_scan_start
                            ))
                            if x_now is not None:
                                last_saved_x = x_now
                            global_frame_idx += 1
                            row_frame_count += 1

                row_end = time.perf_counter()
                handle.dispose()

                # Stop Z motion
                z_drive.halt()

                # Stop polling
                stop_polling.set()
                x_thread.join(timeout=1.0)
                z_thread.join(timeout=1.0)

                row_duration = row_end - row_start

                # Filter position samples to row scan period
                row_x_samples = [(tb, ta, x) for tb, ta, x in x_samples if row_start <= tb <= row_end]

                print(f"  {row_frame_count} saved, {row_skip_count} skipped, {len(row_x_samples)} pos, {row_duration:.2f}s")

                # Add position samples to global list
                for tb, ta, x_um in row_x_samples:
                    all_position_samples.append({
                        "t_before": tb - total_scan_start,
                        "t_after": ta - total_scan_start,
                        "x_um": x_um,
                        "row": row_idx,
                    })

                # Restore move speed for positioning to next row
                stage.x.set_velocity_um_s(args.move_speed_mm * 1000)

        finally:
            # Stop any motion
            z_drive.halt()
            stage.x.halt()

        total_scan_end = time.perf_counter()
        total_duration = total_scan_end - total_scan_start

        # Return to initial position
        print(f"\nReturning to initial position...")
        stage.x.set_velocity_um_s(args.move_speed_mm * 1000)
        stage.y.set_velocity_um_s(args.move_speed_mm * 1000)
        hx = stage.x.move_to_async(initial_x)
        hy = stage.y.move_to_async(initial_y)

        # Wait for savers
        print(f"Waiting for savers ({save_queue.qsize()} frames queued)...")
        for _ in savers:
            save_queue.put(None)
        for t in savers:
            t.join()
        print(f"Saved {len(saved_frames_meta)} frames")

        # Sort frames
        saved_frames_meta.sort(key=lambda f: f["n"])

        # Z tracking error stats
        z_errors = [f["z_error"] for f in saved_frames_meta if f["z_error"] is not None]
        if z_errors:
            import numpy as np
            z_error_arr = np.array(z_errors)
            z_error_mean = float(np.mean(z_error_arr))
            z_error_std = float(np.std(z_error_arr))
            z_error_max = float(np.max(np.abs(z_error_arr)))
            z_error_p95 = float(np.percentile(np.abs(z_error_arr), 95))
        else:
            z_error_mean = z_error_std = z_error_max = z_error_p95 = None

        # Build rows metadata
        rows_meta = []
        for row_idx, (row_y, row_x_min, row_x_max) in enumerate(rows_plan):
            direction = 1 if row_idx % 2 == 0 else -1
            row_frames = [f for f in saved_frames_meta if f["row"] == row_idx]
            frame_start = row_frames[0]["n"] if row_frames else global_frame_idx
            frame_end = (row_frames[-1]["n"] + 1) if row_frames else global_frame_idx
            row_pos_samples = [s for s in all_position_samples if s["row"] == row_idx]

            rows_meta.append({
                "row_idx": row_idx,
                "y_um": row_y,
                "x_min_um": row_x_min,
                "x_max_um": row_x_max,
                "direction": direction,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "duration_s": None,  # Not tracked per-row precisely
                "position_samples": len(row_pos_samples),
            })

        # Build metadata (scan_area_v1.py compatible for stitching)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "x_min_um": min(r[1] for r in rows_plan),
            "x_max_um": max(r[2] for r in rows_plan),
            "y_min_um": rows_plan[0][0],
            "y_max_um": rows_plan[-1][0],
            "y_step_um": y_step,
            "y_overlap_percent": args.y_overlap_percent,
            "downsample": args.downsample,
            "scan_duration_s": total_duration,
            "frame_count": global_frame_idx,
            "position_sample_count": len(all_position_samples),
            "scan_params": {
                "scan_speed_mm_s": args.speed_mm,
                "move_speed_mm_s": args.move_speed_mm,
                "padding_um": args.padding,
                "row_limit": args.row_limit,
            },
            "chip_info": {
                "chips_meta": str(chips_path),
                "chip_index": args.chip,
                "bbox_stage_um": bbox,
                "centroid_stage_um": centroid,
                "area_um2": chip["area_um2"],
                "hull_vertices": len(polygon),
            },
            "focus_plane": {
                "plane_file": str(plane_path),
                "a": plane_a,
                "b": plane_b,
                "c": plane_c,
                "equation": plane.get("equation", f"Z = {plane_a}*X + {plane_b}*Y + {plane_c}"),
                "z_range_um": [z_scan_min, z_scan_max],
                "tracking_error": {
                    "mean_um": z_error_mean,
                    "std_um": z_error_std,
                    "max_um": z_error_max,
                    "p95_um": z_error_p95,
                },
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
                "lamp_name": lamp.name if lamp else None,
                "lamp_intensity": lamp.intensity if lamp else None,
                "lamp_max_intensity": lamp.max_intensity if lamp else None,
                "shutter_name": shutter.name if shutter else None,
                "shutter_open": shutter.is_open if shutter else None,
            },
            "rows": rows_meta,
            "position_stream": all_position_samples,
            "frames": saved_frames_meta,
        }

        # Save metadata
        meta_path = os.path.join(args.output, "scan_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")

        # Compress if requested
        if args.compress:
            print(f"Creating {args.output}.zip...")
            shutil.make_archive(args.output, 'zip', args.output)
            print(f"Created {args.output}.zip")

        # Wait for return
        Stage.wait_all([hx, hy])
        hx.dispose()
        hy.dispose()
        z_drive.move_to_corrected(initial_z)

        # Clean up camera
        camera.dispose()

        # Summary
        print()
        print("=" * 60)
        print("SCAN SUMMARY:")
        print(f"  Total time: {total_duration:.1f}s")
        print(f"  Rows: {len(rows_plan)}")
        print(f"  Total frames: {global_frame_idx}")
        print(f"  Avg FPS: {global_frame_idx / total_duration:.1f}" if total_duration > 0 else "  Avg FPS: N/A")
        if z_error_max is not None:
            dof_20x = 1.7
            print(f"  Z tracking error: mean={z_error_mean:+.2f}, std={z_error_std:.2f}, "
                  f"max={z_error_max:.2f}, p95={z_error_p95:.2f} µm")
            print(f"  {'PASS' if z_error_max < dof_20x else 'NOTE'}: max error "
                  f"{'within' if z_error_max < dof_20x else 'exceeds'} 20x DOF ({dof_20x} µm)")
        print(f"  Output: {args.output}/")
        print()
        print("Done.")

        return 0


if __name__ == "__main__":
    exit(main())
