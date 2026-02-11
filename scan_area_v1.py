"""Multi-row snake scan with parallel position reading and image capture.

Supports both 5x overview scanning and 20x detection scanning for MaskTerial.
See docs/maskterial_integration.md for 20x scanning context.
"""

import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime

from flakefinder.data_utils import compute_frame_size_um, require_microscope_description
from flakefinder.scan_utils import build_microscope_meta, interpolate_position, parse_position, parse_white_balance


def parse_area_rect(value: str) -> tuple[float, float, float, float]:
    """Parse area rectangle from comma-separated string.

    Args:
        value: "x_min,x_max,y_min,y_max" in µm

    Returns:
        (x_min, x_max, y_min, y_max) tuple, with coordinates sorted if swapped.

    Raises:
        ValueError: If format is invalid or values are degenerate.
    """
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("--area-rect must be x_min,x_max,y_min,y_max (4 values)")
    try:
        x1, x2, y1, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        raise ValueError("--area-rect values must be numbers") from None

    # Sort coordinates if swapped
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Check for degenerate (zero-area) rectangles
    if x_min == x_max:
        raise ValueError(f"--area-rect: x_min and x_max cannot be equal ({x_min})")
    if y_min == y_max:
        raise ValueError(f"--area-rect: y_min and y_max cannot be equal ({y_min})")

    return (x_min, x_max, y_min, y_max)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-row snake scan with configurable objective and area",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5x overview scan of full stage area (default)
  python scan_area_v1.py -o scan_5x --objective-mag 5

  # 20x detection scan for MaskTerial over specific area
  python scan_area_v1.py -o scan_20x --objective-mag 20x --z 24699 --area-rect 10000,60000,15000,55000

  # Objective by turret position (position 3 = 20x)
  python scan_area_v1.py -o scan_20x --objective-pos 3 --z 24699
""",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    # Scan area options
    area_group = parser.add_argument_group("Scan area")
    area_group.add_argument(
        "--margin",
        type=float,
        default=1000,
        help="Margin from stage edges in µm (default: 1000, ignored if --area-rect set)",
    )
    area_group.add_argument(
        "--area-rect",
        type=str,
        metavar="X1,X2,Y1,Y2",
        help="Explicit scan area as x_min,x_max,y_min,y_max in µm",
    )

    # Objective/optics options
    optics_group = parser.add_argument_group("Optics")
    optics_group.add_argument(
        "--objective-mag",
        type=str,
        metavar="MAG",
        help="Objective by magnification (e.g., 5, 5x, 20, 2.5) - switches before scan",
    )
    optics_group.add_argument(
        "--objective-pos",
        type=int,
        metavar="POS",
        help="Objective by turret position (1-6) - switches before scan",
    )

    # Motion options
    motion_group = parser.add_argument_group("Motion")
    motion_group.add_argument(
        "--speed-mm",
        type=float,
        default=40,
        help="Scan speed in mm/s for X scanning and Y jogs (default: 40)",
    )
    motion_group.add_argument(
        "--move-speed-mm",
        type=float,
        default=40,
        help="Move speed in mm/s for positioning moves (default: 40)",
    )
    motion_group.add_argument(
        "--z",
        type=float,
        metavar="Z",
        help="Initial Z position in µm (moved before scan, e.g. 24690)",
    )
    motion_group.add_argument(
        "--auto-focus-pos",
        type=parse_position,
        metavar="X,Y",
        help="XY position for autofocus calibration before scan (µm)",
    )

    # Frame options
    frame_group = parser.add_argument_group("Frame capture")
    frame_group.add_argument(
        "--x-overlap-percent",
        type=float,
        default=50,
        help="Target X overlap between saved frames, %% (default: 50). "
        "Frames captured before advancing enough are discarded.",
    )
    frame_group.add_argument(
        "--y-overlap-percent",
        type=float,
        default=12,
        help="Y overlap between rows as %% of frame height (default: 12)",
    )
    frame_group.add_argument("--downsample", type=int, default=1, help="Downsample factor (2 = half dims)")
    frame_group.add_argument(
        "--white-balance",
        type=parse_white_balance,
        default="2.51,1.02,1.41",
        help="White balance as B,G,R gains (default: 2.51,1.02,1.41)",
    )
    frame_group.add_argument("--gamma", type=float, default=1.0, help="Gamma level (default: 1.0)")
    frame_group.add_argument(
        "--binning",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Camera binning NxN (1=full res, 2=2x2, 3=3x3, default: 3)",
    )
    frame_group.add_argument(
        "--exposure-ms",
        type=float,
        default=1.0,
        help="Exposure time in milliseconds (default: 1.0)",
    )
    frame_group.add_argument("--gain", type=float, default=1.0, help="Camera gain multiplier (default: 1.0)")
    frame_group.add_argument(
        "--warmup-frames",
        type=int,
        default=3,
        help="Warmup captures before each row (default: 3, 0 to disable)",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--compress", action="store_true", help="Create .zip of output directory")
    output_group.add_argument("--clean", action="store_true", help="Wipe output directory if it exists")
    output_group.add_argument(
        "--write-threads", type=int, default=2, help="Number of image writer threads (default: 2)"
    )
    output_group.add_argument("-q", "--quiet", action="store_true", help="Suppress per-row progress output")

    args = parser.parse_args()

    # Validate objective args are mutually exclusive
    if args.objective_mag is not None and args.objective_pos is not None:
        print("Error: --objective-mag and --objective-pos are mutually exclusive")
        return 1

    wb = args.white_balance

    # Convert binning to SDK index (1/2/3 -> 0/1/2)
    binning_idx = args.binning - 1

    # Pre-validation using microscope description (avoids slow hardware connection)
    desc = require_microscope_description()

    # Validate area-rect against stage limits
    if args.area_rect:
        try:
            x_min, x_max, y_min, y_max = parse_area_rect(args.area_rect)
            desc_x_min = desc.stage.x.min_um
            desc_x_max = desc.stage.x.max_um
            desc_y_min = desc.stage.y.min_um
            desc_y_max = desc.stage.y.max_um

            if x_min < desc_x_min:
                print(f"Error: x_min ({x_min:.0f}) is below stage minimum ({desc_x_min:.0f})")
                return 1
            if x_max > desc_x_max:
                print(f"Error: x_max ({x_max:.0f}) exceeds stage maximum ({desc_x_max:.0f})")
                return 1
            if y_min < desc_y_min:
                print(f"Error: y_min ({y_min:.0f}) is below stage minimum ({desc_y_min:.0f})")
                return 1
            if y_max > desc_y_max:
                print(f"Error: y_max ({y_max:.0f}) exceeds stage maximum ({desc_y_max:.0f})")
                return 1
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Estimate scan coverage if objective specified
    if args.objective_mag:
        obj_match = re.match(r"^(\d+(?:\.\d+)?)[xX]?$", args.objective_mag.strip())
        if obj_match:
            obj_mag = float(obj_match.group(1))
            frame_size = compute_frame_size_um(desc.camera, obj_mag, binning_idx=binning_idx)
            if frame_size:
                print(
                    f"Pre-check: {obj_mag}x objective @ {args.binning}x{args.binning} binning, frame ~{frame_size[0]:.0f} x {frame_size[1]:.0f} µm"  # noqa: E501
                )

    from PIL import Image as PILImage

    # Check/create output directory before connecting to hardware
    if os.path.exists(args.output):
        if args.clean:
            shutil.rmtree(args.output)
        else:
            print(f"Error: Output directory '{args.output}' already exists. Use --clean to wipe it.")
            return 1
    os.makedirs(args.output)

    from flakefinder.image_utils import sdk_image_to_numpy
    from flakefinder.leica import Microscope, wait_all

    print("Area Scan v1 (Snake Pattern)")
    print("=" * 50)

    with Microscope() as scope:
        scope.validate_description(desc)
        stage = scope.stage
        z = scope.z
        x_bcv = stage.x.bcv  # Native position reader
        x_converter = stage.x.converter  # For native -> um conversion

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

        # Move Z if requested (after objective switch, before scan)
        if args.z is not None:
            current_z = z.position_um
            print(f"Moving Z: {current_z:.1f} -> {args.z:.1f} µm...")
            z.move_to_corrected(args.z)
            print(f"Z at {z.position_um:.1f} µm")

        # Helper to set stage velocity
        def set_stage_speed(speed_mm: float, label: str = "") -> float:
            """Set X and Y velocity, return actual X speed in mm/s."""
            target_um_s = speed_mm * 1000
            actual_mm = speed_mm

            for axis, name in [(stage.x, "X"), (stage.y, "Y")]:
                max_um_s = axis.max_velocity_um_s
                min_um_s = axis.min_velocity_um_s or 0
                clamped = max(min_um_s, min(max_um_s, target_um_s))
                axis.set_velocity_um_s(clamped)
                if name == "X":
                    actual_mm = axis.velocity_um_s / 1000

            return actual_mm

        # Set initial move speed
        move_speed_mm = args.move_speed_mm
        scan_speed_mm = args.speed_mm

        print(f"Move speed: {move_speed_mm:.1f} mm/s")
        print(f"Scan speed: {scan_speed_mm:.1f} mm/s")

        # Start with move speed for initial positioning
        set_stage_speed(move_speed_mm)
        actual_speed_mm = scan_speed_mm  # Will be set before scan

        # Lighting and camera
        scope.light_on()
        camera = scope.camera
        acquisition = scope.acquisition

        # Configure camera
        camera.trigger_mode = 0  # CONTINUOUS for faster capture
        camera.binning = binning_idx
        camera.exposure_time = args.exposure_ms / 1000.0
        camera.gain = args.gain
        camera.gain_rgb = wb
        camera.gamma = args.gamma

        # Autofocus position (already parsed by argparse via type=parse_position)
        auto_focus_pos = args.auto_focus_pos
        af_result = None

        # Build microscope metadata (reads all values back from hardware)
        micro_meta = build_microscope_meta(scope)
        cam_meta = micro_meta["camera"]
        optics_meta = micro_meta["optics"]

        # Validate frame size
        frame_width_um = optics_meta["frame_width_um"]
        frame_height_um = optics_meta["frame_height_um"]
        if frame_width_um is None or frame_height_um is None:
            print("Error: Could not determine frame size. Check objective/camera.")
            return 1

        # Build readout info string
        readout_fps = ""
        if cam_meta["readout_time_s"]:
            rt = cam_meta["readout_time_s"]
            readout_fps = f", Readout: {rt * 1000:.1f}ms ({1 / rt:.0f} fps)"

        print(f"Camera: {cam_meta['name']}")
        exp_str = f"{cam_meta['exposure_s'] * 1000:.1f}ms" if cam_meta["exposure_s"] else "?"
        binning = cam_meta["binning"]
        print(f"  Trigger: CONTINUOUS, Binning: {binning}x{binning}, Exposure: {exp_str}{readout_fps}")
        wb_bgr = cam_meta["white_balance_bgr"]
        print(f"  White balance (B,G,R): {wb_bgr[0]}, {wb_bgr[1]}, {wb_bgr[2]}")
        print(f"  Gamma: {cam_meta['gamma']}")
        print(f"  Frame: {cam_meta['frame_width_px']}x{cam_meta['frame_height_px']} px")
        print(f"  FOV: {frame_width_um:.2f} x {frame_height_um:.2f} µm")
        print(f"  Objective: {optics_meta['objective_mag']}x")
        if args.downsample > 1:
            print(f"  Downsample: {args.downsample}x")

        # Compute position-based frame skip threshold
        target_advance_um = frame_width_um * (1 - args.x_overlap_percent / 100)
        print(f"  X overlap target: {args.x_overlap_percent:.0f}% (advance {target_advance_um:.0f} µm between saves)")

        # Print lighting info
        light = micro_meta["lighting"]
        print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({light['lamp_intensity']}/{light['lamp_max_intensity']})")
        print(f"Shutter: {light['shutter_name']}, {'open' if light['shutter_open'] else 'closed'}")

        # Set up acquisition context
        from LeicaMicrosystems.HardwareModel import Extensions

        context = scope.context
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        # Run autofocus if position specified (before registering scan's image handler)
        if auto_focus_pos:
            from flakefinder.leica.autofocus import continuous_autofocus

            print(f"\nAutofocus at ({auto_focus_pos[0]:.0f}, {auto_focus_pos[1]:.0f}) µm...")

            # Move to autofocus position
            hx, hy = stage.move_to_async(auto_focus_pos[0], auto_focus_pos[1])
            wait_all([hx, hy])

            try:
                af_result = continuous_autofocus(
                    scope,
                    fine_pass=True,
                    super_fine_pass=True,
                )
                print(f"  Z: {af_result.initial_z_um:.1f} -> {af_result.selected_z_um:.1f} µm")
                print(
                    f"  Range: {af_result.z_range_um:.0f}µm, Sharpness: {af_result.initial_sharpness:.1f} -> {af_result.selected_sharpness:.1f}"  # noqa: E501
                )
            except ValueError as e:
                print(f"  Autofocus error: {e}")
                return 1

        # Register scan's image handler (after autofocus, which uses its own handler)
        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # Calculate scan area bounds (already validated pre-connection)
        if args.area_rect:
            x_min, x_max, y_min, y_max = parse_area_rect(args.area_rect)
        else:
            margin = args.margin
            x_min = stage.x.min_um + margin
            x_max = stage.x.max_um - margin
            y_min = stage.y.min_um + margin
            y_max = stage.y.max_um - margin

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Calculate row Y positions (top to bottom, -Y direction)
        y_step = frame_height_um * (1 - args.y_overlap_percent / 100)
        row_y_positions = []
        y = y_max
        while y >= y_min:
            row_y_positions.append(y)
            y -= y_step

        num_rows = len(row_y_positions)

        print(f"Scan area: X={x_min:.0f}-{x_max:.0f} µm, Y={y_min:.0f}-{y_max:.0f} µm")
        print(f"Frame: {frame_width_um:.1f} x {frame_height_um:.1f} µm")
        print(f"Y step: {y_step:.1f} µm ({args.y_overlap_percent:.0f}% overlap)")
        print(f"Rows: {num_rows}")
        print(f"Output: {args.output}/")
        print()

        # Move to start of first row
        print(f"Moving to start (X={x_min:.0f}, Y={row_y_positions[0]:.0f})...")
        hx, hy = stage.move_to_async(x_min, row_y_positions[0])
        wait_all([hx, hy])

        # Switch to scan speed for scanning
        actual_speed_mm = set_stage_speed(scan_speed_mm, "scan")

        # Initialize global metadata
        total_scan_start = time.perf_counter()
        all_position_samples = []  # All position samples across all rows
        global_frame_idx = 0  # Global frame counter

        # Background saver thread
        save_queue = queue.Queue()
        saved_frames_meta = []

        def saver_thread():
            """Background thread: convert, resize, and save frames."""
            while True:
                item = save_queue.get()
                if item is None:  # Poison pill
                    break

                frame_idx, row_idx, t_start, t_end, image, row_y, row_x_samples, t0 = item

                # Convert to numpy and dispose .NET image
                arr = sdk_image_to_numpy(image)
                image.Dispose()

                # Resize if needed
                img = PILImage.fromarray(arr)
                if args.downsample > 1:
                    new_size = (img.width // args.downsample, img.height // args.downsample)
                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                # Save
                path = os.path.join(args.output, f"frame_{frame_idx:04d}.jpg")
                img.save(path, quality=95)

                # Compute metadata
                x_start_interp = interpolate_position(t_start, row_x_samples)
                x_end_interp = interpolate_position(t_end, row_x_samples)
                dt = t_end - t_start
                x_vel = (x_end_interp - x_start_interp) / dt if dt > 0 and x_start_interp and x_end_interp else 0

                saved_frames_meta.append(
                    {
                        "n": frame_idx,
                        "row": row_idx,
                        "t_start": t_start - t0,
                        "t_end": t_end - t0,
                        "x_start": x_start_interp,
                        "x_end": x_end_interp,
                        "x_vel": x_vel,
                        "y_um": row_y,
                    }
                )

                save_queue.task_done()

        savers = []
        for _ in range(args.write_threads):
            t = threading.Thread(target=saver_thread, daemon=True)
            t.start()
            savers.append(t)

        meta = {
            "timestamp": datetime.now().isoformat(),
            "command": sys.argv,
            "x_min_um": x_min,
            "x_max_um": x_max,
            "y_min_um": y_min,
            "y_max_um": y_max,
            "y_step_um": y_step,
            "x_overlap_percent": args.x_overlap_percent,
            "target_advance_um": target_advance_um,
            "y_overlap_percent": args.y_overlap_percent,
            "downsample": args.downsample,
            # Scan parameters
            "scan_params": {
                "scan_speed_mm_s": actual_speed_mm,
                "move_speed_mm_s": move_speed_mm,
                "area_rect": args.area_rect,
                "margin_um": args.margin if not args.area_rect else None,
                "objective_mag_requested": args.objective_mag,
                "objective_pos_requested": args.objective_pos,
                "initial_z_um": args.z,
            },
            "autofocus": {
                "position_um": list(auto_focus_pos) if auto_focus_pos else None,
                "initial_z_um": af_result.initial_z_um if af_result else None,
                "selected_z_um": af_result.selected_z_um if af_result else None,
                "z_range_um": af_result.z_range_um if af_result else None,
                "initial_sharpness": af_result.initial_sharpness if af_result else None,
                "selected_sharpness": af_result.selected_sharpness if af_result else None,
                "final_sharpness": af_result.final_sharpness if af_result else None,
                "objective_position": af_result.objective_position if af_result else None,
                "scan_duration_s": af_result.scan_duration_s if af_result else None,
                "frame_count": af_result.frame_count if af_result else None,
            }
            if auto_focus_pos
            else None,
            # Microscope metadata (camera, optics, lighting)
            "camera": cam_meta,
            "optics": optics_meta,
            "lighting": micro_meta["lighting"],
            "rows": [],
        }

        # Scan each row
        for row_idx, row_y in enumerate(row_y_positions):
            # Snake pattern: even rows +X, odd rows -X
            direction = 1 if row_idx % 2 == 0 else -1
            if direction == 1:
                x_start_pos, x_end_pos = x_min, x_max
                dir_str = "+X"
            else:
                x_start_pos, x_end_pos = x_max, x_min
                dir_str = "-X"

            if not args.quiet:
                print(f"Row {row_idx}/{num_rows - 1}: Y={row_y:.0f}µm, {dir_str}")

            # Move to row start if not already there
            if row_idx > 0:
                hx, hy = stage.move_to_async(x_start_pos, row_y)
                wait_all([hx, hy])

            # Set up position polling for this row
            x_samples = []
            stop_polling = threading.Event()

            def x_poll_thread():
                while not stop_polling.is_set():
                    t_before = time.perf_counter()
                    x_native = x_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    x_um = x_converter.GetMetricsValue(x_native)
                    x_samples.append((t_before, t_after, x_um))

            # Start position polling
            x_thread = threading.Thread(target=x_poll_thread, daemon=True)
            x_thread.start()

            row_start = time.perf_counter()
            row_frame_start = global_frame_idx
            row_frame_count = 0
            row_capture_count = 0
            row_skip_count = 0
            last_saved_x = None

            # Warm up camera with a few captures before starting move
            for _ in range(args.warmup_frames):
                current_image[0] = None
                acquisition.Acquire(context, None)
                if current_image[0] is not None:
                    current_image[0].Dispose()

            # Start async X move
            handle = stage.x.move_to_async(x_end_pos)

            # Capture frames during move, queue to savers immediately
            while not handle.is_complete:
                t_start = time.perf_counter()
                current_image[0] = None
                acquisition.Acquire(context, None)
                t_end = time.perf_counter()

                if current_image[0] is not None:
                    row_capture_count += 1
                    x_now = x_samples[-1][2] if x_samples else None

                    # Position-based frame save/skip
                    if last_saved_x is not None and x_now is not None and abs(x_now - last_saved_x) < target_advance_um:
                        current_image[0].Dispose()
                        row_skip_count += 1
                    else:
                        # Queue frame for background saving
                        save_queue.put(
                            (
                                global_frame_idx,
                                row_idx,
                                t_start,
                                t_end,
                                current_image[0],
                                row_y,
                                x_samples,
                                total_scan_start,
                            )
                        )
                        if x_now is not None:
                            last_saved_x = x_now
                        global_frame_idx += 1
                        row_frame_count += 1

            row_end = time.perf_counter()
            handle.dispose()

            # Stop position polling
            stop_polling.set()
            x_thread.join(timeout=1.0)

            row_duration = row_end - row_start

            # Filter position samples to row scan period
            row_x_samples = [
                (t_before, t_after, x) for t_before, t_after, x in x_samples if row_start <= t_before <= row_end
            ]

            if not args.quiet:
                skip_str = f", {row_skip_count} skipped" if row_skip_count > 0 else ""
                print(
                    f"  {row_frame_count} frames ({row_capture_count} captured{skip_str}),"
                    f" {len(row_x_samples)} pos samples, {row_duration:.2f}s"
                )

            # Add position samples to global list (with adjusted timestamps)
            for t_before, t_after, x_um in row_x_samples:
                all_position_samples.append(
                    {
                        "t_before": t_before - total_scan_start,
                        "t_after": t_after - total_scan_start,
                        "x_um": x_um,
                        "row": row_idx,
                    }
                )

            # Record row metadata
            meta["rows"].append(
                {
                    "row_idx": row_idx,
                    "y_um": row_y,
                    "direction": direction,
                    "frame_start": row_frame_start,
                    "frame_end": global_frame_idx,
                    "captures": row_capture_count,
                    "skipped": row_skip_count,
                    "duration_s": row_duration,
                    "position_samples": len(row_x_samples),
                }
            )

        total_scan_end = time.perf_counter()
        total_duration = total_scan_end - total_scan_start

        # Return to center while saver finishes
        print()
        print("Returning to center...")
        set_stage_speed(move_speed_mm)  # Switch back to move speed
        return_handle_x = stage.x.move_to_async(x_center)
        return_handle_y = stage.y.move_to_async(y_center)

        # Wait for savers to finish
        print(f"Waiting for savers ({save_queue.qsize()} frames queued, {len(savers)} threads)...")
        for _ in savers:
            save_queue.put(None)  # Poison pill for each thread
        for t in savers:
            t.join()
        print(f"Savers done ({len(saved_frames_meta)} frames saved)")

        # Finalize metadata
        # Sort saved_frames_meta by frame index (may be out of order due to threading)
        saved_frames_meta.sort(key=lambda f: f["n"])
        meta["frames"] = saved_frames_meta
        meta["scan_duration_s"] = total_duration
        meta["frame_count"] = global_frame_idx
        total_captures = sum(r["captures"] for r in meta["rows"])
        total_skipped = sum(r["skipped"] for r in meta["rows"])
        meta["total_captures"] = total_captures
        meta["total_skipped"] = total_skipped
        meta["position_sample_count"] = len(all_position_samples)
        meta["position_stream"] = all_position_samples

        # Save metadata
        meta_path = os.path.join(args.output, "scan_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")

        # Compress if requested
        if args.compress:
            print(f"Creating {args.output}.zip...")
            shutil.make_archive(args.output, "zip", args.output)
            print(f"Created {args.output}.zip")

        # Summary stats
        print()
        print("=" * 50)
        print("SCAN SUMMARY:")
        print(f"  Total time: {total_duration:.1f}s")
        print(f"  Rows: {num_rows}")
        print(f"  Total frames saved: {global_frame_idx}")
        if total_skipped > 0:
            print(
                f"  Total captured: {total_captures}"
                f" ({total_skipped} skipped, {args.x_overlap_percent:.0f}% X overlap target)"
            )
        print(f"  Total position samples: {len(all_position_samples)}")
        print(f"  Avg FPS: {global_frame_idx / total_duration:.1f}")

        # Per-row stats
        if meta["rows"]:
            row_frame_counts = [r["frame_end"] - r["frame_start"] for r in meta["rows"]]
            row_durations = [r["duration_s"] for r in meta["rows"]]
            print(
                f"  Frames/row: avg={sum(row_frame_counts) / len(row_frame_counts):.0f}, "
                f"min={min(row_frame_counts)}, max={max(row_frame_counts)}"
            )
            print(f"  Row duration: avg={sum(row_durations) / len(row_durations):.2f}s")

        # Wait for return
        return_handle_x.wait()
        return_handle_y.wait()
        return_handle_x.dispose()
        return_handle_y.dispose()

        print()
        print("Done.")

        return 0


if __name__ == "__main__":
    exit(main())
