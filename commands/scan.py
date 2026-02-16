"""Multi-row snake scan with parallel position reading and image capture.

Supports both 5x overview scanning and 20x detection scanning for MaskTerial.
See docs/maskterial_integration.md for 20x scanning context.

Usage:
    uv run python commands/scan.py -o scan_5x --objective-mag 5
    uv run python commands/scan.py -o scan_20x --objective-mag 20x --z 24699 \\
        --area-rect 10000,60000,15000,55000
    uv run python commands/scan.py -o scan_5x --objective-mag 5 --dry-run
"""

import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from PIL import Image as PILImage

from flakefinder.data_utils import compute_frame_size_um, require_microscope_description
from flakefinder.image_utils import sdk_image_to_numpy
from flakefinder.leica import Microscope, continuous_autofocus, wait_all
from flakefinder.leica.polling import start_motion_polling
from flakefinder.scan_utils import (
    DEFAULT_WB,
    build_microscope_meta,
    interpolate_position,
    parse_area_rect,
    parse_position,
    parse_white_balance,
    validate_area_rect,
)
from flakefinder.types import AreaRect, GainRGB, Point2F


@dataclass
class _Preflight:
    """Validated inputs and computed scan plan from _plan()."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    frame_width_um: float
    frame_height_um: float
    y_step: float
    row_y_positions: list[float]
    target_advance_um: float

    def print_summary(
        self,
        *,
        speed_mm: float,
        move_speed_mm: float,
        x_overlap_percent: float,
        y_overlap_percent: float,
        objective_mag: str | None,
        binning: int,
        downsample: int,
    ) -> None:
        num_rows = len(self.row_y_positions)
        x_dist_mm = (self.x_max - self.x_min) / 1000
        total_distance_mm = x_dist_mm * num_rows
        est_time_s = total_distance_mm / speed_mm + num_rows * 0.5

        print("Area Scan v1 (Snake Pattern)")
        print("=" * 50)
        print(f"Scan area: X={self.x_min:.0f}-{self.x_max:.0f} µm, Y={self.y_min:.0f}-{self.y_max:.0f} µm")
        if objective_mag:
            print(f"Objective: {objective_mag}x, Binning: {binning}x{binning}")
        print(f"Frame FOV: {self.frame_width_um:.1f} x {self.frame_height_um:.1f} µm")
        if downsample > 1:
            print(f"Downsample: {downsample}x")
        print(f"X overlap: {x_overlap_percent:.0f}% (advance {self.target_advance_um:.0f} µm between saves)")
        print(f"Y step: {self.y_step:.1f} µm ({y_overlap_percent:.0f}% overlap)")
        print(f"Rows: {num_rows}")
        print(f"Scan speed: {speed_mm:.1f} mm/s, Move speed: {move_speed_mm:.1f} mm/s")
        print(f"Row X distance: {x_dist_mm:.1f} mm")
        print(f"Total scan distance: {total_distance_mm:.1f} mm")
        print(f"Estimated time: ~{est_time_s:.0f}s")
        print()


def _plan(
    *,
    area_rect: AreaRect | None = None,
    margin: float = 1000,
    objective_mag: str | None = None,
    binning: int = 3,
    x_overlap_percent: float = 100,
    y_overlap_percent: float = 12,
) -> _Preflight:
    """Compute and validate scan plan (pure computation, no hardware).

    Raises:
        ValueError: On invalid inputs.
    """
    desc = require_microscope_description()

    # Resolve scan area
    if area_rect:
        x_min, x_max, y_min, y_max = area_rect
    else:
        x_min = desc.stage.x.min_um + margin
        x_max = desc.stage.x.max_um - margin
        y_min = desc.stage.y.min_um + margin
        y_max = desc.stage.y.max_um - margin

    # Resolve frame size from microscope description
    if objective_mag is None:
        raise ValueError("--objective-mag is required for --dry-run")

    mag_str = objective_mag.lower().rstrip("x")
    obj_mag_float: float | None = None
    for obj in desc.objectives.values():
        if str(obj.magnification) == mag_str:
            obj_mag_float = obj.magnification
            break
    if obj_mag_float is None:
        raise ValueError(f"Unknown objective magnification '{objective_mag}'")

    binning_idx = binning - 1
    frame_size = compute_frame_size_um(desc.camera, obj_mag_float, binning_idx)
    if frame_size is None:
        raise ValueError("Could not compute frame size from microscope description")
    frame_width_um, frame_height_um = frame_size

    # Row Y positions
    y_step = frame_height_um * (1 - y_overlap_percent / 100)
    row_y_positions: list[float] = []
    y = y_min
    while y <= y_max:
        row_y_positions.append(y)
        y += y_step

    # Target X advance
    target_advance_um = frame_width_um * (1 - x_overlap_percent / 100)

    return _Preflight(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        frame_width_um=frame_width_um,
        frame_height_um=frame_height_um,
        y_step=y_step,
        row_y_positions=row_y_positions,
        target_advance_um=target_advance_um,
    )


def run(
    scope: Microscope,
    *,
    output: str,
    margin: float = 1000,
    area_rect: AreaRect | None = None,
    objective_mag: str | None = None,
    speed_mm: float = 40,
    move_speed_mm: float = 40,
    initial_z: float | None = None,
    auto_focus_pos: Point2F | None = None,
    x_overlap_percent: float = 100,
    y_overlap_percent: float = 12,
    downsample: int = 1,
    white_balance: GainRGB = DEFAULT_WB,
    gamma: float = 1.0,
    binning: int = 3,
    exposure_ms: float = 1.0,
    gain: float = 1.0,
    warmup_frames: int = 3,
    compress: bool = False,
    clean: bool = False,
    write_threads: int = 2,
    quiet: bool = False,
) -> None:
    """Multi-row snake scan.

    Raises:
        ValueError: On invalid inputs or scan failure.
    """
    # Convert binning to SDK index (1/2/3 -> 0/1/2)
    binning_idx = binning - 1

    # Check/create output directory before connecting to hardware
    if os.path.exists(output):
        if clean:
            shutil.rmtree(output)
        else:
            raise ValueError(f"Output directory '{output}' already exists. Use --clean to wipe it.")
    os.makedirs(output)

    print("Area Scan v1 (Snake Pattern)")
    print("=" * 50)

    stage = scope.stage
    z = scope.z
    x_bcv = stage.x.bcv  # Native position reader
    x_converter = stage.x.converter  # For native -> um conversion

    if area_rect:
        validate_area_rect(area_rect, stage)

    # Switch objective if requested
    if objective_mag is not None:
        if scope.switch_objective_mag(objective_mag):
            print(f"Switched objective to {scope.objective_mag}x")
        else:
            print(f"Objective: already at {scope.objective_mag}x")

    # Move Z if requested (after objective switch, before scan)
    if initial_z is not None:
        current_z = z.position_um
        print(f"Moving Z: {current_z:.1f} -> {initial_z:.1f} µm...")
        z.move_to_corrected(initial_z)
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
    scan_speed_mm = speed_mm

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
    camera.exposure_time = exposure_ms / 1000.0
    camera.gain = gain
    camera.gain_rgb = white_balance
    camera.gamma = gamma

    af_result = None

    # Build microscope metadata (reads all values back from hardware)
    micro_meta = build_microscope_meta(scope)
    cam_meta = micro_meta["camera"]
    optics_meta = micro_meta["optics"]

    # Validate frame size
    frame_width_um = optics_meta["frame_width_um"]
    frame_height_um = optics_meta["frame_height_um"]
    if frame_width_um is None or frame_height_um is None:
        raise ValueError("Could not determine frame size. Check objective/camera.")

    # Build readout info string
    readout_fps = ""
    if cam_meta["readout_time_s"]:
        rt = cam_meta["readout_time_s"]
        readout_fps = f", Readout: {rt * 1000:.1f}ms ({1 / rt:.0f} fps)"

    print(f"Camera: {cam_meta['name']}")
    exp_str = f"{cam_meta['exposure_s'] * 1000:.1f}ms" if cam_meta["exposure_s"] else "?"
    binning_actual = cam_meta["binning"]
    print(f"  Trigger: CONTINUOUS, Binning: {binning_actual}x{binning_actual}, Exposure: {exp_str}{readout_fps}")
    wb_bgr = cam_meta["white_balance_bgr"]
    print(f"  White balance (B,G,R): {wb_bgr[0]}, {wb_bgr[1]}, {wb_bgr[2]}")
    print(f"  Gamma: {cam_meta['gamma']}")
    print(f"  Frame: {cam_meta['frame_width_px']}x{cam_meta['frame_height_px']} px")
    print(f"  FOV: {frame_width_um:.2f} x {frame_height_um:.2f} µm")
    print(f"  Objective: {optics_meta['objective_mag']}x")
    if downsample > 1:
        print(f"  Downsample: {downsample}x")

    # Compute position-based frame skip threshold
    target_advance_um = frame_width_um * (1 - x_overlap_percent / 100)
    print(f"  X overlap target: {x_overlap_percent:.0f}% (advance {target_advance_um:.0f} µm between saves)")

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
            raise ValueError(f"Autofocus failed: {e}") from e

    # Register scan's image handler (after autofocus, which uses its own handler)
    context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

    # Calculate scan area bounds
    if area_rect:
        x_min, x_max, y_min, y_max = area_rect
    else:
        x_min = stage.x.min_um + margin
        x_max = stage.x.max_um - margin
        y_min = stage.y.min_um + margin
        y_max = stage.y.max_um - margin

    # Calculate row Y positions (low Y to high Y, +Y direction)
    y_step = frame_height_um * (1 - y_overlap_percent / 100)
    row_y_positions = []
    y = y_min
    while y <= y_max:
        row_y_positions.append(y)
        y += y_step

    num_rows = len(row_y_positions)

    print(f"Scan area: X={x_min:.0f}-{x_max:.0f} µm, Y={y_min:.0f}-{y_max:.0f} µm")
    print(f"Frame: {frame_width_um:.1f} x {frame_height_um:.1f} µm")
    print(f"Y step: {y_step:.1f} µm ({y_overlap_percent:.0f}% overlap)")
    print(f"Rows: {num_rows}")
    print(f"Output: {output}/")
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
            if downsample > 1:
                new_size = (img.width // downsample, img.height // downsample)
                img = img.resize(new_size, PILImage.Resampling.LANCZOS)

            # Save
            path = os.path.join(output, f"frame_{frame_idx:04d}.jpg")
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
    for _ in range(write_threads):
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
        "x_overlap_percent": x_overlap_percent,
        "target_advance_um": target_advance_um,
        "y_overlap_percent": y_overlap_percent,
        "downsample": downsample,
        # Scan parameters
        "scan_params": {
            "scan_speed_mm_s": actual_speed_mm,
            "move_speed_mm_s": move_speed_mm,
            "area_rect": area_rect,
            "margin_um": margin if not area_rect else None,
            "objective_mag_requested": objective_mag,
            "initial_z_um": initial_z,
        },
        "autofocus": {
            "position_um": list(af_result.position_um),
            "initial_z_um": af_result.initial_z_um,
            "selected_z_um": af_result.selected_z_um,
            "z_range_um": af_result.z_range_um,
            "initial_sharpness": af_result.initial_sharpness,
            "selected_sharpness": af_result.selected_sharpness,
            "final_sharpness": af_result.final_sharpness,
            "objective_position": af_result.objective_position,
            "scan_duration_s": af_result.scan_duration_s,
            "frame_count": af_result.frame_count,
        }
        if af_result
        else None,
        **micro_meta,
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

        if not quiet:
            print(f"Row {row_idx}/{num_rows - 1}: Y={row_y:.0f}µm, {dir_str}")

        # Move to row start if not already there
        if row_idx > 0:
            hx, hy = stage.move_to_async(x_start_pos, row_y)
            wait_all([hx, hy])

        row_start = time.perf_counter()
        row_frame_start = global_frame_idx
        row_frame_count = 0
        row_capture_count = 0
        row_skip_count = 0
        last_saved_x = None

        # Warm up camera with a few captures before starting move
        for _ in range(warmup_frames):
            current_image[0] = None
            acquisition.Acquire(context, None)
            if current_image[0] is not None:
                current_image[0].Dispose()

        # Start polling at low Hz, issue move, polling ramps to full speed on motion
        x_polling = start_motion_polling(x_bcv, x_converter)
        handle = stage.x.move_to_async(x_end_pos)
        # TODO: derive stop margin from frame size or commanded speed
        _STOP_MARGIN_UM = 50.0
        row_distance_um = abs(x_end_pos - x_start_pos)
        row_timeout_s = max(30.0, (row_distance_um / (actual_speed_mm * 1000)) * 5)

        # Capture frames during move, using position to detect arrival
        # (avoids calling GetState on the same axis as polling — see
        # docs/poll_throttling_plan.md for starvation background)
        while True:
            t_start = time.perf_counter()
            current_image[0] = None
            acquisition.Acquire(context, None)
            t_end = time.perf_counter()

            if current_image[0] is not None:
                row_capture_count += 1
                x_now = x_polling.samples[-1].axis_um if x_polling.samples else None

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
                            x_polling.samples,
                            total_scan_start,
                        )
                    )
                    if x_now is not None:
                        last_saved_x = x_now
                    global_frame_idx += 1
                    row_frame_count += 1

                # Stop when stage reaches target (frees bus for SDK move completion)
                if x_now is not None and (
                    (direction == 1 and x_now >= x_end_pos - _STOP_MARGIN_UM)
                    or (direction == -1 and x_now <= x_end_pos + _STOP_MARGIN_UM)
                ):
                    break

            # Safety timeout
            if time.perf_counter() - row_start > row_timeout_s:
                if not quiet:
                    print(f"  WARNING: row {row_idx} timeout ({row_timeout_s:.0f}s)")
                break

        row_end = time.perf_counter()

        # Stop polling to free bus, then wait for SDK move completion
        x_polling.join()
        if not handle.is_complete:
            handle.wait(timeout=2.0)
        handle.dispose()

        row_duration = row_end - row_start

        # Filter position samples to row scan period
        row_x_samples = [s for s in x_polling.samples if row_start <= s.t_before <= row_end]

        if not quiet:
            skip_str = f", {row_skip_count} skipped" if row_skip_count > 0 else ""
            print(
                f"  {row_frame_count} frames ({row_capture_count} captured{skip_str}),"
                f" {len(row_x_samples)} pos samples, {row_duration:.2f}s"
            )

        # Add position samples to global list (with adjusted timestamps)
        for s in row_x_samples:
            all_position_samples.append(
                {
                    "t_before": s.t_before - total_scan_start,
                    "t_after": s.t_after - total_scan_start,
                    "x_um": s.axis_um,
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

    # Restore move speed (scan leaves it at scan speed)
    set_stage_speed(move_speed_mm)

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
    meta_path = os.path.join(output, "scan_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    # Compress if requested
    if compress:
        print(f"Creating {output}.zip...")
        shutil.make_archive(output, "zip", output)
        print(f"Created {output}.zip")

    # Summary stats
    print()
    print("=" * 50)
    print("SCAN SUMMARY:")
    print(f"  Total time: {total_duration:.1f}s")
    print(f"  Rows: {num_rows}")
    print(f"  Total frames saved: {global_frame_idx}")
    if total_skipped > 0:
        print(
            f"  Total captured: {total_captures} ({total_skipped} skipped, {x_overlap_percent:.0f}% X overlap target)"
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

    print()
    print("Done.")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Multi-row snake scan with configurable objective and area",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5x overview scan of full stage area (default)
  uv run python commands/scan.py -o scan_5x --objective-mag 5

  # 20x detection scan for MaskTerial over specific area
  uv run python commands/scan.py -o scan_20x --objective-mag 20x --z 24699 --area-rect 10000,60000,15000,55000
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
        type=parse_area_rect,
        metavar="X1,X2,Y1,Y2",
        help="Explicit scan area as x_min,x_max,y_min,y_max in µm",
    )

    area_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scan plan and exit (no hardware)",
    )

    # Objective/optics options
    optics_group = parser.add_argument_group("Optics")
    optics_group.add_argument(
        "--objective-mag",
        type=str,
        metavar="MAG",
        help="Objective by magnification (e.g., 5, 5x, 20, 2.5) - switches before scan",
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
        default=100,
        help="Target X overlap between saved frames, %% (default: 100 = save all). "
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
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.dry_run:
        try:
            p = _plan(
                area_rect=args.area_rect,
                margin=args.margin,
                objective_mag=args.objective_mag,
                binning=args.binning,
                x_overlap_percent=args.x_overlap_percent,
                y_overlap_percent=args.y_overlap_percent,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        p.print_summary(
            speed_mm=args.speed_mm,
            move_speed_mm=args.move_speed_mm,
            x_overlap_percent=args.x_overlap_percent,
            y_overlap_percent=args.y_overlap_percent,
            objective_mag=args.objective_mag,
            binning=args.binning,
            downsample=args.downsample,
        )
        print("(dry run -- exiting)")
        return 0

    try:
        with Microscope() as scope:
            run(
                scope,
                output=args.output,
                margin=args.margin,
                area_rect=args.area_rect,
                objective_mag=args.objective_mag,
                speed_mm=args.speed_mm,
                move_speed_mm=args.move_speed_mm,
                initial_z=args.z,
                auto_focus_pos=args.auto_focus_pos,
                x_overlap_percent=args.x_overlap_percent,
                y_overlap_percent=args.y_overlap_percent,
                downsample=args.downsample,
                white_balance=args.white_balance,
                gamma=args.gamma,
                binning=args.binning,
                exposure_ms=args.exposure_ms,
                gain=args.gain,
                warmup_frames=args.warmup_frames,
                compress=args.compress,
                clean=args.clean,
                write_threads=args.write_threads,
                quiet=args.quiet,
            )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
