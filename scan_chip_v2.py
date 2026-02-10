"""Chip-aware multi-row scan with focus plane Z tracking (v2).

Scans only within a chip's convex hull polygon, tracking Z with a plane fit.
Each row's X extent is clipped to the polygon intersection, and Z velocity
is set to match the plane slope during X motion.

Key improvement over v1: Z tracking starts inside the capture loop when X
approaches the real chip edge, eliminating the Z-jump artifact caused by
early Z velocity start.

Run on microscope PC.

Usage:
    python scan_chip_v2.py -o scans/chip0_20x \
        --chips-meta scans/working_overview_5x_stitch_chips.json \
        --chip 0 \
        --plane scans/focus_map_chip0_v10_no_overshoot_plane.json \
        --objective-mag 20x --speed-mm 5

    # Test with just 3 rows
    python scan_chip_v2.py -o scans/chip0_test \
        --chips-meta scans/chips.json --chip 0 --plane scans/plane.json \
        --objective-mag 20x --row-limit 3
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from flakefinder.data_utils import (
    compute_frame_size_um,
    load_chip_geometry,
    require_microscope_description,
)
from flakefinder.scan_utils import (
    build_microscope_meta,
    compute_planar_scan_plan,
    compute_plane_z,
    interpolate_position,
    parse_white_balance,
)
from flakefinder.types import PositionSample


def interpolate_z_position(t: float, z_samples: list[tuple[float, float]]) -> float | None:
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


def measure_x_cruise_speed(
    x_samples: list[PositionSample],
    t_x_started: float,
    t_deadline: float,
    commanded_speed_um_s: float,
    vel_tolerance: float = 0.5,
) -> tuple[float | None, dict[str, Any]]:
    """Measure actual X cruise speed from position samples via linear regression.

    Filters samples to the cruise velocity band: only consecutive pairs whose
    inter-sample speed is within ±vel_tolerance of commanded_speed_um_s are
    kept.  This eliminates both stationary samples (stage hasn't started moving)
    and acceleration-zone samples without relying on a fixed time offset.

    Returns:
        (measured_speed_um_s, measurement_dict) where speed is always positive
        and measurement_dict contains logging info.  Returns (None, {}) if
        insufficient data after filtering.
    """
    # Window to samples between move start and deadline
    window = [s for s in x_samples if t_x_started <= s.t_before < t_deadline]

    if len(window) < 2:
        return None, {}

    # Velocity filter: keep samples where inter-sample speed is within
    # ±vel_tolerance of commanded speed
    speed_lo = commanded_speed_um_s * (1 - vel_tolerance)
    speed_hi = commanded_speed_um_s * (1 + vel_tolerance)
    cruise_mask = [False] * len(window)
    for i in range(1, len(window)):
        dt = window[i].t_before - window[i - 1].t_before
        if dt <= 0:
            continue
        speed = abs(window[i].x_um - window[i - 1].x_um) / dt
        if speed_lo <= speed <= speed_hi:
            cruise_mask[i - 1] = True
            cruise_mask[i] = True

    cruise_samples = [s for s, keep in zip(window, cruise_mask, strict=True) if keep]
    n_filtered = len(window) - len(cruise_samples)

    if len(cruise_samples) < 5:
        return None, {}

    times = np.array([s.t_before for s in cruise_samples])
    positions = np.array([s.x_um for s in cruise_samples])

    t_rel = times - times[0]
    coeffs = np.polyfit(t_rel, positions, 1)
    speed_um_s = abs(coeffs[0])

    # R²
    fitted = np.polyval(coeffs, t_rel)
    ss_res = float(np.sum((positions - fitted) ** 2))
    ss_tot = float(np.sum((positions - np.mean(positions)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    measurement = {
        "measured_x_speed_um_s": round(speed_um_s, 2),
        "n_samples": len(cruise_samples),
        "n_total": len(window),
        "n_filtered": n_filtered,
        "measurement_window_ms": round((times[-1] - times[0]) * 1000, 1),
        "r_squared": round(r_squared, 6),
    }

    return speed_um_s, measurement


# ============================================================================
# scan_row types and function
# ============================================================================


@dataclass
class ScanHardware:
    """Hardware interfaces needed for row scanning."""

    stage: Any  # Stage
    z_drive: Any  # ZDrive
    acquisition: Any  # SDK acquisition
    acq_context: Any  # SDK image acquisition context
    current_image: list  # [None] mutable ref for image callback
    x_bcv: Any  # fast X position reader
    x_converter: Any  # native -> um converter
    z_converter: Any  # native -> um converter
    z_bcv_hysteresis: Any  # hysteresis-corrected Z reader


@dataclass
class RowConfig:
    """Scan configuration for a single row."""

    plane_a: float
    plane_b: float
    plane_c: float
    x_speed_um_s: float
    move_speed_um_s: float
    target_advance_um: float
    lead_in_um: float
    z_lead_s: float
    row_settle_s: float
    warmup_frames: int
    z_max: float
    quiet: bool


@dataclass
class RowResult:
    """Results from scanning a single row."""

    frames_saved: int
    frames_skipped: int
    skipped: bool  # True if row was safety-skipped
    timing_s: dict[str, float | None]  # timestamp dict (see plan doc)
    position_samples: list[dict[str, Any]]  # [{t_before, t_after, x_um, row}]
    z_start_x_um: float | None
    speed_measurement: dict[str, Any] | None  # X speed measurement from lead-in


def scan_row(
    row_idx: int,
    row_y: float,
    row_x_min: float,
    row_x_max: float,
    direction: int,
    total_rows: int,
    *,
    hw: ScanHardware,
    cfg: RowConfig,
    save_queue: queue.Queue,
    global_frame_idx: int,
    scan_t0: float,
) -> RowResult:
    """Scan a single row with focus plane Z tracking.

    Z tracking starts inside the capture loop when X approaches the real
    chip edge, using the lead-in distance as a run-up region.

    Args:
        row_idx: Row index (for metadata and printing).
        row_y: Y position of this row (um).
        row_x_min: Hull-clipped X min (um, unpadded).
        row_x_max: Hull-clipped X max (um, unpadded).
        direction: +1 for +X, -1 for -X.
        total_rows: Total number of rows (for printing).
        hw: Hardware interfaces.
        cfg: Scan configuration.
        save_queue: Queue for frame saving (producer interface).
        global_frame_idx: Starting frame index for this row.
        scan_t0: perf_counter at scan start (for timestamp reference).

    Returns:
        RowResult with frame counts, timing, and position data.
    """
    stage = hw.stage
    z_drive = hw.z_drive

    # ---- Row geometry ----
    if direction == 1:
        chip_edge_x = row_x_min
        x_start_pos = row_x_min - cfg.lead_in_um
        x_end_pos = row_x_max
        dir_str = "+X"
    else:
        chip_edge_x = row_x_max
        x_start_pos = row_x_max + cfg.lead_in_um
        x_end_pos = row_x_min
        dir_str = "-X"

    # Z at chip edge (where tracking begins) and scan end
    z_start = compute_plane_z(cfg.plane_a, cfg.plane_b, cfg.plane_c, chip_edge_x, row_y)
    z_end = compute_plane_z(cfg.plane_a, cfg.plane_b, cfg.plane_c, x_end_pos, row_y)

    # Z velocity: dZ/dt = plane_a * direction * x_speed
    # Initial estimate from commanded speed; corrected below from measured X speed
    z_vel_um_s = cfg.plane_a * direction * cfg.x_speed_um_s
    speed_measurement: dict[str, Any] | None = None

    row_width_mm = (row_x_max - row_x_min) / 1000
    if not cfg.quiet:
        print(
            f"Row {row_idx}/{total_rows - 1}: Y={row_y:.0f}um, {dir_str}, "
            f"X=[{row_x_min:.0f},{row_x_max:.0f}] ({row_width_mm:.1f}mm), "
            f"Z={z_start:.0f}->{z_end:.0f}"
        )

    # Runtime Z safety check
    if max(z_start, z_end) > cfg.z_max:
        print(f"  SKIP: Z would exceed limit ({max(z_start, z_end):.0f} > {cfg.z_max:.0f})")
        return RowResult(
            frames_saved=0,
            frames_skipped=0,
            skipped=True,
            timing_s={},
            position_samples=[],
            z_start_x_um=None,
            speed_measurement=None,
        )

    # ---- 1. Parallel preposition ----
    t_preposition_start = time.perf_counter()

    hy = stage.y.move_to_async(row_y)
    hx = stage.x.move_to_async(x_start_pos)
    # Z corrected is blocking — runs concurrently while X/Y are async
    z_drive.move_to_corrected(z_start)
    hx.wait()
    hy.wait()
    hx.dispose()
    hy.dispose()

    t_preposition_end = time.perf_counter()

    # ---- 2. Settle ----
    time.sleep(cfg.row_settle_s)
    t_settle_end = time.perf_counter()

    # ---- 3. Start per-row polling threads ----
    x_samples: list[PositionSample] = []
    z_samples: list[tuple[float, float]] = []
    stop_polling = threading.Event()

    x_bcv = hw.x_bcv
    x_converter = hw.x_converter
    z_converter = hw.z_converter
    z_bcv_hysteresis = hw.z_bcv_hysteresis

    def x_poll_fn() -> None:
        while not stop_polling.is_set():
            t_before = time.perf_counter()
            x_native = x_bcv.GetControlValue()
            t_after = time.perf_counter()
            x_um = x_converter.GetMetricsValue(x_native)
            x_samples.append(PositionSample(t_before, t_after, x_um))

    def z_poll_fn() -> None:
        while not stop_polling.is_set():
            t = time.perf_counter()
            z_native = z_bcv_hysteresis.GetControlValue()
            z_um = z_converter.GetMetricsValue(z_native)
            z_samples.append((t, z_um))

    x_thread = threading.Thread(target=x_poll_fn, daemon=True)
    z_thread = threading.Thread(target=z_poll_fn, daemon=True)
    x_thread.start()
    z_thread.start()

    # ---- 4. Warmup camera ----
    for _ in range(cfg.warmup_frames):
        hw.current_image[0] = None
        hw.acquisition.Acquire(hw.acq_context, None)
        if hw.current_image[0] is not None:
            hw.current_image[0].Dispose()
    t_warmup_end = time.perf_counter()

    # ---- 5. Start X motion ----
    stage.x.set_velocity_um_s(cfg.x_speed_um_s)
    handle = stage.x.move_to_async(x_end_pos)
    t_x_started = time.perf_counter()

    # ---- 6. Capture loop (owns Z start) ----
    row_frame_count = 0
    row_skip_count = 0
    last_saved_x = None
    z_started = False
    t_z_started = None
    z_start_x_um = None

    while not handle.is_complete:
        t_start = time.perf_counter()
        hw.current_image[0] = None
        hw.acquisition.Acquire(hw.acq_context, None)
        t_end = time.perf_counter()

        if hw.current_image[0] is not None:
            x_now = x_samples[-1].x_um if x_samples else None

            # Start Z tracking when X approaches chip edge
            if not z_started and x_now is not None:
                signed_dist = (chip_edge_x - x_now) * direction
                if signed_dist / cfg.x_speed_um_s <= cfg.z_lead_s:
                    # Measure actual X cruise speed from lead-in samples
                    measured_speed, speed_meas = measure_x_cruise_speed(
                        x_samples, t_x_started, time.perf_counter(), cfg.x_speed_um_s
                    )
                    if measured_speed is not None:
                        z_vel_um_s = cfg.plane_a * direction * measured_speed
                        speed_meas["commanded_x_speed_um_s"] = cfg.x_speed_um_s
                        speed_meas["z_vel_commanded_um_s"] = round(abs(cfg.plane_a * cfg.x_speed_um_s), 4)
                        speed_meas["z_vel_corrected_um_s"] = round(abs(cfg.plane_a * measured_speed), 4)
                        speed_measurement = speed_meas

                    if abs(z_vel_um_s) > 0.1:
                        if z_vel_um_s > 0:
                            z_drive.start_towards_max(abs(z_vel_um_s))
                        else:
                            z_drive.start_towards_min(abs(z_vel_um_s))
                    z_started = True
                    t_z_started = time.perf_counter()
                    z_start_x_um = x_now

            # Position-based frame save/skip
            if last_saved_x is not None and x_now is not None and abs(x_now - last_saved_x) < cfg.target_advance_um:
                hw.current_image[0].Dispose()
                row_skip_count += 1
            else:
                save_queue.put(
                    (
                        global_frame_idx,
                        row_idx,
                        t_start,
                        t_end,
                        hw.current_image[0],
                        row_y,
                        list(x_samples),
                        list(z_samples),
                        scan_t0,
                        chip_edge_x,
                        direction,
                    )
                )
                if x_now is not None:
                    last_saved_x = x_now
                global_frame_idx += 1
                row_frame_count += 1

    t_capture_end = time.perf_counter()
    handle.dispose()

    # ---- 7. Cleanup ----
    z_drive.halt()
    stop_polling.set()
    x_thread.join(timeout=1.0)
    z_thread.join(timeout=1.0)

    # Restore move speed for positioning to next row
    stage.x.set_velocity_um_s(cfg.move_speed_um_s)

    # ---- Collect results ----
    row_duration = t_capture_end - t_x_started

    # Filter position samples to capture period
    row_x_samples = [(tb, ta, x) for tb, ta, x in x_samples if t_x_started <= tb <= t_capture_end]

    # Build timing dict (all relative to scan_t0)
    timing_s: dict[str, float | None] = {
        "preposition_start": t_preposition_start - scan_t0,
        "preposition_end": t_preposition_end - scan_t0,
        "settle_end": t_settle_end - scan_t0,
        "warmup_end": t_warmup_end - scan_t0,
        "x_started": t_x_started - scan_t0,
        "z_started": (t_z_started - scan_t0) if t_z_started is not None else None,
        "z_start_x_um": z_start_x_um,
        "capture_end": t_capture_end - scan_t0,
    }

    # Console summary
    if not cfg.quiet:
        prepos_ms = 1000 * (t_preposition_end - t_preposition_start)
        settle_ms = 1000 * (t_settle_end - t_preposition_end)
        warmup_ms = 1000 * (t_warmup_end - t_settle_end)
        x_start_ms = 1000 * (t_x_started - t_warmup_end)
        z_wait_ms = 1000 * (t_z_started - t_x_started) if t_z_started else 0
        z_start_str = f"Zstart={z_wait_ms:.0f}ms @X={z_start_x_um:.0f}" if z_start_x_um else "Zstart=N/A"

        print(f"  {row_frame_count} saved, {row_skip_count} skipped, {len(row_x_samples)} pos, {row_duration:.2f}s")
        print(
            f"  Startup: prepos={prepos_ms:.0f}ms settle={settle_ms:.0f}ms "
            f"warmup={warmup_ms:.0f}ms Xstart={x_start_ms:.0f}ms {z_start_str}"
        )
        if speed_measurement:
            ms = speed_measurement
            pct = (ms["measured_x_speed_um_s"] - cfg.x_speed_um_s) / cfg.x_speed_um_s * 100
            print(
                f"  X speed: {ms['measured_x_speed_um_s']:.1f} µm/s "
                f"(cmd {cfg.x_speed_um_s:.0f}, {pct:+.2f}%), "
                f"Z vel: {ms['z_vel_commanded_um_s']:.2f} → {ms['z_vel_corrected_um_s']:.2f} µm/s"
            )

    # Build position samples for global list
    position_samples = [
        {
            "t_before": tb - scan_t0,
            "t_after": ta - scan_t0,
            "x_um": x_um,
            "row": row_idx,
        }
        for tb, ta, x_um in row_x_samples
    ]

    return RowResult(
        frames_saved=row_frame_count,
        frames_skipped=row_skip_count,
        skipped=False,
        timing_s=timing_s,
        position_samples=position_samples,
        z_start_x_um=z_start_x_um,
        speed_measurement=speed_measurement,
    )


# ============================================================================
# main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chip-aware scan with focus plane Z tracking (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan chip 0 at 20x with focus plane
  python scan_chip_v2.py -o scans/chip0_20x \\
      --chips-meta scans/chips.json --chip 0 \\
      --plane scans/plane.json --objective-mag 20x

  # Objective by turret position (position 3 = 20x)
  python scan_chip_v2.py -o scans/chip0_20x \\
      --chips-meta scans/chips.json --chip 0 \\
      --plane scans/plane.json --objective-pos 3

  # Test with 3 rows
  python scan_chip_v2.py -o scans/chip0_test \\
      --chips-meta scans/chips.json --chip 0 \\
      --plane scans/plane.json --objective-mag 20x --row-limit 3
""",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    # Chip and plane
    chip_group = parser.add_argument_group("Chip and focus")
    chip_group.add_argument("--chips-meta", type=str, required=True, help="Path to chips JSON from find_chips.py")
    chip_group.add_argument("--chip", type=int, required=True, help="Chip index (from chips JSON)")
    chip_group.add_argument("--plane", type=str, required=True, help="Path to plane JSON from analyze_focus_map.py")

    # Scan control
    scan_group = parser.add_argument_group("Scan control")
    scan_group.add_argument(
        "--x-overlap-percent",
        type=float,
        default=30,
        help="Target X overlap between saved frames, %% (default: 30). "
        "Frames captured before advancing enough are discarded.",
    )
    scan_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scan plan and row summary, then exit (no hardware)",
    )
    scan_group.add_argument("--row-limit", type=int, default=None, help="Scan only N rows then stop (for testing)")
    scan_group.add_argument(
        "--row-settle",
        type=float,
        default=0.1,
        help="Settle time in seconds after pre-position (default: 0.1)",
    )
    scan_group.add_argument(
        "--lead-in-um",
        type=float,
        default=2000,
        help="Lead-in distance before chip edge in um (default: 2000). "
        "X starts this far before the hull boundary so it reaches scan "
        "speed before Z tracking begins.",
    )
    scan_group.add_argument(
        "--z-lead-ms",
        type=float,
        default=30,
        help="Start Z tracking this many ms before X reaches chip edge (default: 30)",
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
        "--objective-pos",
        type=int,
        metavar="POS",
        help="Objective by turret position (1-6) - switches before scan",
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
    frame_group.add_argument(
        "--white-balance", type=parse_white_balance, default="2.51,1.02,1.41", help="White balance as B,G,R gains"
    )
    frame_group.add_argument("--gamma", type=float, default=1.0, help="Gamma (default: 1.0)")
    frame_group.add_argument("--downsample", type=int, default=1, help="Downsample factor")
    frame_group.add_argument(
        "--warmup-frames", type=int, default=3, help="Warmup captures before each row (default: 3)"
    )
    frame_group.add_argument(
        "--y-overlap-percent",
        type=float,
        default=12,
        help="Y overlap between rows as %% of frame height (default: 12)",
    )

    # Safety
    safety_group = parser.add_argument_group("Safety")
    safety_group.add_argument("--z-max", type=float, default=26000.0, help="Hard Z limit in um (default: 26000)")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--clean", action="store_true", help="Wipe output directory if exists")
    output_group.add_argument("--write-threads", type=int, default=2, help="Image writer threads")
    output_group.add_argument("--compress", action="store_true", help="Create .zip of output")
    output_group.add_argument("--quiet", action="store_true", help="Suppress per-row progress output")

    args = parser.parse_args()

    # Validate objective args are mutually exclusive
    if args.objective_mag is not None and args.objective_pos is not None:
        print("Error: --objective-mag and --objective-pos are mutually exclusive")
        return 1

    # ---- Load chip data ----
    chips_path = Path(args.chips_meta)
    if not chips_path.exists():
        print(f"Error: Chips file not found: {chips_path}")
        return 1

    try:
        chip_geo = load_chip_geometry(str(chips_path), args.chip)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    bbox = chip_geo.bbox
    polygon = chip_geo.polygon
    centroid = chip_geo.centroid

    # ---- Load plane data ----
    plane_path = Path(args.plane)
    if not plane_path.exists():
        print(f"Error: Plane file not found: {plane_path}")
        return 1

    with open(plane_path) as f:
        plane_data = json.load(f)

    plane = plane_data["plane"]
    plane_a = plane["a"]  # um/um
    plane_b = plane["b"]  # um/um
    plane_c = plane["c"]  # um

    wb = args.white_balance

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
    if not args.dry_run:
        if os.path.exists(args.output):
            if args.clean:
                shutil.rmtree(args.output)
            else:
                print(f"Error: Output directory '{args.output}' exists. Use --clean to wipe.")
                return 1
        os.makedirs(args.output)

    # ---- Compute frame dimensions from microscope description ----
    desc = require_microscope_description()

    obj_mag_for_plan = None
    if args.objective_mag is not None:
        mag_str = args.objective_mag.lower().rstrip("x")
        for obj in desc.objectives.values():
            if str(obj.magnification) == mag_str:
                obj_mag_for_plan = obj.magnification
                break
        if obj_mag_for_plan is None:
            print(f"Error: Unknown objective magnification '{args.objective_mag}'")
            return 1
    elif args.objective_pos is not None:
        obj_info = desc.objectives.get(args.objective_pos)
        if obj_info:
            obj_mag_for_plan = obj_info.magnification
        else:
            print(f"Error: Unknown objective position {args.objective_pos}")
            return 1
    else:
        print("Error: --objective-mag or --objective-pos required")
        return 1

    frame_size = compute_frame_size_um(desc.camera, obj_mag_for_plan, binning_idx)
    if frame_size is None:
        print("Error: Could not compute frame size from microscope description")
        return 1
    plan_frame_width_um, plan_frame_height_um = frame_size

    # ---- Print scan plan ----
    print("Chip Scan with Focus Plane (v2)")
    print("=" * 60)
    print(f"Chip: #{args.chip}")
    print(f"  BBox: X=[{bbox['x_min']:.0f}, {bbox['x_max']:.0f}], Y=[{bbox['y_min']:.0f}, {bbox['y_max']:.0f}] um")
    print(f"  Centroid: ({centroid[0]:.0f}, {centroid[1]:.0f}) um")
    print(f"  Hull vertices: {len(polygon)}")
    print(f"  Area: {chip_geo.area_um2 / 1e6:.1f} mm2")
    print()
    print(f"Focus plane: Z = {plane_a * 1000:.4f}*X_mm + {plane_b * 1000:.4f}*Y_mm + {plane_c:.2f}")
    print(f"  Expected Z range: {z_min_expected:.0f} - {z_max_expected:.0f} um")
    print(f"  Z limit: {args.z_max:.0f} um")
    print()
    print(f"Scan speed: {args.speed_mm:.1f} mm/s")
    print(f"Lead-in: {args.lead_in_um:.0f} um, Z lead: {args.z_lead_ms:.0f} ms")
    if args.row_limit:
        print(f"Row limit: {args.row_limit}")
    print()

    # ---- Compute row plan (unpadded — lead-in applied per-row) ----
    try:
        plan = compute_planar_scan_plan(
            bbox,
            polygon,
            plane_a=plane_a,
            plane_b=plane_b,
            plane_c=plane_c,
            frame_width_um=plan_frame_width_um,
            frame_height_um=plan_frame_height_um,
            x_overlap_pct=args.x_overlap_percent,
            y_overlap_pct=args.y_overlap_percent,
            padding=0,
            row_limit=args.row_limit,
            speed_mm=args.speed_mm,
            z_max=args.z_max,
        )
    except ValueError as e:
        print(f"ABORT: {e}")
        return 1

    # Print plan summary
    row_widths = [(x_max - x_min) / 1000 for _, x_min, x_max in plan.rows]
    total_distance_mm = sum(row_widths)
    est_total_time_s = total_distance_mm / args.speed_mm + len(plan.rows) * 0.5

    print(f"Frame FOV: {plan.frame_width_um:.1f} x {plan.frame_height_um:.1f} um")
    print(f"Frame skip: target advance {plan.target_advance_um:.0f} um ({args.x_overlap_percent:.0f}% X overlap)")
    print()
    print(f"Row plan: {len(plan.rows)} rows")
    print(f"  Y step: {plan.y_step_um:.1f} um ({args.y_overlap_percent:.0f}% overlap)")
    print(f"  Row widths: {min(row_widths):.1f} - {max(row_widths):.1f} mm")
    print(f"  Total scan distance: {total_distance_mm:.1f} mm")
    print(f"  Z range: {plan.validated_z_min_um:.0f} - {plan.validated_z_max_um:.0f} um")
    z_vel = abs(plan.plane_a * args.speed_mm * 1000)
    print(f"  Z velocity: {z_vel:.1f} um/s (from plane slope)")
    print(f"  Estimated time: ~{est_total_time_s:.0f}s")
    print()

    if args.dry_run:
        print("(dry run -- exiting)")
        return 0

    # ---- Connect to hardware ----
    from PIL import Image as PILImage

    from flakefinder.image_utils import sdk_image_to_numpy
    from flakefinder.leica import Microscope, wait_all

    with Microscope() as scope:
        scope.validate_description(desc)
        stage = scope.stage
        z_drive = scope.z

        # Fast position readers
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter
        z_converter = z_drive.converter
        z_bcv_hysteresis = z_drive.bcv_hysteresis or z_drive.bcv

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

        # Lighting
        scope.light_on()

        # Camera
        camera = scope.camera
        acquisition = scope.acquisition

        # Configure camera
        camera.trigger_mode = 0  # CONTINUOUS
        camera.binning = binning_idx
        camera.exposure_time = args.exposure_ms / 1000.0
        camera.gain = args.gain
        camera.gain_rgb = wb
        camera.gamma = args.gamma

        # ---- Build microscope metadata (reads all values back from hardware) ----
        micro_meta = build_microscope_meta(scope)
        cam_meta = micro_meta["camera"]
        optics_meta = micro_meta["optics"]

        # Validate frame size against plan
        frame_width_um = optics_meta["frame_width_um"]
        frame_height_um = optics_meta["frame_height_um"]
        if frame_width_um is None or frame_height_um is None:
            print("Error: Could not determine frame size. Check objective/camera.")
            return 1
        if abs(frame_width_um - plan.frame_width_um) > 1.0:
            print(f"ABORT: Camera frame width {frame_width_um:.1f} != plan {plan.frame_width_um:.1f} um")
            return 1
        if abs(frame_height_um - plan.frame_height_um) > 1.0:
            print(f"ABORT: Camera frame height {frame_height_um:.1f} != plan {plan.frame_height_um:.1f} um")
            return 1

        exp_str = f"{cam_meta['exposure_s'] * 1000:.2f}ms" if cam_meta["exposure_s"] else "?"
        print(f"Camera: {cam_meta['name']}")
        print(f"  Binning: {cam_meta['binning']}x{cam_meta['binning']}, Exposure: {exp_str}, Gain: {cam_meta['gain']}")
        print(f"  Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")
        print(f"  Frame: {cam_meta['frame_width_px']}x{cam_meta['frame_height_px']} px")
        print(f"  FOV: {frame_width_um:.1f} x {frame_height_um:.1f} um (matches plan)")
        print(f"  Objective: {optics_meta['objective_mag']}x")
        print()

        # ---- Set up acquisition context ----
        from LeicaMicrosystems.HardwareModel import Extensions

        context = scope.context
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # ---- Set up saver threads ----
        save_queue: queue.Queue = queue.Queue()
        saved_frames_meta: list[dict] = []

        def saver_thread():
            while True:
                item = save_queue.get()
                if item is None:
                    break

                (
                    frame_idx,
                    row_idx,
                    t_start,
                    t_end,
                    image,
                    row_y,
                    row_x_samples,
                    row_z_samples,
                    t0,
                    frame_chip_edge_x,
                    frame_direction,
                ) = item

                arr = sdk_image_to_numpy(image)
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
                z_ideal = (
                    compute_plane_z(plan.plane_a, plan.plane_b, plan.plane_c, x_start_interp, row_y)
                    if x_start_interp
                    else None
                )
                z_error = (z_interp - z_ideal) if (z_interp is not None and z_ideal is not None) else None

                dt = t_end - t_start
                x_vel = (x_end_interp - x_start_interp) / dt if dt > 0 and x_start_interp and x_end_interp else 0

                # Lead-in check: frame start hasn't crossed chip edge yet
                if x_start_interp is not None:
                    if frame_direction == 1:
                        in_lead_in = x_start_interp < frame_chip_edge_x
                    else:
                        in_lead_in = x_start_interp > frame_chip_edge_x
                else:
                    in_lead_in = True  # no position data → treat as lead-in

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
                        "z_actual": z_interp,
                        "z_ideal": z_ideal,
                        "z_error": z_error,
                        "in_lead_in": in_lead_in,
                    }
                )

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

        print(f"Initial position: X={initial_x:.0f}, Y={initial_y:.0f}, Z={initial_z:.0f} um")

        # Set move speed for positioning
        stage.x.set_velocity_um_s(args.move_speed_mm * 1000)
        stage.y.set_velocity_um_s(args.move_speed_mm * 1000)

        # Scan speed
        x_speed_um_s = args.speed_mm * 1000

        # ---- Build scan_row config ----
        hw = ScanHardware(
            stage=stage,
            z_drive=z_drive,
            acquisition=acquisition,
            acq_context=context,
            current_image=current_image,
            x_bcv=x_bcv,
            x_converter=x_converter,
            z_converter=z_converter,
            z_bcv_hysteresis=z_bcv_hysteresis,
        )

        cfg = RowConfig(
            plane_a=plan.plane_a,
            plane_b=plan.plane_b,
            plane_c=plan.plane_c,
            x_speed_um_s=x_speed_um_s,
            move_speed_um_s=args.move_speed_mm * 1000,
            target_advance_um=plan.target_advance_um,
            lead_in_um=args.lead_in_um,
            z_lead_s=args.z_lead_ms / 1000.0,
            row_settle_s=args.row_settle,
            warmup_frames=args.warmup_frames,
            z_max=args.z_max,
            quiet=args.quiet,
        )

        # ---- Scan loop ----
        total_scan_start = time.perf_counter()
        all_position_samples: list[dict] = []
        global_frame_idx = 0
        row_timings: list[dict[str, float | None] | None] = [None] * len(plan.rows)
        row_speed_measurements: list[dict[str, Any] | None] = [None] * len(plan.rows)

        try:
            for row_idx, (row_y, row_x_min, row_x_max) in enumerate(plan.rows):
                direction = 1 if row_idx % 2 == 0 else -1

                result = scan_row(
                    row_idx,
                    row_y,
                    row_x_min,
                    row_x_max,
                    direction,
                    len(plan.rows),
                    hw=hw,
                    cfg=cfg,
                    save_queue=save_queue,
                    global_frame_idx=global_frame_idx,
                    scan_t0=total_scan_start,
                )

                if not result.skipped:
                    global_frame_idx += result.frames_saved
                    all_position_samples.extend(result.position_samples)
                    row_timings[row_idx] = result.timing_s
                    row_speed_measurements[row_idx] = result.speed_measurement

        finally:
            # Emergency stop on any error
            z_drive.halt()
            stage.x.halt()

        total_scan_end = time.perf_counter()
        total_duration = total_scan_end - total_scan_start

        # Return to initial position
        print("\nReturning to initial position...")
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

        # Z tracking error stats (exclude lead-in frames)
        tracking_frames = [f for f in saved_frames_meta if f["z_error"] is not None and not f.get("in_lead_in", False)]
        n_lead_in = sum(1 for f in saved_frames_meta if f.get("in_lead_in", False))
        z_errors = [f["z_error"] for f in tracking_frames]
        if z_errors:
            z_error_arr = np.array(z_errors)
            z_error_mean = float(np.mean(z_error_arr))
            z_error_std = float(np.std(z_error_arr))
            z_error_max = float(np.max(np.abs(z_error_arr)))
            z_error_p95 = float(np.percentile(np.abs(z_error_arr), 95))
        else:
            z_error_mean = z_error_std = z_error_max = z_error_p95 = None

        # Per-direction bias and z-jump stats (exclude lead-in)
        dir_stats = {}  # direction -> {mean_error, z_jump}
        for direction in [1, -1]:
            dir_row_idxs = {ri for ri, (_, _, _) in enumerate(plan.rows) if (1 if ri % 2 == 0 else -1) == direction}
            dir_frames = [
                f
                for f in saved_frames_meta
                if f["row"] in dir_row_idxs and f["z_error"] is not None and not f.get("in_lead_in", False)
            ]
            if not dir_frames:
                continue
            dir_errors = [f["z_error"] for f in dir_frames]

            # Z-jump per row: first two non-lead-in frames
            z_jumps = []
            for ri in sorted(dir_row_idxs):
                rf = [
                    f
                    for f in saved_frames_meta
                    if f["row"] == ri and f["z_error"] is not None and not f.get("in_lead_in", False)
                ]
                if len(rf) >= 2:
                    z_jumps.append(abs(rf[1]["z_error"] - rf[0]["z_error"]))

            dir_stats[direction] = {
                "mean_error_um": float(np.mean(dir_errors)),
                "z_jump_um": float(np.mean(z_jumps)) if z_jumps else None,
            }

        # Build rows metadata
        rows_meta = []
        for row_idx, (row_y, row_x_min, row_x_max) in enumerate(plan.rows):
            direction = 1 if row_idx % 2 == 0 else -1
            row_frames = [f for f in saved_frames_meta if f["row"] == row_idx]
            frame_start = row_frames[0]["n"] if row_frames else global_frame_idx
            frame_end = (row_frames[-1]["n"] + 1) if row_frames else global_frame_idx
            row_pos_samples = [s for s in all_position_samples if s["row"] == row_idx]

            timing = row_timings[row_idx]
            if timing and timing.get("capture_end") is not None and timing.get("preposition_start") is not None:
                duration = timing["capture_end"] - timing["preposition_start"]
            else:
                duration = None

            rows_meta.append(
                {
                    "row_idx": row_idx,
                    "y_um": row_y,
                    "x_min_um": row_x_min,
                    "x_max_um": row_x_max,
                    "direction": direction,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "duration_s": duration,
                    "position_samples": len(row_pos_samples),
                    "timing_s": timing,
                    "speed_measurement": row_speed_measurements[row_idx],
                }
            )

        # Build metadata (scan_area_v1.py compatible for stitching)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "command": sys.argv,
            "x_min_um": min(r.x_min_um for r in plan.rows),
            "x_max_um": max(r.x_max_um for r in plan.rows),
            "y_min_um": plan.rows[0].y_um,
            "y_max_um": plan.rows[-1].y_um,
            "y_step_um": plan.y_step_um,
            "y_overlap_percent": args.y_overlap_percent,
            "downsample": args.downsample,
            "scan_duration_s": total_duration,
            "frame_count": global_frame_idx,
            "position_sample_count": len(all_position_samples),
            "scan_params": {
                "scan_speed_mm_s": args.speed_mm,
                "move_speed_mm_s": args.move_speed_mm,
                "lead_in_um": args.lead_in_um,
                "z_lead_ms": args.z_lead_ms,
                "row_limit": args.row_limit,
            },
            "chip_info": {
                "chips_meta": str(chips_path),
                "chip_index": chip_geo.chip_index,
                "bbox_stage_um": bbox,
                "centroid_stage_um": centroid,
                "area_um2": chip_geo.area_um2,
                "hull_vertices": len(polygon),
            },
            "focus_plane": {
                "plane_file": str(plane_path),
                "a": plan.plane_a,
                "b": plan.plane_b,
                "c": plan.plane_c,
                "equation": plane.get("equation", f"Z = {plan.plane_a}*X + {plan.plane_b}*Y + {plan.plane_c}"),
                "z_range_um": [plan.validated_z_min_um, plan.validated_z_max_um],
                "tracking_error": {
                    "mean_um": z_error_mean,
                    "std_um": z_error_std,
                    "max_um": z_error_max,
                    "p95_um": z_error_p95,
                    "mean_error_pos_um": dir_stats.get(1, {}).get("mean_error_um"),
                    "mean_error_neg_um": dir_stats.get(-1, {}).get("mean_error_um"),
                    "z_jump_pos_um": dir_stats.get(1, {}).get("z_jump_um"),
                    "z_jump_neg_um": dir_stats.get(-1, {}).get("z_jump_um"),
                },
            },
            **micro_meta,
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
            shutil.make_archive(args.output, "zip", args.output)
            print(f"Created {args.output}.zip")

        # Summary (printed before cleanup to survive Dispose() crashes)
        print()
        print("=" * 60)
        print("SCAN SUMMARY:")
        print(f"  Total time: {total_duration:.1f}s")
        print(f"  Rows: {len(plan.rows)}")
        print(f"  Total frames: {global_frame_idx}")
        print(f"  Lead-in frames: {n_lead_in} (excluded from Z tracking stats)")
        print(f"  Avg FPS: {global_frame_idx / total_duration:.1f}" if total_duration > 0 else "  Avg FPS: N/A")
        if z_error_max is not None:
            dof_20x = 1.7
            print(
                f"  Z tracking error: mean={z_error_mean:+.2f}, std={z_error_std:.2f}, "
                f"max={z_error_max:.2f}, p95={z_error_p95:.2f} um"
            )
            print(
                f"  {'PASS' if z_error_max < dof_20x else 'NOTE'}: max error "
                f"{'within' if z_error_max < dof_20x else 'exceeds'} 20x DOF ({dof_20x} um)"
            )
            if dir_stats:
                pos = dir_stats.get(1)
                neg = dir_stats.get(-1)
                if pos:
                    zj = f", z-jump={pos['z_jump_um']:.2f}" if pos["z_jump_um"] is not None else ""
                    print(f"  +X rows: mean err={pos['mean_error_um']:+.2f}{zj} um")
                if neg:
                    zj = f", z-jump={neg['z_jump_um']:.2f}" if neg["z_jump_um"] is not None else ""
                    print(f"  -X rows: mean err={neg['mean_error_um']:+.2f}{zj} um")
        print(f"  Output: {args.output}/")
        print()
        print("Done.")

        wait_all([hx, hy])
        z_drive.move_to_corrected(initial_z)

        return 0


if __name__ == "__main__":
    exit(main())
