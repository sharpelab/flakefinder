"""Test Z ramp synchronization at various X speeds.

Tests directed velocity interface with realistic focus plane slope.
Collects tracking error statistics for each speed combination.

Usage:
    python test_z_ramp_speeds.py
    python test_z_ramp_speeds.py --slope 1.45 --x-distance 10
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def run_single_test(
    x_speed_mm_s: float,
    slope_um_per_mm: float,
    x_distance_mm: float = 10.0,
    z_safe_um: float = 20000.0,
) -> dict:
    """Run a single Z ramp test at specified X speed.

    Args:
        x_speed_mm_s: X velocity in mm/s.
        slope_um_per_mm: Focus plane slope (µm Z per mm X).
        x_distance_mm: X travel distance in mm.
        z_safe_um: Safe Z position to start from.

    Returns:
        Dict with test results and tracking error stats.
    """
    from flakefinder.leica import LeicaConnection, Stage, ZDrive

    x_distance_um = x_distance_mm * 1000
    x_speed_um_s = x_speed_mm_s * 1000

    # Calculate Z parameters from slope
    z_delta_um = slope_um_per_mm * x_distance_mm
    expected_duration_s = x_distance_um / x_speed_um_s
    z_speed_um_s = abs(z_delta_um) / expected_duration_s

    print(f"\n{'=' * 60}")
    print(f"Test: X={x_speed_mm_s:.0f} mm/s, slope={slope_um_per_mm:.2f} µm/mm")
    print(f"  Z velocity: {z_speed_um_s:.2f} µm/s")
    print(f"  Z delta: {z_delta_um:+.1f} µm over {x_distance_mm:.0f} mm")
    print(f"  Expected duration: {expected_duration_s:.2f} s")

    with LeicaConnection() as conn:
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        # Get axis interfaces for fast polling
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter

        # Save initial position
        initial_x = stage.x.position_um
        initial_z = z_drive.position_um

        # Calculate start/end positions
        x_margin = 2000
        if initial_x + x_distance_um > stage.x.max_um - x_margin:
            x_start = stage.x.max_um - x_margin - x_distance_um
        else:
            x_start = max(stage.x.min_um + x_margin, initial_x)
        x_end = x_start + x_distance_um

        z_start = z_safe_um
        z_end = z_safe_um + z_delta_um

        x_towards_max = x_end > x_start
        z_towards_max = z_end > z_start

        try:
            # Move to start positions
            z_drive.move_to(z_safe_um)
            time.sleep(0.1)
            stage.x.move_to(x_start)
            time.sleep(0.1)

            # Position sampling
            x_samples = []
            z_samples = []
            stop_polling = threading.Event()

            def poll_x():
                while not stop_polling.is_set():
                    t_before = time.perf_counter()
                    x_native = x_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    x_um = x_converter.GetMetricsValue(x_native)
                    x_samples.append((t_before, t_after, x_um))

            def poll_z():
                while not stop_polling.is_set():
                    t_before = time.perf_counter()
                    z_native = z_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    z_um = z_converter.GetMetricsValue(z_native)
                    z_samples.append((t_before, t_after, z_um))

            # Start polling
            x_thread = threading.Thread(target=poll_x, daemon=True)
            z_thread = threading.Thread(target=poll_z, daemon=True)
            x_thread.start()
            z_thread.start()
            time.sleep(0.1)  # Brief warmup

            # Start motion
            time.perf_counter()

            if x_towards_max:
                stage.x.start_towards_max(x_speed_um_s)
            else:
                stage.x.start_towards_min(x_speed_um_s)

            if z_towards_max:
                z_drive.start_towards_max(z_speed_um_s)
            else:
                z_drive.start_towards_min(z_speed_um_s)

            t_after_start = time.perf_counter()

            # Wait for target with timeout
            timeout_s = expected_duration_s * 2.5
            target_reached = False
            while not target_reached:
                elapsed = time.perf_counter() - t_after_start
                if elapsed > timeout_s:
                    print(f"  TIMEOUT after {elapsed:.2f}s")
                    break
                current_x = stage.x.position_um
                if x_towards_max and current_x >= x_end or not x_towards_max and current_x <= x_end:
                    target_reached = True
                time.sleep(0.001)

            # Halt
            t_halt = time.perf_counter()
            stage.x.halt()
            z_drive.halt()
            time.perf_counter()

            time.sleep(0.05)
            stop_polling.set()
            x_thread.join(timeout=0.5)
            z_thread.join(timeout=0.5)

            # Get final positions
            final_x = stage.x.position_um
            final_z = z_drive.position_um
            motion_duration = t_halt - t_after_start

            print(f"  Duration: {motion_duration * 1000:.0f} ms (expected {expected_duration_s * 1000:.0f} ms)")
            print(f"  X overshoot: {final_x - x_end:+.0f} µm")
            print(f"  Z overshoot: {final_z - z_end:+.1f} µm")

        finally:
            # Restore position
            z_drive.move_to(initial_z)
            stage.x.move_to(initial_x)

    # Analyze tracking error
    x_t = np.array([(s[0] + s[1]) / 2 - t_after_start for s in x_samples])
    x_pos = np.array([s[2] for s in x_samples])
    z_t = np.array([(s[0] + s[1]) / 2 - t_after_start for s in z_samples])
    z_pos = np.array([s[2] for s in z_samples])

    # Interpolate Z at X sample times
    z_interp = np.interp(x_t, z_t, z_pos)

    # Calculate ideal Z for each X position
    slope = (z_end - z_start) / (x_end - x_start)
    z_ideal = z_start + slope * (x_pos - x_start)

    # Filter to constant velocity portion (skip accel/decel)
    accel_time = 0.3  # Conservative estimate
    decel_margin = 0.1
    mask = (x_t >= accel_time) & (x_t <= motion_duration - decel_margin)

    if mask.sum() > 10:
        z_error = z_interp[mask] - z_ideal[mask]
        error_mean = float(np.mean(z_error))
        error_std = float(np.std(z_error))
        error_max = float(np.max(np.abs(z_error)))
    else:
        error_mean = error_std = error_max = float("nan")
        print("  WARNING: Not enough samples in constant velocity region")

    print(f"  Tracking error: mean={error_mean:.2f} µm, std={error_std:.2f} µm, max={error_max:.2f} µm")

    return {
        "x_speed_mm_s": x_speed_mm_s,
        "z_speed_um_s": z_speed_um_s,
        "slope_um_per_mm": slope_um_per_mm,
        "x_distance_mm": x_distance_mm,
        "expected_duration_s": expected_duration_s,
        "actual_duration_s": motion_duration,
        "x_overshoot_um": final_x - x_end,
        "z_overshoot_um": final_z - z_end,
        "target_reached": target_reached,
        "error_mean_um": error_mean,
        "error_std_um": error_std,
        "error_max_um": error_max,
        "num_x_samples": len(x_samples),
        "num_z_samples": len(z_samples),
        "t_after_start": t_after_start,
        "x_start_um": x_start,
        "x_end_um": x_end,
        "z_start_um": z_start,
        "z_end_um": z_end,
        "x_samples": [{"t_before": s[0], "t_after": s[1], "x_um": s[2]} for s in x_samples],
        "z_samples": [{"t_before": s[0], "t_after": s[1], "z_um": s[2]} for s in z_samples],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test Z ramp synchronization at various X speeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--slope", type=float, default=1.45, help="Focus plane slope in µm/mm (Z change per mm X)")
    parser.add_argument("--x-distance", type=float, default=10.0, help="X travel distance in mm")
    parser.add_argument("--z-safe", type=float, default=20000.0, help="Safe Z position to start from (µm)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--speeds", type=str, default="5,10,20,30,40", help="Comma-separated X speeds to test (mm/s)")
    args = parser.parse_args()

    # X speeds to test (mm/s)
    x_speeds = [float(s) for s in args.speeds.split(",")]

    print("Z Ramp Speed Test")
    print("=================")
    print(f"Slope: {args.slope:.2f} µm/mm")
    print(f"X distance: {args.x_distance:.1f} mm")
    print(f"X speeds to test: {x_speeds} mm/s")
    print("")
    print("Expected Z velocities:")
    for x_speed in x_speeds:
        z_vel = args.slope * x_speed
        print(f"  {x_speed:2.0f} mm/s X -> {z_vel:5.1f} µm/s Z")

    results = []
    for x_speed in x_speeds:
        try:
            result = run_single_test(
                x_speed_mm_s=x_speed,
                slope_um_per_mm=args.slope,
                x_distance_mm=args.x_distance,
                z_safe_um=args.z_safe,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "x_speed_mm_s": x_speed,
                    "error": str(e),
                }
            )
        time.sleep(0.5)  # Brief pause between tests

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'X Speed':>8} {'Z Speed':>8} {'Duration':>10} {'Err Mean':>10} {'Err Max':>10}")
    print(f"{'(mm/s)':>8} {'(µm/s)':>8} {'(ms)':>10} {'(µm)':>10} {'(µm)':>10}")
    print(f"{'-' * 60}")

    for r in results:
        if "error" in r:
            print(f"{r['x_speed_mm_s']:>8.0f} {'FAILED':>8}")
        else:
            print(
                f"{r['x_speed_mm_s']:>8.0f} {r['z_speed_um_s']:>8.1f} "
                f"{r['actual_duration_s'] * 1000:>10.0f} "
                f"{r['error_mean_um']:>10.2f} {r['error_max_um']:>10.2f}"
            )

    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"z_ramp_speeds_{timestamp}.json")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "slope_um_per_mm": args.slope,
            "x_distance_mm": args.x_distance,
            "z_safe_um": args.z_safe,
            "x_speeds_mm_s": x_speeds,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
