"""Measure synchronized X+Z ramp motion for focus tracking validation.

Uses directed velocity interface (StartTowardsMax/Min) for better synchronization.
Collects position data during coordinated X+Z motion. Run on microscope PC.
Use test_z_ramp_analyze.py locally to analyze results.

Usage:
    python test_z_ramp_measure.py
    python test_z_ramp_measure.py --x-speed 40 --z-delta 50
    python test_z_ramp_measure.py -o my_test.json
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path


def run_z_ramp_measure(
    x_distance_mm: float = 20.0,
    x_speed_mm_s: float = 10.0,
    z_delta_um: float = 30.0,
    z_safe_um: float = 20000.0,
    output_path: Path | None = None,
) -> dict:
    """Run synchronized X+Z ramp test using directed velocity interface.

    Args:
        x_distance_mm: X travel distance in mm.
        x_speed_mm_s: X velocity in mm/s.
        z_delta_um: Z change during ramp (positive = towards max).
        z_safe_um: Safe Z position to start from (away from sample).
        output_path: Optional path to save results JSON.

    Returns:
        Dict with test results and position samples.
    """
    from flakefinder.leica import LeicaConnection, Stage, ZDrive

    x_distance_um = x_distance_mm * 1000
    x_speed_um_s = x_speed_mm_s * 1000

    # Calculate expected duration and required Z velocity
    expected_duration_s = x_distance_um / x_speed_um_s
    z_speed_um_s = abs(z_delta_um) / expected_duration_s

    print("Z Ramp Measurement (Directed Velocity)")
    print("=" * 50)
    print(f"X distance: {x_distance_mm:.1f} mm")
    print(f"X speed: {x_speed_mm_s:.1f} mm/s")
    print(f"Z delta: {z_delta_um:+.1f} µm")
    print(f"Z speed (calculated): {z_speed_um_s:.2f} µm/s ({z_speed_um_s / 1000:.4f} mm/s)")
    print(f"Expected duration: {expected_duration_s:.2f} s")
    print()

    with LeicaConnection() as conn:
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        # Get axis interfaces for fast polling
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter

        # Save initial position for restoration
        initial_x = stage.x.position_um
        initial_y = stage.y.position_um
        initial_z = z_drive.position_um

        print(f"Initial position (will restore): X={initial_x:.0f}, Y={initial_y:.0f}, Z={initial_z:.0f} µm")

        # Check directed velocity support
        if not stage.x.supports_directed_velocity:
            print("ERROR: X axis does not support directed velocity")
            return {"error": "X directed velocity not supported"}
        if not z_drive.supports_directed_velocity:
            print("ERROR: Z axis does not support directed velocity")
            return {"error": "Z directed velocity not supported"}

        print("Directed velocity: X=supported, Z=supported")

        # Calculate start/end positions
        x_margin = 2000  # 2mm margin
        if initial_x + x_distance_um > stage.x.max_um - x_margin:
            x_start = stage.x.max_um - x_margin - x_distance_um
        else:
            x_start = max(stage.x.min_um + x_margin, initial_x)
        x_end = x_start + x_distance_um

        # Z positions - start at safe height, ramp by z_delta
        z_start = z_safe_um
        z_end = z_safe_um + z_delta_um

        # Determine directions
        x_towards_max = x_end > x_start
        z_towards_max = z_end > z_start

        print("\nTest trajectory:")
        print(
            f"  X: {x_start:.0f} -> {x_end:.0f} µm ({x_distance_mm:.1f} mm, {'towards_max' if x_towards_max else 'towards_min'})"  # noqa: E501
        )
        print(
            f"  Z: {z_start:.0f} -> {z_end:.0f} µm ({z_delta_um:+.1f} µm, {'towards_max' if z_towards_max else 'towards_min'})"  # noqa: E501
        )
        print()

        try:
            # Move to safe Z first (blocking)
            print(f"Moving to safe Z ({z_safe_um:.0f} µm)...")
            z_drive.move_to(z_safe_um)
            time.sleep(0.2)

            # Move to X start position
            print(f"Moving to X start ({x_start:.0f} µm)...")
            stage.x.move_to(x_start)
            time.sleep(0.2)

            actual_x_start = stage.x.position_um
            actual_z_start = z_drive.position_um
            print(f"At start: X={actual_x_start:.0f}, Z={actual_z_start:.0f} µm")

            # Position sampling storage - separate streams for X and Z
            x_samples = []  # [(t_before, t_after, x_um), ...]
            z_samples = []  # [(t_before, t_after, z_um), ...]
            stop_polling = threading.Event()

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
                    t_before = time.perf_counter()
                    z_native = z_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    z_um = z_converter.GetMetricsValue(z_native)
                    z_samples.append((t_before, t_after, z_um))

            # Start polling threads
            print("\nStarting position polling (parallel X and Z threads)...")
            x_thread = threading.Thread(target=poll_x, daemon=True)
            z_thread = threading.Thread(target=poll_z, daemon=True)
            x_thread.start()
            z_thread.start()

            # Let polling run briefly before motion to verify rate
            time.sleep(0.2)
            warmup_x = len(x_samples)
            warmup_z = len(z_samples)
            print(f"  Warmup X polling rate: {warmup_x / 0.2:.1f} Hz ({warmup_x} samples)")
            print(f"  Warmup Z polling rate: {warmup_z / 0.2:.1f} Hz ({warmup_z} samples)")

            print("\nStarting synchronized ramp (directed velocity)...")
            print(f"  X: start_towards_{'max' if x_towards_max else 'min'}({x_speed_um_s:.0f} µm/s)")
            print(f"  Z: start_towards_{'max' if z_towards_max else 'min'}({z_speed_um_s:.2f} µm/s)")

            t_before_start = time.perf_counter()

            # Start both axes with directed velocity - simultaneous!
            if x_towards_max:
                stage.x.start_towards_max(x_speed_um_s)
            else:
                stage.x.start_towards_min(x_speed_um_s)

            if z_towards_max:
                z_drive.start_towards_max(z_speed_um_s)
            else:
                z_drive.start_towards_min(z_speed_um_s)

            t_after_start = time.perf_counter()
            x_samples_at_start = len(x_samples)
            z_samples_at_start = len(z_samples)

            # Wait until X reaches target position, then halt both
            timeout_s = expected_duration_s * 2
            target_reached = False
            while not target_reached:
                elapsed = time.perf_counter() - t_after_start
                if elapsed > timeout_s:
                    print(f"  TIMEOUT after {elapsed:.2f}s (limit: {timeout_s:.2f}s)")
                    break
                current_x = stage.x.position_um
                if x_towards_max and current_x >= x_end or not x_towards_max and current_x <= x_end:
                    target_reached = True
                time.sleep(0.001)

            # Halt both axes
            t_halt = time.perf_counter()
            stage.x.halt()
            z_drive.halt()
            t_after_halt = time.perf_counter()

            x_samples_at_end = len(x_samples)
            z_samples_at_end = len(z_samples)

            # Poll a bit more to capture settling
            time.sleep(0.1)

            # Stop polling
            stop_polling.set()
            x_thread.join(timeout=1.0)
            z_thread.join(timeout=1.0)

            # Get final positions
            final_x = stage.x.position_um
            final_z = z_drive.position_um

            # Calculate motion period stats
            motion_duration = t_halt - t_after_start
            x_samples_during_motion = x_samples_at_end - x_samples_at_start
            z_samples_during_motion = z_samples_at_end - z_samples_at_start
            x_poll_rate = x_samples_during_motion / motion_duration if motion_duration > 0 else 0
            z_poll_rate = z_samples_during_motion / motion_duration if motion_duration > 0 else 0

            print("\nMotion complete!")
            print(f"  Start command latency: {(t_after_start - t_before_start) * 1000:.1f} ms")
            print(f"  Motion duration: {motion_duration * 1000:.1f} ms (expected: {expected_duration_s * 1000:.1f} ms)")
            print(f"  Halt latency: {(t_after_halt - t_halt) * 1000:.1f} ms")
            print(f"  Final X: {final_x:.0f} µm (target: {x_end:.0f}, overshoot: {final_x - x_end:+.0f})")
            print(f"  Final Z: {final_z:.0f} µm (target: {z_end:.0f}, overshoot: {final_z - z_end:+.1f})")
            print("\nPolling during motion (parallel threads):")
            print(f"  X samples: {x_samples_during_motion} ({x_poll_rate:.1f} Hz)")
            print(f"  Z samples: {z_samples_during_motion} ({z_poll_rate:.1f} Hz)")
            print(f"  Total samples: X={len(x_samples)}, Z={len(z_samples)}")

            # Analyze per-sample timing
            import statistics

            if len(x_samples) > 1:
                x_read_times = [(s[1] - s[0]) * 1000 for s in x_samples]
                print("\nSDK call timing:")
                print(f"  X read: {statistics.mean(x_read_times):.2f} ± {statistics.stdev(x_read_times):.2f} ms")
            if len(z_samples) > 1:
                z_read_times = [(s[1] - s[0]) * 1000 for s in z_samples]
                print(f"  Z read: {statistics.mean(z_read_times):.2f} ± {statistics.stdev(z_read_times):.2f} ms")

        finally:
            # Always restore position
            print("\nRestoring initial position...")
            print(f"  Moving to Z={initial_z:.0f} µm...")
            z_drive.move_to(initial_z)

            print(f"  Moving to X={initial_x:.0f} µm...")
            stage.x.move_to(initial_x)

            final_restored_x = stage.x.position_um
            final_restored_z = z_drive.position_um
            print(f"  Restored to: X={final_restored_x:.0f}, Z={final_restored_z:.0f} µm")

    # Build results dict
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "directed_velocity",
        "params": {
            "x_distance_mm": x_distance_mm,
            "x_speed_mm_s": x_speed_mm_s,
            "z_delta_um": z_delta_um,
            "z_safe_um": z_safe_um,
        },
        "initial_position": {
            "x_um": initial_x,
            "y_um": initial_y,
            "z_um": initial_z,
        },
        "commanded": {
            "x_start_um": x_start,
            "x_end_um": x_end,
            "z_start_um": z_start,
            "z_end_um": z_end,
            "x_velocity_um_s": x_speed_um_s,
            "z_velocity_um_s": z_speed_um_s,
            "expected_duration_s": expected_duration_s,
            "x_towards_max": x_towards_max,
            "z_towards_max": z_towards_max,
        },
        "timing": {
            "t_before_start": t_before_start,
            "t_after_start": t_after_start,
            "t_halt": t_halt,
            "t_after_halt": t_after_halt,
            "start_command_latency_ms": (t_after_start - t_before_start) * 1000,
            "motion_duration_s": motion_duration,
            "halt_latency_ms": (t_after_halt - t_halt) * 1000,
        },
        "final_position": {
            "x_um": final_x,
            "z_um": final_z,
            "x_overshoot_um": final_x - x_end,
            "z_overshoot_um": final_z - z_end,
        },
        "polling": {
            "x_total_samples": len(x_samples),
            "z_total_samples": len(z_samples),
            "x_samples_during_motion": x_samples_during_motion,
            "z_samples_during_motion": z_samples_during_motion,
            "x_poll_rate_hz": x_poll_rate,
            "z_poll_rate_hz": z_poll_rate,
            "x_warmup_rate_hz": warmup_x / 0.2,
            "z_warmup_rate_hz": warmup_z / 0.2,
        },
        # Separate sample streams: (t_before, t_after, position_um)
        "x_samples": [{"t_before": s[0], "t_after": s[1], "x_um": s[2]} for s in x_samples],
        "z_samples": [{"t_before": s[0], "t_after": s[1], "z_um": s[2]} for s in z_samples],
    }

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        print(f"Run analysis with: python test_z_ramp_analyze.py {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure synchronized X+Z ramp motion using directed velocity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x-distance", type=float, default=20.0, help="X travel distance in mm")
    parser.add_argument("--x-speed", type=float, default=10.0, help="X velocity in mm/s")
    parser.add_argument(
        "--z-delta", type=float, default=30.0, help="Z change during ramp in µm (positive = towards max)"
    )
    parser.add_argument("--z-safe", type=float, default=20000.0, help="Safe Z position to start from (µm)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output JSON path (default: z_ramp_<timestamp>.json)"
    )
    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"z_ramp_{timestamp}.json")

    # Run measurement
    results = run_z_ramp_measure(
        x_distance_mm=args.x_distance,
        x_speed_mm_s=args.x_speed,
        z_delta_um=args.z_delta,
        z_safe_um=args.z_safe,
        output_path=args.output,
    )

    if "error" in results:
        print(f"\nMeasurement failed: {results['error']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
