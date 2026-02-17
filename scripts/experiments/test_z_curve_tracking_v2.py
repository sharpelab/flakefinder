"""Test Z axis tracking with state feedback + feedforward control.

Implements a proper trajectory tracking controller:
  commanded_vel = Kp * error_pos + Kv * error_vel + Kff * feedforward_vel

Where:
  - error_pos = z_target(x) - z_actual (position error)
  - error_vel = z_vel_target - z_vel_actual (velocity error)
  - feedforward_vel = dZ/dX * v_x (velocity from surface slope)

This replaces the pure feedforward approach in v1, providing:
  - Self-correcting position error feedback
  - Damping from velocity error feedback
  - Anticipatory feedforward velocity

SAFETY: Same safeguards as v1 (pre-flight validation, runtime Z limit, exception handling).

Run on microscope PC.

Usage:
    python test_z_curve_tracking_v2.py
    python test_z_curve_tracking_v2.py --x-speed 10 --surface top
    python test_z_curve_tracking_v2.py --kp 2.0 --kv 0.5 --kff 1.0
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from z_curve_data import SURFACE_DATA, build_z_profile


def run_curve_tracking_v2(
    surface_name: str = "top",
    x_speed_mm_s: float = 10.0,
    z_safe_um: float = 20000.0,
    z_max_um: float = 22000.0,
    control_rate_hz: float = 50.0,
    kp: float = 3.0,  # Position error gain (1/s)
    kv: float = 0.3,  # Velocity error gain (dimensionless)
    kff: float = 1.0,  # Feedforward gain
    lookahead_ms: float = 50.0,  # Feedforward lookahead time (ms)
    output_path: Path | None = None,
) -> dict:
    """Run curved surface tracking test with state feedback + feedforward.

    Control law:
        commanded_vel = Kp * error_pos + Kv * error_vel + Kff * feedforward_vel

    Args:
        surface_name: "top" or "middle" surface profile.
        x_speed_mm_s: X velocity in mm/s.
        z_safe_um: Safe Z offset (µm) - shifts profile to start here.
        z_max_um: Hard Z limit (µm) - emergency halt if exceeded.
        control_rate_hz: Target rate for Z velocity updates.
        kp: Position error gain (converts µm error to µm/s velocity).
        kv: Velocity error gain (dimensionless).
        kff: Feedforward velocity gain (typically 1.0).
        lookahead_ms: Feedforward lookahead time (ms) - predicts where X will be.
        output_path: Optional path to save results JSON.

    Returns:
        Dict with test results and position samples.
    """
    from flakefinder.leica import LeicaConnection, Stage, ZDrive

    # Get surface data bounds
    surface = SURFACE_DATA[surface_name]
    x_points = [p[0] for p in surface["points"]]
    x_data_start = min(x_points)
    x_data_end = max(x_points)

    # Use data range directly
    x_start = x_data_start
    x_end = x_data_end

    # Build Z profile
    z_func, dzdx_func, profile_info = build_z_profile(surface_name, x_start, x_end)

    # Apply safety offset
    z_at_start = z_func(x_start)
    z_offset = z_safe_um - z_at_start

    def z_target(x_um: float) -> float:
        return z_func(x_um) + z_offset

    x_speed_um_s = x_speed_mm_s * 1000
    x_distance_um = x_end - x_start
    expected_duration_s = x_distance_um / x_speed_um_s

    print("Z Curve Tracking V2 (State Feedback + Feedforward)")
    print("=" * 60)
    print(f"Surface: {surface_name} (Y={surface['y_um']} µm)")
    print(f"X range: {x_start:.0f} -> {x_end:.0f} µm ({x_distance_um / 1000:.1f} mm)")
    print(f"X speed: {x_speed_mm_s:.1f} mm/s")
    print(f"Expected duration: {expected_duration_s:.2f} s")
    print()
    print("Control gains:")
    print(f"  Kp (position): {kp:.2f} /s")
    print(f"  Kv (velocity): {kv:.2f}")
    print(f"  Kff (feedforward): {kff:.2f}")
    print(f"  Lookahead: {lookahead_ms:.0f} ms")
    print()
    print("Profile info:")
    print(f"  Z range: {profile_info['z_range_um'][0]:.1f} - {profile_info['z_range_um'][1]:.1f} µm")
    print(f"  Z delta: {profile_info['z_delta_um']:+.1f} µm")
    print(f"  Max gradient: {profile_info['max_gradient_um_per_mm']:.2f} µm/mm")
    print(f"  Max Z velocity needed: {profile_info['max_gradient_um_per_mm'] * x_speed_mm_s:.1f} µm/s")
    print(f"  Control rate target: {control_rate_hz:.0f} Hz")
    print()

    # === SAFETY: Pre-flight Z range validation ===
    x_test_points = np.linspace(x_start, x_end, 100)
    z_test_values = np.array([z_target(x) for x in x_test_points])
    z_min_expected = float(z_test_values.min())
    z_max_expected = float(z_test_values.max())

    print("SAFETY CHECK:")
    print(f"  Expected Z range: {z_min_expected:.0f} - {z_max_expected:.0f} µm")
    print(f"  Z hard limit: {z_max_um:.0f} µm")

    if z_max_expected > z_max_um:
        print(f"  ABORT: Expected max Z ({z_max_expected:.0f}) exceeds limit ({z_max_um:.0f})")
        return {"error": f"Pre-flight check failed: max Z {z_max_expected:.0f} > limit {z_max_um:.0f}"}

    margin = z_max_um - z_max_expected
    print(f"  OK: {margin:.0f} um margin below limit")
    print()

    # Initialize safety tracking
    emergency_stop = threading.Event()
    emergency_reason = [None]

    with LeicaConnection() as conn:
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        # Get interfaces for fast polling
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter

        # Save initial positions
        initial_x = stage.x.position_um
        initial_y = stage.y.position_um
        initial_z = z_drive.position_um

        print(f"Initial position: X={initial_x:.0f}, Y={initial_y:.0f}, Z={initial_z:.0f} µm")
        print(f"Z offset applied: {z_offset:+.1f} µm")

        # Check directed velocity support
        if not stage.x.supports_directed_velocity:
            return {"error": "X axis does not support directed velocity"}
        if not z_drive.supports_directed_velocity:
            return {"error": "Z axis does not support directed velocity"}

        try:
            # Move to start position
            z_start = z_target(x_start)
            print(f"\nMoving to start: X={x_start:.0f}, Z={z_start:.0f} µm...")
            z_drive.move_to(z_start)
            time.sleep(0.2)
            stage.x.move_to(x_start)
            time.sleep(0.2)

            actual_x_start = stage.x.position_um
            actual_z_start = z_drive.position_um
            print(f"At start: X={actual_x_start:.0f}, Z={actual_z_start:.0f} µm")

            # Sampling storage
            x_samples = []  # [(t, x_um), ...]
            z_samples = []  # [(t, z_um), ...]
            control_log = []  # Extended logging for v2
            stop_polling = threading.Event()
            stop_control = threading.Event()

            def poll_x():
                """Poll X position continuously."""
                while not stop_polling.is_set():
                    t = time.perf_counter()
                    x_native = x_bcv.GetControlValue()
                    x_um = x_converter.GetMetricsValue(x_native)
                    x_samples.append((t, x_um))

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
                    return x_samples[-1]  # (t, x_um)
                return (time.perf_counter(), x_start)

            def get_latest_z() -> tuple[float, float]:
                """Get latest Z position and timestamp from polling thread."""
                if z_samples:
                    return z_samples[-1]  # (t, z_um)
                return (time.perf_counter(), z_start)

            # Control loop timing log
            control_timing = []  # [(t, loop_duration_ms), ...]

            def control_z():
                """State feedback + feedforward Z velocity control."""
                control_interval = 1.0 / control_rate_hz

                # State tracking
                last_z = z_start
                last_t_z = time.perf_counter()  # Timestamp of last Z sample
                last_z_vel = 0.0  # Last computed Z velocity
                last_cmd_vel = 0.0
                last_loop_t = time.perf_counter()

                while not stop_control.is_set() and not emergency_stop.is_set():
                    t_now = time.perf_counter()

                    # Read from polling threads (no SDK calls in control loop)
                    _, current_x = get_latest_x()
                    t_z, current_z = get_latest_z()

                    # Skip iteration if no fresh Z sample
                    if t_z == last_t_z:
                        time.sleep(0.001)  # Brief sleep to avoid busy-wait
                        continue

                    # Track loop timing (only for iterations with fresh data)
                    loop_interval = (t_now - last_loop_t) * 1000  # ms
                    control_timing.append((t_now, loop_interval))
                    last_loop_t = t_now

                    # === SAFETY: Runtime Z limit check ===
                    if current_z > z_max_um:
                        emergency_reason[0] = f"Z exceeded limit: {current_z:.0f} > {z_max_um:.0f}"
                        emergency_stop.set()
                        try:
                            z_drive.halt()
                            stage.x.halt()
                        except Exception:
                            pass
                        break

                    # Compute actual Z velocity from sample timestamps (not control loop dt)
                    dt_z = t_z - last_t_z
                    if dt_z > 0.001:
                        actual_z_vel = (current_z - last_z) / dt_z  # µm/s
                    else:
                        actual_z_vel = last_z_vel  # Keep previous velocity if no new sample

                    # Clamp X to data range for spline evaluation
                    x_clamped = max(x_start, min(x_end, current_x))

                    # === State Feedback + Feedforward Control Law ===
                    # Use ideal X velocity (commanded) for all curve lookups
                    ideal_x_vel = x_speed_um_s

                    # Ideal state from curve at current X
                    ideal_z = z_target(x_clamped)
                    ideal_z_vel = dzdx_func(x_clamped) * ideal_x_vel

                    # Feedforward: look ahead using ideal X velocity
                    x_lookahead = current_x + ideal_x_vel * (lookahead_ms / 1000.0)
                    x_lookahead_clamped = max(x_start, min(x_end, x_lookahead))
                    z_vel_feedforward = dzdx_func(x_lookahead_clamped) * ideal_x_vel

                    # Errors: ideal vs actual
                    error_pos = ideal_z - current_z  # Positive = Z too low
                    error_vel = ideal_z_vel - actual_z_vel  # Positive = Z too slow

                    # Control law: u = Kp * e_p + Kv * e_v + Kff * v_ff
                    vel_from_pos_error = kp * error_pos
                    vel_from_vel_error = kv * error_vel
                    vel_from_feedforward = kff * z_vel_feedforward

                    commanded_vel = vel_from_pos_error + vel_from_vel_error + vel_from_feedforward

                    # Update tracking for next iteration (only update Z state if new sample)
                    if t_z != last_t_z:
                        last_z = current_z
                        last_t_z = t_z
                        last_z_vel = actual_z_vel

                    # === SAFETY: Limit velocity near Z max ===
                    z_headroom = z_max_um - current_z
                    if z_headroom < 100 and commanded_vel > 0:
                        scale = max(0, z_headroom / 100)
                        commanded_vel = commanded_vel * scale

                    # Command Z velocity (only if changed significantly)
                    if abs(commanded_vel - last_cmd_vel) > 0.5:
                        if commanded_vel > 0:
                            z_drive.start_towards_max(abs(commanded_vel))
                        elif commanded_vel < 0:
                            z_drive.start_towards_min(abs(commanded_vel))
                        else:
                            z_drive.halt()

                        # Extended control log for v2
                        control_log.append(
                            {
                                "t": t_now,
                                "x_um": current_x,
                                "z_actual_um": current_z,
                                "z_target_um": ideal_z,
                                "z_vel_actual_um_s": actual_z_vel,
                                "z_vel_ideal_um_s": ideal_z_vel,
                                "z_vel_ff_um_s": z_vel_feedforward,
                                "error_pos_um": error_pos,
                                "error_vel_um_s": error_vel,
                                "vel_from_pos": vel_from_pos_error,
                                "vel_from_vel": vel_from_vel_error,
                                "vel_from_ff": vel_from_feedforward,
                                "commanded_vel_um_s": commanded_vel,
                            }
                        )
                        last_cmd_vel = commanded_vel

                    # Sleep for control interval
                    elapsed = time.perf_counter() - t_now
                    sleep_time = max(0, control_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            # Start polling threads
            print("\nStarting position polling...")
            x_thread = threading.Thread(target=poll_x, daemon=True)
            z_thread = threading.Thread(target=poll_z, daemon=True)
            x_thread.start()
            z_thread.start()
            time.sleep(0.1)

            # Record pre-motion samples
            len(x_samples)
            len(z_samples)

            # Start X motion and control simultaneously
            print("\nStarting synchronized X+Z motion with state feedback...")
            t_motion_start = time.perf_counter()
            stage.x.start_towards_max(x_speed_um_s)

            # Start control thread
            control_thread = threading.Thread(target=control_z, daemon=True)
            control_thread.start()

            # Wait until X reaches target or emergency stop
            timeout_s = expected_duration_s * 2
            target_reached = False
            while not target_reached and not emergency_stop.is_set():
                elapsed = time.perf_counter() - t_motion_start
                if elapsed > timeout_s:
                    print(f"  TIMEOUT after {elapsed:.2f}s")
                    break
                current_x = stage.x.position_um
                if current_x >= x_end:
                    target_reached = True
                time.sleep(0.001)

            # Halt both axes
            t_motion_end = time.perf_counter()

            if emergency_stop.is_set():
                print(f"\n  *** EMERGENCY STOP: {emergency_reason[0]} ***")
            stop_control.set()
            stage.x.halt()
            z_drive.halt()

            motion_duration = t_motion_end - t_motion_start
            print("\nMotion complete!")
            print(f"  Duration: {motion_duration * 1000:.0f} ms (expected: {expected_duration_s * 1000:.0f} ms)")

            # Brief settling period
            time.sleep(0.1)

            # Stop polling
            stop_polling.set()
            x_thread.join(timeout=0.5)
            z_thread.join(timeout=0.5)
            control_thread.join(timeout=0.5)

            final_x = stage.x.position_um
            final_z = z_drive.position_um

            print(f"  Final X: {final_x:.0f} µm (target: {x_end:.0f})")
            print(f"  Final Z: {final_z:.0f} µm (target: {z_target(x_end):.0f})")
            print(f"  Samples: X={len(x_samples)}, Z={len(z_samples)}")
            print(f"  Control updates: {len(control_log)}")

            # Analyze control loop timing
            if len(control_timing) > 2:
                intervals = np.array([t[1] for t in control_timing[1:]])  # Skip first (warmup)
                ctrl_freq = 1000 / np.mean(intervals)
                print("\n  Control loop timing:")
                print(f"    Iterations: {len(control_timing)}")
                print(f"    Interval: mean={np.mean(intervals):.2f}ms, std={np.std(intervals):.2f}ms")
                print(f"    Range: {np.min(intervals):.2f} - {np.max(intervals):.2f} ms")
                print(f"    Actual frequency: {ctrl_freq:.1f} Hz (target: {control_rate_hz:.0f} Hz)")

        finally:
            # Restore position
            print("\nRestoring initial position...")
            z_drive.move_to(initial_z)
            stage.x.move_to(initial_x)
            print(f"  Restored to: X={stage.x.position_um:.0f}, Z={z_drive.position_um:.0f} µm")

    # Analyze tracking error
    print("\nAnalyzing tracking error...")

    x_t = np.array([s[0] - t_motion_start for s in x_samples])
    x_pos = np.array([s[1] for s in x_samples])
    z_t = np.array([s[0] - t_motion_start for s in z_samples])
    z_pos = np.array([s[1] for s in z_samples])

    # Interpolate Z at X sample times
    z_at_x_times = np.interp(x_t, z_t, z_pos)

    # Compute ideal Z for each X position
    z_ideal = np.array([z_target(x) for x in x_pos])

    # Compute tracking error
    z_error = z_at_x_times - z_ideal

    # Filter to constant velocity region
    accel_margin_um = 2000
    mask = (x_pos >= x_start + accel_margin_um) & (x_pos <= x_end - accel_margin_um)

    if mask.sum() > 10:
        z_error_cv = z_error[mask]
        error_mean = float(np.mean(z_error_cv))
        error_std = float(np.std(z_error_cv))
        error_max = float(np.max(np.abs(z_error_cv)))
        error_p95 = float(np.percentile(np.abs(z_error_cv), 95))
    else:
        error_mean = error_std = error_max = error_p95 = float("nan")
        print("  WARNING: Not enough samples in constant velocity region")

    dof_20x = 4.0  # µm
    within_dof = error_max < dof_20x

    print("\nTracking Error (constant velocity region):")
    print(f"  Mean: {error_mean:+.2f} µm")
    print(f"  Std: {error_std:.2f} µm")
    print(f"  Max: {error_max:.2f} µm")
    print(f"  95th percentile: {error_p95:.2f} µm")
    print(f"  {'PASS:' if within_dof else 'FAIL:'} {'WITHIN' if within_dof else 'EXCEEDS'} 20x DOF ({dof_20x} um)")

    # Build results
    results = {
        "timestamp": datetime.now().isoformat(),
        "version": 2,
        "params": {
            "surface_name": surface_name,
            "x_speed_mm_s": x_speed_mm_s,
            "z_safe_um": z_safe_um,
            "z_max_um": z_max_um,
            "control_rate_hz": control_rate_hz,
            "kp": kp,
            "kv": kv,
            "kff": kff,
            "lookahead_ms": lookahead_ms,
        },
        "safety": {
            "z_max_expected_um": z_max_expected,
            "z_margin_um": z_max_um - z_max_expected,
            "emergency_stop": emergency_stop.is_set(),
            "emergency_reason": emergency_reason[0],
        },
        "profile": profile_info,
        "commanded": {
            "x_start_um": x_start,
            "x_end_um": x_end,
            "z_offset_um": z_offset,
            "expected_duration_s": expected_duration_s,
        },
        "timing": {
            "t_motion_start": t_motion_start,
            "t_motion_end": t_motion_end,
            "motion_duration_s": motion_duration,
        },
        "final": {
            "x_um": final_x,
            "z_um": final_z,
            "target_reached": target_reached,
        },
        "tracking": {
            "error_mean_um": error_mean,
            "error_std_um": error_std,
            "error_max_um": error_max,
            "error_p95_um": error_p95,
            "within_20x_dof": within_dof,
            "dof_20x_um": dof_20x,
        },
        "polling": {
            "x_samples": len(x_samples),
            "z_samples": len(z_samples),
            "control_updates": len(control_log),
            "control_iterations": len(control_timing),
        },
        "control_loop_timing": {
            "target_hz": control_rate_hz,
            "actual_hz": 1000 / np.mean([t[1] for t in control_timing[1:]]) if len(control_timing) > 2 else None,
            "interval_mean_ms": float(np.mean([t[1] for t in control_timing[1:]])) if len(control_timing) > 2 else None,
            "interval_std_ms": float(np.std([t[1] for t in control_timing[1:]])) if len(control_timing) > 2 else None,
            "interval_min_ms": float(np.min([t[1] for t in control_timing[1:]])) if len(control_timing) > 2 else None,
            "interval_max_ms": float(np.max([t[1] for t in control_timing[1:]])) if len(control_timing) > 2 else None,
        },
        "x_samples": [{"t": s[0], "x_um": s[1]} for s in x_samples],
        "z_samples": [{"t": s[0], "z_um": s[1]} for s in z_samples],
        "control_log": control_log,
        "control_timing": [{"t": t[0], "interval_ms": t[1]} for t in control_timing],
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test Z axis tracking with state feedback + feedforward",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--surface", choices=["top", "middle"], default="top", help="Surface profile to track")
    parser.add_argument("--x-speed", type=float, default=10.0, help="X velocity in mm/s")
    parser.add_argument(
        "--z-safe", type=float, default=20000.0, help="Safe Z offset (µm) - shifts profile to start here"
    )
    parser.add_argument("--z-max", type=float, default=22000.0, help="Hard Z limit (µm) - emergency halt if exceeded")
    parser.add_argument("--control-rate", type=float, default=50.0, help="Z velocity update rate (Hz)")
    parser.add_argument("--kp", type=float, default=3.0, help="Position error gain (1/s) - converts µm error to µm/s")
    parser.add_argument("--kv", type=float, default=0.3, help="Velocity error gain (dimensionless)")
    parser.add_argument("--kff", type=float, default=1.0, help="Feedforward velocity gain")
    parser.add_argument(
        "--lookahead-ms", type=float, default=50.0, help="Feedforward lookahead time (ms) - predicts where X will be"
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"z_curve_v2_{args.surface}_{int(args.x_speed)}mms_{timestamp}.json")

    results = run_curve_tracking_v2(
        surface_name=args.surface,
        x_speed_mm_s=args.x_speed,
        z_safe_um=args.z_safe,
        z_max_um=args.z_max,
        control_rate_hz=args.control_rate,
        kp=args.kp,
        kv=args.kv,
        kff=args.kff,
        lookahead_ms=args.lookahead_ms,
        output_path=args.output,
    )

    if "error" in results:
        print(f"\nTest failed: {results['error']}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
