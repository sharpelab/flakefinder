"""Measure Z axis step response characteristics.

Tests how quickly the Z axis responds to step changes, measuring:
- Rise time (10% to 90% of step)
- Settling time (to within ±2% of final)
- Overshoot (if any)
- Latency (command to first movement)

Run on microscope PC.

Usage:
    python test_z_step_response.py
    python test_z_step_response.py --step-size 20 --num-trials 5
    python test_z_step_response.py -o my_step_test.json
"""

import argparse
import json
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def run_step_response_test(
    step_size_um: float = 10.0,
    z_safe_um: float = 20000.0,
    settle_tolerance: float = 0.02,  # 2% of step for settling
    output_path: Path | None = None,
) -> dict:
    """Run a single Z step response test.

    Args:
        step_size_um: Size of step in µm (positive = towards max).
        z_safe_um: Safe Z position to start from.
        settle_tolerance: Fraction of step for settling criterion.
        output_path: Optional path to save results JSON.

    Returns:
        Dict with test results and position samples.
    """
    from flakefinder.leica import LeicaConnection, ZDrive

    print(f"Z Step Response Test")
    print(f"=" * 50)
    print(f"Step size: {step_size_um:+.1f} µm")
    print(f"Start position: {z_safe_um:.0f} µm")
    print(f"Settling tolerance: ±{settle_tolerance*100:.1f}% of step")
    print()

    with LeicaConnection() as conn:
        z_drive = ZDrive.from_connection(conn)

        # Get interface for fast polling
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter

        # Save initial position
        initial_z = z_drive.position_um
        print(f"Initial Z: {initial_z:.0f} µm (will restore)")

        # Calculate test positions
        z_start = z_safe_um
        z_target = z_safe_um + step_size_um
        step_direction = "towards_max" if step_size_um > 0 else "towards_min"

        print(f"\nStep trajectory: {z_start:.0f} -> {z_target:.0f} µm ({step_direction})")

        try:
            # Move to start position (blocking)
            print(f"\nMoving to start position ({z_start:.0f} µm)...")
            z_drive.move_to(z_start)
            time.sleep(0.3)  # Let settle

            actual_start = z_drive.position_um
            print(f"At start: Z={actual_start:.1f} µm")

            # Position sampling storage
            samples = []  # [(t_before, t_after, z_um), ...]
            stop_polling = threading.Event()

            def poll_z():
                """Poll Z position continuously."""
                while not stop_polling.is_set():
                    t_before = time.perf_counter()
                    z_native = z_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    z_um = z_converter.GetMetricsValue(z_native)
                    samples.append((t_before, t_after, z_um))

            # Start polling
            print(f"\nStarting position polling...")
            poll_thread = threading.Thread(target=poll_z, daemon=True)
            poll_thread.start()

            # Let polling warm up
            time.sleep(0.2)
            warmup_samples = len(samples)
            print(f"  Warmup polling rate: {warmup_samples/0.2:.1f} Hz")

            # Mark pre-step samples
            samples_at_step_start = len(samples)

            # Command the step (non-blocking)
            print(f"\nCommanding step: move_to_async({z_target:.0f} µm)...")
            t_command_start = time.perf_counter()
            handle = z_drive.move_to_async(z_target)
            t_command_end = time.perf_counter()

            # Wait for completion with timeout
            timeout_s = 5.0
            completed = handle.wait(timeout=timeout_s)
            t_complete = time.perf_counter()

            if not completed:
                print(f"  WARNING: Timeout after {timeout_s}s")

            # Continue polling briefly for settling
            time.sleep(0.2)

            # Stop polling
            stop_polling.set()
            poll_thread.join(timeout=1.0)

            # Get final position
            final_z = z_drive.position_um

            motion_time = t_complete - t_command_end
            command_latency = (t_command_end - t_command_start) * 1000

            print(f"\nStep complete!")
            print(f"  Command latency: {command_latency:.1f} ms")
            print(f"  Motion time: {motion_time*1000:.1f} ms")
            print(f"  Final Z: {final_z:.1f} µm (target: {z_target:.0f}, error: {final_z - z_target:+.2f})")
            print(f"  Total samples: {len(samples)} ({len(samples)/motion_time:.1f} Hz during motion)")

        finally:
            # Restore position
            print(f"\nRestoring initial position ({initial_z:.0f} µm)...")
            z_drive.move_to(initial_z)
            time.sleep(0.1)
            restored_z = z_drive.position_um
            print(f"  Restored to: Z={restored_z:.0f} µm")

    # Analyze step response
    print(f"\nAnalyzing step response...")

    # Convert to numpy arrays
    t_samples = np.array([(s[0] + s[1]) / 2 for s in samples])
    z_samples = np.array([s[2] for s in samples])
    t_rel = t_samples - t_command_end  # Relative to command completion

    # Find actual start and end positions from data
    pre_step_z = np.mean(z_samples[:samples_at_step_start]) if samples_at_step_start > 10 else actual_start
    post_step_z = np.mean(z_samples[-20:]) if len(samples) > 20 else final_z
    actual_step = post_step_z - pre_step_z

    print(f"  Pre-step Z: {pre_step_z:.2f} µm")
    print(f"  Post-step Z: {post_step_z:.2f} µm")
    print(f"  Actual step: {actual_step:+.2f} µm (commanded: {step_size_um:+.1f})")

    # Normalize position to 0-1 scale for step response analysis
    z_normalized = (z_samples - pre_step_z) / actual_step if abs(actual_step) > 0.1 else z_samples

    # Find key metrics
    # Motion start: first sample > 10% of step
    motion_start_idx = None
    for i, z_norm in enumerate(z_normalized):
        if z_norm > 0.1:
            motion_start_idx = i
            break

    # 10% and 90% times (for rise time)
    t_10pct = None
    t_90pct = None
    for i, (t, z_norm) in enumerate(zip(t_rel, z_normalized)):
        if t_10pct is None and z_norm >= 0.1:
            t_10pct = t
        if t_90pct is None and z_norm >= 0.9:
            t_90pct = t
            break

    rise_time_ms = (t_90pct - t_10pct) * 1000 if (t_10pct and t_90pct) else None

    # Overshoot: max value above 1.0
    max_normalized = np.max(z_normalized)
    overshoot_pct = (max_normalized - 1.0) * 100 if max_normalized > 1.0 else 0.0

    # Settling time: time to stay within ±tolerance of final value
    settle_band = settle_tolerance
    t_settle = None
    for i in range(len(z_normalized) - 1, -1, -1):
        if abs(z_normalized[i] - 1.0) > settle_band:
            t_settle = t_rel[min(i + 1, len(t_rel) - 1)]
            break
    settling_time_ms = t_settle * 1000 if t_settle else 0.0

    # Latency: command end to first movement (10% threshold)
    latency_ms = t_10pct * 1000 if t_10pct and t_10pct > 0 else 0.0

    # Per-sample SDK timing
    read_times_ms = [(s[1] - s[0]) * 1000 for s in samples]
    mean_read_time = statistics.mean(read_times_ms) if read_times_ms else 0

    print(f"\nStep Response Metrics:")
    print(f"  Latency (to 10%): {latency_ms:.1f} ms")
    print(f"  Rise time (10-90%): {rise_time_ms:.1f} ms" if rise_time_ms else "  Rise time: N/A")
    print(f"  Settling time (±{settle_tolerance*100:.0f}%): {settling_time_ms:.1f} ms")
    print(f"  Overshoot: {overshoot_pct:.1f}%")
    print(f"  SDK read time: {mean_read_time:.2f} ms")

    # Build results
    results = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "step_size_um": step_size_um,
            "z_safe_um": z_safe_um,
            "settle_tolerance": settle_tolerance,
        },
        "commanded": {
            "z_start_um": z_start,
            "z_target_um": z_target,
            "step_direction": step_direction,
        },
        "actual": {
            "pre_step_z_um": float(pre_step_z),
            "post_step_z_um": float(post_step_z),
            "actual_step_um": float(actual_step),
            "final_z_um": final_z,
            "position_error_um": final_z - z_target,
        },
        "timing": {
            "t_command_start": t_command_start,
            "t_command_end": t_command_end,
            "t_complete": t_complete,
            "command_latency_ms": command_latency,
            "motion_time_ms": motion_time * 1000,
        },
        "metrics": {
            "latency_to_10pct_ms": latency_ms,
            "rise_time_10_90_ms": rise_time_ms,
            "settling_time_ms": settling_time_ms,
            "overshoot_percent": overshoot_pct,
        },
        "polling": {
            "total_samples": len(samples),
            "warmup_samples": warmup_samples,
            "samples_at_step_start": samples_at_step_start,
            "poll_rate_hz": len(samples) / (t_samples[-1] - t_samples[0]) if len(samples) > 1 else 0,
            "sdk_read_time_ms": mean_read_time,
        },
        "samples": [
            {"t_before": s[0], "t_after": s[1], "z_um": s[2]}
            for s in samples
        ],
    }

    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def run_multiple_trials(
    step_size_um: float = 10.0,
    num_trials: int = 5,
    z_safe_um: float = 20000.0,
    settle_tolerance: float = 0.02,
    output_path: Path | None = None,
) -> dict:
    """Run multiple step response trials and aggregate statistics.

    Args:
        step_size_um: Size of step in µm.
        num_trials: Number of trials to run.
        z_safe_um: Safe Z position to start from.
        settle_tolerance: Fraction of step for settling criterion.
        output_path: Optional path to save results JSON.

    Returns:
        Dict with aggregated statistics and individual trial results.
    """
    from flakefinder.leica import LeicaConnection, ZDrive

    print(f"Z Step Response - Multiple Trials")
    print(f"=" * 50)
    print(f"Step size: {step_size_um:+.1f} µm")
    print(f"Number of trials: {num_trials}")
    print()

    trials = []
    metrics_lists = {
        "latency_ms": [],
        "rise_time_ms": [],
        "settling_time_ms": [],
        "overshoot_pct": [],
    }

    with LeicaConnection() as conn:
        z_drive = ZDrive.from_connection(conn)
        z_bcv = z_drive.bcv
        z_converter = z_drive.converter

        initial_z = z_drive.position_um
        print(f"Initial Z: {initial_z:.0f} µm (will restore)\n")

        z_start = z_safe_um
        z_target = z_safe_um + step_size_um

        try:
            for trial_num in range(num_trials):
                print(f"--- Trial {trial_num + 1}/{num_trials} ---")

                # Move to start
                z_drive.move_to(z_start)
                time.sleep(0.3)

                # Position sampling
                samples = []
                stop_polling = threading.Event()

                def poll_z():
                    while not stop_polling.is_set():
                        t_before = time.perf_counter()
                        z_native = z_bcv.GetControlValue()
                        t_after = time.perf_counter()
                        z_um = z_converter.GetMetricsValue(z_native)
                        samples.append((t_before, t_after, z_um))

                poll_thread = threading.Thread(target=poll_z, daemon=True)
                poll_thread.start()
                time.sleep(0.1)
                samples_at_start = len(samples)

                # Command step
                t_cmd = time.perf_counter()
                handle = z_drive.move_to_async(z_target)
                handle.wait(timeout=5.0)
                t_done = time.perf_counter()

                time.sleep(0.2)
                stop_polling.set()
                poll_thread.join(timeout=0.5)

                # Analyze this trial
                t_samples = np.array([(s[0] + s[1]) / 2 - t_cmd for s in samples])
                z_samples = np.array([s[2] for s in samples])

                pre_z = np.mean(z_samples[:samples_at_start]) if samples_at_start > 5 else z_start
                post_z = np.mean(z_samples[-10:])
                actual_step = post_z - pre_z

                if abs(actual_step) > 0.1:
                    z_norm = (z_samples - pre_z) / actual_step

                    # Find metrics
                    t_10pct = None
                    t_90pct = None
                    for t, z_n in zip(t_samples, z_norm):
                        if t_10pct is None and z_n >= 0.1:
                            t_10pct = t
                        if t_90pct is None and z_n >= 0.9:
                            t_90pct = t
                            break

                    latency = t_10pct * 1000 if t_10pct and t_10pct > 0 else 0
                    rise_time = (t_90pct - t_10pct) * 1000 if (t_10pct and t_90pct) else None

                    max_norm = np.max(z_norm)
                    overshoot = (max_norm - 1.0) * 100 if max_norm > 1.0 else 0

                    t_settle = 0
                    for i in range(len(z_norm) - 1, -1, -1):
                        if abs(z_norm[i] - 1.0) > settle_tolerance:
                            t_settle = t_samples[min(i + 1, len(t_samples) - 1)] * 1000
                            break

                    metrics_lists["latency_ms"].append(latency)
                    if rise_time is not None:
                        metrics_lists["rise_time_ms"].append(rise_time)
                    metrics_lists["settling_time_ms"].append(t_settle)
                    metrics_lists["overshoot_pct"].append(overshoot)

                    print(f"  Rise: {rise_time:.1f}ms, Settle: {t_settle:.1f}ms, Overshoot: {overshoot:.1f}%")

                    trials.append({
                        "trial": trial_num + 1,
                        "latency_ms": latency,
                        "rise_time_ms": rise_time,
                        "settling_time_ms": t_settle,
                        "overshoot_pct": overshoot,
                        "actual_step_um": actual_step,
                        "motion_time_ms": (t_done - t_cmd) * 1000,
                    })
                else:
                    print(f"  WARNING: No motion detected")

                time.sleep(0.2)  # Brief pause between trials

        finally:
            # Restore
            print(f"\nRestoring initial position...")
            z_drive.move_to(initial_z)

    # Compute statistics
    print(f"\n{'='*50}")
    print(f"SUMMARY ({len(trials)} successful trials)")
    print(f"{'='*50}")

    def stat_str(values, unit=""):
        if not values:
            return "N/A"
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return f"{mean:.1f} ± {std:.1f} {unit}"

    print(f"Latency (to 10%):      {stat_str(metrics_lists['latency_ms'], 'ms')}")
    print(f"Rise time (10-90%):    {stat_str(metrics_lists['rise_time_ms'], 'ms')}")
    print(f"Settling time (±{settle_tolerance*100:.0f}%): {stat_str(metrics_lists['settling_time_ms'], 'ms')}")
    print(f"Overshoot:             {stat_str(metrics_lists['overshoot_pct'], '%')}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "step_size_um": step_size_um,
            "num_trials": num_trials,
            "z_safe_um": z_safe_um,
            "settle_tolerance": settle_tolerance,
        },
        "summary": {
            "latency_ms": {
                "mean": statistics.mean(metrics_lists["latency_ms"]) if metrics_lists["latency_ms"] else None,
                "std": statistics.stdev(metrics_lists["latency_ms"]) if len(metrics_lists["latency_ms"]) > 1 else None,
            },
            "rise_time_ms": {
                "mean": statistics.mean(metrics_lists["rise_time_ms"]) if metrics_lists["rise_time_ms"] else None,
                "std": statistics.stdev(metrics_lists["rise_time_ms"]) if len(metrics_lists["rise_time_ms"]) > 1 else None,
            },
            "settling_time_ms": {
                "mean": statistics.mean(metrics_lists["settling_time_ms"]) if metrics_lists["settling_time_ms"] else None,
                "std": statistics.stdev(metrics_lists["settling_time_ms"]) if len(metrics_lists["settling_time_ms"]) > 1 else None,
            },
            "overshoot_pct": {
                "mean": statistics.mean(metrics_lists["overshoot_pct"]) if metrics_lists["overshoot_pct"] else None,
                "std": statistics.stdev(metrics_lists["overshoot_pct"]) if len(metrics_lists["overshoot_pct"]) > 1 else None,
            },
        },
        "trials": trials,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure Z axis step response characteristics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step-size", type=float, default=10.0,
        help="Step size in µm (positive = towards max)"
    )
    parser.add_argument(
        "--z-safe", type=float, default=20000.0,
        help="Safe Z position to start from (µm)"
    )
    parser.add_argument(
        "--num-trials", type=int, default=1,
        help="Number of trials (>1 runs multi-trial mode with statistics)"
    )
    parser.add_argument(
        "--settle-tolerance", type=float, default=0.02,
        help="Settling criterion as fraction of step (e.g., 0.02 = ±2%%)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output JSON path"
    )
    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"z_step_response_{timestamp}.json")

    if args.num_trials > 1:
        results = run_multiple_trials(
            step_size_um=args.step_size,
            num_trials=args.num_trials,
            z_safe_um=args.z_safe,
            settle_tolerance=args.settle_tolerance,
            output_path=args.output,
        )
    else:
        results = run_step_response_test(
            step_size_um=args.step_size,
            z_safe_um=args.z_safe,
            settle_tolerance=args.settle_tolerance,
            output_path=args.output,
        )

    return 0


if __name__ == "__main__":
    exit(main())
