"""Analyze Z ramp measurement data.

Analyzes data collected by test_z_ramp_measure.py. Run locally.

Usage:
    python test_z_ramp_analyze.py z_ramp_20260205_123456.json
    python test_z_ramp_analyze.py z_ramp_*.json  # Compare multiple
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def interpolate_z_at_x_times(x_times, x_positions, z_times, z_positions):
    """Interpolate Z positions at X sample times for correlation.

    Returns arrays of matched (x, z) pairs at X sample times.
    """
    # Only interpolate within Z time range
    z_min_t, z_max_t = z_times.min(), z_times.max()
    valid_mask = (x_times >= z_min_t) & (x_times <= z_max_t)

    x_times_valid = x_times[valid_mask]
    x_positions_valid = x_positions[valid_mask]

    # Interpolate Z at X times
    z_at_x_times = np.interp(x_times_valid, z_times, z_positions)

    return x_times_valid, x_positions_valid, z_at_x_times


def analyze_z_ramp(data: dict) -> dict:
    """Analyze Z ramp measurement data."""
    params = data["params"]
    commanded = data["commanded"]
    timing = data["timing"]

    # Separate streams
    x_samples = data["x_samples"]
    z_samples = data["z_samples"]

    # Use midpoint time for each sample
    x_times = np.array([(s["t_before"] + s["t_after"]) / 2 for s in x_samples])
    x_positions = np.array([s["x_um"] for s in x_samples])
    z_times = np.array([(s["t_before"] + s["t_after"]) / 2 for s in z_samples])
    z_positions = np.array([s["z_um"] for s in z_samples])

    # Per-sample timing
    x_read_times = np.array([(s["t_after"] - s["t_before"]) * 1000 for s in x_samples])
    z_read_times = np.array([(s["t_after"] - s["t_before"]) * 1000 for s in z_samples])

    # Normalize times to motion start
    t_start = timing["t_after_start"]
    x_times_rel = x_times - t_start
    z_times_rel = z_times - t_start

    # Find X motion period (when X is actually moving)
    x_diff = np.diff(x_positions)
    moving_mask = np.abs(x_diff) > 10  # Moving if >10µm change

    if not np.any(moving_mask):
        return {"error": "No motion detected in X"}

    moving_indices = np.where(moving_mask)[0]
    motion_start_idx = moving_indices[0]
    motion_end_idx = moving_indices[-1] + 1

    # Extract motion period for X
    x_times_motion = x_times[motion_start_idx:motion_end_idx+1]
    x_positions_motion = x_positions[motion_start_idx:motion_end_idx+1]

    motion_duration = x_times_motion[-1] - x_times_motion[0]
    x_traveled = x_positions_motion[-1] - x_positions_motion[0]

    # Interpolate Z at X sample times during motion
    t_matched, x_matched, z_matched = interpolate_z_at_x_times(
        x_times_motion, x_positions_motion, z_times, z_positions
    )

    z_traveled = z_matched[-1] - z_matched[0]

    actual_x_velocity = x_traveled / motion_duration / 1000  # mm/s
    actual_z_velocity = z_traveled / motion_duration  # µm/s

    # Compute ideal Z for each X position (linear ramp)
    x_start = commanded["x_start_um"]
    z_start = commanded["z_start_um"]
    z_delta = params["z_delta_um"]
    x_distance = params["x_distance_mm"] * 1000
    slope = z_delta / x_distance  # µm Z per µm X
    z_ideal = z_start + (x_matched - x_start) * slope

    # Compute errors
    z_error = z_matched - z_ideal
    z_error_mean = np.mean(z_error)
    z_error_std = np.std(z_error)
    z_error_max = np.max(np.abs(z_error))
    z_error_p95 = np.percentile(np.abs(z_error), 95)

    # Linear fit of actual X vs Z
    coeffs = np.polyfit(x_matched, z_matched, 1)
    z_fit = np.polyval(coeffs, x_matched)
    fit_residuals = z_matched - z_fit
    ss_res = np.sum(fit_residuals**2)
    ss_tot = np.sum((z_matched - np.mean(z_matched))**2)
    fit_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # DOF check
    dof_20x = 4.0  # µm

    # Polling rates
    x_poll_rate = data["polling"]["x_poll_rate_hz"]
    z_poll_rate = data["polling"]["z_poll_rate_hz"]

    # Sample intervals
    x_intervals = np.diff(x_times) * 1000  # ms
    z_intervals = np.diff(z_times) * 1000  # ms

    return {
        "params": params,
        "commanded": commanded,
        "timing": {
            "start_command_latency_ms": timing["start_command_latency_ms"],
            "x_duration_s": timing["x_duration_s"],
            "z_duration_s": timing["z_duration_s"],
            "completion_diff_ms": timing["completion_diff_ms"],
            "expected_duration_s": commanded["expected_duration_s"],
        },
        "motion": {
            "duration_s": motion_duration,
            "x_traveled_um": x_traveled,
            "z_traveled_um": z_traveled,
            "actual_x_velocity_mm_s": actual_x_velocity,
            "actual_z_velocity_um_s": actual_z_velocity,
            "commanded_x_velocity_mm_s": params["x_speed_mm_s"],
            "commanded_z_velocity_um_s": commanded["z_velocity_um_s"],
        },
        "tracking": {
            "z_error_mean_um": z_error_mean,
            "z_error_std_um": z_error_std,
            "z_error_max_um": z_error_max,
            "z_error_p95_um": z_error_p95,
            "within_20x_dof": z_error_max < dof_20x,
            "dof_20x_um": dof_20x,
        },
        "linearity": {
            "slope_um_per_mm": coeffs[0] * 1000,
            "expected_slope_um_per_mm": slope * 1000,
            "slope_error_percent": abs(coeffs[0] * 1000 - slope * 1000) / abs(slope * 1000) * 100 if slope != 0 else 0,
            "intercept_um": coeffs[1],
            "r_squared": fit_r2,
            "residual_std_um": float(np.std(fit_residuals)),
        },
        "polling": {
            "x_total_samples": len(x_samples),
            "z_total_samples": len(z_samples),
            "x_poll_rate_hz": x_poll_rate,
            "z_poll_rate_hz": z_poll_rate,
            "x_interval_mean_ms": float(np.mean(x_intervals)),
            "x_interval_std_ms": float(np.std(x_intervals)),
            "z_interval_mean_ms": float(np.mean(z_intervals)),
            "z_interval_std_ms": float(np.std(z_intervals)),
            "x_read_time_mean_ms": float(np.mean(x_read_times)),
            "z_read_time_mean_ms": float(np.mean(z_read_times)),
            "matched_samples": len(x_matched),
        },
        # Raw data for plotting
        "_arrays": {
            "x_times_rel": x_times_rel,
            "x_positions": x_positions,
            "z_times_rel": z_times_rel,
            "z_positions": z_positions,
            "t_matched": t_matched - t_start,
            "x_matched": x_matched,
            "z_matched": z_matched,
            "z_ideal": z_ideal,
            "z_error": z_error,
            "z_fit": z_fit,
            "x_intervals": x_intervals,
            "z_intervals": z_intervals,
            "motion_start_idx": motion_start_idx,
            "motion_end_idx": motion_end_idx,
        },
    }


def print_report(analysis: dict, filename: str = "") -> None:
    """Print analysis report."""
    print()
    print("=" * 60)
    print(f"Z RAMP ANALYSIS{f': {filename}' if filename else ''}")
    print("=" * 60)

    params = analysis["params"]
    print(f"\nTest parameters:")
    print(f"  X distance: {params['x_distance_mm']:.1f} mm")
    print(f"  X speed: {params['x_speed_mm_s']:.1f} mm/s")
    print(f"  Z delta: {params['z_delta_um']:+.1f} µm")

    timing = analysis["timing"]
    print(f"\nTiming:")
    print(f"  Start command latency: {timing['start_command_latency_ms']:.1f} ms")
    print(f"  X duration: {timing['x_duration_s']*1000:.1f} ms (expected: {timing['expected_duration_s']*1000:.1f} ms)")
    print(f"  Z duration: {timing['z_duration_s']*1000:.1f} ms")
    print(f"  Completion difference: {timing['completion_diff_ms']:.1f} ms")

    motion = analysis["motion"]
    print(f"\nActual motion:")
    print(f"  Duration: {motion['duration_s']:.3f} s")
    print(f"  X velocity: {motion['actual_x_velocity_mm_s']:.2f} mm/s (commanded: {motion['commanded_x_velocity_mm_s']:.1f})")
    print(f"  Z velocity: {motion['actual_z_velocity_um_s']:.2f} µm/s (commanded: {motion['commanded_z_velocity_um_s']:.2f})")

    tracking = analysis["tracking"]
    print(f"\nZ tracking error (vs ideal linear ramp):")
    print(f"  Mean: {tracking['z_error_mean_um']:+.2f} µm")
    print(f"  Std: {tracking['z_error_std_um']:.2f} µm")
    print(f"  Max absolute: {tracking['z_error_max_um']:.2f} µm")
    print(f"  95th percentile: {tracking['z_error_p95_um']:.2f} µm")

    dof = tracking["dof_20x_um"]
    if tracking["within_20x_dof"]:
        print(f"  ✓ Max error ({tracking['z_error_max_um']:.2f} µm) is WITHIN 20x DOF ({dof} µm)")
    else:
        print(f"  ✗ Max error ({tracking['z_error_max_um']:.2f} µm) EXCEEDS 20x DOF ({dof} µm)")

    lin = analysis["linearity"]
    print(f"\nLinearity (X vs Z fit):")
    print(f"  Slope: {lin['slope_um_per_mm']:.4f} µm/mm (expected: {lin['expected_slope_um_per_mm']:.4f})")
    print(f"  Slope error: {lin['slope_error_percent']:.2f}%")
    print(f"  R²: {lin['r_squared']:.6f}")
    print(f"  Residual std: {lin['residual_std_um']:.3f} µm")

    poll = analysis["polling"]
    print(f"\nPolling performance (parallel threads):")
    print(f"  X: {poll['x_total_samples']} samples, {poll['x_poll_rate_hz']:.1f} Hz, interval {poll['x_interval_mean_ms']:.2f} ± {poll['x_interval_std_ms']:.2f} ms")
    print(f"  Z: {poll['z_total_samples']} samples, {poll['z_poll_rate_hz']:.1f} Hz, interval {poll['z_interval_mean_ms']:.2f} ± {poll['z_interval_std_ms']:.2f} ms")
    print(f"  SDK read time: X={poll['x_read_time_mean_ms']:.2f}ms, Z={poll['z_read_time_mean_ms']:.2f}ms")
    print(f"  Matched samples for analysis: {poll['matched_samples']}")

    print()
    print("=" * 60)
    if tracking["within_20x_dof"]:
        print("✓ TEST PASSED: Z tracking error within 20x DOF")
    else:
        print("✗ TEST FAILED: Z tracking error exceeds 20x DOF")
    print("=" * 60)


def plot_results(analysis: dict, output_path: Path | None = None, title_suffix: str = "") -> None:
    """Plot Z ramp analysis results."""
    arr = analysis["_arrays"]
    params = analysis["params"]
    tracking = analysis["tracking"]
    lin = analysis["linearity"]
    poll = analysis["polling"]

    x_times_rel = arr["x_times_rel"]
    x_positions = arr["x_positions"]
    z_times_rel = arr["z_times_rel"]
    z_positions = arr["z_positions"]
    t_matched = arr["t_matched"]
    x_matched = arr["x_matched"]
    z_matched = arr["z_matched"]
    z_ideal = arr["z_ideal"]
    z_error = arr["z_error"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    title = f"Z Ramp Test: {params['x_distance_mm']:.0f}mm @ {params['x_speed_mm_s']:.0f}mm/s, ΔZ={params['z_delta_um']:+.0f}µm"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=12)

    # 1. X and Z vs time (separate traces)
    ax1 = axes[0, 0]
    ax1_z = ax1.twinx()

    ax1.plot(x_times_rel * 1000, x_positions / 1000, 'b-', label='X', linewidth=0.5, alpha=0.8)
    ax1_z.plot(z_times_rel * 1000, z_positions, 'r-', label='Z', linewidth=0.5, alpha=0.8)

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('X (mm)', color='b')
    ax1_z.set_ylabel('Z (µm)', color='r')
    ax1.set_title(f'Position vs Time (X: {poll["x_poll_rate_hz"]:.0f}Hz, Z: {poll["z_poll_rate_hz"]:.0f}Hz)')

    # 2. X vs Z scatter (matched samples)
    ax2 = axes[0, 1]
    ax2.scatter(x_matched / 1000, z_matched, s=3, alpha=0.5, label='Actual', zorder=2)
    ax2.plot(x_matched / 1000, z_ideal, 'r-', linewidth=2, label='Ideal ramp', zorder=3)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Z (µm)')
    ax2.set_title(f'X vs Z (R² = {lin["r_squared"]:.6f})')
    ax2.legend()

    # 3. Z error vs X
    ax3 = axes[0, 2]
    ax3.scatter(x_matched / 1000, z_error, s=3, alpha=0.5, c='blue')
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.axhline(4, color='r', linestyle='--', alpha=0.7, label='±4µm (20x DOF)')
    ax3.axhline(-4, color='r', linestyle='--', alpha=0.7)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z error (µm)')
    ax3.set_title(f'Z Tracking Error (max: {tracking["z_error_max_um"]:.2f} µm)')
    ax3.legend()

    # 4. Z error histogram
    ax4 = axes[1, 0]
    ax4.hist(z_error, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='k', linestyle='-', linewidth=1)
    ax4.axvline(z_error.mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {z_error.mean():.2f} µm')
    ax4.axvline(4, color='orange', linestyle='--', alpha=0.7)
    ax4.axvline(-4, color='orange', linestyle='--', alpha=0.7, label='±4µm DOF')
    ax4.set_xlabel('Z error (µm)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Z Error Distribution (std: {tracking["z_error_std_um"]:.2f} µm)')
    ax4.legend()

    # 5. Z error vs time
    ax5 = axes[1, 1]
    ax5.scatter(t_matched * 1000, z_error, s=3, alpha=0.5, c='blue')
    ax5.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax5.axhline(4, color='r', linestyle='--', alpha=0.7)
    ax5.axhline(-4, color='r', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Z error (µm)')
    ax5.set_title('Z Error vs Time')

    # 6. Polling interval histograms (overlaid)
    ax6 = axes[1, 2]
    ax6.hist(arr["x_intervals"], bins=50, alpha=0.5, label=f'X ({poll["x_interval_mean_ms"]:.1f}ms)', color='blue')
    ax6.hist(arr["z_intervals"], bins=50, alpha=0.5, label=f'Z ({poll["z_interval_mean_ms"]:.1f}ms)', color='red')
    ax6.set_xlabel('Sample interval (ms)')
    ax6.set_ylabel('Count')
    ax6.set_title('Polling Intervals')
    ax6.legend()

    # Summary text
    summary = (
        f"Tracking: max={tracking['z_error_max_um']:.2f}µm, "
        f"std={tracking['z_error_std_um']:.2f}µm\n"
        f"Linearity: R²={lin['r_squared']:.6f}, "
        f"slope error={lin['slope_error_percent']:.2f}%\n"
        f"Polling: X={poll['x_poll_rate_hz']:.0f}Hz, Z={poll['z_poll_rate_hz']:.0f}Hz\n"
        f"Result: {'✓ PASS' if tracking['within_20x_dof'] else '✗ FAIL'} "
        f"(max error {'<' if tracking['within_20x_dof'] else '>'} 4µm DOF)"
    )
    fig.text(0.02, 0.02, summary, fontsize=10, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Z ramp measurement data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files", type=Path, nargs="+",
        help="JSON file(s) from test_z_ramp_measure.py"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plots interactively instead of saving"
    )
    args = parser.parse_args()

    all_passed = True

    for filepath in args.files:
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        if "error" in data:
            print(f"Skipping {filepath}: {data['error']}")
            continue

        analysis = analyze_z_ramp(data)

        if "error" in analysis:
            print(f"Analysis error for {filepath}: {analysis['error']}")
            continue

        print_report(analysis, filepath.name)

        if not analysis["tracking"]["within_20x_dof"]:
            all_passed = False

        if not args.no_plot:
            if args.show:
                plot_results(analysis, output_path=None, title_suffix=filepath.stem)
            else:
                plot_path = filepath.with_suffix('.png')
                plot_results(analysis, output_path=plot_path, title_suffix=filepath.stem)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
