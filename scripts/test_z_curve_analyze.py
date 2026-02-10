"""Analyze Z curve tracking test data.

Generates plots comparing actual vs desired Z profile.
Run locally after collecting data with test_z_curve_tracking.py.

Usage:
    python test_z_curve_analyze.py z_curve_*.json
    python test_z_curve_analyze.py z_curve_*.json --show
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from z_curve_data import SURFACE_DATA, compute_ideal_z


def analyze_curve_tracking(data: dict) -> dict:
    """Analyze curve tracking data.

    Args:
        data: Dict loaded from test_z_curve_tracking.py output.

    Returns:
        Analysis dict with computed metrics and arrays.
    """
    params = data["params"]
    commanded = data["commanded"]
    timing = data["timing"]
    tracking = data["tracking"]

    # Extract sample arrays
    x_samples = data["x_samples"]
    z_samples = data["z_samples"]
    control_log = data.get("control_log", [])

    t_start = timing["t_motion_start"]

    x_t = np.array([s["t"] - t_start for s in x_samples])
    x_pos = np.array([s["x_um"] for s in x_samples])
    z_t = np.array([s["t"] - t_start for s in z_samples])
    z_pos = np.array([s["z_um"] for s in z_samples])

    # Interpolate Z at X sample times
    z_at_x_times = np.interp(x_t, z_t, z_pos)

    # Compute ideal Z
    z_offset = commanded["z_offset_um"]
    z_ideal = compute_ideal_z(params["surface_name"], x_pos, z_offset)

    # Tracking error
    z_error = z_at_x_times - z_ideal

    # Control log
    if control_log:
        ctrl_t = np.array([c["t"] - t_start for c in control_log])
        ctrl_x = np.array([c["x_um"] for c in control_log])
        ctrl_z_target = np.array([c["z_target_um"] for c in control_log])
        ctrl_z_vel = np.array([c["z_vel_um_s"] for c in control_log])
    else:
        ctrl_t = ctrl_x = ctrl_z_target = ctrl_z_vel = np.array([])

    # Constant velocity region mask
    x_start = commanded["x_start_um"]
    x_end = commanded["x_end_um"]
    accel_margin = 2000
    cv_mask = (x_pos >= x_start + accel_margin) & (x_pos <= x_end - accel_margin)

    return {
        "x_t": x_t,
        "x_pos": x_pos,
        "z_t": z_t,
        "z_pos": z_pos,
        "z_at_x_times": z_at_x_times,
        "z_ideal": z_ideal,
        "z_error": z_error,
        "cv_mask": cv_mask,
        "ctrl_t": ctrl_t,
        "ctrl_x": ctrl_x,
        "ctrl_z_target": ctrl_z_target,
        "ctrl_z_vel": ctrl_z_vel,
        "params": params,
        "commanded": commanded,
        "tracking": tracking,
        "timing": timing,
    }


def plot_curve_tracking(data: dict, output_path: Path | None = None, title_suffix: str = "") -> None:
    """Plot curve tracking analysis.

    Args:
        data: Dict loaded from test_z_curve_tracking.py output.
        output_path: Path to save plot, or None to show interactively.
        title_suffix: Additional text for title.
    """
    analysis = analyze_curve_tracking(data)

    params = analysis["params"]
    tracking = analysis["tracking"]
    x_pos = analysis["x_pos"]
    z_at_x = analysis["z_at_x_times"]
    z_ideal = analysis["z_ideal"]
    z_error = analysis["z_error"]
    cv_mask = analysis["cv_mask"]
    ctrl_z_vel = analysis["ctrl_z_vel"]
    ctrl_t = analysis["ctrl_t"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    title = f"Z Curve Tracking: {params['surface_name']} surface @ {params['x_speed_mm_s']:.0f}mm/s"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=12)

    # 1. X and Z vs time
    ax1 = axes[0, 0]
    ax1_z = ax1.twinx()
    x_t = analysis["x_t"]
    z_t = analysis["z_t"]
    ax1.plot(x_t * 1000, x_pos / 1000, "b-", linewidth=0.5, alpha=0.8, label="X")
    ax1_z.plot(z_t * 1000, analysis["z_pos"], "r-", linewidth=0.5, alpha=0.8, label="Z")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("X (mm)", color="b")
    ax1_z.set_ylabel("Z (µm)", color="r")
    ax1.set_title("Position vs Time")

    # 2. Actual vs ideal Z vs X
    ax2 = axes[0, 1]
    ax2.plot(x_pos / 1000, z_at_x, "b-", linewidth=0.8, alpha=0.7, label="Actual Z")
    ax2.plot(x_pos / 1000, z_ideal, "r--", linewidth=1.5, label="Ideal Z")

    # Mark surface data points
    surface = SURFACE_DATA[params["surface_name"]]
    for x_um, _residual in surface["points"]:
        x_mm = x_um / 1000
        ax2.axvline(x_mm, color="gray", linestyle=":", alpha=0.3)

    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Z (µm)")
    ax2.set_title("Actual vs Ideal Z Profile")
    ax2.legend()

    # 3. Tracking error vs X
    ax3 = axes[0, 2]
    ax3.scatter(x_pos[cv_mask] / 1000, z_error[cv_mask], s=2, alpha=0.5, c="blue", label="CV region")
    ax3.scatter(x_pos[~cv_mask] / 1000, z_error[~cv_mask], s=2, alpha=0.3, c="gray", label="Accel/decel")
    ax3.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax3.axhline(4, color="r", linestyle="--", alpha=0.7, label="±4µm (20x DOF)")
    ax3.axhline(-4, color="r", linestyle="--", alpha=0.7)
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z error (µm)")
    ax3.set_title(f"Tracking Error (max CV: {tracking['error_max_um']:.2f}µm)")
    ax3.legend()

    # 4. Z velocity commands vs time
    ax4 = axes[1, 0]
    if len(ctrl_z_vel) > 0:
        ax4.plot(ctrl_t * 1000, ctrl_z_vel, "g-", linewidth=0.8)
        ax4.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("Commanded Z velocity (µm/s)")
        ax4.set_title(f"Z Velocity Commands ({len(ctrl_z_vel)} updates)")
    else:
        ax4.text(0.5, 0.5, "No control log data", transform=ax4.transAxes, ha="center", va="center")
        ax4.set_title("Z Velocity Commands")

    # 5. Error histogram
    ax5 = axes[1, 1]
    z_error_cv = z_error[cv_mask]
    ax5.hist(z_error_cv, bins=50, edgecolor="black", alpha=0.7)
    ax5.axvline(0, color="k", linestyle="-", linewidth=1)
    ax5.axvline(z_error_cv.mean(), color="r", linestyle="--", linewidth=2, label=f"Mean: {z_error_cv.mean():.2f}µm")
    ax5.axvline(4, color="orange", linestyle="--", alpha=0.7)
    ax5.axvline(-4, color="orange", linestyle="--", alpha=0.7, label="±4µm DOF")
    ax5.set_xlabel("Z error (µm)")
    ax5.set_ylabel("Count")
    ax5.set_title(f"Error Distribution (std: {tracking['error_std_um']:.2f}µm)")
    ax5.legend()

    # 6. Summary text
    ax6 = axes[1, 2]
    ax6.axis("off")

    profile = data.get("profile", {})
    summary_text = f"""Curve Tracking Summary
{"=" * 30}

Surface: {params["surface_name"]} (Y={profile.get("y_um", "N/A")} µm)
X speed: {params["x_speed_mm_s"]:.1f} mm/s
Control rate: {params["control_rate_hz"]:.0f} Hz

Profile stats:
  Z range: {profile.get("z_range_um", ["N/A", "N/A"])[0]:.1f} - {profile.get("z_range_um", ["N/A", "N/A"])[1]:.1f} µm
  Max gradient: {profile.get("max_gradient_um_per_mm", "N/A"):.2f} µm/mm

Tracking (CV region):
  Mean error: {tracking["error_mean_um"]:+.2f} µm
  Std error: {tracking["error_std_um"]:.2f} µm
  Max error: {tracking["error_max_um"]:.2f} µm
  95th pct: {tracking["error_p95_um"]:.2f} µm

Result: {"✓ PASS" if tracking["within_20x_dof"] else "✗ FAIL"}
  (max error {"<" if tracking["within_20x_dof"] else ">"} {tracking["dof_20x_um"]} µm DOF)
"""
    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def print_report(data: dict, filename: str = "") -> None:
    """Print analysis report."""
    params = data["params"]
    tracking = data["tracking"]
    profile = data.get("profile", {})
    timing = data["timing"]

    print()
    print("=" * 60)
    print(f"Z CURVE TRACKING ANALYSIS{f': {filename}' if filename else ''}")
    print("=" * 60)

    print("\nTest parameters:")
    print(f"  Surface: {params['surface_name']}")
    print(f"  X speed: {params['x_speed_mm_s']:.1f} mm/s")
    print(f"  Control rate: {params['control_rate_hz']:.0f} Hz")

    print("\nProfile:")
    print(f"  Max gradient: {profile.get('max_gradient_um_per_mm', 'N/A'):.2f} µm/mm")
    print(f"  Max Z velocity: {profile.get('max_gradient_um_per_mm', 0) * params['x_speed_mm_s']:.1f} µm/s")

    print("\nTiming:")
    print(f"  Motion duration: {timing['motion_duration_s'] * 1000:.0f} ms")

    print("\nTracking error (constant velocity region):")
    print(f"  Mean: {tracking['error_mean_um']:+.2f} µm")
    print(f"  Std: {tracking['error_std_um']:.2f} µm")
    print(f"  Max: {tracking['error_max_um']:.2f} µm")
    print(f"  95th percentile: {tracking['error_p95_um']:.2f} µm")

    print()
    print("=" * 60)
    dof = tracking["dof_20x_um"]
    if tracking["within_20x_dof"]:
        print(f"✓ PASS: Max error ({tracking['error_max_um']:.2f}µm) within 20x DOF ({dof}µm)")
    else:
        print(f"✗ FAIL: Max error ({tracking['error_max_um']:.2f}µm) exceeds 20x DOF ({dof}µm)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Z curve tracking test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", type=Path, nargs="+", help="JSON file(s) from test_z_curve_tracking.py")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively instead of saving")
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

        print_report(data, filepath.name)

        if not data["tracking"]["within_20x_dof"]:
            all_passed = False

        if not args.no_plot:
            plot_path = None if args.show else filepath.with_suffix(".png")
            plot_curve_tracking(data, plot_path, filepath.stem)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
