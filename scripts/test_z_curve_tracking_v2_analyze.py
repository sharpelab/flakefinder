"""Analyze Z curve tracking V2 test data (state feedback + feedforward).

Generates plots comparing actual vs desired Z profile, plus detailed
analysis of the control components (position error, velocity error, feedforward).

Run locally after collecting data with test_z_curve_tracking_v2.py.

Usage:
    python test_z_curve_tracking_v2_analyze.py z_curve_v2_*.json
    python test_z_curve_tracking_v2_analyze.py z_curve_v2_*.json --show
    python test_z_curve_tracking_v2_analyze.py z_curve_v2_*.json --compare
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Surface data (duplicated from test script for standalone analysis)
SURFACE_DATA = {
    "top": {
        "y_um": 8175,
        "points": [
            (12112, -4.6),
            (16612, +6.0),
            (21112, +2.0),
            (25612, -4.6),
            (30112, +4.3),
            (34612, -2.4),
        ],
    },
    "middle": {
        "y_um": 12675,
        "points": [
            (12112, -3.5),
            (16612, -0.7),
            (21112, +4.7),
            (25612, -2.0),
            (30112, +1.7),
            (34612, -4.9),
        ],
    },
}

PLANE_FIT = {
    "a_um_per_mm": 1.4722,
    "b_um_per_mm": -0.6527,
    "c_um": 24666.85,
}


def compute_ideal_z(surface_name: str, x_um: np.ndarray, z_offset: float) -> np.ndarray:
    """Compute ideal Z profile for given X positions."""
    surface = SURFACE_DATA[surface_name]
    y_um = surface["y_um"]
    points = surface["points"]

    x_points = np.array([p[0] for p in points])
    residuals = np.array([p[1] for p in points])

    spline = CubicSpline(x_points, residuals, extrapolate=True)

    a = PLANE_FIT["a_um_per_mm"]
    b = PLANE_FIT["b_um_per_mm"]
    c = PLANE_FIT["c_um"]
    y_mm = y_um / 1000

    z_ideal = []
    for x in x_um:
        x_mm = x / 1000
        z_plane = a * x_mm + b * y_mm + c
        z_residual = float(spline(x))
        z_ideal.append(z_plane + z_residual + z_offset)

    return np.array(z_ideal)


def analyze_curve_tracking_v2(data: dict) -> dict:
    """Analyze V2 curve tracking data with control breakdown."""
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

    # Control log arrays (V2 has extended logging)
    if control_log:
        ctrl_t = np.array([c["t"] - t_start for c in control_log])
        ctrl_x = np.array([c["x_um"] for c in control_log])
        ctrl_z_actual = np.array([c["z_actual_um"] for c in control_log])
        ctrl_z_target = np.array([c["z_target_um"] for c in control_log])
        ctrl_z_vel_actual = np.array([c["z_vel_actual_um_s"] for c in control_log])
        ctrl_z_vel_ff = np.array([c["z_vel_ff_um_s"] for c in control_log])
        ctrl_error_pos = np.array([c["error_pos_um"] for c in control_log])
        ctrl_error_vel = np.array([c["error_vel_um_s"] for c in control_log])
        ctrl_vel_from_pos = np.array([c["vel_from_pos"] for c in control_log])
        ctrl_vel_from_vel = np.array([c["vel_from_vel"] for c in control_log])
        ctrl_vel_from_ff = np.array([c["vel_from_ff"] for c in control_log])
        ctrl_cmd_vel = np.array([c["commanded_vel_um_s"] for c in control_log])
    else:
        ctrl_t = ctrl_x = np.array([])
        ctrl_z_actual = ctrl_z_target = np.array([])
        ctrl_z_vel_actual = ctrl_z_vel_ff = np.array([])
        ctrl_error_pos = ctrl_error_vel = np.array([])
        ctrl_vel_from_pos = ctrl_vel_from_vel = ctrl_vel_from_ff = np.array([])
        ctrl_cmd_vel = np.array([])

    # Constant velocity region mask
    x_start = commanded["x_start_um"]
    x_end = commanded["x_end_um"]
    accel_margin = 2000
    cv_mask = (x_pos >= x_start + accel_margin) & (x_pos <= x_end - accel_margin)

    # Control log CV mask
    if len(ctrl_x) > 0:
        ctrl_cv_mask = (ctrl_x >= x_start + accel_margin) & (ctrl_x <= x_end - accel_margin)
    else:
        ctrl_cv_mask = np.array([], dtype=bool)

    return {
        "x_t": x_t,
        "x_pos": x_pos,
        "z_t": z_t,
        "z_pos": z_pos,
        "z_at_x_times": z_at_x_times,
        "z_ideal": z_ideal,
        "z_error": z_error,
        "cv_mask": cv_mask,
        # Control log data
        "ctrl_t": ctrl_t,
        "ctrl_x": ctrl_x,
        "ctrl_z_actual": ctrl_z_actual,
        "ctrl_z_target": ctrl_z_target,
        "ctrl_z_vel_actual": ctrl_z_vel_actual,
        "ctrl_z_vel_ff": ctrl_z_vel_ff,
        "ctrl_error_pos": ctrl_error_pos,
        "ctrl_error_vel": ctrl_error_vel,
        "ctrl_vel_from_pos": ctrl_vel_from_pos,
        "ctrl_vel_from_vel": ctrl_vel_from_vel,
        "ctrl_vel_from_ff": ctrl_vel_from_ff,
        "ctrl_cmd_vel": ctrl_cmd_vel,
        "ctrl_cv_mask": ctrl_cv_mask,
        # Metadata
        "params": params,
        "commanded": commanded,
        "tracking": tracking,
        "timing": timing,
    }


def plot_curve_tracking_v2(data: dict, output_path: Path | None = None, title_suffix: str = "") -> None:
    """Plot V2 curve tracking analysis with control breakdown."""
    analysis = analyze_curve_tracking_v2(data)

    params = analysis["params"]
    tracking = analysis["tracking"]
    x_pos = analysis["x_pos"]
    z_at_x = analysis["z_at_x_times"]
    z_ideal = analysis["z_ideal"]
    z_error = analysis["z_error"]
    cv_mask = analysis["cv_mask"]

    # Control data
    ctrl_t = analysis["ctrl_t"]
    ctrl_error_pos = analysis["ctrl_error_pos"]
    ctrl_error_vel = analysis["ctrl_error_vel"]
    ctrl_vel_from_pos = analysis["ctrl_vel_from_pos"]
    ctrl_vel_from_vel = analysis["ctrl_vel_from_vel"]
    ctrl_vel_from_ff = analysis["ctrl_vel_from_ff"]
    ctrl_cmd_vel = analysis["ctrl_cmd_vel"]
    ctrl_cv_mask = analysis["ctrl_cv_mask"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    title = f"Z Curve Tracking V2: {params['surface_name']} @ {params['x_speed_mm_s']:.0f}mm/s"
    title += f" (Kp={params['kp']}, Kv={params['kv']}, Kff={params['kff']})"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12)

    # Row 1: Basic tracking plots (same as V1)

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
    surface = SURFACE_DATA[params["surface_name"]]
    for x_um, _residual in surface["points"]:
        ax2.axvline(x_um / 1000, color="gray", linestyle=":", alpha=0.3)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Z (µm)")
    ax2.set_title("Actual vs Ideal Z Profile")
    ax2.legend()

    # 3. Tracking error vs X
    ax3 = axes[0, 2]
    ax3.scatter(x_pos[cv_mask] / 1000, z_error[cv_mask], s=2, alpha=0.5, c="blue", label="CV region")
    ax3.scatter(x_pos[~cv_mask] / 1000, z_error[~cv_mask], s=2, alpha=0.3, c="gray", label="Accel/decel")
    ax3.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax3.axhline(4, color="r", linestyle="--", alpha=0.7, label="±4µm DOF")
    ax3.axhline(-4, color="r", linestyle="--", alpha=0.7)
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Z error (µm)")
    ax3.set_title(f"Tracking Error (max CV: {tracking['error_max_um']:.2f}µm)")
    ax3.legend()

    # Row 2: Control component analysis

    # 4. Position error vs time
    ax4 = axes[1, 0]
    if len(ctrl_error_pos) > 0:
        ax4.plot(ctrl_t * 1000, ctrl_error_pos, "b-", linewidth=0.8, alpha=0.8)
        ax4.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax4.axhline(4, color="r", linestyle="--", alpha=0.5)
        ax4.axhline(-4, color="r", linestyle="--", alpha=0.5)
        if len(ctrl_error_pos[ctrl_cv_mask]) > 0:
            mean_err = np.mean(ctrl_error_pos[ctrl_cv_mask])
            ax4.axhline(mean_err, color="orange", linestyle="--", label=f"CV mean: {mean_err:.2f}µm")
            ax4.legend()
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Position error (µm)")
    ax4.set_title("Position Error (z_target - z_actual)")

    # 5. Velocity error vs time
    ax5 = axes[1, 1]
    if len(ctrl_error_vel) > 0:
        ax5.plot(ctrl_t * 1000, ctrl_error_vel, "g-", linewidth=0.8, alpha=0.8)
        ax5.axhline(0, color="k", linestyle="-", linewidth=0.5)
        if len(ctrl_error_vel[ctrl_cv_mask]) > 0:
            mean_vel_err = np.mean(ctrl_error_vel[ctrl_cv_mask])
            ax5.axhline(mean_vel_err, color="orange", linestyle="--", label=f"CV mean: {mean_vel_err:.1f}µm/s")
            ax5.legend()
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Velocity error (µm/s)")
    ax5.set_title("Velocity Error (v_target - v_actual)")

    # 6. Control components stacked
    ax6 = axes[1, 2]
    if len(ctrl_t) > 0:
        ax6.fill_between(ctrl_t * 1000, 0, ctrl_vel_from_ff, alpha=0.4, label="Feedforward", color="blue")
        ax6.fill_between(
            ctrl_t * 1000,
            ctrl_vel_from_ff,
            ctrl_vel_from_ff + ctrl_vel_from_pos,
            alpha=0.4,
            label="From pos error",
            color="green",
        )
        ax6.fill_between(
            ctrl_t * 1000,
            ctrl_vel_from_ff + ctrl_vel_from_pos,
            ctrl_vel_from_ff + ctrl_vel_from_pos + ctrl_vel_from_vel,
            alpha=0.4,
            label="From vel error",
            color="red",
        )
        ax6.plot(ctrl_t * 1000, ctrl_cmd_vel, "k-", linewidth=1, alpha=0.8, label="Total cmd")
        ax6.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax6.legend(fontsize=8)
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("Velocity component (µm/s)")
    ax6.set_title("Control Law Decomposition")

    # Row 3: Statistics and summary

    # 7. Error histogram
    ax7 = axes[2, 0]
    z_error_cv = z_error[cv_mask]
    if len(z_error_cv) > 0:
        ax7.hist(z_error_cv, bins=50, edgecolor="black", alpha=0.7)
        ax7.axvline(0, color="k", linestyle="-", linewidth=1)
        ax7.axvline(z_error_cv.mean(), color="r", linestyle="--", linewidth=2, label=f"Mean: {z_error_cv.mean():.2f}µm")
        ax7.axvline(4, color="orange", linestyle="--", alpha=0.7)
        ax7.axvline(-4, color="orange", linestyle="--", alpha=0.7, label="±4µm DOF")
        ax7.legend()
    ax7.set_xlabel("Z error (µm)")
    ax7.set_ylabel("Count")
    ax7.set_title(f"Error Distribution (std: {tracking['error_std_um']:.2f}µm)")

    # 8. Control contribution histogram (CV region only)
    ax8 = axes[2, 1]
    if len(ctrl_t) > 0 and ctrl_cv_mask.sum() > 0:
        pos_contrib = ctrl_vel_from_pos[ctrl_cv_mask]
        vel_contrib = ctrl_vel_from_vel[ctrl_cv_mask]
        ff_contrib = ctrl_vel_from_ff[ctrl_cv_mask]

        x_labels = ["Pos err", "Vel err", "Feedfwd"]
        means = [np.mean(np.abs(pos_contrib)), np.mean(np.abs(vel_contrib)), np.mean(np.abs(ff_contrib))]
        colors = ["green", "red", "blue"]

        bars = ax8.bar(x_labels, means, color=colors, alpha=0.7, edgecolor="black")
        ax8.set_ylabel("Mean |contribution| (µm/s)")
        ax8.set_title("Control Component Magnitudes (CV region)")

        # Add value labels on bars
        for bar, val in zip(bars, means, strict=False):
            ax8.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 9. Summary text
    ax9 = axes[2, 2]
    ax9.axis("off")

    profile = data.get("profile", {})

    # Calculate additional stats from control log
    if len(ctrl_error_pos) > 0 and ctrl_cv_mask.sum() > 0:
        pos_err_rms = np.sqrt(np.mean(ctrl_error_pos[ctrl_cv_mask] ** 2))
        vel_err_rms = np.sqrt(np.mean(ctrl_error_vel[ctrl_cv_mask] ** 2))
        pos_contrib_mean = np.mean(np.abs(ctrl_vel_from_pos[ctrl_cv_mask]))
        vel_contrib_mean = np.mean(np.abs(ctrl_vel_from_vel[ctrl_cv_mask]))
        ff_contrib_mean = np.mean(np.abs(ctrl_vel_from_ff[ctrl_cv_mask]))
    else:
        pos_err_rms = vel_err_rms = 0
        pos_contrib_mean = vel_contrib_mean = ff_contrib_mean = 0

    # Control loop timing stats
    ctrl_timing = data.get("control_loop_timing", {})
    timing_text = ""
    if ctrl_timing.get("actual_hz"):
        timing_text = f"""
Control loop timing:
  Target: {ctrl_timing.get("target_hz", "N/A"):.0f} Hz
  Actual: {ctrl_timing.get("actual_hz", 0):.1f} Hz
  Interval: {ctrl_timing.get("interval_mean_ms", 0):.2f} ± {ctrl_timing.get("interval_std_ms", 0):.2f} ms
  Range: {ctrl_timing.get("interval_min_ms", 0):.2f} - {ctrl_timing.get("interval_max_ms", 0):.2f} ms
"""

    summary_text = f"""Curve Tracking V2 Summary
{"=" * 32}

Surface: {params["surface_name"]} (Y={profile.get("y_um", "N/A")} µm)
X speed: {params["x_speed_mm_s"]:.1f} mm/s
Control rate: {params["control_rate_hz"]:.0f} Hz

Control gains:
  Kp: {params["kp"]:.2f} /s
  Kv: {params["kv"]:.2f}
  Kff: {params["kff"]:.2f}
{timing_text}
Tracking (CV region):
  Mean error: {tracking["error_mean_um"]:+.2f} µm
  Std error: {tracking["error_std_um"]:.2f} µm
  Max error: {tracking["error_max_um"]:.2f} µm
  95th pct: {tracking["error_p95_um"]:.2f} µm

Control stats (CV):
  Pos err RMS: {pos_err_rms:.2f} µm
  Vel err RMS: {vel_err_rms:.1f} µm/s
  |Pos contrib|: {pos_contrib_mean:.1f} µm/s
  |Vel contrib|: {vel_contrib_mean:.1f} µm/s
  |FF contrib|: {ff_contrib_mean:.1f} µm/s

Result: {"✓ PASS" if tracking["within_20x_dof"] else "✗ FAIL"}
  (max {tracking["error_max_um"]:.2f}µm {"<" if tracking["within_20x_dof"] else ">"} {tracking["dof_20x_um"]}µm DOF)
"""
    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
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


def plot_comparison(data_list: list[tuple[str, dict]], output_path: Path | None = None) -> None:
    """Plot comparison of multiple test runs.

    Args:
        data_list: List of (filename, data_dict) tuples.
        output_path: Path to save plot, or None to show interactively.
    """
    n = len(data_list)
    if n < 2:
        print("Need at least 2 files for comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("V2 Tracking Comparison", fontsize=12)

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    # 1. Tracking error vs X (all runs)
    ax1 = axes[0, 0]
    for i, (_fname, data) in enumerate(data_list):
        analysis = analyze_curve_tracking_v2(data)
        x_pos = analysis["x_pos"]
        z_error = analysis["z_error"]
        cv_mask = analysis["cv_mask"]
        label = Path(fname).stem.split("_")[-1]  # Use timestamp as label

        params = analysis["params"]
        label = f"Kp={params['kp']}, Kv={params['kv']}"

        ax1.plot(x_pos[cv_mask] / 1000, z_error[cv_mask], linewidth=0.8, alpha=0.7, color=colors[i], label=label)

    ax1.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax1.axhline(4, color="r", linestyle="--", alpha=0.5)
    ax1.axhline(-4, color="r", linestyle="--", alpha=0.5)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Z error (µm)")
    ax1.set_title("Tracking Error vs Position")
    ax1.legend(fontsize=8)

    # 2. Error metrics bar chart
    ax2 = axes[0, 1]
    labels = []
    mean_errors = []
    max_errors = []
    std_errors = []

    for _fname, data in data_list:
        params = data["params"]
        tracking = data["tracking"]
        labels.append(f"Kp={params['kp']}\nKv={params['kv']}")
        mean_errors.append(abs(tracking["error_mean_um"]))
        max_errors.append(tracking["error_max_um"])
        std_errors.append(tracking["error_std_um"])

    x = np.arange(len(labels))
    width = 0.25

    ax2.bar(x - width, mean_errors, width, label="|Mean|", color="blue", alpha=0.7)
    ax2.bar(x, max_errors, width, label="Max", color="red", alpha=0.7)
    ax2.bar(x + width, std_errors, width, label="Std", color="green", alpha=0.7)

    ax2.axhline(4, color="orange", linestyle="--", label="DOF limit")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Error (µm)")
    ax2.set_title("Error Metrics Comparison")
    ax2.legend()

    # 3. Position error vs time (all runs)
    ax3 = axes[1, 0]
    for i, (_fname, data) in enumerate(data_list):
        analysis = analyze_curve_tracking_v2(data)
        ctrl_t = analysis["ctrl_t"]
        ctrl_error_pos = analysis["ctrl_error_pos"]
        params = analysis["params"]

        if len(ctrl_t) > 0:
            label = f"Kp={params['kp']}, Kv={params['kv']}"
            ax3.plot(ctrl_t * 1000, ctrl_error_pos, linewidth=0.8, alpha=0.7, color=colors[i], label=label)

    ax3.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax3.axhline(4, color="r", linestyle="--", alpha=0.5)
    ax3.axhline(-4, color="r", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Position error (µm)")
    ax3.set_title("Position Error vs Time")
    ax3.legend(fontsize=8)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = []
    headers = ["Kp", "Kv", "Kff", "Mean", "Max", "Std", "Pass"]

    for _fname, data in data_list:
        params = data["params"]
        tracking = data["tracking"]
        table_data.append(
            [
                f"{params['kp']:.1f}",
                f"{params['kv']:.1f}",
                f"{params['kff']:.1f}",
                f"{tracking['error_mean_um']:+.2f}",
                f"{tracking['error_max_um']:.2f}",
                f"{tracking['error_std_um']:.2f}",
                "✓" if tracking["within_20x_dof"] else "✗",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["lightblue"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title("Results Summary", pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Comparison plot saved to {output_path}")
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
    print(f"Z CURVE TRACKING V2 ANALYSIS{f': {filename}' if filename else ''}")
    print("=" * 60)

    print("\nTest parameters:")
    print(f"  Surface: {params['surface_name']}")
    print(f"  X speed: {params['x_speed_mm_s']:.1f} mm/s")
    print(f"  Control rate: {params['control_rate_hz']:.0f} Hz")

    print("\nControl gains:")
    print(f"  Kp (position): {params['kp']:.2f} /s")
    print(f"  Kv (velocity): {params['kv']:.2f}")
    print(f"  Kff (feedforward): {params['kff']:.2f}")

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

    # Control statistics from log
    control_log = data.get("control_log", [])
    if control_log:
        commanded = data["commanded"]
        x_start = commanded["x_start_um"]
        x_end = commanded["x_end_um"]
        accel_margin = 2000

        ctrl_x = np.array([c["x_um"] for c in control_log])
        cv_mask = (ctrl_x >= x_start + accel_margin) & (ctrl_x <= x_end - accel_margin)

        if cv_mask.sum() > 0:
            cv_logs = [c for c, m in zip(control_log, cv_mask, strict=False) if m]

            pos_err = np.array([c["error_pos_um"] for c in cv_logs])
            vel_err = np.array([c["error_vel_um_s"] for c in cv_logs])
            pos_contrib = np.array([c["vel_from_pos"] for c in cv_logs])
            vel_contrib = np.array([c["vel_from_vel"] for c in cv_logs])
            ff_contrib = np.array([c["vel_from_ff"] for c in cv_logs])

            print("\nControl statistics (CV region):")
            print(f"  Position error RMS: {np.sqrt(np.mean(pos_err**2)):.2f} µm")
            print(f"  Velocity error RMS: {np.sqrt(np.mean(vel_err**2)):.1f} µm/s")
            print(f"  Mean |pos contrib|: {np.mean(np.abs(pos_contrib)):.1f} µm/s")
            print(f"  Mean |vel contrib|: {np.mean(np.abs(vel_contrib)):.1f} µm/s")
            print(f"  Mean |ff contrib|:  {np.mean(np.abs(ff_contrib)):.1f} µm/s")

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
        description="Analyze Z curve tracking V2 test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", type=Path, nargs="+", help="JSON file(s) from test_z_curve_tracking_v2.py")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively instead of saving")
    parser.add_argument("--compare", action="store_true", help="Generate comparison plot across all files")
    args = parser.parse_args()

    all_passed = True
    data_list = []

    for filepath in args.files:
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        if "error" in data:
            print(f"Skipping {filepath}: {data['error']}")
            continue

        data_list.append((str(filepath), data))
        print_report(data, filepath.name)

        if not data["tracking"]["within_20x_dof"]:
            all_passed = False

        if not args.no_plot and not args.compare:
            plot_path = None if args.show else filepath.with_suffix(".png")
            plot_curve_tracking_v2(data, plot_path, filepath.stem)

    # Comparison plot
    if args.compare and len(data_list) >= 2:
        if args.show:
            plot_comparison(data_list, None)
        else:
            plot_comparison(data_list, Path("z_curve_v2_comparison.png"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
