"""Analyze Z step response test data.

Generates plots of step response characteristics.
Run locally after collecting data with test_z_step_response.py.

Usage:
    python test_z_step_analyze.py z_step_response_*.json
    python test_z_step_analyze.py z_step_response_*.json --show
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def analyze_step_response(data: dict) -> dict:
    """Analyze step response data.

    Args:
        data: Dict loaded from test_z_step_response.py output.

    Returns:
        Analysis dict with computed metrics.
    """
    samples = data["samples"]
    timing = data["timing"]
    params = data["params"]

    # Extract arrays
    t_samples = np.array([(s["t_before"] + s["t_after"]) / 2 for s in samples])
    z_samples = np.array([s["z_um"] for s in samples])

    # Normalize to command time
    t_rel = t_samples - timing["t_command_end"]

    # Get actual step characteristics
    actual = data["actual"]
    pre_z = actual["pre_step_z_um"]
    post_z = actual["post_step_z_um"]
    step_size = actual["actual_step_um"]

    # Normalize position
    if abs(step_size) > 0.1:
        z_norm = (z_samples - pre_z) / step_size
    else:
        z_norm = z_samples - pre_z

    return {
        "t_rel_ms": t_rel * 1000,
        "z_um": z_samples,
        "z_normalized": z_norm,
        "pre_z_um": pre_z,
        "post_z_um": post_z,
        "step_size_um": step_size,
        "metrics": data.get("metrics", {}),
        "params": params,
    }


def plot_step_response(data: dict, output_path: Path | None = None, title_suffix: str = "") -> None:
    """Plot step response analysis.

    Args:
        data: Dict loaded from test_z_step_response.py output.
        output_path: Path to save plot, or None to show interactively.
        title_suffix: Additional text for title.
    """
    analysis = analyze_step_response(data)

    t_ms = analysis["t_rel_ms"]
    z_um = analysis["z_um"]
    z_norm = analysis["z_normalized"]
    metrics = analysis["metrics"]
    params = analysis["params"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    title = f"Z Step Response: {params['step_size_um']:+.0f}µm step"
    if title_suffix:
        title += f" ({title_suffix})"
    fig.suptitle(title, fontsize=12)

    # 1. Raw Z position vs time
    ax1 = axes[0, 0]
    ax1.plot(t_ms, z_um, 'b-', linewidth=0.8)
    ax1.axhline(analysis["pre_z_um"], color='gray', linestyle='--', alpha=0.5, label='Start')
    ax1.axhline(analysis["post_z_um"], color='gray', linestyle='--', alpha=0.5, label='Target')
    ax1.axvline(0, color='r', linestyle='-', alpha=0.3, label='Command')
    ax1.set_xlabel('Time from command (ms)')
    ax1.set_ylabel('Z position (µm)')
    ax1.set_title('Raw Position')
    ax1.legend()

    # 2. Normalized step response
    ax2 = axes[0, 1]
    ax2.plot(t_ms, z_norm, 'b-', linewidth=0.8)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(0.1, color='orange', linestyle=':', alpha=0.7, label='10%')
    ax2.axhline(0.9, color='orange', linestyle=':', alpha=0.7, label='90%')
    ax2.axvline(0, color='r', linestyle='-', alpha=0.3)

    # Mark rise time if available
    if metrics.get("latency_to_10pct_ms"):
        ax2.axvline(metrics["latency_to_10pct_ms"], color='green', linestyle='--', alpha=0.5, label=f'10%: {metrics["latency_to_10pct_ms"]:.1f}ms')
    if metrics.get("rise_time_10_90_ms") and metrics.get("latency_to_10pct_ms"):
        t_90 = metrics["latency_to_10pct_ms"] + metrics["rise_time_10_90_ms"]
        ax2.axvline(t_90, color='green', linestyle='--', alpha=0.5, label=f'90%: {t_90:.1f}ms')

    ax2.set_xlabel('Time from command (ms)')
    ax2.set_ylabel('Normalized position')
    ax2.set_title('Normalized Step Response')
    ax2.legend(loc='lower right')
    ax2.set_ylim(-0.1, 1.3)

    # 3. Zoomed in on step (first 200ms)
    ax3 = axes[1, 0]
    mask = (t_ms >= -10) & (t_ms <= 200)
    if mask.sum() > 0:
        ax3.plot(t_ms[mask], z_norm[mask], 'b-', linewidth=1)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(1, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(0.1, color='orange', linestyle=':', alpha=0.7)
        ax3.axhline(0.9, color='orange', linestyle=':', alpha=0.7)
        ax3.axvline(0, color='r', linestyle='-', alpha=0.3)

        # Settling band
        settle_tol = params.get("settle_tolerance", 0.02)
        ax3.axhline(1 + settle_tol, color='red', linestyle=':', alpha=0.5)
        ax3.axhline(1 - settle_tol, color='red', linestyle=':', alpha=0.5, label=f'±{settle_tol*100:.0f}% band')

    ax3.set_xlabel('Time from command (ms)')
    ax3.set_ylabel('Normalized position')
    ax3.set_title('Step Response Detail (first 200ms)')
    ax3.legend()

    # 4. Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    metrics_text = f"""Step Response Metrics
{'='*30}

Step size: {params['step_size_um']:+.1f} µm
Actual step: {analysis['step_size_um']:+.2f} µm

Latency (to 10%): {metrics.get('latency_to_10pct_ms', 'N/A'):.1f} ms
Rise time (10-90%): {metrics.get('rise_time_10_90_ms', 'N/A'):.1f} ms
Settling time (±{params.get('settle_tolerance', 0.02)*100:.0f}%): {metrics.get('settling_time_ms', 'N/A'):.1f} ms
Overshoot: {metrics.get('overshoot_percent', 0):.1f}%

Position error: {data['actual']['position_error_um']:+.2f} µm

Polling rate: {data['polling']['poll_rate_hz']:.0f} Hz
SDK read time: {data['polling']['sdk_read_time_ms']:.2f} ms
"""
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_trial_summary(data: dict, output_path: Path | None = None) -> None:
    """Plot summary of multi-trial step response test.

    Args:
        data: Dict loaded from multi-trial test output.
        output_path: Path to save plot, or None to show interactively.
    """
    trials = data["trials"]
    params = data["params"]
    summary = data["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Z Step Response Summary: {params['step_size_um']:+.0f}µm step ({params['num_trials']} trials)", fontsize=12)

    # Extract metrics
    rise_times = [t["rise_time_ms"] for t in trials if t.get("rise_time_ms") is not None]
    settling_times = [t["settling_time_ms"] for t in trials]
    overshoots = [t["overshoot_pct"] for t in trials]
    latencies = [t["latency_ms"] for t in trials]

    # 1. Rise time histogram
    ax1 = axes[0, 0]
    if rise_times:
        ax1.hist(rise_times, bins=max(5, len(rise_times)//2), edgecolor='black', alpha=0.7)
        mean_rise = summary["rise_time_ms"]["mean"]
        if mean_rise:
            ax1.axvline(mean_rise, color='r', linestyle='--', label=f'Mean: {mean_rise:.1f}ms')
        ax1.set_xlabel('Rise time (ms)')
        ax1.set_ylabel('Count')
        ax1.set_title('Rise Time (10-90%)')
        ax1.legend()

    # 2. Settling time histogram
    ax2 = axes[0, 1]
    ax2.hist(settling_times, bins=max(5, len(settling_times)//2), edgecolor='black', alpha=0.7)
    mean_settle = summary["settling_time_ms"]["mean"]
    if mean_settle:
        ax2.axvline(mean_settle, color='r', linestyle='--', label=f'Mean: {mean_settle:.1f}ms')
    ax2.set_xlabel('Settling time (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Settling Time (±{params["settle_tolerance"]*100:.0f}%)')
    ax2.legend()

    # 3. Metrics vs trial number
    ax3 = axes[1, 0]
    trial_nums = [t["trial"] for t in trials]
    ax3.plot(trial_nums, rise_times if rise_times else [0]*len(trial_nums), 'o-', label='Rise time')
    ax3.plot(trial_nums, settling_times, 's-', label='Settling time')
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Consistency Across Trials')
    ax3.legend()

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    def fmt_stat(stat_dict, unit=""):
        if stat_dict["mean"] is None:
            return "N/A"
        mean = stat_dict["mean"]
        std = stat_dict["std"] if stat_dict["std"] else 0
        return f"{mean:.1f} ± {std:.1f} {unit}"

    summary_text = f"""Multi-Trial Summary
{'='*30}

Step size: {params['step_size_um']:+.1f} µm
Trials: {params['num_trials']}

Latency (to 10%): {fmt_stat(summary['latency_ms'], 'ms')}
Rise time (10-90%): {fmt_stat(summary['rise_time_ms'], 'ms')}
Settling time: {fmt_stat(summary['settling_time_ms'], 'ms')}
Overshoot: {fmt_stat(summary['overshoot_pct'], '%')}
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Z step response test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files", type=Path, nargs="+",
        help="JSON file(s) from test_z_step_response.py"
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

    for filepath in args.files:
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"Analyzing: {filepath.name}")
        print(f"{'='*60}")

        # Check if multi-trial or single trial
        if "trials" in data:
            # Multi-trial summary
            summary = data["summary"]
            print(f"Multi-trial test: {data['params']['num_trials']} trials")
            print(f"  Rise time: {summary['rise_time_ms']['mean']:.1f} ± {summary['rise_time_ms']['std']:.1f} ms")
            print(f"  Settling time: {summary['settling_time_ms']['mean']:.1f} ± {summary['settling_time_ms']['std']:.1f} ms")
            print(f"  Overshoot: {summary['overshoot_pct']['mean']:.1f} ± {summary['overshoot_pct']['std']:.1f}%")

            if not args.no_plot:
                plot_path = None if args.show else filepath.with_suffix('.png')
                plot_multi_trial_summary(data, plot_path)

        else:
            # Single trial
            metrics = data.get("metrics", {})
            print(f"Single trial test: {data['params']['step_size_um']:+.1f}µm step")
            print(f"  Latency: {metrics.get('latency_to_10pct_ms', 'N/A'):.1f} ms")
            print(f"  Rise time: {metrics.get('rise_time_10_90_ms', 'N/A'):.1f} ms")
            print(f"  Settling time: {metrics.get('settling_time_ms', 'N/A'):.1f} ms")
            print(f"  Overshoot: {metrics.get('overshoot_percent', 0):.1f}%")

            if not args.no_plot:
                plot_path = None if args.show else filepath.with_suffix('.png')
                plot_step_response(data, plot_path, filepath.stem)

    return 0


if __name__ == "__main__":
    exit(main())
