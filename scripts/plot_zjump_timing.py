"""Plot frame timing and position data to investigate Z-jump hypothesis.

Shows T on X axis with sampled X positions, Z error, and camera frame markers.
Hypothesis: large delay between frame 0 and frame 1 capture correlates with Z-jump magnitude.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_scan(scan_dir):
    with open(Path(scan_dir) / "scan_meta.json") as f:
        return json.load(f)


def plot_row_timing(ax_x, ax_z, meta, row_idx, label_prefix=""):
    """Plot X position samples and Z error for a single row, with frame markers."""
    frames = [f for f in meta["frames"] if f["line"] == row_idx]
    pos_samples = meta["position_stream"]
    row_samples = [s for s in pos_samples if s["line"] == row_idx]

    if not frames or not row_samples:
        return

    # Time origin: first position sample for this row
    t0 = row_samples[0]["t_before"]

    # Position stream: X vs T
    ps_t = [(s["t_before"] + s["t_after"]) / 2 - t0 for s in row_samples]
    ps_x = [s["x_um"] / 1000 for s in row_samples]  # mm

    ax_x.plot(ps_t, ps_x, ".-", color="C0", markersize=2, linewidth=0.8, alpha=0.7, label="X (polled)")

    # Frame captures: vertical spans + Z error markers
    for i, f in enumerate(frames):
        ft_start = f["t_capture"] - t0
        ft_end = f["t_capture"] + f["capture_duration_s"] - t0
        ft_mid = (ft_start + ft_end) / 2

        # Frame span on X plot
        alpha = 0.4 if i == 0 else 0.15
        color = "red" if i == 0 else ("orange" if i == 1 else "gray")
        ax_x.axvspan(ft_start, ft_end, alpha=alpha, color=color, zorder=0)

        # Z error markers
        ax_z.plot(
            ft_mid,
            f["z_error_um"],
            "o",
            color=color,
            markersize=6 if i < 2 else 3,
            zorder=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    # Annotate frame 0→1 gap
    if len(frames) >= 2:
        dt = (frames[1]["t_capture"] - frames[0]["t_capture"]) * 1000
        mid_t = (frames[0]["t_capture"] + frames[1]["t_capture"]) / 2 - t0
        ax_x.annotate(
            f"{dt:.0f}ms",
            xy=(mid_t, ax_x.get_ylim()[0]),
            xytext=(mid_t, ax_x.get_ylim()[0]),
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
            fontweight="bold",
        )

    # Z error line
    frame_t = [(f["t_capture"] + (f["t_capture"] + f["capture_duration_s"])) / 2 - t0 for f in frames]
    frame_z = [f["z_error_um"] for f in frames]
    ax_z.plot(frame_t, frame_z, "-", color="C1", linewidth=1, alpha=0.5)
    ax_z.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    row_info = meta["lines"][row_idx]
    direction = "→+X" if row_info["direction"] == 1 else "←-X"
    ax_x.set_title(f"{label_prefix}Row {row_idx} ({direction})", fontsize=9)


def plot_scan(scan_dir, scan_label=None):
    meta = load_scan(scan_dir)
    n_rows = len(meta["lines"])

    if scan_label is None:
        scan_label = Path(scan_dir).name

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows), squeeze=False, gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"Z-Jump Timing Analysis: {scan_label}", fontsize=12, fontweight="bold")

    for row_idx in range(n_rows):
        ax_x = axes[row_idx, 0]
        ax_z = axes[row_idx, 1]

        plot_row_timing(ax_x, ax_z, meta, row_idx)

        ax_x.set_ylabel("X (mm)")
        ax_z.set_ylabel("Z error (µm)")
        if row_idx == n_rows - 1:
            ax_x.set_xlabel("Time (s)")
            ax_z.set_xlabel("Time (s)")

    plt.tight_layout()
    return fig


def plot_multi_scan_comparison(scan_dirs, labels=None):
    """Plot frame 0→1 gap vs Z-jump magnitude across multiple scans."""
    all_gaps = []
    all_jumps = []
    all_labels = []

    for i, scan_dir in enumerate(scan_dirs):
        meta = load_scan(scan_dir)
        label = labels[i] if labels else Path(scan_dir).name

        for row_idx in range(len(meta["lines"])):
            frames = [f for f in meta["frames"] if f["line"] == row_idx]
            if len(frames) >= 2:
                dt_ms = (frames[1]["t_capture"] - frames[0]["t_capture"]) * 1000
                zjump = abs(frames[1]["z_error_um"] - frames[0]["z_error_um"])
                all_gaps.append(dt_ms)
                all_jumps.append(zjump)
                all_labels.append(f"{label} r{row_idx}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(all_gaps, all_jumps, s=40, alpha=0.7, edgecolors="black", linewidths=0.5)

    for gap, jump, lbl in zip(all_gaps, all_jumps, all_labels, strict=False):
        ax.annotate(lbl, (gap, jump), fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")

    # Fit line
    gaps = np.array(all_gaps)
    jumps = np.array(all_jumps)
    if len(gaps) > 2:
        coeffs = np.polyfit(gaps, jumps, 1)
        x_fit = np.linspace(gaps.min(), gaps.max(), 100)
        ax.plot(
            x_fit,
            np.polyval(coeffs, x_fit),
            "--",
            color="red",
            alpha=0.5,
            label=f"fit: {coeffs[0]:.3f}·dt + {coeffs[1]:.1f}",
        )
        r2 = 1 - np.sum((jumps - np.polyval(coeffs, gaps)) ** 2) / np.sum((jumps - jumps.mean()) ** 2)
        ax.set_title(f"Frame 0→1 Gap vs Z-Jump (R²={r2:.3f})", fontweight="bold")
        ax.legend()

    ax.set_xlabel("Frame 0→1 gap (ms)")
    ax.set_ylabel("|Z-jump| (µm)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    scan_dirs = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else [
            "scans/chip7_baseline_3row",
            *[f"scans/chip7_repeat_{i}" for i in range(1, 6)],
            "scans/settle_test_steep",
        ]
    )

    # Per-scan row plots for the first scan
    fig1 = plot_scan(scan_dirs[0])
    fig1.savefig("scans/zjump_timing_rows.png", dpi=150, bbox_inches="tight")
    print("Saved scans/zjump_timing_rows.png")

    # Correlation plot across all scans
    fig2 = plot_multi_scan_comparison(scan_dirs)
    fig2.savefig("scans/zjump_gap_vs_magnitude.png", dpi=150, bbox_inches="tight")
    print("Saved scans/zjump_gap_vs_magnitude.png")

    plt.show()
