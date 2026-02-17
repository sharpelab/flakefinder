"""Compare frame pacing between scan types.

Shows horizontal bars for each frame's capture window, colored by row.
Accepts multiple scan dirs to compare side-by-side.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_scan(scan_dir):
    with open(Path(scan_dir) / "scan_meta.json") as f:
        return json.load(f)


def plot_pacing(scan_dirs, output=None):
    n = len(scan_dirs)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n), squeeze=False)

    for idx, scan_dir in enumerate(scan_dirs):
        ax = axes[idx, 0]
        meta = load_scan(scan_dir)
        frames = meta["frames"]
        rows = meta["lines"]
        label = Path(scan_dir).name

        row_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

        for f in frames:
            ri = f["line"]
            row = rows[ri]
            is_f0 = f["n"] == row["frame_start"]
            is_f1 = f["n"] == row["frame_start"] + 1
            color = "red" if is_f0 else ("orange" if is_f1 else row_colors[ri % len(row_colors)])
            ax.barh(
                ri,
                f["capture_duration_s"],
                left=f["t_capture"],
                height=0.6,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.15,
            )

        # Annotate f0→f1 gaps
        for row in rows:
            rf = sorted([f for f in frames if f["line"] == row["line_idx"]], key=lambda f: f["t_capture"])
            if len(rf) >= 2:
                dt01_ms = (rf[1]["t_capture"] - rf[0]["t_capture"]) * 1000
                mid_t = ((rf[0]["t_capture"] + rf[0]["capture_duration_s"]) + rf[1]["t_capture"]) / 2
                ax.annotate(
                    f"{dt01_ms:.0f}ms",
                    xy=(mid_t, row["line_idx"]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="red",
                    fontweight="bold",
                    xytext=(0, -12),
                    textcoords="offset points",
                )

        ax.set_ylabel("Row")
        ax.set_title(label, fontsize=10, loc="left", fontweight="bold")
        ax.set_yticks(range(len(rows)))
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.2)

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()

    out_path = output or "scans/frame_pacing_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    dirs = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else [
            "scans/area_facade_test",
            "scans/chip7_baseline_3row",
        ]
    )
    plot_pacing(dirs)
