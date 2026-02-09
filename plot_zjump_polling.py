"""Plot full-run Z and X position timeseries with camera frame pacing.

Uses polling_debug.json (Z/X timeseries) + scan_meta.json (frames, rows).
Three vertically stacked panels sharing a time axis:
  1. Polled Z (µm) — raw Z position over the entire run
  2. Polled X (mm) — raw X position over the entire run
  3. Frame pacing — horizontal bars showing each frame's capture window
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_data(scan_dir):
    scan_dir = Path(scan_dir)
    with open(scan_dir / "scan_meta.json") as f:
        meta = json.load(f)
    with open(scan_dir / "polling_debug.json") as f:
        polling = json.load(f)
    return meta, polling


def plot_polling(scan_dir, output=None):
    meta, polling = load_data(scan_dir)
    rows = meta["rows"]
    frames = meta["frames"]

    z_t = np.array([s["t"] for s in polling["z_samples"]])
    z_um = np.array([s["z_um"] for s in polling["z_samples"]])
    x_t = np.array([(s["t_before"] + s["t_after"]) / 2 for s in polling["x_samples"]])
    x_um = np.array([s["x_um"] for s in polling["x_samples"]])

    fig, (ax_z, ax_x, ax_f) = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.suptitle(f"Full Run: {Path(scan_dir).name}", fontsize=13, fontweight="bold")

    # Row background shading (all panels)
    row_colors = ["#e8f0fe", "#fef3e8"]
    for row in rows:
        rf = [f for f in frames if f["row"] == row["row_idx"]]
        if not rf:
            continue
        t0 = rf[0]["t_start"] - 0.1
        t1 = rf[-1]["t_end"] + 0.1
        color = row_colors[row["row_idx"] % 2]
        for ax in (ax_z, ax_x, ax_f):
            ax.axvspan(t0, t1, alpha=0.4, color=color, zorder=0)
        direction = "+X" if row["direction"] == 1 else "-X"
        ax_z.text((t0 + t1) / 2, ax_z.get_ylim()[0] if ax_z.get_ylim()[0] != 0 else z_um.min(),
                  f"R{row['row_idx']} ({direction})", ha="center", va="bottom",
                  fontsize=7, alpha=0.6)

    # Panel 1: Z position
    ax_z.plot(z_t, z_um, "-", color="C0", linewidth=0.6, alpha=0.8)
    ax_z.set_ylabel("Z position (µm)")
    ax_z.grid(True, alpha=0.2)

    # Panel 2: X position
    ax_x.plot(x_t, x_um / 1000, "-", color="C2", linewidth=0.6, alpha=0.8)
    ax_x.set_ylabel("X position (mm)")
    ax_x.grid(True, alpha=0.2)

    # Panel 3: Frame pacing — horizontal bars per row
    for f in frames:
        row_idx = f["row"]
        color = "red" if f["n"] == meta["rows"][row_idx]["frame_start"] else (
            "orange" if f["n"] == meta["rows"][row_idx]["frame_start"] + 1 else "C0")
        ax_f.barh(row_idx, f["t_end"] - f["t_start"], left=f["t_start"],
                  height=0.6, color=color, alpha=0.7, edgecolor="black", linewidth=0.2)

    # Annotate frame 0→1 gaps
    for row in rows:
        rf = sorted([f for f in frames if f["row"] == row["row_idx"]], key=lambda f: f["t_start"])
        if len(rf) >= 2:
            dt_ms = (rf[1]["t_start"] - rf[0]["t_end"]) * 1000
            mid_t = (rf[0]["t_end"] + rf[1]["t_start"]) / 2
            ax_f.annotate(f"{dt_ms:.0f}ms", xy=(mid_t, row["row_idx"]),
                          ha="center", va="center", fontsize=7, color="red", fontweight="bold",
                          xytext=(0, 10), textcoords="offset points")

    ax_f.set_ylabel("Row")
    ax_f.set_xlabel("Time (s)")
    ax_f.set_yticks(range(len(rows)))
    ax_f.invert_yaxis()
    ax_f.grid(True, axis="x", alpha=0.2)

    # Re-annotate row labels on Z panel now that ylim is set
    for row in rows:
        rf = [f for f in frames if f["row"] == row["row_idx"]]
        if not rf:
            continue
        t_mid = (rf[0]["t_start"] + rf[-1]["t_end"]) / 2
        direction = "+X" if row["direction"] == 1 else "-X"
        ax_z.text(t_mid, ax_z.get_ylim()[1], f"R{row['row_idx']} ({direction})",
                  ha="center", va="top", fontsize=8, alpha=0.5)

    plt.tight_layout()

    out_path = output or str(Path(scan_dir) / "zjump_polling.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    return fig


if __name__ == "__main__":
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else "scans/chip7_debug_polling"
    output = sys.argv[2] if len(sys.argv) > 2 else None
    plot_polling(scan_dir, output)
