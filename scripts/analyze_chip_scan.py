"""Analyze a completed chip scan for focus quality and sharpness.

Reads scan_meta.json from a scan directory, computes per-frame sharpness
from saved JPGs, and produces quality metrics and plots.

Usage:
    uv run python scripts/analyze_chip_scan.py scans/chip1_20x/
    uv run python scripts/analyze_chip_scan.py scans/chip1_20x/ --sample 5 --min-sharpness 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flakefinder.leica.autofocus import sharpness


def plot_row_map(meta: dict, output_path: Path, *, notes: str | None = None) -> None:
    """Plot chip hull outline with per-row extents and lead-in arrows."""
    rows = meta.get("rows", [])
    if not rows:
        print("  No row data for row map")
        return

    lead_in = meta.get("scan_params", {}).get("lead_in_um", 2000)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # All rows as light background
    for r in rows:
        ax.plot([r["x_min_um"], r["x_max_um"]], [r["y_um"], r["y_um"]], color="lightblue", linewidth=1, alpha=0.5)

    # Hull outline from row extents
    left_edge = [(r["x_min_um"], r["y_um"]) for r in rows]
    right_edge = [(r["x_max_um"], r["y_um"]) for r in rows]
    outline_x = [p[0] for p in left_edge] + [p[0] for p in reversed(right_edge)] + [left_edge[0][0]]
    outline_y = [p[1] for p in left_edge] + [p[1] for p in reversed(right_edge)] + [left_edge[0][1]]
    ax.plot(outline_x, outline_y, "k-", linewidth=1.5, alpha=0.7, label=f"Chip hull ({len(rows)} rows)")

    # Highlight rows with colors (cycle through a palette)
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999"]
    # For large row counts, only label every Nth row
    label_every = max(1, len(rows) // 15)
    for i, r in enumerate(rows):
        ri = r["row_idx"]
        d = r["direction"]
        x_min, x_max = r["x_min_um"], r["x_max_um"]
        color = palette[i % len(palette)]
        width_mm = (x_max - x_min) / 1000
        label = f"Row {ri} ({width_mm:.1f}mm)" if i % label_every == 0 else None
        ax.plot([x_min, x_max], [r["y_um"], r["y_um"]], color=color, linewidth=2, label=label)

        # Lead-in arrow
        if d == 1:
            x_start = x_min - lead_in
            ax.annotate(
                "",
                xy=(x_min, r["y_um"]),
                xytext=(x_start, r["y_um"]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "ls": "--"},
            )
        else:
            x_start = x_max + lead_in
            ax.annotate(
                "",
                xy=(x_max, r["y_um"]),
                xytext=(x_start, r["y_um"]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "ls": "--"},
            )

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    title = "Row Map — Hull extents with lead-in"
    if notes:
        title += f" ({notes})"
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Row map saved to {output_path}")


def load_scan_meta(scan_dir: Path) -> dict:
    """Load scan_meta.json from a scan directory."""
    meta_path = scan_dir / "scan_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"scan_meta.json not found in {scan_dir}")
    with open(meta_path) as f:
        return json.load(f)


def compute_z_tracking_stats(meta: dict) -> dict:
    """Compute Z tracking error statistics from frame metadata.

    Lead-in frames (where ``in_lead_in`` is true) are excluded from all
    statistics.  Older scans without the field are treated as all non-lead-in.

    Returns dict with overall and per-row stats, plus ``n_lead_in``.
    """
    frames = meta["frames"]
    rows = meta["rows"]

    # Separate lead-in vs tracking frames (backwards compatible)
    tracking_frames = [f for f in frames if not f.get("in_lead_in", False)]
    n_lead_in = len(frames) - len(tracking_frames)

    # Overall arrays (tracking frames only)
    z_errors = np.array([f["z_error"] for f in tracking_frames])
    abs_errors = np.abs(z_errors)

    overall = {
        "mean_error_um": float(np.mean(z_errors)),
        "std_um": float(np.std(z_errors)),
        "max_abs_um": float(np.max(abs_errors)),
        "p95_um": float(np.percentile(abs_errors, 95)),
        "pct_outside_2um": float(np.mean(abs_errors > 2.0) * 100),
        "pct_outside_4um": float(np.mean(abs_errors > 4.0) * 100),
        "n_frames": len(tracking_frames),
        "n_lead_in": n_lead_in,
    }

    # Per-row stats (exclude lead-in)
    per_row = []
    for row in rows:
        row_idx = row["row_idx"]
        row_frames = [f for f in tracking_frames if f["row"] == row_idx]
        if not row_frames:
            continue
        row_errors = np.array([f["z_error"] for f in row_frames])
        row_abs = np.abs(row_errors)

        direction = row["direction"]

        # Z-jump: first two non-lead-in frames
        if len(row_frames) >= 2:
            z_jump = abs(row_frames[1]["z_error"] - row_frames[0]["z_error"])
        else:
            z_jump = 0.0

        per_row.append(
            {
                "row_idx": row_idx,
                "y_um": row["y_um"],
                "n_frames": len(row_frames),
                "mean_error_um": float(np.mean(row_errors)),
                "std_um": float(np.std(row_errors)),
                "max_abs_um": float(np.max(row_abs)),
                "direction": direction,
                "z_jump_um": z_jump,
            }
        )

    return {"overall": overall, "per_row": per_row}


def compute_slope_metrics(meta: dict) -> dict:
    """Compute per-row Z slope accuracy, drift, and velocity metrics.

    Uses non-lead-in frames only.  Returns dict with per_row list and overall
    summary.
    """
    frames = meta["frames"]
    rows_meta = meta["rows"]
    plane_a = meta["focus_plane"]["a"]  # um/um  (dZ/dX)
    cmd_speed_mm = meta["scan_params"]["scan_speed_mm_s"]

    tracking = [f for f in frames if not f.get("in_lead_in", False)]

    per_row: list[dict] = []
    for row in rows_meta:
        ri = row["row_idx"]
        direction = row["direction"]
        rf = [f for f in tracking if f["row"] == ri]
        if len(rf) < 5:
            continue

        x = np.array([f["x_start"] for f in rf])
        z = np.array([f["z_actual"] for f in rf])
        t = np.array([f["t_start"] for f in rf])
        z_err = np.array([f["z_error"] for f in rf])
        frame_ns = np.array([f["n"] for f in rf])

        # 1. Z slope: linear fit of z_actual vs x (dZ/dX, direction-independent)
        coeffs = np.polyfit(x, z, 1)  # [slope_um_um, intercept]
        actual_slope = coeffs[0] * 1000  # um/mm
        ideal_slope = plane_a * 1000  # um/mm
        slope_err_pct = (actual_slope - ideal_slope) / abs(ideal_slope) * 100 if abs(ideal_slope) > 0.001 else 0.0

        # Fit line for z_error vs frame_num (for plotting)
        err_coeffs = np.polyfit(frame_ns.astype(float), z_err, 1)

        # 2. Z drift: mean z_error first 5 vs last 5
        n_edge = min(5, len(rf) // 2)
        drift_start = float(np.mean(z_err[:n_edge]))
        drift_end = float(np.mean(z_err[-n_edge:]))
        drift_um = drift_end - drift_start

        # 3. Velocities via linear regression (accurate cruise speed)
        t_rel = t - t[0]
        t_var = float(np.sum((t_rel - np.mean(t_rel)) ** 2))

        x_coeffs = np.polyfit(t_rel, x, 1)
        x_speed_mean = float(abs(x_coeffs[0])) / 1000  # mm/s
        x_fit = np.polyval(x_coeffs, t_rel)
        x_resid_var = float(np.sum((x - x_fit) ** 2)) / (len(x) - 2) if len(x) > 2 else 0.0
        x_speed_std = float(np.sqrt(x_resid_var / t_var)) / 1000 if t_var > 0 else 0.0  # mm/s

        z_coeffs = np.polyfit(t_rel, z, 1)
        z_speed_mean = float(z_coeffs[0])  # um/s (signed, matches direction)
        z_fit = np.polyval(z_coeffs, t_rel)
        z_resid_var = float(np.sum((z - z_fit) ** 2)) / (len(z) - 2) if len(z) > 2 else 0.0
        z_speed_std = float(np.sqrt(z_resid_var / t_var)) if t_var > 0 else 0.0  # um/s

        ideal_z_vel = plane_a * direction * cmd_speed_mm * 1000  # um/s
        row_width_mm = (row["x_max_um"] - row["x_min_um"]) / 1000

        per_row.append(
            {
                "row_idx": ri,
                "direction": direction,
                "row_width_mm": row_width_mm,
                "n_frames": len(rf),
                "actual_slope_um_mm": float(actual_slope),
                "ideal_slope_um_mm": float(ideal_slope),
                "slope_error_pct": float(slope_err_pct),
                "drift_start_um": drift_start,
                "drift_end_um": drift_end,
                "drift_um": drift_um,
                "x_speed_mean_mm_s": x_speed_mean,
                "x_speed_std_mm_s": x_speed_std,
                "z_speed_mean_um_s": z_speed_mean,
                "z_speed_std_um_s": z_speed_std,
                "ideal_z_vel_um_s": float(ideal_z_vel),
                # For plotting: fit line coefficients (z_error vs frame_num)
                "err_fit_coeffs": [float(err_coeffs[0]), float(err_coeffs[1])],
                "frame_range": [int(frame_ns[0]), int(frame_ns[-1])],
            }
        )

    # Overall summary
    if per_row:
        slope_errs = [r["slope_error_pct"] for r in per_row]
        drifts = [r["drift_um"] for r in per_row]
        overall = {
            "mean_slope_error_pct": float(np.mean(slope_errs)),
            "mean_abs_slope_error_pct": float(np.mean(np.abs(slope_errs))),
            "mean_drift_um": float(np.mean(drifts)),
            "mean_abs_drift_um": float(np.mean(np.abs(drifts))),
            "max_abs_drift_um": float(np.max(np.abs(drifts))),
            "cmd_speed_mm_s": cmd_speed_mm,
            "plane_a_um_um": plane_a,
        }
    else:
        overall = {}

    return {"overall": overall, "per_row": per_row}


def compute_sharpness_values(scan_dir: Path, meta: dict, sample_every: int = 1) -> dict:
    """Compute tenengrad sharpness for saved frame JPGs.

    Args:
        scan_dir: Directory containing frame_NNNN.jpg files.
        meta: Loaded scan_meta.json dict.
        sample_every: Only process every Nth frame (1 = all).

    Returns:
        Dict with frame indices, sharpness values, positions, and per-row stats.
    """
    frames = meta["frames"]
    total = len(frames)

    indices = []
    sharpness_vals = []
    x_positions = []
    y_positions = []
    row_ids = []

    t0 = time.monotonic()
    processed = 0

    for i, frame in enumerate(frames):
        if i % sample_every != 0:
            continue

        img_path = scan_dir / f"frame_{frame['n']:04d}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        s = sharpness(img, method="tenengrad")
        indices.append(frame["n"])
        sharpness_vals.append(s)
        x_positions.append(frame["x_start"])
        y_positions.append(frame["y_um"])
        row_ids.append(frame["row"])
        processed += 1

        # Progress every 100 frames
        if processed % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = processed / elapsed
            remaining = (total / sample_every - processed) / rate if rate > 0 else 0
            print(
                f"  Processed {processed}/{total // sample_every} frames ({rate:.0f} fps, ~{remaining:.0f}s remaining)",
                end="\r",
            )

    elapsed = time.monotonic() - t0
    print(f"  Processed {processed} frames in {elapsed:.1f}s ({processed / elapsed:.0f} fps)        ")

    sharpness_arr = np.array(sharpness_vals)
    indices_arr = np.array(indices)
    x_arr = np.array(x_positions)
    y_arr = np.array(y_positions)
    row_arr = np.array(row_ids)

    # Per-row sharpness stats
    unique_rows = sorted(set(row_ids))
    per_row = []
    for r in unique_rows:
        mask = row_arr == r
        row_sharp = sharpness_arr[mask]
        per_row.append(
            {
                "row_idx": r,
                "n_frames": int(mask.sum()),
                "mean": float(np.mean(row_sharp)),
                "std": float(np.std(row_sharp)),
                "min": float(np.min(row_sharp)),
            }
        )

    return {
        "indices": indices_arr,
        "sharpness": sharpness_arr,
        "x_um": x_arr,
        "y_um": y_arr,
        "rows": row_arr,
        "per_row": per_row,
        "sample_every": sample_every,
        "overall_mean": float(np.mean(sharpness_arr)),
        "overall_std": float(np.std(sharpness_arr)),
        "overall_min": float(np.min(sharpness_arr)),
        "overall_max": float(np.max(sharpness_arr)),
    }


def _plot_spatial_z_error(
    ax: plt.Axes,
    frames: list[dict],
    z_errors: list,
    lead_in_mask: list[bool],
) -> None:
    """Render the spatial Z error map on *ax*."""
    x_mm = np.array([f["x_start"] for f in frames]) / 1000
    y_mm = np.array([f["y_um"] for f in frames]) / 1000
    z_err_arr = np.array(z_errors)
    li_mask = np.array(lead_in_mask)

    abs_max = max(np.percentile(np.abs(z_err_arr), 99), 0.5)
    track = ~li_mask
    if np.any(track):
        sc = ax.scatter(
            x_mm[track],
            y_mm[track],
            c=z_err_arr[track],
            cmap="coolwarm",
            s=2,
            alpha=0.6,
            vmin=-abs_max,
            vmax=abs_max,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Z error (um)")
    if np.any(li_mask):
        ax.scatter(
            x_mm[li_mask],
            y_mm[li_mask],
            c="gray",
            marker="x",
            s=6,
            alpha=0.3,
            rasterized=True,
        )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Spatial Z Error Map")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # +Y down


def plot_analysis(
    meta: dict,
    z_stats: dict,
    sharpness_data: dict | None,
    output_path: Path,
    min_sharpness: float | None,
    *,
    slope_metrics: dict | None = None,
    **kwargs,
) -> None:
    """Generate combined analysis plot and save to output_path."""
    frames = meta["frames"]
    n_rows = len(meta["rows"])
    spatial_z = kwargs.get("spatial_z", False)

    # Determine layout
    has_sharpness = sharpness_data is not None
    if has_sharpness and spatial_z:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_zerr, ax_sharp = axes[0, 0], axes[0, 1]
        ax_zmap, ax_sharpmap = axes[1, 0], axes[1, 1]
    elif has_sharpness:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_zerr, ax_sharp = axes[0, 0], axes[0, 1]
        ax_sharpmap = axes[1, 1]
        fig.delaxes(axes[1, 0])  # spatial Z hidden by default
    elif spatial_z:
        fig, (ax_zerr, ax_zmap) = plt.subplots(1, 2, figsize=(16, 5))
    else:
        fig, ax_zerr = plt.subplots(1, 1, figsize=(10, 5))

    # Color map for rows
    cmap = plt.colormaps["viridis"]
    row_indices = sorted(set(f["row"] for f in frames))
    row_colors = {r: cmap(i / max(1, len(row_indices) - 1)) for i, r in enumerate(row_indices)}

    # --- Panel 1: Z error vs frame number ---
    # Separate lead-in from tracking frames for distinct styling
    lead_in_mask = [f.get("in_lead_in", False) for f in frames]
    tracking_idx = [i for i, li in enumerate(lead_in_mask) if not li]
    lead_in_idx = [i for i, li in enumerate(lead_in_mask) if li]

    frame_nums = [f["n"] for f in frames]
    z_errors = [f["z_error"] for f in frames]
    frame_rows = [f["row"] for f in frames]
    colors = [row_colors[r] for r in frame_rows]

    # Tracking frames: colored dots
    if tracking_idx:
        ax_zerr.scatter(
            [frame_nums[i] for i in tracking_idx],
            [z_errors[i] for i in tracking_idx],
            c=[colors[i] for i in tracking_idx],
            s=2,
            alpha=0.6,
            rasterized=True,
        )
    # Lead-in frames: gray x markers
    if lead_in_idx:
        ax_zerr.scatter(
            [frame_nums[i] for i in lead_in_idx],
            [z_errors[i] for i in lead_in_idx],
            c="gray",
            marker="x",
            s=6,
            alpha=0.3,
            rasterized=True,
            label=f"Lead-in ({len(lead_in_idx)})",
        )

    # DOF bands
    for dof, alpha, label in [(2, 0.15, "2 um DOF"), (4, 0.08, "4 um DOF")]:
        ax_zerr.axhspan(-dof, dof, alpha=alpha, color="green", label=label)

    ax_zerr.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ov = z_stats["overall"]
    n_lead_in = ov.get("n_lead_in", 0)
    lead_in_note = f"  [{n_lead_in} lead-in excluded]" if n_lead_in > 0 else ""
    ax_zerr.set_xlabel("Frame number")
    ax_zerr.set_ylabel("Z error (um)")
    ax_zerr.set_title(
        f"Z Tracking Error: mean={ov['mean_error_um']:.3f}, "
        f"std={ov['std_um']:.3f}, max={ov['max_abs_um']:.3f}, "
        f"p95={ov['p95_um']:.3f} um\n"
        f"Outside 2um: {ov['pct_outside_2um']:.1f}%, "
        f"Outside 4um: {ov['pct_outside_4um']:.1f}%{lead_in_note}"
    )
    ax_zerr.legend(loc="upper right", fontsize=8)

    # Add row boundary markers
    for row in meta["rows"]:
        ax_zerr.axvline(row["frame_start"], color="gray", alpha=0.15, linewidth=0.5)

    # Overlay per-row linear fit lines (slope drift)
    if slope_metrics:
        for sr in slope_metrics["per_row"]:
            f0, f1 = sr["frame_range"]
            xs = np.array([f0, f1], dtype=float)
            ys = sr["err_fit_coeffs"][0] * xs + sr["err_fit_coeffs"][1]
            ax_zerr.plot(xs, ys, color="black", linewidth=1.0, alpha=0.7)
        # Annotation
        sm_ov = slope_metrics["overall"]
        ax_zerr.text(
            0.01,
            0.01,
            f"|Slope err|: {sm_ov['mean_abs_slope_error_pct']:.2f}%  |Drift|: {sm_ov['mean_abs_drift_um']:.2f} um",
            transform=ax_zerr.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
        )

    # --- Spatial Z error map (only when --spatial-z) ---
    if spatial_z and not has_sharpness:
        _plot_spatial_z_error(ax_zmap, frames, z_errors, lead_in_mask)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {output_path}")
        return

    if not has_sharpness:
        # Z-only, no spatial → single panel done
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {output_path}")
        return

    # --- Panel 2: Sharpness vs frame number ---
    assert sharpness_data is not None  # guaranteed by has_sharpness branch above
    s_indices = sharpness_data["indices"]
    s_vals = sharpness_data["sharpness"]
    s_rows = sharpness_data["rows"]
    s_colors = [row_colors[r] for r in s_rows]

    ax_sharp.scatter(s_indices, s_vals, c=s_colors, s=4, alpha=0.6, rasterized=True)

    if min_sharpness is not None:
        ax_sharp.axhline(
            min_sharpness,
            color="red",
            linewidth=1,
            linestyle="--",
            label=f"Threshold ({min_sharpness})",
        )
        below = s_vals < min_sharpness
        n_below = int(np.sum(below))
        if n_below > 0:
            ax_sharp.scatter(
                s_indices[below],
                s_vals[below],
                s=20,
                facecolors="none",
                edgecolors="red",
                linewidths=1,
                zorder=5,
                label=f"Below threshold ({n_below})",
            )

    ax_sharp.set_xlabel("Frame number")
    ax_sharp.set_ylabel("Tenengrad sharpness")
    sample_note = f" (every {sharpness_data['sample_every']}th)" if sharpness_data["sample_every"] > 1 else ""
    ax_sharp.set_title(
        f"Frame Sharpness{sample_note}: mean={sharpness_data['overall_mean']:.1f}, "
        f"std={sharpness_data['overall_std']:.1f}, "
        f"min={sharpness_data['overall_min']:.1f}"
    )
    ax_sharp.legend(loc="lower right", fontsize=8)

    # Row boundaries
    for row in meta["rows"]:
        ax_sharp.axvline(row["frame_start"], color="gray", alpha=0.15, linewidth=0.5)

    # --- Panel 3: Spatial Z error map (only when --spatial-z) ---
    if spatial_z:
        _plot_spatial_z_error(ax_zmap, frames, z_errors, lead_in_mask)

    # --- Panel 4: Spatial sharpness map ---
    sx_mm = sharpness_data["x_um"] / 1000
    sy_mm = sharpness_data["y_um"] / 1000

    sc2 = ax_sharpmap.scatter(
        sx_mm,
        sy_mm,
        c=sharpness_data["sharpness"],
        cmap="plasma",
        s=4,
        alpha=0.6,
        rasterized=True,
    )
    plt.colorbar(sc2, ax=ax_sharpmap, label="Tenengrad sharpness")

    if min_sharpness is not None:
        below = sharpness_data["sharpness"] < min_sharpness
        if np.any(below):
            ax_sharpmap.scatter(
                sx_mm[below],
                sy_mm[below],
                s=20,
                facecolors="none",
                edgecolors="red",
                linewidths=1,
                zorder=5,
                label=f"Below {min_sharpness}",
            )
            ax_sharpmap.legend(loc="lower right", fontsize=8)

    ax_sharpmap.set_xlabel("X (mm)")
    ax_sharpmap.set_ylabel("Y (mm)")
    ax_sharpmap.set_title("Spatial Sharpness Map")
    ax_sharpmap.set_aspect("equal")
    ax_sharpmap.invert_yaxis()  # +Y down

    # Suptitle with scan summary
    scan_time = meta.get("scan_duration_s", 0)
    n_frames = meta.get("frame_count", len(frames))
    chip_info = meta.get("chip_info", {})
    chip_idx = chip_info.get("chip_index", "?")
    obj_mag = meta.get("optics", {}).get("objective_mag", "?")
    notes = kwargs.get("notes")
    notes_str = f"  [{notes}]" if notes else ""
    fig.suptitle(
        f"Scan Analysis: chip {chip_idx} @ {obj_mag}x  |  "
        f"{n_frames} frames, {scan_time:.0f}s  |  {n_rows} rows{notes_str}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def print_summary(
    meta: dict,
    z_stats: dict,
    sharpness_data: dict | None,
    min_sharpness: float | None,
    slope_metrics: dict | None = None,
    *,
    quiet: bool = False,
) -> None:
    """Print text summary to console.

    When *quiet* is True, print only a compact 2-3 line summary suitable
    for pipeline use.
    """
    chip_info = meta.get("chip_info", {})
    chip_idx = chip_info.get("chip_index", "?")
    obj_mag = meta.get("optics", {}).get("objective_mag", "?")
    n_frames = meta.get("frame_count", "?")
    n_rows = len(meta.get("rows", []))
    duration = meta.get("scan_duration_s", 0)
    ov = z_stats["overall"]

    if quiet:
        print(f"Scan: chip {chip_idx} @ {obj_mag}x | {n_frames} frames, {duration:.1f}s | {n_rows} rows")
        drift_str = ""
        if slope_metrics and slope_metrics.get("overall"):
            drift_str = f", drift={slope_metrics['overall']['max_abs_drift_um']:.2f} um"
        print(
            f"Z tracking: std={ov['std_um']:.2f} um, p95={ov['p95_um']:.2f} um"
            f"{drift_str}, "
            f">2um: {ov['pct_outside_2um']:.1f}%, >4um: {ov['pct_outside_4um']:.1f}%"
        )
        if sharpness_data is not None:
            n_below = 0
            if min_sharpness is not None:
                n_below = int(np.sum(sharpness_data["sharpness"] < min_sharpness))
                thresh_str = f", {n_below} below {min_sharpness}"
            else:
                thresh_str = ""
            print(
                f"Sharpness: mean={sharpness_data['overall_mean']:.1f}, "
                f"min={sharpness_data['overall_min']:.1f}{thresh_str}"
            )
        return

    print()
    print("=" * 65)
    print("SCAN ANALYSIS")
    print("=" * 65)

    # Scan info
    print(f"Timestamp:    {meta.get('timestamp', 'unknown')}")
    print(f"Chip index:   {chip_idx}")
    print(f"Objective:    {obj_mag}x")
    n_lead_in = ov.get("n_lead_in", 0)
    lead_in_str = f" ({n_lead_in} lead-in excluded from stats)" if n_lead_in > 0 else ""
    print(f"Frames:       {n_frames}{lead_in_str}")
    print(f"Rows:         {n_rows}")
    print(f"Duration:     {duration:.1f}s")

    # Focus plane
    fp = meta.get("focus_plane", {})
    if fp:
        print(f"\nFocus plane:  {fp.get('equation', 'N/A')}")

    # Z tracking
    print()
    print("--- Z Tracking ---")
    print(f"Mean error:     {ov['mean_error_um']:+.4f} um")
    print(f"Std:            {ov['std_um']:.4f} um")
    print(f"Max |error|:    {ov['max_abs_um']:.4f} um")
    print(f"P95 |error|:    {ov['p95_um']:.4f} um")
    print(f"Outside 2 um:   {ov['pct_outside_2um']:.1f}% ({int(ov['n_frames'] * ov['pct_outside_2um'] / 100)} frames)")
    print(f"Outside 4 um:   {ov['pct_outside_4um']:.1f}% ({int(ov['n_frames'] * ov['pct_outside_4um'] / 100)} frames)")

    # Per-row Z table
    print("\n--- Per-Row Z Error ---")
    print(f"{'Row':>3}  {'Dir':>3}  {'Frames':>6}  {'Mean err':>9}  {'Max |err|':>9}  {'Z-jump':>9}")
    print(f"{'---':>3}  {'---':>3}  {'------':>6}  {'---------':>9}  {'---------':>9}  {'---------':>9}")
    for r in z_stats["per_row"]:
        dir_str = "+X" if r["direction"] > 0 else "-X"
        zjump_str = f"{r['z_jump_um']:.3f}"
        me = r["mean_error_um"]
        mx = r["max_abs_um"]
        print(f"{r['row_idx']:3d}   {dir_str:>2}  {r['n_frames']:6d}  {me:+9.3f}  {mx:9.3f}  {zjump_str:>9}")

    # Directional bias summary
    pos_rows = [r for r in z_stats["per_row"] if r["direction"] > 0]
    neg_rows = [r for r in z_stats["per_row"] if r["direction"] < 0]
    print("\n--- Directional Bias ---")
    if pos_rows:
        pos_mean = np.mean([r["mean_error_um"] for r in pos_rows])
        pos_zjump = np.mean([r["z_jump_um"] for r in pos_rows])
        print(f"+X rows ({len(pos_rows):2d}):  mean Z err = {pos_mean:+.3f} um,  mean z-jump = {pos_zjump:.3f} um")
    if neg_rows:
        neg_mean = np.mean([r["mean_error_um"] for r in neg_rows])
        neg_zjump = np.mean([r["z_jump_um"] for r in neg_rows])
        print(f"-X rows ({len(neg_rows):2d}):  mean Z err = {neg_mean:+.3f} um,  mean z-jump = {neg_zjump:.3f} um")
    if pos_rows and neg_rows:
        print(f"Directional split:  {pos_mean - neg_mean:+.3f} um (+X minus -X)")

    # Slope & drift
    if slope_metrics and slope_metrics["per_row"]:
        sm = slope_metrics
        print("\n--- Z Slope & Drift ---")
        hdr = (
            f"{'Row':>3}  {'Dir':>3}  {'Width':>6}  "
            f"{'Slope':>7} {'Ideal':>7} {'Err%':>6}  "
            f"{'Drift':>7}  {'X mm/s':>7} {'Z um/s':>7}"
        )
        print(hdr)
        sep = (
            f"{'---':>3}  {'---':>3}  {'-----':>6}  "
            f"{'-----':>7} {'-----':>7} {'----':>6}  "
            f"{'-----':>7}  {'------':>7} {'------':>7}"
        )
        print(sep)
        for r in sm["per_row"]:
            d = "+X" if r["direction"] > 0 else "-X"
            print(
                f"{r['row_idx']:3d}   {d:>2}  {r['row_width_mm']:5.1f}m  "
                f"{r['actual_slope_um_mm']:+7.3f} {r['ideal_slope_um_mm']:+7.3f} "
                f"{r['slope_error_pct']:+5.1f}%  "
                f"{r['drift_um']:+7.3f}  "
                f"{r['x_speed_mean_mm_s']:7.2f} {r['z_speed_mean_um_s']:+7.1f}"
            )
        ov = sm["overall"]
        print(
            f"\nMean |slope error|: {ov['mean_abs_slope_error_pct']:.2f}% "
            f"(signed: {ov['mean_slope_error_pct']:+.2f}%), "
            f"Mean |drift|: {ov['mean_abs_drift_um']:.2f} um "
            f"(signed: {ov['mean_drift_um']:+.2f}), "
            f"Max |drift|: {ov['max_abs_drift_um']:.2f} um"
        )

    # Sharpness
    if sharpness_data is not None:
        print()
        print("--- Sharpness (Tenengrad) ---")
        sample_note = (
            f" (sampled every {sharpness_data['sample_every']}th frame)" if sharpness_data["sample_every"] > 1 else ""
        )
        print(f"Frames analyzed: {len(sharpness_data['indices'])}{sample_note}")
        print(f"Mean:   {sharpness_data['overall_mean']:.1f}")
        print(f"Std:    {sharpness_data['overall_std']:.1f}")
        print(f"Min:    {sharpness_data['overall_min']:.1f}")
        print(f"Max:    {sharpness_data['overall_max']:.1f}")

        if min_sharpness is not None:
            below = sharpness_data["sharpness"] < min_sharpness
            n_below = int(np.sum(below))
            pct_below = np.mean(below) * 100
            print(f"\nBelow threshold ({min_sharpness}):")
            print(f"  {n_below} frames ({pct_below:.1f}%)")

            if n_below > 0 and n_below <= 20:
                idx_below = sharpness_data["indices"][below]
                row_below = sharpness_data["rows"][below]
                s_below = sharpness_data["sharpness"][below]
                for fi, ri, si in zip(idx_below, row_below, s_below, strict=True):
                    print(f"  frame_{fi:04d}.jpg  row {ri}  sharpness={si:.1f}")

        # Per-row sharpness (worst rows)
        worst_sharp = sorted(sharpness_data["per_row"], key=lambda r: r["min"])
        print("\nWorst sharpness rows (top 5 by min):")
        for r in worst_sharp[:5]:
            print(
                f"  Row {r['row_idx']:2d}: "
                f"mean={r['mean']:.1f}, std={r['std']:.1f}, "
                f"min={r['min']:.1f}  [{r['n_frames']} frames]"
            )

    print()
    print("=" * 65)


def print_comparison(
    z_stats_a: dict,
    z_stats_b: dict,
    label_a: str,
    label_b: str,
) -> None:
    """Print side-by-side comparison of two scans."""
    ov_a, ov_b = z_stats_a["overall"], z_stats_b["overall"]

    print()
    print("=" * 65)
    print("SCAN COMPARISON")
    print("=" * 65)

    # Overall stats table
    wa = max(len(label_a), 10)
    wb = max(len(label_b), 10)
    print(f"{'Metric':<20}  {label_a:>{wa}}  {label_b:>{wb}}")
    print(f"{'------':<20}  {'-' * wa}  {'-' * wb}")

    rows = [
        ("Frames", f"{ov_a['n_frames']}", f"{ov_b['n_frames']}"),
        ("Mean error (um)", f"{ov_a['mean_error_um']:+.4f}", f"{ov_b['mean_error_um']:+.4f}"),
        ("Std (um)", f"{ov_a['std_um']:.4f}", f"{ov_b['std_um']:.4f}"),
        ("Max |error| (um)", f"{ov_a['max_abs_um']:.4f}", f"{ov_b['max_abs_um']:.4f}"),
        ("P95 |error| (um)", f"{ov_a['p95_um']:.4f}", f"{ov_b['p95_um']:.4f}"),
        ("Outside 2 um", f"{ov_a['pct_outside_2um']:.1f}%", f"{ov_b['pct_outside_2um']:.1f}%"),
        ("Outside 4 um", f"{ov_a['pct_outside_4um']:.1f}%", f"{ov_b['pct_outside_4um']:.1f}%"),
    ]
    for label, va, vb in rows:
        print(f"{label:<20}  {va:>{wa}}  {vb:>{wb}}")

    # Directional bias comparison
    def _dir_stats(per_row, sign):
        matched = [r for r in per_row if r["direction"] * sign > 0]
        if not matched:
            return None, None
        return (
            float(np.mean([r["mean_error_um"] for r in matched])),
            float(np.mean([r["z_jump_um"] for r in matched])) if matched else None,
        )

    print(f"\n{'Directional bias':<20}  {label_a:>{wa}}  {label_b:>{wb}}")
    print(f"{'----------------':<20}  {'-' * wa}  {'-' * wb}")
    for dir_name, sign in [("+X mean err", 1), ("-X mean err", -1)]:
        mean_a, _ = _dir_stats(z_stats_a["per_row"], sign)
        mean_b, _ = _dir_stats(z_stats_b["per_row"], sign)
        va = f"{mean_a:+.3f}" if mean_a is not None else "N/A"
        vb = f"{mean_b:+.3f}" if mean_b is not None else "N/A"
        print(f"{dir_name:<20}  {va:>{wa}}  {vb:>{wb}}")
    for dir_name, sign in [("+X z-jump", 1), ("-X z-jump", -1)]:
        _, zjump_a = _dir_stats(z_stats_a["per_row"], sign)
        _, zjump_b = _dir_stats(z_stats_b["per_row"], sign)
        va = f"{zjump_a:.3f}" if zjump_a is not None else "N/A"
        vb = f"{zjump_b:.3f}" if zjump_b is not None else "N/A"
        print(f"{dir_name:<20}  {va:>{wa}}  {vb:>{wb}}")

    # Per-row comparison (paired by row index)
    rows_a = {r["row_idx"]: r for r in z_stats_a["per_row"]}
    rows_b = {r["row_idx"]: r for r in z_stats_b["per_row"]}
    common_rows = sorted(set(rows_a.keys()) & set(rows_b.keys()))

    if common_rows:
        hdr = f"{'Row':>3}  {'Dir':>3}  {'Mean A':>8}  {'Mean B':>8}"
        hdr += f"  {'Delta':>7}  {'Max A':>7}  {'Max B':>7}"
        print(f"\n{hdr}")
        sep = f"{'---':>3}  {'---':>3}  {'------':>8}  {'------':>8}"
        sep += f"  {'-----':>7}  {'-----':>7}  {'-----':>7}"
        print(sep)
        for ri in common_rows:
            ra, rb = rows_a[ri], rows_b[ri]
            dir_str = "+X" if ra["direction"] > 0 else "-X"
            delta = rb["mean_error_um"] - ra["mean_error_um"]
            line = f"{ri:3d}   {dir_str:>2}  {ra['mean_error_um']:+8.3f}"
            line += f"  {rb['mean_error_um']:+8.3f}  {delta:+7.3f}"
            line += f"  {ra['max_abs_um']:7.3f}  {rb['max_abs_um']:7.3f}"
            print(line)

    print()
    print("=" * 65)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a completed chip scan for focus quality and sharpness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scan_dir",
        type=Path,
        help="Path to scan directory containing scan_meta.json and frame JPGs",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=None,
        help="Threshold for flagging poorly focused frames",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="Compute sharpness on every Nth frame (for speed on large scans)",
    )
    parser.add_argument(
        "--no-sharpness",
        action="store_true",
        help="Skip sharpness computation (only analyze Z tracking from metadata)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output plot path (default: scan_analysis.png in scan dir)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Label for plot title and output filename (e.g. 'raw_async')",
    )
    parser.add_argument(
        "--spatial-z",
        action="store_true",
        help="Include spatial Z error map panel in plots (off by default)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Compact output: one-line scan info + Z quality verdict only",
    )
    parser.add_argument(
        "--row-map",
        action="store_true",
        help="Generate chip hull + row extent plot with lead-in arrows",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        metavar="SCAN_DIR_B",
        help="Second scan directory for side-by-side comparison",
    )
    args = parser.parse_args()

    scan_dir = args.scan_dir.resolve()
    if not scan_dir.is_dir():
        print(f"Error: Not a directory: {scan_dir}")
        return 1

    # Load metadata
    if not args.quiet:
        print(f"Loading scan metadata from {scan_dir}")
    meta = load_scan_meta(scan_dir)
    n_frames = meta.get("frame_count", len(meta["frames"]))
    n_rows = len(meta.get("rows", []))
    if not args.quiet:
        print(f"  {n_frames} frames, {n_rows} rows, {meta.get('scan_duration_s', 0):.1f}s")

    # Z tracking analysis (always, from metadata only)
    z_stats = compute_z_tracking_stats(meta)
    slope_metrics = compute_slope_metrics(meta)

    # Sharpness analysis (optional, reads JPGs)
    sharpness_data = None
    if not args.no_sharpness:
        if not args.quiet:
            print(f"Computing sharpness (sample every {args.sample})...")
        sharpness_data = compute_sharpness_values(scan_dir, meta, sample_every=max(1, args.sample))

    # Print summary
    print_summary(meta, z_stats, sharpness_data, args.min_sharpness, slope_metrics, quiet=args.quiet)

    # Comparison mode
    if args.compare:
        compare_dir = args.compare.resolve()
        if not compare_dir.is_dir():
            print(f"Error: Compare directory not found: {compare_dir}")
            return 1
        print(f"\nLoading comparison scan from {compare_dir}")
        meta_b = load_scan_meta(compare_dir)
        z_stats_b = compute_z_tracking_stats(meta_b)
        print_comparison(
            z_stats,
            z_stats_b,
            label_a=scan_dir.name,
            label_b=compare_dir.name,
        )

    # Generate plot
    if args.output:
        output_path = args.output
    else:
        output_path = scan_dir / "scan_analysis.png"
    if not args.quiet:
        print("Generating plot...")
    plot_analysis(
        meta,
        z_stats,
        sharpness_data,
        output_path,
        args.min_sharpness,
        slope_metrics=slope_metrics,
        notes=args.notes,
        spatial_z=args.spatial_z,
    )

    # Row map plot
    if args.row_map:
        suffix = f"_{args.notes}" if args.notes else ""
        row_map_path = scan_dir / f"row_map{suffix}.png"
        plot_row_map(meta, row_map_path, notes=args.notes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
