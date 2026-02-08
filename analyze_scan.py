"""Analyze a completed chip scan for focus quality and sharpness.

Reads scan_meta.json from a scan directory, computes per-frame sharpness
from saved JPGs, and produces quality metrics and plots.

Usage:
    uv run python analyze_scan.py scans/chip1_20x/
    uv run python analyze_scan.py scans/chip1_20x/ --sample 5 --min-sharpness 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.flakefinder.leica.autofocus import sharpness


def load_scan_meta(scan_dir: Path) -> dict:
    """Load scan_meta.json from a scan directory."""
    meta_path = scan_dir / "scan_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"scan_meta.json not found in {scan_dir}")
    with open(meta_path) as f:
        return json.load(f)


def compute_z_tracking_stats(meta: dict) -> dict:
    """Compute Z tracking error statistics from frame metadata.

    Returns dict with overall and per-row stats.
    """
    frames = meta["frames"]
    rows = meta["rows"]

    # Overall arrays
    z_errors = np.array([f["z_error"] for f in frames])
    abs_errors = np.abs(z_errors)

    overall = {
        "mean_error_um": float(np.mean(z_errors)),
        "std_um": float(np.std(z_errors)),
        "max_abs_um": float(np.max(abs_errors)),
        "p95_um": float(np.percentile(abs_errors, 95)),
        "pct_outside_2um": float(np.mean(abs_errors > 2.0) * 100),
        "pct_outside_4um": float(np.mean(abs_errors > 4.0) * 100),
        "n_frames": len(frames),
    }

    # Per-row stats
    per_row = []
    for row in rows:
        row_idx = row["row_idx"]
        fs = row["frame_start"]
        fe = row["frame_end"]
        row_frames = [f for f in frames if f["row"] == row_idx]
        if not row_frames:
            continue
        row_errors = np.array([f["z_error"] for f in row_frames])
        row_abs = np.abs(row_errors)
        per_row.append({
            "row_idx": row_idx,
            "y_um": row["y_um"],
            "n_frames": len(row_frames),
            "mean_error_um": float(np.mean(row_errors)),
            "std_um": float(np.std(row_errors)),
            "max_abs_um": float(np.max(row_abs)),
        })

    return {"overall": overall, "per_row": per_row}


def compute_sharpness_values(
    scan_dir: Path, meta: dict, sample_every: int = 1
) -> dict:
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

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        s = sharpness(img, method="tenengrad")
        indices.append(frame["n"])
        sharpness_vals.append(s)
        # Use midpoint of x_start/x_end as frame X position
        x_positions.append((frame["x_start"] + frame["x_end"]) / 2)
        y_positions.append(frame["y_um"])
        row_ids.append(frame["row"])
        processed += 1

        # Progress every 100 frames
        if processed % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = processed / elapsed
            remaining = (total / sample_every - processed) / rate if rate > 0 else 0
            print(
                f"  Processed {processed}/{total // sample_every} frames "
                f"({rate:.0f} fps, ~{remaining:.0f}s remaining)",
                end="\r",
            )

    elapsed = time.monotonic() - t0
    print(
        f"  Processed {processed} frames in {elapsed:.1f}s "
        f"({processed / elapsed:.0f} fps)        "
    )

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
        per_row.append({
            "row_idx": r,
            "n_frames": int(mask.sum()),
            "mean": float(np.mean(row_sharp)),
            "std": float(np.std(row_sharp)),
            "min": float(np.min(row_sharp)),
        })

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


def plot_analysis(
    meta: dict,
    z_stats: dict,
    sharpness_data: dict | None,
    output_path: Path,
    min_sharpness: float | None,
    **kwargs,
) -> None:
    """Generate combined analysis plot and save to output_path."""
    frames = meta["frames"]
    n_rows = len(meta["rows"])

    # Determine layout: 2x2 if sharpness data, 1x1 if only Z
    has_sharpness = sharpness_data is not None
    if has_sharpness:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        ax_zerr, ax_sharp, ax_zmap, ax_sharpmap = (
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
        )
    else:
        fig, ax_zerr = plt.subplots(1, 1, figsize=(12, 5))

    # Color map for rows
    cmap = plt.cm.viridis
    row_indices = sorted(set(f["row"] for f in frames))
    row_colors = {r: cmap(i / max(1, len(row_indices) - 1)) for i, r in enumerate(row_indices)}

    # --- Panel 1: Z error vs frame number ---
    frame_nums = [f["n"] for f in frames]
    z_errors = [f["z_error"] for f in frames]
    frame_rows = [f["row"] for f in frames]
    colors = [row_colors[r] for r in frame_rows]

    ax_zerr.scatter(frame_nums, z_errors, c=colors, s=2, alpha=0.6, rasterized=True)

    # DOF bands
    for dof, alpha, label in [(2, 0.15, "2 um DOF"), (4, 0.08, "4 um DOF")]:
        ax_zerr.axhspan(-dof, dof, alpha=alpha, color="green", label=label)

    ax_zerr.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ov = z_stats["overall"]
    ax_zerr.set_xlabel("Frame number")
    ax_zerr.set_ylabel("Z error (um)")
    ax_zerr.set_title(
        f"Z Tracking Error: mean={ov['mean_error_um']:.3f}, "
        f"std={ov['std_um']:.3f}, max={ov['max_abs_um']:.3f}, "
        f"p95={ov['p95_um']:.3f} um\n"
        f"Outside 2um: {ov['pct_outside_2um']:.1f}%, "
        f"Outside 4um: {ov['pct_outside_4um']:.1f}%"
    )
    ax_zerr.legend(loc="upper right", fontsize=8)

    # Add row boundary markers
    for row in meta["rows"]:
        ax_zerr.axvline(row["frame_start"], color="gray", alpha=0.15, linewidth=0.5)

    if not has_sharpness:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Plot saved to {output_path}")
        return

    # --- Panel 2: Sharpness vs frame number ---
    s_indices = sharpness_data["indices"]
    s_vals = sharpness_data["sharpness"]
    s_rows = sharpness_data["rows"]
    s_colors = [row_colors[r] for r in s_rows]

    ax_sharp.scatter(s_indices, s_vals, c=s_colors, s=4, alpha=0.6, rasterized=True)

    if min_sharpness is not None:
        ax_sharp.axhline(
            min_sharpness, color="red", linewidth=1, linestyle="--",
            label=f"Threshold ({min_sharpness})",
        )
        below = s_vals < min_sharpness
        n_below = int(np.sum(below))
        if n_below > 0:
            ax_sharp.scatter(
                s_indices[below], s_vals[below],
                s=20, facecolors="none", edgecolors="red", linewidths=1,
                zorder=5, label=f"Below threshold ({n_below})",
            )

    ax_sharp.set_xlabel("Frame number")
    ax_sharp.set_ylabel("Tenengrad sharpness")
    sample_note = (
        f" (every {sharpness_data['sample_every']}th)"
        if sharpness_data["sample_every"] > 1
        else ""
    )
    ax_sharp.set_title(
        f"Frame Sharpness{sample_note}: mean={sharpness_data['overall_mean']:.1f}, "
        f"std={sharpness_data['overall_std']:.1f}, "
        f"min={sharpness_data['overall_min']:.1f}"
    )
    ax_sharp.legend(loc="lower right", fontsize=8)

    # Row boundaries
    for row in meta["rows"]:
        ax_sharp.axvline(row["frame_start"], color="gray", alpha=0.15, linewidth=0.5)

    # --- Panel 3: Spatial Z error map ---
    x_mm = np.array([(f["x_start"] + f["x_end"]) / 2 for f in frames]) / 1000
    y_mm = np.array([f["y_um"] for f in frames]) / 1000
    z_err_arr = np.array(z_errors)

    # Use absolute Z error for color
    abs_max = max(np.percentile(np.abs(z_err_arr), 99), 0.5)
    sc = ax_zmap.scatter(
        x_mm, y_mm, c=z_err_arr, cmap="coolwarm",
        s=2, alpha=0.6, vmin=-abs_max, vmax=abs_max, rasterized=True,
    )
    plt.colorbar(sc, ax=ax_zmap, label="Z error (um)")
    ax_zmap.set_xlabel("X (mm)")
    ax_zmap.set_ylabel("Y (mm)")
    ax_zmap.set_title("Spatial Z Error Map")
    ax_zmap.set_aspect("equal")
    ax_zmap.invert_yaxis()  # +Y down

    # --- Panel 4: Spatial sharpness map ---
    sx_mm = sharpness_data["x_um"] / 1000
    sy_mm = sharpness_data["y_um"] / 1000

    sc2 = ax_sharpmap.scatter(
        sx_mm, sy_mm, c=sharpness_data["sharpness"], cmap="plasma",
        s=4, alpha=0.6, rasterized=True,
    )
    plt.colorbar(sc2, ax=ax_sharpmap, label="Tenengrad sharpness")

    if min_sharpness is not None:
        below = sharpness_data["sharpness"] < min_sharpness
        if np.any(below):
            ax_sharpmap.scatter(
                sx_mm[below], sy_mm[below],
                s=20, facecolors="none", edgecolors="red", linewidths=1,
                zorder=5, label=f"Below {min_sharpness}",
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
) -> None:
    """Print text summary to console."""
    print()
    print("=" * 65)
    print("SCAN ANALYSIS")
    print("=" * 65)

    # Scan info
    chip_info = meta.get("chip_info", {})
    print(f"Timestamp:    {meta.get('timestamp', 'unknown')}")
    print(f"Chip index:   {chip_info.get('chip_index', '?')}")
    print(f"Objective:    {meta.get('optics', {}).get('objective_mag', '?')}x")
    print(f"Frames:       {meta.get('frame_count', '?')}")
    print(f"Rows:         {len(meta.get('rows', []))}")
    print(f"Duration:     {meta.get('scan_duration_s', 0):.1f}s")

    # Focus plane
    fp = meta.get("focus_plane", {})
    if fp:
        print(f"\nFocus plane:  {fp.get('equation', 'N/A')}")

    # Z tracking
    ov = z_stats["overall"]
    print()
    print("--- Z Tracking ---")
    print(f"Mean error:     {ov['mean_error_um']:+.4f} um")
    print(f"Std:            {ov['std_um']:.4f} um")
    print(f"Max |error|:    {ov['max_abs_um']:.4f} um")
    print(f"P95 |error|:    {ov['p95_um']:.4f} um")
    print(f"Outside 2 um:   {ov['pct_outside_2um']:.1f}% ({int(ov['n_frames'] * ov['pct_outside_2um'] / 100)} frames)")
    print(f"Outside 4 um:   {ov['pct_outside_4um']:.1f}% ({int(ov['n_frames'] * ov['pct_outside_4um'] / 100)} frames)")

    # Per-row Z (worst rows)
    worst_rows = sorted(z_stats["per_row"], key=lambda r: r["max_abs_um"], reverse=True)
    print(f"\nWorst Z rows (top 5 by max |error|):")
    for r in worst_rows[:5]:
        print(
            f"  Row {r['row_idx']:2d} (y={r['y_um']/1000:.2f} mm): "
            f"mean={r['mean_error_um']:+.3f}, std={r['std_um']:.3f}, "
            f"max={r['max_abs_um']:.3f} um  [{r['n_frames']} frames]"
        )

    # Sharpness
    if sharpness_data is not None:
        print()
        print("--- Sharpness (Tenengrad) ---")
        sample_note = (
            f" (sampled every {sharpness_data['sample_every']}th frame)"
            if sharpness_data["sample_every"] > 1
            else ""
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
                for fi, ri, si in zip(idx_below, row_below, s_below):
                    print(f"  frame_{fi:04d}.jpg  row {ri}  sharpness={si:.1f}")

        # Per-row sharpness (worst rows)
        worst_sharp = sorted(sharpness_data["per_row"], key=lambda r: r["min"])
        print(f"\nWorst sharpness rows (top 5 by min):")
        for r in worst_sharp[:5]:
            print(
                f"  Row {r['row_idx']:2d}: "
                f"mean={r['mean']:.1f}, std={r['std']:.1f}, "
                f"min={r['min']:.1f}  [{r['n_frames']} frames]"
            )

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
    args = parser.parse_args()

    scan_dir = args.scan_dir.resolve()
    if not scan_dir.is_dir():
        print(f"Error: Not a directory: {scan_dir}")
        return 1

    # Load metadata
    print(f"Loading scan metadata from {scan_dir}")
    meta = load_scan_meta(scan_dir)
    n_frames = meta.get("frame_count", len(meta["frames"]))
    n_rows = len(meta.get("rows", []))
    print(f"  {n_frames} frames, {n_rows} rows, {meta.get('scan_duration_s', 0):.1f}s")

    # Z tracking analysis (always, from metadata only)
    z_stats = compute_z_tracking_stats(meta)

    # Sharpness analysis (optional, reads JPGs)
    sharpness_data = None
    if not args.no_sharpness:
        print(f"Computing sharpness (sample every {args.sample})...")
        sharpness_data = compute_sharpness_values(
            scan_dir, meta, sample_every=args.sample
        )

    # Print summary
    print_summary(meta, z_stats, sharpness_data, args.min_sharpness)

    # Generate plot
    if args.output:
        output_path = args.output
    elif args.notes:
        output_path = scan_dir / f"scan_analysis_{args.notes}.png"
    else:
        output_path = scan_dir / "scan_analysis.png"
    print(f"Generating plot...")
    plot_analysis(meta, z_stats, sharpness_data, output_path, args.min_sharpness, notes=args.notes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
