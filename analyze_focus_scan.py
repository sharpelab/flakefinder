"""Analyze focus quality from scan_area_with_focus.py output.

Computes sharpness metrics per frame, correlates with Z tracking error,
and generates visualizations to validate focus tracking performance.

Usage:
    python analyze_focus_scan.py scans/test_focus_scan/
    python analyze_focus_scan.py scans/test_focus_scan/ --show
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def analyze_frame(image_path: Path) -> dict:
    """Compute sharpness metrics for a single frame.

    Returns:
        Dict with tenengrad and laplacian variance metrics.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"tenengrad": None, "laplacian": None}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tenengrad - Sobel gradient magnitude (good for edges)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    tenengrad = float(cv2.mean(cv2.magnitude(sobel_x, sobel_y))[0])

    # Laplacian variance - high frequency content
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {"tenengrad": tenengrad, "laplacian": laplacian}


def z_error_color(z_error: float | None) -> tuple[int, int, int]:
    """Map z_error to border color (BGR).

    Green: < 0.5µm (excellent)
    Yellow: 0.5-1.7µm (acceptable)
    Red: > 1.7µm (exceeds 20x DOF)
    Gray: unknown
    """
    if z_error is None:
        return (128, 128, 128)  # Gray

    abs_err = abs(z_error)
    if abs_err < 0.5:
        return (0, 200, 0)      # Green
    elif abs_err < 1.7:
        return (0, 200, 200)    # Yellow
    else:
        return (0, 0, 200)      # Red


def create_filmstrip(
    scan_dir: Path,
    frames_data: list[dict],
    output_path: Path,
    thumb_height: int = 120,
    border_width: int = 4,
    max_frames: int = 100,
):
    """Create horizontal filmstrip with z_error-colored borders.

    Args:
        scan_dir: Directory containing frame images.
        frames_data: List of frame dicts with 'n' and 'z_error'.
        output_path: Where to save filmstrip.
        thumb_height: Height of thumbnails in pixels.
        border_width: Border width in pixels.
        max_frames: Max frames to include (evenly sampled if exceeded).
    """
    # Sample frames if too many
    if len(frames_data) > max_frames:
        indices = np.linspace(0, len(frames_data) - 1, max_frames, dtype=int)
        frames_data = [frames_data[i] for i in indices]

    thumbnails = []
    for frame in frames_data:
        frame_path = scan_dir / f"frame_{frame['n']:04d}.jpg"
        if not frame_path.exists():
            continue

        img = Image.open(frame_path)

        # Compute thumbnail size maintaining aspect ratio
        aspect = img.width / img.height
        thumb_width = int(thumb_height * aspect)
        thumb = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

        # Convert to numpy for border drawing
        thumb_arr = np.array(thumb)

        # Add colored border
        color = z_error_color(frame.get("z_error"))
        bordered = cv2.copyMakeBorder(
            thumb_arr, border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, value=color[::-1]  # RGB for PIL
        )

        thumbnails.append(bordered)

    if not thumbnails:
        print("No thumbnails generated")
        return

    # Stack horizontally
    filmstrip = np.hstack(thumbnails)
    Image.fromarray(filmstrip).save(output_path, quality=90)
    print(f"Saved filmstrip ({len(thumbnails)} frames): {output_path}")


def create_comparison(
    scan_dir: Path,
    best_frame: dict,
    worst_frame: dict,
    output_path: Path,
):
    """Create side-by-side comparison of best and worst sharpness frames."""
    best_path = scan_dir / f"frame_{best_frame['n']:04d}.jpg"
    worst_path = scan_dir / f"frame_{worst_frame['n']:04d}.jpg"

    if not best_path.exists() or not worst_path.exists():
        print("Could not find best/worst frames")
        return

    best_img = cv2.imread(str(best_path))
    worst_img = cv2.imread(str(worst_path))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(best_img, f"BEST: frame {best_frame['n']}", (20, 50),
                font, 1.5, (0, 255, 0), 3)
    cv2.putText(best_img, f"tenengrad={best_frame['tenengrad']:.1f}", (20, 100),
                font, 1.0, (255, 255, 255), 2)
    if best_frame.get('z_error') is not None:
        cv2.putText(best_img, f"z_error={best_frame['z_error']:+.2f}um", (20, 140),
                    font, 1.0, (255, 255, 255), 2)

    cv2.putText(worst_img, f"WORST: frame {worst_frame['n']}", (20, 50),
                font, 1.5, (0, 0, 255), 3)
    cv2.putText(worst_img, f"tenengrad={worst_frame['tenengrad']:.1f}", (20, 100),
                font, 1.0, (255, 255, 255), 2)
    if worst_frame.get('z_error') is not None:
        cv2.putText(worst_img, f"z_error={worst_frame['z_error']:+.2f}um", (20, 140),
                    font, 1.0, (255, 255, 255), 2)

    # Stack horizontally
    comparison = np.hstack([best_img, worst_img])
    cv2.imwrite(str(output_path), comparison)
    print(f"Saved comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze focus quality from scan with Z tracking"
    )
    parser.add_argument("scan_dir", type=Path, help="Scan output directory")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--dof", type=float, default=1.7,
                       help="DOF threshold in µm for flagging (default: 1.7 for 20x)")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    meta_path = scan_dir / "scan_meta.json"

    if not meta_path.exists():
        print(f"Error: {meta_path} not found")
        return 1

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    frames_meta = meta.get("frames", [])
    if not frames_meta:
        print("No frames in metadata")
        return 1

    print(f"Analyzing {len(frames_meta)} frames from {scan_dir}")

    # Compute sharpness for each frame
    frames_data = []
    for i, frame in enumerate(frames_meta):
        frame_path = scan_dir / f"frame_{frame['n']:04d}.jpg"
        if not frame_path.exists():
            print(f"  Warning: {frame_path} not found")
            continue

        metrics = analyze_frame(frame_path)

        frames_data.append({
            "n": frame["n"],
            "x_um": frame.get("x_start"),
            "z_actual": frame.get("z_actual"),
            "z_ideal": frame.get("z_ideal"),
            "z_error": frame.get("z_error"),
            "tenengrad": metrics["tenengrad"],
            "laplacian": metrics["laplacian"],
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(frames_meta)} frames")

    print(f"Analyzed {len(frames_data)} frames")

    # Write CSV
    csv_path = scan_dir / "focus_quality.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "x_um", "z_actual", "z_ideal", "z_error", "tenengrad", "laplacian"
        ])
        writer.writeheader()
        writer.writerows(frames_data)
    print(f"Saved CSV: {csv_path}")

    # Compute statistics
    tenengrad_vals = [f["tenengrad"] for f in frames_data if f["tenengrad"] is not None]
    laplacian_vals = [f["laplacian"] for f in frames_data if f["laplacian"] is not None]
    z_errors = [f["z_error"] for f in frames_data if f["z_error"] is not None]

    print("\nSharpness Statistics:")
    if tenengrad_vals:
        print(f"  Tenengrad: mean={np.mean(tenengrad_vals):.1f}, "
              f"std={np.std(tenengrad_vals):.1f}, "
              f"min={np.min(tenengrad_vals):.1f}, max={np.max(tenengrad_vals):.1f}")
    if laplacian_vals:
        print(f"  Laplacian: mean={np.mean(laplacian_vals):.1f}, "
              f"std={np.std(laplacian_vals):.1f}, "
              f"min={np.min(laplacian_vals):.1f}, max={np.max(laplacian_vals):.1f}")

    # Z error statistics
    if z_errors:
        abs_errors = [abs(e) for e in z_errors]
        print(f"\nZ Tracking Error:")
        print(f"  Mean: {np.mean(z_errors):+.3f} µm")
        print(f"  Std: {np.std(z_errors):.3f} µm")
        print(f"  Max |error|: {np.max(abs_errors):.3f} µm")
        print(f"  95th percentile: {np.percentile(abs_errors, 95):.3f} µm")

    # Flag frames exceeding DOF
    flagged = [f for f in frames_data if f["z_error"] is not None and abs(f["z_error"]) > args.dof]
    if flagged:
        print(f"\nFlagged frames (|z_error| > {args.dof} µm): {len(flagged)}")
        for f in flagged[:10]:  # Show first 10
            print(f"  Frame {f['n']}: z_error={f['z_error']:+.2f}µm, "
                  f"tenengrad={f['tenengrad']:.1f}")
        if len(flagged) > 10:
            print(f"  ... and {len(flagged) - 10} more")
    else:
        print(f"\nNo frames exceed DOF threshold ({args.dof} µm)")

    # Find best/worst frames
    valid_frames = [f for f in frames_data if f["tenengrad"] is not None]
    if valid_frames:
        best_frame = max(valid_frames, key=lambda f: f["tenengrad"])
        worst_frame = min(valid_frames, key=lambda f: f["tenengrad"])
        print(f"\nBest frame: {best_frame['n']} (tenengrad={best_frame['tenengrad']:.1f})")
        print(f"Worst frame: {worst_frame['n']} (tenengrad={worst_frame['tenengrad']:.1f})")

    # Create visualizations
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Focus Quality Analysis: {scan_dir.name}", fontsize=14)

    # 1. Sharpness vs X position
    ax1 = axes[0, 0]
    x_vals = [f["x_um"] for f in frames_data if f["x_um"] and f["tenengrad"]]
    ten_vals = [f["tenengrad"] for f in frames_data if f["x_um"] and f["tenengrad"]]
    if x_vals:
        ax1.plot(np.array(x_vals) / 1000, ten_vals, "b-", linewidth=0.8, alpha=0.7)
        ax1.scatter(np.array(x_vals) / 1000, ten_vals, c="blue", s=10, alpha=0.5)
    ax1.set_xlabel("X position (mm)")
    ax1.set_ylabel("Tenengrad sharpness")
    ax1.set_title("Sharpness vs X Position")
    ax1.grid(True, alpha=0.3)

    # 2. Z error vs X position
    ax2 = axes[0, 1]
    x_vals = [f["x_um"] for f in frames_data if f["x_um"] and f["z_error"] is not None]
    z_err = [f["z_error"] for f in frames_data if f["x_um"] and f["z_error"] is not None]
    if x_vals:
        ax2.plot(np.array(x_vals) / 1000, z_err, "r-", linewidth=0.8, alpha=0.7)
        ax2.scatter(np.array(x_vals) / 1000, z_err, c="red", s=10, alpha=0.5)
        ax2.axhline(y=args.dof, color="orange", linestyle="--", label=f"+{args.dof}µm DOF")
        ax2.axhline(y=-args.dof, color="orange", linestyle="--", label=f"-{args.dof}µm DOF")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax2.set_xlabel("X position (mm)")
    ax2.set_ylabel("Z error (µm)")
    ax2.set_title("Z Tracking Error vs X Position")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3. Sharpness vs Z error scatter
    ax3 = axes[1, 0]
    z_err = [f["z_error"] for f in frames_data if f["z_error"] is not None and f["tenengrad"]]
    ten_vals = [f["tenengrad"] for f in frames_data if f["z_error"] is not None and f["tenengrad"]]
    if z_err:
        ax3.scatter(z_err, ten_vals, c="purple", s=20, alpha=0.6)
        ax3.axvline(x=args.dof, color="orange", linestyle="--", alpha=0.7)
        ax3.axvline(x=-args.dof, color="orange", linestyle="--", alpha=0.7)

        # Compute correlation
        if len(z_err) > 2:
            corr = np.corrcoef(np.abs(z_err), ten_vals)[0, 1]
            ax3.text(0.05, 0.95, f"corr(|z_err|, sharpness) = {corr:.3f}",
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top')
    ax3.set_xlabel("Z error (µm)")
    ax3.set_ylabel("Tenengrad sharpness")
    ax3.set_title("Sharpness vs Z Error")
    ax3.grid(True, alpha=0.3)

    # 4. Frame-by-frame timeline
    ax4 = axes[1, 1]
    frame_nums = [f["n"] for f in frames_data if f["tenengrad"]]
    ten_vals = [f["tenengrad"] for f in frames_data if f["tenengrad"]]
    if frame_nums:
        # Normalize sharpness for comparison
        ten_norm = (np.array(ten_vals) - np.mean(ten_vals)) / np.std(ten_vals) if np.std(ten_vals) > 0 else ten_vals
        ax4.plot(frame_nums, ten_norm, "b-", linewidth=0.8, alpha=0.7, label="Sharpness (normalized)")

    z_err = [f["z_error"] for f in frames_data if f["z_error"] is not None]
    frame_nums_z = [f["n"] for f in frames_data if f["z_error"] is not None]
    if z_err:
        ax4.plot(frame_nums_z, z_err, "r-", linewidth=0.8, alpha=0.7, label="Z error (µm)")
    ax4.set_xlabel("Frame number")
    ax4.set_ylabel("Value")
    ax4.set_title("Timeline: Sharpness & Z Error")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = scan_dir / "focus_quality.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plots: {plot_path}")

    if args.show:
        plt.show()
    plt.close()

    # Create filmstrip
    filmstrip_path = scan_dir / "filmstrip.jpg"
    create_filmstrip(scan_dir, frames_data, filmstrip_path)

    # Create best/worst comparison
    if valid_frames:
        comparison_path = scan_dir / "comparison_best_worst.jpg"
        create_comparison(scan_dir, best_frame, worst_frame, comparison_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check success criteria
    success = True
    issues = []

    if tenengrad_vals:
        cv = np.std(tenengrad_vals) / np.mean(tenengrad_vals) * 100
        print(f"Sharpness coefficient of variation: {cv:.1f}%")
        if cv > 20:
            issues.append(f"High sharpness variation ({cv:.1f}% CV)")
            success = False

    if flagged:
        pct_flagged = len(flagged) / len(frames_data) * 100
        print(f"Frames exceeding DOF: {len(flagged)} ({pct_flagged:.1f}%)")
        if pct_flagged > 5:
            issues.append(f"{pct_flagged:.1f}% frames exceed DOF threshold")
            success = False

    # Check for startup transients (first 10% vs rest)
    if len(frames_data) >= 20:
        n_startup = len(frames_data) // 10
        startup_sharpness = [f["tenengrad"] for f in frames_data[:n_startup] if f["tenengrad"]]
        rest_sharpness = [f["tenengrad"] for f in frames_data[n_startup:] if f["tenengrad"]]
        if startup_sharpness and rest_sharpness:
            startup_mean = np.mean(startup_sharpness)
            rest_mean = np.mean(rest_sharpness)
            diff_pct = (startup_mean - rest_mean) / rest_mean * 100
            print(f"Startup vs steady-state sharpness: {diff_pct:+.1f}%")
            if abs(diff_pct) > 15:
                issues.append(f"Startup transient detected ({diff_pct:+.1f}% difference)")
                success = False

    if success:
        print("\nPASS: Focus tracking appears successful")
    else:
        print(f"\nISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")

    return 0


if __name__ == "__main__":
    exit(main())
