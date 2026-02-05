"""Analyze focus map data and produce visualizations.

Loads focus_map_chip*.json, identifies out-of-focus points, fits a tilt plane,
and generates surface plots.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from PIL import Image


def sharpness_tenengrad(image: np.ndarray) -> float:
    """Tenengrad sharpness - Sobel gradient magnitude mean."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.mean(cv2.magnitude(sobel_x, sobel_y))[0]


def sharpness_laplacian(image: np.ndarray) -> float:
    """Laplacian variance - measures edges/high frequency content."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def sharpness_normalized_variance(image: np.ndarray) -> float:
    """Normalized variance - variance / mean, texture measure."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    mean = gray.mean()
    if mean == 0:
        return 0
    return gray.var() / mean


def load_focus_map(path: Path) -> dict:
    """Load focus map JSON."""
    with open(path) as f:
        return json.load(f)


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane z = ax + by + c to the data.

    Returns:
        (coefficients [a, b, c], r_squared)
    """
    # Build design matrix [x, y, 1]
    A = np.column_stack([x, y, np.ones_like(x)])

    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)

    # R-squared
    z_pred = A @ coeffs
    ss_res = np.sum((z - z_pred) ** 2)
    ss_tot = np.sum((z - np.mean(z)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return coeffs, r_squared


def get_sharpness(point: dict, prefer_best: bool = True) -> float | None:
    """Get sharpness value from point, preferring best_sharpness over legacy sharpness."""
    if prefer_best and "best_sharpness" in point:
        return point.get("best_sharpness")
    # Fall back to legacy 'sharpness' field or final_sharpness
    return point.get("sharpness") or point.get("final_sharpness")


def analyze_focus_map(data: dict) -> dict:
    """Analyze focus map data.

    Returns dict with analysis results.
    """
    points = data["sample_points"]

    # Extract arrays
    x = np.array([p["x_um"] for p in points])
    y = np.array([p["y_um"] for p in points])
    z = np.array([p["best_z_um"] for p in points if p["best_z_um"] is not None])

    # Prefer best_sharpness, fall back to legacy sharpness field
    sharpness = np.array([get_sharpness(p) for p in points if get_sharpness(p) is not None])
    types = [p["type"] for p in points]

    # Filter to successful points
    valid_mask = np.array([p["best_z_um"] is not None for p in points])
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Check for sharpness drift (best vs final differ significantly)
    sharpness_drift_indices = []
    sharpness_drift_threshold = 0.2  # 20% difference
    for i, p in enumerate(points):
        if not valid_mask[i]:
            continue
        best = p.get("best_sharpness")
        final = p.get("final_sharpness")
        if best is not None and final is not None and best > 0:
            drift_ratio = abs(best - final) / best
            if drift_ratio > sharpness_drift_threshold:
                sharpness_drift_indices.append(i)

    # Basic stats
    z_mean = np.mean(z)
    z_std = np.std(z)
    z_range = np.ptp(z)

    sharpness_mean = np.mean(sharpness)
    sharpness_std = np.std(sharpness)

    # Identify low sharpness points (below mean - 1 std)
    sharpness_threshold = sharpness_mean - sharpness_std
    low_sharpness_mask = sharpness < sharpness_threshold
    low_sharpness_indices = np.where(low_sharpness_mask)[0]

    # Fit tilt plane
    coeffs, r_squared = fit_plane(x_valid, y_valid, z)

    # Tilt in µm per mm
    tilt_x = coeffs[0] * 1000  # µm/mm
    tilt_y = coeffs[1] * 1000  # µm/mm
    tilt_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)
    tilt_direction = np.degrees(np.arctan2(tilt_y, tilt_x))

    # Residuals from plane fit
    z_pred = coeffs[0] * x_valid + coeffs[1] * y_valid + coeffs[2]
    residuals = z - z_pred
    residual_std = np.std(residuals)

    # Outliers (residuals > 2 std from plane)
    outlier_mask = np.abs(residuals) > 2 * residual_std
    outlier_indices = np.where(outlier_mask)[0]

    return {
        "n_points": len(points),
        "n_valid": len(z),
        "z_mean": z_mean,
        "z_std": z_std,
        "z_range": z_range,
        "z_min": np.min(z),
        "z_max": np.max(z),
        "sharpness_mean": sharpness_mean,
        "sharpness_std": sharpness_std,
        "sharpness_min": np.min(sharpness),
        "sharpness_max": np.max(sharpness),
        "sharpness_threshold": sharpness_threshold,
        "low_sharpness_indices": low_sharpness_indices,
        "sharpness_drift_indices": sharpness_drift_indices,
        "plane_coeffs": coeffs,
        "plane_r_squared": r_squared,
        "tilt_x_um_per_mm": tilt_x,
        "tilt_y_um_per_mm": tilt_y,
        "tilt_magnitude_um_per_mm": tilt_magnitude,
        "tilt_direction_deg": tilt_direction,
        "residual_std": residual_std,
        "outlier_indices": outlier_indices,
        "x": x_valid,
        "y": y_valid,
        "z": z,
        "sharpness": sharpness,
        "types": [types[i] for i in range(len(types)) if valid_mask[i]],
        "residuals": residuals,
    }


def plot_focus_map(data: dict, analysis: dict, output_path: Path) -> None:
    """Generate focus map visualization."""
    fig = plt.figure(figsize=(18, 12))

    x = analysis["x"] / 1000  # Convert to mm
    y = analysis["y"] / 1000
    z = analysis["z"]
    sharpness = analysis["sharpness"]
    residuals = analysis["residuals"]
    coeffs = analysis["plane_coeffs"]

    # 1. 3D interpolated surface from sample points
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # Create interpolated surface
    x_grid = np.linspace(x.min(), x.max(), 50)
    y_grid = np.linspace(y.min(), y.max(), 50)
    X_surf, Y_surf = np.meshgrid(x_grid, y_grid)

    # Interpolate Z values onto grid
    Z_surf = griddata((x, y), z, (X_surf, Y_surf), method='cubic')

    # Plot interpolated surface
    surf = ax1.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', alpha=0.8,
                            linewidth=0, antialiased=True)

    # Overlay sample points
    colors = ['red' if t == 'contour' else 'white' for t in analysis["types"]]
    ax1.scatter(x, y, z, c=colors, s=30, edgecolors='black', linewidths=0.5, zorder=5)

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (µm)')
    ax1.set_title('Interpolated Focus Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='Z (µm)')

    # 2. 3D points with fitted plane
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')

    # Color by type
    colors = ['blue' if t == 'contour' else 'green' for t in analysis["types"]]
    ax2.scatter(x, y, z, c=colors, s=50, alpha=0.8)

    # Plot fitted plane
    Z_plane = coeffs[0] * X_surf * 1000 + coeffs[1] * Y_surf * 1000 + coeffs[2]
    ax2.plot_surface(X_surf, Y_surf, Z_plane, alpha=0.3, color='red')

    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (µm)')
    ax2.set_title(f'Fitted Plane (R²={analysis["plane_r_squared"]:.3f})\n'
                  f'Tilt: {analysis["tilt_magnitude_um_per_mm"]:.1f} µm/mm @ {analysis["tilt_direction_deg"]:.0f}°')

    # 3. Interpolated surface - top-down heatmap view
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.pcolormesh(X_surf, Y_surf, Z_surf, cmap='viridis', shading='auto')
    ax3.scatter(x, y, c='white', s=30, edgecolors='black', linewidths=0.5)
    plt.colorbar(im, ax=ax3, label='Z (µm)')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_title('Interpolated Surface (top view)')
    ax3.set_aspect('equal')

    # 4. Sharpness heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    scatter = ax4.scatter(x, y, c=sharpness, cmap='plasma', s=100, edgecolors='black')
    plt.colorbar(scatter, ax=ax4, label='Sharpness')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_title(f'Sharpness Map (mean={analysis["sharpness_mean"]:.1f}, std={analysis["sharpness_std"]:.1f})')
    ax4.set_aspect('equal')

    # Mark outliers
    if len(analysis["outlier_indices"]) > 0:
        out_x = x[analysis["outlier_indices"]]
        out_y = y[analysis["outlier_indices"]]
        ax4.scatter(out_x, out_y, s=200, facecolors='none', edgecolors='cyan',
                   linewidths=2, label='Z outliers')
        ax4.legend()

    # 5. Residuals from plane fit
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(x, y, c=residuals, cmap='coolwarm', s=100, edgecolors='black',
                         vmin=-3*analysis["residual_std"], vmax=3*analysis["residual_std"])
    plt.colorbar(scatter, ax=ax5, label='Residual (µm)')
    ax5.set_xlabel('X (mm)')
    ax5.set_ylabel('Y (mm)')
    ax5.set_title(f'Residuals from Plane Fit (std={analysis["residual_std"]:.1f} µm)')
    ax5.set_aspect('equal')

    # 6. Z heatmap (top-down view)
    ax6 = fig.add_subplot(2, 3, 6)
    scatter = ax6.scatter(x, y, c=z, cmap='viridis', s=100, edgecolors='black')
    plt.colorbar(scatter, ax=ax6, label='Z (µm)')
    ax6.set_xlabel('X (mm)')
    ax6.set_ylabel('Y (mm)')
    ax6.set_title('Z Position Map')
    ax6.set_aspect('equal')

    # Mark low sharpness points
    if len(analysis["low_sharpness_indices"]) > 0:
        low_x = x[analysis["low_sharpness_indices"]]
        low_y = y[analysis["low_sharpness_indices"]]
        ax6.scatter(low_x, low_y, s=200, facecolors='none', edgecolors='red',
                   linewidths=2, label='Low sharpness')
        ax6.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def print_report(data: dict, analysis: dict) -> None:
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("FOCUS MAP ANALYSIS")
    print("=" * 60)

    print(f"\nSource: {data.get('source_chips_meta', 'unknown')}")
    print(f"Chip ID: {data.get('chip_id', 'unknown')}")
    print(f"Duration: {data.get('duration_s', 0):.1f}s")

    print(f"\n--- Points ---")
    print(f"Total points: {analysis['n_points']}")
    print(f"Valid points: {analysis['n_valid']}")

    print(f"\n--- Z Statistics ---")
    print(f"Z range: {analysis['z_min']:.1f} - {analysis['z_max']:.1f} µm ({analysis['z_range']:.1f} µm total)")
    print(f"Z mean: {analysis['z_mean']:.1f} µm")
    print(f"Z std: {analysis['z_std']:.1f} µm")

    print(f"\n--- Tilt Analysis ---")
    print(f"Plane fit R²: {analysis['plane_r_squared']:.3f}")
    print(f"Tilt X: {analysis['tilt_x_um_per_mm']:.2f} µm/mm")
    print(f"Tilt Y: {analysis['tilt_y_um_per_mm']:.2f} µm/mm")
    print(f"Tilt magnitude: {analysis['tilt_magnitude_um_per_mm']:.2f} µm/mm")
    print(f"Tilt direction: {analysis['tilt_direction_deg']:.1f}°")
    print(f"Residual std: {analysis['residual_std']:.1f} µm (surface roughness/error)")

    print(f"\n--- Sharpness Statistics ---")
    print(f"Sharpness range: {analysis['sharpness_min']:.1f} - {analysis['sharpness_max']:.1f}")
    print(f"Sharpness mean: {analysis['sharpness_mean']:.1f}")
    print(f"Sharpness std: {analysis['sharpness_std']:.1f}")

    print(f"\n--- Potential Issues ---")

    low_sharp = analysis["low_sharpness_indices"]
    if len(low_sharp) > 0:
        print(f"Low sharpness points ({len(low_sharp)}, < {analysis['sharpness_threshold']:.1f}):")
        points = data["sample_points"]
        for idx in low_sharp:
            p = points[idx]
            sharpness = p.get('best_sharpness', p.get('sharpness', 0))
            print(f"  {p['type']} {p['index']}: sharpness={sharpness:.1f}, "
                  f"pos=({p['x_um']/1000:.2f}, {p['y_um']/1000:.2f}) mm")
    else:
        print("No low sharpness points detected.")

    outliers = analysis["outlier_indices"]
    if len(outliers) > 0:
        print(f"\nZ outliers ({len(outliers)}, > 2σ from plane):")
        points = data["sample_points"]
        valid_points = [p for p in points if p["best_z_um"] is not None]
        for idx in outliers:
            p = valid_points[idx]
            res = analysis["residuals"][idx]
            print(f"  {p['type']} {p['index']}: Z={p['best_z_um']:.1f} µm, "
                  f"residual={res:+.1f} µm")
    else:
        print("No Z outliers detected.")

    drift_indices = analysis["sharpness_drift_indices"]
    if len(drift_indices) > 0:
        print(f"\nSharpness drift detected ({len(drift_indices)} points, best vs final >20% diff):")
        points = data["sample_points"]
        for idx in drift_indices:
            p = points[idx]
            best = p.get("best_sharpness", 0)
            final = p.get("final_sharpness", 0)
            drift_pct = abs(best - final) / best * 100 if best > 0 else 0
            print(f"  {p['type']} {p['index']}: best={best:.1f}, final={final:.1f} "
                  f"({drift_pct:+.0f}%)")
    else:
        print("No sharpness drift detected.")

    # DOF warning
    dof_20x = 4  # µm, approximate for 20x
    if analysis["z_range"] > 5 * dof_20x:
        print(f"\n⚠️  WARNING: Z range ({analysis['z_range']:.0f} µm) is ~{analysis['z_range']/dof_20x:.0f}x "
              f"the DOF of a 20x objective (~{dof_20x} µm)")
        print(f"   Consider leveling the sample or using focus tracking during scans.")

    print("\n" + "=" * 60)


def create_mosaic(
    data: dict,
    images_dir: Path,
    output_path: Path,
    max_dim: int = 3000,
    margin_px: int = 5,
) -> None:
    """Create mosaic image with focus map images laid out in a grid.

    Algorithm:
    1. Determine grid dimensions from unique X/Y positions
    2. Add 2 to each dimension for contour images on edges
    3. Compute thumbnail size to fit: (canvas - margins) / n_images
    4. Place images at their relative grid positions
    5. Label each image with ID, sharpness, and relative Z offset

    Args:
        data: Focus map data dict.
        images_dir: Directory containing the after images.
        output_path: Output mosaic image path.
        max_dim: Maximum canvas dimension in pixels.
        margin_px: Gap between images in pixels.
    """
    points = data["sample_points"]

    # Filter to points with images
    grid_points = [p for p in points if p.get("image") and p["type"] == "grid"]
    contour_points = [p for p in points if p.get("image") and p["type"] == "contour"]
    all_points = grid_points + contour_points

    if not all_points:
        print("No images found in focus map data")
        return

    # Compute mean Z for relative offsets
    z_values = [p["best_z_um"] for p in all_points if p.get("best_z_um") is not None]
    z_mean = np.mean(z_values) if z_values else 0

    # Load first image to get aspect ratio
    first_img_path = images_dir / all_points[0]["image"]
    if not first_img_path.exists():
        print(f"Image not found: {first_img_path}")
        return

    first_img = cv2.imread(str(first_img_path))
    img_h, img_w = first_img.shape[:2]
    img_aspect = img_w / img_h  # e.g., 1.5 for landscape

    # Determine grid dimensions from unique positions
    if grid_points:
        grid_x = np.array([p["x_um"] for p in grid_points])
        grid_y = np.array([p["y_um"] for p in grid_points])

        # Find unique X and Y values (with tolerance for floating point)
        def unique_with_tolerance(arr, tol=100):
            sorted_arr = np.sort(arr)
            unique = [sorted_arr[0]]
            for v in sorted_arr[1:]:
                if v - unique[-1] > tol:
                    unique.append(v)
            return np.array(unique)

        unique_x = unique_with_tolerance(grid_x)
        unique_y = unique_with_tolerance(grid_y)
        n_cols_grid = len(unique_x)
        n_rows_grid = len(unique_y)
    else:
        n_cols_grid = 1
        n_rows_grid = 1
        unique_x = np.array([0])
        unique_y = np.array([0])

    # Total columns/rows including contours on edges
    n_cols = n_cols_grid + 2
    n_rows = n_rows_grid + 2

    print(f"Grid: {n_cols_grid}x{n_rows_grid}, Total with contours: {n_cols}x{n_rows}")

    # Margins: double margin between contour and grid regions
    contour_margin = 2 * margin_px
    grid_margin = margin_px

    # Compute thumbnail size to fit canvas
    # Layout: [contour col][contour_margin][grid cols with grid_margin between][contour_margin][contour col]
    # Width = 2*thumb_w + 2*contour_margin + n_cols_grid*thumb_w + (n_cols_grid-1)*grid_margin
    # Width = (n_cols_grid + 2)*thumb_w + 2*contour_margin + (n_cols_grid-1)*grid_margin

    # First determine canvas aspect from grid aspect
    grid_aspect = n_cols / n_rows * img_aspect

    if grid_aspect > 1:
        canvas_w = max_dim
        canvas_h = int(max_dim / grid_aspect)
    else:
        canvas_h = max_dim
        canvas_w = int(max_dim * grid_aspect)

    # Compute thumbnail dimensions accounting for double margin at contour/grid boundary
    # canvas_w = n_cols * thumb_w + 2 * contour_margin + (n_cols_grid - 1) * grid_margin
    total_h_margin = 2 * contour_margin + max(0, n_cols_grid - 1) * grid_margin
    total_v_margin = 2 * contour_margin + max(0, n_rows_grid - 1) * grid_margin

    thumb_w = int((canvas_w - total_h_margin) / n_cols)
    thumb_h = int((canvas_h - total_v_margin) / n_rows)

    # Maintain image aspect ratio - use the smaller dimension
    if thumb_w / thumb_h > img_aspect:
        thumb_w = int(thumb_h * img_aspect)
    else:
        thumb_h = int(thumb_w / img_aspect)

    # Recalculate canvas to fit exactly
    canvas_w = n_cols * thumb_w + total_h_margin
    canvas_h = n_rows * thumb_h + total_v_margin

    print(f"Thumbnails: {thumb_w}x{thumb_h} px")
    print(f"Canvas: {canvas_w}x{canvas_h} px")

    # Create canvas
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    # Grid bounds for determining contour placement
    if grid_points:
        grid_x_min, grid_x_max = grid_x.min(), grid_x.max()
        grid_y_min, grid_y_max = grid_y.min(), grid_y.max()
        grid_x_span = grid_x_max - grid_x_min if grid_x_max > grid_x_min else 1
        grid_y_span = grid_y_max - grid_y_min if grid_y_max > grid_y_min else 1
    else:
        grid_x_min = grid_x_max = grid_x_span = 0
        grid_y_min = grid_y_max = grid_y_span = 1

    # Label drawing parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, thumb_h / 350)  # Scale font based on thumbnail size
    font_thickness = max(1, int(thumb_h / 200))
    line_height = int(font_scale * 28)  # Increased spacing between lines

    def draw_label(canvas, x0, y0, point, z_mean, img=None):
        """Draw ID, sharpness metrics, and Z offset label on image."""
        # Build label: "g05" or "c02" format
        prefix = "g" if point["type"] == "grid" else "c"
        label_id = f"{prefix}{point['index']:02d}"

        # Use pre-computed metrics from full-size image if available
        if "_s" in point:
            s0 = point["_s"]
            s1 = point["_l"]
            s2 = point["_v"]
            sharpness_str = f"S:{s0:.0f} L:{s1:.0f}"
            sharpness_str2 = f"V:{s2:.1f}"
        else:
            # Fall back to stored sharpness
            sharpness = get_sharpness(point)
            sharpness_str = f"S:{sharpness:.0f}" if sharpness else "S:--"
            sharpness_str2 = None
            s0 = sharpness

        # Z offset from mean
        z = point.get("best_z_um")
        if z is not None:
            z_offset = z - z_mean
            z_str = f"Z:{z_offset:+.0f}"
        else:
            z_str = "Z:--"

        # Text color: red for low sharpness, white otherwise
        low_sharpness = s0 is not None and s0 < 30
        # Also flag if significant drift between best and final
        best = point.get("best_sharpness")
        final = point.get("final_sharpness")
        has_drift = (best is not None and final is not None and best > 0 and
                     abs(best - final) / best > 0.2)
        text_color = (0, 0, 255) if (low_sharpness or has_drift) else (255, 255, 255)  # BGR

        # Draw background rectangle for readability
        label_lines = [label_id, sharpness_str]
        if sharpness_str2:
            label_lines.append(sharpness_str2)
        label_lines.append(z_str)

        max_text_width = 0
        for line in label_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            max_text_width = max(max_text_width, text_w)

        bg_height = len(label_lines) * line_height + 6
        bg_width = max_text_width + 8

        # Draw semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0 + 2, y0 + 2), (x0 + bg_width + 2, y0 + bg_height + 2),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # Draw text lines
        for i, line in enumerate(label_lines):
            text_y = y0 + 6 + line_height * (i + 1)
            cv2.putText(canvas, line, (x0 + 5, text_y), font, font_scale,
                       text_color, font_thickness, cv2.LINE_AA)

    # Grid interior starts after contour column + contour_margin
    grid_x_start = thumb_w + contour_margin
    grid_y_start = thumb_h + contour_margin

    # Place grid images in interior cells (columns 1 to n_cols-2, rows 1 to n_rows-2)
    for p in grid_points:
        img_path = images_dir / p["image"]
        if not img_path.exists():
            continue

        img_full = cv2.imread(str(img_path))
        if img_full is None:
            continue

        # Compute metrics from full-size image before resize
        p["_s"] = sharpness_tenengrad(img_full)
        p["_l"] = sharpness_laplacian(img_full)
        p["_v"] = sharpness_normalized_variance(img_full)

        img = cv2.resize(img_full, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Map grid position to interior cell (0-indexed within grid)
        nx = (p["x_um"] - grid_x_min) / grid_x_span if grid_x_span > 0 else 0.5
        ny = (p["y_um"] - grid_y_min) / grid_y_span if grid_y_span > 0 else 0.5

        # Grid cell index (0 to n_cols_grid-1)
        grid_col = int(nx * (n_cols_grid - 1) + 0.5)
        grid_row = int(ny * (n_rows_grid - 1) + 0.5)

        x0 = grid_x_start + grid_col * (thumb_w + grid_margin)
        y0 = grid_y_start + grid_row * (thumb_h + grid_margin)

        canvas[y0:y0+thumb_h, x0:x0+thumb_w] = img
        draw_label(canvas, x0, y0, p, z_mean, img)

    # Place contour images on outer edge at continuous positions
    # Grid region pixel bounds
    interior_x_start = grid_x_start
    interior_x_end = grid_x_start + (n_cols_grid - 1) * (thumb_w + grid_margin) + thumb_w
    interior_y_start = grid_y_start
    interior_y_end = grid_y_start + (n_rows_grid - 1) * (thumb_h + grid_margin) + thumb_h

    for p in contour_points:
        img_path = images_dir / p["image"]
        if not img_path.exists():
            continue

        img_full = cv2.imread(str(img_path))
        if img_full is None:
            continue

        # Compute metrics from full-size image before resize
        p["_s"] = sharpness_tenengrad(img_full)
        p["_l"] = sharpness_laplacian(img_full)
        p["_v"] = sharpness_normalized_variance(img_full)

        img = cv2.resize(img_full, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Determine which edge based on position relative to grid
        px, py = p["x_um"], p["y_um"]

        # Normalized position relative to grid (can be outside 0-1)
        nx = (px - grid_x_min) / grid_x_span if grid_x_span > 0 else 0.5
        ny = (py - grid_y_min) / grid_y_span if grid_y_span > 0 else 0.5

        # Determine edge: which side is furthest outside grid bounds?
        outside_left = grid_x_min - px if px < grid_x_min else 0
        outside_right = px - grid_x_max if px > grid_x_max else 0
        outside_top = grid_y_min - py if py < grid_y_min else 0
        outside_bottom = py - grid_y_max if py > grid_y_max else 0

        max_outside = max(outside_left, outside_right, outside_top, outside_bottom)

        if max_outside == 0:
            # Inside grid bounds - place based on nearest edge
            dist_left = px - grid_x_min
            dist_right = grid_x_max - px
            dist_top = py - grid_y_min
            dist_bottom = grid_y_max - py
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            if min_dist == dist_left:
                edge = 'left'
            elif min_dist == dist_right:
                edge = 'right'
            elif min_dist == dist_top:
                edge = 'top'
            else:
                edge = 'bottom'
        elif max_outside == outside_left:
            edge = 'left'
        elif max_outside == outside_right:
            edge = 'right'
        elif max_outside == outside_top:
            edge = 'top'
        else:
            edge = 'bottom'

        # Calculate continuous position along the edge
        # Clamp normalized coords for positioning along edge
        nx_clamped = max(0, min(1, nx))
        ny_clamped = max(0, min(1, ny))

        if edge == 'left':
            x0 = 0  # Left column
            # Map ny to interior Y range
            y0 = int(interior_y_start + ny_clamped * (interior_y_end - interior_y_start - thumb_h))
        elif edge == 'right':
            x0 = canvas_w - thumb_w  # Right column
            y0 = int(interior_y_start + ny_clamped * (interior_y_end - interior_y_start - thumb_h))
        elif edge == 'top':
            # Map nx to interior X range
            x0 = int(interior_x_start + nx_clamped * (interior_x_end - interior_x_start - thumb_w))
            y0 = 0  # Top row
        else:  # bottom
            x0 = int(interior_x_start + nx_clamped * (interior_x_end - interior_x_start - thumb_w))
            y0 = canvas_h - thumb_h  # Bottom row

        # Clamp to canvas bounds
        x0 = max(0, min(canvas_w - thumb_w, x0))
        y0 = max(0, min(canvas_h - thumb_h, y0))

        canvas[y0:y0+thumb_h, x0:x0+thumb_w] = img
        draw_label(canvas, x0, y0, p, z_mean, img)

    # Draw legend in bottom right
    legend_lines = [
        "Legend:",
        "S: Tenengrad (Sobel)",
        "L: Laplacian var",
        "V: Norm variance",
        "Z: offset from mean",
        "Red = low S or drift",
    ]
    legend_font_scale = font_scale * 0.9
    legend_line_height = int(legend_font_scale * 25)

    max_legend_width = 0
    for line in legend_lines:
        (text_w, _), _ = cv2.getTextSize(line, font, legend_font_scale, font_thickness)
        max_legend_width = max(max_legend_width, text_w)

    legend_width = max_legend_width + 16
    legend_height = len(legend_lines) * legend_line_height + 12
    legend_x = canvas_w - legend_width - 10
    legend_y = canvas_h - legend_height - 10

    # Draw legend background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (legend_x, legend_y),
                  (legend_x + legend_width, legend_y + legend_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Draw legend text
    for i, line in enumerate(legend_lines):
        text_y = legend_y + 8 + legend_line_height * (i + 1)
        color = (200, 200, 200) if i == 0 else (255, 255, 255)
        cv2.putText(canvas, line, (legend_x + 8, text_y), font, legend_font_scale,
                    color, font_thickness, cv2.LINE_AA)

    # Draw notes in bottom left if present
    notes = data.get("notes")
    if notes:
        notes_font_scale = font_scale * 0.9
        notes_line_height = int(notes_font_scale * 25)
        # Support both actual newlines and literal \n in notes string
        notes_lines = notes.replace("\\n", "\n").split("\n")

        max_notes_width = 0
        for line in notes_lines:
            (text_w, _), _ = cv2.getTextSize(line, font, notes_font_scale, font_thickness)
            max_notes_width = max(max_notes_width, text_w)

        notes_width = max_notes_width + 16
        notes_height = len(notes_lines) * notes_line_height + 12
        notes_x = 10
        notes_y = canvas_h - notes_height - 10

        # Draw notes background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (notes_x, notes_y),
                      (notes_x + notes_width, notes_y + notes_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Draw notes text
        for i, line in enumerate(notes_lines):
            text_y_pos = notes_y + 8 + notes_line_height * (i + 1)
            cv2.putText(canvas, line, (notes_x + 8, text_y_pos), font, notes_font_scale,
                        (255, 255, 200), font_thickness, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Mosaic saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze focus map data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "focus_map",
        type=Path,
        help="Path to focus_map_chip*.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output plot path (default: focus_map_analysis.png in same dir)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plot",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=3000,
        help="Maximum canvas dimension for mosaic in pixels",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=5,
        help="Gap between images in mosaic (pixels)",
    )
    parser.add_argument(
        "--no-mosaic",
        action="store_true",
        help="Skip generating mosaic image",
    )
    args = parser.parse_args()

    if not args.focus_map.exists():
        print(f"Error: File not found: {args.focus_map}")
        return 1

    # Load data
    data = load_focus_map(args.focus_map)

    # Analyze
    analysis = analyze_focus_map(data)

    # Print report
    print_report(data, analysis)

    # Generate plot
    if not args.no_plot:
        output_path = args.output or args.focus_map.with_name(
            args.focus_map.stem + "_analysis.png"
        )
        plot_focus_map(data, analysis, output_path)

    # Generate mosaic
    if not args.no_mosaic:
        # Find images directory
        images_dir = args.focus_map.with_name(args.focus_map.stem + "_images")
        if images_dir.exists():
            mosaic_path = args.focus_map.with_name(args.focus_map.stem + "_mosaic.jpg")
            create_mosaic(
                data,
                images_dir,
                mosaic_path,
                max_dim=args.max_dim,
                margin_px=args.margin,
            )
        else:
            print(f"Images directory not found: {images_dir}")
            print("  (Run focus_map.py with --save-images to generate)")

    return 0


if __name__ == "__main__":
    exit(main())
