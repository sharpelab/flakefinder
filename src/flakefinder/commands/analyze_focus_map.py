"""Analyze focus map data and produce visualizations.

Loads focus_map_chip*.json, identifies out-of-focus points, fits a tilt plane,
and generates surface plots.

Usage:
    uv run python commands/analyze_focus_map.py scans/focus_map_chip0.json
    uv run python commands/analyze_focus_map.py scans/focus_map_chip0.json --export-plane scans/plane.json
    uv run python commands/analyze_focus_map.py scans/focus_map_chip0.json --min-sharpness 20 -q
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


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


def get_sharpness(point: dict) -> float | None:
    """Get selected sharpness from a focus map point (nested to_dict format)."""
    return point.get("selected", {}).get("sharpness")


def plot_af_curves(
    ax,
    *,
    coarse_curve: list[dict],
    fine_curve: list[dict],
    super_fine_curve: list[dict],
    initial_z_um: float,
    selected_z_um: float,
    selected_sharpness: float,
    initial_label: str = "Initial Z",
    selected_label: str = "Selected Z",
    markersize: float = 4,
    star_markersize: float = 16,
) -> None:
    """Plot autofocus sharpness curves with markers on the given axes.

    Shared between standalone analyze_autofocus plots and focus map curve mosaics.
    Draws coarse/fine/super_fine curves, shaded scan ranges, initial Z vline,
    and selected Z star marker.
    """
    sf_ms = max(1, markersize - 1)  # super_fine slightly smaller

    if coarse_curve:
        coarse_z = [r["z_um"] for r in coarse_curve]
        coarse_s = [r["sharpness"] for r in coarse_curve]
        ax.plot(
            coarse_z,
            coarse_s,
            "o-",
            color="#4488cc",
            markersize=markersize,
            linewidth=1.2,
            label="Coarse",
            zorder=3,
        )

    if fine_curve:
        fine_z = [r["z_um"] for r in fine_curve]
        fine_s = [r["sharpness"] for r in fine_curve]
        ax.plot(
            fine_z,
            fine_s,
            "o-",
            color="#ee8833",
            markersize=markersize,
            linewidth=1.2,
            label="Fine",
            zorder=3,
        )

    if super_fine_curve:
        sf_z = [r["z_um"] for r in super_fine_curve]
        sf_s = [r["sharpness"] for r in super_fine_curve]
        ax.plot(sf_z, sf_s, "o-", color="#cc4488", markersize=sf_ms, linewidth=1.2, label="Super fine", zorder=3)

    # Shaded scan ranges
    if coarse_curve:
        ax.axvspan(min(coarse_z), max(coarse_z), alpha=0.08, color="#4488cc", zorder=0)
    if fine_curve:
        ax.axvspan(min(fine_z), max(fine_z), alpha=0.12, color="#ee8833", zorder=1)
    if super_fine_curve:
        ax.axvspan(min(sf_z), max(sf_z), alpha=0.15, color="#cc4488", zorder=1)

    # Initial Z marker
    ax.axvline(initial_z_um, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label=initial_label, zorder=2)

    # Selected Z marker
    ax.plot(
        selected_z_um,
        selected_sharpness,
        "*",
        color="gold",
        markersize=star_markersize,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=selected_label,
        zorder=5,
    )

    ax.grid(True, alpha=0.3)


def analyze_focus_map(data: dict) -> dict:
    """Analyze focus map data.

    Returns dict with analysis results.
    """
    points = data["sample_points"]

    # Extract arrays
    x = np.array([p["x_um"] for p in points])
    y = np.array([p["y_um"] for p in points])
    z = np.array([p["selected"]["z_um"] for p in points if "selected" in p])

    sharpness = np.array([p["selected"]["sharpness"] for p in points if "selected" in p])
    types = [p["type"] for p in points]
    indices = [p["index"] for p in points]

    # Filter to successful points
    valid_mask = np.array(["selected" in p for p in points])
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Check for sharpness drift (best vs final differ significantly)
    sharpness_drift_indices = []
    sharpness_drift_threshold = 0.2  # 20% difference
    for i, p in enumerate(points):
        if not valid_mask[i]:
            continue
        best = get_sharpness(p)
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
        "indices": [indices[i] for i in range(len(indices)) if valid_mask[i]],
        "residuals": residuals,
    }


def plot_focus_map(data: dict, analysis: dict, output_path: Path, quiet: bool = False) -> None:
    """Generate focus map visualization."""
    fig = plt.figure(figsize=(18, 12))

    x = analysis["x"] / 1000  # Convert to mm
    y = analysis["y"] / 1000
    z = analysis["z"]
    sharpness = analysis["sharpness"]
    residuals = analysis["residuals"]
    coeffs = analysis["plane_coeffs"]

    # Build point labels like "c01", "g11"
    labels = [
        f"{'c' if t == 'contour' else 'g'}{idx:02d}"
        for t, idx in zip(analysis["types"], analysis["indices"], strict=True)
    ]

    def annotate_points(ax, x, y, labels, fontsize=6):
        for xi, yi, label in zip(x, y, labels, strict=True):
            ax.annotate(
                label,
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=fontsize,
                alpha=0.8,
            )

    # 1. 3D interpolated surface from sample points
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")

    # Create interpolated surface
    x_grid = np.linspace(x.min(), x.max(), 50)
    y_grid = np.linspace(y.min(), y.max(), 50)
    X_surf, Y_surf = np.meshgrid(x_grid, y_grid)

    # Interpolate Z values onto grid
    Z_surf = griddata((x, y), z, (X_surf, Y_surf), method="cubic")

    # Plot interpolated surface
    surf = ax1.plot_surface(X_surf, Y_surf, Z_surf, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True)

    # Overlay sample points
    colors = ["red" if t == "contour" else "white" for t in analysis["types"]]
    ax1.scatter(x, y, z, c=colors, s=30, edgecolors="black", linewidths=0.5, zorder=5)

    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (µm)")
    ax1.set_title("Interpolated Focus Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5, label="Z (µm)")

    # 2. 3D points with fitted plane
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")

    # Color by type
    colors = ["blue" if t == "contour" else "green" for t in analysis["types"]]
    ax2.scatter(x, y, z, c=colors, s=50, alpha=0.8)

    # Plot fitted plane
    Z_plane = coeffs[0] * X_surf * 1000 + coeffs[1] * Y_surf * 1000 + coeffs[2]
    ax2.plot_surface(X_surf, Y_surf, Z_plane, alpha=0.3, color="red")

    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (µm)")
    ax2.set_title(
        f"Fitted Plane (R²={analysis['plane_r_squared']:.3f})\n"
        f"Tilt: {analysis['tilt_magnitude_um_per_mm']:.1f} µm/mm @ {analysis['tilt_direction_deg']:.0f}°"
    )

    # 3. Interpolated surface - top-down heatmap view
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.pcolormesh(X_surf, Y_surf, Z_surf, cmap="viridis", shading="auto")
    ax3.scatter(x, y, c="white", s=30, edgecolors="black", linewidths=0.5)
    plt.colorbar(im, ax=ax3, label="Z (µm)")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.set_title("Interpolated Surface (top view)")
    ax3.set_aspect("equal")
    ax3.invert_yaxis()
    annotate_points(ax3, x, y, labels)

    # 4. Sharpness heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    scatter = ax4.scatter(x, y, c=sharpness, cmap="plasma", s=100, edgecolors="black")
    plt.colorbar(scatter, ax=ax4, label="Sharpness")
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    ax4.set_title(f"Sharpness Map (mean={analysis['sharpness_mean']:.1f}, std={analysis['sharpness_std']:.1f})")
    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    annotate_points(ax4, x, y, labels)

    # Mark outliers
    if len(analysis["outlier_indices"]) > 0:
        out_x = x[analysis["outlier_indices"]]
        out_y = y[analysis["outlier_indices"]]
        ax4.scatter(
            out_x,
            out_y,
            s=200,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2,
            label="Z outliers",
        )
        ax4.legend()

    # 5. Residuals from plane fit
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(
        x,
        y,
        c=residuals,
        cmap="coolwarm",
        s=100,
        edgecolors="black",
        vmin=-3 * analysis["residual_std"],
        vmax=3 * analysis["residual_std"],
    )
    plt.colorbar(scatter, ax=ax5, label="Residual (µm)")
    ax5.set_xlabel("X (mm)")
    ax5.set_ylabel("Y (mm)")
    ax5.set_title(f"Residuals from Plane Fit (std={analysis['residual_std']:.1f} µm)")
    ax5.set_aspect("equal")
    ax5.invert_yaxis()
    annotate_points(ax5, x, y, labels)

    # 6. Z heatmap (top-down view)
    ax6 = fig.add_subplot(2, 3, 6)
    scatter = ax6.scatter(x, y, c=z, cmap="viridis", s=100, edgecolors="black")
    plt.colorbar(scatter, ax=ax6, label="Z (µm)")
    ax6.set_xlabel("X (mm)")
    ax6.set_ylabel("Y (mm)")
    ax6.set_title("Z Position Map")
    ax6.set_aspect("equal")
    ax6.invert_yaxis()
    annotate_points(ax6, x, y, labels)

    # Mark low sharpness points
    if len(analysis["low_sharpness_indices"]) > 0:
        low_x = x[analysis["low_sharpness_indices"]]
        low_y = y[analysis["low_sharpness_indices"]]
        ax6.scatter(
            low_x,
            low_y,
            s=200,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            label="Low sharpness",
        )
        ax6.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    if not quiet:
        print(f"Plot saved to {output_path}")


def print_report(data: dict, analysis: dict) -> None:
    """Print analysis report."""
    print("\n" + "=" * 60)
    print("FOCUS MAP ANALYSIS")
    print("=" * 60)

    print(f"\nSource: {data.get('source_chips_meta', 'unknown')}")
    print(f"Chip ID: {data.get('chip_id', 'unknown')}")
    print(f"Duration: {data.get('duration_s', 0):.1f}s")

    print("\n--- Points ---")
    print(f"Total points: {analysis['n_points']}")
    print(f"Valid points: {analysis['n_valid']}")

    print("\n--- Z Statistics ---")
    print(f"Z range: {analysis['z_min']:.1f} - {analysis['z_max']:.1f} µm ({analysis['z_range']:.1f} µm total)")
    print(f"Z mean: {analysis['z_mean']:.1f} µm")
    print(f"Z std: {analysis['z_std']:.1f} µm")

    print("\n--- Tilt Analysis ---")
    print(f"Plane fit R²: {analysis['plane_r_squared']:.3f}")
    print(f"Tilt X: {analysis['tilt_x_um_per_mm']:.2f} µm/mm")
    print(f"Tilt Y: {analysis['tilt_y_um_per_mm']:.2f} µm/mm")
    print(f"Tilt magnitude: {analysis['tilt_magnitude_um_per_mm']:.2f} µm/mm")
    print(f"Tilt direction: {analysis['tilt_direction_deg']:.1f}°")
    print(f"Residual std: {analysis['residual_std']:.1f} µm (surface roughness/error)")

    print("\n--- Sharpness Statistics ---")
    print(f"Sharpness range: {analysis['sharpness_min']:.1f} - {analysis['sharpness_max']:.1f}")
    print(f"Sharpness mean: {analysis['sharpness_mean']:.1f}")
    print(f"Sharpness std: {analysis['sharpness_std']:.1f}")

    print("\n--- Potential Issues ---")

    low_sharp = analysis["low_sharpness_indices"]
    if len(low_sharp) > 0:
        print(f"Low sharpness points ({len(low_sharp)}, < {analysis['sharpness_threshold']:.1f}):")
        points = data["sample_points"]
        for idx in low_sharp:
            p = points[idx]
            sharpness = get_sharpness(p) or 0
            print(
                f"  {p['type']} {p['index']}: sharpness={sharpness:.1f}, "
                f"pos=({p['x_um'] / 1000:.2f}, {p['y_um'] / 1000:.2f}) mm"
            )
    else:
        print("No low sharpness points detected.")

    outliers = analysis["outlier_indices"]
    if len(outliers) > 0:
        print(f"\nZ outliers ({len(outliers)}, > 2σ from plane):")
        points = data["sample_points"]
        valid_points = [p for p in points if "selected" in p]
        for idx in outliers:
            p = valid_points[idx]
            res = analysis["residuals"][idx]
            print(f"  {p['type']} {p['index']}: Z={p['selected']['z_um']:.1f} µm, residual={res:+.1f} µm")
    else:
        print("No Z outliers detected.")

    drift_indices = analysis["sharpness_drift_indices"]
    if len(drift_indices) > 0:
        print(f"\nSharpness drift detected ({len(drift_indices)} points, best vs final >20% diff):")
        points = data["sample_points"]
        for idx in drift_indices:
            p = points[idx]
            sel = get_sharpness(p) or 0
            final = p.get("final_sharpness", 0)
            drift_pct = abs(sel - final) / sel * 100 if sel > 0 else 0
            print(f"  {p['type']} {p['index']}: selected={sel:.1f}, final={final:.1f} ({drift_pct:+.0f}%)")
    else:
        print("No sharpness drift detected.")

    # DOF warning
    dof_20x = 4  # µm, approximate for 20x
    if analysis["z_range"] > 5 * dof_20x:
        print(
            f"\n⚠️  WARNING: Z range ({analysis['z_range']:.0f} µm) is ~{analysis['z_range'] / dof_20x:.0f}x "
            f"the DOF of a 20x objective (~{dof_20x} µm)"
        )
        print("   Consider leveling the sample or using focus tracking during scans.")

    print("\n" + "=" * 60)


def create_mosaic(
    data: dict,
    images_dir: Path,
    output_path: Path,
    max_dim: int = 3000,
    margin_px: int = 5,
    quiet: bool = False,
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
        if not quiet:
            print("No images found in focus map data")
        return

    # Compute mean Z for relative offsets
    z_values = [p["selected"]["z_um"] for p in all_points if "selected" in p]
    z_mean = np.mean(z_values) if z_values else 0

    # Load first image to get aspect ratio
    first_img_path = images_dir / all_points[0]["image"]
    if not first_img_path.exists():
        if not quiet:
            print(f"Image not found: {first_img_path}")
        return

    first_img = cv2.imread(first_img_path)
    assert first_img is not None, f"Failed to read image: {first_img_path}"
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

    if not quiet:
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

    if not quiet:
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
        z = point.get("selected", {}).get("z_um")
        if z is not None:
            z_offset = z - z_mean
            z_str = f"Z:{z_offset:+.0f}"
        else:
            z_str = "Z:--"

        # Text color: red for low sharpness, white otherwise
        low_sharpness = s0 is not None and s0 < 30
        # Also flag if significant drift between best and final
        sel = get_sharpness(point)
        final = point.get("final_sharpness")
        has_drift = (  # noqa: F841 — TODO: use to color drift points
            sel is not None and final is not None and sel > 0 and abs(sel - final) / sel > 0.2
        )
        text_color = (0, 0, 255) if low_sharpness else (255, 255, 255)  # BGR: red if low sharpness

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
        cv2.rectangle(overlay, (x0 + 2, y0 + 2), (x0 + bg_width + 2, y0 + bg_height + 2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # Draw text lines
        for i, line in enumerate(label_lines):
            text_y = y0 + 6 + line_height * (i + 1)
            cv2.putText(
                canvas,
                line,
                (x0 + 5, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

    # Grid interior starts after contour column + contour_margin
    grid_x_start = thumb_w + contour_margin
    grid_y_start = thumb_h + contour_margin

    # Place grid images in interior cells (columns 1 to n_cols-2, rows 1 to n_rows-2)
    for p in grid_points:
        img_path = images_dir / p["image"]
        if not img_path.exists():
            continue

        img_full = cv2.imread(img_path)
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

        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = img
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

        img_full = cv2.imread(img_path)
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
                edge = "left"
            elif min_dist == dist_right:
                edge = "right"
            elif min_dist == dist_top:
                edge = "top"
            else:
                edge = "bottom"
        elif max_outside == outside_left:
            edge = "left"
        elif max_outside == outside_right:
            edge = "right"
        elif max_outside == outside_top:
            edge = "top"
        else:
            edge = "bottom"

        # Calculate continuous position along the edge
        # Clamp normalized coords for positioning along edge
        nx_clamped = max(0, min(1, nx))
        ny_clamped = max(0, min(1, ny))

        if edge == "left":
            x0 = 0  # Left column
            # Map ny to interior Y range
            y0 = int(interior_y_start + ny_clamped * (interior_y_end - interior_y_start - thumb_h))
        elif edge == "right":
            x0 = canvas_w - thumb_w  # Right column
            y0 = int(interior_y_start + ny_clamped * (interior_y_end - interior_y_start - thumb_h))
        elif edge == "top":
            # Map nx to interior X range
            x0 = int(interior_x_start + nx_clamped * (interior_x_end - interior_x_start - thumb_w))
            y0 = 0  # Top row
        else:  # bottom
            x0 = int(interior_x_start + nx_clamped * (interior_x_end - interior_x_start - thumb_w))
            y0 = canvas_h - thumb_h  # Bottom row

        # Clamp to canvas bounds
        x0 = max(0, min(canvas_w - thumb_w, x0))
        y0 = max(0, min(canvas_h - thumb_h, y0))

        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = img
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
    cv2.rectangle(
        overlay,
        (legend_x, legend_y),
        (legend_x + legend_width, legend_y + legend_height),
        (40, 40, 40),
        -1,
    )
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Draw legend text
    for i, line in enumerate(legend_lines):
        text_y = legend_y + 8 + legend_line_height * (i + 1)
        color = (200, 200, 200) if i == 0 else (255, 255, 255)
        cv2.putText(
            canvas,
            line,
            (legend_x + 8, text_y),
            font,
            legend_font_scale,
            color,
            font_thickness,
            cv2.LINE_AA,
        )

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
        cv2.rectangle(
            overlay,
            (notes_x, notes_y),
            (notes_x + notes_width, notes_y + notes_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Draw notes text
        for i, line in enumerate(notes_lines):
            text_y_pos = notes_y + 8 + notes_line_height * (i + 1)
            cv2.putText(
                canvas,
                line,
                (notes_x + 8, text_y_pos),
                font,
                notes_font_scale,
                (255, 255, 200),
                font_thickness,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not quiet:
        print(f"Mosaic saved to {output_path}")


def plot_curves_mosaic(
    data: dict,
    output_path: Path,
    *,
    min_sharpness: float = 20.0,
    quiet: bool = False,
) -> None:
    """Generate mosaic of per-point sharpness curves from focus map data.

    Each subplot shows coarse/fine/super_fine sharpness curves for one sample point,
    with initial Z and selected Z marked. Dropped points are labeled with reason.
    """
    points = data["sample_points"]
    if not points:
        if not quiet:
            print("No sample points found")
        return

    # Get drop reasons from robust plane fit
    try:
        result = compute_robust_plane_fit(data, min_sharpness=min_sharpness)
        dropped_map: dict[tuple[str, int], str] = {
            (d["type"], d["index"]): d["reason"] for d in result["points_dropped"]
        }
    except ValueError:
        dropped_map = {}

    n = len(points)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), squeeze=False)

    for i, point in enumerate(points):
        ax = axes[i // cols, i % cols]

        prefix = "c" if point["type"] == "contour" else "g"
        label = f"{prefix}{point['index']:02d}"

        # Build subtitle
        parts: list[str] = []
        drop_key = (point["type"], point["index"])
        if drop_key in dropped_map:
            parts.append(f"DROPPED {dropped_map[drop_key]}")

        dr = point.get("dynamic_range")
        if dr is not None:
            parts.append(f"DR={dr:.0%}")

        sel = point.get("selected", {})
        s = sel.get("sharpness")
        if s is not None:
            parts.append(f"S={s:.1f}")

        subtitle = "  ".join(parts)

        # Plot curves if point has data
        has_selected = "selected" in point
        has_curves = bool(point.get("sharpness_curve"))

        if has_selected and has_curves:
            initial = point.get("initial", {})
            plot_af_curves(
                ax,
                coarse_curve=point.get("sharpness_curve", []),
                fine_curve=point.get("fine_sharpness_curve", []),
                super_fine_curve=point.get("super_fine_sharpness_curve", []),
                initial_z_um=initial.get("z_um", sel["z_um"]),
                selected_z_um=sel["z_um"],
                selected_sharpness=sel["sharpness"],
                markersize=2,
                star_markersize=10,
            )
            # Add y-padding so curves don't crowd the title/bottom
            y_lo, y_hi = ax.get_ylim()
            y_pad = (y_hi - y_lo) * 0.1
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # Title color: red for dropped/error points
        title_color = "red" if drop_key in dropped_map or not has_selected else "black"
        if not has_selected:
            subtitle = "ERROR (no selected Z)"

        ax.set_title(f"{label}\n{subtitle}", fontsize=7, color=title_color)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    # Shared legend from first subplot with data
    for i in range(min(n, rows * cols)):
        handles, labels = axes[i // cols, i % cols].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=7)
            break

    plt.tight_layout()
    plt.savefig(output_path, dpi=210)
    plt.close()
    if not quiet:
        print(f"Curves mosaic saved to {output_path}")


def compute_robust_plane_fit(
    data: dict,
    corner_margin_um: float = 5000.0,
    min_sharpness: float = 0.0,
    reject_sharpness: float = 10.0,
    min_dynamic_range: float = 0.05,
    edge_peak_max_dr: float = 0.10,
    mad_sigma_threshold: float = 3.0,
    max_reject_frac: float = 0.3,
    min_points_after_reject: int = 4,
    max_tilt_um_per_mm: float = 30.0,
) -> dict:
    """Compute robust plane fit with outlier rejection and coverage analysis.

    Two-stage filtering:
      Stage 1 (pre-filter): Hard-reject points with sharpness below noise floor
        (< reject_sharpness), flat AF curves (DR < min_dynamic_range), black frames,
        zero refinement delta (fine/super_fine didn't improve on coarse), or
        edge peaks with low DR (peak_near_edge AND DR < edge_peak_max_dr).
      Stage 2 (post-fit): MAD-based residual rejection. Fit initial plane,
        compute MAD of residuals, reject points > mad_sigma_threshold * robust_sigma.

    After fitting, computes whole-FM confidence assessment from aggregate
    quality signals (peak_near_edge fraction, median DR). Reports low_confidence
    when the pattern matches a false-success focus map.

    Args:
        data: Focus map data dict.
        corner_margin_um: Distance from edge to consider "corner" region.
        min_sharpness: Sharpness warning threshold (points below flagged, not rejected).
        reject_sharpness: Hard reject threshold (below = noise floor, Z meaningless).
        min_dynamic_range: Hard reject threshold for DR (below = flat curve).
        edge_peak_max_dr: Reject edge peaks (peak_near_edge=True) with DR below this.
        mad_sigma_threshold: Reject residuals beyond this many robust-sigma (MAD * 1.4826).
        max_reject_frac: Never reject more than this fraction of pre-filtered points.
        min_points_after_reject: Require at least this many points after rejection.
        max_tilt_um_per_mm: Fall back to flat plane if tilt exceeds this (default 30;
            normal chips are 2-6 µm/mm, degenerate fits produce 100+).

    Returns:
        Dict with plane parameters, quality metrics, coverage info, and confidence.
    """
    points = [p for p in data["sample_points"] if "selected" in p]

    if len(points) < 3:
        raise ValueError(f"Need at least 3 valid points, got {len(points)}")

    # Extract arrays
    x = np.array([p["x_um"] for p in points])
    y = np.array([p["y_um"] for p in points])
    z = np.array([p["selected"]["z_um"] for p in points])
    sel_sharpness = np.array([p["selected"]["sharpness"] for p in points])
    final_sharpness = np.array([p["final_sharpness"] for p in points])
    # Compute quality metrics
    drift_pct = np.clip((sel_sharpness - final_sharpness) / sel_sharpness * 100, 0, 100)
    coarse_best_z = np.array([p.get("coarse", {}).get("best_z_um", p["selected"]["z_um"]) for p in points])
    refinement_delta = np.abs(coarse_best_z - z)
    dynamic_range = np.array([p.get("dynamic_range", 0) for p in points])
    peak_near_edge = np.array([p.get("peak_near_edge", False) for p in points])
    mean_intensity = np.array([p.get("mean_intensity", float("inf")) for p in points])

    # --- Stage 1: Pre-filter (hard rejects only) ---
    # Reject points with truly bad measurements:
    # - sharpness below reject threshold (noise floor, Z is meaningless)
    # - dynamic range near zero (flat curve, no focus signal)
    # - black frame (mean intensity near zero — missed chip or light off)
    # - zero refinement delta (fine/super_fine didn't improve on coarse — false peak)
    BLACK_FRAME_THRESHOLD = 5.0
    prefilter_mask = (
        (sel_sharpness >= reject_sharpness)
        & (dynamic_range >= min_dynamic_range)
        & (mean_intensity >= BLACK_FRAME_THRESHOLD)
        & (refinement_delta > 0)
        & ~(peak_near_edge & (dynamic_range < edge_peak_max_dr))
    )

    if prefilter_mask.sum() < 3:
        # Fall back to all points if not enough pass pre-filter
        print(f"  Warning: Only {prefilter_mask.sum()} pre-filtered points, using all points")
        prefilter_mask = np.ones(len(points), dtype=bool)

    # Classify pre-filter drop reasons (order matters: first match wins)
    prefilter_reasons: dict[int, str] = {}
    for i in range(len(points)):
        if not prefilter_mask[i]:
            if mean_intensity[i] < BLACK_FRAME_THRESHOLD:
                prefilter_reasons[i] = "black_frame"
            elif sel_sharpness[i] < reject_sharpness:
                prefilter_reasons[i] = "low_sharpness"
            elif dynamic_range[i] < min_dynamic_range:
                prefilter_reasons[i] = "flat_curve"
            elif refinement_delta[i] == 0:
                prefilter_reasons[i] = "no_refinement"
            elif peak_near_edge[i] and dynamic_range[i] < edge_peak_max_dr:
                prefilter_reasons[i] = "edge_peak_low_dr"
            else:
                prefilter_reasons[i] = "flat_curve"

    # --- Stage 2: MAD-based residual rejection ---
    # Initial plane fit on pre-filtered points
    x_pf = x[prefilter_mask]
    y_pf = y[prefilter_mask]
    z_pf = z[prefilter_mask]

    A_pf = np.column_stack([x_pf, y_pf, np.ones_like(x_pf)])
    coeffs_pf, _, _, _ = np.linalg.lstsq(A_pf, z_pf, rcond=None)

    # Compute residuals for all pre-filtered points
    z_pred_pf = coeffs_pf[0] * x_pf + coeffs_pf[1] * y_pf + coeffs_pf[2]
    residuals_pf = z_pf - z_pred_pf

    # MAD-based outlier detection
    mad = float(np.median(np.abs(residuals_pf)))
    robust_sigma = mad * 1.4826  # Scale to match Gaussian sigma
    mad_threshold_um = mad_sigma_threshold * robust_sigma if robust_sigma > 0 else float("inf")

    # Identify outliers among pre-filtered points
    pf_indices = np.where(prefilter_mask)[0]
    mad_reject_mask = np.abs(residuals_pf) > mad_threshold_um

    # Safety: don't reject too many points
    n_prefiltered = int(prefilter_mask.sum())
    n_would_reject = int(mad_reject_mask.sum())
    n_remaining = n_prefiltered - n_would_reject

    too_few = n_remaining < min_points_after_reject
    too_many = n_would_reject > max_reject_frac * n_prefiltered
    if n_would_reject > 0 and (too_few or too_many):
        print(
            f"  Warning: MAD would reject {n_would_reject}/{n_prefiltered} points "
            f"(remaining {n_remaining} < {min_points_after_reject} or "
            f">{max_reject_frac:.0%}), skipping residual rejection"
        )
        mad_reject_mask = np.zeros_like(mad_reject_mask, dtype=bool)

    # Build final mask: pre-filtered minus MAD-rejected
    high_conf_mask = prefilter_mask.copy()
    mad_reject_reasons: dict[int, float] = {}  # index -> residual
    for j, pf_idx in enumerate(pf_indices):
        if mad_reject_mask[j]:
            high_conf_mask[pf_idx] = False
            mad_reject_reasons[pf_idx] = float(residuals_pf[j])

    # Final plane fit on surviving points
    x_hc = x[high_conf_mask]
    y_hc = y[high_conf_mask]
    z_hc = z[high_conf_mask]

    A = np.column_stack([x_hc, y_hc, np.ones_like(x_hc)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_hc, rcond=None)
    a, b, c = coeffs

    # Tilt sanity check: degenerate fits (collinear points) produce absurd tilt
    tilt_magnitude = float(np.sqrt(a**2 + b**2) * 1000)  # µm/mm
    tilt_clamped = tilt_magnitude > max_tilt_um_per_mm
    if tilt_clamped:
        print(
            f"  Warning: tilt {tilt_magnitude:.1f} µm/mm exceeds {max_tilt_um_per_mm} µm/mm, "
            f"falling back to flat plane at median Z"
        )
        a, b, c = 0.0, 0.0, float(np.median(z_hc))

    # Compute residuals and R²
    z_pred = a * x_hc + b * y_hc + c
    residuals = z_hc - z_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z_hc - np.mean(z_hc)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Corner coverage analysis
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    corners = {
        "BL": (x_min, y_min),  # Bottom-left
        "BR": (x_max, y_min),  # Bottom-right
        "TL": (x_min, y_max),  # Top-left
        "TR": (x_max, y_max),  # Top-right
    }

    corners_covered = []
    corners_extrapolated = []

    for corner_name, (cx, cy) in corners.items():
        # Check if any high-confidence point is near this corner
        near_corner = (np.abs(x_hc - cx) < corner_margin_um) & (np.abs(y_hc - cy) < corner_margin_um)
        if near_corner.any():
            corners_covered.append(corner_name)
        else:
            corners_extrapolated.append(corner_name)

    # --- Whole-FM confidence assessment ---
    # Evaluate aggregate quality of points that survived hard reject.
    # A false-success FM has: majority peak_near_edge + low median DR.
    pf_edge = peak_near_edge[prefilter_mask]
    pf_dr = dynamic_range[prefilter_mask]
    pf_sharpness = sel_sharpness[prefilter_mask]

    n_pf = len(pf_edge)
    edge_frac = float(pf_edge.sum() / n_pf) if n_pf > 0 else 0.0
    median_dr = float(np.median(pf_dr)) if n_pf > 0 else 0.0
    median_sharpness = float(np.median(pf_sharpness)) if n_pf > 0 else 0.0

    confidence_reasons: list[str] = []
    if edge_frac >= 0.5:
        confidence_reasons.append("majority_peak_near_edge")
    if median_dr < 0.20:
        confidence_reasons.append("low_median_dr")

    low_confidence = len(confidence_reasons) >= 2

    # Per-point warnings (informational, not rejected from fit)
    point_warnings = []
    for i in range(len(points)):
        if not prefilter_mask[i]:
            continue  # already rejected
        warnings: list[str] = []
        if peak_near_edge[i]:
            warnings.append("peak_near_edge")
        if sel_sharpness[i] < min_sharpness:
            warnings.append("low_sharpness")
        if warnings:
            point_warnings.append(
                {
                    "type": points[i]["type"],
                    "index": points[i]["index"],
                    "warnings": warnings,
                }
            )

    # Build points_dropped with specific reasons
    points_dropped = []
    for i in range(len(points)):
        if high_conf_mask[i]:
            continue
        entry = {
            "type": points[i]["type"],
            "index": points[i]["index"],
            "x_um": float(x[i]),
            "y_um": float(y[i]),
            "z_um": float(z[i]),
            "selected_sharpness": float(sel_sharpness[i]),
            "dynamic_range": float(dynamic_range[i]),
            "peak_near_edge": bool(peak_near_edge[i]),
            "mean_intensity": float(mean_intensity[i]),
        }
        if i in mad_reject_reasons:
            entry["reason"] = "residual_outlier"
            entry["residual_um"] = mad_reject_reasons[i]
        elif i in prefilter_reasons:
            entry["reason"] = prefilter_reasons[i]
        else:
            entry["reason"] = "unknown"
        points_dropped.append(entry)

    # Build result
    result = {
        "plane": {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "equation": f"Z = {a * 1000:.4f}*X_mm + {b * 1000:.4f}*Y_mm + {c:.2f}",
        },
        "units": "um",
        "quality": {
            "r_squared": float(r_squared),
            "residual_std_um": float(np.std(residuals)),
            "residual_max_um": float(np.max(np.abs(residuals))),
            "points_total": len(points),
            "points_used": int(high_conf_mask.sum()),
            "points_prefiltered": n_prefiltered,
            "residual_outliers_rejected": int(mad_reject_mask.sum()),
            "mad_threshold_um": float(mad_threshold_um),
            "reject_sharpness": reject_sharpness,
            "min_dynamic_range": min_dynamic_range,
            "min_sharpness": min_sharpness,
        },
        "tilt": {
            "x_um_per_mm": float(a * 1000),
            "y_um_per_mm": float(b * 1000),
            "magnitude_um_per_mm": float(np.sqrt(a**2 + b**2) * 1000),
            "clamped": tilt_clamped,
        },
        "coverage": {
            "x_range_um": [float(x_hc.min()), float(x_hc.max())],
            "y_range_um": [float(y_hc.min()), float(y_hc.max())],
            "full_x_range_um": [float(x_min), float(x_max)],
            "full_y_range_um": [float(y_min), float(y_max)],
            "corners_covered": corners_covered,
            "corners_extrapolated": corners_extrapolated,
        },
        "points_used": [
            {
                "type": points[i]["type"],
                "index": points[i]["index"],
                "x_um": float(x[i]),
                "y_um": float(y[i]),
                "z_um": float(z[i]),
                "refinement_delta_um": float(refinement_delta[i]),
                "drift_pct": float(drift_pct[i]),
                "dynamic_range": float(dynamic_range[i]),
                "peak_near_edge": bool(peak_near_edge[i]),
                "mean_intensity": float(mean_intensity[i]),
            }
            for i in range(len(points))
            if high_conf_mask[i]
        ],
        "points_dropped": points_dropped,
        "confidence": {
            "low_confidence": low_confidence,
            "reasons": confidence_reasons,
            "edge_frac": edge_frac,
            "median_dr": median_dr,
            "median_sharpness": median_sharpness,
        },
        "point_warnings": point_warnings,
    }

    # Compute interpolated Z grid from good points
    n_grid = 50
    grid_x = np.linspace(x_min, x_max, n_grid)
    grid_y = np.linspace(y_min, y_max, n_grid)
    G_x, G_y = np.meshgrid(grid_x, grid_y)

    # Use cubic interpolation from good points, fall back to plane for extrapolation
    Z_interp = griddata((x_hc, y_hc), z_hc, (G_x, G_y), method="cubic")
    # Fill NaN (extrapolated regions) with plane values
    nan_mask = np.isnan(Z_interp)
    Z_interp[nan_mask] = a * G_x[nan_mask] + b * G_y[nan_mask] + c

    result["interpolated_grid"] = {
        "x_um": grid_x.tolist(),
        "y_um": grid_y.tolist(),
        "z_um": Z_interp.tolist(),
    }

    # Store arrays for plotting (not serialized)
    result["_plot_data"] = {
        "all_x": x,
        "all_y": y,
        "all_z": z,
        "all_sharpness": sel_sharpness,
        "high_conf_mask": high_conf_mask,
        "G_x": G_x,
        "G_y": G_y,
        "Z_interp": Z_interp,
    }

    return result


def plot_contour_map(result: dict, output_path: Path, quiet: bool = False) -> None:
    """Generate contour map showing interpolated Z surface from filtered points."""
    pd = result["_plot_data"]
    G_x = pd["G_x"] / 1000  # mm
    G_y = pd["G_y"] / 1000
    Z_interp = pd["Z_interp"]
    all_x = pd["all_x"] / 1000
    all_y = pd["all_y"] / 1000
    mask = pd["high_conf_mask"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Filled contour
    cf = ax.contourf(G_x, G_y, Z_interp, levels=20, cmap="viridis")
    # Contour lines
    cl = ax.contour(G_x, G_y, Z_interp, levels=10, colors="white", linewidths=0.5, alpha=0.5)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.0f")

    plt.colorbar(cf, ax=ax, label="Z (µm)")

    # Good points (white)
    ax.scatter(
        all_x[mask],
        all_y[mask],
        c="white",
        s=60,
        edgecolors="black",
        linewidths=1,
        zorder=5,
        label=f"Used ({mask.sum()})",
    )
    # Dropped points (red X)
    ax.scatter(
        all_x[~mask],
        all_y[~mask],
        c="red",
        s=60,
        marker="x",
        linewidths=2,
        zorder=5,
        label=f"Dropped ({(~mask).sum()})",
    )

    # Label all points
    points_used = result["points_used"]
    points_dropped = result["points_dropped"]
    all_labeled = points_used + points_dropped
    for p in all_labeled:
        label = f"{'c' if p['type'] == 'contour' else 'g'}{p['index']:02d}"
        ax.annotate(
            label,
            (p["x_um"] / 1000, p["y_um"] / 1000),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.8,
        )

    q = result["quality"]
    t = result["tilt"]
    ax.set_title(
        f"Interpolated Focus Surface (R²={q['r_squared']:.3f}, "
        f"{q['points_used']}/{q['points_total']} pts, "
        f"tilt={t['magnitude_um_per_mm']:.1f} µm/mm)\n"
        f"Residual std={q['residual_std_um']:.1f} µm, "
        f"min_sharpness={q['min_sharpness']:.0f}"
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    if not quiet:
        print(f"Contour map saved to {output_path}")


def export_plane(data: dict, output_path: Path, min_sharpness: float = 0.0) -> dict:
    """Export robust plane fit to JSON file.

    Args:
        data: Focus map data dict.
        output_path: Output JSON path.
        min_sharpness: Minimum selected_sharpness to include a point.

    Returns:
        The plane fit result dict.
    """
    result = compute_robust_plane_fit(data, min_sharpness=min_sharpness)

    # Add metadata
    result["source"] = {
        "focus_map": data.get("source_chips_meta", "unknown"),
        "chip_id": data.get("chip_id", 0),
        "timestamp": data.get("timestamp", "unknown"),
        "objective_mag": data.get("optics", {}).get("objective_mag"),
    }

    # Strip non-serializable plot data before saving
    serializable = {k: v for k, v in result.items() if k != "_plot_data"}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return result


def run(
    focus_map_path: Path,
    *,
    output: Path | None = None,
    plot: bool = False,
    max_dim: int = 4500,
    margin: int = 5,
    mosaic: bool = False,
    curves: bool = False,
    export_plane_path: Path | None = None,
    min_sharpness: float = 20.0,
    quiet: bool = False,
) -> None:
    """Analyze focus map data, generate plots/mosaic, optionally export plane fit.

    Args:
        focus_map_path: Path to focus_map_chip*.json.
        output: Output plot path (default: auto from focus_map_path).
        plot: Generate analysis plot.
        max_dim: Maximum canvas dimension for mosaic.
        margin: Gap between images in mosaic (pixels).
        mosaic: Generate mosaic image.
        curves: Generate per-point sharpness curve mosaic.
        export_plane_path: Export robust plane fit to this JSON path.
        min_sharpness: Minimum sharpness to include in plane fit.
        quiet: Suppress verbose output.

    Raises:
        FileNotFoundError: If focus_map_path doesn't exist.
    """
    if not focus_map_path.exists():
        raise FileNotFoundError(f"File not found: {focus_map_path}")

    # Load data
    data = load_focus_map(focus_map_path)

    # Analyze
    analysis = analyze_focus_map(data)

    # Print report
    if not quiet:
        print_report(data, analysis)

    # Generate plot
    if plot:
        output_path = output or focus_map_path.with_name(focus_map_path.stem + "_analysis.png")
        plot_focus_map(data, analysis, output_path, quiet=quiet)

    # Generate mosaic
    if mosaic:
        images_dir = focus_map_path.with_name(focus_map_path.stem + "_images")
        if images_dir.exists():
            mosaic_path = focus_map_path.with_name(focus_map_path.stem + "_mosaic.jpg")
            create_mosaic(data, images_dir, mosaic_path, max_dim=max_dim, margin_px=margin, quiet=quiet)
        elif not quiet:
            print(f"Images directory not found: {images_dir}")
            print("  (Run focus_map.py with --save-images to generate)")

    # Generate curves mosaic
    if curves:
        curves_path = focus_map_path.with_name(focus_map_path.stem + "_curves.png")
        plot_curves_mosaic(data, curves_path, min_sharpness=min_sharpness, quiet=quiet)

    # Export plane fit
    if export_plane_path:
        result = export_plane(data, export_plane_path, min_sharpness=min_sharpness)

        q = result["quality"]
        t = result["tilt"]
        c = result["coverage"]

        # One-line summary (always printed)
        corners_str = (
            ",".join(c["corners_extrapolated"]) + " extrapolated" if c["corners_extrapolated"] else "all covered"
        )
        print(
            f"analyze_focus_map: {q['points_used']}/{q['points_total']} pts, "
            f"R\u00b2={q['r_squared']:.3f}, residual={q['residual_std_um']:.1f}\u00b5m, "
            f"tilt={t['magnitude_um_per_mm']:.2f}\u00b5m/mm, "
            f"corners: {corners_str}, {export_plane_path}"
        )

        # Warn if low confidence (always printed, not gated by quiet)
        conf = result.get("confidence", {})
        if conf.get("low_confidence"):
            reasons = ", ".join(conf["reasons"])
            print(f"  \u26a0\ufe0f  LOW CONFIDENCE: {reasons}")

        if not quiet:
            print()
            print("=" * 60)
            print("ROBUST PLANE FIT EXPORT")
            print("=" * 60)
            rej_s = q["reject_sharpness"]
            rej_dr = q["min_dynamic_range"]
            print(f"Points used: {q['points_used']}/{q['points_total']} (reject S<{rej_s}, DR<{rej_dr})")
            print(f"R²: {q['r_squared']:.4f}")
            print(f"Residual std: {q['residual_std_um']:.2f} um")
            print(f"Residual max: {q['residual_max_um']:.2f} um")
            print()
            print(f"Plane: {result['plane']['equation']}")
            print(f"Tilt: {t['magnitude_um_per_mm']:.2f} um/mm")
            print()
            print(f"Coverage X: {c['x_range_um'][0] / 1000:.1f} - {c['x_range_um'][1] / 1000:.1f} mm")
            print(f"Coverage Y: {c['y_range_um'][0] / 1000:.1f} - {c['y_range_um'][1] / 1000:.1f} mm")
            print(f"Corners covered: {', '.join(c['corners_covered']) or 'none'}")
            print(f"Corners extrapolated: {', '.join(c['corners_extrapolated']) or 'none'}")
            print()
            print(f"Exported to: {export_plane_path}")

        # Generate contour map
        if plot:
            contour_path = export_plane_path.with_name(export_plane_path.stem.replace("_plane", "") + "_contour.png")
            plot_contour_map(result, contour_path, quiet=quiet)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze focus map data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("focus_map", type=Path, help="Path to focus_map_chip*.json")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output plot path")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plot")
    parser.add_argument("--max-dim", type=int, default=4500, help="Max canvas dimension for mosaic")
    parser.add_argument("--margin", type=int, default=5, help="Gap between images in mosaic")
    parser.add_argument("--no-mosaic", action="store_true", help="Skip generating mosaic image")
    parser.add_argument("--curves", action="store_true", help="Generate per-point sharpness curve mosaic")
    parser.add_argument("--export-plane", type=Path, default=None, help="Export plane fit to JSON")
    parser.add_argument("--min-sharpness", type=float, default=20.0, help="Min sharpness for plane fit")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        run(
            args.focus_map,
            output=args.output,
            plot=not args.no_plot,
            max_dim=args.max_dim,
            margin=args.margin,
            mosaic=not args.no_mosaic,
            curves=args.curves,
            export_plane_path=args.export_plane,
            min_sharpness=args.min_sharpness,
            quiet=args.quiet,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
