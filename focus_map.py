"""Compute focus map for a chip using autofocus at sample points.

Samples points along the chip's convex hull edge and on an interior grid,
runs autofocus at each point, and outputs a focus map with best Z positions.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from PIL import Image as PILImage


def load_chips_meta(chips_meta_path: Path) -> dict:
    """Load chips metadata from JSON file."""
    if not chips_meta_path.exists():
        raise FileNotFoundError(f"Chips metadata not found: {chips_meta_path}")
    with open(chips_meta_path) as f:
        return json.load(f)


def sample_contour_points(
    convex_hull: list[list[float]],
    num_samples: int,
) -> list[tuple[float, float]]:
    """Sample points evenly along a convex hull perimeter.

    Args:
        convex_hull: List of [x, y] points defining the hull in stage coords.
        num_samples: Number of points to sample along the perimeter.

    Returns:
        List of (x, y) tuples in stage coordinates.
    """
    if len(convex_hull) < 2:
        return [(convex_hull[0][0], convex_hull[0][1])] if convex_hull else []

    # Calculate cumulative distance along perimeter
    hull = np.array(convex_hull)
    # Close the loop by appending first point
    hull_closed = np.vstack([hull, hull[0:1]])

    # Calculate segment lengths
    diffs = np.diff(hull_closed, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_perimeter = cumulative_dist[-1]

    # Sample at evenly spaced distances
    sample_distances = np.linspace(0, total_perimeter, num_samples, endpoint=False)

    points = []
    for d in sample_distances:
        # Find which segment contains this distance
        idx = np.searchsorted(cumulative_dist, d, side="right") - 1
        idx = max(0, min(idx, len(hull_closed) - 2))

        # Interpolate within segment
        seg_start_dist = cumulative_dist[idx]
        seg_length = segment_lengths[idx]
        if seg_length > 0:
            t = (d - seg_start_dist) / seg_length
        else:
            t = 0

        p1 = hull_closed[idx]
        p2 = hull_closed[idx + 1]
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        points.append((float(x), float(y)))

    return points


def sample_grid_points(
    bbox: dict,
    spacing_um: float,
) -> list[tuple[float, float]]:
    """Sample interior grid points within a bounding box.

    Args:
        bbox: Dict with x_min, y_min, x_max, y_max in stage coords.
        spacing_um: Grid spacing in micrometers.

    Returns:
        List of (x, y) tuples in stage coordinates.
    """
    x_min, x_max = bbox["x_min"], bbox["x_max"]
    y_min, y_max = bbox["y_min"], bbox["y_max"]

    # Add margin to avoid sampling exactly on edge
    margin = spacing_um / 4

    x_start = x_min + margin
    x_end = x_max - margin
    y_start = y_min + margin
    y_end = y_max - margin

    if x_start >= x_end or y_start >= y_end:
        # Box too small for grid
        return [((x_min + x_max) / 2, (y_min + y_max) / 2)]

    # Generate grid
    x_vals = np.arange(x_start, x_end, spacing_um)
    y_vals = np.arange(y_start, y_end, spacing_um)

    points = []
    for y in y_vals:
        for x in x_vals:
            points.append((float(x), float(y)))

    return points


def main():
    parser = argparse.ArgumentParser(
        description="Compute focus map for a chip using autofocus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chips-meta",
        type=Path,
        required=True,
        help="Path to *_chips.json file",
    )
    parser.add_argument(
        "--chip",
        type=int,
        default=0,
        help="Chip number to map (0-indexed)",
    )
    parser.add_argument(
        "--contour-samples",
        type=int,
        default=8,
        help="Number of points to sample along contour edge",
    )
    parser.add_argument(
        "--grid-spacing-um",
        type=float,
        default=4500,
        help="Interior grid spacing in µm",
    )
    parser.add_argument(
        "--z-range",
        type=float,
        default=None,
        help="Z scan range in µm (default: auto from objective)",
    )
    parser.add_argument(
        "--no-fine-pass",
        action="store_true",
        help="Disable two-pass autofocus (use single coarse pass only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as chips meta)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save 'after' image at each focus point",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample points without running autofocus",
    )
    args = parser.parse_args()

    # Load chips metadata
    print(f"Loading chips metadata from {args.chips_meta}")
    chips_meta = load_chips_meta(args.chips_meta)

    chips = chips_meta.get("chips", [])
    if not chips:
        print("Error: No chips found in metadata")
        return 1

    if args.chip < 0 or args.chip >= len(chips):
        print(f"Error: Chip {args.chip} not found (have {len(chips)} chips)")
        return 1

    chip = chips[args.chip]
    print(f"Processing chip {args.chip}")

    # Get hull and bbox
    convex_hull = chip.get("convex_hull_stage_um", [])
    bbox = chip.get("bbox_stage_um", {})

    if not convex_hull:
        print("Error: Chip has no convex hull data")
        return 1
    if not bbox:
        print("Error: Chip has no bounding box data")
        return 1

    # Generate sample points
    contour_points = sample_contour_points(convex_hull, args.contour_samples)
    grid_points = sample_grid_points(bbox, args.grid_spacing_um)

    print(f"Sample points:")
    print(f"  Contour: {len(contour_points)} points")
    print(f"  Grid: {len(grid_points)} points")
    print(f"  Total: {len(contour_points) + len(grid_points)} points")

    # All sample points with labels
    all_points = []
    for i, (x, y) in enumerate(contour_points):
        all_points.append({"x_um": x, "y_um": y, "type": "contour", "index": i})
    for i, (x, y) in enumerate(grid_points):
        all_points.append({"x_um": x, "y_um": y, "type": "grid", "index": i})

    if args.dry_run:
        print("\n[DRY RUN] Would autofocus at these points:")
        for pt in all_points:
            print(f"  {pt['type']:7} {pt['index']:2}: ({pt['x_um']/1000:.2f}, {pt['y_um']/1000:.2f}) mm")
        return 0

    # Output directory
    output_dir = args.output_dir or args.chips_meta.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import hardware libraries (after dry-run check)
    from flakefinder.leica import LeicaConnection, Stage, Lamp, Shutter, ZDrive
    from flakefinder.leica.camera import Camera
    from flakefinder.leica.enums import UCAPI_IID
    from flakefinder.leica.core import get_interface_required
    from flakefinder.leica.autofocus import continuous_autofocus

    start_time = time.perf_counter()

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up hardware
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        current_x, current_y = stage.position_um
        current_z = z_drive.position_um

        print(f"\nCurrent position: X={current_x:.1f}, Y={current_y:.1f}, Z={current_z:.1f} µm")

        # Lighting
        try:
            shutter = Shutter.from_connection(conn)
            shutter.open()
        except LookupError:
            pass

        try:
            lamp = Lamp.from_connection(conn)
            lamp.full()
        except LookupError:
            lamp = None

        # Camera
        try:
            camera = Camera.from_connection(conn)
        except LookupError:
            print("Error: Camera not found")
            return 1

        acquisition = get_interface_required(camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)

        # Configure camera for fast capture
        camera.trigger_mode = 0  # CONTINUOUS
        camera.binning = 2       # 3x3 binning for speed
        camera.exposure_time = 0.001  # 1ms

        print(f"Camera: {camera.name}")
        if lamp:
            print(f"Lamp: {lamp.name}, intensity={lamp.intensity}/{lamp.max_intensity}")

        # Acquisition context
        context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory

        # Create images directory if saving images
        images_dir = None
        if args.save_images:
            images_dir = output_dir / f"focus_map_chip{args.chip}_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving images to {images_dir}")

        # Collect results
        sample_results = []
        total_points = len(all_points)

        print(f"\nRunning autofocus at {total_points} points...")

        for i, pt in enumerate(all_points):
            x_um, y_um = pt["x_um"], pt["y_um"]

            print(f"  [{i+1}/{total_points}] {pt['type']} {pt['index']}: "
                  f"({x_um/1000:.2f}, {y_um/1000:.2f}) mm ... ", end="", flush=True)

            # Move to position
            hx, hy = stage.move_to_async(x_um, y_um)
            Stage.wait_all([hx, hy])
            hx.dispose()
            hy.dispose()

            # Run autofocus
            image_path = None
            try:
                af_result = continuous_autofocus(
                    conn=conn,
                    camera=camera,
                    acquisition=acquisition,
                    context=context,
                    z_range_um=args.z_range,
                    fine_pass=not args.no_fine_pass,
                )
                best_z = af_result.best_z_um
                best_sharpness = af_result.best_sharpness
                final_sharpness = af_result.final_sharpness
                print(f"Z={best_z:.1f} µm, sharpness={best_sharpness:.1f}/{final_sharpness:.1f}", end="")

                # Save after image if requested
                if images_dir is not None:
                    img = camera.capture()
                    if img is not None:
                        fname = f"{pt['type']}_{pt['index']:02d}.jpg"
                        image_path = images_dir / fname
                        PILImage.fromarray(img).save(str(image_path), quality=95)
                        print(f" -> {fname}", end="")
                print()
            except Exception as e:
                print(f"FAILED: {e}")
                best_z = None
                best_sharpness = None
                final_sharpness = None

            sample_results.append({
                "x_um": x_um,
                "y_um": y_um,
                "type": pt["type"],
                "index": pt["index"],
                "best_z_um": best_z,
                "best_sharpness": best_sharpness,
                "final_sharpness": final_sharpness,
                "image": str(image_path.name) if image_path else None,
            })

        # Dispose resources before connection closes
        try:
            context.Dispose()
        except Exception:
            pass
        camera.dispose()

    duration_s = time.perf_counter() - start_time

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(duration_s, 2),
        "chip_id": args.chip,
        "source_chips_meta": str(args.chips_meta),
        "grid_params": {
            "contour_samples": args.contour_samples,
            "grid_spacing_um": args.grid_spacing_um,
            "z_range_um": args.z_range,
            "fine_pass": not args.no_fine_pass,
            "save_images": args.save_images,
        },
        "sample_points": sample_results,
    }

    # Save
    output_path = output_dir / f"focus_map_chip{args.chip}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFocus map saved to {output_path}")

    # Summary
    successful = [r for r in sample_results if r["best_z_um"] is not None]
    z_values = [r["best_z_um"] for r in successful]
    best_sharpness_values = [r["best_sharpness"] for r in successful]
    final_sharpness_values = [r["final_sharpness"] for r in successful]

    print(f"\nSummary:")
    print(f"  Duration: {duration_s:.1f}s")
    print(f"  Points sampled: {len(sample_results)}")
    print(f"  Successful: {len(successful)}")

    if z_values:
        print(f"  Z range: {min(z_values):.1f} - {max(z_values):.1f} µm")
        print(f"  Z mean: {np.mean(z_values):.1f} µm")
        print(f"  Z std: {np.std(z_values):.1f} µm")
        print(f"  Sharpness (best): {np.mean(best_sharpness_values):.1f} ± {np.std(best_sharpness_values):.1f}")
        print(f"  Sharpness (final): {np.mean(final_sharpness_values):.1f} ± {np.std(final_sharpness_values):.1f}")

    return 0


if __name__ == "__main__":
    exit(main())
