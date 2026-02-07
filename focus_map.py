"""Compute focus map for a chip using autofocus at sample points.

Samples points along the chip's convex hull edge and on an interior grid,
runs autofocus at each point, and outputs a focus map with best Z positions.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from PIL import Image as PILImage

from flakefinder.leica.autofocus import ALL_SHARPNESS_METRICS


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
    convex_hull: list[list[float]],
    spacing_um: float,
    inset_um: float = 1500.0,
) -> list[tuple[float, float]]:
    """Sample interior grid points within a convex hull, centered on centroid.

    Grid is centered on the hull centroid and only includes points inside
    the hull (with an inset margin to avoid crowding contour points).

    Args:
        convex_hull: List of [x, y] points defining the hull in stage coords.
        spacing_um: Grid spacing in micrometers.
        inset_um: Inset margin from hull edge in µm (avoids crowding contour points).

    Returns:
        List of (x, y) tuples in stage coordinates.
    """
    hull = np.array(convex_hull)

    # Compute centroid
    cx = hull[:, 0].mean()
    cy = hull[:, 1].mean()

    # Shrink hull inward by inset_um for the containment test
    if inset_um > 0:
        # Shrink toward centroid
        dists = np.sqrt((hull[:, 0] - cx) ** 2 + (hull[:, 1] - cy) ** 2)
        min_dist = dists.min()
        if min_dist > inset_um:
            scale = (min_dist - inset_um) / min_dist
        else:
            scale = 1.0
        inset_hull = np.column_stack([
            cx + (hull[:, 0] - cx) * scale,
            cy + (hull[:, 1] - cy) * scale,
        ])
    else:
        inset_hull = hull

    def point_in_polygon(px, py, polygon):
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # Bounding box of full hull
    x_min, x_max = hull[:, 0].min(), hull[:, 0].max()
    y_min, y_max = hull[:, 1].min(), hull[:, 1].max()

    # Generate grid centered on centroid
    # Expand outward from centroid in both directions
    x_half = int((x_max - cx) / spacing_um) + 1
    y_half = int((y_max - cy) / spacing_um) + 1
    x_vals = cx + np.arange(-x_half, x_half + 1) * spacing_um
    y_vals = cy + np.arange(-y_half, y_half + 1) * spacing_um

    # Filter to points inside inset hull
    points = []
    for y in y_vals:
        for x in x_vals:
            if point_in_polygon(x, y, inset_hull):
                points.append((float(x), float(y)))

    # Ensure centroid is included (deduplicate if already present)
    centroid = (float(cx), float(cy))
    min_dist_to_existing = min(
        ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in points
    ) if points else float('inf')
    if min_dist_to_existing > (spacing_um / 4) ** 2:
        points.append(centroid)

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
        default=16,
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
        "--fine-range",
        type=float,
        default=50.0,
        help="Fine pass range in µm (default: 50)",
    )
    parser.add_argument(
        "--super-fine",
        action="store_true",
        help="Enable super fine third pass: 10µm range at 20µm/s",
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
        "--debug-dir",
        type=Path,
        default=None,
        help="Save all AF scan frames per point (coarse/fine PNGs + sharpness CSV)",
    )
    parser.add_argument(
        "--sharpness-method",
        choices=list(ALL_SHARPNESS_METRICS.keys()),
        default="tenengrad",
        help="Sharpness metric for autofocus",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Reference Z position in µm (default: autofocus at chip centroid)",
    )
    parser.add_argument(
        "--z-speed",
        type=float,
        default=None,
        help="Z axis speed in µm/s (default: use current)",
    )
    parser.add_argument(
        "--move-settle",
        type=float,
        default=0,
        help="Settle time in seconds after stage XY move",
    )
    parser.add_argument(
        "--af-settle",
        type=float,
        default=0,
        help="Settle time in seconds after autofocus (before image capture)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix for output filenames (e.g. 'v6' -> focus_map_chip0_v6.json)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes to display on rendered mosaic",
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
    grid_points = sample_grid_points(convex_hull, args.grid_spacing_um)

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
    from flakefinder.autofocus_util import save_debug_frames

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

        # Determine reference Z (autofocus at centroid if not provided)
        if args.z is None:
            hull = np.array(convex_hull)
            cx, cy = float(hull[:, 0].mean()), float(hull[:, 1].mean())
            print(f"\nNo --z provided, autofocusing at chip centroid ({cx/1000:.2f}, {cy/1000:.2f}) mm...")
            hx, hy = stage.move_to_async(cx, cy)
            Stage.wait_all([hx, hy])
            hx.dispose()
            hy.dispose()
            centroid_af = continuous_autofocus(
                conn=conn,
                camera=camera,
                acquisition=acquisition,
                context=context,
                z_range_um=args.z_range,
                z_speed_um_s=args.z_speed,
                fine_pass=not args.no_fine_pass,
                fine_range_um=args.fine_range,
                super_fine_pass=args.super_fine,
                sharpness_method=args.sharpness_method,
            )
            args.z = centroid_af.selected_z_um
            print(f"  Centroid AF: Z={args.z:.1f} µm, sharpness={centroid_af.selected_sharpness:.1f}")

        # Create images directory if saving images
        images_dir = None
        if args.save_images:
            stem = f"focus_map_chip{args.chip}_{args.suffix}" if args.suffix else f"focus_map_chip{args.chip}"
            images_dir = output_dir / f"{stem}_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving images to {images_dir}")

        # Collect results
        sample_results = []
        total_points = len(all_points)
        save_executor = ThreadPoolExecutor(max_workers=1) if (args.debug_dir or args.save_images) else None
        save_futures = []

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
            if args.move_settle > 0:
                time.sleep(args.move_settle)

            # Run autofocus
            image_path = None
            label = f"{'c' if pt['type'] == 'contour' else 'g'}{pt['index']:02d}"

            try:
                af_result = continuous_autofocus(
                    conn=conn,
                    camera=camera,
                    acquisition=acquisition,
                    context=context,
                    z_start_um=args.z,
                    z_range_um=args.z_range,
                    z_speed_um_s=args.z_speed,
                    fine_pass=not args.no_fine_pass,
                    fine_range_um=args.fine_range,
                    super_fine_pass=args.super_fine,
                    sharpness_method=args.sharpness_method,
                    store_frames=bool(args.debug_dir),
                    compute_all_metrics=bool(args.debug_dir),
                )
                best_z = af_result.selected_z_um
                selected_sharpness = af_result.selected_sharpness
                final_sharpness = af_result.final_sharpness
                print(f"Z={best_z:.1f} µm, sharpness={selected_sharpness:.1f}/{final_sharpness:.1f}", end="")

                # Capture after image while still at this position (needs camera)
                after_img = None
                if images_dir is not None:
                    if args.af_settle > 0:
                        time.sleep(args.af_settle)
                    after_img = camera.capture()
                    if after_img is not None:
                        fname = f"{pt['type']}_{pt['index']:02d}.jpg"
                        image_path = images_dir / fname
                        print(f" -> {fname}", end="")

                # Queue all disk writes to background thread
                if save_executor and (af_result.frames is not None or after_img is not None):
                    _af = af_result
                    _after = after_img
                    _label = label
                    _image_path = image_path
                    _debug_dir = args.debug_dir

                    def _save(af=_af, after=_after, lbl=_label, img_path=_image_path, dbg=_debug_dir):
                        if dbg and af.frames:
                            save_debug_frames(af, dbg / lbl)
                        if after is not None:
                            if img_path:
                                PILImage.fromarray(after).save(str(img_path), quality=95)
                            if dbg:
                                point_dir = dbg / lbl
                                point_dir.mkdir(parents=True, exist_ok=True)
                                PILImage.fromarray(after).save(str(point_dir / "after.png"))

                    save_futures.append(save_executor.submit(_save))

                print()
            except Exception as e:
                print(f"FAILED: {e}")
                af_result = None

            sample_results.append({
                "x_um": x_um,
                "y_um": y_um,
                "type": pt["type"],
                "index": pt["index"],
                "selected_z_um": af_result.selected_z_um if af_result else None,
                "selected_sharpness": af_result.selected_sharpness if af_result else None,
                "scan_best_z_um": af_result.scan_best_z_um if af_result else None,
                "scan_best_sharpness": af_result.scan_best_sharpness if af_result else None,
                "final_sharpness": af_result.final_sharpness if af_result else None,
                "dynamic_range": af_result.dynamic_range if af_result else None,
                "initial_z_um": af_result.initial_z_um if af_result else None,
                "coarse_z_start_um": af_result.coarse_z_start_um if af_result else None,
                "coarse_z_end_um": af_result.coarse_z_end_um if af_result else None,
                "coarse_best_z_um": af_result.coarse_best_z_um if af_result else None,
                "coarse_best_sharpness": af_result.coarse_best_sharpness if af_result else None,
                "fine_z_start_um": af_result.fine_z_start_um if af_result else None,
                "fine_z_end_um": af_result.fine_z_end_um if af_result else None,
                "stayed_at_initial": af_result.stayed_at_initial if af_result else None,
                "image": str(image_path.name) if image_path else None,
            })

        # Wait for background saves to finish
        if save_executor:
            for fut in save_futures:
                fut.result()  # raises if any save failed
            save_executor.shutdown(wait=True)

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
        "notes": args.notes,
        "grid_params": {
            "contour_samples": args.contour_samples,
            "grid_spacing_um": args.grid_spacing_um,
            "z_start_um": args.z,
            "z_range_um": args.z_range,
            "z_speed_um_s": args.z_speed,
            "fine_pass": not args.no_fine_pass,
            "fine_range_um": args.fine_range,
            "super_fine_pass": args.super_fine,
            "sharpness_method": args.sharpness_method,
            "move_settle_s": args.move_settle,
            "af_settle_s": args.af_settle,
            "save_images": args.save_images,
            "debug_dir": str(args.debug_dir) if args.debug_dir else None,
        },
        "sample_points": sample_results,
    }

    # Save
    stem = f"focus_map_chip{args.chip}_{args.suffix}" if args.suffix else f"focus_map_chip{args.chip}"
    output_path = output_dir / f"{stem}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFocus map saved to {output_path}")

    # Summary
    successful = [r for r in sample_results if r["selected_z_um"] is not None]
    z_values = [r["selected_z_um"] for r in successful]
    selected_sharpness_values = [r["selected_sharpness"] for r in successful]
    final_sharpness_values = [r["final_sharpness"] for r in successful]

    print(f"\nSummary:")
    print(f"  Duration: {duration_s:.1f}s")
    print(f"  Points sampled: {len(sample_results)}")
    print(f"  Successful: {len(successful)}")

    if z_values:
        print(f"  Z range: {min(z_values):.1f} - {max(z_values):.1f} µm")
        print(f"  Z mean: {np.mean(z_values):.1f} µm")
        print(f"  Z std: {np.std(z_values):.1f} µm")
        print(f"  Sharpness (selected): {np.mean(selected_sharpness_values):.1f} ± {np.std(selected_sharpness_values):.1f}")
        print(f"  Sharpness (final): {np.mean(final_sharpness_values):.1f} ± {np.std(final_sharpness_values):.1f}")

    return 0


if __name__ == "__main__":
    exit(main())
