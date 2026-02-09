"""Compute focus map for a chip using autofocus at sample points.

Samples points along the chip's convex hull edge and on an interior grid,
runs autofocus at each point, and outputs a focus map with best Z positions.
"""

import argparse
import contextlib
import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image as PILImage

from flakefinder.data_utils import load_chip_geometry
from flakefinder.leica.autofocus import ALL_SHARPNESS_METRICS, AutofocusResult
from flakefinder.types import Point2F


class SamplePoint(NamedTuple):
    x_um: float
    y_um: float
    type: str  # "contour" or "grid"
    index: int


@dataclass
class FocusMapSample:
    point: SamplePoint
    af_result: AutofocusResult | None = None  # None on error
    image: str | None = None  # filename only

    @property
    def error(self) -> bool:
        return self.af_result is None

    def to_dict(self) -> dict:
        d: dict = {
            "x_um": self.point.x_um,
            "y_um": self.point.y_um,
            "type": self.point.type,
            "index": self.point.index,
        }
        if self.error:
            d["error"] = True
        else:
            d.update(self.af_result.to_dict())
            d["image"] = self.image
        return d


def hull_perimeter_um(convex_hull: Sequence[Point2F]) -> float:
    """Compute perimeter of a convex hull in µm.

    Args:
        convex_hull: List of (x, y) vertices in stage coords (µm).

    Returns:
        Perimeter length in µm.
    """
    hull = np.array(convex_hull)
    hull_closed = np.vstack([hull, hull[0:1]])
    diffs = np.diff(hull_closed, axis=0)
    return float(np.sum(np.sqrt((diffs**2).sum(axis=1))))


def sample_contour_points(
    convex_hull: Sequence[Point2F],
    num_samples: int,
) -> list[Point2F]:
    """Sample points evenly along a convex hull perimeter.

    Args:
        convex_hull: List of (x, y) vertices in stage coords.
        num_samples: Number of points to sample along the perimeter.

    Returns:
        List of (x, y) tuples in stage coordinates.
    """
    if len(convex_hull) < 2:
        return [convex_hull[0]] if convex_hull else []

    # Calculate cumulative distance along perimeter
    hull = np.array(convex_hull)
    hull_closed = np.vstack([hull, hull[0:1]])

    diffs = np.diff(hull_closed, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
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
    convex_hull: Sequence[Point2F],
    spacing_um: float,
    inset_um: float = 1500.0,
) -> list[Point2F]:
    """Sample interior grid points within a convex hull, centered on centroid.

    Grid is centered on the hull centroid and only includes points inside
    the hull (with an inset margin to avoid crowding contour points).

    Args:
        convex_hull: List of (x, y) vertices in stage coords.
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
        inset_hull = np.column_stack(
            [
                cx + (hull[:, 0] - cx) * scale,
                cy + (hull[:, 1] - cy) * scale,
            ]
        )
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
    x_max = hull[:, 0].max()
    y_max = hull[:, 1].max()

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
    min_dist_to_existing = min(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) for p in points) if points else float("inf")
    if min_dist_to_existing > (spacing_um / 4) ** 2:
        points.append(centroid)

    return points


def run_focus_map(
    all_points: list[SamplePoint],
    reference_z_um: float,
    conn,
    stage,
    camera,
    acquisition,
    context,
    args,
    images_dir: "Path | None",
    save_executor: "ThreadPoolExecutor | None",
) -> list[FocusMapSample]:
    """Run autofocus at each sample point and return results.

    Args:
        all_points: Sample points to autofocus at.
        reference_z_um: Starting Z position for each autofocus sweep.
        conn: LeicaConnection.
        stage: Stage instance.
        camera: Camera instance.
        acquisition: Image acquisition interface.
        context: Acquisition context.
        args: Parsed CLI arguments (read-only, used for AF tuning params).
        images_dir: Directory for after-images, or None.
        save_executor: ThreadPoolExecutor for background disk writes, or None.

    Returns:
        List of FocusMapSample (one per point).
    """
    from flakefinder.autofocus_util import save_debug_frames
    from flakefinder.leica import Stage as StageClass
    from flakefinder.leica.autofocus import continuous_autofocus

    sample_results = []
    save_futures = []
    total_points = len(all_points)

    print(f"\nRunning autofocus at {total_points} points...")

    for i, pt in enumerate(all_points):
        print(
            f"  [{i + 1}/{total_points}] {pt.type} {pt.index}: ({pt.x_um / 1000:.2f}, {pt.y_um / 1000:.2f}) mm ... ",
            end="",
            flush=True,
        )

        # Move to position
        hx, hy = stage.move_to_async(pt.x_um, pt.y_um)
        StageClass.wait_all([hx, hy])
        hx.dispose()
        hy.dispose()
        if args.move_settle > 0:
            time.sleep(args.move_settle)

        # Run autofocus
        image_path = None
        label = f"{'c' if pt.type == 'contour' else 'g'}{pt.index:02d}"

        try:
            af_result = continuous_autofocus(
                conn=conn,
                camera=camera,
                acquisition=acquisition,
                context=context,
                z_start_um=reference_z_um,
                z_range_um=args.z_range,
                z_speed_um_s=args.z_speed,
                fine_pass=not args.no_fine_pass,
                fine_range_um=args.fine_range,
                super_fine_pass=not args.no_super_fine,
                sharpness_method=args.sharpness_method,
                store_frames=bool(args.debug_dir),
                compute_all_metrics=args.all_metrics,
            )
            best_z = af_result.selected_z_um
            selected_sharpness = af_result.selected_sharpness
            final_sharpness = af_result.final_sharpness
            print(
                f"Z={best_z:.1f} µm, sharpness={selected_sharpness:.1f}/{final_sharpness:.1f}",
                end="",
            )

            # Capture after image while still at this position (needs camera)
            after_img = None
            if images_dir is not None:
                if args.af_settle > 0:
                    time.sleep(args.af_settle)
                after_img = camera.capture()
                if after_img is not None:
                    fname = f"{pt.type}_{pt.index:02d}.jpg"
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

        sample_results.append(
            FocusMapSample(
                point=pt,
                af_result=af_result,
                image=str(image_path.name) if image_path else None,
            )
        )

    # Wait for background saves to finish
    if save_executor:
        for fut in save_futures:
            fut.result()  # raises if any save failed

    return sample_results


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
        "--contour-spacing-mm",
        type=float,
        default=5.0,
        help="Target spacing between contour points in mm (count = perimeter/spacing, clamped to 4-24)",
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
        "--no-super-fine",
        action="store_true",
        help="Disable super fine third pass (10µm range at 20µm/s)",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Compute all 5 sharpness metrics per frame (slow; default: primary only)",
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

    # Load chip geometry
    print(f"Loading chips metadata from {args.chips_meta}")
    try:
        chip_geo = load_chip_geometry(str(args.chips_meta), args.chip)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print(f"Processing chip {args.chip}")
    convex_hull = chip_geo.polygon

    # Compute contour sample count from perimeter and target spacing
    perimeter_mm = hull_perimeter_um(convex_hull) / 1000
    contour_samples = max(4, min(24, round(perimeter_mm / args.contour_spacing_mm)))

    # Generate sample points
    contour_points = sample_contour_points(convex_hull, contour_samples)
    grid_points = sample_grid_points(convex_hull, args.grid_spacing_um)

    spacing_mm = args.contour_spacing_mm
    actual_mm = perimeter_mm / len(contour_points)
    print(f"Sample points (perimeter={perimeter_mm:.1f} mm, spacing={spacing_mm:.1f} mm):")
    print(f"  Contour: {len(contour_points)} points ({actual_mm:.1f} mm apart)")
    print(f"  Grid: {len(grid_points)} points")
    print(f"  Total: {len(contour_points) + len(grid_points)} points")

    # All sample points with labels
    all_points: list[SamplePoint] = []
    for i, (x, y) in enumerate(contour_points):
        all_points.append(SamplePoint(x, y, "contour", i))
    for i, (x, y) in enumerate(grid_points):
        all_points.append(SamplePoint(x, y, "grid", i))

    if args.dry_run:
        print("\n[DRY RUN] Would autofocus at these points:")
        for pt in all_points:
            print(f"  {pt.type:7} {pt.index:2}: ({pt.x_um / 1000:.2f}, {pt.y_um / 1000:.2f}) mm")
        return 0

    # Output directory and stem
    output_dir = args.output_dir or args.chips_meta.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"focus_map_chip{args.chip}_{args.suffix}" if args.suffix else f"focus_map_chip{args.chip}"

    # Import hardware libraries (after dry-run check)
    from flakefinder.leica import Lamp, LeicaConnection, Shutter, Stage, ZDrive
    from flakefinder.leica.autofocus import continuous_autofocus
    from flakefinder.leica.camera import Camera
    from flakefinder.leica.core import get_interface_required
    from flakefinder.leica.enums import UCAPI_IID

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
        camera.binning = 2  # 3x3 binning for speed
        camera.exposure_time = 0.001  # 1ms

        print(f"Camera: {camera.name}")
        if lamp:
            print(f"Lamp: {lamp.name}, intensity={lamp.intensity}/{lamp.max_intensity}")

        # Acquisition context
        context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory

        # Resolve reference Z
        if args.z is not None:
            reference_z_um = args.z
        else:
            cx, cy = chip_geo.centroid
            cx_mm, cy_mm = cx / 1000, cy / 1000
            print(f"\nNo --z provided, autofocusing at centroid ({cx_mm:.2f}, {cy_mm:.2f}) mm...")
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
                super_fine_pass=not args.no_super_fine,
                sharpness_method=args.sharpness_method,
            )
            reference_z_um = centroid_af.selected_z_um
            sharpness = centroid_af.selected_sharpness
            print(f"  Centroid AF: Z={reference_z_um:.1f} µm, sharpness={sharpness:.1f}")

        # Create images directory if saving images
        images_dir = None
        if args.save_images:
            images_dir = output_dir / f"{stem}_images"
            images_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving images to {images_dir}")

        save_executor = ThreadPoolExecutor(max_workers=1) if (args.debug_dir or args.save_images) else None

        sample_results = run_focus_map(
            all_points=all_points,
            reference_z_um=reference_z_um,
            conn=conn,
            stage=stage,
            camera=camera,
            acquisition=acquisition,
            context=context,
            args=args,
            images_dir=images_dir,
            save_executor=save_executor,
        )

        # Wait for background saves to finish
        if save_executor:
            save_executor.shutdown(wait=True)

        # Dispose resources before connection closes
        with contextlib.suppress(Exception):
            context.Dispose()
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
            "contour_samples": contour_samples,
            "contour_spacing_mm": args.contour_spacing_mm,
            "grid_spacing_um": args.grid_spacing_um,
            "z_start_um": reference_z_um,
            "z_range_um": args.z_range,
            "z_speed_um_s": args.z_speed,
            "fine_pass": not args.no_fine_pass,
            "fine_range_um": args.fine_range,
            "super_fine_pass": not args.no_super_fine,
            "sharpness_method": args.sharpness_method,
            "move_settle_s": args.move_settle,
            "af_settle_s": args.af_settle,
            "save_images": args.save_images,
            "debug_dir": str(args.debug_dir) if args.debug_dir else None,
        },
        "sample_points": [s.to_dict() for s in sample_results],
    }

    # Save
    output_path = output_dir / f"{stem}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFocus map saved to {output_path}")

    # Summary
    successful = [s for s in sample_results if not s.error]
    z_values = [s.af_result.selected_z_um for s in successful]
    selected_sharpness_values = [s.af_result.selected_sharpness for s in successful]
    final_sharpness_values = [s.af_result.final_sharpness for s in successful]

    print("\nSummary:")
    print(f"  Duration: {duration_s:.1f}s")
    print(f"  Points sampled: {len(sample_results)}")
    print(f"  Successful: {len(successful)}")

    if z_values:
        print(f"  Z range: {min(z_values):.1f} - {max(z_values):.1f} µm")
        print(f"  Z mean: {np.mean(z_values):.1f} µm")
        print(f"  Z std: {np.std(z_values):.1f} µm")
        sel_mean, sel_std = np.mean(selected_sharpness_values), np.std(selected_sharpness_values)
        fin_mean, fin_std = np.mean(final_sharpness_values), np.std(final_sharpness_values)
        print(f"  Sharpness (selected): {sel_mean:.1f} ± {sel_std:.1f}")
        print(f"  Sharpness (final): {fin_mean:.1f} ± {fin_std:.1f}")

    return 0


if __name__ == "__main__":
    exit(main())
