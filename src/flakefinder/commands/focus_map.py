"""Compute focus map for a chip using autofocus at sample points.

Samples points along the chip's convex hull edge and on an interior grid,
runs autofocus at each point, and outputs a focus map with best Z positions.

Usage:
    uv run python commands/focus_map.py --chips-meta scans/chips.json --chip 0 \\
        --gain 1 --exposure-ms 1
    uv run python commands/focus_map.py --chips-meta scans/chips.json --chip 0 \\
        --gain 1 --exposure-ms 2 --save-images --suffix v6 --z 24700
    uv run python commands/focus_map.py --chips-meta scans/chips.json --chip 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image as PILImage

from flakefinder.autofocus_util import save_debug_frames
from flakefinder.data_utils import load_chip_geometry
from flakefinder.leica import wait_all
from flakefinder.leica.autofocus import ALL_SHARPNESS_METRICS, AutofocusResult, continuous_autofocus
from flakefinder.leica.microscope import Microscope
from flakefinder.scan_utils import DEFAULT_WB, build_microscope_meta, get_git_version, parse_white_balance
from flakefinder.types import ChipGeometry, GainRGB, Point2F


class FMAfDefaults(NamedTuple):
    """Focus-map-specific autofocus defaults per objective."""

    z_range_um: float
    z_speed_um_s: float


# Tighter coarse scan for focus map AF (vs standalone autofocus auto-range).
# 10x: 250 µm at 1250 µm/s gives ~3.5 µm/frame spacing (was 500/2500 → ~7 µm/frame).
FM_AF_DEFAULTS: dict[int, FMAfDefaults] = {
    2: FMAfDefaults(z_range_um=250, z_speed_um_s=1250),  # 10x
    3: FMAfDefaults(z_range_um=500, z_speed_um_s=1250),  # 20x
}


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
        if self.error or self.af_result is None:
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
    scope: Microscope,
    *,
    z_range_um: float | None = None,
    z_speed_um_s: float | None = None,
    fine_pass: bool = True,
    fine_range_um: float = 50.0,
    super_fine_pass: bool = True,
    sharpness_method: str = "tenengrad",
    all_metrics: bool = False,
    move_settle_s: float = 0,
    af_settle_s: float = 0,
    move_to_best_z: bool = True,
    images_dir: Path | None = None,
    debug_dir: Path | None = None,
    save_executor: ThreadPoolExecutor | None = None,
    quiet: bool = False,
) -> list[FocusMapSample]:
    """Run autofocus at each sample point and return results.

    Args:
        all_points: Sample points to autofocus at.
        reference_z_um: Starting Z position for each autofocus sweep.
        scope: Microscope facade instance.
        z_range_um: Z scan range in µm (None = auto from objective).
        z_speed_um_s: Z axis speed in µm/s (None = use current).
        fine_pass: Enable two-pass autofocus (coarse + fine).
        fine_range_um: Fine pass range in µm.
        super_fine_pass: Enable super fine third pass.
        sharpness_method: Sharpness metric name.
        all_metrics: Compute all sharpness metrics per frame.
        move_settle_s: Settle time after XY move.
        af_settle_s: Settle time after autofocus (before image capture).
        move_to_best_z: Move Z back to best position and verify sharpness.
            False skips the return move, settle, and final capture.
        images_dir: Directory for after-images, or None.
        debug_dir: Directory for AF debug frames, or None.
        save_executor: ThreadPoolExecutor for background disk writes, or None.
        quiet: Suppress per-point progress output.

    Returns:
        List of FocusMapSample (one per point).
    """
    stage = scope.stage

    sample_results = []
    save_futures = []
    total_points = len(all_points)

    if not quiet:
        print(f"\nRunning autofocus at {total_points} points...")

    for i, pt in enumerate(all_points):
        if not quiet:
            x_mm, y_mm = pt.x_um / 1000, pt.y_um / 1000
            print(
                f"  [{i + 1}/{total_points}] {pt.type} {pt.index}: ({x_mm:.2f}, {y_mm:.2f}) mm ... ",
                end="",
                flush=True,
            )

        # Move to position
        hx, hy = stage.move_to_async(pt.x_um, pt.y_um)
        wait_all([hx, hy])
        if move_settle_s > 0:
            time.sleep(move_settle_s)

        # Run autofocus
        image_path = None
        label = f"{'c' if pt.type == 'contour' else 'g'}{pt.index:02d}"

        try:
            af_result = continuous_autofocus(
                scope,
                z_start_um=reference_z_um,
                z_range_um=z_range_um,
                z_speed_um_s=z_speed_um_s,
                fine_pass=fine_pass,
                fine_range_um=fine_range_um,
                super_fine_pass=super_fine_pass,
                sharpness_method=sharpness_method,
                store_frames=bool(debug_dir),
                compute_all_metrics=all_metrics,
                move_to_best_z=move_to_best_z,
            )
            best_z = af_result.selected_z_um
            selected_sharpness = af_result.selected_sharpness
            final_sharpness = af_result.final_sharpness
            if not quiet:
                print(
                    f"Z={best_z:.1f} µm, sharpness={selected_sharpness:.1f}/{final_sharpness:.1f}",
                    end="",
                )

            # Capture after image while still at this position (needs camera)
            after_img = None
            if images_dir is not None:
                if af_settle_s > 0:
                    time.sleep(af_settle_s)
                after_img = scope.camera.capture()
                if after_img is not None:
                    fname = f"{pt.type}_{pt.index:02d}.jpg"
                    image_path = images_dir / fname
                    if not quiet:
                        print(f" -> {fname}", end="")

            # Queue all disk writes to background thread
            if save_executor and (af_result.frames is not None or after_img is not None):
                _af = af_result
                _after = after_img
                _label = label
                _image_path = image_path
                _debug_dir = debug_dir

                def _save(af=_af, after=_after, lbl=_label, img_path=_image_path, dbg=_debug_dir):
                    if dbg and af.frames:
                        save_debug_frames(af, dbg / lbl)
                    if after is not None:
                        if img_path:
                            PILImage.fromarray(after).save(img_path, quality=95)
                        if dbg:
                            point_dir = dbg / lbl
                            point_dir.mkdir(parents=True, exist_ok=True)
                            PILImage.fromarray(after).save(point_dir / "after.png")

                save_futures.append(save_executor.submit(_save))

            if not quiet:
                print()
        except Exception as e:
            print(f"  [{i + 1}/{total_points}] FAILED: {e}" if quiet else f"FAILED: {e}")
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


@dataclass
class _Preflight:
    """Validated sample points and chip geometry from _plan()."""

    points: list[SamplePoint]
    chip_geo: ChipGeometry
    contour_count: int
    grid_count: int
    perimeter_mm: float

    def print_summary(self, contour_spacing_mm: float) -> None:
        actual_mm = self.perimeter_mm / self.contour_count
        print(f"{len(self.points)} sample points")
        print(f"  Perimeter: {self.perimeter_mm:.1f} mm, spacing: {contour_spacing_mm:.1f} mm")
        print(f"  Contour: {self.contour_count} points ({actual_mm:.1f} mm apart)")
        print(f"  Grid: {self.grid_count} points")


def _nearest_neighbor_order(
    points: list[SamplePoint],
    start: Point2F,
) -> list[SamplePoint]:
    """Reorder points by nearest-neighbor traversal from start position."""
    if len(points) <= 1:
        return list(points)

    remaining = list(points)
    ordered: list[SamplePoint] = []
    cx, cy = start

    while remaining:
        best_idx = 0
        best_dist_sq = float("inf")
        for i, pt in enumerate(remaining):
            d = (pt.x_um - cx) ** 2 + (pt.y_um - cy) ** 2
            if d < best_dist_sq:
                best_dist_sq = d
                best_idx = i
        nearest = remaining.pop(best_idx)
        ordered.append(nearest)
        cx, cy = nearest.x_um, nearest.y_um

    return ordered


def _plan(
    *,
    chips_meta: Path,
    chip: int = 0,
    contour_spacing_mm: float = 15.0,
    grid_spacing_um: float = 10000,
) -> _Preflight:
    """Compute sample points for focus map (pure computation, no hardware).

    Raises:
        ValueError: On invalid inputs (missing file, bad chip index).
        FileNotFoundError: If chips_meta file doesn't exist.
    """
    chip_geo = load_chip_geometry(chips_meta, chip)
    convex_hull = chip_geo.polygon

    # Compute contour sample count from perimeter and target spacing
    perimeter_mm = hull_perimeter_um(convex_hull) / 1000
    contour_samples = max(4, min(24, round(perimeter_mm / contour_spacing_mm)))

    # Generate sample points
    contour_points = sample_contour_points(convex_hull, contour_samples)
    grid_points = sample_grid_points(convex_hull, grid_spacing_um)

    # All sample points with labels
    all_points: list[SamplePoint] = []
    for i, (x, y) in enumerate(contour_points):
        all_points.append(SamplePoint(x, y, "contour", i))
    for i, (x, y) in enumerate(grid_points):
        all_points.append(SamplePoint(x, y, "grid", i))

    # Reorder for efficient traversal (nearest-neighbor from centroid)
    all_points = _nearest_neighbor_order(all_points, chip_geo.centroid)

    return _Preflight(
        points=all_points,
        chip_geo=chip_geo,
        contour_count=len(contour_points),
        grid_count=len(grid_points),
        perimeter_mm=perimeter_mm,
    )


def run(
    scope: Microscope,
    *,
    chips_meta: Path,
    gain: float,
    exposure_ms: float,
    chip: int = 0,
    contour_spacing_mm: float = 15.0,
    grid_spacing_um: float = 10000,
    objective_mag: str | None = None,
    z_range: float | None = None,
    fine_pass: bool = True,
    fine_range: float = 50.0,
    super_fine_pass: bool = True,
    all_metrics: bool = False,
    output_dir: Path | None = None,
    save_images: bool = False,
    debug_dir: Path | None = None,
    sharpness_method: str = "tenengrad",
    z: float | None = None,
    z_speed: float | None = None,
    move_settle: float = 0,
    af_settle: float = 0,
    move_to_best_z: bool = True,
    white_balance: GainRGB = DEFAULT_WB,
    suffix: str | None = None,
    notes: str | None = None,
    quiet: bool = False,
) -> None:
    """Run focus map computation.

    Raises:
        ValueError: On invalid inputs.
        FileNotFoundError: If chips_meta file doesn't exist.
    """
    p = _plan(
        chips_meta=chips_meta,
        chip=chip,
        contour_spacing_mm=contour_spacing_mm,
        grid_spacing_um=grid_spacing_um,
    )
    all_points = p.points
    chip_geo = p.chip_geo

    if not quiet:
        p.print_summary(contour_spacing_mm)

    # Output directory and stem
    output_dir = output_dir or chips_meta.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"focus_map_chip{chip}_{suffix}" if suffix else f"focus_map_chip{chip}"

    start_time = time.perf_counter()

    # Switch objective if requested
    if objective_mag is not None:
        if scope.switch_objective_mag(objective_mag):
            if not quiet:
                print(f"Switched objective to {scope.objective_mag}x")
        else:
            if not quiet:
                print(f"Objective: already at {scope.objective_mag}x")

    scope.light_on()

    if not quiet:
        x, y = scope.stage.position_um
        print(f"\nCurrent position: X={x:.1f}, Y={y:.1f}, Z={scope.z.position_um:.1f} µm")

    # Configure camera for fast capture
    camera = scope.camera
    camera.trigger_mode = 0  # CONTINUOUS
    camera.binning = 2  # 3x3 binning for speed
    camera.exposure_time = exposure_ms / 1000.0
    camera.gain = gain
    camera.gain_rgb = white_balance
    camera.gamma = 1.0

    if not quiet:
        print(f"Camera: {camera.name}")
        print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")

    # Apply focus-map-specific AF defaults if not explicitly set
    fm_defaults = FM_AF_DEFAULTS.get(scope.nosepiece.position)
    if fm_defaults is not None:
        if z_range is None:
            z_range = fm_defaults.z_range_um
        if z_speed is None:
            z_speed = fm_defaults.z_speed_um_s

    # Resolve reference Z
    if z is not None:
        reference_z_um = z
    else:
        # Centroid AF uses wider range than grid points (2x FM default, cap 500 µm)
        # to tolerate initial Z uncertainty after objective swap.
        CENTROID_RANGE_CAP_UM = 500.0
        centroid_z_range = min(z_range * 2, CENTROID_RANGE_CAP_UM) if z_range is not None else None

        cx, cy = chip_geo.centroid
        cx_mm, cy_mm = cx / 1000, cy / 1000
        if not quiet:
            print(f"\nNo --z provided, autofocusing at centroid ({cx_mm:.2f}, {cy_mm:.2f}) mm...")
        hx, hy = scope.stage.move_to_async(cx, cy)
        wait_all([hx, hy])
        centroid_af = continuous_autofocus(
            scope,
            z_range_um=centroid_z_range,
            z_speed_um_s=z_speed,
            fine_pass=fine_pass,
            fine_range_um=fine_range,
            super_fine_pass=super_fine_pass,
            sharpness_method=sharpness_method,
        )
        reference_z_um = centroid_af.selected_z_um
        sharpness = centroid_af.selected_sharpness
        if not quiet:
            print(f"  Centroid AF: Z={reference_z_um:.1f} µm, sharpness={sharpness:.1f}")

    # Create images directory if saving images
    images_dir = None
    if save_images:
        images_dir = output_dir / f"{stem}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"Saving images to {images_dir}")

    save_executor = ThreadPoolExecutor(max_workers=1) if (debug_dir or save_images) else None

    sample_results = run_focus_map(
        all_points=all_points,
        reference_z_um=reference_z_um,
        scope=scope,
        z_range_um=z_range,
        z_speed_um_s=z_speed,
        fine_pass=fine_pass,
        fine_range_um=fine_range,
        super_fine_pass=super_fine_pass,
        sharpness_method=sharpness_method,
        all_metrics=all_metrics,
        move_settle_s=move_settle,
        af_settle_s=af_settle,
        move_to_best_z=move_to_best_z,
        images_dir=images_dir,
        debug_dir=debug_dir,
        save_executor=save_executor,
        quiet=quiet,
    )

    # Wait for background saves to finish
    if save_executor:
        save_executor.shutdown(wait=True)

    # Save results before __exit__ (SDK Dispose can crash the process)
    duration_s = time.perf_counter() - start_time

    output = {
        "git_version": get_git_version(),
        "timestamp": datetime.now().isoformat(),
        "command": sys.argv,
        "duration_s": round(duration_s, 2),
        "chip_id": chip,
        "source_chips_meta": str(chips_meta),
        "notes": notes,
        **build_microscope_meta(scope),
        "grid_params": {
            "contour_samples": p.contour_count,
            "contour_spacing_mm": contour_spacing_mm,
            "grid_spacing_um": grid_spacing_um,
            "z_start_um": reference_z_um,
            "z_range_um": z_range,
            "z_speed_um_s": z_speed,
            "fine_pass": fine_pass,
            "fine_range_um": fine_range,
            "super_fine_pass": super_fine_pass,
            "sharpness_method": sharpness_method,
            "move_settle_s": move_settle,
            "af_settle_s": af_settle,
            "save_images": save_images,
            "debug_dir": str(debug_dir) if debug_dir else None,
        },
        "sample_points": [s.to_dict() for s in sample_results],
    }

    output_path = output_dir / f"{stem}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    successful_results = [s.af_result for s in sample_results if s.af_result is not None]
    z_values = [r.selected_z_um for r in successful_results]
    n_ok = len(successful_results)
    n_total = len(sample_results)

    # One-line summary (always printed)
    z_range_str = f"Z={min(z_values):.0f}-{max(z_values):.0f} µm" if z_values else "no Z data"
    print(f"focus_map: {n_ok}/{n_total} OK, {z_range_str}, {duration_s:.1f}s, {output_path}")

    if not quiet and z_values:
        selected_sharpness_values = [r.selected_sharpness for r in successful_results]
        final_sharpness_values = [r.final_sharpness for r in successful_results]
        sel_mean = np.mean(selected_sharpness_values)
        sel_std = np.std(selected_sharpness_values)
        fin_mean = np.mean(final_sharpness_values)
        fin_std = np.std(final_sharpness_values)
        print(f"\n  Z mean: {np.mean(z_values):.1f} µm")
        print(f"  Z std: {np.std(z_values):.1f} µm")
        print(f"  Sharpness (selected): {sel_mean:.1f} ± {sel_std:.1f}")
        print(f"  Sharpness (final): {fin_mean:.1f} ± {fin_std:.1f}")


def _build_parser():
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
        "--gain",
        type=float,
        required=True,
        help="Camera gain (e.g. 1 for AF, 4 for chip scan)",
    )
    parser.add_argument(
        "--exposure-ms",
        type=float,
        required=True,
        help="Exposure time in ms (e.g. 1 for 10x, 2 for 20x/50x)",
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
        default=15.0,
        help="Target spacing between contour points in mm (count = perimeter/spacing, clamped to 4-24)",
    )
    parser.add_argument(
        "--grid-spacing-um",
        type=float,
        default=10000,
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
        "--no-verify",
        action="store_true",
        help="Skip move-back-to-best-Z verify step (matches pipeline behavior)",
    )
    parser.add_argument(
        "--white-balance",
        type=parse_white_balance,
        default="2.51,1.02,1.41",
        help="White balance as B,G,R gains (default: 2.51,1.02,1.41)",
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
        "--objective-mag",
        type=str,
        metavar="MAG",
        help="Objective by magnification (e.g., 5, 5x, 20, 2.5) - switches before running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample points without running autofocus",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress verbose output; print one summary line",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.dry_run:
        try:
            p = _plan(
                chips_meta=args.chips_meta,
                chip=args.chip,
                contour_spacing_mm=args.contour_spacing_mm,
                grid_spacing_um=args.grid_spacing_um,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            return 1
        p.print_summary(args.contour_spacing_mm)
        print("\n[DRY RUN] Would autofocus at these points:")
        for pt in p.points:
            print(f"  {pt.type:7} {pt.index:2}: ({pt.x_um / 1000:.2f}, {pt.y_um / 1000:.2f}) mm")
        return 0

    try:
        with Microscope() as scope:
            run(
                scope,
                chips_meta=args.chips_meta,
                gain=args.gain,
                exposure_ms=args.exposure_ms,
                chip=args.chip,
                contour_spacing_mm=args.contour_spacing_mm,
                grid_spacing_um=args.grid_spacing_um,
                objective_mag=args.objective_mag,
                z_range=args.z_range,
                fine_pass=not args.no_fine_pass,
                fine_range=args.fine_range,
                super_fine_pass=not args.no_super_fine,
                all_metrics=args.all_metrics,
                output_dir=args.output_dir,
                save_images=args.save_images,
                debug_dir=args.debug_dir,
                sharpness_method=args.sharpness_method,
                z=args.z,
                z_speed=args.z_speed,
                move_settle=args.move_settle,
                af_settle=0 if args.no_verify else args.af_settle,
                move_to_best_z=not args.no_verify,
                white_balance=args.white_balance,
                suffix=args.suffix,
                notes=args.notes,
                quiet=args.quiet,
            )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
