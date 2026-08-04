"""Revisit stage points with autofocus and capture.

Takes a list of (x, y, z) points, plans a time-efficient route, moves to
each, autofocuses using z as initial guess, and captures the result image.

Usage:
    uv run python commands/revisit.py -o scans/revisit_01 \\
        --points revisit_top20_50x.json

    uv run python commands/revisit.py -o scans/revisit_01 \\
        --point 50000,40000,24700,flake_A --point 51000,41000,24750 \\
        --objective-mag 20x

    uv run python commands/revisit.py -o scans/revisit_01 \\
        --points revisit_top20_50x.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image as PILImage

from flakefinder.leica import wait_all
from flakefinder.leica.autofocus import focus_and_capture
from flakefinder.leica.microscope import Microscope
from flakefinder.scan_utils import DEFAULT_WB, build_microscope_meta, parse_white_balance
from flakefinder.types import GainRGB

XY_MAX_SPEED_MM_S = 40.0


class RevisitPoint(NamedTuple):
    x: float
    y: float
    z: float
    label: str | None


class RouteStep(NamedTuple):
    point: RevisitPoint
    original_index: int
    distance_mm: float
    travel_time_s: float


class RevisitFile(NamedTuple):
    points: list[RevisitPoint]
    objective_mag: str
    material: str | None = None  # material preset the points came from


def _parse_points_file(points_file: Path) -> RevisitFile:
    """Parse points from a JSON file.

    Expected format: {"objective_mag": ..., "points": [{x, y, z, label?}, ...], ...}
    """
    with open(points_file) as f:
        data = json.load(f)
    if not isinstance(data, dict) or "points" not in data:
        raise ValueError(f"Expected JSON object with 'points' key in {points_file}")
    obj_mag = data.get("objective_mag")
    if obj_mag is None:
        raise ValueError(f"Missing 'objective_mag' in {points_file}")
    points: list[RevisitPoint] = []
    for obj in data["points"]:
        points.append(
            RevisitPoint(
                x=float(obj["x"]),
                y=float(obj["y"]),
                z=float(obj["z"]),
                label=obj.get("label"),
            )
        )
    return RevisitFile(points=points, objective_mag=f"{obj_mag:g}x", material=data.get("material"))


def _parse_point_args(point_args: list[str]) -> list[RevisitPoint]:
    """Parse points from --point CLI args."""
    points: list[RevisitPoint] = []
    for arg in point_args:
        parts = arg.split(",")
        if len(parts) < 3:
            raise ValueError(f"--point requires at least X,Y,Z: got '{arg}'")
        label = parts[3].strip() if len(parts) >= 4 else None
        points.append(
            RevisitPoint(
                x=float(parts[0]),
                y=float(parts[1]),
                z=float(parts[2]),
                label=label,
            )
        )
    return points


def _two_opt(
    points: list[RevisitPoint],
    order: list[int],
    start_x: float,
    start_y: float,
) -> list[int]:
    """2-opt improvement on point visit order. First point is fixed."""
    n = len(order)
    if n <= 2:
        return order

    def px(i: int) -> float:
        return points[order[i]].x

    def py(i: int) -> float:
        return points[order[i]].y

    def dist(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):  # skip 0 to keep first point fixed
            for j in range(i + 1, n):
                # Cost of edges adjacent to the segment [i..j]
                prev_x = px(i - 1)
                prev_y = py(i - 1)
                old_cost = dist(prev_x, prev_y, px(i), py(i))
                new_cost = dist(prev_x, prev_y, px(j), py(j))

                if j < n - 1:
                    old_cost += dist(px(j), py(j), px(j + 1), py(j + 1))
                    new_cost += dist(px(i), py(i), px(j + 1), py(j + 1))

                if new_cost < old_cost - 1e-10:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True

    return order


def _or_opt(
    points: list[RevisitPoint],
    order: list[int],
    start_x: float,
    start_y: float,
) -> list[int]:
    """Or-opt: relocate 1/2/3-node segments to better positions. First point fixed."""
    n = len(order)
    if n <= 2:
        return order

    def total_distance(seq: list[int]) -> float:
        d = 0.0
        cx, cy = start_x, start_y
        for idx in seq:
            px, py = points[idx].x, points[idx].y
            d += math.hypot(px - cx, py - cy)
            cx, cy = px, py
        return d

    for seg_len in (1, 2, 3):
        improved = True
        while improved:
            improved = False
            best_dist = total_distance(order)
            best_order = None

            for i in range(1, n - seg_len + 1):  # skip 0 to keep first fixed
                seg = order[i : i + seg_len]
                rest = order[:i] + order[i + seg_len :]

                for j in range(1, len(rest) + 1):  # skip 0 to keep first fixed
                    candidate = rest[:j] + seg + rest[j:]
                    d = total_distance(candidate)
                    if d < best_dist - 1e-10:
                        best_dist = d
                        best_order = candidate

            if best_order is not None:
                order = best_order
                improved = True

    return order


def _plan_route(
    points: list[RevisitPoint],
    start_x: float,
    start_y: float,
    *,
    or_opt: bool = False,
) -> list[RouteStep]:
    """Nearest-neighbor route with 2-opt improvement. Optionally add or-opt."""
    # Phase 1: Nearest-neighbor ordering
    remaining = list(range(len(points)))
    order: list[int] = []
    cur_x, cur_y = start_x, start_y

    while remaining:
        best_idx = -1
        best_dist = float("inf")
        for idx in remaining:
            p = points[idx]
            dist = math.hypot(p.x - cur_x, p.y - cur_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        remaining.remove(best_idx)
        order.append(best_idx)
        cur_x, cur_y = points[best_idx].x, points[best_idx].y

    # Phase 2: Route improvement (first point fixed)
    order = _two_opt(points, order, start_x, start_y)
    if or_opt:
        order = _or_opt(points, order, start_x, start_y)

    # Phase 3: Build route steps with distances
    route: list[RouteStep] = []
    cur_x, cur_y = start_x, start_y
    for idx in order:
        p = points[idx]
        dist_mm = math.hypot(p.x - cur_x, p.y - cur_y) / 1000.0
        travel_s = dist_mm / XY_MAX_SPEED_MM_S if dist_mm > 0 else 0.0
        route.append(RouteStep(point=p, original_index=idx, distance_mm=dist_mm, travel_time_s=travel_s))
        cur_x, cur_y = p.x, p.y

    return route


def _output_filename(index: int, label: str | None, mag_str: str) -> str:
    """Compute output filename for a point."""
    return f"{label}_{mag_str}x.png" if label else f"revisit_{index}_{mag_str}x.png"


def _print_route_table(route: list[RouteStep], mag_str: str | None = None) -> None:
    """Print the planned route as a formatted table."""
    header = f"{'#':<4} {'Label':<16} {'X':>10} {'Y':>10} {'Z':>10} {'Dist(mm)':>10} {'Time(s)':>9}"
    if mag_str:
        header += f"  {'File'}"
    print(header)
    sep_width = len(header) if not mag_str else 73
    print("-" * sep_width)
    total_dist = 0.0
    total_time = 0.0
    for i, step in enumerate(route):
        p = step.point
        dist_str = f"{step.distance_mm:.2f}" if i > 0 else "--"
        time_str = f"{step.travel_time_s:.2f}" if i > 0 else "--"
        label_str = p.label or ""
        line = f"{i + 1:<4} {label_str:<16} {p.x:>10.0f} {p.y:>10.0f} {p.z:>10.0f} {dist_str:>10} {time_str:>9}"
        if mag_str:
            line += f"  {_output_filename(i, p.label, mag_str)}"
        print(line)
        total_dist += step.distance_mm
        total_time += step.travel_time_s
    print("-" * sep_width)
    print(f"{'':38} Total travel: {total_dist:.2f} mm, ~{total_time:.1f}s (XY only)")


def _per_chip_path(output: str, label: str | None, filename: str) -> str | None:
    """Compute per-chip output path from label's chip index.

    Parses chip index from label format ``rank{NN}_c{N}_{frame}_d{det}``.
    Returns ``{output_parent}/chip_{N}/{output_basename}/{filename}``
    or None if the label doesn't contain a chip index.
    """
    if label is None:
        return None
    m = re.search(r"_c(\d+)_", label)
    if m is None:
        return None
    chip_idx = int(m.group(1))
    base_dir = os.path.dirname(output)
    revisit_subdir = os.path.basename(output)
    return os.path.join(base_dir, f"chip_{chip_idx}", revisit_subdir, filename)


def run(
    scope: Microscope,
    *,
    output: str,
    points: list[RevisitPoint],
    objective_mag: str,
    material: str | None = None,
    z_speed: float | None = None,
    z_range: float | None = None,
    settle: float = 0,
    exposure_ms: float | None = None,
    gain: float | None = None,
    white_balance: GainRGB = DEFAULT_WB,
    quiet: bool = False,
    or_opt: bool = False,
    per_chip: bool = False,
) -> None:
    """Run revisit loop: move, focus-scan, save best frame at each point.

    Capture settings resolve as explicit gain/exposure_ms args > the
    material preset's per-mag revisit_capture entry > FC_DEFAULTS
    (applied inside focus_and_capture).
    """

    def vprint(*a, **kw):
        if not quiet:
            print(*a, **kw)

    # Switch objective
    if scope.switch_objective_mag(objective_mag):
        vprint(f"Switched objective to {scope.objective_mag}x")
    else:
        vprint(f"Objective: already at {scope.objective_mag}x")

    mag = scope.objective_mag

    # Per-material revisit capture overrides (explicit args win)
    if material is not None and (gain is None or exposure_ms is None):
        # Local import: keeps the heavy segmentation module (matplotlib,
        # scipy) out of revisit startup when no material is involved.
        from flakefinder.segmentation import DetectorConfig

        cap = DetectorConfig.from_material(material).revisit_capture.get(mag)
        if cap is not None:
            if gain is None:
                gain = cap.gain
            if exposure_ms is None:
                exposure_ms = cap.exposure_ms
            vprint(f"Capture: gain={gain:g}, exposure={exposure_ms:g}ms [material {material}]")

    # Lighting and camera
    scope.light_on()
    camera = scope.camera
    camera.trigger_mode = 0  # CONTINUOUS
    camera.binning = 2  # 3x3 binning
    camera.gain_rgb = white_balance

    # Build microscope metadata
    micro_meta = build_microscope_meta(scope)

    # Read current position for route planning
    cur_x, cur_y = scope.stage.position_um
    vprint(f"Starting position: X={cur_x:.0f}, Y={cur_y:.0f} um")

    # Plan route
    mag_str = f"{mag:g}"
    route = _plan_route(points, cur_x, cur_y, or_opt=or_opt)
    vprint()
    _print_route_table(route, mag_str)
    vprint()

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Background save worker — overlaps PNG encode with next point's move.
    # Potential optimization: cv2.imwrite with low compression level would be
    # faster than PIL, but needs BGR conversion since images are stored as RGB.
    save_q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue()

    def save_worker():
        while True:
            item = save_q.get()
            if item is None:
                break
            path, img = item
            PILImage.fromarray(img).save(path)

    saver = threading.Thread(target=save_worker, daemon=True)
    saver.start()

    # Execution loop
    results: list[dict] = []
    t_total_start = time.perf_counter()

    for i, step in enumerate(route):
        p = step.point
        label_str = p.label or f"#{i}"
        vprint(f"[{i + 1}/{len(route)}] {label_str}: X={p.x:.0f}, Y={p.y:.0f}, Z={p.z:.0f}")

        # Phase 1: Parallel move to XYZ at max speed
        t_move_start = time.perf_counter()
        scope.z.set_velocity_um_s(scope.z.max_velocity_um_s)
        hx, hy = scope.stage.move_to_async(p.x, p.y)
        scope.z.move_to_corrected(p.z)  # blocking Z while XY runs async
        wait_all([hx, hy])
        t_move_end = time.perf_counter()

        # Settle after move
        if settle > 0:
            time.sleep(settle)

        # Phase 2: Focus scan + capture best frame
        t_af_start = time.perf_counter()
        fc = focus_and_capture(
            scope,
            z_center_um=p.z,
            z_range_um=z_range,
            z_speed_um_s=z_speed,
            exposure_ms=exposure_ms,
            gain=gain,
        )
        t_af_end = time.perf_counter()

        # Phase 3: Enqueue save (runs in background, overlaps with next move)
        filename = _output_filename(i, p.label, mag_str)
        if per_chip:
            filepath = _per_chip_path(output, p.label, filename)
            if filepath is None:
                filepath = os.path.join(output, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else:
            filepath = os.path.join(output, filename)
        save_q.put((filepath, fc.image))

        t_point_end = time.perf_counter()

        # Phase durations
        move_s = t_move_end - t_move_start
        af_s = t_af_end - t_af_start
        total_s = t_point_end - t_move_start
        z_adj = fc.z_um - p.z

        # Verbose per-point output
        vprint(
            f"  Move: {move_s * 1000:.0f}ms | AF: {af_s * 1000:.0f}ms ({fc.frame_count} frames) | Total: {total_s:.1f}s"
        )
        fq_str = f" [{fc.focus_quality.value}]" if fc.focus_quality.value != "ok" else ""
        vprint(f"  Z: {p.z:.1f} -> {fc.z_um:.1f} ({z_adj:+.1f}), sharpness={fc.sharpness:.1f}{fq_str}")
        vprint(f"  Saving: {filename}")

        results.append(
            {
                "order": i,
                "label": p.label,
                "original_index": step.original_index,
                "x_um": p.x,
                "y_um": p.y,
                "z_initial_um": p.z,
                "z_focused_um": fc.z_um,
                "z_adjustment_um": round(z_adj, 2),
                "sharpness": fc.sharpness,
                "best_sharpness_laplacian": fc.best_sharpness_laplacian,
                "monotonicity": fc.monotonicity,
                "focus_quality": fc.focus_quality.value,
                "frame_count": fc.frame_count,
                "timing_s": {
                    "move": round(move_s, 3),
                    "autofocus": round(af_s, 3),
                    "af_position": round(fc.timing.position_s, 3),
                    "af_set_speed": round(fc.timing.set_speed_s, 3),
                    "af_pre_scan": round(fc.timing.pre_scan_s, 3),
                    "af_scan": round(fc.timing.scan_s, 3),
                    "af_sharpness_tail": round(fc.timing.sharpness_tail_s, 3),
                    "af_restore_speed": round(fc.timing.restore_speed_s, 3),
                    "total": round(total_s, 3),
                },
                "image": filename,
                "sharpness_curve": fc.sharpness_curve,
            }
        )

    # Drain save worker before writing metadata
    t_save_drain_start = time.perf_counter()
    save_q.put(None)
    saver.join()
    save_tail_s = time.perf_counter() - t_save_drain_start

    t_total_end = time.perf_counter()
    total_elapsed = t_total_end - t_total_start

    # Aggregate timing
    move_total = sum(r["timing_s"]["move"] for r in results)
    af_total = sum(r["timing_s"]["autofocus"] for r in results)

    # Save metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "command": sys.argv,
        "total_elapsed_s": round(total_elapsed, 2),
        "point_count": len(route),
        "objective_mag": mag,
        "z_range": z_range,
        "aggregate_timing_s": {
            "move": round(move_total, 2),
            "autofocus": round(af_total, 2),
            "save_tail": round(save_tail_s, 2),
            "total": round(total_elapsed, 2),
        },
        **micro_meta,
        "points": results,
    }

    meta_path = os.path.join(output, "revisit_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Summary (always printed)
    n = len(route)
    n_bad = sum(1 for r in results if r["focus_quality"] != "ok")
    if quiet:
        fq_str = f", {n_bad} bad/borderline" if n_bad else ""
        print(f"Revisited {n} points in {total_elapsed:.1f}s{fq_str} -> {output}/")
    else:
        print()
        print("=" * 50)
        print("REVISIT SUMMARY")
        print(f"  Points: {n}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"  Breakdown: move={move_total:.1f}s, AF={af_total:.1f}s (avg {af_total / n:.1f}s)")
        print(f"  Output: {output}/")
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revisit stage points with autofocus and capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Revisit from JSON file
  uv run python commands/revisit.py -o scans/revisit_01 \\
      --points points.json --objective-mag 20x

  # Revisit individual points
  uv run python commands/revisit.py -o scans/revisit_01 \\
      --point 50000,40000,24700,flake_A --point 51000,41000,24750 \\
      --objective-mag 50x

  # Dry run (print route only)
  uv run python commands/revisit.py -o scans/revisit_01 \\
      --points points.json --objective-mag 20x --dry-run
""",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")

    # Points (mutually exclusive)
    pts_group = parser.add_mutually_exclusive_group(required=True)
    pts_group.add_argument(
        "--points",
        type=Path,
        help="Revisit JSON file with {objective_mag, points: [{x, y, z, label?}, ...]}",
    )
    pts_group.add_argument(
        "--point",
        action="append",
        dest="point_args",
        metavar="X,Y,Z[,LABEL]",
        help="Individual point (repeatable)",
    )

    # Optics
    optics_group = parser.add_argument_group("Optics")
    optics_group.add_argument(
        "--objective-mag",
        type=str,
        default=None,
        metavar="MAG",
        help="Objective magnification (e.g. 20x, 50x). Read from --points file if not specified.",
    )

    # Autofocus
    af_group = parser.add_argument_group("Autofocus")
    af_group.add_argument("--z-speed", type=float, default=None, help="Z speed in um/s (default: from objective)")
    af_group.add_argument(
        "--z-range", type=float, default=None, help="AF search range in um (default: auto from objective)"
    )
    af_group.add_argument(
        "--settle", type=float, default=0, help="Settle time in seconds after XY move, before AF (default: 0)"
    )

    # Camera
    cam_group = parser.add_argument_group("Camera")
    cam_group.add_argument(
        "--exposure-ms", type=float, default=None, help="Exposure time in ms (default: from objective)"
    )
    cam_group.add_argument("--gain", type=float, default=None, help="Camera gain (default: unchanged)")
    cam_group.add_argument(
        "--white-balance", type=parse_white_balance, default="2.51,1.02,1.41", help="White balance as B,G,R gains"
    )

    # Output control
    out_group = parser.add_argument_group("Output control")
    out_group.add_argument("--dry-run", action="store_true", help="Print route plan and exit (no hardware)")
    out_group.add_argument("--clean", action="store_true", help="Remove output directory before starting")
    out_group.add_argument("-q", "--quiet", action="store_true", help="Reduced output")
    out_group.add_argument(
        "--per-chip",
        action="store_true",
        help="Distribute images into per-chip directories (parse chip index from label _c{N}_ pattern)",
    )

    # Route optimization
    route_group = parser.add_argument_group("Route optimization")
    route_group.add_argument(
        "--or-opt", action="store_true", help="Additional or-opt node/segment relocation after 2-opt"
    )

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # Parse points and resolve objective mag
    try:
        if args.points is not None:
            if args.objective_mag is not None:
                print("Error: --objective-mag cannot be used with --points (mag comes from the file)")
                return 1
            revisit_file = _parse_points_file(args.points)
            points = revisit_file.points
            objective_mag = revisit_file.objective_mag
            material = revisit_file.material
        else:
            points = _parse_point_args(args.point_args)
            if args.objective_mag is None:
                print("Error: --objective-mag is required when using --point")
                return 1
            objective_mag = args.objective_mag
            material = None
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1

    print(f"Loaded {len(points)} points (objective: {objective_mag})")

    # Dry run
    if args.dry_run:
        mag_str = objective_mag.lower().rstrip("x")
        route = _plan_route(points, 0.0, 0.0, or_opt=args.or_opt)
        print()
        print("Route plan (from origin):")
        _print_route_table(route, mag_str)
        print()
        print("(dry run -- exiting)")
        return 0

    # Validate/clean output directory (skip exists check in per-chip mode)
    if not args.per_chip and os.path.exists(args.output):
        if args.clean:
            shutil.rmtree(args.output)
            print(f"Removed existing: {args.output}")
        else:
            print(f"Error: Directory '{args.output}' already exists (use --clean to remove)")
            return 1

    # Connect to hardware
    try:
        with Microscope() as scope:
            run(
                scope,
                output=args.output,
                points=points,
                objective_mag=objective_mag,
                material=material,
                z_speed=args.z_speed,
                z_range=args.z_range,
                settle=args.settle,
                exposure_ms=args.exposure_ms,
                gain=args.gain,
                white_balance=args.white_balance,
                quiet=args.quiet,
                or_opt=args.or_opt,
                per_chip=args.per_chip,
            )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
