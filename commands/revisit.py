"""Revisit stage points with autofocus and capture.

Takes a list of (x, y, z) points, plans a time-efficient route, moves to
each, autofocuses using z as initial guess, and captures the result image.

Usage:
    uv run python commands/revisit.py -o scans/revisit_01 \\
        --points points.json --objective-mag 20x

    uv run python commands/revisit.py -o scans/revisit_01 \\
        --point 50000,40000,24700,flake_A --point 51000,41000,24750 \\
        --objective-mag 20x

    uv run python commands/revisit.py -o scans/revisit_01 \\
        --points points.json --objective-mag 50x --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from PIL import Image as PILImage

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


def _parse_points_file(points_file: Path) -> list[RevisitPoint]:
    """Parse points from a JSON file."""
    with open(points_file) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {points_file}, got {type(data).__name__}")
    points: list[RevisitPoint] = []
    for obj in data:
        points.append(
            RevisitPoint(
                x=float(obj["x"]),
                y=float(obj["y"]),
                z=float(obj["z"]),
                label=obj.get("label"),
            )
        )
    return points


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


def _plan_route(
    points: list[RevisitPoint],
    start_x: float,
    start_y: float,
) -> list[RouteStep]:
    """Greedy nearest-neighbor route from start position."""
    remaining = list(range(len(points)))
    route: list[RouteStep] = []
    cur_x, cur_y = start_x, start_y

    while remaining:
        best_idx = -1
        best_dist = float("inf")
        for idx in remaining:
            p = points[idx]
            dist = math.hypot(p.x - cur_x, p.y - cur_y) / 1000.0  # mm
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        remaining.remove(best_idx)
        p = points[best_idx]
        travel_s = best_dist / XY_MAX_SPEED_MM_S if best_dist > 0 else 0.0
        route.append(RouteStep(point=p, original_index=best_idx, distance_mm=best_dist, travel_time_s=travel_s))
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


def run(
    scope: Microscope,
    *,
    output: str,
    points: list[RevisitPoint],
    objective_mag: str,
    z_speed: float = 1250,
    z_range: float | None = None,
    exposure_ms: float = 1.0,
    gain: float | None = None,
    white_balance: GainRGB = DEFAULT_WB,
    quiet: bool = False,
) -> None:
    """Run revisit loop: move, focus-scan, save best frame at each point."""
    from flakefinder.leica import wait_all

    def vprint(*a, **kw):
        if not quiet:
            print(*a, **kw)

    # Switch objective
    if scope.switch_objective_mag(objective_mag):
        vprint(f"Switched objective to {scope.objective_mag}x")
    else:
        vprint(f"Objective: already at {scope.objective_mag}x")

    mag = scope.objective_mag

    # Lighting and camera
    scope.light_on()
    camera = scope.camera
    camera.trigger_mode = 0  # CONTINUOUS
    camera.binning = 2  # 3x3 binning
    camera.exposure_time = exposure_ms / 1000.0
    if gain is not None:
        camera.gain = gain
    camera.gain_rgb = white_balance

    # Build microscope metadata
    micro_meta = build_microscope_meta(scope)

    # Read current position for route planning
    cur_x, cur_y = scope.stage.position_um
    vprint(f"Starting position: X={cur_x:.0f}, Y={cur_y:.0f} um")

    # Plan route
    mag_str = f"{mag:g}"
    route = _plan_route(points, cur_x, cur_y)
    vprint()
    _print_route_table(route, mag_str)
    vprint()

    # Create output directory
    os.makedirs(output)

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

        # Phase 2: Focus scan + capture best frame
        t_af_start = time.perf_counter()
        fc = focus_and_capture(
            scope,
            z_center_um=p.z,
            z_range_um=z_range,
            z_speed_um_s=z_speed,
        )
        t_af_end = time.perf_counter()

        # Phase 3: Save best frame
        t_save_start = time.perf_counter()
        filename = _output_filename(i, p.label, mag_str)
        filepath = os.path.join(output, filename)
        PILImage.fromarray(fc.image).save(filepath)
        t_save_end = time.perf_counter()

        t_point_end = time.perf_counter()

        # Phase durations
        move_s = t_move_end - t_move_start
        af_s = t_af_end - t_af_start
        save_s = t_save_end - t_save_start
        total_s = t_point_end - t_move_start
        z_adj = fc.z_um - p.z

        # Verbose per-point output
        vprint(
            f"  Move: {move_s * 1000:.0f}ms | AF: {af_s * 1000:.0f}ms "
            f"({fc.frame_count} frames) | Save: {save_s * 1000:.0f}ms | "
            f"Total: {total_s:.1f}s"
        )
        vprint(f"  Z: {p.z:.1f} -> {fc.z_um:.1f} ({z_adj:+.1f}), sharpness={fc.sharpness:.1f}")
        vprint(f"  Saved: {filename}")

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
                "frame_count": fc.frame_count,
                "timing_s": {
                    "move": round(move_s, 3),
                    "autofocus": round(af_s, 3),
                    "af_position": round(fc.timing.position_s, 3),
                    "af_scan": round(fc.timing.scan_s, 3),
                    "save": round(save_s, 3),
                    "total": round(total_s, 3),
                },
                "image": filename,
            }
        )

    t_total_end = time.perf_counter()
    total_elapsed = t_total_end - t_total_start

    # Aggregate timing
    move_total = sum(r["timing_s"]["move"] for r in results)
    af_total = sum(r["timing_s"]["autofocus"] for r in results)
    save_total = sum(r["timing_s"]["save"] for r in results)

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
            "save": round(save_total, 2),
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
    if quiet:
        print(f"Revisited {n} points in {total_elapsed:.1f}s -> {output}/")
    else:
        print()
        print("=" * 50)
        print("REVISIT SUMMARY")
        print(f"  Points: {n}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(
            f"  Breakdown: move={move_total:.1f}s, AF={af_total:.1f}s (avg {af_total / n:.1f}s), save={save_total:.1f}s"
        )
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
        help="JSON file with [{x, y, z, label?}, ...]",
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
        required=True,
        metavar="MAG",
        help="Objective magnification (e.g. 20x, 50x)",
    )

    # Autofocus
    af_group = parser.add_argument_group("Autofocus")
    af_group.add_argument("--z-speed", type=float, default=1250, help="Z speed in um/s (default: 1250)")
    af_group.add_argument(
        "--z-range", type=float, default=None, help="AF search range in um (default: auto from objective)"
    )

    # Camera
    cam_group = parser.add_argument_group("Camera")
    cam_group.add_argument("--exposure-ms", type=float, default=1.0, help="Exposure time in ms (default: 1.0)")
    cam_group.add_argument("--gain", type=float, default=None, help="Camera gain (default: unchanged)")
    cam_group.add_argument(
        "--white-balance", type=parse_white_balance, default="2.51,1.02,1.41", help="White balance as B,G,R gains"
    )

    # Output control
    out_group = parser.add_argument_group("Output control")
    out_group.add_argument("--dry-run", action="store_true", help="Print route plan and exit (no hardware)")
    out_group.add_argument("--clean", action="store_true", help="Remove output directory before starting")
    out_group.add_argument("-q", "--quiet", action="store_true", help="Reduced output")

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # Parse points
    try:
        if args.points is not None:
            points = _parse_points_file(args.points)
        else:
            points = _parse_point_args(args.point_args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1

    print(f"Loaded {len(points)} points")

    # Dry run
    if args.dry_run:
        mag_str = args.objective_mag.lower().rstrip("x")
        route = _plan_route(points, 0.0, 0.0)
        print()
        print("Route plan (from origin):")
        _print_route_table(route, mag_str)
        print()
        print("(dry run -- exiting)")
        return 0

    # Validate/clean output directory
    if os.path.exists(args.output):
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
                objective_mag=args.objective_mag,
                z_speed=args.z_speed,
                z_range=args.z_range,
                exposure_ms=args.exposure_ms,
                gain=args.gain,
                white_balance=args.white_balance,
                quiet=args.quiet,
            )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
