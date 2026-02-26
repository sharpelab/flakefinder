#!/usr/bin/env python3
"""Simple microscope stage status and position utility.

Usage:
    uv run python commands/stage.py                          # Report current status
    uv run python commands/stage.py --x 5000 --y 10000       # Move XY absolute
    uv run python commands/stage.py --dx 100 --dy -50        # Move XY relative
    uv run python commands/stage.py --z 24500                # Move Z absolute
    uv run python commands/stage.py --shutter open           # Open shutter
    uv run python commands/stage.py --lamp 50                # Set lamp to 50%
    uv run python commands/stage.py --objective-mag 5x       # Switch to 5x objective
    uv run python commands/stage.py --objective-pos 3        # Switch to position 3
    uv run python commands/stage.py --park                   # Park microscope
"""

import argparse
import sys

from flakefinder.cli_utils import park_microscope, report_status
from flakefinder.leica import Microscope, wait_all

# =============================================================================
# Axis Conventions
# =============================================================================
# Z AXIS: +Z = CLOSER to sample (higher values = more crash risk)
#         To retract/safe: DECREASE Z
#         To approach: INCREASE Z
# =============================================================================


def change_objective_mag(scope: Microscope, mag: str) -> None:
    """Change objective by magnification string with verbose logging.

    Args:
        scope: Microscope instance.
        mag: Magnification string (e.g. '20x', '5', '2.5').
    """
    nosepiece = scope.nosepiece
    target_pos = nosepiece.parse_magnification(mag)
    _change_objective_pos(scope, target_pos)


def change_objective_pos(scope: Microscope, pos: int) -> None:
    """Change objective by turret position with verbose logging.

    Args:
        scope: Microscope instance.
        pos: Turret position (1-6).
    """
    nosepiece = scope.nosepiece
    target_pos = nosepiece.validate_position(pos)
    _change_objective_pos(scope, target_pos)


def _change_objective_pos(scope: Microscope, target_pos: int) -> None:
    """Change objective to validated turret position with verbose logging.

    Future: Add parfocal compensation once offsets are calibrated.

    Args:
        scope: Microscope instance.
        target_pos: Validated turret position.
    """
    nosepiece = scope.nosepiece

    target_mag = nosepiece.magnifications.get(target_pos)

    if not scope.switch_objective_pos(target_pos):
        print(f"Objective: already at position {target_pos} ({target_mag}x)")
        return

    z_after = scope.z.position_um
    print(f"Objective: switched to {target_mag}x (pos {target_pos})")
    print(f"  Z after: {z_after:.1f} µm")


def run(
    scope: Microscope,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
    shutter: str | None = None,
    lamp: float | None = None,
    aperture: int | None = None,
    objective_mag: str | None = None,
    objective_pos: int | None = None,
    z_speed: float | None = None,
    park: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> None:
    """Microscope stage operations.

    Raises:
        ValueError: On invalid inputs or operation failure.
    """
    if objective_mag is not None and objective_pos is not None:
        raise ValueError("objective_mag and objective_pos are mutually exclusive")

    if park:
        park_microscope(scope)
        return

    if shutter is not None:
        before = "open" if scope.shutter.is_open else "closed"
        if shutter == "open":
            scope.shutter.open()
        else:
            scope.shutter.close()
        after = "open" if scope.shutter.is_open else "closed"
        print(f"Shutter: {before} -> {after}")

    if lamp is not None:
        before_pct = scope.lamp.intensity_pct
        scope.lamp.intensity_pct = lamp
        after_pct = scope.lamp.intensity_pct
        print(f"Lamp: {before_pct:.0f}% -> {after_pct:.0f}%")

    if aperture is not None:
        before_val = scope.aperture.value
        scope.aperture.value = aperture
        after_val = scope.aperture.value
        print(f"Aperture: {before_val} -> {after_val}")

    if x is not None or y is not None:
        x_before, y_before = scope.stage.position_um
        x_target = x if x is not None else x_before
        y_target = y if y is not None else y_before
        print(f"XY: ({x_before:.1f}, {y_before:.1f}) -> ({x_target:.1f}, {y_target:.1f}) µm")
        hx, hy = scope.stage.move_to_async(x_target, y_target)
        wait_all([hx, hy])
        x_after, y_after = scope.stage.position_um
        print(f"XY: done at ({x_after:.1f}, {y_after:.1f}) µm")

    if dx is not None or dy is not None:
        x_before, y_before = scope.stage.position_um
        dx_val = dx if dx is not None else 0
        dy_val = dy if dy is not None else 0
        print(f"XY: ({x_before:.1f}, {y_before:.1f}) + ({dx_val:+.1f}, {dy_val:+.1f}) µm")
        hx, hy = scope.stage.move_rel_async(dx_val, dy_val)
        wait_all([hx, hy])
        x_after, y_after = scope.stage.position_um
        print(f"XY: done at ({x_after:.1f}, {y_after:.1f}) µm")

    if z is not None:
        z_before = scope.z.position_um
        print(f"Z: {z_before:.1f} -> {z:.1f} µm")
        scope.z.move_to(z)
        print(f"Z: done at {scope.z.position_um:.1f} µm")

    if dz is not None:
        z_before = scope.z.position_um
        print(f"Z: {z_before:.1f} + {dz:+.1f} µm")
        scope.z.move_rel(dz)
        print(f"Z: done at {scope.z.position_um:.1f} µm")

    if z_speed is not None:
        before_vel = scope.z.velocity_um_s
        scope.z.set_velocity_um_s(z_speed)
        after_vel = scope.z.velocity_um_s
        print(f"Z speed: {before_vel:.0f} -> {after_vel:.0f} µm/s")

    if objective_mag is not None:
        change_objective_mag(scope, objective_mag)
    elif objective_pos is not None:
        change_objective_pos(scope, objective_pos)

    if quiet:
        sx, sy = scope.stage.position_um
        sz = scope.z.position_um
        mag = scope.nosepiece.magnification
        obj_str = f" ({mag}x)" if mag else ""
        shutter_str = "open" if scope.shutter.is_open else "closed"
        lamp_pct = scope.lamp.intensity_pct
        print(f"X={sx:.1f} Y={sy:.1f} Z={sz:.1f} µm{obj_str} shutter={shutter_str} lamp={lamp_pct:.0f}%")
    else:
        print()
        report_status(scope, verbose=verbose)


def _build_parser():
    parser = argparse.ArgumentParser(description="Microscope stage status and position utility")
    parser.add_argument("--x", type=float, help="Target X position (µm)")
    parser.add_argument("--y", type=float, help="Target Y position (µm)")
    parser.add_argument("--z", type=float, help="Target Z position (µm)")
    parser.add_argument("--dx", type=float, help="Relative X move (µm)")
    parser.add_argument("--dy", type=float, help="Relative Y move (µm)")
    parser.add_argument("--dz", type=float, help="Relative Z move (µm)")
    parser.add_argument("--shutter", choices=["open", "close"], help="Open or close shutter")
    parser.add_argument("--lamp", type=float, help="Set lamp intensity (0-100%%)")
    parser.add_argument("--aperture", type=int, metavar="VALUE", help="Set aperture diaphragm value")
    parser.add_argument("--objective-mag", type=str, metavar="MAG", help="Switch objective by magnification")
    parser.add_argument("--objective-pos", type=int, metavar="POS", help="Switch objective by turret position (1-6)")
    parser.add_argument("--z-speed", type=float, metavar="UM_S", help="Set Z velocity (µm/s)")
    parser.add_argument("--park", action="store_true", help="Park microscope in safe idle state")
    parser.add_argument("-q", "--quiet", action="store_true", help="Print only single-line position summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show velocity limits and conversion factors")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        with Microscope() as scope:
            run(
                scope,
                objective_mag=args.objective_mag,
                objective_pos=args.objective_pos,
                x=args.x,
                y=args.y,
                z=args.z,
                dx=args.dx,
                dy=args.dy,
                dz=args.dz,
                shutter=args.shutter,
                lamp=args.lamp,
                aperture=args.aperture,
                z_speed=args.z_speed,
                park=args.park,
                quiet=args.quiet,
                verbose=args.verbose,
            )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
