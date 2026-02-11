#!/usr/bin/env python3
"""Simple microscope stage status and position utility.

Usage:
    python stage_util.py                      # Report current status
    python stage_util.py --x 5000 --y 10000   # Move XY absolute
    python stage_util.py --dx 100 --dy -50    # Move XY relative
    python stage_util.py --z 24500            # Move Z absolute
    python stage_util.py --dz -100            # Move Z relative
    python stage_util.py --shutter open       # Open shutter
    python stage_util.py --shutter close      # Close shutter
    python stage_util.py --lamp 50            # Set lamp to 50%
    python stage_util.py --aperture 500       # Set aperture diaphragm
    python stage_util.py --objective-mag 5x    # Switch to 5x objective (SDK handles z-hop)
    python stage_util.py --objective-pos 3    # Switch to position 3
    python stage_util.py --z-speed 5000       # Set Z velocity to 5000 µm/s
    python stage_util.py --park               # Park microscope (safe idle state)
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

# =============================================================================
# Objective Safety Configuration
# =============================================================================

# Working distances in µm (conservative estimates for Sharpe Lab DM6M)
# These are approximate - actual values depend on specific objective models.
# Position -> working distance mapping (1-indexed positions)
WORKING_DISTANCES_UM: dict[int, float] = {
    1: 12700,  # 5x N PLAN - 12.7mm working distance
    2: 11000,  # 10x - ~11mm (typical)
    3: 1900,  # 20x - ~1.9mm (typical)
    4: 380,  # 50x - ~0.38mm (typical long WD)
    5: 210,  # 150x - ~0.21mm (short WD, highest risk)
    6: 15000,  # 2.5x - ~15mm (very safe)
}

# Parfocal offsets in µm (to be calibrated)
# Offset = Z_focused(this_obj) - Z_focused(reference_obj)
# Positive means this objective focuses at higher Z than reference.
# All zeros until calibrated - set reference objective to position 1 (5x).
PARFOCAL_OFFSETS_UM: dict[int, float] = {
    1: 0,  # 5x - reference
    2: 0,  # 10x
    3: 0,  # 20x
    4: 0,  # 50x
    5: 0,  # 150x
    6: 0,  # 2.5x
}

# Safety margin added to Z retraction (µm)
Z_SAFETY_MARGIN_UM = 500


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

    # Future: parfocal compensation would go here once calibrated
    # current_offset = PARFOCAL_OFFSETS_UM.get(current_position, 0)
    # target_offset = PARFOCAL_OFFSETS_UM.get(target_position, 0)
    # ...


def main() -> int:
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
    parser.add_argument(
        "--objective-mag",
        type=str,
        metavar="MAG",
        help="Switch objective by magnification (e.g., 5, 5x, 20, 2.5)",
    )
    parser.add_argument("--objective-pos", type=int, metavar="POS", help="Switch objective by turret position (1-6)")
    parser.add_argument("--z-speed", type=float, metavar="UM_S", help="Set Z velocity (µm/s)")
    parser.add_argument("--park", action="store_true", help="Park microscope in safe idle state")
    parser.add_argument("-q", "--quiet", action="store_true", help="Print only single-line position summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show velocity limits and conversion factors")
    args = parser.parse_args()

    # Validate objective args are mutually exclusive
    if args.objective_mag is not None and args.objective_pos is not None:
        print("Error: --objective-mag and --objective-pos are mutually exclusive")
        return 1

    # Handle --park specially (ignores other position args)
    if args.park:
        with Microscope() as scope:
            park_microscope(scope)
        return 0

    with Microscope() as scope:
        # Shutter control
        if args.shutter is not None:
            before = "open" if scope.shutter.is_open else "closed"
            if args.shutter == "open":
                scope.shutter.open()
            else:
                scope.shutter.close()
            after = "open" if scope.shutter.is_open else "closed"
            print(f"Shutter: {before} -> {after}")

        # Lamp control
        if args.lamp is not None:
            before_pct = scope.lamp.intensity_pct
            scope.lamp.intensity_pct = args.lamp
            after_pct = scope.lamp.intensity_pct
            print(f"Lamp: {before_pct:.0f}% -> {after_pct:.0f}%")

        # Aperture control
        if args.aperture is not None:
            before = scope.aperture.value
            scope.aperture.value = args.aperture
            after = scope.aperture.value
            print(f"Aperture: {before} -> {after}")

        # Absolute XY move
        if args.x is not None or args.y is not None:
            x_before, y_before = scope.stage.position_um
            x_target = args.x if args.x is not None else x_before
            y_target = args.y if args.y is not None else y_before

            print(f"XY: ({x_before:.1f}, {y_before:.1f}) -> ({x_target:.1f}, {y_target:.1f}) µm")

            hx, hy = scope.stage.move_to_async(x_target, y_target)
            wait_all([hx, hy])

            x_after, y_after = scope.stage.position_um
            print(f"XY: done at ({x_after:.1f}, {y_after:.1f}) µm")

        # Relative XY move
        if args.dx is not None or args.dy is not None:
            x_before, y_before = scope.stage.position_um
            dx = args.dx if args.dx is not None else 0
            dy = args.dy if args.dy is not None else 0

            print(f"XY: ({x_before:.1f}, {y_before:.1f}) + ({dx:+.1f}, {dy:+.1f}) µm")

            hx, hy = scope.stage.move_rel_async(dx, dy)
            wait_all([hx, hy])

            x_after, y_after = scope.stage.position_um
            print(f"XY: done at ({x_after:.1f}, {y_after:.1f}) µm")

        # Absolute Z move
        if args.z is not None:
            z_before = scope.z.position_um
            print(f"Z: {z_before:.1f} -> {args.z:.1f} µm")
            scope.z.move_to(args.z)
            print(f"Z: done at {scope.z.position_um:.1f} µm")

        # Relative Z move
        if args.dz is not None:
            z_before = scope.z.position_um
            print(f"Z: {z_before:.1f} + {args.dz:+.1f} µm")
            scope.z.move_rel(args.dz)
            print(f"Z: done at {scope.z.position_um:.1f} µm")

        # Z speed
        if args.z_speed is not None:
            before = scope.z.velocity_um_s
            scope.z.set_velocity_um_s(args.z_speed)
            after = scope.z.velocity_um_s
            print(f"Z speed: {before:.0f} -> {after:.0f} µm/s")

        # Objective change
        if args.objective_mag is not None:
            try:
                change_objective_mag(scope, args.objective_mag)
            except ValueError as e:
                print(f"Error: {e}")
                return 1
        elif args.objective_pos is not None:
            try:
                change_objective_pos(scope, args.objective_pos)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        # Report status
        if args.quiet:
            x, y = scope.stage.position_um
            z = scope.z.position_um
            mag = scope.nosepiece.magnification
            obj_str = f" ({mag}x)" if mag else ""
            print(f"X={x:.1f} Y={y:.1f} Z={z:.1f} µm{obj_str}")
        else:
            print()
            report_status(scope, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
