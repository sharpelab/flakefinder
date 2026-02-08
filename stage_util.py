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
    python stage_util.py --lamp 50            # Set lamp intensity
    python stage_util.py --objective-mag 5x    # Switch to 5x objective (SDK handles z-hop)
    python stage_util.py --objective-pos 3    # Switch to position 3
    python stage_util.py --z-speed 5000       # Set Z velocity to 5000 µm/s
    python stage_util.py --park               # Park microscope (safe idle state)
"""

import argparse
import sys

from flakefinder.leica import LeicaConnection, Stage, ZDrive, Lamp, Nosepiece, Shutter


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
    3: 1900,   # 20x - ~1.9mm (typical)
    4: 380,    # 50x - ~0.38mm (typical long WD)
    5: 210,    # 150x - ~0.21mm (short WD, highest risk)
    6: 15000,  # 2.5x - ~15mm (very safe)
}

# Parfocal offsets in µm (to be calibrated)
# Offset = Z_focused(this_obj) - Z_focused(reference_obj)
# Positive means this objective focuses at higher Z than reference.
# All zeros until calibrated - set reference objective to position 1 (5x).
PARFOCAL_OFFSETS_UM: dict[int, float] = {
    1: 0,    # 5x - reference
    2: 0,    # 10x
    3: 0,    # 20x
    4: 0,    # 50x
    5: 0,    # 150x
    6: 0,    # 2.5x
}

# Safety margin added to Z retraction (µm)
Z_SAFETY_MARGIN_UM = 500


def change_objective(
    nosepiece: Nosepiece,
    z: ZDrive,
    target_position: int,
) -> None:
    """Change objective, trusting SDK's built-in z-hop for safety.

    The Leica SDK automatically performs a z-hop (retract, rotate, return)
    when switching objectives. We just log the operation and let the SDK
    handle safety.

    Future: Add parfocal compensation once offsets are calibrated.

    Args:
        nosepiece: Nosepiece instance.
        z: ZDrive instance.
        target_position: Target objective position (1-indexed).
    """
    current_position = nosepiece.position
    current_mag = nosepiece.magnification
    target_mag = nosepiece.magnifications.get(target_position)

    if current_position == target_position:
        print(f"Objective: already at position {target_position} ({target_mag}x)")
        return

    z_before = z.position_um

    print(f"Objective: {current_mag}x (pos {current_position}) -> {target_mag}x (pos {target_position})")
    print(f"  Z before: {z_before:.1f} µm")
    print(f"  Switching (SDK handles z-hop)...")

    nosepiece.set_position(target_position, z=z)

    z_after = z.position_um
    print(f"  Z after: {z_after:.1f} µm (delta: {z_after - z_before:+.1f} µm)")

    # Future: parfocal compensation would go here once calibrated
    # current_offset = PARFOCAL_OFFSETS_UM.get(current_position, 0)
    # target_offset = PARFOCAL_OFFSETS_UM.get(target_position, 0)
    # ...

    print(f"Objective: done at {nosepiece.magnification}x")


def park_microscope(conn: LeicaConnection) -> None:
    """Put microscope in a safe idle state.

    Operations in order:
    1. Retract Z to safe position (15000 µm)
    2. Move XY to origin (0, 0)
    3. Switch to 5x objective
    4. Turn lamp off
    5. Close shutter
    """
    print("Parking microscope...")

    stage = Stage.from_connection(conn)
    z = ZDrive.from_connection(conn)

    # 1. Z retract first (safety)
    z_target = 15000.0
    z.move_to(z_target)
    print(f"  Z -> {z.position_um:.0f} um [ok]")

    # 2. XY to origin
    hx, hy = stage.move_to_async(0.0, 0.0)
    Stage.wait_all([hx, hy])
    hx.dispose()
    hy.dispose()
    print(f"  X -> {stage.x.position_um:.0f} um [ok]")
    print(f"  Y -> {stage.y.position_um:.0f} um [ok]")

    # 3. Switch to 5x objective
    try:
        nosepiece = Nosepiece.from_connection(conn)
        # 5x is at position 1 on this microscope
        target_pos = None
        for pos, mag in nosepiece.magnifications.items():
            if mag == 5.0:
                target_pos = pos
                break
        if target_pos is not None and nosepiece.position != target_pos:
            nosepiece.set_position(target_pos, z=z)
        print(f"  Objective -> {nosepiece.magnification}x [ok]")
    except LookupError:
        print("  Objective -> (not available)")

    # 4. Turn lamp off
    try:
        lamp = Lamp.from_connection(conn)
        lamp.intensity = 0
        print("  Lamp off [ok]")
    except LookupError:
        print("  Lamp -> (not available)")

    # 5. Close shutter
    try:
        shutter = Shutter.from_connection(conn)
        shutter.close()
        print("  Shutter closed [ok]")
    except LookupError:
        print("  Shutter -> (not available)")

    print("Parked.")


def report_status(conn: LeicaConnection, verbose: bool = False) -> None:
    """Print current microscope status."""
    # Stage XY
    stage = Stage.from_connection(conn)
    x, y = stage.position_um
    print(f"Stage X: {x:.1f} µm ({stage.x.min_um:.0f} - {stage.x.max_um:.0f})")
    print(f"Stage Y: {y:.1f} µm ({stage.y.min_um:.0f} - {stage.y.max_um:.0f})")

    # Z axis
    z = ZDrive.from_connection(conn)
    print(f"Stage Z: {z.position_um:.1f} µm ({z.min_um:.0f} - {z.max_um:.0f})")

    # Velocity info (verbose mode)
    if verbose:
        print()
        print("Velocity (from SDK converter):")
        print(f"  X: {stage.x.velocity_um_s/1000:.1f} mm/s (max: {stage.x.max_velocity_um_s/1000:.1f} mm/s)")
        print(f"  Y: {stage.y.velocity_um_s/1000:.1f} mm/s (max: {stage.y.max_velocity_um_s/1000:.1f} mm/s)")
        print(f"  Z: {z.velocity_um_s/1000:.1f} mm/s (max: {z.max_velocity_um_s/1000:.1f} mm/s)")

    # Nosepiece/objective
    try:
        nosepiece = Nosepiece.from_connection(conn)
        mag = nosepiece.magnification
        if mag:
            print(f"Objective: position {nosepiece.position} ({mag}x)")
        else:
            print(f"Objective: position {nosepiece.position}")
    except LookupError:
        pass  # Nosepiece not available

    # Lamp
    try:
        lamp = Lamp.from_connection(conn)
        print(f"Lamp: {lamp.intensity}/{lamp.max_intensity}")
    except LookupError:
        pass  # Lamp not available

    # Shutter
    try:
        shutter = Shutter.from_connection(conn)
        state = "open" if shutter.is_open else "closed"
        print(f"Shutter: {state}")
    except LookupError:
        pass  # Shutter not available


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Microscope stage status and position utility"
    )
    parser.add_argument("--x", type=float, help="Target X position (µm)")
    parser.add_argument("--y", type=float, help="Target Y position (µm)")
    parser.add_argument("--z", type=float, help="Target Z position (µm)")
    parser.add_argument("--dx", type=float, help="Relative X move (µm)")
    parser.add_argument("--dy", type=float, help="Relative Y move (µm)")
    parser.add_argument("--dz", type=float, help="Relative Z move (µm)")
    parser.add_argument("--shutter", choices=["open", "close"], help="Open or close shutter")
    parser.add_argument("--lamp", type=int, help="Set lamp intensity")
    parser.add_argument("--objective-mag", type=str, metavar="MAG",
                        help="Switch objective by magnification (e.g., 5, 5x, 20, 2.5)")
    parser.add_argument("--objective-pos", type=int, metavar="POS",
                        help="Switch objective by turret position (1-6)")
    parser.add_argument("--z-speed", type=float, metavar="UM_S", help="Set Z velocity (µm/s)")
    parser.add_argument("--park", action="store_true", help="Park microscope in safe idle state")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show velocity limits and conversion factors")
    args = parser.parse_args()

    # Validate objective args are mutually exclusive
    if args.objective_mag is not None and args.objective_pos is not None:
        print("Error: --objective-mag and --objective-pos are mutually exclusive")
        return 1

    # Handle --park specially (ignores other position args)
    if args.park:
        with LeicaConnection() as conn:
            park_microscope(conn)
        return 0

    with LeicaConnection() as conn:
        stage = Stage.from_connection(conn)
        z = ZDrive.from_connection(conn)

        # Shutter control
        if args.shutter is not None:
            shutter = Shutter.from_connection(conn)
            before = "open" if shutter.is_open else "closed"
            if args.shutter == "open":
                shutter.open()
            else:
                shutter.close()
            after = "open" if shutter.is_open else "closed"
            print(f"Shutter: {before} -> {after}")

        # Lamp control
        if args.lamp is not None:
            lamp = Lamp.from_connection(conn)
            before = lamp.intensity
            lamp.intensity = args.lamp
            after = lamp.intensity
            print(f"Lamp: {before} -> {after}")

        # Absolute XY move
        if args.x is not None or args.y is not None:
            x_before, y_before = stage.position_um
            x_target = args.x if args.x is not None else x_before
            y_target = args.y if args.y is not None else y_before

            print(f"XY: ({x_before:.1f}, {y_before:.1f}) -> ({x_target:.1f}, {y_target:.1f}) µm")

            hx, hy = stage.move_to_async(x_target, y_target)
            Stage.wait_all([hx, hy])
            hx.dispose()
            hy.dispose()

            x_after, y_after = stage.position_um
            print(f"XY: done at ({x_after:.1f}, {y_after:.1f}) µm")

        # Relative XY move
        if args.dx is not None or args.dy is not None:
            x_before, y_before = stage.position_um
            dx = args.dx if args.dx is not None else 0
            dy = args.dy if args.dy is not None else 0

            print(f"XY: ({x_before:.1f}, {y_before:.1f}) + ({dx:+.1f}, {dy:+.1f}) µm")

            hx, hy = stage.move_rel_async(dx, dy)
            Stage.wait_all([hx, hy])
            hx.dispose()
            hy.dispose()

            x_after, y_after = stage.position_um
            print(f"XY: done at ({x_after:.1f}, {y_after:.1f}) µm")

        # Absolute Z move
        if args.z is not None:
            z_before = z.position_um
            print(f"Z: {z_before:.1f} -> {args.z:.1f} µm")
            z.move_to(args.z)
            print(f"Z: done at {z.position_um:.1f} µm")

        # Relative Z move
        if args.dz is not None:
            z_before = z.position_um
            print(f"Z: {z_before:.1f} + {args.dz:+.1f} µm")
            z.move_rel(args.dz)
            print(f"Z: done at {z.position_um:.1f} µm")

        # Z speed
        if args.z_speed is not None:
            before = z.velocity_um_s
            z.set_velocity_um_s(args.z_speed)
            after = z.velocity_um_s
            print(f"Z speed: {before:.0f} -> {after:.0f} µm/s")

        # Objective change
        if args.objective_mag is not None or args.objective_pos is not None:
            nosepiece = Nosepiece.from_connection(conn)
            try:
                if args.objective_mag is not None:
                    target_pos = nosepiece.parse_magnification(args.objective_mag)
                else:
                    target_pos = nosepiece.validate_position(args.objective_pos)
                change_objective(nosepiece, z, target_pos)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        # Always report full status
        print()
        report_status(conn, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
