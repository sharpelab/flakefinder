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
"""

import argparse
import sys

from flakefinder.leica import LeicaConnection, Stage, ZDrive, Lamp, Nosepiece, Shutter


def report_status(conn: LeicaConnection) -> None:
    """Print current microscope status."""
    # Stage XY
    stage = Stage.from_connection(conn)
    x, y = stage.position_um
    print(f"Stage X: {x:.1f} µm ({stage.x.min_um:.0f} - {stage.x.max_um:.0f})")
    print(f"Stage Y: {y:.1f} µm ({stage.y.min_um:.0f} - {stage.y.max_um:.0f})")

    # Z axis
    z = ZDrive.from_connection(conn)
    print(f"Stage Z: {z.position_um:.1f} µm ({z.min_um:.0f} - {z.max_um:.0f})")

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
    args = parser.parse_args()

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

        # Always report full status
        print()
        report_status(conn)

    return 0


if __name__ == "__main__":
    sys.exit(main())
