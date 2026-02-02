"""FlakeFinder CLI - Microscope control and flake detection."""

from __future__ import annotations

import argparse
import sys


def cmd_connect(args: argparse.Namespace) -> int:
    """Attempt to connect to the microscope and report status."""
    print("FlakeFinder - Microscope Connection Test")
    print("=" * 40)

    try:
        from .leica import LeicaConnection, TID, print_unit_tree

        config_dir = args.config_dir
        print(f"Config directory: {config_dir or '(default)'}")
        print("Attempting to connect...")
        print()

        with LeicaConnection(config_dir) as conn:
            print(f"Connected: {conn.root.GetName()}")
            print()

            # Find key units
            units_to_find = [
                (TID.MICROSCOPE_STAGE, "Stage"),
                (TID.MICROSCOPE_ZDRIVE, "Z-Drive"),
                (TID.MICROSCOPE_NOSEPIECE, "Nosepiece"),
                (TID.MICROSCOPE_LAMP, "Lamp"),
                (TID.MICROSCOPE_IL_SHUTTER, "IL Shutter"),
            ]

            print("Units found:")
            for tid, label in units_to_find:
                unit = conn.find_unit(tid)
                if unit:
                    print(f"  {label}: {unit.GetName()}")
                else:
                    print(f"  {label}: (not found)")

            if args.tree:
                print()
                print("Unit tree:")
                print_unit_tree(conn.root)

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print()
        print("This usually means pythonnet or the Leica DLLs are not available.")
        print("See docs/setup.md for installation instructions.")
        return 1

    except Exception as e:
        print(f"Connection failed: {e}")
        print()
        print("This is expected if you're not on the microscope PC.")
        print("The driver requires the Leica hardware and DLLs to be present.")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show FlakeFinder version and configuration info."""
    print("FlakeFinder v0.1.0")
    print()
    print("API: flakefinder.leica")
    print("  - Async-capable microscope control")
    print("  - Event-based position monitoring")
    print()
    print("See: https://github.com/sharpelab/flakefinder")
    return 0


def cmd_test_axis(args: argparse.Namespace) -> int:
    """Test Axis class with Z-drive and optionally stage."""
    import time

    try:
        from .leica import LeicaConnection, TID, Axis, Stage, PositionMonitor, AxisEvents, EventQueue

        print("FlakeFinder - Axis Test")
        print("=" * 40)

        with LeicaConnection(args.config_dir) as conn:
            # Test Z-drive
            z_unit = conn.find_unit(TID.MICROSCOPE_ZDRIVE)
            if z_unit is None:
                print("Z-drive not found!")
                return 1

            z = Axis(z_unit)
            print(f"Z-Drive: {z.name}")
            print(f"  Position: {z.position_um:.2f} µm")
            print(f"  Range: {z.min_um:.2f} - {z.max_um:.2f} µm")
            print(f"  Supports async: {z.supports_async}")
            print(f"  Supports halt: {z.supports_halt}")
            print(f"  Supports velocity: {z.supports_velocity}")
            print(f"  Calibrated: {z.is_calibrated}")
            print()

            # Test Stage
            try:
                stage = Stage.from_connection(conn)
                print(f"Stage X: {stage.x.name}")
                print(f"  Position: {stage.x.position_um:.2f} µm")
                print(f"  Range: {stage.x.min_um:.2f} - {stage.x.max_um:.2f} µm")
                print(f"  Supports async: {stage.x.supports_async}")
                print()
                print(f"Stage Y: {stage.y.name}")
                print(f"  Position: {stage.y.position_um:.2f} µm")
                print(f"  Range: {stage.y.min_um:.2f} - {stage.y.max_um:.2f} µm")
                print(f"  Supports async: {stage.y.supports_async}")
                print()
            except LookupError as e:
                print(f"Stage not found: {e}")
                print()

            # Test async move if requested
            if args.test_move:
                # Safety margin from edges (µm)
                MARGIN = 1000

                # First, move Z to center (sync) for safety
                z_mid = (z.min_um + z.max_um) / 2
                print(f"Moving Z to center ({z_mid:.0f} µm) for safety...")
                z.move_to(z_mid)
                print(f"  Z now at: {z.position_um:.2f} µm")
                print()

                # Test async with X axis (safer than Z)
                if not stage.x.supports_async:
                    print("X axis doesn't support async moves")
                    return 1

                x_start = stage.x.position_um
                x_left = stage.x.min_um + MARGIN
                x_right = stage.x.max_um - MARGIN

                print("Testing async X moves...")
                print(f"  Start: {x_start:.2f} µm")
                print(f"  Left target: {x_left:.2f} µm")
                print(f"  Right target: {x_right:.2f} µm")
                print()

                if args.events:
                    # Use event-based position monitoring
                    print("  Using EVENT-BASED position monitoring (blocking on events)")
                    print()

                    events = AxisEvents(stage.x.unit)
                    eq = EventQueue()
                    converter = stage.x._converter

                    with events.subscribe_position(eq.handler):
                        # Move to left edge
                        print("  Moving to left edge (async)...")
                        handle = stage.x.move_to_async(x_left)
                        event_count = 0
                        while not handle.is_complete:
                            pos = eq.get(timeout=0.2)  # Block until event or timeout
                            if pos is not None:
                                event_count += 1
                                pos_um = converter.GetMetricsValue(pos)
                                print(f"    [event {event_count}] pos={pos_um:.0f}µm", flush=True)
                        # Drain any remaining events
                        for pos in eq.drain():
                            event_count += 1
                        print(f"  Final: {stage.x.position_um:.2f} µm ({event_count} events)")
                        print()

                        # Move to right edge
                        print("  Moving to right edge (async)...")
                        handle = stage.x.move_to_async(x_right)
                        event_count = 0
                        while not handle.is_complete:
                            pos = eq.get(timeout=0.2)
                            if pos is not None:
                                event_count += 1
                                pos_um = converter.GetMetricsValue(pos)
                                print(f"    [event {event_count}] pos={pos_um:.0f}µm", flush=True)
                        for pos in eq.drain():
                            event_count += 1
                        print(f"  Final: {stage.x.position_um:.2f} µm ({event_count} events)")
                        print()

                        # Return to start
                        print(f"  Returning to start ({x_start:.2f} µm)...")
                        handle = stage.x.move_to_async(x_start)
                        handle.wait()
                        print(f"  Final: {stage.x.position_um:.2f} µm")
                else:
                    # Use polling-based position monitoring
                    print("  Using POLLING-based position monitoring")
                    print()

                    # Move to left edge
                    print("  Moving to left edge (async)...", end="", flush=True)
                    handle = stage.x.move_to_async(x_left)
                    print(f" [raw state: {handle.state_raw}]")
                    while not handle.is_complete:
                        print(f"    pos={stage.x.position_um:.0f} state={handle.state_raw}", flush=True)
                        time.sleep(0.1)
                    print(f"  Position: {stage.x.position_um:.2f} µm")
                    print(f"  State: {handle.state.name} (raw={handle.state_raw})")
                    print()

                    # Move to right edge
                    print("  Moving to right edge (async)...", end="", flush=True)
                    handle = stage.x.move_to_async(x_right)
                    print(f" [raw state: {handle.state_raw}]")
                    while not handle.is_complete:
                        print(f"    pos={stage.x.position_um:.0f} state={handle.state_raw}", flush=True)
                        time.sleep(0.1)
                    print(f"  Position: {stage.x.position_um:.2f} µm")
                    print(f"  State: {handle.state.name} (raw={handle.state_raw})")
                    print()

                    # Return to start
                    print(f"  Returning to start ({x_start:.2f} µm)...")
                    handle = stage.x.move_to_async(x_start)
                    handle.wait()
                    print(f"  Final: {stage.x.position_um:.2f} µm")

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


def cmd_test_camera(args: argparse.Namespace) -> int:
    """Test Camera class with single-shot and optional streaming."""
    import time

    try:
        from .leica import LeicaConnection, Camera, Stage, Lamp

        print("FlakeFinder - Camera Test")
        print("=" * 40)

        with LeicaConnection(args.config_dir) as conn:
            camera = Camera.from_connection(conn)

            # Set up lamp
            try:
                lamp = Lamp.from_connection(conn)
                if args.lamp is not None:
                    lamp.intensity = args.lamp
                else:
                    lamp.full()  # Default to full
                print(f"Lamp: {lamp.name}")
                print(f"  Intensity: {lamp.intensity}/{lamp.max_intensity}")
            except LookupError:
                print("Lamp: (not found)")

            # Apply settings from CLI
            if args.exposure_ms is not None:
                camera.exposure_time = args.exposure_ms / 1000.0
            if args.gain is not None:
                camera.gain = args.gain

            print(f"Camera: {camera.name}")
            print(f"  Exposure: {camera.exposure_time * 1000:.1f}ms")
            print(f"  Gain: {camera.gain}")
            print(f"  Binning: {camera.binning} (0=1x1, 1=2x2, 2=3x3)")
            print()

            # Single-shot test
            print("Taking single image...")
            start = time.monotonic()
            image = camera.capture()
            elapsed = time.monotonic() - start
            print(f"  Captured: {image.shape} in {elapsed:.3f}s")
            print()

            # Streaming test
            if args.stream:
                duration = args.stream_duration
                print(f"Streaming for {duration}s...")

                # Get stage for position tagging if available
                try:
                    stage = Stage.from_connection(conn)
                    print(f"  Position tagging enabled (stage found)")
                except LookupError:
                    stage = None
                    print(f"  No stage found, positions will be None")

                with camera.stream(stage) as stream:
                    start = time.monotonic()
                    while (time.monotonic() - start) < duration:
                        frame = stream.get_frame(timeout=0.5)
                        if frame:
                            pos_str = (
                                f"({frame.position[0]:.0f}, {frame.position[1]:.0f})"
                                if frame.position
                                else "None"
                            )
                            print(
                                f"  Frame {frame.frame_number}: "
                                f"{frame.image.shape} @ {pos_str}"
                            )

                print()
                print(f"  Captured: {stream.frames_captured} frames")
                print(f"  Dropped: {stream.frames_dropped} frames")
                print(f"  Rate: {stream.frame_rate:.1f} fps")

            # Save image if requested
            if args.output:
                try:
                    from PIL import Image as PILImage

                    pil_image = PILImage.fromarray(image)
                    pil_image.save(args.output)
                    print(f"Saved to: {args.output}")
                except ImportError:
                    print("Install PIL to save images: pip install Pillow")

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        import traceback

        print(f"Error: {e}")
        traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for the FlakeFinder CLI."""
    parser = argparse.ArgumentParser(
        prog="flakefinder",
        description="Automated 2D material flake detection system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # connect command
    connect_parser = subparsers.add_parser(
        "connect",
        help="Test microscope connection",
    )
    connect_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    connect_parser.add_argument(
        "--tree",
        action="store_true",
        help="Print full unit tree",
    )
    connect_parser.set_defaults(func=cmd_connect)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show version and configuration info",
    )
    info_parser.set_defaults(func=cmd_info)

    # test-axis command
    axis_parser = subparsers.add_parser(
        "test-axis",
        help="Test Axis/Stage classes (Phase 2)",
    )
    axis_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    axis_parser.add_argument(
        "--test-move",
        action="store_true",
        help="Actually move the X axis to test async",
    )
    axis_parser.add_argument(
        "--events",
        action="store_true",
        help="Use event-based position monitoring (requires --test-move)",
    )
    axis_parser.set_defaults(func=cmd_test_axis)

    # test-camera command
    camera_parser = subparsers.add_parser(
        "test-camera",
        help="Test Camera class (single-shot and streaming)",
    )
    camera_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    camera_parser.add_argument(
        "--stream",
        action="store_true",
        help="Test continuous streaming",
    )
    camera_parser.add_argument(
        "--stream-duration",
        type=float,
        default=3.0,
        help="Streaming test duration in seconds (default: 3.0)",
    )
    camera_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save captured image to file (e.g., test.png)",
    )
    camera_parser.add_argument(
        "--exposure-ms",
        type=float,
        default=10.0,
        help="Exposure time in milliseconds (default: 10)",
    )
    camera_parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Camera gain (default: 1.0)",
    )
    camera_parser.add_argument(
        "--lamp",
        type=int,
        default=None,
        help="Lamp intensity (default: max)",
    )
    camera_parser.set_defaults(func=cmd_test_camera)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
