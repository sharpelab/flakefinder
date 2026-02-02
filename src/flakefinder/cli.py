"""FlakeFinder CLI - Microscope control and flake detection."""

from __future__ import annotations

import argparse
import sys


def cmd_connect(args: argparse.Namespace) -> int:
    """Attempt to connect to the microscope and report status."""
    print("FlakeFinder - Microscope Connection Test")
    print("=" * 40)

    if args.legacy:
        return _connect_legacy(args)
    else:
        return _connect_new(args)


def _connect_new(args: argparse.Namespace) -> int:
    """Connect using the new leica API."""
    try:
        from .leica import LeicaConnection, TID, get_interface, IID, print_unit_tree

        config_dir = args.config_dir
        print(f"API: flakefinder.leica (new)")
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


def _connect_legacy(args: argparse.Namespace) -> int:
    """Connect using the legacy driver API."""
    try:
        from .driver.microscope import Microscope

        config_dir = args.config_dir or "./"
        print(f"API: flakefinder.driver (legacy)")
        print(f"Config directory: {config_dir}")
        print("Attempting to initialize microscope...")

        scope = Microscope(config_dir=config_dir)
        print("Connected successfully!")
        print()
        print("Subsystems:")
        print(f"  Stage: {scope.stage}")
        print(f"  Lamp: {scope.lamp}")
        print(f"  Camera: {scope.camera}")
        print(f"  Nosepiece: {scope.nosepiece}")
        print(f"  Z-Drive: {scope.zDrive}")
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
    print("APIs:")
    print("  - flakefinder.driver: Legacy microscope driver")
    print("  - flakefinder.leica: New async-capable API (in progress)")
    print()
    print("See: https://github.com/sharpelab/flakefinder")
    return 0


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
        "--legacy",
        action="store_true",
        help="Use legacy driver API instead of new leica API",
    )
    connect_parser.add_argument(
        "--tree",
        action="store_true",
        help="Print full unit tree (new API only)",
    )
    connect_parser.set_defaults(func=cmd_connect)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show version and configuration info",
    )
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
