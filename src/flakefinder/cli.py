"""FlakeFinder CLI - Microscope control and flake detection."""

from __future__ import annotations

import argparse
import sys


def cmd_connect(args: argparse.Namespace) -> int:
    """Attempt to connect to the microscope and report status."""
    print("FlakeFinder - Microscope Connection Test")
    print("=" * 40)

    try:
        # This will fail without the hardware/DLLs, which is expected
        from .driver.microscope import Microscope

        config_dir = args.config_dir or "./"
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
    print("Phase 0: Bootstrap")
    print("  - Minimal microscope driver (from 2DMatGMM-System)")
    print("  - Connection test only")
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
