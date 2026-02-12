"""Chip scan analysis pipeline: rsync from microscope, analyze, show results.

Wraps the common post-scan workflow into a single command.

Usage:
    uv run python scripts/process_chip_scan.py scans/chip0_20x
    uv run python scripts/process_chip_scan.py scans/chip0_20x --sharpness --show
    uv run python scripts/process_chip_scan.py scans/chip0_20x --notes "v10 plane"
    uv run python scripts/process_chip_scan.py scans/chip0_20x --local  # skip rsync
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rsync a chip scan from the microscope and run analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scan_dir",
        type=str,
        help="Scan directory relative to flakefinder/ (e.g. scans/chip0_20x)",
    )
    parser.add_argument(
        "--sharpness",
        action="store_true",
        help="Enable sharpness computation (disabled by default for speed)",
    )
    parser.add_argument("--show", action="store_true", help="Open the analysis plot after generation")
    parser.add_argument("--local", action="store_true", help="Skip rsync (data already local)")
    # Pass-through args for analyze_chip_scan.py
    parser.add_argument("--notes", type=str, default=None, help="Label for plot title")
    parser.add_argument("--sample", type=int, default=None, help="Sharpness: process every Nth frame")
    parser.add_argument("--min-sharpness", type=float, default=None, help="Sharpness threshold for flagging")
    parser.add_argument("--spatial-z", action="store_true", help="Include spatial Z error map panel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Full analyze_chip_scan output (default: quiet)")
    args = parser.parse_args()

    local_scan_dir = (REPO_DIR / args.scan_dir).resolve()

    # --- Step 1: rsync from microscope ---
    if not args.local:
        remote = f"sharpelab-microscope:flakefinder/{args.scan_dir}/"
        local = f"{local_scan_dir}/"
        print(f"Syncing {remote} -> {local}")
        result = subprocess.run(["rsync", "-a", "--quiet", remote, local])
        if result.returncode != 0:
            print(f"Error: rsync failed (exit {result.returncode})")
            return 1

    if not local_scan_dir.is_dir():
        print(f"Error: Not a directory: {local_scan_dir}")
        return 1

    # --- Step 2: run analyze_chip_scan.py ---
    cmd = [sys.executable, str(REPO_DIR / "scripts" / "analyze_chip_scan.py"), str(local_scan_dir)]
    if not args.verbose:
        cmd.append("--quiet")
    if not args.sharpness:
        cmd.append("--no-sharpness")
    if args.notes is not None:
        cmd.extend(["--notes", args.notes])
    if args.sample is not None:
        cmd.extend(["--sample", str(args.sample)])
    if args.min_sharpness is not None:
        cmd.extend(["--min-sharpness", str(args.min_sharpness)])
    if args.spatial_z:
        cmd.append("--spatial-z")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: analyze_chip_scan failed (exit {result.returncode})")
        return 1

    # --- Step 3: determine plot path and optionally show ---
    plot_path = local_scan_dir / "scan_analysis.png"
    if not plot_path.exists():
        print("Warning: expected plot not found at", plot_path)
        return 0

    if args.show:
        subprocess.run(["present", str(plot_path)])

    # --- Step 4: print plot path for easy scan-nb --attach ---
    print(f"\n{plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
