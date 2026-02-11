"""Overview scan post-processing pipeline: rsync, stitch, detect chips, show results.

Wraps the common overview workflow into a single command.

Usage:
    uv run python scripts/process_overview.py scans/overview_5x
    uv run python scripts/process_overview.py scans/overview_5x --downsample 4 --show
    uv run python scripts/process_overview.py scans/overview_5x --local  # skip rsync
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rsync an overview scan, stitch it, and detect chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scan_dir",
        type=str,
        help="Scan directory relative to flakefinder/ (e.g. scans/overview_5x)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for stitching",
    )
    parser.add_argument("--show", action="store_true", help="Open the chip detection image")
    parser.add_argument("--local", action="store_true", help="Skip rsync (data already local)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full stitch output")
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

    # --- Step 2: stitch (capture output, suppress unless verbose/error) ---
    cmd = [
        sys.executable,
        str(REPO_DIR / "stitch_area.py"),
        str(local_scan_dir),
        "--downsample",
        str(args.downsample),
    ]
    print("Stitching...")
    if args.verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if not args.verbose and result.stdout:
            print(result.stdout)
        if not args.verbose and result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"Error: stitch_area failed (exit {result.returncode})")
        return 1

    # --- Step 3: find chips (output passes through — already compact) ---
    stitch_path = local_scan_dir.parent / f"{local_scan_dir.name}_stitch.jpg"
    if not stitch_path.exists():
        print(f"Error: expected stitch not found at {stitch_path}")
        return 1

    cmd = [sys.executable, str(REPO_DIR / "find_chips.py"), str(stitch_path)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: find_chips failed (exit {result.returncode})")
        return 1

    # --- Step 4: optionally show detection image ---
    detection_path = stitch_path.with_name(stitch_path.stem + "_chips_detected.png")
    if args.show and detection_path.exists():
        subprocess.run(["show", str(detection_path)])

    # --- Step 5: print all artifact paths ---
    chips_json_path = stitch_path.with_name(stitch_path.stem + "_chips.json")
    print("\nArtifacts:")
    print(f"  Stitch:    {stitch_path}")
    print(f"  Chips:     {chips_json_path}")
    print(f"  Detection: {detection_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
