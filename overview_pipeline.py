"""Overview scan post-processing pipeline: stitch, detect chips, show results.

Wraps the common overview workflow into a single command.

Usage:
    uv run python overview_pipeline.py scans/overview_5x
    uv run python overview_pipeline.py scans/overview_5x --downsample 4 --show
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stitch an overview scan and detect chips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scan_dir",
        type=Path,
        help="Scan directory containing scan_meta.json and frame JPGs",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Downsample factor for stitching",
    )
    parser.add_argument("--show", action="store_true", help="Open the chip detection image")
    args = parser.parse_args()

    scan_dir = args.scan_dir.resolve()
    if not scan_dir.is_dir():
        print(f"Error: Not a directory: {scan_dir}")
        return 1

    # --- Step 1: stitch ---
    cmd = [
        sys.executable,
        str(REPO_DIR / "stitch_area.py"),
        str(scan_dir),
        "--downsample",
        str(args.downsample),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: stitch_area failed (exit {result.returncode})")
        return 1

    # --- Step 2: find chips ---
    stitch_path = scan_dir.parent / f"{scan_dir.name}_stitch.jpg"
    if not stitch_path.exists():
        print(f"Error: expected stitch not found at {stitch_path}")
        return 1

    cmd = [sys.executable, str(REPO_DIR / "find_chips.py"), str(stitch_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: find_chips failed (exit {result.returncode})")
        return 1

    # --- Step 3: optionally show detection image ---
    detection_path = stitch_path.with_name(stitch_path.stem + "_chips_detected.png")
    if args.show and detection_path.exists():
        subprocess.run(["show", str(detection_path)])

    # --- Step 4: print all artifact paths ---
    chips_json_path = stitch_path.with_name(stitch_path.stem + "_chips.json")
    print("\nArtifacts:")
    print(f"  Stitch:    {stitch_path}")
    print(f"  Chips:     {chips_json_path}")
    print(f"  Detection: {detection_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
