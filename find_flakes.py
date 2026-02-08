"""End-to-end flake finding pipeline.

Orchestrates: initial Z → 5x overview → stitch → chip detection →
per-chip focus mapping → plane analysis → 20x scanning.

Calls existing scripts as subprocesses. If any step fails, prints what
completed and exits. Output goes under a single timestamped run directory.

Usage:
    # Full pipeline
    uv run python find_flakes.py

    # Only process chips 0 and 2
    uv run python find_flakes.py --chips 0,2

    # Preview commands without running
    uv run python find_flakes.py --dry-run

    # Pause for confirmation between each stage
    uv run python find_flakes.py --pause
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Defaults
DEFAULT_AREA_RECT = "8000,95000,0,75000"
DEFAULT_INITIAL_Z = 24690
DEFAULT_SCAN_SPEED = 5
DEFAULT_SCAN_Z_SPEED = 625
DEFAULT_CHIP_PADDING = 2000

SCRIPT_DIR = Path(__file__).parent


def run_step(name, cmd, dry_run=False, pause=False):
    """Run a subprocess step, printing the command and streaming output.

    Args:
        name: Human-readable step name.
        cmd: Command list for subprocess.
        dry_run: If True, print command but don't execute.
        pause: If True, prompt user before running.

    Returns:
        (duration_s, returncode) tuple. duration_s is 0 for dry runs.

    Raises:
        SystemExit if step fails (nonzero return code).
    """
    cmd_str = " ".join(str(c) for c in cmd)

    print()
    print("=" * 70)
    print(f"STEP: {name}")
    print(f"  {cmd_str}")
    print("=" * 70)

    if dry_run:
        print("  [dry-run] skipped")
        return 0, 0

    if pause:
        try:
            response = input("\nPress Enter to run, or 'q' to quit: ").strip().lower()
        except EOFError:
            response = ""
        if response in ("q", "quit", "exit"):
            print("Aborted by user.")
            sys.exit(0)

    print()
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    duration = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        print(f"  Command: {cmd_str}")
        sys.exit(result.returncode)

    print(f"\n  [{name}] completed in {format_duration(duration)}")
    return duration, result.returncode


def format_duration(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end flake finding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  uv run python find_flakes.py

  # Only scan chips 0 and 2
  uv run python find_flakes.py --chips 0,2

  # Custom scan area
  uv run python find_flakes.py --area-rect 10000,80000,5000,70000

  # Preview all commands
  uv run python find_flakes.py --dry-run

  # Step through with confirmation between each stage
  uv run python find_flakes.py --pause
""",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Run directory (default: scans/run_YYYYMMDD_HHMM/)",
    )
    parser.add_argument(
        "--area-rect", type=str, default=DEFAULT_AREA_RECT,
        help=f"Scan area x_min,x_max,y_min,y_max in µm (default: {DEFAULT_AREA_RECT})",
    )
    parser.add_argument(
        "--initial-z", type=float, default=DEFAULT_INITIAL_Z,
        help=f"Z position before overview in µm (default: {DEFAULT_INITIAL_Z})",
    )
    parser.add_argument(
        "--chips", type=str, default=None,
        help="Comma-separated chip indices to process (default: all)",
    )
    parser.add_argument(
        "--scan-speed", type=float, default=DEFAULT_SCAN_SPEED,
        help=f"20x scan speed in mm/s (default: {DEFAULT_SCAN_SPEED})",
    )
    parser.add_argument(
        "--scan-z-speed", type=float, default=DEFAULT_SCAN_Z_SPEED,
        help=f"Focus map AF Z speed in µm/s (default: {DEFAULT_SCAN_Z_SPEED})",
    )
    parser.add_argument(
        "--chip-padding", type=float, default=DEFAULT_CHIP_PADDING,
        help=f"Chip scan padding in µm (default: {DEFAULT_CHIP_PADDING})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Prompt for confirmation before each step",
    )
    args = parser.parse_args()

    # Build run directory
    if args.output:
        run_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = Path("scans") / f"run_{timestamp}"

    overview_dir = run_dir / "overview_5x"

    # Stitch output is placed by stitch_area.py as <scan_dir>_stitch.jpg
    # in the parent of the scan dir, i.e. run_dir/overview_5x_stitch.jpg
    stitch_path = run_dir / "overview_5x_stitch.jpg"
    stitch_meta_path = run_dir / "overview_5x_stitch_meta.json"
    chips_json_path = run_dir / "overview_5x_stitch_chips.json"

    # Parse chip filter
    chip_filter = None
    if args.chips is not None:
        chip_filter = [int(c.strip()) for c in args.chips.split(",")]

    print("FlakeFinder Pipeline")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Area rect:     {args.area_rect}")
    print(f"Initial Z:     {args.initial_z} µm")
    print(f"Scan speed:    {args.scan_speed} mm/s")
    print(f"AF Z speed:    {args.scan_z_speed} µm/s")
    print(f"Chip padding:  {args.chip_padding} µm")
    if chip_filter:
        print(f"Chips:         {chip_filter}")
    if args.dry_run:
        print(f"Mode:          DRY RUN")
    if args.pause:
        print(f"Mode:          PAUSE between steps")

    # Create run directory
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    timing = {}
    pipeline_start = time.perf_counter()

    # ----------------------------------------------------------------
    # Step 1: 5x overview scan (includes initial Z move via --z)
    # ----------------------------------------------------------------
    duration, _ = run_step(
        "5x Overview Scan",
        [
            "uv", "run", "python", "scan_area_v1.py",
            "-o", str(overview_dir),
            "--objective-mag", "5x",
            "--z", str(args.initial_z),
            "--area-rect", args.area_rect,
            "--downsample", "4",
            "--clean",
        ],
        dry_run=args.dry_run,
        pause=args.pause,
    )
    timing["overview_scan"] = duration

    # ----------------------------------------------------------------
    # Step 2: Stitch overview
    # ----------------------------------------------------------------
    duration, _ = run_step(
        "Stitch Overview",
        [
            "uv", "run", "python", "stitch_area.py",
            str(overview_dir),
        ],
        dry_run=args.dry_run,
        pause=args.pause,
    )
    timing["stitch"] = duration

    # ----------------------------------------------------------------
    # Step 3: Detect chips
    # ----------------------------------------------------------------
    duration, _ = run_step(
        "Detect Chips",
        [
            "uv", "run", "python", "find_chips.py",
            str(stitch_path),
        ],
        dry_run=args.dry_run,
        pause=args.pause,
    )
    timing["detect_chips"] = duration

    # ----------------------------------------------------------------
    # Step 4: Switch to 20x for focus mapping and chip scans
    # ----------------------------------------------------------------
    duration, _ = run_step(
        "Switch to 20x",
        [
            "uv", "run", "python", "stage_util.py",
            "--objective-mag", "20x",
        ],
        dry_run=args.dry_run,
        pause=args.pause,
    )
    timing["switch_20x"] = duration

    # ----------------------------------------------------------------
    # Load chip data to plan per-chip steps
    # ----------------------------------------------------------------
    if args.dry_run:
        # For dry run, show example commands for chip 0
        chip_indices = chip_filter or [0]
        print(f"\n  [dry-run] Would process chips: {chip_indices} (showing chip 0 as example)")
    else:
        with open(chips_json_path) as f:
            chips_data = json.load(f)
        all_chips = chips_data.get("chips", [])
        n_chips = len(all_chips)
        print(f"\nDetected {n_chips} chips")

        if chip_filter:
            chip_indices = [i for i in chip_filter if i < n_chips]
            if len(chip_indices) < len(chip_filter):
                skipped = [i for i in chip_filter if i >= n_chips]
                print(f"  Skipping out-of-range chips: {skipped}")
        else:
            chip_indices = list(range(n_chips))

        print(f"Processing chips: {chip_indices}")

    # ----------------------------------------------------------------
    # Per-chip loop
    # ----------------------------------------------------------------
    timing["chips"] = {}

    for chip_idx in chip_indices:
        chip_dir = run_dir / f"chip_{chip_idx}"
        focus_map_path = chip_dir / f"focus_map_chip{chip_idx}.json"
        plane_path = chip_dir / f"focus_map_chip{chip_idx}_plane.json"
        scan_20x_dir = chip_dir / "scan_20x"

        chip_timing = {}

        print(f"\n{'#' * 70}")
        print(f"# CHIP {chip_idx}")
        print(f"{'#' * 70}")

        # Step 4a: Focus map
        duration, _ = run_step(
            f"Chip {chip_idx} - Focus Map",
            [
                "uv", "run", "python", "focus_map.py",
                "--chips-meta", str(chips_json_path),
                "--chip", str(chip_idx),
                "--save-images",
                "--z-speed", str(int(args.scan_z_speed)),
                "--af-settle", "0.2",
                "--output-dir", str(chip_dir),
            ],
            dry_run=args.dry_run,
            pause=args.pause,
        )
        chip_timing["focus_map"] = duration

        # Step 4b: Analyze focus map + export plane
        duration, _ = run_step(
            f"Chip {chip_idx} - Analyze Focus Map",
            [
                "uv", "run", "python", "analyze_focus_map.py",
                str(focus_map_path),
                "--export-plane", str(plane_path),
                "--min-sharpness", "20",
                "-q",
            ],
            dry_run=args.dry_run,
            pause=args.pause,
        )
        chip_timing["analyze"] = duration

        # Step 4c: 20x scan
        duration, _ = run_step(
            f"Chip {chip_idx} - 20x Scan",
            [
                "uv", "run", "python", "scan_chip.py",
                "-o", str(scan_20x_dir),
                "--chips-meta", str(chips_json_path),
                "--chip", str(chip_idx),
                "--plane", str(plane_path),
                "--objective-mag", "20x",
                "--padding", str(int(args.chip_padding)),
                "--speed-mm", str(args.scan_speed),
                "--clean",
            ],
            dry_run=args.dry_run,
            pause=args.pause,
        )
        chip_timing["scan_20x"] = duration

        timing["chips"][str(chip_idx)] = chip_timing

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    pipeline_duration = time.perf_counter() - pipeline_start

    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"{'Step':<30} {'Duration':>12}")
    print("-" * 42)
    print(f"{'Overview scan':<30} {format_duration(timing.get('overview_scan', 0)):>12}")
    print(f"{'Stitch':<30} {format_duration(timing.get('stitch', 0)):>12}")
    print(f"{'Detect chips':<30} {format_duration(timing.get('detect_chips', 0)):>12}")
    print(f"{'Switch to 20x':<30} {format_duration(timing.get('switch_20x', 0)):>12}")

    for chip_idx_str, chip_t in timing.get("chips", {}).items():
        print(f"{'  Chip ' + chip_idx_str + ' focus map':<30} {format_duration(chip_t.get('focus_map', 0)):>12}")
        print(f"{'  Chip ' + chip_idx_str + ' analyze':<30} {format_duration(chip_t.get('analyze', 0)):>12}")
        print(f"{'  Chip ' + chip_idx_str + ' 20x scan':<30} {format_duration(chip_t.get('scan_20x', 0)):>12}")

    print("-" * 42)
    print(f"{'TOTAL':<30} {format_duration(pipeline_duration):>12}")
    print()
    print(f"Output: {run_dir}/")

    # Save timing data
    if not args.dry_run:
        timing["total_s"] = pipeline_duration
        timing["timestamp"] = datetime.now().isoformat()
        timing["args"] = {
            "area_rect": args.area_rect,
            "initial_z": args.initial_z,
            "scan_speed": args.scan_speed,
            "scan_z_speed": args.scan_z_speed,
            "chip_padding": args.chip_padding,
            "chips": args.chips,
        }

        timing_path = run_dir / "timing.json"
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"Timing saved to {timing_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
