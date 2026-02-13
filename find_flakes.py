"""End-to-end flake finding pipeline.

Orchestrates: initial Z → 5x overview → stitch → chip detection →
per-chip focus mapping → plane analysis → 20x scanning.

Calls command modules in-process with a shared Microscope connection.
If any step fails, prints what completed and exits. Output goes under
a single timestamped run directory.

Supports checkpointing: re-running with the same -o directory resumes
from where the previous run left off.

Usage:
    # Full pipeline
    uv run python find_flakes.py

    # Only process chips 0 and 2
    uv run python find_flakes.py --chips 0,2

    # Resume a previous run
    uv run python find_flakes.py -o scans/run_20260208_1430

    # Process chips after chip 3, limit to 2 chips
    uv run python find_flakes.py --after 3 --limit 2

    # Preview commands without running
    uv run python find_flakes.py --dry-run

    # Pause for confirmation between each stage
    uv run python find_flakes.py --pause
"""

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from commands import analyze_focus_map, chip_scan, find_chips, focus_map, scan, stage, stitch
from flakefinder.cli_utils import park_microscope
from flakefinder.leica import Microscope
from flakefinder.scan_utils import parse_area_rect, parse_white_balance, validate_area_rect
from flakefinder.types import AreaRect, GainRGB


class TeeWriter:
    """Write to both a stream and a log file."""

    def __init__(self, stream, log_file, *, suppress_console: bool = False):
        self._stream = stream
        self._log = log_file
        self.suppress_console = suppress_console

    def write(self, data):
        if not self.suppress_console:
            self._stream.write(data)
            self._stream.flush()
        # Replace \r with \n in log so progress lines become separate lines
        self._log.write(data.replace("\r", "\n"))
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


# Defaults
DEFAULT_AREA_RECT = "8000,95000,0,78000"
DEFAULT_INITIAL_Z = 24690
DEFAULT_SCAN_SPEED = 5
DEFAULT_SCAN_Z_SPEED = 1250


def load_checkpoint(run_dir):
    """Load checkpoint from run directory, or return empty checkpoint."""
    path = run_dir / "checkpoint.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_steps": [], "step_timing": {}, "n_chips": None, "chip_indices": None}


def save_checkpoint(run_dir, checkpoint):
    """Write checkpoint to run directory (atomic via temp file)."""
    path = run_dir / "checkpoint.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(path)


def step_done(checkpoint, step_key):
    """Check if a step is already checkpointed."""
    return step_key in checkpoint["completed_steps"]


def mark_step(run_dir, checkpoint, step_key, duration=0):
    """Mark a step as complete, record its timing, and persist."""
    checkpoint["completed_steps"].append(step_key)
    checkpoint["step_timing"][step_key] = duration
    save_checkpoint(run_dir, checkpoint)


def run_in_process(name, fn, *, dry_run=False, pause=False, quiet=False):
    """Run a pipeline step by calling fn() in-process.

    Args:
        name: Human-readable step name.
        fn: Callable that raises on failure or returns a result.
        dry_run: If True, print step name but don't execute.
        pause: If True, prompt user before running.
        quiet: If True, suppress banner/progress output (console already suppressed).

    Returns:
        (duration_s, result) tuple. (0, None) for dry runs.

    Raises:
        SystemExit if step fails.
    """
    if not quiet:
        print()
        print("=" * 70)
        print(f"STEP: {name}")
        print("=" * 70)

    if dry_run:
        if not quiet:
            print("  [dry-run] skipped")
        return 0, None

    if pause:
        with _always_console():
            try:
                response = input("\nPress Enter to run, or 'q' to quit: ").strip().lower()
            except EOFError:
                response = ""
            if response in ("q", "quit", "exit"):
                print("Aborted by user.")
                sys.exit(0)

    if not quiet:
        print()
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        with _always_console():
            print(f"\nFAILED: {name} ({e})")
        sys.exit(1)
    duration = time.perf_counter() - start

    if not quiet:
        print(f"\n  [{name}] completed in {format_duration(duration)}")
    return duration, result


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


def _format_duration_compact(seconds):
    """Format seconds as compact string (e.g., 3m07s, 1h23m)."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


@contextlib.contextmanager
def _always_console():
    """Temporarily force console output, restoring previous state on exit.

    In quiet mode (TeeWriter.suppress_console=True), this unsuppresses so
    summary lines always reach the terminal. In verbose mode it's a no-op.
    """
    prev: list[tuple[TeeWriter, bool]] = []
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TeeWriter):
            prev.append((stream, stream.suppress_console))
            stream.suppress_console = False
    try:
        yield
    finally:
        for stream, was_suppressed in prev:
            stream.suppress_console = was_suppressed


@dataclass
class _Preflight:
    """Validated pipeline configuration from _plan()."""

    run_dir: Path
    overview_dir: Path
    stitch_path: Path
    chips_json_path: Path
    chip_filter: list[int] | None
    wb: GainRGB
    area: AreaRect
    args: argparse.Namespace  # raw CLI args for forwarding


def _plan(args: argparse.Namespace) -> _Preflight:
    """Parse and validate pipeline configuration (no hardware).

    Args:
        args: Parsed CLI arguments.

    Returns:
        _Preflight with resolved paths and parsed values.
    """

    wb = parse_white_balance(args.white_balance)
    area = parse_area_rect(args.area_rect)

    if args.output:
        run_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = Path("scans") / f"run_{timestamp}"

    overview_dir = run_dir / "overview_5x"
    stitch_path = run_dir / "overview_5x_stitch.jpg"
    chips_json_path = run_dir / "overview_5x_stitch_chips.json"

    chip_filter = None
    if args.chips is not None:
        chip_filter = [int(c.strip()) for c in args.chips.split(",")]

    return _Preflight(
        run_dir=run_dir,
        overview_dir=overview_dir,
        stitch_path=stitch_path,
        chips_json_path=chips_json_path,
        chip_filter=chip_filter,
        wb=wb,
        area=area,
        args=args,
    )


def _build_parser():
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

  # Resume a previous run
  uv run python find_flakes.py -o scans/run_20260208_1430

  # Process chips after chip 3, limit to 2
  uv run python find_flakes.py --after 3 --limit 2

  # Step through with confirmation between each stage
  uv run python find_flakes.py --pause
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Run directory (default: scans/run_YYYYMMDD_HHMM/)",
    )
    parser.add_argument(
        "--area-rect",
        type=str,
        default=DEFAULT_AREA_RECT,
        help=f"Scan area x_min,x_max,y_min,y_max in µm (default: {DEFAULT_AREA_RECT})",
    )
    parser.add_argument(
        "--initial-z",
        type=float,
        default=DEFAULT_INITIAL_Z,
        help=f"Z position before overview in µm (default: {DEFAULT_INITIAL_Z})",
    )
    parser.add_argument(
        "--chips",
        type=str,
        default=None,
        help="Comma-separated chip indices to process (default: all)",
    )
    parser.add_argument(
        "--after",
        type=int,
        default=None,
        metavar="N",
        help="Skip chips with index <= N (applied after --chips filter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N chips (applied after --after filter)",
    )
    parser.add_argument(
        "--scan-speed",
        type=float,
        default=DEFAULT_SCAN_SPEED,
        help=f"20x scan speed in mm/s (default: {DEFAULT_SCAN_SPEED})",
    )
    parser.add_argument(
        "--scan-z-speed",
        type=float,
        default=DEFAULT_SCAN_Z_SPEED,
        help=f"Focus map AF Z speed in µm/s (default: {DEFAULT_SCAN_Z_SPEED})",
    )
    parser.add_argument(
        "--white-balance",
        type=str,
        default="2.51,1.02,1.41",
        help="White balance as B,G,R gains, passed to all capture scripts (default: 2.51,1.02,1.41)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Free-text notes stored in checkpoint.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Prompt for confirmation before each step",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print one summary line per step (full output still in pipeline.log)",
    )
    return parser


def _print_header(p: _Preflight) -> None:
    args = p.args
    print("FlakeFinder Pipeline")
    print("=" * 70)
    print(f"Run directory: {p.run_dir}")
    print(f"Area rect:     {args.area_rect}")
    print(f"Initial Z:     {args.initial_z} µm")
    print(f"Scan speed:    {args.scan_speed} mm/s")
    print(f"AF Z speed:    {args.scan_z_speed} µm/s")
    print(f"White balance: {args.white_balance} (B,G,R)")
    if p.chip_filter:
        print(f"Chips:         {p.chip_filter}")
    if args.after is not None:
        print(f"After:         {args.after}")
    if args.limit is not None:
        print(f"Limit:         {args.limit}")
    if args.dry_run:
        print("Mode:          DRY RUN")
    if args.pause:
        print("Mode:          PAUSE between steps")


def _resolve_chip_indices(p: _Preflight) -> list[int]:
    """Load chip data and apply --chips/--after/--limit filters."""
    args = p.args
    with open(p.chips_json_path) as f:
        chips_data = json.load(f)
    all_chips = chips_data.get("chips", [])
    n_chips = len(all_chips)
    print(f"\nDetected {n_chips} chips")

    if p.chip_filter:
        chip_indices = [i for i in p.chip_filter if i < n_chips]
        if len(chip_indices) < len(p.chip_filter):
            skipped = [i for i in p.chip_filter if i >= n_chips]
            print(f"  Skipping out-of-range chips: {skipped}")
    else:
        chip_indices = list(range(n_chips))

    if args.after is not None:
        before = len(chip_indices)
        chip_indices = [i for i in chip_indices if i > args.after]
        dropped = before - len(chip_indices)
        if dropped:
            print(f"  --after {args.after}: dropped {dropped} chip(s)")

    if args.limit is not None and len(chip_indices) > args.limit:
        chip_indices = chip_indices[: args.limit]
        print(f"  --limit {args.limit}: capped to {args.limit} chip(s)")

    print(f"Processing chips: {chip_indices}")
    return chip_indices


def _print_focus_map_summary(chip_idx: int, plane_path: Path) -> None:
    """Print [chip N] focus_map summary line from exported plane JSON."""
    with _always_console():
        if plane_path.exists():
            with open(plane_path) as f:
                plane_data = json.load(f)
            q = plane_data.get("quality", {})
            pts_used = q.get("points_used", "?")
            pts_total = q.get("points_total", "?")
            r2 = q.get("r_squared", 0)
            resid = q.get("residual_std_um", 0)
            print(
                f"[chip {chip_idx}] focus_map {pts_used}/{pts_total} pts, "
                f"R\u00b2={r2:.3f}, residual {resid:.1f} \u00b5m"
            )


def _print_chip_scan_summary(chip_idx: int, scan_dir: Path) -> None:
    """Print [chip N] scan summary line from chip scan metadata."""
    with _always_console():
        meta_path = scan_dir / "scan_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                csm = json.load(f)
            frames = csm.get("frame_count", "?")
            rows = len(csm.get("rows", []))
            dur = csm.get("scan_duration_s", 0)
            te = csm.get("tracking_error", {})
            z_std = te.get("std_um")
            z_max_err = te.get("max_um")
            z_str = ""
            if z_std is not None and z_max_err is not None:
                dof_20x = 1.7
                verdict = "PASS" if z_max_err < dof_20x else "NOTE"
                z_str = f", Z std {z_std:.2f} \u00b5m {verdict}"
            print(f"[chip {chip_idx}] scan {frames} frames, {rows} rows, {_format_duration_compact(dur)}{z_str}")


def _print_chip_summary(chip_idx: int, plane_path: Path, scan_dir: Path) -> None:
    """Print both focus_map and scan summaries for a chip."""
    _print_focus_map_summary(chip_idx, plane_path)
    _print_chip_scan_summary(chip_idx, scan_dir)


def run(scope: Microscope, p: _Preflight) -> int:
    """Execute the flake-finding pipeline with a live Microscope.

    Returns:
        0 on success (non-zero exits via sys.exit from run_in_process).
    """
    validate_area_rect(p.area, scope.stage)

    args = p.args
    quiet = args.quiet
    run_dir = p.run_dir
    checkpoint = load_checkpoint(run_dir)

    with _always_console():
        print(f"[run] {run_dir}/")

    notes = args.notes or checkpoint.get("notes")
    if notes:
        checkpoint["notes"] = notes
        print(f"Notes:         {notes}")

    if checkpoint["completed_steps"]:
        print(f"\nResuming from checkpoint ({len(checkpoint['completed_steps'])} steps complete)")
        for s in checkpoint["completed_steps"]:
            print(f"  [checkpoint] {s}")

    pipeline_start = time.perf_counter()

    # ----------------------------------------------------------------
    # Step 1: 5x overview scan
    # ----------------------------------------------------------------
    if step_done(checkpoint, "overview_scan"):
        print("\n  [checkpoint] Skipping 5x Overview Scan (already complete)")
    else:
        duration, _ = run_in_process(
            "5x Overview Scan",
            lambda: scan.run(
                scope=scope,
                output=str(p.overview_dir),
                objective_mag="5x",
                initial_z=args.initial_z,
                area_rect=p.area,
                downsample=4,
                white_balance=p.wb,
                clean=True,
                quiet=quiet,
            ),
            pause=args.pause,
            quiet=quiet,
        )
        mark_step(run_dir, checkpoint, "overview_scan", duration)

    with _always_console():
        meta_path = p.overview_dir / "scan_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                scan_meta = json.load(f)
            frames = scan_meta.get("frame_count", "?")
            rows = len(scan_meta.get("rows", []))
            dur = checkpoint["step_timing"].get("overview_scan", 0)
            print(f"[overview] {frames} frames, {rows} rows, {_format_duration_compact(dur)}")

    # ----------------------------------------------------------------
    # Step 2: Stitch overview
    # ----------------------------------------------------------------
    if step_done(checkpoint, "stitch"):
        print("\n  [checkpoint] Skipping Stitch Overview (already complete)")
    else:
        duration, _ = run_in_process(
            "Stitch Overview",
            lambda: stitch.run(scan_dir=p.overview_dir, quiet=True),
            pause=args.pause,
            quiet=quiet,
        )
        mark_step(run_dir, checkpoint, "stitch", duration)

    with _always_console():
        stitch_meta_path = p.stitch_path.with_name(p.stitch_path.stem + "_meta.json")
        if stitch_meta_path.exists():
            with open(stitch_meta_path) as f:
                sm = json.load(f)
            w, h = sm["image_size_px"]
            print(f"[stitch] {w}x{h} px")

    # ----------------------------------------------------------------
    # Step 3: Detect chips
    # ----------------------------------------------------------------
    if step_done(checkpoint, "detect_chips"):
        print("\n  [checkpoint] Skipping Detect Chips (already complete)")
    else:
        duration, result = run_in_process(
            "Detect Chips",
            lambda: find_chips.run(image_path=p.stitch_path),
            pause=args.pause,
            quiet=quiet,
        )
        n_chips = len(result.get("chips", []))
        checkpoint["n_chips"] = n_chips
        checkpoint["chip_indices"] = list(range(n_chips))
        mark_step(run_dir, checkpoint, "detect_chips", duration)

    with _always_console():
        with open(p.chips_json_path) as f:
            chips_data = json.load(f)
        print(f"[detect] {len(chips_data.get('chips', []))} chips")

    # ----------------------------------------------------------------
    # Step 4: Switch to 20x
    # ----------------------------------------------------------------
    if step_done(checkpoint, "switch_20x"):
        print("\n  [checkpoint] Skipping Switch to 20x (already complete)")
    else:
        duration, _ = run_in_process(
            "Switch to 20x",
            lambda: stage.run(scope=scope, objective_mag="20x"),
            pause=args.pause,
            quiet=quiet,
        )
        mark_step(run_dir, checkpoint, "switch_20x", duration)

    # ----------------------------------------------------------------
    # Load chip data and apply filters
    # ----------------------------------------------------------------
    chip_indices = _resolve_chip_indices(p)

    # ----------------------------------------------------------------
    # Per-chip loop
    # ----------------------------------------------------------------
    for chip_idx in chip_indices:
        chip_dir = run_dir / f"chip_{chip_idx}"
        focus_map_path = chip_dir / f"focus_map_chip{chip_idx}.json"
        plane_path = chip_dir / f"focus_map_chip{chip_idx}_plane.json"
        scan_20x_dir = chip_dir / "scan_20x"

        fm_key = f"chip_{chip_idx}_focus_map"
        an_key = f"chip_{chip_idx}_analyze"
        sc_key = f"chip_{chip_idx}_scan"

        if step_done(checkpoint, fm_key) and step_done(checkpoint, an_key) and step_done(checkpoint, sc_key):
            print(f"\n  [checkpoint] Skipping chip {chip_idx} (all steps complete)")
            _print_chip_summary(chip_idx, plane_path, scan_20x_dir)
            continue

        print(f"\n{'#' * 70}")
        print(f"# CHIP {chip_idx}")
        print(f"{'#' * 70}")

        # Step 5a: Focus map
        if step_done(checkpoint, fm_key):
            print(f"\n  [checkpoint] Skipping Chip {chip_idx} - Focus Map (already complete)")
        else:
            duration, _ = run_in_process(
                f"Chip {chip_idx} - Focus Map",
                lambda ci=chip_idx, cd=chip_dir: focus_map.run(
                    scope=scope,
                    chips_meta=p.chips_json_path,
                    chip=ci,
                    save_images=True,
                    z_speed=args.scan_z_speed,
                    af_settle=0.2,
                    white_balance=p.wb,
                    output_dir=cd,
                    quiet=True,
                ),
                pause=args.pause,
                quiet=quiet,
            )
            mark_step(run_dir, checkpoint, fm_key, duration)

        # Step 5b: Analyze focus map + export plane
        if step_done(checkpoint, an_key):
            print(f"\n  [checkpoint] Skipping Chip {chip_idx} - Analyze (already complete)")
        else:
            duration, _ = run_in_process(
                f"Chip {chip_idx} - Analyze Focus Map",
                lambda fmp=focus_map_path, pp=plane_path: analyze_focus_map.run(
                    focus_map_path=fmp,
                    export_plane_path=pp,
                    min_sharpness=20.0,
                    quiet=True,
                ),
                pause=args.pause,
                quiet=quiet,
            )
            mark_step(run_dir, checkpoint, an_key, duration)

        # Summary: focus_map (after analyze exports plane)
        _print_focus_map_summary(chip_idx, plane_path)

        # Step 5c: 20x scan
        if step_done(checkpoint, sc_key):
            print(f"\n  [checkpoint] Skipping Chip {chip_idx} - 20x Scan (already complete)")
        else:
            duration, _ = run_in_process(
                f"Chip {chip_idx} - 20x Scan",
                lambda ci=chip_idx, sd=scan_20x_dir, pp=plane_path: chip_scan.run(
                    scope=scope,
                    output=str(sd),
                    chips_meta=p.chips_json_path,
                    chip=ci,
                    plane_path=pp,
                    objective_mag="20x",
                    speed_mm=args.scan_speed,
                    white_balance=p.wb,
                    clean=True,
                    quiet=True,
                ),
                pause=args.pause,
                quiet=quiet,
            )
            mark_step(run_dir, checkpoint, sc_key, duration)

        # Summary: chip scan
        _print_chip_scan_summary(chip_idx, scan_20x_dir)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    pipeline_duration = time.perf_counter() - pipeline_start
    t = checkpoint["step_timing"]

    with _always_console():
        print(f"[done] {len(chip_indices)} chips, {_format_duration_compact(pipeline_duration)} total")

    if not quiet:
        print()
        print("=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"{'Step':<30} {'Duration':>12}")
        print("-" * 42)
        print(f"{'Overview scan':<30} {format_duration(t.get('overview_scan', 0)):>12}")
        print(f"{'Stitch':<30} {format_duration(t.get('stitch', 0)):>12}")
        print(f"{'Detect chips':<30} {format_duration(t.get('detect_chips', 0)):>12}")
        print(f"{'Switch to 20x':<30} {format_duration(t.get('switch_20x', 0)):>12}")

        for chip_idx in chip_indices:
            fm = t.get(f"chip_{chip_idx}_focus_map", 0)
            an = t.get(f"chip_{chip_idx}_analyze", 0)
            sc = t.get(f"chip_{chip_idx}_scan", 0)
            print(f"{'  Chip ' + str(chip_idx) + ' focus map':<30} {format_duration(fm):>12}")
            print(f"{'  Chip ' + str(chip_idx) + ' analyze':<30} {format_duration(an):>12}")
            print(f"{'  Chip ' + str(chip_idx) + ' 20x scan':<30} {format_duration(sc):>12}")

        total_recorded = sum(t.values())
        print("-" * 42)
        print(f"{'Total (recorded)':<30} {format_duration(total_recorded):>12}")
        print(f"{'This invocation':<30} {format_duration(pipeline_duration):>12}")
        print()
        print(f"Output: {run_dir}/")

    # Save args to checkpoint for reference
    checkpoint["args"] = {
        "area_rect": args.area_rect,
        "initial_z": args.initial_z,
        "scan_speed": args.scan_speed,
        "scan_z_speed": args.scan_z_speed,
        "chips": args.chips,
        "after": args.after,
        "limit": args.limit,
    }
    save_checkpoint(run_dir, checkpoint)

    # ----------------------------------------------------------------
    # Step 6: Park microscope
    # ----------------------------------------------------------------
    if not step_done(checkpoint, "park"):
        duration, _ = run_in_process(
            "Park Microscope",
            lambda: park_microscope(scope),
            pause=args.pause,
            quiet=quiet,
        )
        mark_step(run_dir, checkpoint, "park", duration)

    return 0


def main() -> int:
    args = _build_parser().parse_args()
    p = _plan(args)

    if args.dry_run:
        _print_header(p)
        run_in_process("5x Overview Scan", lambda: None, dry_run=True)
        run_in_process("Stitch Overview", lambda: None, dry_run=True)
        run_in_process("Detect Chips", lambda: None, dry_run=True)
        run_in_process("Switch to 20x", lambda: None, dry_run=True)
        if p.chips_json_path.exists():
            chip_indices = _resolve_chip_indices(p)
            for ci in chip_indices:
                run_in_process(f"Chip {ci} - Focus Map", lambda: None, dry_run=True)
                run_in_process(f"Chip {ci} - Analyze Focus Map", lambda: None, dry_run=True)
                run_in_process(f"Chip {ci} - 20x Scan", lambda: None, dry_run=True)
        else:
            print("\n  Per-chip steps: Focus Map -> Analyze -> 20x Scan (chips not yet detected)")
        return 0

    # Create run directory and set up log tee
    p.run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(p.run_dir / "pipeline.log", "a")  # noqa: SIM115
    quiet = args.quiet
    sys.stdout = TeeWriter(sys.__stdout__, log_file, suppress_console=quiet)
    sys.stderr = TeeWriter(sys.__stderr__, log_file, suppress_console=quiet)

    try:
        _print_header(p)
        with Microscope() as scope:
            return run(scope, p)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log: {p.run_dir / 'pipeline.log'}")


if __name__ == "__main__":
    sys.exit(main())
