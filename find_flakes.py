"""End-to-end flake finding pipeline.

Orchestrates: overview scan → stitch → chip detection →
per-chip focus mapping → plane analysis → chip scanning →
background segmentation.

Overview and chip scan magnifications are configurable (default: 5x
overview, 20x chip scan). Scan speeds scale automatically with
magnification.

Segmentation runs in a background thread after each chip scan completes,
using reduced workers (--seg-jobs-bg) to limit CPU contention with
hardware operations. After the microscope is parked, remaining jobs are
boosted to --seg-jobs-idle workers. Use --seg-wait for a safe baseline
that blocks between chips, or --no-segment to disable entirely.

Calls command modules in-process with a shared Microscope connection.
If any step fails, prints what completed and exits. Output goes under
a single timestamped run directory.

Supports checkpointing: re-running with the same -o directory resumes
from where the previous run left off.

Usage:
    # Full pipeline with default preset (5x overview + 20x chip scan)
    uv run python find_flakes.py

    # Fast screening preset (2.5x overview + 10x chip scan)
    uv run python find_flakes.py --preset 2.5_10

    # Only process chips 0 and 2
    uv run python find_flakes.py --chips 0,2

    # Resume a previous run (config loaded from checkpoint)
    uv run python find_flakes.py --resume scans/run_20260208_1430

    # Disable background segmentation
    uv run python find_flakes.py --no-segment

    # Safe seg baseline (blocks between chips)
    uv run python find_flakes.py --seg-wait

    # Preview commands without running
    uv run python find_flakes.py --dry-run
"""

import argparse
import contextlib
import json
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, TypedDict

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
REPO_DIR = Path(__file__).resolve().parent


class ScanPreset(TypedDict):
    overview_mag: str
    chip_scan_mag: str
    chip_scan_speed_mm: float
    chip_scan_gain: float
    chip_scan_exposure_ms: float
    focus_map_gain: float
    focus_map_exposure_ms: float


PRESETS: dict[str, ScanPreset] = {
    "5_20": {
        "overview_mag": "5x",
        "chip_scan_mag": "20x",
        "chip_scan_speed_mm": 5.0,
        "chip_scan_gain": 4.0,
        "chip_scan_exposure_ms": 0.25,
        "focus_map_gain": 1.0,
        "focus_map_exposure_ms": 1.0,
    },
    "2.5_10": {
        "overview_mag": "2.5x",
        "chip_scan_mag": "10x",
        "chip_scan_speed_mm": 10.0,
        "chip_scan_gain": 4.0,
        "chip_scan_exposure_ms": 0.25,
        "focus_map_gain": 1.0,
        "focus_map_exposure_ms": 1.0,
    },
}

DEFAULT_PRESET = "5_20"

# Config flags that --resume forbids (must come from checkpoint instead)
_RESUME_FORBIDDEN_FLAGS = frozenset(
    {
        "--preset",
        "--overview-mag",
        "--chip-scan-mag",
        "--scan-speed",
        "--scan-z-speed",
        "--area-rect",
        "--initial-z",
        "--white-balance",
        "--notes",
    }
)


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
class SegConfig:
    """Segmentation pipeline configuration."""

    enabled: bool
    wait: bool  # --seg-wait: block before next chip until seg completes
    flatfield: Path | None
    material: str
    jobs_bg: int  # workers during hardware ops
    jobs_idle: int  # workers after park (boost)


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
    preset_name: str
    overview_mag: str
    chip_scan_mag: str
    scan_speed: float
    scan_z_speed: float
    chip_scan_gain: float
    chip_scan_exposure_ms: float
    focus_map_gain: float
    focus_map_exposure_ms: float
    seg: SegConfig
    args: argparse.Namespace  # raw CLI args for forwarding


def _resolve_flatfield(chip_scan_mag: str) -> Path | None:
    """Auto-detect flatfield file for a chip scan magnification."""
    path = REPO_DIR / "calibration" / f"flatfield_{chip_scan_mag}_bin3.npy"
    return path if path.exists() else None


def _plan(args: argparse.Namespace) -> _Preflight:
    """Parse and validate pipeline configuration (no hardware).

    Args:
        args: Parsed CLI arguments.

    Returns:
        _Preflight with resolved paths and parsed values.
    """

    wb = parse_white_balance(args.white_balance)
    area = parse_area_rect(args.area_rect)

    # Resolve preset
    preset_name: str = args.preset
    preset = PRESETS[preset_name]

    # Normalize magnification: "5" → "5x", "2.5X" → "2.5x"
    def _fmt_mag(s: str) -> str:
        n = float(s.lower().strip().rstrip("x"))
        return f"{int(n)}x" if n == int(n) else f"{n}x"

    # CLI flags override preset values when provided
    overview_mag = _fmt_mag(args.overview_mag) if args.overview_mag is not None else preset["overview_mag"]
    chip_scan_mag = _fmt_mag(args.chip_scan_mag) if args.chip_scan_mag is not None else preset["chip_scan_mag"]

    if args.output:
        run_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_dir = Path("scans") / f"run_{timestamp}"

    overview_dir = run_dir / f"overview_{overview_mag}"
    stitch_path = run_dir / f"overview_{overview_mag}_stitch.jpg"
    chips_json_path = run_dir / f"overview_{overview_mag}_stitch_chips.json"

    chip_filter = None
    if args.chips is not None:
        chip_filter = [int(c.strip()) for c in args.chips.split(",")]

    # Scan speed from preset; CLI overrides. AF Z speed still auto-derived from mag.
    scan_speed = args.scan_speed if args.scan_speed is not None else preset["chip_scan_speed_mm"]
    chip_mag_num = float(chip_scan_mag.rstrip("x"))
    mag_scale = 20.0 / chip_mag_num
    scan_z_speed = args.scan_z_speed if args.scan_z_speed is not None else min(1250.0 * mag_scale, 5000.0)

    # Segmentation config
    if args.flatfield:
        flatfield = Path(args.flatfield)
    else:
        flatfield = _resolve_flatfield(chip_scan_mag)

    seg = SegConfig(
        enabled=not args.no_segment,
        wait=args.seg_wait,
        flatfield=flatfield,
        material=args.material,
        jobs_bg=args.seg_jobs_bg,
        jobs_idle=args.seg_jobs_idle,
    )

    return _Preflight(
        run_dir=run_dir,
        overview_dir=overview_dir,
        stitch_path=stitch_path,
        chips_json_path=chips_json_path,
        chip_filter=chip_filter,
        wb=wb,
        area=area,
        preset_name=preset_name,
        overview_mag=overview_mag,
        chip_scan_mag=chip_scan_mag,
        scan_speed=scan_speed,
        scan_z_speed=scan_z_speed,
        chip_scan_gain=preset["chip_scan_gain"],
        chip_scan_exposure_ms=preset["chip_scan_exposure_ms"],
        focus_map_gain=preset["focus_map_gain"],
        focus_map_exposure_ms=preset["focus_map_exposure_ms"],
        seg=seg,
        args=args,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="End-to-end flake finding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Presets: {", ".join(PRESETS)}

Examples:
  # Full pipeline with default preset ({DEFAULT_PRESET})
  uv run python find_flakes.py

  # Fast screening preset
  uv run python find_flakes.py --preset 2.5_10

  # Only scan chips 0 and 2
  uv run python find_flakes.py --chips 0,2

  # Preview all commands
  uv run python find_flakes.py --dry-run

  # Resume a previous run (config from checkpoint)
  uv run python find_flakes.py --resume scans/run_20260208_1430

  # Process chips after chip 3, limit to 2
  uv run python find_flakes.py --after 3 --limit 2
""",
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Run directory for new run (default: scans/run_YYYYMMDD_HHMM/)",
    )
    run_group.add_argument(
        "--resume",
        type=str,
        metavar="DIR",
        default=None,
        help="Resume a previous run (loads all config from checkpoint)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=list(PRESETS),
        help=f"Scan preset (default: {DEFAULT_PRESET})",
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
        "--overview-mag",
        type=str,
        default=None,
        help="Override preset overview magnification",
    )
    parser.add_argument(
        "--chip-scan-mag",
        type=str,
        default=None,
        help="Override preset chip scan magnification",
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
        default=None,
        help="Chip scan speed in mm/s (default: scales with magnification, 5 at 20x)",
    )
    parser.add_argument(
        "--scan-z-speed",
        type=float,
        default=None,
        help="Focus map AF Z speed in µm/s (default: scales with magnification, 1250 at 20x)",
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

    # Segmentation (runs in background after each chip scan)
    seg_group = parser.add_argument_group("Segmentation")
    seg_group.add_argument(
        "--no-segment",
        action="store_true",
        help="Disable background segmentation",
    )
    seg_group.add_argument(
        "--seg-wait",
        action="store_true",
        help="Wait for segmentation to complete before next chip (safe baseline for comparison)",
    )
    seg_group.add_argument(
        "--flatfield",
        type=str,
        default=None,
        help="Flatfield .npy file (default: auto-detect from calibration/flatfield_{mag}_bin3.npy)",
    )
    seg_group.add_argument(
        "--material",
        type=str,
        default="hbn",
        choices=["hbn", "graphene"],
        help="Material preset for segmentation (default: hbn)",
    )
    seg_group.add_argument(
        "--seg-jobs-bg",
        type=int,
        default=4,
        help="Segmentation workers during hardware ops (default: 4)",
    )
    seg_group.add_argument(
        "--seg-jobs-idle",
        type=int,
        default=8,
        help="Segmentation workers after park / boost (default: 8)",
    )
    return parser


def _print_header(p: _Preflight) -> None:
    args = p.args
    print("FlakeFinder Pipeline")
    print("=" * 70)
    print(f"Run directory: {p.run_dir}")
    print(f"Preset:        {p.preset_name}")
    print(f"Overview:      {p.overview_mag}")
    print(f"Chip scan:     {p.chip_scan_mag}")
    print(f"Area rect:     {args.area_rect}")
    print(f"Initial Z:     {args.initial_z} µm")
    print(f"Scan speed:    {p.scan_speed} mm/s")
    print(f"AF Z speed:    {p.scan_z_speed} µm/s")
    print(f"Chip camera:   gain={p.chip_scan_gain}, exposure={p.chip_scan_exposure_ms}ms")
    print(f"AF camera:     gain={p.focus_map_gain}, exposure={p.focus_map_exposure_ms}ms")
    print(f"White balance: {args.white_balance} (B,G,R)")
    if p.chip_filter:
        print(f"Chips:         {p.chip_filter}")
    if args.after is not None:
        print(f"After:         {args.after}")
    if args.limit is not None:
        print(f"Limit:         {args.limit}")
    if p.seg.enabled:
        ff_label = str(p.seg.flatfield) if p.seg.flatfield else "none"
        print(f"Segmentation:  {p.seg.material}, {p.seg.jobs_bg}j bg / {p.seg.jobs_idle}j idle")
        print(f"Flatfield:     {ff_label}")
        if p.seg.wait:
            print("Seg mode:      WAIT (blocking)")
    else:
        print("Segmentation:  disabled")
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
            rows = len(csm.get("lines", []))
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


# ============================================================================
# Background segmentation
# ============================================================================


@dataclass
class _SegJob:
    """Tracks one chip's segmentation work."""

    chip_idx: int
    seg_key: str  # checkpoint key
    scan_dir: Path  # chip scan directory (frame_NNNN.jpg)
    seg_dir: Path  # segmentation output directory


class _SegResult(NamedTuple):
    success: bool
    duration: float
    summary: str  # one-line summary for console
    error: str  # error message on failure


def _run_chip_seg(
    job: _SegJob,
    flatfield: Path | None,
    material: str,
    jobs: int,
) -> _SegResult:
    """Run segmentation for one chip in-process. Called in background thread."""
    from flakefinder.segmentation import (
        DetectorConfig,
        FrameResult,
        natural_sort_key,
        process_frame,
        strip_geometry,
    )

    start = time.perf_counter()
    try:
        # Read pixel size from scan metadata
        pixel_size = 0.36  # fallback for 20x bin3
        scan_meta_path = job.scan_dir / "scan_meta.json"
        if scan_meta_path.exists():
            with open(scan_meta_path) as f:
                sm = json.load(f)
            pixel_size = sm.get("optics", {}).get("sample_pixel_x_um", pixel_size)

        # Discover frames
        frames = sorted(job.scan_dir.glob("frame_*.jpg"), key=natural_sort_key)
        if not frames:
            return _SegResult(success=False, duration=0, summary="", error="No frame_*.jpg files found")

        job.seg_dir.mkdir(parents=True, exist_ok=True)
        flatfield_str = str(flatfield) if flatfield else None
        config = DetectorConfig.from_material(material)
        dark_frac_cutoff = 0.05

        # Process all frames in parallel
        results: dict[str, FrameResult] = {}
        total_det_count = 0

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            future_to_name = {}
            for fp in frames:
                fut = pool.submit(process_frame, str(fp), flatfield_str, config, pixel_size, dark_frac_cutoff)
                future_to_name[fut] = fp.stem

            for fut in as_completed(future_to_name):
                r = fut.result()
                results[r.frame_name] = r
                total_det_count += len(r.detections)

        # Write per-frame JSONs (geometry only)
        for fp in frames:
            name = fp.stem
            r = results[name]
            if not r.detections:
                continue
            geom_dets = [
                {k: v for k, v in d.items() if k not in ("tier", "score", "classification")} for d in r.detections
            ]
            frame_json = {"frame": name, "dark_frac": round(r.dark_frac, 4), "detections": geom_dets}
            with open(job.seg_dir / f"{name}.json", "w") as f:
                json.dump(frame_json, f, indent=2)

        # Build summary
        tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        skipped_count = 0
        frames_with_dets = 0
        all_detections: dict[str, list[dict]] = {}

        for fp in frames:
            name = fp.stem
            r = results[name]
            if r.skipped:
                skipped_count += 1
            if r.detections:
                frames_with_dets += 1
                stripped = [strip_geometry(d) for d in r.detections]
                for d in stripped:
                    d["frame"] = name
                all_detections[name] = stripped
                for d in r.detections:
                    tier = d.get("tier", 3)
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

        seg_elapsed = time.perf_counter() - start

        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "duration_s": round(seg_elapsed, 2),
            "scan_dir": str(job.scan_dir),
            "params": {
                "material": material,
                "flatfield": flatfield_str,
                "contrast_offset": config.contrast_offset,
                "min_size_um2": config.min_size_um2,
                "min_size_px": int(config.min_size_um2 / (pixel_size**2)),
                "edge_margin_px": config.edge_margin_px,
                "dark_frac_cutoff": dark_frac_cutoff,
                "pixel_size_um": pixel_size,
            },
            "stats": {
                "total_frames": len(frames),
                "skipped_frames": skipped_count,
                "frames_with_detections": frames_with_dets,
                "total_detections": total_det_count,
                "tier_1": tier_counts[1],
                "tier_2": tier_counts[2],
                "tier_3": tier_counts[3],
            },
            "detections_by_frame": all_detections,
        }

        with open(job.seg_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        summary_str = (
            f"{len(frames)} frames, "
            f"{total_det_count} det "
            f"(T1:{tier_counts[1]} T2:{tier_counts[2]} "
            f"T3:{tier_counts[3]}), {seg_elapsed:.1f}s @ {jobs}j"
        )

        return _SegResult(success=True, duration=seg_elapsed, summary=summary_str, error="")

    except Exception as e:
        duration = time.perf_counter() - start
        return _SegResult(success=False, duration=duration, summary="", error=str(e))


def _on_seg_done(future: Future, chip_idx: int) -> None:
    """Print seg completion summary. Runs in executor thread."""
    try:
        result = future.result()
        with _always_console():
            if result.success:
                print(f"[seg chip {chip_idx}] {result.summary}")
            else:
                print(f"[seg chip {chip_idx}] FAILED: {result.error[:200]}")
    except Exception as e:
        with _always_console():
            print(f"[seg chip {chip_idx}] ERROR: {e}")


def _drain_seg(
    seg_futures: list[tuple[_SegJob, Future]],
    seg_executor: ThreadPoolExecutor,
    seg: SegConfig,
    checkpoint: dict,
    run_dir: Path,
) -> float:
    """Wait for all seg futures, boost queued jobs, checkpoint completions.

    Returns total segmentation wall-clock time across all chips.
    """
    total_seg_time = 0.0
    to_boost: list[_SegJob] = []

    for job, future in seg_futures:
        if future.cancel():
            to_boost.append(job)
            continue
        # Done or running — result() blocks if still running
        try:
            result = future.result()
            total_seg_time += result.duration
            if result.success:
                mark_step(run_dir, checkpoint, job.seg_key, result.duration)
        except Exception:
            pass  # callback already logged

    # Re-submit cancelled jobs at idle (boosted) speed
    if to_boost:
        with _always_console():
            print(f"[seg] Boosting {len(to_boost)} job(s) to {seg.jobs_idle} workers")
        for job in to_boost:
            future = seg_executor.submit(_run_chip_seg, job, seg.flatfield, seg.material, seg.jobs_idle)
            try:
                result = future.result()
                total_seg_time += result.duration
                with _always_console():
                    if result.success:
                        print(f"[seg chip {job.chip_idx}] {result.summary}")
                        mark_step(run_dir, checkpoint, job.seg_key, result.duration)
                    else:
                        print(f"[seg chip {job.chip_idx}] FAILED: {result.error[:200]}")
            except Exception as e:
                with _always_console():
                    print(f"[seg chip {job.chip_idx}] ERROR: {e}")

    return total_seg_time


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

    # Background segmentation executor (max 1 concurrent seg job)
    seg_executor: ThreadPoolExecutor | None = None
    seg_futures: list[tuple[_SegJob, Future]] = []
    if p.seg.enabled:
        seg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seg")

    pipeline_start = time.perf_counter()

    # ----------------------------------------------------------------
    # Step 1: Overview scan
    # ----------------------------------------------------------------
    if step_done(checkpoint, "overview_scan"):
        print(f"\n  [checkpoint] Skipping {p.overview_mag} Overview Scan (already complete)")
    else:
        duration, _ = run_in_process(
            f"{p.overview_mag} Overview Scan",
            lambda: scan.run(
                scope=scope,
                output=str(p.overview_dir),
                objective_mag=p.overview_mag,
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
            rows = len(scan_meta.get("lines", []))
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
    # Step 4: Switch to chip scan objective
    # ----------------------------------------------------------------
    switch_key = f"switch_{p.chip_scan_mag}"
    if step_done(checkpoint, switch_key):
        print(f"\n  [checkpoint] Skipping Switch to {p.chip_scan_mag} (already complete)")
    else:
        duration, _ = run_in_process(
            f"Switch to {p.chip_scan_mag}",
            lambda: stage.run(scope=scope, objective_mag=p.chip_scan_mag),
            pause=args.pause,
            quiet=quiet,
        )
        mark_step(run_dir, checkpoint, switch_key, duration)

    # ----------------------------------------------------------------
    # Load chip data and apply filters
    # ----------------------------------------------------------------
    chip_indices = _resolve_chip_indices(p)

    # ----------------------------------------------------------------
    # Per-chip loop
    # ----------------------------------------------------------------
    chip_loop_start = time.perf_counter()
    chips_processed = 0

    for loop_pos, chip_idx in enumerate(chip_indices):
        chip_dir = run_dir / f"chip_{chip_idx}"
        focus_map_path = chip_dir / f"focus_map_chip{chip_idx}.json"
        plane_path = chip_dir / f"focus_map_chip{chip_idx}_plane.json"
        chip_scan_dir = chip_dir / f"scan_{p.chip_scan_mag}"

        fm_key = f"chip_{chip_idx}_focus_map"
        an_key = f"chip_{chip_idx}_analyze"
        sc_key = f"chip_{chip_idx}_scan"

        if step_done(checkpoint, fm_key) and step_done(checkpoint, an_key) and step_done(checkpoint, sc_key):
            print(f"\n  [checkpoint] Skipping chip {chip_idx} (all steps complete)")
            _print_chip_summary(chip_idx, plane_path, chip_scan_dir)
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
                    gain=p.focus_map_gain,
                    exposure_ms=p.focus_map_exposure_ms,
                    chip=ci,
                    save_images=True,
                    z_speed=p.scan_z_speed,
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

        # Step 5c: Chip scan
        if step_done(checkpoint, sc_key):
            print(f"\n  [checkpoint] Skipping Chip {chip_idx} - {p.chip_scan_mag} Scan (already complete)")
        else:
            duration, _ = run_in_process(
                f"Chip {chip_idx} - {p.chip_scan_mag} Scan",
                lambda ci=chip_idx, sd=chip_scan_dir, pp=plane_path: chip_scan.run(
                    scope=scope,
                    output=str(sd),
                    chips_meta=p.chips_json_path,
                    chip=ci,
                    plane_path=pp,
                    objective_mag=p.chip_scan_mag,
                    speed_mm=p.scan_speed,
                    gain=p.chip_scan_gain,
                    exposure_ms=p.chip_scan_exposure_ms,
                    white_balance=p.wb,
                    clean=True,
                    quiet=True,
                ),
                pause=args.pause,
                quiet=quiet,
            )
            mark_step(run_dir, checkpoint, sc_key, duration)

        # Summary: chip scan
        _print_chip_scan_summary(chip_idx, chip_scan_dir)

        # Submit background segmentation
        seg_key = f"chip_{chip_idx}_segment"
        if seg_executor is not None and not step_done(checkpoint, seg_key):
            seg_dir = chip_dir / "seg"
            job = _SegJob(chip_idx, seg_key, chip_scan_dir, seg_dir)
            future = seg_executor.submit(_run_chip_seg, job, p.seg.flatfield, p.seg.material, p.seg.jobs_bg)
            future.add_done_callback(lambda f, ci=chip_idx: _on_seg_done(f, ci))
            seg_futures.append((job, future))
            if p.seg.wait:
                # Safe mode: block until this chip's seg completes before next chip
                try:
                    result = future.result()
                    if result.success:
                        mark_step(run_dir, checkpoint, seg_key, result.duration)
                except Exception:
                    pass  # callback already logged

        # ETA for remaining chips
        chips_processed += 1
        chips_remaining = len(chip_indices) - (loop_pos + 1)
        if chips_remaining > 0 and chips_processed > 0:
            avg_s = (time.perf_counter() - chip_loop_start) / chips_processed
            eta_s = avg_s * chips_remaining
            with _always_console():
                print(
                    f"[{loop_pos + 1}/{len(chip_indices)} chips] "
                    f"~{_format_duration_compact(eta_s)} remaining "
                    f"(avg {_format_duration_compact(avg_s)}/chip)"
                )

    # ----------------------------------------------------------------
    # Scans complete
    # ----------------------------------------------------------------
    with _always_console():
        seg_pending = sum(1 for _, f in seg_futures if not f.done())
        seg_note = f", {seg_pending} seg pending" if seg_pending > 0 else ""
        print(f"[scans done] {len(chip_indices)} chips{seg_note}")

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

    # ----------------------------------------------------------------
    # Step 7: Drain background segmentation (boost remaining jobs)
    # ----------------------------------------------------------------
    if seg_futures and seg_executor is not None:
        seg_wall_start = time.perf_counter()
        total_seg_time = _drain_seg(seg_futures, seg_executor, p.seg, checkpoint, run_dir)
        seg_wall = time.perf_counter() - seg_wall_start
        with _always_console():
            print(
                f"[seg done] {len(seg_futures)} chips, "
                f"{_format_duration_compact(total_seg_time)} cpu, "
                f"{_format_duration_compact(seg_wall)} waited after park"
            )
    if seg_executor is not None:
        seg_executor.shutdown(wait=False)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    pipeline_duration = time.perf_counter() - pipeline_start
    t = checkpoint["step_timing"]

    with _always_console():
        print(f"[done] {_format_duration_compact(pipeline_duration)} total")

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
        print(f"{'Switch to ' + p.chip_scan_mag:<30} {format_duration(t.get(switch_key, 0)):>12}")

        for chip_idx in chip_indices:
            fm = t.get(f"chip_{chip_idx}_focus_map", 0)
            an = t.get(f"chip_{chip_idx}_analyze", 0)
            sc = t.get(f"chip_{chip_idx}_scan", 0)
            sg = t.get(f"chip_{chip_idx}_segment", 0)
            print(f"{'  Chip ' + str(chip_idx) + ' focus map':<30} {format_duration(fm):>12}")
            print(f"{'  Chip ' + str(chip_idx) + ' analyze':<30} {format_duration(an):>12}")
            print(f"{'  Chip ' + str(chip_idx) + ' ' + p.chip_scan_mag + ' scan':<30} {format_duration(sc):>12}")
            if sg > 0:
                print(f"{'  Chip ' + str(chip_idx) + ' segment':<30} {format_duration(sg):>12}")

        hw_total = sum(v for k, v in t.items() if "segment" not in k)
        seg_total = sum(v for k, v in t.items() if "segment" in k)
        print("-" * 42)
        print(f"{'Hardware total':<30} {format_duration(hw_total):>12}")
        if seg_total > 0:
            print(f"{'Segmentation total (bg)':<30} {format_duration(seg_total):>12}")
        print(f"{'This invocation':<30} {format_duration(pipeline_duration):>12}")
        print()
        print(f"Output: {run_dir}/")

    # Save args to checkpoint for reference
    checkpoint["args"] = {
        "preset": p.preset_name,
        "area_rect": args.area_rect,
        "initial_z": args.initial_z,
        "overview_mag": p.overview_mag,
        "chip_scan_mag": p.chip_scan_mag,
        "scan_speed": p.scan_speed,
        "scan_z_speed": p.scan_z_speed,
        "chip_scan_gain": p.chip_scan_gain,
        "chip_scan_exposure_ms": p.chip_scan_exposure_ms,
        "focus_map_gain": p.focus_map_gain,
        "focus_map_exposure_ms": p.focus_map_exposure_ms,
        "white_balance": args.white_balance,
        "chips": args.chips,
        "after": args.after,
        "limit": args.limit,
        "seg_jobs_bg": p.seg.jobs_bg,
        "seg_jobs_idle": p.seg.jobs_idle,
        "material": p.seg.material,
        "flatfield": str(p.seg.flatfield) if p.seg.flatfield else None,
    }
    save_checkpoint(run_dir, checkpoint)

    return 0


def _apply_resume(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Load pipeline config from checkpoint for --resume mode.

    Mutates args in place: sets output and all config fields from the
    saved checkpoint. Calls parser.error() on invalid state.
    """
    used = [f for f in sorted(_RESUME_FORBIDDEN_FLAGS) if f in sys.argv]
    if used:
        parser.error(f"--resume cannot be combined with: {', '.join(used)}")

    resume_dir = Path(args.resume)
    cp_path = resume_dir / "checkpoint.json"
    if not cp_path.exists():
        parser.error(f"No checkpoint.json in {resume_dir}")

    with open(cp_path) as f:
        cp = json.load(f)
    saved = cp.get("args")
    if not saved:
        parser.error(f"checkpoint.json in {resume_dir} has no saved args (old format?)")

    args.output = str(resume_dir)
    args.preset = saved.get("preset", DEFAULT_PRESET)
    args.area_rect = saved["area_rect"]
    args.initial_z = saved["initial_z"]
    args.overview_mag = saved["overview_mag"]
    args.chip_scan_mag = saved["chip_scan_mag"]
    args.scan_speed = saved["scan_speed"]
    args.scan_z_speed = saved["scan_z_speed"]
    if "white_balance" in saved:
        args.white_balance = saved["white_balance"]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.resume:
        _apply_resume(args, parser)

    p = _plan(args)

    if args.dry_run:
        _print_header(p)
        run_in_process(f"{p.overview_mag} Overview Scan", lambda: None, dry_run=True)
        run_in_process("Stitch Overview", lambda: None, dry_run=True)
        run_in_process("Detect Chips", lambda: None, dry_run=True)
        run_in_process(f"Switch to {p.chip_scan_mag}", lambda: None, dry_run=True)
        if p.chips_json_path.exists():
            chip_indices = _resolve_chip_indices(p)
            for ci in chip_indices:
                run_in_process(f"Chip {ci} - Focus Map", lambda: None, dry_run=True)
                run_in_process(f"Chip {ci} - Analyze Focus Map", lambda: None, dry_run=True)
                run_in_process(f"Chip {ci} - {p.chip_scan_mag} Scan", lambda: None, dry_run=True)
                if p.seg.enabled:
                    run_in_process(f"Chip {ci} - Segment (bg, {p.seg.jobs_bg}j)", lambda: None, dry_run=True)
        else:
            seg_str = " -> Segment (bg)" if p.seg.enabled else ""
            print(
                f"\n  Per-chip steps: Focus Map -> Analyze -> {p.chip_scan_mag} Scan{seg_str} (chips not yet detected)"
            )
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
