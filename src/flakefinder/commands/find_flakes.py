"""End-to-end flake finding pipeline.

Orchestrates: overview scan → stitch → chip detection →
per-chip focus mapping → plane analysis → chip scanning →
background segmentation.

Overview and chip scan magnifications are configurable (default: 2.5x
overview, 20x chip scan). Scan speeds scale automatically with
magnification.

Segmentation runs in a background thread after each chip scan completes,
Use --seg-jobs to control parallelism (default 4). Use --seg-wait for
a safe baseline that blocks between chips, or --no-segment to disable
entirely.

Calls command modules in-process with a shared Microscope connection.
If any step fails, prints what completed and exits. Output goes under
a single timestamped run directory.

Usage:
    # Full pipeline with default preset (5x overview + 20x chip scan)
    uv run python find_flakes.py

    # Fast screening preset (2.5x overview + 10x chip scan)
    uv run python find_flakes.py --preset 2.5_10

    # Only process chips 0 and 2
    uv run python find_flakes.py --chips 0,2

    # Disable background segmentation
    uv run python find_flakes.py --no-segment

    # Safe seg baseline (blocks between chips)
    uv run python find_flakes.py --seg-wait

    # Preview commands without running
    uv run python find_flakes.py --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Callable
from concurrent.futures import (
    CancelledError,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from flakefinder.cli_utils import park_microscope
from flakefinder.commands import (
    analyze_focus_map,
    chip_scan,
    find_chips,
    focus_map,
    revisit,
    scan,
    stage,
    stitch,
    upload,
)
from flakefinder.data_utils import add_stage_coords
from flakefinder.flakes_api import get_auth
from flakefinder.leica import Microscope
from flakefinder.scan_utils import (
    CALIBRATION_DIR,
    PARFOCAL_Z_UM,
    add_colour_matrix_arg,
    build_lighting_meta,
    build_revisit_json,
    get_git_version,
    parse_area_rect,
    parse_white_balance,
    validate_area_rect,
)
from flakefinder.segmentation import (
    BGCheckResult,
    BGRatios,
    CaptureSettings,
    DetectorConfig,
    FrameResult,
    dedup_detections,
    evaluate_bg_check,
    measure_bg_frame_modes,
    measure_bg_images,
    measure_bg_stitch,
    natural_sort_key,
    process_frame,
    strip_geometry,
)
from flakefinder.types import AreaRect, ColourMatrix, GainRGB

if TYPE_CHECKING:
    from flakefinder.segmentation import Detection


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


class ScanPreset(TypedDict):
    name: str
    overview_mag: str
    chip_scan_mag: str
    chip_scan_speed_mm: float
    chip_scan_gain: float
    chip_scan_exposure_ms: float
    focus_map_gain: float
    focus_map_exposure_ms: float


PRESETS: dict[str, ScanPreset] = {
    "2.5_20": {
        "name": "2.5x overview, 20x scan",
        "overview_mag": "2.5x",
        "chip_scan_mag": "20x",
        "chip_scan_speed_mm": 5.0,
        "chip_scan_gain": 4.0,
        "chip_scan_exposure_ms": 0.25,
        "focus_map_gain": 1.0,
        "focus_map_exposure_ms": 1.0,
    },
    "2.5_10": {
        "name": "2.5x overview, 10x scan",
        "overview_mag": "2.5x",
        "chip_scan_mag": "10x",
        "chip_scan_speed_mm": 10.0,
        "chip_scan_gain": 4.0,
        "chip_scan_exposure_ms": 0.25,
        "focus_map_gain": 1.0,
        "focus_map_exposure_ms": 1.0,
    },
}

DEFAULT_PRESET = "2.5_10"


def _write_run_meta(run_dir: Path, meta: dict) -> None:
    """Write run metadata to run_dir/checkpoint.json (atomic via temp file).

    The on-disk filename is `checkpoint.json` for compatibility with
    `upload.py` and `run_viewer.py`, which read `operator`, `name`,
    `notes`, `args`, `n_chips`, and `step_timing` from it.
    """
    path = run_dir / "checkpoint.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    tmp.replace(path)


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
    jobs: int  # parallel workers for frame processing
    revisit_mags: list[float]  # target magnifications for revisit JSONs
    revisit_top: int | None  # cap on T1 detections for revisit (None = all T1)


@dataclass
class _Preflight:
    """Validated pipeline configuration from _plan()."""

    run_dir: Path
    overview_dir: Path
    stitch_path: Path
    chips_json_path: Path
    chip_filter: list[int] | None
    wb: GainRGB
    colour_matrix: ColourMatrix | None
    area: AreaRect
    preset_name: str
    overview_mag: str
    chip_scan_mag: str
    scan_speed: float
    chip_scan_gain: float
    chip_scan_gain_source: str  # "CLI" | "material" | "preset"
    chip_scan_exposure_ms: float
    chip_scan_exposure_source: str  # "CLI" | "material" | "preset"
    revisit_capture: dict[float, CaptureSettings]  # material overrides by mag
    focus_map_gain: float
    focus_map_exposure_ms: float
    seg: SegConfig
    upload: bool
    substrate: str
    detector_cfg: DetectorConfig  # material preset (bg references, WB, substrate)
    bg_check_active: bool  # False when CLI WB differs from the material preset WB
    args: argparse.Namespace  # raw CLI args for forwarding


def _resolve_flatfield(chip_scan_mag: str) -> Path | None:
    """Auto-detect flatfield file for a chip scan magnification."""
    path = CALIBRATION_DIR / f"flatfield_{chip_scan_mag}_bin3.npy"
    return path if path.exists() else None


def _plan(args: argparse.Namespace) -> _Preflight:
    """Parse and validate pipeline configuration (no hardware).

    Args:
        args: Parsed CLI arguments.

    Returns:
        _Preflight with resolved paths and parsed values.
    """

    material_cfg = DetectorConfig.from_material(args.material)
    if args.white_balance is not None:
        wb = parse_white_balance(args.white_balance)
    else:
        wb = material_cfg.white_balance
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

    # Scan speed from preset; CLI overrides.
    scan_speed = args.scan_speed if args.scan_speed is not None else preset["chip_scan_speed_mm"]

    # Chip-scan camera settings: CLI flag > material preset > ScanPreset.
    if args.chip_scan_gain is not None:
        chip_scan_gain, chip_scan_gain_source = args.chip_scan_gain, "CLI"
    elif material_cfg.chip_scan_gain is not None:
        chip_scan_gain, chip_scan_gain_source = material_cfg.chip_scan_gain, "material"
    else:
        chip_scan_gain, chip_scan_gain_source = preset["chip_scan_gain"], "preset"
    if args.chip_scan_exposure_ms is not None:
        chip_scan_exposure_ms, chip_scan_exposure_source = args.chip_scan_exposure_ms, "CLI"
    elif material_cfg.chip_scan_exposure_ms is not None:
        chip_scan_exposure_ms, chip_scan_exposure_source = material_cfg.chip_scan_exposure_ms, "material"
    else:
        chip_scan_exposure_ms, chip_scan_exposure_source = preset["chip_scan_exposure_ms"], "preset"

    # Segmentation config
    if args.flatfield:
        flatfield = Path(args.flatfield)
    else:
        flatfield = _resolve_flatfield(chip_scan_mag)

    revisit_mags = []
    if args.revisit_mags:
        for m in args.revisit_mags:
            revisit_mags.append(float(m.lower().rstrip("x")))

    seg = SegConfig(
        enabled=not args.no_segment,
        wait=args.seg_wait,
        flatfield=flatfield,
        material=args.material,
        jobs=args.seg_jobs,
        revisit_mags=revisit_mags,
        revisit_top=args.revisit_top,
    )

    # Upload config — substrate is derived from the material preset
    do_upload = args.upload
    substrate = material_cfg.substrate.value
    if do_upload:
        get_auth()  # fail fast if credentials are missing

    return _Preflight(
        run_dir=run_dir,
        overview_dir=overview_dir,
        stitch_path=stitch_path,
        chips_json_path=chips_json_path,
        chip_filter=chip_filter,
        wb=wb,
        colour_matrix=args.colour_matrix,
        area=area,
        preset_name=preset_name,
        overview_mag=overview_mag,
        chip_scan_mag=chip_scan_mag,
        scan_speed=scan_speed,
        chip_scan_gain=chip_scan_gain,
        chip_scan_gain_source=chip_scan_gain_source,
        chip_scan_exposure_ms=chip_scan_exposure_ms,
        chip_scan_exposure_source=chip_scan_exposure_source,
        revisit_capture=material_cfg.revisit_capture,
        focus_map_gain=preset["focus_map_gain"],
        focus_map_exposure_ms=preset["focus_map_exposure_ms"],
        seg=seg,
        upload=do_upload,
        substrate=substrate,
        detector_cfg=material_cfg,
        bg_check_active=wb == material_cfg.white_balance,
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

  # Process chips after chip 3, limit to 2
  uv run python find_flakes.py --after 3 --limit 2
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Run directory for new run (default: scans/run_YYYYMMDD_HHMM/)",
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
        "--chip-scan-gain",
        type=float,
        default=None,
        help="Override preset chip-scan camera gain",
    )
    parser.add_argument(
        "--chip-scan-exposure-ms",
        type=float,
        default=None,
        help="Override preset chip-scan camera exposure in ms",
    )
    parser.add_argument(
        "--white-balance",
        type=str,
        default=None,
        help="White balance as B,G,R gains (default: from material preset)",
    )
    add_colour_matrix_arg(parser)
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Operator name (who is running the scan). Required.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Scan name / description (e.g. 'SF119 A-H'). Required.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional free-text notes stored with the run metadata",
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
    parser.add_argument(
        "--debug-focus-map",
        action="store_true",
        help="Save per-point images and verify sharpness during focus map (slower)",
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
        default="hbn_medium",
        choices=DetectorConfig.material_names(),
        help="Material preset for segmentation (default: hbn_medium)",
    )
    seg_group.add_argument(
        "--seg-jobs",
        type=int,
        default=4,
        help="Segmentation parallel workers (default: 4)",
    )
    seg_group.add_argument(
        "--revisit-mag",
        type=str,
        action="append",
        dest="revisit_mags",
        metavar="MAG",
        help="Target mag for parfocal-adjusted revisit JSON (e.g. 50x). Repeatable.",
    )
    seg_group.add_argument(
        "--revisit-top",
        type=int,
        default=None,
        help="Cap revisit to top N tier-1 detections per chip (default: all T1)",
    )

    # Upload (post-processing, after microscope is released)
    upload_group = parser.add_argument_group("Upload")
    upload_group.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to flakes.sharpelab.science after pipeline completes",
    )
    return parser


def _print_header(p: _Preflight) -> None:
    args = p.args
    print("FlakeFinder Pipeline")
    print("=" * 70)
    print(f"Operator:      {args.operator}")
    print(f"Scan name:     {args.name}")
    print(f"Run directory: {p.run_dir}")
    print(f"Preset:        {p.preset_name}")
    print(f"Overview:      {p.overview_mag}")
    print(f"Chip scan:     {p.chip_scan_mag}")
    print(f"Area rect:     {args.area_rect}")
    print(f"Initial Z:     {args.initial_z} µm")
    print(f"Scan speed:    {p.scan_speed} mm/s")
    print(
        f"Chip camera:   gain={p.chip_scan_gain} [{p.chip_scan_gain_source}], "
        f"exposure={p.chip_scan_exposure_ms}ms [{p.chip_scan_exposure_source}]"
    )
    print(f"AF camera:     gain={p.focus_map_gain}, exposure={p.focus_map_exposure_ms}ms")
    if p.revisit_capture:
        cap_str = ", ".join(
            f"{mag:g}x gain={c.gain:g}/{c.exposure_ms:g}ms" for mag, c in sorted(p.revisit_capture.items())
        )
        print(f"Revisit cam:   {cap_str} [material]; other mags FC defaults")
    else:
        print("Revisit cam:   FC defaults (per-objective)")
    wb_str = f"{p.wb.blue},{p.wb.green},{p.wb.red}"
    wb_source = "preset" if args.white_balance is None else "CLI"
    print(f"White balance: {wb_str} (B,G,R) [{wb_source}]")
    cm_source = "connection default" if p.colour_matrix is None else "CLI"
    print(f"Colour matrix: {p.colour_matrix or ColourMatrix.CCM_5800K} [{cm_source}]")
    if p.chip_filter:
        print(f"Chips:         {p.chip_filter}")
    if args.after is not None:
        print(f"After:         {args.after}")
    if args.limit is not None:
        print(f"Limit:         {args.limit}")
    if p.seg.enabled:
        ff_label = str(p.seg.flatfield) if p.seg.flatfield else "none"
        print(f"Segmentation:  {p.seg.material}, {p.seg.jobs}j")
        print(f"Flatfield:     {ff_label}")
        if p.seg.wait:
            print("Seg mode:      WAIT (blocking)")
        if p.seg.revisit_mags:
            mags = ", ".join(f"{m:g}x" for m in p.seg.revisit_mags)
            top_str = f"top {p.seg.revisit_top} " if p.seg.revisit_top is not None else ""
            print(f"Revisit:       {top_str}T1 → {mags} (capture after scans)")
    else:
        print("Segmentation:  disabled")
    if p.upload:
        print(f"Upload:        flakes.sharpelab.science (substrate={p.substrate})")
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


def _format_illumination(lighting: dict) -> str:
    """Compact one-line illumination summary for console/pipeline.log."""
    return (
        f"IL turret {lighting['il_turret_pos']}/{lighting['il_turret_max']}, "
        f"IL FD {lighting['il_field_diaphragm']}/{lighting['il_field_diaphragm_max']}, "
        f"IL AP {lighting['aperture_value']}/{lighting['aperture_max_value']}, "
        f"DIC {lighting['dic_turret_pos']}/{lighting['dic_turret_max']}, "
        f"TL/IL switch {lighting['tl_il_lamp_switch']}, "
        f"tube port {lighting['tube_port']}/{lighting['tube_port_max']}, "
        f"method {lighting['contrasting_method']}, "
        f"TL FD {lighting['tl_field_diaphragm']}/{lighting['tl_field_diaphragm_max']}, "
        f"TL AP {lighting['tl_aperture_diaphragm']}/{lighting['tl_aperture_diaphragm_max']}, "
        f"TL shutter {'open' if lighting['tl_shutter_open'] else 'closed'}"
    )


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
            conf = plane_data.get("confidence", {})
            if conf.get("low_confidence"):
                reasons = ", ".join(conf["reasons"])
                print(f"[chip {chip_idx}] \u26a0\ufe0f  FOCUS MAP LOW CONFIDENCE: {reasons}")


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
# Background sanity check (docs/illum_sanity_check_plan.md)
# ============================================================================


def _mag_to_float(mag: str) -> float:
    """'10x' → 10.0"""
    return float(mag.lower().rstrip("x"))


def _run_bg_check(
    label: str,
    key: str,
    mag: float,
    measure: Callable[[], BGRatios | None],
    p: _Preflight,
    run_dir: Path,
    run_meta: dict,
) -> None:
    """Evaluate one background sanity-check layer and act on the verdict.

    Catches illumination-path changes the SDK cannot see (manual sliders/
    filters at the stand) by comparing measured substrate background mode
    ratios against golden blank-chip references.

    Reporting only — the check never halts a run. Outcome is always recorded
    in run metadata; ok → pipeline.log line, warn → loud console warning,
    deviant → loud console warning naming the band. Operators decide what to
    do with the run.
    """
    result: BGCheckResult | None = None
    skip_reason: str | None = None
    if not p.bg_check_active:
        skip_reason = "custom white balance (no reference)"
    else:
        ref = p.detector_cfg.bg_reference.get(mag)
        if ref is None:
            skip_reason = f"no bg reference for {p.seg.material} at {mag:g}x"
        else:
            ratios = measure()
            if ratios is None:
                skip_reason = "no background measurement available"
            else:
                result = evaluate_bg_check(ratios, ref)

    checks = run_meta.setdefault("bg_check", {})
    if result is None:
        checks[key] = {"verdict": "skipped", "reason": skip_reason}
        _write_run_meta(run_dir, run_meta)
        print(f"[bg {label}] skipped: {skip_reason}")
        return

    checks[key] = dict(result._asdict())
    _write_run_meta(run_dir, run_meta)
    line = f"R/G {result.rg:.3f} (Δ{result.delta_rg:+.3f}), B/G {result.bg:.3f} (Δ{result.delta_bg:+.3f})"
    if result.verdict == "ok":
        print(f"[bg {label}] {line} OK")
    elif result.verdict == "warn":
        with _always_console():
            print(f"[bg {label}] ⚠️  BACKGROUND WARN: {line} — check the illumination path")
    else:
        with _always_console():
            print(f"[bg {label}] \U0001f6d1 BACKGROUND DEVIANT: {line}")
            print(
                f"[bg {label}] Substrate colour is outside the deviant band for "
                f"{p.seg.material} at {mag:g}x — the illumination path may have "
                "been changed at the stand (manual slider/filter, invisible to the SDK)."
            )
            print(f"[bg {label}] See docs/illum_sanity_check_plan.md. Continuing.")


# ============================================================================
# Background segmentation
# ============================================================================


@dataclass
class _SegJob:
    """Tracks one chip's segmentation work."""

    chip_idx: int
    scan_dir: Path  # chip scan directory (frame_NNNN.jpg)
    seg_dir: Path  # segmentation output directory
    plane_path: Path | None  # focus plane JSON for revisit Z computation


class _SegResult(NamedTuple):
    success: bool
    duration: float
    summary: str  # one-line summary for console
    error: str  # error message on failure


def _generate_revisits(
    job: _SegJob,
    all_detections: dict[str, list[Detection]],
    material: str,
    revisit_mags: list[float],
    revisit_top: int | None,
) -> str:
    """Generate revisit JSONs from segmentation results. Returns summary string."""

    # Flatten (stage coords already in detections from _run_chip_seg)
    all_flat = [d for dets in all_detections.values() for d in dets]

    # Rank by (tier asc, score desc), dedup, filter to T1
    ranked = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
    ranked = dedup_detections(ranked)
    ranked = [d for d in ranked if d.get("tier") == 1]
    if revisit_top is not None:
        ranked = ranked[:revisit_top]

    # Load focus plane
    assert job.plane_path is not None
    with open(job.plane_path) as f:
        plane_data = json.load(f)
    plane = plane_data["plane"]
    a, b, c = plane["a"], plane["b"], plane["c"]
    scan_mag = plane_data.get("source", {}).get("objective_mag")
    if scan_mag is None or scan_mag not in PARFOCAL_Z_UM:
        return ""

    # Build base points at scan magnification
    base_points = []
    for i, d in enumerate(ranked):
        sx = d.get("stage_x")
        sy = d.get("stage_y")
        if sx is None or sy is None:
            continue
        sx, sy = float(sx), float(sy)
        z = a * sx + b * sy + c
        label = f"rank{i + 1:02d}_{d['frame']}_d{d['det_id']}"
        base_points.append({"x": round(sx, 2), "y": round(sy, 2), "z": round(z, 2), "label": label})

    if not base_points:
        return ""

    # Write parfocal + parcentric adjusted revisit JSONs
    written = []
    for target_mag in revisit_mags:
        if target_mag not in PARFOCAL_Z_UM:
            continue
        revisit_obj = build_revisit_json(base_points, scan_mag, target_mag)
        revisit_obj["material"] = material
        revisit_obj["plane_source"] = str(job.plane_path)
        mag_label = f"{target_mag:g}"
        revisit_path = job.seg_dir / f"revisit_{mag_label}x.json"
        with open(revisit_path, "w") as f:
            json.dump(revisit_obj, f, indent=2)
        written.append(f"{mag_label}x")

    if written:
        return f"revisit {len(base_points)}pts → {', '.join(written)}"
    return ""


def _run_chip_seg(
    job: _SegJob,
    flatfield: Path | None,
    material: str,
    jobs: int,
    revisit_mags: list[float],
    revisit_top: int | None,
    bg_check_active: bool = True,
) -> _SegResult:
    """Run segmentation for one chip in-process. Called in background thread."""
    start = time.perf_counter()
    try:
        # Read pixel size + objective mag from scan metadata
        pixel_size = 0.36  # fallback for 20x bin3
        objective_mag: float | None = None
        scan_meta_path = job.scan_dir / "scan_meta.json"
        if scan_meta_path.exists():
            with open(scan_meta_path) as f:
                sm = json.load(f)
            pixel_size = sm.get("optics", {}).get("sample_pixel_x_um", pixel_size)
            mag = sm.get("optics", {}).get("objective_mag")
            objective_mag = float(mag) if mag is not None else None

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
            geom_dets = [{k: v for k, v in d.items() if k not in ("tier", "score")} for d in r.detections]
            frame_json = {"frame": name, "dark_frac": round(r.dark_frac, 4), "detections": geom_dets}
            with open(job.seg_dir / f"{name}.json", "w") as f:
                json.dump(frame_json, f, indent=2)

        # Build summary
        tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        skipped_count = 0
        frames_with_dets = 0
        all_detections: dict[str, list[Detection]] = {}

        for fp in frames:
            name = fp.stem
            r = results[name]
            if r.skipped:
                skipped_count += 1
            if r.detections:
                frames_with_dets += 1
                stripped = [strip_geometry(d) for d in r.detections]
                for i, d in enumerate(stripped):
                    d["frame"] = name
                    d["det_id"] = i
                all_detections[name] = stripped
                for d in r.detections:
                    tier = d.get("tier", 3)
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Add stage coordinates to all detections (with rolling shutter correction)
        if all_detections and scan_meta_path.exists():
            with open(scan_meta_path) as f:
                scan_meta = json.load(f)
            all_flat = [d for dets in all_detections.values() for d in dets]
            add_stage_coords(all_flat, scan_meta)

        # Per-frame background modes (local clip ceiling, substrate
        # fingerprint, lamp-drift visibility for downstream consumers)
        bg_mode_by_frame = {}
        for fp in frames:
            bg = results[fp.stem].bg_mode_rgb
            if bg is not None:
                bg_mode_by_frame[fp.stem] = list(bg)

        # Background monitor: per-chip median mode ratios vs golden reference
        # (catches mid-run illumination drift; see docs/illum_sanity_check_plan.md)
        bg_marker = ""
        ratios = measure_bg_frame_modes(bg_mode_by_frame)
        bg_check: dict = {"verdict": "skipped"}
        if ratios is not None:
            bg_check["rg"] = round(ratios.rg, 4)
            bg_check["bg"] = round(ratios.bg, 4)
        ref = config.bg_reference.get(objective_mag) if objective_mag is not None else None
        if ratios is None:
            bg_check["reason"] = "no background modes"
        elif not bg_check_active:
            bg_check["reason"] = "custom white balance (no reference)"
        elif ref is None:
            bg_check["reason"] = f"no bg reference for {material} at {objective_mag:g}x"
        else:
            res = evaluate_bg_check(ratios, ref)
            bg_check = dict(res._asdict())
            if res.verdict != "ok":
                tag = "DEVIANT" if res.verdict == "abort" else "WARN"
                bg_marker = (
                    f", ⚠️ bg R/G {res.rg:.3f} (Δ{res.delta_rg:+.3f}) B/G {res.bg:.3f} (Δ{res.delta_bg:+.3f}) {tag}"
                )

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
            "bg_mode_by_frame": bg_mode_by_frame,
            "bg_check": bg_check,
            "detections_by_frame": all_detections,
        }

        with open(job.seg_dir / "summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        # Generate revisit JSONs if configured
        revisit_str = ""
        if revisit_mags and job.plane_path and total_det_count > 0:
            revisit_str = _generate_revisits(
                job,
                all_detections,
                material,
                revisit_mags,
                revisit_top,
            )

        summary_str = (
            f"{len(frames)} frames, "
            f"{total_det_count} det "
            f"(T1:{tier_counts[1]} T2:{tier_counts[2]} "
            f"T3:{tier_counts[3]}), {seg_elapsed:.1f}s @ {jobs}j"
        )
        if revisit_str:
            summary_str += f", {revisit_str}"
        summary_str += bg_marker

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
    except CancelledError:
        pass  # will be re-submitted at boosted worker count
    except Exception as e:
        with _always_console():
            print(f"[seg chip {chip_idx}] ERROR: {e}")


def _drain_seg(
    seg_futures: list[tuple[_SegJob, Future]],
    step_timing: dict[str, float],
    run_dir: Path,
    run_meta: dict,
) -> float:
    """Wait for all seg futures, recording timings.

    Returns total segmentation wall-clock time across all chips.
    """
    total_seg_time = 0.0
    for job, future in seg_futures:
        try:
            result = future.result()
            total_seg_time += result.duration
            if result.success:
                step_timing[f"chip_{job.chip_idx}_segment"] = result.duration
                _write_run_meta(run_dir, run_meta)
        except Exception:
            pass  # callback already logged
    return total_seg_time


def _run_revisit_phase(
    scope: Microscope,
    seg_futures: list[tuple[_SegJob, Future]],
    seg: SegConfig,
    chip_indices: list[int],
    run_dir: Path,
    step_timing: dict[str, float],
    run_meta: dict,
    p: _Preflight,
) -> tuple[float, float]:
    """Drain seg and run revisit captures, overlapping where possible.

    Iterates magnifications (one at a time).  Within each mag, visits each
    chip's detections as soon as its seg completes — overlapping microscope
    I/O with background seg CPU work.

    Returns (total_seg_cpu_time, total_revisit_time).
    """
    # Map chip_idx → (job, future) for lookup
    job_by_chip: dict[int, tuple[_SegJob, Future]] = {}
    for job, future in seg_futures:
        job_by_chip[job.chip_idx] = (job, future)

    total_seg_time = 0.0
    total_revisit_time = 0.0
    seg_done: set[int] = set()

    def _wait_for_seg(chip_idx: int) -> None:
        """Block until chip's seg is done, record timing if successful."""
        nonlocal total_seg_time
        if chip_idx not in job_by_chip or chip_idx in seg_done:
            return
        job, future = job_by_chip[chip_idx]
        seg_done.add(chip_idx)
        try:
            result = future.result()
            total_seg_time += result.duration
            if result.success:
                step_timing[f"chip_{job.chip_idx}_segment"] = result.duration
                _write_run_meta(run_dir, run_meta)
        except Exception:
            pass  # _on_seg_done callback already logged

    # Process revisits per magnification (one mag at a time)
    for mag in seg.revisit_mags:
        mag_label = f"{mag:g}x"

        with _always_console():
            print(f"\n[revisit] {mag_label}: {len(chip_indices)} chip(s)")

        for ci in chip_indices:
            # Wait for seg to finish for this chip
            _wait_for_seg(ci)

            # Load revisit JSON
            chip_dir = run_dir / f"chip_{ci}"
            revisit_json = chip_dir / "seg" / f"revisit_{mag_label}.json"
            output_dir = chip_dir / f"revisit_{mag_label}"

            key = f"chip_{ci}_revisit_{mag_label}"
            if not revisit_json.exists():
                with _always_console():
                    print(f"[revisit chip {ci}] No {mag_label} detections, skipping")
                step_timing[key] = 0
                _write_run_meta(run_dir, run_meta)
                continue

            revisit_file = revisit._parse_points_file(revisit_json)
            n_pts = len(revisit_file.points)

            t_start = time.perf_counter()
            revisit.run(
                scope,
                output=str(output_dir),
                points=revisit_file.points,
                objective_mag=mag_label,
                material=p.seg.material,
                white_balance=p.wb,
                colour_matrix=p.colour_matrix,
                quiet=True,
            )
            duration = time.perf_counter() - t_start

            with _always_console():
                print(f"[revisit chip {ci}] {mag_label}: {n_pts} pts, {_format_duration_compact(duration)}")

            step_timing[key] = duration
            _write_run_meta(run_dir, run_meta)
            total_revisit_time += duration

    # Drain any remaining seg futures (chips not involved in revisit)
    for ci in chip_indices:
        _wait_for_seg(ci)

    return total_seg_time, total_revisit_time


def run(scope: Microscope, p: _Preflight) -> int:
    """Execute the flake-finding pipeline with a live Microscope.

    Returns:
        0 on success (non-zero exits via sys.exit from run_in_process).
    """
    validate_area_rect(p.area, scope.stage)

    args = p.args
    quiet = args.quiet
    run_dir = p.run_dir

    # Illumination-path snapshot at pipeline start (lamp/shutter state here
    # is pre-light_on; per-scan scan_meta.json records the lit state).
    lighting = build_lighting_meta(scope)

    with _always_console():
        print(f"[run] {run_dir}/")
        print(f"[illum] {_format_illumination(dict(lighting))}")

    step_timing: dict[str, float] = {}
    run_meta: dict = {
        "illumination": dict(lighting),
        "operator": args.operator,
        "name": args.name,
        "step_timing": step_timing,
        "n_chips": None,
        "args": {
            "git_version": get_git_version(),
            "preset": p.preset_name,
            "area_rect": args.area_rect,
            "initial_z": args.initial_z,
            "overview_mag": p.overview_mag,
            "chip_scan_mag": p.chip_scan_mag,
            "scan_speed": p.scan_speed,
            "chip_scan_gain": p.chip_scan_gain,
            "chip_scan_exposure_ms": p.chip_scan_exposure_ms,
            "focus_map_gain": p.focus_map_gain,
            "focus_map_exposure_ms": p.focus_map_exposure_ms,
            "white_balance": f"{p.wb.blue},{p.wb.green},{p.wb.red}",
            "colour_matrix": p.colour_matrix.value if p.colour_matrix else None,
            "chips": args.chips,
            "after": args.after,
            "limit": args.limit,
            "seg_jobs": p.seg.jobs,
            "material": p.seg.material,
            "flatfield": str(p.seg.flatfield) if p.seg.flatfield else None,
        },
    }
    if args.notes:
        run_meta["notes"] = args.notes

    print(f"Operator:      {args.operator}")
    print(f"Scan name:     {args.name}")
    if args.notes:
        print(f"Notes:         {args.notes}")

    _write_run_meta(run_dir, run_meta)

    def _record(key: str, duration: float) -> None:
        step_timing[key] = duration
        _write_run_meta(run_dir, run_meta)

    # Background segmentation executor (max 1 concurrent seg job)
    seg_executor: ThreadPoolExecutor | None = None
    seg_futures: list[tuple[_SegJob, Future]] = []
    if p.seg.enabled:
        seg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seg")

    pipeline_start = time.perf_counter()

    # ----------------------------------------------------------------
    # Step 1: Overview scan
    # ----------------------------------------------------------------
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
            colour_matrix=p.colour_matrix,
            clean=True,
            quiet=quiet,
        ),
        pause=args.pause,
        quiet=quiet,
    )
    _record("overview_scan", duration)

    with _always_console():
        meta_path = p.overview_dir / "scan_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                scan_meta = json.load(f)
            frames = scan_meta.get("frame_count", "?")
            rows = len(scan_meta.get("lines", []))
            print(f"[overview] {frames} frames, {rows} rows, {_format_duration_compact(duration)}")

    # ----------------------------------------------------------------
    # Step 2: Stitch overview
    # ----------------------------------------------------------------
    duration, _ = run_in_process(
        "Stitch Overview",
        lambda: stitch.run(scan_dir=p.overview_dir, quiet=True),
        pause=args.pause,
        quiet=quiet,
    )
    _record("stitch", duration)

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
    duration, result = run_in_process(
        "Detect Chips",
        lambda: find_chips.run(image_path=p.stitch_path),
        pause=args.pause,
        quiet=quiet,
    )
    n_chips = len(result.get("chips", []))
    run_meta["n_chips"] = n_chips
    _record("detect_chips", duration)

    with _always_console():
        with open(p.chips_json_path) as f:
            chips_data = json.load(f)
        print(f"[detect] {len(chips_data.get('chips', []))} chips")

    # Background sanity check on the stitched overview (reporting only)
    _run_bg_check(
        "overview",
        "overview",
        _mag_to_float(p.overview_mag),
        lambda: measure_bg_stitch(p.stitch_path, chips_data.get("chips", [])),
        p,
        run_dir,
        run_meta,
    )

    # ----------------------------------------------------------------
    # Step 4: Switch to chip scan objective
    # ----------------------------------------------------------------
    switch_key = f"switch_{p.chip_scan_mag}"

    def _switch_and_set_z():
        stage.run(scope=scope, objective_mag=p.chip_scan_mag)
        # Override SDK parfocal Z with operator's initial_z
        stage.run(scope=scope, z=args.initial_z)

    duration, _ = run_in_process(
        f"Switch to {p.chip_scan_mag}",
        _switch_and_set_z,
        pause=args.pause,
        quiet=quiet,
    )
    _record(switch_key, duration)

    # ----------------------------------------------------------------
    # Load chip data and apply filters
    # ----------------------------------------------------------------
    chip_indices = _resolve_chip_indices(p)

    # ----------------------------------------------------------------
    # Per-chip loop
    # ----------------------------------------------------------------
    chip_loop_start = time.perf_counter()
    prev_chip_z: float | None = None  # chain Z between chips

    for loop_pos, chip_idx in enumerate(chip_indices):
        chip_dir = run_dir / f"chip_{chip_idx}"
        focus_map_path = chip_dir / f"focus_map_chip{chip_idx}.json"
        plane_path = chip_dir / f"focus_map_chip{chip_idx}_plane.json"
        chip_scan_dir = chip_dir / f"scan_{p.chip_scan_mag}"

        print(f"\n{'#' * 70}")
        print(f"# CHIP {chip_idx}")
        print(f"{'#' * 70}")

        # Chain Z from previous chip's focus map
        if prev_chip_z is not None:
            with _always_console():
                print(f"[chain] Z → {prev_chip_z:.0f} µm from previous chip")
            stage.run(scope=scope, z=prev_chip_z)

        # Step 5a: Focus map
        duration, _ = run_in_process(
            f"Chip {chip_idx} - Focus Map",
            lambda ci=chip_idx, cd=chip_dir: focus_map.run(
                scope=scope,
                chips_meta=p.chips_json_path,
                gain=p.focus_map_gain,
                exposure_ms=p.focus_map_exposure_ms,
                chip=ci,
                save_best_image=True,
                af_settle=0.2 if args.debug_focus_map else 0,
                move_to_best_z=args.debug_focus_map,
                white_balance=p.wb,
                colour_matrix=p.colour_matrix,
                output_dir=cd,
                quiet=True,
            ),
            pause=args.pause,
            quiet=quiet,
        )
        _record(f"chip_{chip_idx}_focus_map", duration)

        # Extract Z for chaining to next chip
        if focus_map_path.exists():
            with open(focus_map_path) as f:
                prev_chip_z = json.load(f).get("grid_params", {}).get("z_start_um")

        # Step 5b: Analyze focus map + export plane
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
        _record(f"chip_{chip_idx}_analyze", duration)

        # Summary: focus_map (after analyze exports plane)
        _print_focus_map_summary(chip_idx, plane_path)

        # Background gate: focus-map best-AF images vs golden reference
        fm_images_dir = chip_dir / f"focus_map_chip{chip_idx}_images"
        _run_bg_check(
            f"chip {chip_idx}",
            f"chip_{chip_idx}",
            _mag_to_float(p.chip_scan_mag),
            lambda d=fm_images_dir: measure_bg_images(sorted(d.glob("*.jpg"))),
            p,
            run_dir,
            run_meta,
        )

        # Step 5c: Chip scan
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
                colour_matrix=p.colour_matrix,
                clean=True,
                quiet=True,
            ),
            pause=args.pause,
            quiet=quiet,
        )
        _record(f"chip_{chip_idx}_scan", duration)

        # Summary: chip scan
        _print_chip_scan_summary(chip_idx, chip_scan_dir)

        # Submit background segmentation
        if seg_executor is not None:
            seg_dir = chip_dir / "seg"
            job = _SegJob(chip_idx, chip_scan_dir, seg_dir, plane_path)
            future = seg_executor.submit(
                _run_chip_seg,
                job,
                p.seg.flatfield,
                p.seg.material,
                p.seg.jobs,
                p.seg.revisit_mags,
                p.seg.revisit_top,
                p.bg_check_active,
            )
            future.add_done_callback(lambda f, ci=chip_idx: _on_seg_done(f, ci))
            seg_futures.append((job, future))
            if p.seg.wait:
                # Safe mode: block until this chip's seg completes before next chip
                try:
                    result = future.result()
                    if result.success:
                        _record(f"chip_{chip_idx}_segment", result.duration)
                except Exception:
                    pass  # callback already logged

        # ETA for remaining chips
        chips_processed = loop_pos + 1
        chips_remaining = len(chip_indices) - chips_processed
        if chips_remaining > 0:
            avg_s = (time.perf_counter() - chip_loop_start) / chips_processed
            eta_s = avg_s * chips_remaining
            with _always_console():
                print(
                    f"[{chips_processed}/{len(chip_indices)} chips] "
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
    # Step 6+7: Seg drain + revisit captures, or park + seg drain
    # ----------------------------------------------------------------
    has_revisits = bool(p.seg.revisit_mags) and seg_futures and seg_executor is not None

    if has_revisits:
        assert seg_executor is not None
        # Revisit path: drain seg interleaved with revisit captures, then park
        seg_wall_start = time.perf_counter()
        total_seg_time, total_revisit_time = _run_revisit_phase(
            scope,
            seg_futures,
            p.seg,
            chip_indices,
            run_dir,
            step_timing,
            run_meta,
            p,
        )
        seg_wall = time.perf_counter() - seg_wall_start
        with _always_console():
            print(
                f"[seg+revisit done] seg {_format_duration_compact(total_seg_time)} cpu, "
                f"revisit {_format_duration_compact(total_revisit_time)}, "
                f"{_format_duration_compact(seg_wall)} wall"
            )
        seg_executor.shutdown(wait=False)

        # Park after revisits
        duration, _ = run_in_process(
            "Park Microscope",
            lambda: park_microscope(scope),
            pause=args.pause,
            quiet=quiet,
        )
        _record("park", duration)
    else:
        # No-revisit path: park first, then drain seg
        duration, _ = run_in_process(
            "Park Microscope",
            lambda: park_microscope(scope),
            pause=args.pause,
            quiet=quiet,
        )
        _record("park", duration)

        if seg_futures and seg_executor is not None:
            seg_wall_start = time.perf_counter()
            total_seg_time = _drain_seg(seg_futures, step_timing, run_dir, run_meta)
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
    t = step_timing

    with _always_console():
        print(f"[scan done] {_format_duration_compact(pipeline_duration)} total")

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
            for mag in p.seg.revisit_mags:
                mag_label = f"{mag:g}x"
                rv = t.get(f"chip_{chip_idx}_revisit_{mag_label}", 0)
                if rv > 0:
                    print(f"{'  Chip ' + str(chip_idx) + ' revisit ' + mag_label:<30} {format_duration(rv):>12}")

        hw_total = sum(v for k, v in t.items() if "segment" not in k and "revisit" not in k)
        seg_total = sum(v for k, v in t.items() if "segment" in k)
        revisit_total = sum(v for k, v in t.items() if "revisit" in k)
        print("-" * 42)
        print(f"{'Hardware total':<30} {format_duration(hw_total):>12}")
        if seg_total > 0:
            print(f"{'Segmentation total (bg)':<30} {format_duration(seg_total):>12}")
        if revisit_total > 0:
            print(f"{'Revisit total':<30} {format_duration(revisit_total):>12}")
        print(f"{'This invocation':<30} {format_duration(pipeline_duration):>12}")
        print()
        print(f"Output: {run_dir}/")

    _write_run_meta(run_dir, run_meta)

    return 0


def main() -> int:
    # On Windows, CTRL_BREAK_EVENT sends SIGBREAK which terminates by default.
    # Re-register it to raise KeyboardInterrupt so graceful shutdown works.
    if sys.platform == "win32":
        import signal

        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    parser = _build_parser()
    args = parser.parse_args()

    if not args.operator:
        parser.error("--operator is required")
    if not args.name:
        parser.error("--name is required")

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
                    run_in_process(f"Chip {ci} - Segment ({p.seg.jobs}j)", lambda: None, dry_run=True)
            if p.seg.revisit_mags:
                for mag in p.seg.revisit_mags:
                    mag_label = f"{mag:g}x"
                    top_str = f"top {p.seg.revisit_top} " if p.seg.revisit_top is not None else ""
                    run_in_process(
                        f"Revisit {mag_label} ({len(chip_indices)} chips, {top_str}T1)",
                        lambda: None,
                        dry_run=True,
                    )
        else:
            seg_str = " -> Segment (bg)" if p.seg.enabled else ""
            revisit_str = ""
            if p.seg.revisit_mags:
                mags = ", ".join(f"{m:g}x" for m in p.seg.revisit_mags)
                revisit_str = f" -> Revisit ({mags})"
            print(
                f"\n  Per-chip steps: Focus Map -> Analyze -> {p.chip_scan_mag} Scan"
                f"{seg_str}{revisit_str} (chips not yet detected)"
            )
        if p.upload:
            run_in_process(f"Upload to flakes.sharpelab.science (substrate={p.substrate})", lambda: None, dry_run=True)
        return 0

    # Create run directory and set up log tee
    main_start = time.perf_counter()
    p.run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(p.run_dir / "pipeline.log", "a")  # noqa: SIM115
    quiet = args.quiet
    sys.stdout = TeeWriter(sys.__stdout__, log_file, suppress_console=quiet)
    sys.stderr = TeeWriter(sys.__stderr__, log_file, suppress_console=quiet)

    try:
        _print_header(p)
        rc = 1
        with Microscope() as scope:
            try:
                rc = run(scope, p)
            except KeyboardInterrupt:
                with _always_console():
                    print("\n[interrupted] Parking microscope...")
                    try:
                        park_microscope(scope)
                        print("[parked]")
                    except Exception as e:
                        import traceback

                        print(f"[park failed] {type(e).__name__}: {e}")
                        traceback.print_exc()
                return 1

        # Upload runs after microscope is released
        if rc == 0 and p.upload:
            try:
                t_upload = time.perf_counter()
                with _always_console():
                    upload.run(
                        p.run_dir,
                        material=DetectorConfig.from_material(p.seg.material).name,
                        substrate=p.substrate,
                        quiet=quiet,
                    )
                upload_duration = time.perf_counter() - t_upload
                meta_path = p.run_dir / "checkpoint.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta.setdefault("step_timing", {})["upload"] = upload_duration
                    _write_run_meta(p.run_dir, meta)
            except Exception as e:
                with _always_console():
                    print(f"[upload] FAILED: {e}")

        with _always_console():
            total_duration = time.perf_counter() - main_start
            print(f"[done] {_format_duration_compact(total_duration)} total")

        return rc
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Log: {p.run_dir / 'pipeline.log'}")


if __name__ == "__main__":
    sys.exit(main())
