"""Batch capture a series of camera conditions in one microscope connection.

Runs many captures from a JSON spec without paying per-capture connection
overhead. Intended for calibration sweeps (exposure ramps, lamp ramps,
dark frames) where the sample stays put and only camera/lamp settings vary.

**This command never moves anything.** No stage, no Z drive, no objective
switch. Position and focus the scope first, then run the series. Spec keys
that would imply motion are rejected outright.

Usage:
    sls capture-series -o calibration/cc_20260806/frames --spec ramp.json
    sls capture-series -o out --spec ramp.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image as PILImage

from flakefinder.data_utils import require_microscope_description
from flakefinder.leica.microscope import Microscope
from flakefinder.scan_utils import DEFAULT_WB, build_microscope_meta
from flakefinder.types import GainRGB


@cache
def _binning_index_by_factor() -> dict[int, int]:
    """Map binning factor (NxN) -> SDK binning index, from the hardware description."""
    desc = require_microscope_description()
    return {level.factor: index for index, level in desc.camera.binning_levels.items()}


@cache
def _binning_factor_by_index() -> dict[int, int]:
    """Map SDK binning index -> binning factor (NxN)."""
    return {index: factor for factor, index in _binning_index_by_factor().items()}


# Spec keys that would move hardware. Rejected with a pointed message so a
# stale revisit-style spec can't crash a thick sample into an objective.
MOTION_KEYS = {
    "x",
    "y",
    "z",
    "objective",
    "objective_mag",
    "focus",
    "autofocus",
    "z_range",
    "z_speed",
}

IMAGE_FORMATS = {"png", "tif", "tiff", "jpg", "jpeg"}

TOP_LEVEL_KEYS = {"defaults", "conditions", "notes"}

# Built-in defaults, overridable by the spec's "defaults" block and then
# per-condition. Camera pipeline values match what Camera._init_defaults
# leaves behind, stated explicitly so the spec is self-documenting.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "lamp": 100.0,
    "exposure_ms": 1.0,
    "gain": None,
    "white_balance": {"r": DEFAULT_WB.red, "g": DEFAULT_WB.green, "b": DEFAULT_WB.blue},
    "binning": 3,
    "gamma": 1.0,
    "saturation": 100,
    "auto_brightness": False,
    "aperture": None,
    "frames": 1,
    "warmup_frames": 1,
    "settle_s": 0.2,
    "format": "png",
    "quality": 95,
}

SETTING_KEYS = set(BUILTIN_DEFAULTS)
CONDITION_KEYS = SETTING_KEYS | {"name"}


class Condition(NamedTuple):
    """A fully resolved capture condition (defaults merged with overrides)."""

    name: str
    lamp: float
    exposure_ms: float
    gain: float | None
    white_balance: GainRGB
    binning: int
    gamma: float
    saturation: int
    auto_brightness: bool
    aperture: int | None
    frames: int
    warmup_frames: int
    settle_s: float
    image_format: str
    quality: int

    def filename(self, frame: int) -> str:
        """Output filename for 1-indexed frame number."""
        return f"{self.name}_f{frame}.{self.image_format}"

    def filenames(self) -> list[str]:
        return [self.filename(i) for i in range(1, self.frames + 1)]

    def requested(self) -> dict[str, Any]:
        """Requested settings, for the metadata record."""
        r, g, b = self.white_balance
        return {
            "lamp_pct": self.lamp,
            "exposure_ms": self.exposure_ms,
            "gain": self.gain,
            "white_balance_bgr": [b, g, r],
            "binning": self.binning,
            "gamma": self.gamma,
            "saturation": self.saturation,
            "auto_brightness": self.auto_brightness,
            "aperture": self.aperture,
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "settle_s": self.settle_s,
        }


class Series(NamedTuple):
    """A parsed capture series."""

    conditions: list[Condition]
    notes: Any
    defaults: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return sum(c.frames for c in self.conditions)

    @property
    def warmup_count(self) -> int:
        return sum(c.warmup_frames for c in self.conditions)

    @property
    def exposure_total_s(self) -> float:
        """Total exposure time across all frames, warmups included."""
        return sum((c.frames + c.warmup_frames) * c.exposure_ms / 1000.0 for c in self.conditions)

    @property
    def settle_total_s(self) -> float:
        return sum(c.settle_s for c in self.conditions)


# --- Spec parsing / validation ---


class SpecError(ValueError):
    """Invalid capture series spec."""


def _check_keys(keys: Any, allowed: set[str], where: str) -> None:
    """Reject motion keys and unknown keys with actionable messages."""
    motion = sorted(set(keys) & MOTION_KEYS)
    if motion:
        raise SpecError(
            f"{where} has key '{motion[0]}' — capture-series never moves the stage, "
            f"Z drive, or objective. Position the scope first (sls stage / sls revisit), "
            f"then run the series."
        )
    unknown = sorted(set(keys) - allowed)
    if unknown:
        raise SpecError(f"{where} has unknown key '{unknown[0]}'. Allowed: {', '.join(sorted(allowed))}")


def _as_number(value: Any, field: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{where}: '{field}' must be a number, got {value!r}")
    return float(value)


def _parse_white_balance(value: Any, where: str) -> GainRGB:
    if not isinstance(value, dict):
        raise SpecError(f'{where}: \'white_balance\' must be an object like {{"r": 1.0, "g": 1.0, "b": 1.0}}')
    missing = sorted({"r", "g", "b"} - set(value))
    if missing:
        raise SpecError(f"{where}: 'white_balance' is missing '{missing[0]}' (needs r, g, and b)")
    _check_keys(value, {"r", "g", "b"}, f"{where}: 'white_balance'")
    gains = {k: _as_number(value[k], f"white_balance.{k}", where) for k in ("r", "g", "b")}
    for channel, gain in gains.items():
        if gain <= 0:
            raise SpecError(f"{where}: white_balance.{channel} must be > 0, got {gain}")
    return GainRGB(red=gains["r"], green=gains["g"], blue=gains["b"])


def _resolve_condition(raw: dict[str, Any], defaults: dict[str, Any], index: int) -> Condition:
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SpecError(f"condition #{index}: 'name' is required and must be a non-empty string")
    where = f"condition '{name}'"
    _check_keys(raw, CONDITION_KEYS, where)

    merged = {**defaults, **{k: v for k, v in raw.items() if k != "name"}}

    lamp = _as_number(merged["lamp"], "lamp", where)
    if not 0 <= lamp <= 100:
        raise SpecError(f"{where}: lamp must be 0-100%, got {lamp}")

    exposure_ms = _as_number(merged["exposure_ms"], "exposure_ms", where)
    if exposure_ms <= 0:
        raise SpecError(f"{where}: exposure_ms must be > 0, got {exposure_ms}")

    gain = None
    if merged["gain"] is not None:
        gain = _as_number(merged["gain"], "gain", where)
        if gain <= 0:
            raise SpecError(f"{where}: gain must be > 0, got {gain}")

    binning = merged["binning"]
    valid_binning = sorted(_binning_index_by_factor())
    if binning not in valid_binning:
        raise SpecError(
            f"{where}: binning must be one of {', '.join(str(b) for b in valid_binning)} (NxN), got {binning!r}"
        )

    gamma = _as_number(merged["gamma"], "gamma", where)
    if gamma <= 0:
        raise SpecError(f"{where}: gamma must be > 0, got {gamma}")

    saturation = merged["saturation"]
    if isinstance(saturation, bool) or not isinstance(saturation, int):
        raise SpecError(f"{where}: saturation must be an integer, got {saturation!r}")

    auto_brightness = merged["auto_brightness"]
    if not isinstance(auto_brightness, bool):
        raise SpecError(f"{where}: auto_brightness must be true or false, got {auto_brightness!r}")

    aperture = merged["aperture"]
    if aperture is not None and (isinstance(aperture, bool) or not isinstance(aperture, int)):
        raise SpecError(f"{where}: aperture must be an integer or null, got {aperture!r}")

    frames = merged["frames"]
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise SpecError(f"{where}: frames must be an integer >= 1, got {frames!r}")

    warmup_frames = merged["warmup_frames"]
    if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int) or warmup_frames < 0:
        raise SpecError(f"{where}: warmup_frames must be an integer >= 0, got {warmup_frames!r}")

    settle_s = _as_number(merged["settle_s"], "settle_s", where)
    if settle_s < 0:
        raise SpecError(f"{where}: settle_s must be >= 0, got {settle_s}")

    image_format = merged["format"]
    if not isinstance(image_format, str) or image_format.lower() not in IMAGE_FORMATS:
        raise SpecError(f"{where}: format must be one of {', '.join(sorted(IMAGE_FORMATS))}, got {image_format!r}")

    quality = merged["quality"]
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise SpecError(f"{where}: quality must be an integer 1-100, got {quality!r}")

    return Condition(
        name=name,
        lamp=lamp,
        exposure_ms=exposure_ms,
        gain=gain,
        white_balance=_parse_white_balance(merged["white_balance"], where),
        binning=binning,
        gamma=gamma,
        saturation=saturation,
        auto_brightness=auto_brightness,
        aperture=aperture,
        frames=frames,
        warmup_frames=warmup_frames,
        settle_s=settle_s,
        image_format=image_format.lower(),
        quality=quality,
    )


def parse_series(data: Any) -> Series:
    """Parse and validate a capture series spec.

    Args:
        data: Decoded JSON spec object.

    Returns:
        Series with fully resolved conditions.

    Raises:
        SpecError: If the spec is malformed, out of range, or asks for motion.
    """
    if not isinstance(data, dict):
        raise SpecError("spec must be a JSON object with a 'conditions' list")
    _check_keys(data, TOP_LEVEL_KEYS, "spec")

    raw_defaults = data.get("defaults", {})
    if not isinstance(raw_defaults, dict):
        raise SpecError("spec: 'defaults' must be an object")
    _check_keys(raw_defaults, SETTING_KEYS, "spec: 'defaults'")
    defaults = {**BUILTIN_DEFAULTS, **raw_defaults}

    raw_conditions = data.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise SpecError("spec: 'conditions' must be a non-empty list")

    conditions: list[Condition] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_conditions):
        if not isinstance(raw, dict):
            raise SpecError(f"condition #{i}: must be an object, got {raw!r}")
        cond = _resolve_condition(raw, defaults, i)
        if cond.name in seen:
            raise SpecError(f"duplicate condition name '{cond.name}' — names become filenames and must be unique")
        seen.add(cond.name)
        conditions.append(cond)

    return Series(conditions=conditions, notes=data.get("notes"), defaults=defaults)


def load_series(path: Path) -> Series:
    """Load and validate a series spec from a JSON file."""
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise SpecError(f"{path}: invalid JSON: {e}") from e
    return parse_series(data)


# --- Plan reporting ---

SAFETY_BANNER = "SAFETY: no stage / Z / objective motion — capture-series captures in place."


def format_plan(series: Series) -> str:
    """Render the resolved condition table and totals as text."""
    header = (
        f"{'#':<4} {'Name':<24} {'Lamp%':>6} {'Exp(ms)':>9} {'Gain':>6} "
        f"{'WB(R,G,B)':>18} {'Bin':>4} {'N':>3}  First file"
    )
    lines = [header, "-" * len(header)]
    for i, c in enumerate(series.conditions):
        r, g, b = c.white_balance
        gain_str = f"{c.gain:g}" if c.gain is not None else "--"
        wb_str = f"{r:g},{g:g},{b:g}"
        lines.append(
            f"{i + 1:<4} {c.name:<24} {c.lamp:>6.1f} {c.exposure_ms:>9g} {gain_str:>6} "
            f"{wb_str:>18} {c.binning:>4} {c.frames:>3}  {c.filename(1)}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{len(series.conditions)} conditions, {series.frame_count} frames (+{series.warmup_count} warmup, discarded)"
    )
    lines.append(
        f"Exposure total: {series.exposure_total_s:.1f}s, settle total: {series.settle_total_s:.1f}s "
        f"(excludes readout and file save)"
    )
    return "\n".join(lines)


# --- Execution ---


def _read_applied(scope: Microscope) -> dict[str, Any]:
    """Read back the settings the hardware actually holds right now."""
    camera = scope.camera
    r, g, b = camera.gain_rgb
    return {
        "lamp_pct": scope.lamp.intensity_pct,
        "lamp_native": scope.lamp.intensity,
        "shutter_open": scope.shutter.is_open,
        "exposure_ms": camera.exposure_time * 1000.0,
        "gain": camera.gain,
        "white_balance_bgr": [b, g, r],
        "binning": _binning_factor_by_index().get(camera.binning, camera.binning),
        "gamma": camera.gamma,
        "saturation": camera.saturation,
        "auto_brightness": camera.auto_brightness,
        "aperture": scope.aperture.value,
    }


def _apply(scope: Microscope, cond: Condition, prev: Condition | None) -> None:
    """Apply a condition's settings, skipping anything unchanged from prev."""
    camera = scope.camera

    if prev is None or cond.lamp != prev.lamp:
        scope.lamp.intensity_pct = cond.lamp
    if cond.aperture is not None and (prev is None or cond.aperture != prev.aperture):
        scope.aperture.value = cond.aperture
    if prev is None or cond.binning != prev.binning:
        camera.binning = _binning_index_by_factor()[cond.binning]
    if prev is None or cond.exposure_ms != prev.exposure_ms:
        camera.exposure_time = cond.exposure_ms / 1000.0
    if cond.gain is not None and (prev is None or cond.gain != prev.gain):
        camera.gain = cond.gain
    if prev is None or cond.saturation != prev.saturation:
        camera.saturation = cond.saturation
    if prev is None or cond.gamma != prev.gamma:
        camera.gamma = cond.gamma
    if prev is None or cond.white_balance != prev.white_balance:
        camera.gain_rgb = cond.white_balance
    if prev is None or cond.auto_brightness != prev.auto_brightness:
        camera.auto_brightness = cond.auto_brightness


def _save(path: str, image: np.ndarray, quality: int) -> None:
    img = PILImage.fromarray(image)
    if path.lower().endswith((".jpg", ".jpeg")):
        img.save(path, quality=quality)
    else:
        img.save(path)


def run(
    scope: Microscope,
    *,
    output: str,
    series: Series,
    spec_path: Path | None = None,
    quiet: bool = False,
) -> None:
    """Capture every condition in the series at the current stage position.

    Opens the shutter once, then walks the conditions applying only the
    settings that changed. Never commands the stage, Z drive, or nosepiece.
    """

    def vprint(*a, **kw):
        if not quiet:
            print(*a, **kw)

    os.makedirs(output, exist_ok=True)

    x_um, y_um, z_um = scope.position
    objective_mag = scope.objective_mag

    print(SAFETY_BANNER)
    print(f"        Position: X={x_um:.1f} Y={y_um:.1f} Z={z_um:.1f} µm, {objective_mag:g}x")
    vprint()

    scope.shutter.open()

    # Background save worker — PNG encode overlaps the next exposure.
    save_q: queue.Queue[tuple[str, np.ndarray, int] | None] = queue.Queue()

    def save_worker():
        while True:
            item = save_q.get()
            if item is None:
                break
            _save(*item)

    saver = threading.Thread(target=save_worker, daemon=True)
    saver.start()

    results: list[dict[str, Any]] = []
    prev: Condition | None = None
    t_total_start = time.perf_counter()

    for i, cond in enumerate(series.conditions):
        t_cond_start = time.perf_counter()

        _apply(scope, cond, prev)
        t_setup = time.perf_counter()

        if cond.settle_s > 0:
            time.sleep(cond.settle_s)
        t_settle = time.perf_counter()

        # Warmup frames are discarded: after a settings change the first
        # acquisition can still carry the previous exposure/lamp state.
        for _ in range(cond.warmup_frames):
            scope.camera.capture()
        t_warmup = time.perf_counter()

        applied = _read_applied(scope)

        frame_times: list[float] = []
        frame_records: list[dict[str, Any]] = []
        for frame_no in range(1, cond.frames + 1):
            t_frame = time.perf_counter()
            timestamp = datetime.now().isoformat()
            image = scope.camera.capture()
            frame_times.append(time.perf_counter() - t_frame)
            filename = cond.filename(frame_no)
            save_q.put((os.path.join(output, filename), image, cond.quality))
            frame_records.append({"file": filename, "timestamp": timestamp})

        total_s = time.perf_counter() - t_cond_start
        gain_str = f"{cond.gain:g}" if cond.gain is not None else "--"
        vprint(
            f"[{i + 1}/{len(series.conditions)}] {cond.name}: "
            f"lamp={cond.lamp:g}% exp={cond.exposure_ms:g}ms gain={gain_str} "
            f"-> {cond.frames} frames in {total_s:.2f}s"
        )

        results.append(
            {
                "index": i,
                "name": cond.name,
                "requested": cond.requested(),
                "applied": applied,
                "timing_s": {
                    "setup": round(t_setup - t_cond_start, 4),
                    "settle": round(t_settle - t_setup, 4),
                    "warmup": round(t_warmup - t_settle, 4),
                    "frames": [round(t, 4) for t in frame_times],
                    "total": round(total_s, 4),
                },
                "frames": frame_records,
            }
        )
        prev = cond

    t_save_drain_start = time.perf_counter()
    save_q.put(None)
    saver.join()
    save_tail_s = time.perf_counter() - t_save_drain_start

    total_elapsed = time.perf_counter() - t_total_start

    # Built after the loop so the camera block reflects the last applied state.
    micro_meta = build_microscope_meta(scope)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "command": sys.argv,
        "spec_path": str(spec_path) if spec_path else None,
        "spec_defaults": series.defaults,
        "notes": series.notes,
        "condition_count": len(series.conditions),
        "frame_count": series.frame_count,
        "warmup_frame_count": series.warmup_count,
        "total_elapsed_s": round(total_elapsed, 2),
        "save_tail_s": round(save_tail_s, 2),
        "position_um": {"x": x_um, "y": y_um, "z": z_um},
        "objective_mag": objective_mag,
        **micro_meta,
        "conditions": results,
    }

    meta_path = os.path.join(output, "capture_series_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    n = len(series.conditions)
    if quiet:
        print(f"Captured {series.frame_count} frames over {n} conditions in {total_elapsed:.1f}s -> {output}/")
    else:
        print()
        print("=" * 50)
        print("CAPTURE SERIES SUMMARY")
        print(f"  Conditions: {n}")
        print(f"  Frames: {series.frame_count} (+{series.warmup_count} warmup, discarded)")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / series.frame_count:.2f}s/frame)")
        print(f"  Output: {output}/")
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch capture a series of camera conditions at the current position",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
capture-series NEVER moves the stage, Z drive, or objective. Position and
focus the scope first, then run the series.

Examples:
  sls capture-series -o calibration/cc_20260806/frames --spec white_ramp.json
  sls capture-series -o out --spec white_ramp.json --dry-run
""",
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory (created if missing)")
    parser.add_argument("--spec", type=Path, required=True, help="JSON spec with {defaults?, conditions, notes?}")
    parser.add_argument("--settle", type=float, default=None, help="Override settle_s for every condition")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan and exit (no hardware)")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before starting")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduced output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        series = load_series(args.spec)
    except (SpecError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    if args.settle is not None:
        if args.settle < 0:
            print("Error: --settle must be >= 0")
            return 1
        series = series._replace(conditions=[c._replace(settle_s=args.settle) for c in series.conditions])

    print(f"Loaded {len(series.conditions)} conditions from {args.spec}")
    print()
    print(SAFETY_BANNER)
    print()
    print(format_plan(series))
    print()

    if args.dry_run:
        print("(dry run -- exiting)")
        return 0

    if args.clean and os.path.exists(args.output):
        shutil.rmtree(args.output)
        print(f"Removed existing: {args.output}")

    with Microscope() as scope:
        run(scope, output=args.output, series=series, spec_path=args.spec, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
