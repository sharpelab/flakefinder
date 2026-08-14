"""Offline background sanity-check audit over completed runs.

Replays the bg-check layers (overview gate, per-chip focus-map gate, seg
monitor) against on-disk run artifacts — no hardware. Used to validate the
check against historical runs and for post-hoc auditing.

Usage:
    uv run python scripts/check_bg.py scans/run_20260807_1630
    uv run python scripts/check_bg.py scans/run_* --material graphene_thin_90nm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flakefinder.scan_utils import parse_white_balance
from flakefinder.segmentation import (
    BGRatios,
    DetectorConfig,
    evaluate_bg_check,
    measure_bg_frame_modes,
    measure_bg_images,
    measure_bg_stitch,
)


def _print_check(
    run_name: str,
    layer: str,
    mag: float | None,
    config: DetectorConfig,
    material: str,
    ratios: BGRatios | None,
) -> None:
    prefix = f"{run_name:22} {layer:18}"
    if ratios is None:
        print(f"{prefix} — no measurement")
        return
    ref = config.bg_reference.get(mag) if mag is not None else None
    if ref is None:
        print(f"{prefix} R/G {ratios.rg:.3f}  B/G {ratios.bg:.3f}  (no reference for {material} at {mag:g}x)")
        return
    res = evaluate_bg_check(ratios, ref)
    print(
        f"{prefix} R/G {res.rg:.3f} (Δ{res.delta_rg:+.3f})  "
        f"B/G {res.bg:.3f} (Δ{res.delta_bg:+.3f})  {res.verdict.upper()}"
    )


def check_run(run_dir: Path, material_override: str | None) -> None:
    ckpt_args: dict = {}
    ckpt_path = run_dir / "checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt_args = json.load(f).get("args", {})

    material = material_override or ckpt_args.get("material")
    if material is None:
        print(f"{run_dir.name}: no material in checkpoint.json (use --material)")
        return
    config = DetectorConfig.from_material(material)

    wb_str = ckpt_args.get("white_balance")
    if wb_str is not None and parse_white_balance(wb_str) != config.white_balance:
        print(f"{run_dir.name}: custom white balance {wb_str} — {material} references do not apply")
        return

    # Overview gate: stitched image + detected chip bboxes
    for stitch_path in sorted(run_dir.glob("overview_*_stitch.jpg")):
        chips_path = stitch_path.with_name(stitch_path.stem + "_chips.json")
        if not chips_path.exists():
            continue
        with open(chips_path) as f:
            chips = json.load(f).get("chips", [])
        mag_str = stitch_path.stem.removeprefix("overview_").removesuffix("_stitch")
        ratios = measure_bg_stitch(stitch_path, chips)
        _print_check(run_dir.name, f"overview {mag_str}", float(mag_str.rstrip("x")), config, material, ratios)

    # Per-chip layers
    fallback_mag = ckpt_args.get("chip_scan_mag")
    for chip_dir in sorted(run_dir.glob("chip_*")):
        chip = chip_dir.name.removeprefix("chip_")

        scan_mag = float(fallback_mag.rstrip("x")) if fallback_mag else None
        for meta_path in chip_dir.glob("scan_*/scan_meta.json"):
            with open(meta_path) as f:
                m = json.load(f).get("optics", {}).get("objective_mag")
            if m is not None:
                scan_mag = float(m)

        # Focus-map gate
        images = sorted((chip_dir / f"focus_map_chip{chip}_images").glob("*.jpg"))
        if images:
            _print_check(run_dir.name, f"chip {chip} gate", scan_mag, config, material, measure_bg_images(images))
        else:
            print(f"{run_dir.name:22} {f'chip {chip} gate':18} — no focus-map images")

        # Seg monitor
        summary_path = chip_dir / "seg" / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                bg_modes = json.load(f).get("bg_mode_by_frame", {})
            ratios = measure_bg_frame_modes(bg_modes)
            _print_check(run_dir.name, f"chip {chip} monitor", scan_mag, config, material, ratios)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline background sanity-check audit (replays all bg-check layers)")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories (scans/run_*)")
    parser.add_argument(
        "--material",
        default=None,
        choices=DetectorConfig.material_names(),
        help="Material preset override (default: from checkpoint.json)",
    )
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        check_run(run_dir, args.material)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
