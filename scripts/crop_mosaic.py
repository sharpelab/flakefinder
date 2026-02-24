"""Flexible detection cropping and mosaic assembly from scan runs.

Usage:
    python scripts/crop_mosaic.py scans/run_20260220_1543/ --tier 1 --top 10
    python scripts/crop_mosaic.py scans/run_20260220_1543/ --ranks 3,6,48
    python scripts/crop_mosaic.py scans/run_20260220_1543/ --where "entropy > 4.4" --chip 0
    python scripts/crop_mosaic.py scans/run_20260220_1543/ --where "G > 1.5" --tier 1 \\
        --label "#{rank} c{chip} G={G:+.2f} ent={ent:.2f}"
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
from crop_util import crop_detection
from mosaic_util import make_mosaic

from flakefinder.segmentation import (
    Detection,
    DetectorConfig,
    dedup_detections,
    score_detections,
)

_SAFE_BUILTINS = {"abs": abs, "max": max, "min": min, "round": round}

DEFAULT_LABEL = "#{rank} c{chip} G={G:+.2f} ent={ent:.2f}"


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def _find_seg_dirs(run_dir: Path, seg_name: str) -> dict[int, Path]:
    """Find seg directories for all chips in a run directory."""
    result: dict[int, Path] = {}
    for chip_dir in sorted(run_dir.glob("chip_*")):
        if not chip_dir.is_dir():
            continue
        try:
            idx = int(chip_dir.name.removeprefix("chip_"))
        except ValueError:
            continue
        seg_dir = chip_dir / seg_name
        if (seg_dir / "summary.json").exists():
            result[idx] = seg_dir
    return result


def _find_scan_dir(chip_dir: Path) -> Path | None:
    """Find the first scan_* subdirectory within a chip directory."""
    for d in sorted(chip_dir.iterdir()):
        if d.is_dir() and d.name.startswith("scan_"):
            return d
    return None


def _det_namespace(det: Detection, rank: int) -> dict:
    """Build a namespace dict from a detection for eval/format."""
    ns = dict(det)
    r, g, b = det["contrast_rgb"]
    ns.update(
        {
            "R": r,
            "G": g,
            "B": b,
            "rank": rank,
            "chip": det.get("chip_idx", 0),
            "nm": det.get("thickness_nm"),
            "ent": det.get("entropy", det.get("g_entropy", 0)),
        }
    )
    return ns


def _eval_where(det: Detection, expr: str) -> bool:
    """Evaluate a filter expression against detection metrics."""
    ns = _det_namespace(det, 0)
    return bool(eval(expr, {"__builtins__": _SAFE_BUILTINS}, ns))


def _format_label(det: Detection, rank: int, fmt: str) -> str:
    """Format a label string using detection metrics."""
    ns = _det_namespace(det, rank)
    return fmt.format_map(ns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop detections and assemble labeled mosaics from scan runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Selection is applied in order: filter -> sort -> pick.

Label format uses Python str.format_map() with detection metrics as keys.
Shortcuts: R, G, B (contrast), ent (entropy), nm (thickness_nm), chip, rank.

Where expressions use Python eval with detection metrics as variables.
Example: --where "G > 1.5" --where "entropy < 4.4"
""",
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing chip_*/seg/")

    # Selection
    sel = parser.add_argument_group("selection")
    sel.add_argument("--tier", type=int, default=None, help="Keep only tier N detections")
    sel.add_argument(
        "--chip", type=_parse_int_list, default=None, help="Keep only these chip indices (comma-separated)"
    )
    sel.add_argument("--where", action="append", default=None, help="Metric filter expression (repeatable, AND'd)")
    sel.add_argument(
        "--ranks", type=_parse_int_list, default=None, help="Pick explicit ranks after filtering (1-indexed)"
    )
    sel.add_argument("--top", type=int, default=10, help="Pick top N (ignored if --ranks is set)")

    # Pipeline
    pipe = parser.add_argument_group("pipeline")
    pipe.add_argument("--seg-name", default="seg", help="Seg subdir name within each chip dir")
    pipe.add_argument(
        "--material",
        default="hbn_medium",
        choices=DetectorConfig.material_names(),
        help="Material scoring preset",
    )
    pipe.add_argument("--no-dedup", action="store_true", help="Skip spatial deduplication")
    pipe.add_argument("--dedup-radius", type=float, default=50.0, help="Dedup radius in um")

    # Output
    out = parser.add_argument_group("output")
    out.add_argument(
        "-o", "--output", type=Path, default=None, help="Output directory (default: <run_dir>/crop_mosaic)"
    )
    out.add_argument("--label", default=DEFAULT_LABEL, help="Label format string")
    out.add_argument("--overlay", choices=["contour", "bbox", "none"], default="contour", help="Overlay on crops")
    out.add_argument("--cols", type=int, default=5, help="Mosaic columns")
    out.add_argument("--pad", type=int, default=250, help="Bbox padding in pixels")
    out.add_argument("--no-mosaic", action="store_true", help="Only produce crops, skip mosaic")

    args = parser.parse_args()

    # --- Load detections ---
    seg_dirs = _find_seg_dirs(args.run_dir, args.seg_name)
    if not seg_dirs:
        print(f"No seg directories found in {args.run_dir}")
        return 1

    config = DetectorConfig.from_material(args.material)
    all_flat: list[Detection] = []
    scan_dirs: dict[int, Path | None] = {}
    um_per_px = 0.36

    for chip_idx, seg_dir in sorted(seg_dirs.items()):
        chip_dir = seg_dir.parent
        scan_dirs[chip_idx] = _find_scan_dir(chip_dir)

        with open(seg_dir / "summary.json") as f:
            summary = json.load(f)
        um_per_px = summary.get("params", {}).get("pixel_size_um", um_per_px)

        for frame_name, dets in summary.get("detections_by_frame", {}).items():
            if not dets:
                continue
            score_detections(dets, config)
            for idx, d in enumerate(dets):
                d["det_id"] = idx
                d.setdefault("frame", frame_name)
                d["chip_idx"] = chip_idx
            all_flat.extend(dets)

    print(f"Loaded {len(all_flat)} detections from {len(seg_dirs)} chips")

    # --- Dedup ---
    if not args.no_dedup and all_flat:
        sorted_all = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        n_before = len(all_flat)
        all_flat = dedup_detections(sorted_all, radius_um=args.dedup_radius)
        print(f"Dedup: {n_before} -> {len(all_flat)} (radius={args.dedup_radius:.0f} um)")

    # --- Filter ---
    if args.tier is not None:
        n_before = len(all_flat)
        all_flat = [d for d in all_flat if d.get("tier") == args.tier]
        print(f"Tier {args.tier}: {n_before} -> {len(all_flat)}")

    if args.chip is not None:
        chip_set = set(args.chip)
        n_before = len(all_flat)
        all_flat = [d for d in all_flat if d.get("chip_idx") in chip_set]
        print(f"Chips {sorted(chip_set)}: {n_before} -> {len(all_flat)}")

    for expr in args.where or []:
        n_before = len(all_flat)
        all_flat = [d for d in all_flat if _eval_where(d, expr)]
        print(f"Where '{expr}': {n_before} -> {len(all_flat)}")

    if not all_flat:
        print("No detections match filters.")
        return 0

    # --- Sort and select ---
    ranked = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))

    if args.ranks:
        selected = [(r, ranked[r - 1]) for r in args.ranks if 1 <= r <= len(ranked)]
        invalid = [r for r in args.ranks if r < 1 or r > len(ranked)]
        if invalid:
            print(f"Warning: ranks {invalid} out of range (1-{len(ranked)})")
    else:
        n = min(args.top, len(ranked))
        selected = [(i + 1, ranked[i]) for i in range(n)]

    # --- Print table ---
    print(f"\nSelected {len(selected)} detections:")
    hdr = f"{'#':>4}  {'chip':>6} {'frame':<16} {'det':>3} {'score':>7} {'G':>7} {'ent':>6}"
    print(hdr)
    for rank, d in selected:
        _, g, _ = d["contrast_rgb"]
        chip_str = f"c{d.get('chip_idx', '?')}"
        ent = d.get("entropy", d.get("g_entropy", 0))
        print(
            f"{rank:>4}  {chip_str:>6} {d['frame']:<16} {d['det_id']:>3} "
            f"{d.get('score', 0):>7.3f} {g:>+7.3f} {ent:>6.2f}"
        )

    # --- Crop ---
    output_dir = args.output or (args.run_dir / "crop_mosaic")
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[str] = []
    crop_labels: list[str] = []

    for rank, d in selected:
        chip_idx = d.get("chip_idx", 0)
        frame_name = d["frame"]
        det_id = d["det_id"]

        scan_dir = scan_dirs.get(chip_idx)
        if scan_dir is None or not scan_dir.is_dir():
            print(f"  Warning: no scan dir for chip_{chip_idx}, skipping crop")
            continue

        frame_path = scan_dir / f"{frame_name}.jpg"
        frame_json_path = seg_dirs[chip_idx] / f"{frame_name}.json"
        crop = crop_detection(frame_path, frame_json_path, det_id, um_per_px, args.pad, args.overlay)
        if crop is None:
            continue

        crop_filename = f"rank{rank:03d}_c{chip_idx}_{frame_name}_d{det_id}.jpg"
        crop_path = crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))
        crop_labels.append(_format_label(d, rank, args.label))

    print(f"\nCropped {len(crop_paths)} detections -> {crops_dir}/")

    # --- Mosaic ---
    if not args.no_mosaic and crop_paths:
        n_cols = args.cols
        n_rows = math.ceil(len(crop_paths) / n_cols)
        mosaic = make_mosaic(
            crop_paths,
            rows=n_rows,
            cols=n_cols,
            max_dim=8000,
            margin=4,
            bg_color=(30, 30, 30),
            labels=crop_labels,
            label_size=16,
            label_color=(255, 255, 255),
            label_bg=(0, 0, 0, 180),
        )
        mosaic_path = output_dir / "mosaic.jpg"
        mosaic.save(str(mosaic_path), quality=95)
        print(f"Saved mosaic: {mosaic_path}")
        subprocess.Popen(
            ["present", str(mosaic_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
