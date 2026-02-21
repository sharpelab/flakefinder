"""Parallel flake segmentation over a chip scan directory.

Usage:
    python scripts/segment_chip_scan.py scans/run_20260212_1843/chip_0/scan_20x \
        -o /tmp/seg_run1843 -j 16

Flatfield is auto-detected from scan_meta.json objective magnification
(e.g. 10x -> calibration/flatfield_10x_bin3.npy). Override with --flatfield.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2

from flakefinder.segmentation import (
    Detection,
    DetectorConfig,
    FrameResult,
    draw_detections,
    natural_sort_key,
    process_frame,
    strip_geometry,
)


class SegStats(NamedTuple):
    """Summary statistics from a segmentation run."""

    total_frames: int
    skipped_frames: int
    frames_with_detections: int
    total_detections: int
    tier_1: int
    tier_2: int
    tier_3: int
    duration_s: float


def run(
    scan_dir: Path,
    output: Path,
    *,
    flatfield: Path | None = None,
    material: str = "hbn",
    contrast_offset: float | None = None,
    min_size_um: float | None = None,
    edge_margin: int | None = None,
    dark_frac_cutoff: float = 0.05,
    jobs: int = 16,
    viz: bool = False,
    quiet: bool = False,
) -> SegStats:
    """Run parallel flake segmentation over a chip scan directory.

    Args:
        scan_dir: Directory containing frame_NNNN.jpg files.
        output: Output directory for results.
        flatfield: Flatfield .npy file for correction.
        material: Material preset name.
        jobs: Number of parallel workers.
        quiet: Suppress progress output.

    Returns:
        SegStats with detection counts and timing.

    Raises:
        ValueError: If no frames found in scan_dir or scan_meta.json missing.
    """
    from dataclasses import replace

    # Read pixel size from scan metadata
    scan_meta_path = scan_dir / "scan_meta.json"
    if not scan_meta_path.exists():
        raise ValueError(f"scan_meta.json not found in {scan_dir}")
    with open(scan_meta_path) as f:
        scan_meta = json.load(f)
    pixel_size = scan_meta["optics"]["sample_pixel_x_um"]
    mag = scan_meta["optics"].get("objective_mag")
    if not quiet:
        print(f"pixel_size: {pixel_size} µm/px ({mag or '?'}x, from scan_meta.json)")

    # Auto-detect flatfield from objective magnification
    if flatfield is None and mag is not None:
        from flakefinder.scan_utils import CALIBRATION_DIR

        mag_str = f"{mag}x" if not str(mag).endswith("x") else str(mag)
        candidate = CALIBRATION_DIR / f"flatfield_{mag_str}_bin3.npy"
        if candidate.exists():
            flatfield = candidate
            if not quiet:
                print(f"flatfield: {candidate.name} (auto-detected)")
        elif not quiet:
            print(f"flatfield: none (no {candidate.name} found)")

    # Discover frames
    frames = sorted(scan_dir.glob("frame_*.jpg"), key=natural_sort_key)
    if not frames:
        raise ValueError(f"No frame_*.jpg files found in {scan_dir}")
    if not quiet:
        print(f"Found {len(frames)} frames in {scan_dir}")

    output.mkdir(parents=True, exist_ok=True)
    flatfield_str = str(flatfield) if flatfield else None

    config = DetectorConfig.from_material(material)
    overrides = {}
    if contrast_offset is not None:
        overrides["contrast_offset"] = contrast_offset
    if min_size_um is not None:
        overrides["min_size_um2"] = min_size_um
    if edge_margin is not None:
        overrides["edge_margin_px"] = edge_margin
    if overrides:
        config = replace(config, **overrides)

    # Submit all frames to worker pool
    t0 = time.monotonic()
    results: dict[str, FrameResult] = {}
    total_det_count = 0

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        future_to_name = {}
        for fp in frames:
            fut = pool.submit(
                process_frame,
                str(fp),
                flatfield_str,
                config,
                pixel_size,
                dark_frac_cutoff,
            )
            future_to_name[fut] = fp.stem

        for done_count, fut in enumerate(as_completed(future_to_name), 1):
            result = fut.result()
            results[result.frame_name] = result
            total_det_count += len(result.detections)
            if not quiet and done_count % 100 == 0:
                print(f"  [{done_count}/{len(frames)}] {total_det_count} detections so far...")

    elapsed = time.monotonic() - t0
    if not quiet:
        print(f"Processed {len(frames)} frames in {elapsed:.1f}s ({len(frames) / elapsed:.1f} fps)")

    # Write per-frame JSONs and optional viz
    for fp in frames:
        name = fp.stem
        r = results[name]
        if not r.detections:
            continue

        # Per-frame JSON (geometry only — tier/score live in summary.json)
        geom_dets = [{k: v for k, v in d.items() if k not in ("tier", "score", "classification")} for d in r.detections]
        frame_json = {"frame": name, "dark_frac": round(r.dark_frac, 4), "detections": geom_dets}
        with open(output / f"{name}.json", "w") as f:
            json.dump(frame_json, f, indent=2)

        # Annotated image
        if viz:
            raw = cv2.imread(fp)
            if raw is not None:
                vis = draw_detections(raw, r.detections, um_per_px=pixel_size)
                cv2.imwrite(output / f"{name}.jpg", vis)

    # Build summary
    tier_counts = {1: 0, 2: 0, 3: 0}
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
            # Tag each detection with its frame name
            for d in stripped:
                d["frame"] = name
            all_detections[name] = stripped
            for d in r.detections:
                tier = d.get("tier", 3)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "command": sys.argv,
        "duration_s": round(elapsed, 2),
        "scan_dir": str(scan_dir),
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

    summary_path = output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if not quiet:
        print(f"Saved: {summary_path}")

    if not quiet:
        print(f"\n{'=' * 50}")
        print(f"Total frames:       {len(frames)}")
        print(f"Skipped (dark):     {skipped_count}")
        print(f"Frames w/ dets:     {frames_with_dets}")
        print(f"Total detections:   {total_det_count}")
        print(f"  Tier 1:           {tier_counts[1]}")
        print(f"  Tier 2:           {tier_counts[2]}")
        print(f"  Tier 3:           {tier_counts[3]}")

        all_flat = [d for dets in all_detections.values() for d in dets]
        if all_flat:
            print("\n--- Top 10 by score ---")
            by_score = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))[:10]
            for i, d in enumerate(by_score):
                print(
                    f"  {i + 1}. {d['frame']} "
                    f"score={d.get('score', 0):.3f} "
                    f"tier={d.get('tier', '?')} "
                    f"size={d['size_px']}px "
                    f"cal_dist={d.get('cal_dist', 0):.3f}"
                )

    return SegStats(
        total_frames=len(frames),
        skipped_frames=skipped_count,
        frames_with_detections=frames_with_dets,
        total_detections=total_det_count,
        tier_1=tier_counts[1],
        tier_2=tier_counts[2],
        tier_3=tier_counts[3],
        duration_s=round(elapsed, 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel flake segmentation over a chip scan directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scan_dir", type=Path, help="Scan directory containing frame_NNNN.jpg files")
    parser.add_argument("--flatfield", type=Path, default=None, help="Flatfield .npy file")
    parser.add_argument(
        "--material",
        default="hbn",
        choices=["hbn", "hbn_thin", "hbn_thick", "graphene"],
        help="Material preset",
    )
    parser.add_argument("--contrast-offset", type=float, default=None, help="Override contrast offset from preset")
    parser.add_argument("--min-size-um", type=float, default=None, help="Override min detection area (µm²)")
    parser.add_argument("--edge-margin", type=int, default=None, help="Override edge margin (px)")
    parser.add_argument(
        "--dark-frac-cutoff", type=float, default=0.05, help="Skip frame if dark pixel fraction exceeds this"
    )
    parser.add_argument("-j", "--jobs", type=int, default=16, help="Worker count")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--viz", action="store_true", help="Write annotated frame images (expensive)")
    args = parser.parse_args()

    try:
        run(
            scan_dir=args.scan_dir,
            output=args.output,
            flatfield=args.flatfield,
            material=args.material,
            contrast_offset=args.contrast_offset,
            min_size_um=args.min_size_um,
            edge_margin=args.edge_margin,
            dark_frac_cutoff=args.dark_frac_cutoff,
            jobs=args.jobs,
            viz=args.viz,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
