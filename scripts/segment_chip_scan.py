"""Parallel flake segmentation over a chip scan directory.

Usage:
    python scripts/segment_chip_scan.py scans/run_20260212_1843/chip_0/scan_20x \
        --flatfield calibration/flatfield_20x_bin3.npy \
        -o /tmp/seg_run1843 -j 16
"""

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from flakefinder.scan_utils import apply_flatfield


class FrameResult(NamedTuple):
    frame_name: str
    detections: list[dict]
    dark_frac: float
    skipped: bool


def _process_frame(
    frame_path: str,
    flatfield_path: str | None,
    contrast_offset: float,
    min_size: int,
    edge_margin: int,
    dark_frac_cutoff: float,
) -> FrameResult:
    """Worker: load image, apply flatfield, segment. Imports inside worker for pickling."""
    from segment_flakes import compute_dark_frac, segment_frame

    frame_name = Path(frame_path).stem
    raw = cv2.imread(frame_path)
    if raw is None:
        return FrameResult(frame_name, [], 0.0, True)

    if flatfield_path:
        ff = np.load(flatfield_path).astype(np.float32)
        ff = ff[:, :, ::-1]  # RGB -> BGR
        corrected = apply_flatfield(raw, ff)
    else:
        corrected = raw

    dark_frac = compute_dark_frac(corrected)
    if dark_frac > dark_frac_cutoff:
        return FrameResult(frame_name, [], dark_frac, True)

    detections = segment_frame(
        corrected,
        contrast_offset=contrast_offset,
        min_size_px=min_size,
        edge_margin_px=edge_margin,
    )
    return FrameResult(frame_name, detections, dark_frac, False)


def _natural_sort_key(path: Path) -> int:
    """Extract frame number for natural sorting."""
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else 0


def _strip_geometry(det: dict) -> dict:
    """Return detection dict without hull/contour (large point lists)."""
    return {k: v for k, v in det.items() if k not in ("hull", "contour")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel flake segmentation over a chip scan directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scan_dir", type=Path, help="Scan directory containing frame_NNNN.jpg files")
    parser.add_argument("--flatfield", type=Path, default=None, help="Flatfield .npy file")
    parser.add_argument("--contrast-offset", type=float, default=15.0, help="Threshold above bg mode")
    parser.add_argument("--min-size", type=int, default=1000, help="Min detection size (px)")
    parser.add_argument("--edge-margin", type=int, default=50, help="Ignore detections near frame edge (px)")
    parser.add_argument(
        "--dark-frac-cutoff", type=float, default=0.05, help="Skip frame if dark pixel fraction exceeds this"
    )
    parser.add_argument("-j", "--jobs", type=int, default=16, help="Worker count")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--viz", action="store_true", help="Write annotated frame images (expensive)")
    parser.add_argument("--pixel-size", type=float, default=0.36, help="µm per pixel (default: 0.36 for 20x bin3)")
    args = parser.parse_args()

    # Discover frames
    frames = sorted(args.scan_dir.glob("frame_*.jpg"), key=_natural_sort_key)
    if not frames:
        print(f"No frame_*.jpg files found in {args.scan_dir}")
        return 1
    print(f"Found {len(frames)} frames in {args.scan_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    flatfield_str = str(args.flatfield) if args.flatfield else None

    # Submit all frames to worker pool
    t0 = time.monotonic()
    results: dict[str, FrameResult] = {}
    total_det_count = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        future_to_name = {}
        for fp in frames:
            fut = pool.submit(
                _process_frame,
                str(fp),
                flatfield_str,
                args.contrast_offset,
                args.min_size,
                args.edge_margin,
                args.dark_frac_cutoff,
            )
            future_to_name[fut] = fp.stem

        for done_count, fut in enumerate(as_completed(future_to_name), 1):
            result = fut.result()
            results[result.frame_name] = result
            total_det_count += len(result.detections)
            if done_count % 100 == 0:
                print(f"  [{done_count}/{len(frames)}] {total_det_count} detections so far...")

    elapsed = time.monotonic() - t0
    print(f"Processed {len(frames)} frames in {elapsed:.1f}s ({len(frames) / elapsed:.1f} fps)")

    # Write per-frame JSONs and optional viz
    from segment_flakes import draw_detections

    for fp in frames:
        name = fp.stem
        r = results[name]
        if not r.detections:
            continue

        # Per-frame JSON (full detections with hull/contour)
        frame_json = {"frame": name, "dark_frac": round(r.dark_frac, 4), "detections": r.detections}
        with open(args.output / f"{name}.json", "w") as f:
            json.dump(frame_json, f, indent=2)

        # Annotated image
        if args.viz:
            raw = cv2.imread(fp)
            if raw is not None:
                vis = draw_detections(raw, r.detections, um_per_px=args.pixel_size)
                cv2.imwrite(args.output / f"{name}.jpg", vis)

    # Build summary
    tier_counts = {1: 0, 2: 0, 3: 0}
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
            stripped = [_strip_geometry(d) for d in r.detections]
            # Tag each detection with its frame name
            for d in stripped:
                d["frame"] = name
            all_detections[name] = stripped
            for d in r.detections:
                tier = d.get("tier", 3)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

    summary = {
        "timestamp": datetime.now().isoformat(),
        "scan_dir": str(args.scan_dir),
        "params": {
            "flatfield": flatfield_str,
            "contrast_offset": args.contrast_offset,
            "min_size_px": args.min_size,
            "edge_margin_px": args.edge_margin,
            "dark_frac_cutoff": args.dark_frac_cutoff,
            "pixel_size_um": args.pixel_size,
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

    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Final stats
    print(f"\n{'=' * 50}")
    print(f"Total frames:       {len(frames)}")
    print(f"Skipped (dark):     {skipped_count}")
    print(f"Frames w/ dets:     {frames_with_dets}")
    print(f"Total detections:   {total_det_count}")
    print(f"  Tier 1:           {tier_counts[1]}")
    print(f"  Tier 2:           {tier_counts[2]}")
    print(f"  Tier 3:           {tier_counts[3]}")

    # Top 10 by score
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
