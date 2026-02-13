"""Re-rank existing segmentation output without re-running segmentation.

Usage:
    python scripts/rerank_detections.py /tmp/seg_run1843
    python scripts/rerank_detections.py /tmp/seg_run1843 --top 20
    python scripts/rerank_detections.py /tmp/seg_run1843 --reclassify
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from segment_flakes import classify_detections, score_detections


def _natural_sort_key(path: Path) -> int:
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else 0


def _strip_geometry(det: dict) -> dict:
    return {k: v for k, v in det.items() if k not in ("hull", "contour")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-rank detections in a segmentation output directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seg_dir", type=Path, help="Segmentation output directory")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to print")
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Also re-run tape classification (default: only rescore tier+score)",
    )
    args = parser.parse_args()

    summary_path = args.seg_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json in {args.seg_dir}")
        return 1

    with open(summary_path) as f:
        summary = json.load(f)

    # Discover per-frame JSONs
    frame_jsons = sorted(args.seg_dir.glob("frame_*.json"), key=_natural_sort_key)
    if not frame_jsons:
        print(f"No frame_*.json files in {args.seg_dir}")
        return 1

    rescore_fn = classify_detections if args.reclassify else score_detections

    # Re-score each frame and overwrite
    tier_counts = {1: 0, 2: 0, 3: 0}
    total_detections = 0
    frames_with_dets = 0
    all_detections: dict[str, list[dict]] = {}

    for fj in frame_jsons:
        with open(fj) as f:
            frame_data = json.load(f)

        dets = frame_data["detections"]
        if not dets:
            continue

        rescore_fn(dets)

        # Overwrite per-frame JSON
        with open(fj, "w") as f:
            json.dump(frame_data, f, indent=2)

        frame_name = frame_data["frame"]
        total_detections += len(dets)
        frames_with_dets += 1

        stripped = [_strip_geometry(d) for d in dets]
        for idx, d in enumerate(stripped):
            d["frame"] = frame_name
            d["det_idx"] = idx
        all_detections[frame_name] = stripped

        for d in dets:
            tier = d.get("tier", 3)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Update summary
    summary["stats"]["total_detections"] = total_detections
    summary["stats"]["frames_with_detections"] = frames_with_dets
    summary["stats"]["tier_1"] = tier_counts[1]
    summary["stats"]["tier_2"] = tier_counts[2]
    summary["stats"]["tier_3"] = tier_counts[3]
    summary["detections_by_frame"] = all_detections
    summary["reranked_at"] = datetime.now().isoformat()

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Re-scored {len(frame_jsons)} frames, {total_detections} detections")
    print(f"  Tier 1: {tier_counts[1]}  Tier 2: {tier_counts[2]}  Tier 3: {tier_counts[3]}")

    # Print top N
    all_flat = [d for dets in all_detections.values() for d in dets]
    if all_flat:
        ranked = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        print(f"\n--- Top {args.top} ---")
        hdr = (
            f"{'#':>3}  {'frame':<16} {'det':>3} {'size':>7}"
            f" {'R':>7} {'G':>7} {'score':>7} {'cal_d':>7} {'grad':>7} {'entr':>6}"
        )
        print(hdr)
        for i, d in enumerate(ranked[: args.top]):
            r, g, _ = d["contrast_rgb"]
            print(
                f"{i + 1:>3}  {d['frame']:<16} {d.get('det_idx', '-'):>3} {d['size_px']:>7} "
                f"{r:>+7.3f} {g:>+7.3f} {d.get('score', 0):>7.3f} "
                f"{d.get('cal_dist', 0):>7.3f} {d.get('grad_energy', 0):>7.1f} {d.get('g_entropy', 0):>6.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
