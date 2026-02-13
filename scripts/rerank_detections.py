"""Re-rank existing segmentation output without re-running segmentation.

Usage:
    python scripts/rerank_detections.py /tmp/seg_run1843
    python scripts/rerank_detections.py /tmp/seg_run1843 --top 20
    python scripts/rerank_detections.py /tmp/seg_run1843 --reclassify
    python scripts/rerank_detections.py /tmp/seg_run1843 --no-mosaic
    python scripts/rerank_detections.py /tmp/seg_run1843 --scan-dir path/to/scan_20x
"""

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from mosaic_util import make_mosaic
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
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=None,
        help="Directory with raw frame_NNNN.jpg files (default: <seg_dir>/../scan_20x)",
    )
    parser.add_argument("--no-mosaic", action="store_true", help="Skip crop and mosaic generation")
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
    full_detections: dict[str, list[dict]] = {}  # with hull/contour for mosaic

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
        full_detections[frame_name] = dets

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

    # Generate cropped mosaic
    if not args.no_mosaic and all_flat:
        scan_dir = args.scan_dir or (args.seg_dir / ".." / "scan_20x").resolve()
        if not scan_dir.is_dir():
            print(f"Warning: scan dir not found: {scan_dir} (use --scan-dir or --no-mosaic)")
        else:
            top_dets = ranked[: args.top]
            crops_dir = args.seg_dir / "crops"
            crops_dir.mkdir(exist_ok=True)

            crop_paths = []
            crop_labels = []
            pad = 150

            for i, d in enumerate(top_dets):
                frame_name = d["frame"]
                det_idx = d.get("det_idx", 0)
                frame_path = scan_dir / f"{frame_name}.jpg"
                if not frame_path.exists():
                    print(f"  Warning: {frame_path} not found, skipping crop")
                    continue

                # Get full detection with contour
                full_det = full_detections[frame_name][det_idx]
                img = cv2.imread(str(frame_path))
                if img is None:
                    print(f"  Warning: failed to read {frame_path}, skipping crop")
                    continue
                h, w = img.shape[:2]

                # Draw contour
                contour = full_det.get("contour")
                if contour and len(contour) >= 3:
                    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                # Crop with padding
                bx, by, bw, bh = full_det["bbox"]
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(w, bx + bw + pad)
                y1 = min(h, by + bh + pad)
                crop = img[y0:y1, x0:x1]

                crop_path = crops_dir / f"rank{i + 1:02d}_{frame_name}_d{det_idx}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crop_paths.append(str(crop_path))

                r, g, _ = d["contrast_rgb"]
                crop_labels.append(f"#{i + 1} {frame_name} R={r:+.2f} G={g:+.2f}")

            if crop_paths:
                mosaic_path = args.seg_dir / f"top{args.top}.jpg"
                mosaic = make_mosaic(
                    crop_paths,
                    rows=2,
                    cols=None,
                    max_dim=5000,
                    margin=4,
                    bg_color=(30, 30, 30),
                    labels=crop_labels,
                    label_size=16,
                    label_color=(255, 255, 255),
                    label_bg=(0, 0, 0, 180),
                )
                mosaic.save(str(mosaic_path), quality=95)
                print(f"\nSaved mosaic: {mosaic_path}")
                print(f"Saved {len(crop_paths)} crops: {crops_dir}/")
                subprocess.Popen(["present", str(mosaic_path)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
