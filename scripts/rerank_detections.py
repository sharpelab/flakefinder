"""Re-rank existing segmentation output without re-running segmentation.

Usage:
    python scripts/rerank_detections.py /tmp/seg_run1843
    python scripts/rerank_detections.py /tmp/seg_run1843 --top 20
    python scripts/rerank_detections.py /tmp/seg_run1843 --reclassify
    python scripts/rerank_detections.py /tmp/seg_run1843 --no-mosaic
    python scripts/rerank_detections.py /tmp/seg_run1843 --scan-dir path/to/scan_20x
    python scripts/rerank_detections.py /tmp/seg_run1843 --plane plane.json \\
        --revisit-mag 20x --revisit-mag 50x
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from mosaic_util import make_mosaic

from flakefinder.scan_utils import PARFOCAL_Z_UM
from flakefinder.segmentation import DetectorConfig, classify_detections, draw_scale_bar, score_detections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-rank detections in a segmentation output directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seg_dir", type=Path, help="Segmentation output directory")
    parser.add_argument("--material", default="hbn", choices=["hbn", "graphene"], help="Material preset")
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
    parser.add_argument("--name", type=str, default=None, help="Name for mosaic/crops (e.g. v4_entropy)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip spatial deduplication")
    parser.add_argument(
        "--plane",
        type=Path,
        default=None,
        help="Focus plane JSON — enables revisit JSON output for top N",
    )
    parser.add_argument(
        "--dedup-radius",
        type=float,
        default=50.0,
        help="Merge radius for dedup in µm",
    )
    parser.add_argument(
        "--revisit-mag",
        type=str,
        action="append",
        dest="revisit_mags",
        metavar="MAG",
        help="Target mag for parfocal-adjusted revisit JSON (e.g. 20x, 50x). Repeatable. Requires --plane.",
    )
    args = parser.parse_args()

    if args.revisit_mags and not args.plane:
        parser.error("--revisit-mag requires --plane")

    t0 = time.monotonic()

    summary_path = args.seg_dir / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json in {args.seg_dir}")
        return 1

    with open(summary_path) as f:
        summary = json.load(f)

    all_detections: dict[str, list[dict]] = summary.get("detections_by_frame", {})
    if not all_detections:
        print(f"No detections_by_frame in {summary_path}")
        return 1

    config = DetectorConfig.from_material(args.material)

    # Re-score from summary (no per-frame JSON I/O)
    tier_counts = {1: 0, 2: 0, 3: 0}
    total_detections = 0
    frames_with_dets = 0

    for _frame_name, dets in all_detections.items():
        if not dets:
            continue
        if args.reclassify:
            classify_detections(dets, config)
        else:
            score_detections(dets, config)
        total_detections += len(dets)
        frames_with_dets += 1
        for idx, d in enumerate(dets):
            d["det_idx"] = idx
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
    summary["rerank_command"] = sys.argv
    summary["rerank_duration_s"] = round(time.monotonic() - t0, 2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    n_frames = summary["stats"].get("total_frames", len(all_detections))
    print(f"Re-scored {n_frames} frames, {total_detections} detections")
    print(f"  Tier 1: {tier_counts[1]}  Tier 2: {tier_counts[2]}  Tier 3: {tier_counts[3]}")

    all_flat = [d for dets in all_detections.values() for d in dets]

    # Compute stage coordinates for each detection (needed by dedup and revisit)
    if all_flat:
        scan_dir = args.scan_dir or (args.seg_dir / ".." / "scan_20x").resolve()
        scan_meta_path = scan_dir / "scan_meta.json"
        if not scan_meta_path.exists():
            print(f"Warning: {scan_meta_path} not found, no stage coords")
        else:
            with open(scan_meta_path) as f:
                scan_meta = json.load(f)

            um_per_px = scan_meta["optics"]["sample_pixel_x_um"]
            frame_w_px = scan_meta["camera"]["frame_width_px"]
            frame_h_px = scan_meta["camera"]["frame_height_px"]
            frame_positions: dict[int, tuple[float, float]] = {}
            for fr in scan_meta["frames"]:
                frame_positions[fr["n"]] = (fr["x_um"], fr["y_um"])

            for d in all_flat:
                m = re.search(r"\d+", d["frame"])
                if m is None:
                    continue
                frame_n = int(m.group())
                if frame_n not in frame_positions:
                    continue
                fx, fy = frame_positions[frame_n]
                px_x, px_y = d["center"]
                d["stage_x"] = fx + (px_x - frame_w_px / 2) * um_per_px
                d["stage_y"] = fy + (px_y - frame_h_px / 2) * um_per_px

    # Spatial deduplication via greedy NMS on stage coordinates
    if not args.no_dedup and all_flat:
        from scipy.spatial import KDTree

        sorted_all = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))

        # Separate detections with/without stage coords
        with_coords = []
        without_coords = []
        for d in sorted_all:
            if d.get("stage_x") is not None:
                with_coords.append(d)
            else:
                without_coords.append(d)

        if with_coords:
            coords = np.array([(d["stage_x"], d["stage_y"]) for d in with_coords])
            tree = KDTree(coords)
            suppressed: set[int] = set()
            kept = list(without_coords)
            for idx, d in enumerate(with_coords):
                if idx in suppressed:
                    continue
                kept.append(d)
                neighbors = tree.query_ball_point(coords[idx], args.dedup_radius)
                for n_idx in neighbors:
                    if n_idx > idx:
                        suppressed.add(n_idx)
        else:
            kept = sorted_all

        n_before = len(all_flat)
        all_flat = kept
        print(f"Dedup: {n_before} → {len(all_flat)} unique (radius={args.dedup_radius:.0f} µm)")

    # Print top N
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
            if args.name:
                crops_dir = args.seg_dir / "crops" / args.name
            else:
                crops_dir = args.seg_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            crop_paths = []
            crop_labels = []
            pad = 250
            um_per_px = summary.get("params", {}).get("pixel_size_um", 0.36)

            for i, d in enumerate(top_dets):
                frame_name = d["frame"]
                det_idx = d.get("det_idx", 0)
                frame_path = scan_dir / f"{frame_name}.jpg"
                if not frame_path.exists():
                    print(f"  Warning: {frame_path} not found, skipping crop")
                    continue

                # Load per-frame JSON for contour geometry
                frame_json_path = args.seg_dir / f"{frame_name}.json"
                if not frame_json_path.exists():
                    print(f"  Warning: {frame_json_path} not found, skipping crop")
                    continue
                with open(frame_json_path) as f:
                    full_det = json.load(f)["detections"][det_idx]

                img = cv2.imread(frame_path)
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
                draw_scale_bar(crop, um_per_px)

                crop_path = crops_dir / f"rank{i + 1:02d}_{frame_name}_d{det_idx}.jpg"
                cv2.imwrite(crop_path, crop)
                crop_paths.append(str(crop_path))

                r, g, _ = d["contrast_rgb"]
                crop_labels.append(f"#{i + 1} {frame_name} R={r:+.2f} G={g:+.2f}")

            if crop_paths:
                mosaic_name = args.name or f"top{args.top}"
                mosaic_path = args.seg_dir / f"{mosaic_name}.jpg"
                import math

                n_cols = 5
                n_rows = math.ceil(len(crop_paths) / n_cols)
                mosaic = make_mosaic(
                    crop_paths,
                    rows=n_rows,
                    cols=n_cols,
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

    # Write revisit JSON for top N
    if args.plane and all_flat:
        with open(args.plane) as f:
            plane_data = json.load(f)
        plane = plane_data["plane"]
        a, b, c = plane["a"], plane["b"], plane["c"]

        # Base points at scan magnification
        base_points = []
        for i, d in enumerate(ranked[: args.top]):
            sx = d.get("stage_x")
            sy = d.get("stage_y")
            if sx is None:
                continue
            z = a * sx + b * sy + c
            label = f"rank{i + 1:02d}_{d['frame']}_d{d.get('det_idx', 0)}"
            base_points.append({"x": round(sx, 2), "y": round(sy, 2), "z": round(z, 2), "label": label})

        if args.revisit_mags:
            # Parfocal-adjusted revisit JSONs
            scan_mag = plane_data.get("source", {}).get("objective_mag")
            if scan_mag is None:
                print("Error: plane JSON missing source.objective_mag (re-export with updated analyze_focus_map)")
                return 1
            if scan_mag not in PARFOCAL_Z_UM:
                print(f"Error: scan mag {scan_mag}x not in PARFOCAL_Z_UM")
                return 1

            for mag_str in args.revisit_mags:
                target_mag = float(mag_str.lower().rstrip("x"))
                if target_mag not in PARFOCAL_Z_UM:
                    print(f"Error: revisit mag {target_mag}x not in PARFOCAL_Z_UM")
                    return 1
                delta = PARFOCAL_Z_UM[target_mag] - PARFOCAL_Z_UM[scan_mag]
                mag_label = f"{target_mag:g}"
                adjusted = [{**pt, "z": round(pt["z"] + delta, 2)} for pt in base_points]
                revisit_obj = {
                    "objective_mag": target_mag,
                    "scan_mag": scan_mag,
                    "applied_parfocal_delta_um": round(delta, 1),
                    "plane_source": str(args.plane),
                    "points": adjusted,
                }
                name_base = args.name or f"top{args.top}"
                revisit_name = f"revisit_{name_base}_{mag_label}x.json"
                revisit_path = args.seg_dir / revisit_name
                with open(revisit_path, "w") as f:
                    json.dump(revisit_obj, f, indent=2)
                print(
                    f"\nSaved revisit JSON ({len(adjusted)} pts, {mag_label}x, delta={delta:+.1f} µm): {revisit_path}"
                )
        else:
            # Un-adjusted revisit JSON
            scan_mag = plane_data.get("source", {}).get("objective_mag")
            revisit_obj = {
                "objective_mag": scan_mag,
                "scan_mag": scan_mag,
                "applied_parfocal_delta_um": 0,
                "plane_source": str(args.plane),
                "points": base_points,
            }
            revisit_name = f"revisit_{args.name}.json" if args.name else f"revisit_top{args.top}.json"
            revisit_path = args.seg_dir / revisit_name
            with open(revisit_path, "w") as f:
                json.dump(revisit_obj, f, indent=2)
            print(f"\nSaved revisit JSON ({len(base_points)} points): {revisit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
