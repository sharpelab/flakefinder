"""Re-rank existing segmentation output without re-running segmentation.

Usage (single chip):
    python scripts/rerank_detections.py /tmp/seg_run1843
    python scripts/rerank_detections.py /tmp/seg_run1843 --top 20
    python scripts/rerank_detections.py /tmp/seg_run1843 --reclassify
    python scripts/rerank_detections.py /tmp/seg_run1843 --no-mosaic
    python scripts/rerank_detections.py /tmp/seg_run1843 --scan-dir path/to/scan_20x
    python scripts/rerank_detections.py /tmp/seg_run1843 --plane plane.json \\
        --revisit-mag 20x --revisit-mag 50x

Usage (scan-wide — all chips in a run):
    python scripts/rerank_detections.py scans/run_20260220_1543/
    python scripts/rerank_detections.py scans/run_20260220_1543/ --top 20
    python scripts/rerank_detections.py scans/run_20260220_1543/ --seg-name seg_10x
    python scripts/rerank_detections.py scans/run_20260220_1543/ --tier 1 \\
        --revisit-mag 20x --revisit-mag 50x
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mosaic_util import make_mosaic

from flakefinder.scan_utils import PARFOCAL_Z_UM
from flakefinder.segmentation import (
    Detection,
    DetectorConfig,
    classify_detections,
    dedup_detections,
    draw_scale_bar,
    score_detections,
)


def plot_rg_scatter(
    ax: plt.Axes,
    detections: list[Detection],
    config: DetectorConfig,
    top_indices: set[int] | None = None,
    title: str | None = None,
) -> None:
    """Plot R-G contrast scatter on the given axes.

    Points colored by calibration distance (RdYlGn_r: green=near curve, red=far).
    Top-N detections drawn as stars.  Reusable for single-chip or multi-chip grids.

    Args:
        ax: Matplotlib axes to draw on.
        detections: Flat list of detection dicts with ``contrast_rgb`` and ``cal_dist``.
        config: DetectorConfig (provides ``cal_poly`` for the calibration curve).
        top_indices: Indices into *detections* to highlight as stars.
        title: Optional axes title.
    """
    if not detections:
        return
    top_indices = top_indices or set()

    rs = np.array([d["contrast_rgb"][0] for d in detections])
    gs = np.array([d["contrast_rgb"][1] for d in detections])
    cal_dists = np.array([d.get("cal_dist", 0.0) for d in detections])

    # Regular points (not in top-N)
    mask_reg = np.array([i not in top_indices for i in range(len(detections))])
    if mask_reg.any():
        ax.scatter(
            gs[mask_reg],
            rs[mask_reg],
            c=cal_dists[mask_reg],
            s=8,
            alpha=0.5,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=2,
            zorder=2,
        )

    # Top-N as stars
    mask_top = ~mask_reg
    if mask_top.any():
        ax.scatter(
            gs[mask_top],
            rs[mask_top],
            c=cal_dists[mask_top],
            s=120,
            alpha=0.9,
            marker="*",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=2,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    # Calibration curve
    g_range = np.linspace(-0.5, 5.0, 200)
    r_curve = np.polyval(config.cal_poly, g_range)
    ax.plot(g_range, r_curve, "k-", linewidth=2, label="Cal curve")

    ax.set_xlabel("G contrast")
    ax.set_ylabel("R contrast")
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-2, 5)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)


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
    if not chip_dir.is_dir():
        return None
    for d in sorted(chip_dir.iterdir()):
        if d.is_dir() and d.name.startswith("scan_"):
            return d
    return None


def _find_chip_planes(run_dir: Path, chip_indices: list[int]) -> dict[int, dict]:
    """Auto-discover per-chip focus plane JSONs."""
    planes: dict[int, dict] = {}
    for idx in chip_indices:
        plane_path = run_dir / f"chip_{idx}" / f"focus_map_chip{idx}_plane.json"
        if plane_path.exists():
            with open(plane_path) as f:
                planes[idx] = json.load(f)
    return planes


def _run_wide_main(args: argparse.Namespace, seg_dirs: dict[int, Path]) -> int:
    """Scan-wide reranking: aggregate detections from all chips in a run."""
    import math

    t0 = time.monotonic()
    run_dir = args.seg_dir
    config = DetectorConfig.from_material(args.material)

    # Load detections from all chips
    all_flat: list[Detection] = []
    scan_dirs: dict[int, Path | None] = {}
    um_per_px = 0.36
    chip_tier_counts: dict[int, dict[int, int]] = {}

    for chip_idx, seg_dir in sorted(seg_dirs.items()):
        chip_dir = seg_dir.parent
        scan_dirs[chip_idx] = _find_scan_dir(chip_dir)

        with open(seg_dir / "summary.json") as f:
            summary = json.load(f)
        um_per_px = summary.get("params", {}).get("pixel_size_um", um_per_px)
        detections_by_frame = summary.get("detections_by_frame", {})

        tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
        for frame_name, dets in detections_by_frame.items():
            if not dets:
                continue
            if args.reclassify:
                classify_detections(dets, config)
            else:
                score_detections(dets, config)
            for idx, d in enumerate(dets):
                d["det_idx"] = idx
                d.setdefault("frame", frame_name)
                d["chip_idx"] = chip_idx
            for d in dets:
                tier = d.get("tier", 3)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            all_flat.extend(dets)

        chip_tier_counts[chip_idx] = tier_counts

    # Per-chip stats table
    total_t1 = total_t2 = total_t3 = 0
    print(f"Loaded {len(all_flat)} detections from {len(seg_dirs)} chips\n")
    print(f"  {'Chip':<10} {'T1':>5} {'T2':>5} {'T3':>5} {'Total':>7}")
    print(f"  {'-' * 10} {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 7}")
    for chip_idx in sorted(chip_tier_counts):
        tc = chip_tier_counts[chip_idx]
        t1, t2, t3 = tc[1], tc[2], tc[3]
        total_t1 += t1
        total_t2 += t2
        total_t3 += t3
        total = t1 + t2 + t3
        print(f"  chip_{chip_idx:<5} {t1:>5} {t2:>5} {t3:>5} {total:>7}")
    print(f"  {'-' * 10} {'-' * 5} {'-' * 5} {'-' * 5} {'-' * 7}")
    total_all = total_t1 + total_t2 + total_t3
    print(f"  {'TOTAL':<10} {total_t1:>5} {total_t2:>5} {total_t3:>5} {total_all:>7}")

    # Warn about missing stage coords
    if all_flat and all_flat[0].get("stage_x") is None:
        print("\nWarning: detections missing stage coords -- dedup may not work")

    # Spatial deduplication across all chips
    if not args.no_dedup and all_flat:
        sorted_all = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        n_before = len(all_flat)
        all_flat = dedup_detections(sorted_all, radius_um=args.dedup_radius)
        print(f"\nDedup: {n_before} -> {len(all_flat)} unique (radius={args.dedup_radius:.0f} um)")

    # Tier filtering
    if args.tier is not None and all_flat:
        n_before_tier = len(all_flat)
        all_flat = [d for d in all_flat if d.get("tier") == args.tier]
        print(f"\nTier {args.tier} filter: {n_before_tier} -> {len(all_flat)}")

    # Rank and print top N
    ranked: list[Detection] = []
    if all_flat:
        ranked = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        n_show = min(args.top, len(ranked))
        tier_label = f" T{args.tier}" if args.tier is not None else ""
        print(f"\n--- Top {n_show}{tier_label} ---")
        hdr = (
            f"{'#':>3}  {'chip':>6} {'frame':<16} {'det':>3} {'size':>7}"
            f" {'R':>7} {'G':>7} {'score':>7} {'cal_d':>7} {'grad':>7} {'entr':>6}"
        )
        print(hdr)
        for i, d in enumerate(ranked[: args.top]):
            r, g, _ = d["contrast_rgb"]
            chip_str = f"c{d.get('chip_idx', '?')}"
            print(
                f"{i + 1:>3}  {chip_str:>6} {d['frame']:<16} {d.get('det_idx', '-'):>3} {d['size_px']:>7} "
                f"{r:>+7.3f} {g:>+7.3f} {d.get('score', 0):>7.3f} "
                f"{d.get('cal_dist', 0):>7.3f} {d.get('grad_energy', 0):>7.1f}"
                f" {d.get('entropy', d.get('g_entropy', 0)):>6.2f}"
            )

    # Output directory
    output_dir = run_dir / "rerank"
    if args.name:
        output_dir = run_dir / "rerank" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate cropped mosaic
    if not args.no_mosaic and ranked:
        top_dets = ranked[: args.top]
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        crop_paths: list[str] = []
        crop_labels: list[str] = []
        pad = 250

        for i, d in enumerate(top_dets):
            chip_idx = d.get("chip_idx", 0)
            frame_name = d["frame"]
            det_idx = d.get("det_idx", 0)

            scan_dir = scan_dirs.get(chip_idx)
            if scan_dir is None or not scan_dir.is_dir():
                print(f"  Warning: no scan dir for chip_{chip_idx}, skipping crop")
                continue

            frame_path = scan_dir / f"{frame_name}.jpg"
            if not frame_path.exists():
                print(f"  Warning: {frame_path} not found, skipping crop")
                continue

            frame_json_path = seg_dirs[chip_idx] / f"{frame_name}.json"
            if not frame_json_path.exists():
                print(f"  Warning: {frame_json_path} not found, skipping crop")
                continue
            with open(frame_json_path) as f:
                full_det = json.load(f)["detections"][det_idx]

            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"  Warning: failed to read {frame_path}, skipping crop")
                continue
            h, w = img.shape[:2]

            bx, by, bw, bh = full_det["bbox"]

            if args.overlay == "contour":
                contour = full_det.get("contour")
                if contour and len(contour) >= 3:
                    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            elif args.overlay == "bbox":
                bp = 4
                cv2.rectangle(img, (bx - bp, by - bp), (bx + bw + bp, by + bh + bp), (0, 255, 0), 1)

            x0 = max(0, bx - pad)
            y0 = max(0, by - pad)
            x1 = min(w, bx + bw + pad)
            y1 = min(h, by + bh + pad)
            crop = img[y0:y1, x0:x1]
            draw_scale_bar(crop, um_per_px)

            crop_path = crops_dir / f"rank{i + 1:02d}_c{chip_idx}_{frame_name}_d{det_idx}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))

            r, g, _ = d["contrast_rgb"]
            crop_labels.append(f"#{i + 1} chip_{chip_idx} {frame_name} R={r:+.2f} G={g:+.2f}")

        if crop_paths:
            mosaic_name = args._name_base
            mosaic_path = output_dir / f"{mosaic_name}.jpg"
            n_cols = 5
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
            mosaic.save(str(mosaic_path), quality=95)
            print(f"\nSaved mosaic: {mosaic_path}")
            print(f"Saved {len(crop_paths)} crops: {crops_dir}/")
            subprocess.Popen(
                ["present", str(mosaic_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    # R-G scatter plot
    if not args.no_scatter and all_flat:
        top_set = set(id(d) for d in ranked[: args.top]) if ranked else set()
        top_idx = {i for i, d in enumerate(all_flat) if id(d) in top_set}

        fig, ax = plt.subplots(figsize=(10, 7))
        n_chips = len(seg_dirs)
        plot_rg_scatter(
            ax,
            all_flat,
            config,
            top_indices=top_idx,
            title=(
                f"{run_dir.name}: {len(all_flat)} detections, {n_chips} chips"
                f" (top {min(args.top, len(all_flat))} starred)"
            ),
        )
        scatter_name = f"{args.name}_rg_scatter.png" if args.name else "rg_scatter.png"
        scatter_path = output_dir / scatter_name
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        print(f"Saved scatter: {scatter_path}")
        subprocess.Popen(
            ["present", str(scatter_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Revisit JSON for top N — use --plane override or auto-discovered per-chip planes
    chip_planes: dict[int, dict] | None = None
    if not args.plane:
        chip_planes = _find_chip_planes(run_dir, list(seg_dirs.keys()))
        if chip_planes:
            print(f"\nAuto-discovered focus planes for {len(chip_planes)} chips: {sorted(chip_planes.keys())}")

    has_planes = args.plane or chip_planes
    if has_planes and ranked:
        # Single-plane override
        if args.plane:
            with open(args.plane) as f:
                single_plane = json.load(f)
        else:
            single_plane = None

        def _get_plane_for_chip(chip_idx: int) -> dict | None:
            if single_plane:
                return single_plane
            if chip_planes:
                return chip_planes.get(chip_idx)
            return None

        base_points = []
        scan_mag = None
        for i, d in enumerate(ranked[: args.top]):
            sx = d.get("stage_x")
            sy = d.get("stage_y")
            if sx is None or sy is None:
                continue
            chip_idx = d.get("chip_idx", 0)
            pd = _get_plane_for_chip(chip_idx)
            if pd is None:
                print(f"  Warning: no plane for chip_{chip_idx}, skipping revisit point")
                continue
            plane = pd["plane"]
            z = plane["a"] * sx + plane["b"] * sy + plane["c"]
            if scan_mag is None:
                scan_mag = pd.get("source", {}).get("objective_mag")
            label = f"rank{i + 1:02d}_c{chip_idx}_{d['frame']}_d{d.get('det_idx', 0)}"
            base_points.append({"x": round(sx, 2), "y": round(sy, 2), "z": round(z, 2), "label": label})

        plane_source = str(args.plane) if args.plane else "per-chip auto-discovery"

        if args.revisit_mags:
            if scan_mag is None:
                print("Error: plane JSON(s) missing source.objective_mag")
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
                    "plane_source": plane_source,
                    "points": adjusted,
                }
                name_base = args._name_base
                revisit_name = f"revisit_{name_base}_{mag_label}x.json"
                revisit_path = output_dir / revisit_name
                with open(revisit_path, "w") as f:
                    json.dump(revisit_obj, f, indent=2)
                print(
                    f"\nSaved revisit JSON ({len(adjusted)} pts, {mag_label}x, delta={delta:+.1f} um): {revisit_path}"
                )
        else:
            revisit_obj = {
                "objective_mag": scan_mag,
                "scan_mag": scan_mag,
                "applied_parfocal_delta_um": 0,
                "plane_source": plane_source,
                "points": base_points,
            }
            revisit_name = f"revisit_{args._name_base}.json"
            revisit_path = output_dir / revisit_name
            with open(revisit_path, "w") as f:
                json.dump(revisit_obj, f, indent=2)
            print(f"\nSaved revisit JSON ({len(base_points)} points): {revisit_path}")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-rank detections in a segmentation output directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("seg_dir", type=Path, help="Segmentation output dir (single chip) or run dir (scan-wide)")
    parser.add_argument(
        "--material",
        default="hbn",
        choices=DetectorConfig.material_names(),
        help="Material preset",
    )
    parser.add_argument(
        "--top", type=int, default=None, help="Number of top results (default: 10, or all when --tier is set)"
    )
    parser.add_argument("--tier", type=int, default=None, help="Filter to tier N detections (e.g. --tier 1 for all T1)")
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
    parser.add_argument(
        "--overlay",
        choices=["contour", "bbox", "none"],
        default="contour",
        help="What to draw on crops: contour outline, bounding box, or nothing",
    )
    parser.add_argument("--no-mosaic", action="store_true", help="Skip crop and mosaic generation")
    parser.add_argument("--no-scatter", action="store_true", help="Skip R-G scatter plot generation")
    parser.add_argument("--name", type=str, default=None, help="Name for mosaic/crops (e.g. v4_entropy)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip spatial deduplication")
    parser.add_argument(
        "--seg-name",
        default="seg",
        help="Seg subdir name within each chip dir (scan-wide mode only)",
    )
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

    # Resolve --top default: 10 normally, unlimited when --tier is set
    explicit_top = args.top is not None
    if args.top is None:
        args.top = 10 if args.tier is None else 999999

    # Auto-generate a name base for output files when not explicitly named
    if args.name:
        args._name_base = args.name
    elif args.tier is not None and not explicit_top:
        args._name_base = f"t{args.tier}"
    elif args.tier is not None:
        args._name_base = f"t{args.tier}_top{args.top}"
    else:
        args._name_base = f"top{args.top}"

    if args.revisit_mags and not args.plane:
        # In scan-wide mode, per-chip planes are auto-discovered — defer the check
        summary_path = args.seg_dir / "summary.json"
        if summary_path.exists():
            parser.error("--revisit-mag requires --plane (single-chip mode)")

    t0 = time.monotonic()

    summary_path = args.seg_dir / "summary.json"
    if not summary_path.exists():
        # Try scan-wide mode: look for chip_*/seg/summary.json
        seg_dirs = _find_seg_dirs(args.seg_dir, args.seg_name)
        if seg_dirs:
            return _run_wide_main(args, seg_dirs)
        print(f"No summary.json in {args.seg_dir} and no chip seg directories found")
        return 1

    with open(summary_path) as f:
        summary = json.load(f)

    all_detections: dict[str, list[Detection]] = summary.get("detections_by_frame", {})
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

    # Stage coordinates are now included in summary.json from segmentation.
    # Warn if missing (old data generated before this change).
    if all_flat and all_flat[0].get("stage_x") is None:
        scan_dir = args.scan_dir or (args.seg_dir / ".." / "scan_20x").resolve()
        scan_meta_path = scan_dir / "scan_meta.json"
        if not scan_meta_path.exists():
            print(f"Warning: {scan_meta_path} not found, no stage coords")
        else:
            from flakefinder.data_utils import add_stage_coords

            with open(scan_meta_path) as f:
                scan_meta = json.load(f)
            add_stage_coords(all_flat, scan_meta)
            print("Note: added stage coords (old summary.json without coords)")

    # Spatial deduplication via greedy NMS on stage coordinates
    if not args.no_dedup and all_flat:
        sorted_all = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        n_before = len(all_flat)
        all_flat = dedup_detections(sorted_all, radius_um=args.dedup_radius)
        print(f"Dedup: {n_before} → {len(all_flat)} unique (radius={args.dedup_radius:.0f} µm)")

    # Tier filtering
    if args.tier is not None and all_flat:
        n_before_tier = len(all_flat)
        all_flat = [d for d in all_flat if d.get("tier") == args.tier]
        print(f"\nTier {args.tier} filter: {n_before_tier} -> {len(all_flat)}")

    # Print top N
    if all_flat:
        ranked = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        n_show = min(args.top, len(ranked))
        tier_label = f" T{args.tier}" if args.tier is not None else ""
        print(f"\n--- Top {n_show}{tier_label} ---")
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
                f"{d.get('cal_dist', 0):>7.3f} {d.get('grad_energy', 0):>7.1f}"
                f" {d.get('entropy', d.get('g_entropy', 0)):>6.2f}"
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

                bx, by, bw, bh = full_det["bbox"]

                if args.overlay == "contour":
                    contour = full_det.get("contour")
                    if contour and len(contour) >= 3:
                        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                elif args.overlay == "bbox":
                    bp = 4
                    cv2.rectangle(img, (bx - bp, by - bp), (bx + bw + bp, by + bh + bp), (0, 255, 0), 1)

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
                mosaic_name = args._name_base
                mosaic_path = args.seg_dir / f"{mosaic_name}.jpg"
                import math

                n_cols = 5
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
                mosaic.save(str(mosaic_path), quality=95)
                print(f"\nSaved mosaic: {mosaic_path}")
                print(f"Saved {len(crop_paths)} crops: {crops_dir}/")
                subprocess.Popen(
                    ["present", str(mosaic_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

    # Generate R-G scatter plot
    if not args.no_scatter and all_flat:
        top_set = set(id(d) for d in ranked[: args.top]) if ranked else set()
        # Map back to indices in all_flat for plot_rg_scatter
        top_idx = {i for i, d in enumerate(all_flat) if id(d) in top_set}

        fig, ax = plt.subplots(figsize=(10, 7))
        chip_label = args.seg_dir.parent.name if args.seg_dir.parent.name.startswith("chip") else args.seg_dir.name
        plot_rg_scatter(
            ax,
            all_flat,
            config,
            top_indices=top_idx,
            title=f"{chip_label}: {len(all_flat)} detections (top {min(args.top, len(all_flat))} starred)",
        )
        scatter_name = f"{args.name}_rg_scatter.png" if args.name else "rg_scatter.png"
        scatter_path = args.seg_dir / scatter_name
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        print(f"Saved scatter: {scatter_path}")
        subprocess.Popen(
            ["present", str(scatter_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

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
                name_base = args._name_base
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
            revisit_name = f"revisit_{args._name_base}.json"
            revisit_path = args.seg_dir / revisit_name
            with open(revisit_path, "w") as f:
                json.dump(revisit_obj, f, indent=2)
            print(f"\nSaved revisit JSON ({len(base_points)} points): {revisit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
