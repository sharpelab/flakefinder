"""Upload a completed find-flakes run to flakes.sharpelab.science.

Packages the run directory into the ZIP format expected by the 2DMatGMM
website's POST /upload endpoint, then uploads it.

Usage:
    python scripts/upload_run.py scans/run_20260220_1543/
    python scripts/upload_run.py scans/run_20260220_1543/ --dry-run
    python scripts/upload_run.py scans/run_20260220_1543/ --tier 1 --top 20
    python scripts/upload_run.py scans/run_20260220_1543/ --user Zack --material hBN
"""

import argparse
import contextlib
import json
import math
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from flakefinder.segmentation import Detection, DetectorConfig, dedup_detections

BASE_URL = "https://flakes.sharpelab.science"
AUTH = ("dgglab", "***REMOVED***")

# Classification → thickness label for the website
THICKNESS_MAP = {
    "thin": "thin",
    "medium": "medium",
    "thick": "thick",
    "possible": "possible",
}


def load_summary(seg_dir: Path) -> dict:
    """Load seg/summary.json."""
    path = seg_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json in {seg_dir}")
    with open(path) as f:
        return json.load(f)


def load_frame_geometry(seg_dir: Path, frame_name: str, det_idx: int) -> dict | None:
    """Load contour/hull from per-frame JSON for a specific detection.

    Per-frame detections are in the same order as summary detections,
    so det_idx indexes directly into the list.
    """
    path = seg_dir / f"{frame_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        frame_data = json.load(f)
    dets = frame_data.get("detections", [])
    if det_idx < len(dets):
        return dets[det_idx]
    return None


def select_flakes(
    summary: dict,
    max_tier: int,
    top_n: int | None,
) -> list[Detection]:
    """Select and rank flakes from summary detections.

    Returns flat list of detection dicts with frame/det_id, sorted by
    (tier asc, score desc), filtered by tier, deduped.
    """
    all_dets: list[Detection] = []
    for frame_name, dets in summary.get("detections_by_frame", {}).items():
        for d in dets:
            d.setdefault("frame", frame_name)
            all_dets.append(d)

    # Filter by tier
    filtered = [d for d in all_dets if d.get("tier", 3) <= max_tier]

    # Sort by (tier asc, score desc)
    filtered.sort(key=lambda d: (d.get("tier", 3), -d.get("score", 0)))

    # Dedup by spatial proximity
    filtered = dedup_detections(filtered)

    if top_n is not None:
        filtered = filtered[:top_n]

    return filtered


def render_eval_img(
    frame_path: str,
    contour: list[list[int]] | None,
    bbox: list[int],
) -> np.ndarray | None:
    """Read a scan frame and draw detection overlay.

    Draws red contour outline and green rotated bounding box,
    matching 2DMatGMM eval_img convention.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return None

    # Red contour outline
    if contour and len(contour) >= 3:
        pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
        # Morphological gradient effect: draw thick then thin for outline look
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Green bounding box
    bx, by, bw, bh = bbox
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    return img


def _render_and_save(args: tuple) -> str | None:
    """Worker function for parallel eval_img rendering.

    Args is (frame_path, contour, bbox, output_path).
    Returns output_path on success, None on failure.
    """
    frame_path, contour, bbox, output_path = args
    img = render_eval_img(frame_path, contour, bbox)
    if img is None:
        return None
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output_path


def make_overview_compressed(stitch_path: Path, output_path: Path) -> None:
    """Resize stitched overview to 2000x2000 JPEG."""
    img = cv2.imread(str(stitch_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read stitch image: {stitch_path}")
    resized = cv2.resize(img, (2000, 2000), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 80])


def make_overview_marked(
    overview_compressed: np.ndarray,
    flake_x_um: float,
    flake_y_um: float,
    flake_number: int,
    stage_bounds: dict,
    overview_size_px: tuple[int, int],
) -> np.ndarray:
    """Draw a green circle and red flake number on the overview image.

    Maps stage coordinates to overview pixel coordinates using the stitch
    metadata's stage_bounds_um.
    """
    img = overview_compressed.copy()
    ow, oh = overview_size_px

    # Map stage µm → overview pixel (the overview_compressed is 2000x2000,
    # but it was resized from overview_size_px which maps to stage_bounds)
    x_min = stage_bounds["x_min"]
    x_max = stage_bounds["x_max"]
    y_min = stage_bounds["y_min"]
    y_max = stage_bounds["y_max"]

    # Map to 2000x2000 compressed image
    px_x = int((flake_x_um - x_min) / (x_max - x_min) * 2000)
    px_y = int((flake_y_um - y_min) / (y_max - y_min) * 2000)

    # Green circle
    cv2.circle(img, (px_x, px_y), 20, (0, 255, 0), 2)

    # Red flake number
    label = str(flake_number)
    cv2.putText(
        img,
        label,
        (px_x + 25, px_y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    return img


def build_scan_meta(
    run_dir: Path,
    user: str,
    material: str,
    substrate: str,
) -> dict:
    """Build scan-level meta.json for upload."""
    scan_time = time.time()
    checkpoint_path = run_dir / "checkpoint.json"

    # Try to get timestamp from stitch meta
    for meta_file in run_dir.glob("overview_*_stitch_meta.json"):
        with open(meta_file) as f:
            stitch_meta = json.load(f)
        ts = stitch_meta.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                scan_time = dt.timestamp()
            except (ValueError, TypeError):
                pass
        break

    comment = ""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            cp = json.load(f)
        notes = cp.get("notes", "")
        args = cp.get("args", {})
        preset = args.get("preset", "")
        mag = args.get("chip_scan_mag", "")
        parts = []
        if notes:
            parts.append(notes)
        if preset:
            parts.append(f"preset={preset}")
        if mag:
            parts.append(f"scan_mag={mag}")
        comment = " | ".join(parts)

    return {
        "scan_user": user,
        "scan_time": scan_time,
        "chip_thickness": substrate,
        "scan_exfoliated_material": material,
        "comment": comment or "FlakeFinder upload",
    }


def classify_thickness(det: dict, material: str) -> str:
    """Compute thickness label from detection contrast using DetectorConfig.

    The pipeline's classification field is usually None (it only marks "tape").
    This recomputes the classification from R/G contrast.
    """
    contrast = det["contrast_rgb"]
    r, g = contrast[0], contrast[1]
    config = DetectorConfig.from_material(material)
    label = config.classify(r, g)
    return THICKNESS_MAP.get(label, label)


def build_flake_meta(
    det: dict,
    camera_meta: dict,
    scan_mag: float,
    material: str,
) -> dict:
    """Build per-flake meta.json for upload."""
    # Position: µm → mm
    pos_x = det["stage_x"] / 1000.0
    pos_y = det["stage_y"] / 1000.0

    size_um2 = det["size_um2"]
    entropy = det["entropy"]
    aspect_ratio = det["aspect_ratio"]

    # Thickness from R/G contrast classification
    thickness = classify_thickness(det, material)

    # Sidelengths: derive from area + aspect_ratio (rotated bounding box)
    min_side_um = math.sqrt(size_um2 / aspect_ratio)
    max_side_um = min_side_um * aspect_ratio
    # Website expects nanometers
    max_side = max_side_um * 1000
    min_side = min_side_um * 1000

    contrast = det["contrast_rgb"]

    # Camera settings for the images dict
    wb_bgr = camera_meta["white_balance_bgr"]
    cam_entry = {
        "aperture": 6,
        "light": 6.2,
        "nosepiece": float(scan_mag),
        "gamma": int(camera_meta["gamma"] * 100),
        "gain": camera_meta["gain"],
        "exposure": camera_meta["exposure_s"],
        "white_balance": [
            int(wb_bgr[2] * 25),  # R
            int(wb_bgr[1] * 25),  # G
            int(wb_bgr[0] * 25),  # B
        ],
    }

    mag_str = f"{scan_mag:g}x"

    flake_meta = {
        "flake": {
            "position_x": round(pos_x, 4),
            "position_y": round(pos_y, 4),
            "size": round(size_um2, 1),
            "thickness": thickness,
            "entropy": round(entropy, 4),
            "max_sidelength": round(max_side, 1),
            "min_sidelength": round(min_side, 1),
            "false_positive_probability": 0.0,
            # Extra FlakeFinder fields (informational, not in DB schema)
            "mean_contrast_r": round(contrast[0], 4),
            "mean_contrast_g": round(contrast[1], 4),
            "mean_contrast_b": round(contrast[2], 4),
            "flakefinder_tier": det["tier"],
            "flakefinder_score": det["score"],
            "flakefinder_classification": det.get("classification"),
            "flakefinder_cal_dist": det["cal_dist"],
        },
        "images": {
            mag_str: cam_entry,
        },
    }

    return flake_meta


def discover_chips(run_dir: Path) -> list[int]:
    """Find chip directories with seg/summary.json."""
    chips = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.startswith("chip_"):
            idx = int(d.name.split("_")[1])
            seg_dir = d / "seg"
            if (seg_dir / "summary.json").exists():
                chips.append(idx)
    return chips


def find_revisit_image(
    chip_dir: Path,
    frame_name: str,
    det_id: int,
    mag: float,
) -> Path | None:
    """Find a revisit image for a detection at a given magnification.

    Revisit images are named like rank{NN}_{frame}_d{det_id}_{mag}x.png.
    We match on the frame+det_id part.
    """
    mag_str = f"{mag:g}x"
    search_key = f"{frame_name}_d{det_id}"

    # Check all revisit directories for this mag
    for revisit_dir in chip_dir.iterdir():
        if not revisit_dir.is_dir() or not revisit_dir.name.startswith("revisit_"):
            continue
        if mag_str not in revisit_dir.name:
            continue
        for img in revisit_dir.iterdir():
            if search_key in img.stem and img.suffix == ".png":
                return img

    return None


def discover_revisit_mags(chip_dir: Path) -> list[float]:
    """Find available revisit magnifications for a chip."""
    mags = set()
    for d in chip_dir.iterdir():
        if d.is_dir() and d.name.startswith("revisit_"):
            # Extract magnification from dir name like revisit_20x, revisit_50x,
            # revisit_hbn_thick_t1_50x, etc.
            parts = d.name.split("_")
            for part in parts:
                if part.endswith("x"):
                    try:
                        mags.add(float(part[:-1]))
                    except ValueError:
                        continue
    return sorted(mags)


def find_scan_dir(chip_dir: Path) -> Path | None:
    """Find the scan directory (scan_10x, scan_20x, etc.) inside a chip dir."""
    for d in sorted(chip_dir.iterdir()):
        if d.is_dir() and d.name.startswith("scan_") and (d / "scan_meta.json").exists():
            return d
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a find-flakes run to flakes.sharpelab.science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload_run.py scans/run_20260220_1543/
  python scripts/upload_run.py scans/run_20260220_1543/ --dry-run
  python scripts/upload_run.py scans/run_20260220_1543/ --tier 1 --top 20
  python scripts/upload_run.py scans/run_20260220_1543/ --user Sandesh --substrate 285nm
""",
    )
    parser.add_argument("run_dir", type=Path, help="Find-flakes run directory")
    parser.add_argument("--user", default=None, help="Scan user (default: from checkpoint notes or 'FlakeFinder')")
    parser.add_argument("--material", default="hBN", help="Exfoliated material (default: hBN)")
    parser.add_argument("--substrate", default="285nm", help="Chip thickness / substrate (default: 285nm)")
    parser.add_argument("--tier", type=int, default=1, help="Max tier to include (default: 1)")
    parser.add_argument("--top", type=int, default=None, help="Max flakes per chip (default: all passing tier filter)")
    parser.add_argument("--dry-run", action="store_true", help="Build ZIP but don't upload")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel workers for eval_img rendering")
    parser.add_argument("--name", default=None, help="Scan name override (default: run directory name)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist", file=sys.stderr)
        return 1

    scan_name = args.name or run_dir.name

    # Resolve user from checkpoint notes if not provided
    user = args.user
    if user is None:
        cp_path = run_dir / "checkpoint.json"
        if cp_path.exists():
            with open(cp_path) as f:
                cp = json.load(f)
            notes = cp.get("notes", "")
            # Try to extract "Operator: Name" from notes
            for line in notes.split("\n"):
                if line.lower().startswith("operator:"):
                    user = line.split(":", 1)[1].strip()
                    break
        if user is None:
            user = "FlakeFinder"

    # Discover chips
    chip_indices = discover_chips(run_dir)
    if not chip_indices:
        print("Error: no chips with seg/summary.json found", file=sys.stderr)
        return 1
    print(f"Found {len(chip_indices)} chips with segmentation data: {chip_indices}")

    # Find overview stitch
    stitch_files = list(run_dir.glob("overview_*_stitch.jpg"))
    if not stitch_files:
        print("Error: no overview stitch image found", file=sys.stderr)
        return 1
    stitch_path = stitch_files[0]

    # Load stitch metadata for coordinate mapping
    stitch_meta_path = stitch_path.with_name(stitch_path.stem + "_meta.json")
    if not stitch_meta_path.exists():
        print(f"Error: no stitch metadata at {stitch_meta_path}", file=sys.stderr)
        return 1
    with open(stitch_meta_path) as f:
        stitch_meta = json.load(f)
    stage_bounds = stitch_meta["stage_bounds_um"]
    overview_size_px = tuple(stitch_meta["image_size_px"])

    # Build upload tree in temp directory
    with tempfile.TemporaryDirectory(prefix="upload_") as tmp_root:
        upload_dir = Path(tmp_root) / scan_name
        upload_dir.mkdir()

        # Scan-level meta.json
        scan_meta = build_scan_meta(run_dir, user, args.material, args.substrate)
        with open(upload_dir / "meta.json", "w") as f:
            json.dump(scan_meta, f, indent=2)
        print(f"Scan: {scan_name} (user={user}, material={args.material}, substrate={args.substrate})")

        # overview_compressed.jpg
        overview_compressed_path = upload_dir / "overview_compressed.jpg"
        print(f"Generating overview_compressed.jpg from {stitch_path.name}...")
        make_overview_compressed(stitch_path, overview_compressed_path)

        # Load overview for marking
        overview_compressed = cv2.imread(str(overview_compressed_path))
        assert overview_compressed is not None, f"Failed to read {overview_compressed_path}"

        # Process each chip
        total_flakes = 0
        eval_img_jobs: list[tuple] = []
        # Deferred work: (chip_upload_dir, flake_idx, det, mag, src_path)
        revisit_copies: list[tuple[Path, str]] = []
        # Deferred work: (overview_img, det, flake_number, output_path)
        overview_marked_jobs: list[tuple[float, float, int, str]] = []

        for chip_idx in chip_indices:
            chip_dir = run_dir / f"chip_{chip_idx}"
            seg_dir = chip_dir / "seg"

            # Load summary
            summary = load_summary(seg_dir)

            # Find scan directory for frame paths and camera meta
            scan_dir = find_scan_dir(chip_dir)
            if scan_dir is None:
                print(f"  Chip {chip_idx}: no scan directory found, skipping")
                continue

            scan_meta_path = scan_dir / "scan_meta.json"
            with open(scan_meta_path) as f:
                chip_scan_meta = json.load(f)
            camera_meta = chip_scan_meta["camera"]
            optics_meta = chip_scan_meta["optics"]
            scan_mag = optics_meta["objective_mag"]

            # Material preset used for segmentation (for thickness classification)
            seg_material = summary.get("params", {}).get("material", "hbn")

            # Select flakes
            flakes = select_flakes(summary, args.tier, args.top)
            if not flakes:
                print(f"  Chip {chip_idx}: no flakes passing tier<={args.tier} filter")
                continue

            n_t1 = sum(1 for d in flakes if d.get("tier") == 1)
            n_t2 = sum(1 for d in flakes if d.get("tier") == 2)
            print(f"  Chip {chip_idx}: {len(flakes)} flakes (T1:{n_t1}, T2:{n_t2})")

            # Discover available revisit magnifications
            revisit_mags = discover_revisit_mags(chip_dir)

            # Create chip directory in upload tree
            chip_upload_name = f"Chip_{chip_idx + 1}"
            chip_upload_dir = upload_dir / chip_upload_name

            for flake_idx, det in enumerate(flakes):
                flake_number = flake_idx + 1
                flake_upload_name = f"Flake_{flake_number}"
                flake_dir = chip_upload_dir / flake_upload_name
                flake_dir.mkdir(parents=True, exist_ok=True)

                # Flake meta.json
                flake_meta = build_flake_meta(det, camera_meta, scan_mag, seg_material)

                # Add revisit mag entries to images dict
                for mag in revisit_mags:
                    revisit_img = find_revisit_image(chip_dir, det["frame"], det["det_id"], mag)
                    if revisit_img is not None:
                        mag_str = f"{mag:g}x"
                        # Add image entry (reuse camera meta — not exact but the
                        # website needs an entry per mag to create Image rows)
                        flake_meta["images"][mag_str] = {
                            "aperture": 6,
                            "light": 6.2,
                            "nosepiece": float(mag),
                            "gamma": int(camera_meta["gamma"] * 100),
                            "gain": camera_meta["gain"],
                            "exposure": camera_meta["exposure_s"],
                            "white_balance": flake_meta["images"][f"{scan_mag:g}x"]["white_balance"],
                        }
                        # Queue file copy
                        revisit_copies.append((revisit_img, str(flake_dir / f"{mag_str}.png")))

                with open(flake_dir / "meta.json", "w") as f:
                    json.dump(flake_meta, f, indent=2)

                # Queue eval_img rendering
                frame_path = str(scan_dir / f"{det['frame']}.jpg")
                # Load contour from per-frame JSON
                geom = load_frame_geometry(seg_dir, det["frame"], det["det_id"])
                contour = geom.get("contour") if geom else None
                eval_img_path = str(flake_dir / "eval_img.jpg")
                eval_img_jobs.append((frame_path, contour, det["bbox"], eval_img_path))

                # Queue overview_marked
                stage_x = det["stage_x"]
                stage_y = det["stage_y"]
                overview_marked_path = str(flake_dir / "overview_marked.jpg")
                overview_marked_jobs.append((stage_x, stage_y, flake_number, overview_marked_path))

                total_flakes += 1

        if total_flakes == 0:
            print("Error: no flakes to upload", file=sys.stderr)
            return 1

        print(f"\nTotal: {total_flakes} flakes across {len(chip_indices)} chips")

        # Render eval images in parallel
        print(f"Rendering {len(eval_img_jobs)} eval images ({args.jobs} workers)...")
        t0 = time.monotonic()
        rendered = 0
        failed = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_render_and_save, job): job for job in eval_img_jobs}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    rendered += 1
                else:
                    failed += 1
        elapsed = time.monotonic() - t0
        print(f"  Rendered {rendered} eval images in {elapsed:.1f}s ({rendered / max(elapsed, 0.001):.0f}/s)")
        if failed:
            print(f"  WARNING: {failed} eval images failed to render")

        # Generate overview_marked images
        print(f"Generating {len(overview_marked_jobs)} overview_marked images...")
        for stage_x, stage_y, flake_number, output_path in overview_marked_jobs:
            marked = make_overview_marked(
                overview_compressed,
                stage_x,
                stage_y,
                flake_number,
                stage_bounds,
                overview_size_px,
            )
            cv2.imwrite(output_path, marked, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # Copy revisit images
        if revisit_copies:
            print(f"Copying {len(revisit_copies)} revisit images...")
            for src, dst in revisit_copies:
                shutil.copy2(src, dst)

        # Create ZIP
        print("Creating ZIP archive...")
        zip_path = Path(tmp_root) / scan_name
        zip_file = shutil.make_archive(str(zip_path), "zip", str(tmp_root), scan_name)
        zip_size_mb = Path(zip_file).stat().st_size / (1024 * 1024)
        print(f"  ZIP: {zip_file} ({zip_size_mb:.1f} MB)")

        if args.dry_run:
            # Copy ZIP to current directory for inspection
            final_zip = Path(f"{scan_name}_upload.zip")
            shutil.copy2(zip_file, final_zip)
            print(f"\n[dry-run] ZIP saved to {final_zip}")
            print("  Upload skipped. Inspect the ZIP to verify contents.")
            return 0

        # Upload
        print(f"\nUploading to {BASE_URL}/api/upload ...")
        t0 = time.monotonic()
        with open(zip_file, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/api/upload",
                files={"zip": (f"{scan_name}.zip", f, "application/zip")},
                auth=AUTH,
                timeout=600,
            )
        elapsed = time.monotonic() - t0

        if resp.status_code == 200:
            print(f"  Upload successful ({elapsed:.1f}s, {zip_size_mb / max(elapsed, 0.001):.1f} MB/s)")
            print(f"  View at: {BASE_URL}")
            return 0
        else:
            print(f"  Upload FAILED: {resp.status_code} {resp.reason}", file=sys.stderr)
            with contextlib.suppress(Exception):
                print(f"  Response: {resp.text[:500]}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
