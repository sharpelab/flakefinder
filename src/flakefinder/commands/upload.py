"""Upload a completed find-flakes run to flakes.sharpelab.science.

Uses the resource-oriented REST API: upload files as tokens, then create
scan/chips/flakes via the REST endpoints.

Usage:
    sls upload scans/run_20260220_1543/
    sls upload scans/run_20260220_1543/ --dry-run
    sls upload scans/run_20260220_1543/ --tier 1 --top 20
    sls upload scans/run_20260220_1543/ --user Sandesh --substrate 285nm
"""

import argparse
import contextlib
import json
import math
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import requests
from PIL import Image as PILImage
from PIL import ImageDraw

from flakefinder.flakes_api import BASE_URL, get_auth
from flakefinder.scan_utils import get_git_version
from flakefinder.segmentation import Detection, dedup_detections


class UploadResult(NamedTuple):
    total_flakes: int
    upload_mb: float
    uploaded: bool


class OverviewAssets(NamedTuple):
    """Loaded overview stitch + metadata used for rendering detail crops."""

    stitch: PILImage.Image
    um_per_px: float
    stage_x_min: float
    stage_y_min: float
    fov_w_um: float
    fov_h_um: float


def load_overview_assets(run_dir: Path) -> OverviewAssets:
    """Load stitched overview image + scan metadata for detail rendering.

    Reads the stitch image once into memory; subsequent crops are cheap.
    """
    stitch_files = list(run_dir.glob("overview_*_stitch.jpg"))
    if not stitch_files:
        raise FileNotFoundError("No overview stitch image found")
    stitch_path = stitch_files[0]

    stitch_meta_path = stitch_path.with_name(stitch_path.stem + "_meta.json")
    with open(stitch_meta_path) as f:
        stitch_meta = json.load(f)

    # Find matching overview scan dir for FOV dimensions
    obj_mag = stitch_meta["objective_mag"]
    mag_label = f"{obj_mag:g}x"
    overview_dir = run_dir / f"overview_{mag_label}"
    with open(overview_dir / "scan_meta.json") as f:
        overview_meta = json.load(f)

    return OverviewAssets(
        stitch=PILImage.open(stitch_path).convert("RGB"),
        um_per_px=stitch_meta["scale_um_per_px"],
        stage_x_min=stitch_meta["stage_bounds_um"]["x_min"],
        stage_y_min=stitch_meta["stage_bounds_um"]["y_min"],
        fov_w_um=overview_meta["optics"]["frame_width_um"],
        fov_h_um=overview_meta["optics"]["frame_height_um"],
    )


def render_wide_detail(
    overview: OverviewAssets,
    stage_x_um: float,
    stage_y_um: float,
    output_path: Path,
) -> bool:
    """Render the 'wide' detail: 2× FOV crop centered on flake, with red circle.

    Crops a region (2× linear, 4× area) around the flake from the stitched
    overview, draws a red circle at the flake center, and saves a PNG.

    Returns True on success, False if the flake is fully outside the stitch.
    """
    box_w_px = overview.fov_w_um * 2 / overview.um_per_px
    box_h_px = overview.fov_h_um * 2 / overview.um_per_px

    cx = (stage_x_um - overview.stage_x_min) / overview.um_per_px
    cy = (stage_y_um - overview.stage_y_min) / overview.um_per_px

    left = int(round(cx - box_w_px / 2))
    top = int(round(cy - box_h_px / 2))
    right = int(round(cx + box_w_px / 2))
    bottom = int(round(cy + box_h_px / 2))

    stitch_w, stitch_h = overview.stitch.size
    left_c = max(0, left)
    top_c = max(0, top)
    right_c = min(stitch_w, right)
    bottom_c = min(stitch_h, bottom)
    if right_c <= left_c or bottom_c <= top_c:
        return False

    crop = overview.stitch.crop((left_c, top_c, right_c, bottom_c))

    # Red circle at flake center, radius ~3% of crop's smaller dim
    cir_x = cx - left_c
    cir_y = cy - top_c
    radius = max(8, int(min(box_w_px, box_h_px) * 0.03))
    draw = ImageDraw.Draw(crop)
    draw.ellipse(
        [(cir_x - radius, cir_y - radius), (cir_x + radius, cir_y + radius)],
        outline=(255, 0, 0),
        width=3,
    )
    crop.save(output_path)
    return True


def load_summary(seg_dir: Path) -> dict:
    """Load seg/summary.json."""
    path = seg_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json in {seg_dir}")
    with open(path) as f:
        return json.load(f)


def load_frame_geometry(seg_dir: Path, frame_name: str, det_id: int) -> dict | None:
    """Load contour/hull from per-frame JSON for a specific detection.

    Per-frame detections are in the same order as summary detections,
    so det_id indexes directly into the list.
    """
    path = seg_dir / f"{frame_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        frame_data = json.load(f)
    dets = frame_data.get("detections", [])
    if det_id < len(dets):
        return dets[det_id]
    return None


def select_flakes(
    summary: dict,
    tier: int,
    top_n: int | None,
) -> list[Detection]:
    """Select and rank flakes from summary detections.

    Returns flat list of detection dicts with frame/det_id, sorted by
    (tier asc, score desc), filtered to exact tier match, deduped.
    """
    all_dets: list[Detection] = []
    for frame_name, dets in summary.get("detections_by_frame", {}).items():
        for d in dets:
            d.setdefault("frame", frame_name)
            all_dets.append(d)

    # Filter by exact tier match
    filtered = [d for d in all_dets if d.get("tier", 3) == tier]

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


_REVISIT_FILENAME_RE = re.compile(r"^rank\d+_(?P<frame>frame_\d+)_d(?P<det>\d+)_(?P<mag>[\d.]+)x$")


def find_revisit_image(
    chip_dir: Path,
    frame_name: str,
    det_id: int,
    mag: float,
) -> Path | None:
    """Find a revisit image for a detection at a given magnification.

    Revisit images are named ``rank{NN}_{frame}_d{det_id}_{mag:g}x.png``.
    Parses the filename and compares frame/det_id/mag by value so that, e.g.,
    det_id=2 doesn't substring-match files for det_id=28 from the same frame.
    """
    mag_str = f"{mag:g}"
    mag_dir_token = f"{mag_str}x"

    for revisit_dir in chip_dir.iterdir():
        if not revisit_dir.is_dir() or not revisit_dir.name.startswith("revisit_"):
            continue
        if mag_dir_token not in revisit_dir.name:
            continue
        for img in revisit_dir.iterdir():
            if img.suffix != ".png":
                continue
            m = _REVISIT_FILENAME_RE.match(img.stem)
            if m is None:
                continue
            if m.group("frame") == frame_name and int(m.group("det")) == det_id and m.group("mag") == mag_str:
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


def _load_checkpoint(run_dir: Path) -> dict:
    """Load checkpoint.json from run directory, or empty dict."""
    cp_path = run_dir / "checkpoint.json"
    if cp_path.exists():
        with open(cp_path) as f:
            return json.load(f)
    return {}


def _resolve_user(run_dir: Path, user: str | None, checkpoint: dict) -> str:
    """Resolve upload user from explicit value or checkpoint operator field."""
    if user is not None:
        return user
    return checkpoint.get("operator", "FlakeFinder")


def _resolve_scan_name(run_dir: Path, name: str | None, checkpoint: dict) -> str:
    """Resolve scan name from explicit value or checkpoint name field.

    Uses checkpoint["name"] if present, falls back to run_dir.name.
    """
    if name is not None:
        return name
    desc = checkpoint.get("name", "")
    if desc:
        return re.sub(r"[^a-zA-Z0-9_.\-]", "_", desc)
    return run_dir.name


def _get_scan_time(run_dir: Path) -> float:
    """Get scan timestamp from stitch metadata, or current time as fallback."""
    for meta_file in run_dir.glob("overview_*_stitch_meta.json"):
        with open(meta_file) as f:
            stitch_meta = json.load(f)
        ts = stitch_meta.get("timestamp")
        if ts:
            try:
                return datetime.fromisoformat(ts).timestamp()
            except (ValueError, TypeError):
                pass
        break
    return time.time()


def _build_comment(checkpoint: dict) -> str:
    """Build the human-readable scan comment from checkpoint metadata.

    Only the operator's own words go here — the comment is editable on the
    server. Machine-recorded provenance belongs in :func:`_build_meta`.
    """
    parts = []
    if cp_name := checkpoint.get("name", ""):
        parts.append(cp_name)
    if cp_notes := checkpoint.get("notes", ""):
        parts.append(cp_notes)
    return " | ".join(parts) or "FlakeFinder upload"


# Checkpoint args worth recording as provenance. Excludes run-shaping
# arguments (chips/after/limit/seg_jobs) that say nothing about the result.
_META_ARG_KEYS = (
    "git_version",
    "preset",
    "area_rect",
    "initial_z",
    "overview_mag",
    "chip_scan_mag",
    "scan_speed",
    "chip_scan_gain",
    "chip_scan_exposure_ms",
    "focus_map_gain",
    "focus_map_exposure_ms",
    "white_balance",
    "colour_matrix",
    "material",
    "flatfield",
)


def _build_meta(checkpoint: dict) -> dict:
    """Build the read-only provenance dict recorded against the scan.

    ``git_version`` is the revision that ran the scan, taken from the
    checkpoint; ``upload_git_version`` is the revision running this upload,
    which differs whenever a run is uploaded from a different machine or
    after a later pull. Keys absent from older checkpoints are omitted.
    """
    cp_args = checkpoint.get("args", {})
    meta = {key: cp_args[key] for key in _META_ARG_KEYS if cp_args.get(key) is not None}
    if operator := checkpoint.get("operator", ""):
        meta["operator"] = operator
    meta["upload_git_version"] = get_git_version()
    return meta


def _build_flake_payload(
    det: dict,
    eval_token: str | None,
    revisit_tokens: dict[float, str],
    detail_tokens: dict[str, str],
    camera_meta: dict,
) -> dict:
    """Build a flake payload for POST /chips/:id/flakes."""
    pos_x = det["stage_x"] / 1000.0  # um -> mm
    pos_y = det["stage_y"] / 1000.0
    size_um2 = det["size_um2"]
    aspect_ratio = det["aspect_ratio"]
    min_side_um = math.sqrt(size_um2 / aspect_ratio)
    max_side_um = min_side_um * aspect_ratio
    contrast = det["contrast_rgb"]

    payload: dict = {
        "position_x": round(pos_x, 4),
        "position_y": round(pos_y, 4),
        "size": round(size_um2, 1),
        "thickness": det.get("classification"),
        "entropy": round(det["entropy"], 4),
        "max_sidelength": round(max_side_um * 1000, 1),  # um -> nm
        "min_sidelength": round(min_side_um * 1000, 1),
        "false_positive_probability": 0.0,
        "mean_r": round(contrast[0], 4),
        "mean_g": round(contrast[1], 4),
        "mean_b": round(contrast[2], 4),
        "score": det.get("score"),
        "tier": det.get("tier"),
        "thickness_nm": det["thickness_nm"],
        "layers": det.get("layers"),
    }

    if eval_token:
        payload["eval_image"] = eval_token

    if revisit_tokens:
        images = {}
        for mag, token in revisit_tokens.items():
            images[f"{mag:g}"] = {
                "aperture": 6,
                "light_voltage": 6.2,
                "magnification": float(mag),
                "gain": camera_meta["gain"],
                "gamma": int(camera_meta["gamma"] * 100),
                "exposure_time": camera_meta["exposure_s"],
                "white_balance_r": int(camera_meta["white_balance_bgr"][2] * 25),
                "white_balance_g": int(camera_meta["white_balance_bgr"][1] * 25),
                "white_balance_b": int(camera_meta["white_balance_bgr"][0] * 25),
                "file": token,
            }
        payload["images"] = images

    if detail_tokens:
        payload["details"] = dict(detail_tokens)

    return payload


def _upload_file_token(
    base_url: str,
    auth: tuple[str, str],
    key: str,
    path: Path,
) -> tuple[str, str, int]:
    """Upload a file to the staging area. Returns (key, token, size_bytes)."""
    data = path.read_bytes()
    resp = requests.post(
        f"{base_url}/api/uploads",
        data=data,
        auth=auth,
        timeout=120,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"Failed to upload {key} ({path.name}): {resp.status_code} {resp.text[:200]}")
    return key, resp.json()["token"], len(data)


def run(
    run_dir: Path,
    *,
    user: str | None = None,
    material: str = "hBN",
    substrate: str = "285nm",
    tier: int = 1,
    top: int | None = None,
    dry_run: bool = False,
    stage_dir: Path | None = None,
    quiet: bool = False,
    jobs: int = 4,
    name: str | None = None,
    base_url: str = BASE_URL,
    wide_detail: bool = True,
) -> UploadResult:
    """Upload a find-flakes run via the incremental REST API.

    Args:
        run_dir: Path to the run directory.
        user: Scan user name (resolved from checkpoint if None).
        material: Exfoliated material label.
        substrate: Chip thickness / substrate label.
        tier: Tier to select (exact match).
        top: Max flakes per chip (None = all passing tier filter).
        dry_run: Discover and render but don't upload.
        stage_dir: If set, use as the staging directory instead of a tempdir
            and force ``dry_run=True``. Lets you inspect rendered files +
            payload JSON without hitting the network.
        quiet: Only print [upload] status lines, suppress detail.
        jobs: Parallel workers for eval_img rendering.
        name: Scan name override (default: run directory name).
        wide_detail: If True (default), render and upload a 'wide' detail
            image (2× FOV stitch crop, red circle) per flake.

    Returns:
        UploadResult with flake count, upload size, and upload status.

    Raises:
        FileNotFoundError: If run_dir or required files don't exist.
        RuntimeError: If no flakes pass the tier filter or an API call fails.
    """
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if stage_dir is not None:
        dry_run = True

    checkpoint = _load_checkpoint(run_dir)
    scan_name = _resolve_scan_name(run_dir, name, checkpoint)
    resolved_user = _resolve_user(run_dir, user, checkpoint)

    # Discover chips
    chip_indices = discover_chips(run_dir)
    if not chip_indices:
        raise FileNotFoundError("No chips with seg/summary.json found")
    print(f"[upload] {len(chip_indices)} chips with segmentation data")

    # Find overview stitch
    stitch_files = list(run_dir.glob("overview_*_stitch.jpg"))
    if not stitch_files:
        raise FileNotFoundError("No overview stitch image found")
    stitch_path = stitch_files[0]

    # Load stitch metadata for stage bounds
    stitch_meta_path = stitch_path.with_name(stitch_path.stem + "_meta.json")
    if not stitch_meta_path.exists():
        raise FileNotFoundError(f"No stitch metadata at {stitch_meta_path}")
    with open(stitch_meta_path) as f:
        stitch_meta = json.load(f)
    stage_bounds = stitch_meta["stage_bounds_um"]

    # Load overview assets once if we're rendering detail crops
    overview_assets: OverviewAssets | None = None
    if wide_detail:
        overview_assets = load_overview_assets(run_dir)

    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_ctx = contextlib.nullcontext(str(stage_dir))
    else:
        stage_ctx = tempfile.TemporaryDirectory(prefix="upload_")

    with stage_ctx as tmp_root:
        tmp = Path(tmp_root)

        # Create overview_compressed.jpg
        overview_path = tmp / "overview_compressed.jpg"
        make_overview_compressed(stitch_path, overview_path)

        # Collect files to upload and per-chip data for resource creation
        upload_files: list[tuple[str, Path]] = [("overview", overview_path)]
        eval_img_jobs: list[tuple] = []
        # Each entry: (chip_idx, camera_meta, flake_entries)
        # flake_entries: list of (det, eval_key, revisit_keys, detail_keys)
        chip_data: list[tuple[int, dict, list]] = []
        total_flakes = 0

        print(f"[upload] {scan_name} (user={resolved_user}, material={material}, substrate={substrate})")

        for chip_idx in chip_indices:
            chip_dir = run_dir / f"chip_{chip_idx}"
            seg_dir = chip_dir / "seg"

            summary = load_summary(seg_dir)

            scan_dir = find_scan_dir(chip_dir)
            if scan_dir is None:
                if not quiet:
                    print(f"  Chip {chip_idx}: no scan directory found, skipping")
                continue

            scan_meta_path = scan_dir / "scan_meta.json"
            with open(scan_meta_path) as f:
                chip_scan_meta = json.load(f)
            camera_meta = chip_scan_meta["camera"]

            flakes = select_flakes(summary, tier, top)
            if not flakes:
                continue

            n_t1 = sum(1 for d in flakes if d.get("tier") == 1)
            n_t2 = sum(1 for d in flakes if d.get("tier") == 2)
            if not quiet:
                print(f"  Chip {chip_idx}: {len(flakes)} flakes (T1:{n_t1}, T2:{n_t2})")

            revisit_mags = discover_revisit_mags(chip_dir)

            flake_entries: list[tuple[dict, str, dict[float, str], dict[str, str]]] = []
            for flake_idx, det in enumerate(flakes):
                # Queue eval_img rendering
                eval_key = f"c{chip_idx}_f{flake_idx}_eval"
                eval_path = tmp / f"{eval_key}.jpg"
                frame_path = str(scan_dir / f"{det['frame']}.jpg")
                geom = load_frame_geometry(seg_dir, det["frame"], det["det_id"])
                contour = geom.get("contour") if geom else None
                eval_img_jobs.append((frame_path, contour, det["bbox"], str(eval_path)))
                upload_files.append((eval_key, eval_path))

                # Find revisit images
                revisit_keys: dict[float, str] = {}
                for mag in revisit_mags:
                    revisit_img = find_revisit_image(chip_dir, det["frame"], det["det_id"], mag)
                    if revisit_img is not None:
                        rkey = f"c{chip_idx}_f{flake_idx}_{mag:g}x"
                        upload_files.append((rkey, revisit_img))
                        revisit_keys[mag] = rkey

                # Render detail images (synthesized from overview stitch)
                detail_keys: dict[str, str] = {}
                if overview_assets is not None:
                    wide_key = f"c{chip_idx}_f{flake_idx}_wide"
                    wide_path = tmp / f"{wide_key}.png"
                    if render_wide_detail(overview_assets, det["stage_x"], det["stage_y"], wide_path):
                        upload_files.append((wide_key, wide_path))
                        detail_keys["wide"] = wide_key

                flake_entries.append((det, eval_key, revisit_keys, detail_keys))
                total_flakes += 1

            chip_data.append((chip_idx, camera_meta, flake_entries))

        if total_flakes == 0:
            raise RuntimeError("No flakes pass the tier filter")

        if dry_run:
            # In stage_dir mode, render eval images and dump payloads for inspection.
            if stage_dir is not None:
                print(f"[upload] Rendering {len(eval_img_jobs)} eval images into {stage_dir}...")
                with ProcessPoolExecutor(max_workers=jobs) as pool:
                    list(pool.map(_render_and_save, eval_img_jobs))

                # Build mock tokens (using the upload key as the token) so we
                # can render representative payloads without server roundtrip.
                mock_tokens = {key: f"mock-{key}" for key, _ in upload_files}
                for chip_idx, camera_meta, flake_entries in chip_data:
                    flake_payloads = [
                        _build_flake_payload(
                            det,
                            mock_tokens.get(eval_key),
                            {mag: mock_tokens[k] for mag, k in revisit_keys.items() if k in mock_tokens},
                            {name: mock_tokens[k] for name, k in detail_keys.items() if k in mock_tokens},
                            camera_meta,
                        )
                        for det, eval_key, revisit_keys, detail_keys in flake_entries
                    ]
                    payload_path = Path(stage_dir) / f"c{chip_idx}_flake_payloads.json"
                    with open(payload_path, "w") as f:
                        json.dump(flake_payloads, f, indent=2)
                    print(f"  Wrote {payload_path} ({len(flake_payloads)} flakes)")
            print(f"[upload] Dry-run: {total_flakes} flakes, {len(upload_files)} files to upload")
            return UploadResult(total_flakes=total_flakes, upload_mb=0.0, uploaded=False)

        # Render eval images in parallel
        print(f"[upload] Rendering {len(eval_img_jobs)} eval images...")
        t0 = time.monotonic()
        rendered = 0
        failed_eval_paths: set[str] = set()
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_render_and_save, job): job for job in eval_img_jobs}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    rendered += 1
                else:
                    failed_eval_paths.add(futures[fut][3])
        elapsed = time.monotonic() - t0
        if not quiet:
            print(f"  Rendered {rendered} eval images in {elapsed:.1f}s")
        if failed_eval_paths:
            print(f"  WARNING: {len(failed_eval_paths)} eval images failed to render")

        # Remove failed eval images from upload list
        upload_files = [(k, p) for k, p in upload_files if str(p) not in failed_eval_paths]

        # Upload all files in parallel to get tokens
        print(f"[upload] Uploading {len(upload_files)} files...")
        t0 = time.monotonic()
        auth_tuple = get_auth().as_tuple()
        tokens: dict[str, str] = {}
        total_bytes = 0

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(_upload_file_token, base_url, auth_tuple, key, path): key for key, path in upload_files
            }
            for fut in as_completed(futures):
                key, token, size = fut.result()
                tokens[key] = token
                total_bytes += size

        upload_mb = total_bytes / (1024 * 1024)
        elapsed = time.monotonic() - t0
        if not quiet:
            print(f"  Staged {upload_mb:.1f} MB in {elapsed:.1f}s ({upload_mb / max(elapsed, 0.001):.1f} MB/s)")

        # Create resources via REST API
        session = requests.Session()
        session.auth = auth_tuple

        # Create scan
        scan_time = _get_scan_time(run_dir)
        comment = _build_comment(checkpoint)
        resp = session.post(
            f"{base_url}/api/scans",
            json={
                "name": scan_name,
                "user": resolved_user,
                "time": int(scan_time),
                "source": "flakefinder",
                "comment": comment,
                "meta": _build_meta(checkpoint),
            },
            timeout=30,
        )
        if resp.status_code != 201:
            raise RuntimeError(f"Create scan failed: {resp.status_code} {resp.text[:500]}")
        scan_id = resp.json()["id"]

        # Upload overview
        resp = session.put(
            f"{base_url}/api/scans/{scan_id}/overview",
            json={"image": tokens["overview"], "stage_bounds": stage_bounds},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Upload overview failed: {resp.status_code} {resp.text[:500]}")

        # Create chips and flakes
        for chip_idx, camera_meta, flake_entries in chip_data:
            resp = session.post(
                f"{base_url}/api/scans/{scan_id}/chips",
                json={"wafer": substrate, "material": material},
                timeout=30,
            )
            if resp.status_code != 201:
                raise RuntimeError(f"Create chip {chip_idx} failed: {resp.status_code} {resp.text[:500]}")
            chip_id = resp.json()["id"]

            flake_payloads = []
            for det, eval_key, revisit_keys, detail_keys in flake_entries:
                resolved_revisit_tokens = {mag: tokens[rkey] for mag, rkey in revisit_keys.items() if rkey in tokens}
                resolved_detail_tokens = {name: tokens[k] for name, k in detail_keys.items() if k in tokens}
                payload = _build_flake_payload(
                    det,
                    tokens.get(eval_key),
                    resolved_revisit_tokens,
                    resolved_detail_tokens,
                    camera_meta,
                )
                flake_payloads.append(payload)

            resp = session.post(
                f"{base_url}/api/chips/{chip_id}/flakes",
                json=flake_payloads,
                timeout=60,
            )
            if resp.status_code != 201:
                raise RuntimeError(f"Create flakes for chip {chip_idx} failed: {resp.status_code} {resp.text[:500]}")
            if not quiet:
                print(f"  Chip {chip_idx}: {len(flake_payloads)} flakes created")

        print(f"[upload] Complete -- {total_flakes} flakes, {upload_mb:.1f} MB uploaded")
        print(f"[upload] View at: {base_url}")
        return UploadResult(total_flakes=total_flakes, upload_mb=upload_mb, uploaded=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a find-flakes run to flakes.sharpelab.science",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sls upload scans/run_20260220_1543/
  sls upload scans/run_20260220_1543/ --dry-run
  sls upload scans/run_20260220_1543/ --tier 1 --top 20
  sls upload scans/run_20260220_1543/ --user Sandesh --substrate 285nm
""",
    )
    parser.add_argument("run_dir", type=Path, help="Find-flakes run directory")
    parser.add_argument("--user", default=None, help="Scan user (default: from checkpoint notes or 'FlakeFinder')")
    parser.add_argument("--material", default="hBN", help="Exfoliated material (default: hBN)")
    parser.add_argument("--substrate", default="285nm", help="Chip thickness / substrate (default: 285nm)")
    parser.add_argument("--tier", type=int, default=1, help="Tier to select, exact match (default: 1)")
    parser.add_argument("--top", type=int, default=None, help="Max flakes per chip (default: all passing tier filter)")
    parser.add_argument("--dry-run", action="store_true", help="Discover flakes but don't upload")
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Render files into this directory and write payload JSONs. Implies --dry-run.",
    )
    parser.add_argument(
        "--no-detail-wide",
        action="store_true",
        help="Disable the per-flake 'wide' detail image (default: enabled)",
    )
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel workers for eval_img rendering")
    parser.add_argument("--name", default=None, help="Scan name override (default: run directory name)")
    parser.add_argument("--url", default=BASE_URL, help=f"Target server URL (default: {BASE_URL})")
    args = parser.parse_args()

    try:
        run(
            args.run_dir,
            user=args.user,
            material=args.material,
            substrate=args.substrate,
            tier=args.tier,
            top=args.top,
            dry_run=args.dry_run,
            stage_dir=args.stage_dir,
            jobs=args.jobs,
            name=args.name,
            base_url=args.url,
            wide_detail=not args.no_detail_wide,
        )
        return 0
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
