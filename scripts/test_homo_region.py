"""Compare 4 candidate metrics for "largest homogeneous region" within flakes.

Evaluates: tile variance, local entropy, uniform flood-fill,
and Otsu sub-segmentation on real detections from a chip scan.

Usage:
    uv run python scripts/test_homo_region.py scans/run_20260221_1651 --chip 0
    uv run python scripts/test_homo_region.py scans/run_20260221_1651 --chip 0 --top 5
    uv run python scripts/test_homo_region.py scans/run_20260221_1651 --chip 0 --top 5 --visualize 1
    uv run python scripts/test_homo_region.py scans/run_20260221_1651 --chip 0 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy import ndimage

REPO_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class MetricResult(NamedTuple):
    area_um2: float
    time_ms: float
    mask: np.ndarray  # bool mask of the homogeneous region (frame-sized)


class DetectionResult(NamedTuple):
    rank: int
    frame: str
    det_idx: int
    score: float
    size_um2: float
    tile: MetricResult
    entropy: MetricResult
    sliding: MetricResult
    subseg: MetricResult


# ---------------------------------------------------------------------------
# Metric 1: Tile variance
# ---------------------------------------------------------------------------


def metric_tile_variance(
    image: np.ndarray,
    mask: np.ndarray,
    um_per_px: float,
    tile_size: int = 5,
) -> MetricResult:
    t0 = time.perf_counter()
    h, w = mask.shape

    # Number of full tiles
    th = h // tile_size
    tw = w // tile_size

    # Crop to exact tile grid
    crop_h = th * tile_size
    crop_w = tw * tile_size
    img_crop = image[:crop_h, :crop_w].astype(np.float32)
    mask_crop = mask[:crop_h, :crop_w]

    # Reshape into tiles: (th, tile_size, tw, tile_size, 3)
    tiles_img = img_crop.reshape(th, tile_size, tw, tile_size, 3)
    tiles_mask = mask_crop.reshape(th, tile_size, tw, tile_size)

    # Per-tile: fraction of mask pixels
    tile_coverage = tiles_mask.mean(axis=(1, 3))  # (th, tw)
    valid_tiles = tile_coverage > 0.5  # tile mostly inside flake

    # Per-tile color variance (mean across channels of per-channel variance)
    tile_var = tiles_img.var(axis=(1, 3)).mean(axis=-1)  # (th, tw)

    # Threshold: only consider valid tiles
    valid_vars = tile_var[valid_tiles]
    if valid_vars.size < 2:
        return MetricResult(0.0, (time.perf_counter() - t0) * 1000, np.zeros_like(mask))

    # Use just the median — median+1σ was too permissive and included
    # nearly the entire flake even with defects present
    threshold = float(np.median(valid_vars))
    uniform_tiles = valid_tiles & (tile_var < threshold)

    # Flood-fill contiguous uniform tiles
    labeled, n_labels = ndimage.label(uniform_tiles)
    if n_labels == 0:
        return MetricResult(0.0, (time.perf_counter() - t0) * 1000, np.zeros_like(mask))

    sizes = ndimage.sum(uniform_tiles, labeled, range(1, n_labels + 1))
    best_label = int(np.argmax(sizes)) + 1
    best_tile_mask = labeled == best_label

    # Expand tile mask back to pixel coordinates
    region_mask = np.zeros_like(mask)
    for ty in range(th):
        for tx in range(tw):
            if best_tile_mask[ty, tx]:
                y0, y1 = ty * tile_size, (ty + 1) * tile_size
                x0, x1 = tx * tile_size, (tx + 1) * tile_size
                region_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]

    area_px = int(region_mask.sum())
    area_um2 = area_px * um_per_px**2
    elapsed = (time.perf_counter() - t0) * 1000
    return MetricResult(round(area_um2, 1), round(elapsed, 1), region_mask)


# ---------------------------------------------------------------------------
# Metric 2: Local entropy
# ---------------------------------------------------------------------------


def _local_entropy_cv(channel: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Compute local entropy using uniform-filter histogramming (no skimage)."""
    # Use local std as a proxy for entropy — faster and no extra deps
    # Local std via: sqrt(E[X²] - E[X]²)
    ch = channel.astype(np.float32)
    ksize = 2 * radius + 1

    mean = cv2.blur(ch, (ksize, ksize))
    mean_sq = cv2.blur(ch * ch, (ksize, ksize))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    return local_std


def metric_local_entropy(
    image: np.ndarray,
    mask: np.ndarray,
    um_per_px: float,
    radius: int = 5,
) -> MetricResult:
    t0 = time.perf_counter()

    # Use green channel (most informative for hBN contrast)
    green = image[:, :, 1]
    local_std = _local_entropy_cv(green, mask, radius)

    # Only consider pixels inside the flake
    flake_stds = local_std[mask]
    if flake_stds.size < 10:
        return MetricResult(0.0, (time.perf_counter() - t0) * 1000, np.zeros_like(mask))

    # Threshold: 40th percentile (low std = uniform)
    threshold = float(np.percentile(flake_stds, 40))
    uniform = mask & (local_std <= threshold)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    uniform_clean = cv2.morphologyEx(uniform.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Largest connected component
    labeled, n_labels = ndimage.label(uniform_clean)
    if n_labels == 0:
        return MetricResult(0.0, (time.perf_counter() - t0) * 1000, np.zeros_like(mask))

    sizes = ndimage.sum(uniform_clean, labeled, range(1, n_labels + 1))
    best_label = int(np.argmax(sizes)) + 1
    region_mask = (labeled == best_label).astype(bool)

    area_px = int(region_mask.sum())
    area_um2 = area_px * um_per_px**2
    elapsed = (time.perf_counter() - t0) * 1000
    return MetricResult(round(area_um2, 1), round(elapsed, 1), region_mask)


# ---------------------------------------------------------------------------
# Metric 3: Uniform flood-fill from interior
# ---------------------------------------------------------------------------


def metric_flood_fill(
    image: np.ndarray,
    mask: np.ndarray,
    um_per_px: float,
    std_threshold: float = 4.0,
    block_size: int = 3,
) -> MetricResult:
    """Flood-fill from the deepest interior point, growing while local color stays uniform.

    Uses a per-pixel local std map (same as local entropy metric) to define
    "uniform" pixels, then finds the largest connected component of uniform
    pixels within the flake. Seeds from the distance-transform peak to
    break ties when multiple uniform regions exist.
    """
    t0 = time.perf_counter()

    green = image[:, :, 1].astype(np.float32)
    ksize = 2 * block_size + 1
    mean = cv2.blur(green, (ksize, ksize))
    mean_sq = cv2.blur(green * green, (ksize, ksize))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    # Uniform pixels: low local std AND inside the flake mask
    uniform = mask & (local_std <= std_threshold)

    # Morphological cleanup to remove isolated noise pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    uniform_clean = cv2.morphologyEx(uniform.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Label connected components
    labeled, n_labels = ndimage.label(uniform_clean)
    if n_labels == 0:
        elapsed = (time.perf_counter() - t0) * 1000
        return MetricResult(0.0, round(elapsed, 1), np.zeros_like(mask))

    # Find the component containing the deepest interior point (distance
    # transform peak). This biases toward the "main" uniform region rather
    # than a small isolated patch.
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peak_idx = np.argmax(dist)
    peak_y, peak_x = np.unravel_index(peak_idx, dist.shape)
    seed_label = labeled[peak_y, peak_x]

    if seed_label > 0:
        # Use the component containing the seed
        region_mask = (labeled == seed_label).astype(bool)
    else:
        # Seed not in any uniform region — fall back to largest CC
        sizes = ndimage.sum(uniform_clean, labeled, range(1, n_labels + 1))
        best_label = int(np.argmax(sizes)) + 1
        region_mask = (labeled == best_label).astype(bool)

    area_px = int(region_mask.sum())
    area_um2 = area_px * um_per_px**2
    elapsed = (time.perf_counter() - t0) * 1000
    return MetricResult(round(area_um2, 1), round(elapsed, 1), region_mask)


# ---------------------------------------------------------------------------
# Metric 4: Sub-segmentation (reuse _otsu_split from segmentation.py)
# ---------------------------------------------------------------------------


def _sensitive_otsu_split(
    norm_contrast: np.ndarray,
    component: np.ndarray,
    min_size_px: int,
    std_threshold: float = 0.3,
    range_threshold: float = 0.15,
) -> list[np.ndarray] | None:
    """Otsu split with much lower thresholds than production segmentation.

    Production uses std<0.8 and range<0.5 — far too coarse to split a T1
    flake with a subtle pinch defect. This version catches smaller contrast
    differences.
    """
    rows = np.any(component, axis=1)
    cols = np.any(component, axis=0)
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    sl = (slice(y_idx[0], y_idx[-1] + 1), slice(x_idx[0], x_idx[-1] + 1))
    roi_comp = component[sl]
    roi_cc = norm_contrast[sl]

    stds = [roi_cc[:, :, c][roi_comp].std() for c in range(3)]
    best_ch = int(np.argmax(stds))
    roi_ch = roi_cc[:, :, best_ch]
    blob_vals = roi_ch[roi_comp]

    if blob_vals.std() < std_threshold:
        return None

    v_min, v_max = blob_vals.min(), blob_vals.max()
    if v_max - v_min < range_threshold:
        return None

    scaled = ((blob_vals - v_min) / (v_max - v_min) * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = v_min + (thresh_val / 255) * (v_max - v_min)

    sub_components = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    comp_size = int(roi_comp.sum())
    for is_high in [False, True]:
        roi_sub = roi_comp & ((roi_ch >= thresh) if is_high else (roi_ch < thresh))
        roi_clean = cv2.morphologyEx(
            roi_sub.astype(np.uint8),
            cv2.MORPH_OPEN,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        roi_labels, n_sub = ndimage.label(roi_clean)
        for j in range(1, n_sub + 1):
            roi_piece = roi_labels == j
            if int(roi_piece.sum()) >= min_size_px:
                full = np.zeros(component.shape, dtype=bool)
                full[sl] = roi_piece
                sub_components.append(full)

    if len(sub_components) >= 2:
        return sub_components
    if len(sub_components) == 1 and int(sub_components[0].sum()) < comp_size * 0.95:
        return sub_components
    return None


def metric_subseg(
    image: np.ndarray,
    mask: np.ndarray,
    um_per_px: float,
) -> MetricResult:
    t0 = time.perf_counter()

    # Build normalized contrast (same as segment_frame)
    bg_modes = np.array(
        [
            float(np.argmax(np.histogram(image[:, :, c].ravel(), bins=256, range=(0, 256))[0][20:230]) + 20)
            for c in range(3)
        ]
    )
    img_f = image.astype(np.float32) - bg_modes[np.newaxis, np.newaxis, :]
    norm_contrast = img_f / np.maximum(bg_modes[np.newaxis, np.newaxis, :], 1.0)

    min_size_px = max(50, int(mask.sum() * 0.05))

    # Iterative splitting with sensitive thresholds
    final: list[np.ndarray] = []
    queue: list[tuple[np.ndarray, int]] = [(mask, 0)]
    max_depth = 3

    while queue:
        comp, depth = queue.pop()
        if depth >= max_depth:
            final.append(comp)
            continue
        pieces = _sensitive_otsu_split(norm_contrast, comp, min_size_px)
        if pieces is None:
            final.append(comp)
        else:
            for piece in pieces:
                queue.append((piece, depth + 1))

    if not final:
        elapsed = (time.perf_counter() - t0) * 1000
        return MetricResult(0.0, round(elapsed, 1), np.zeros_like(mask))

    # Find largest piece
    piece_sizes = [int(p.sum()) for p in final]
    best_idx = int(np.argmax(piece_sizes))
    best_mask = final[best_idx]

    area_px = piece_sizes[best_idx]
    area_um2 = area_px * um_per_px**2
    elapsed = (time.perf_counter() - t0) * 1000
    return MetricResult(round(area_um2, 1), round(elapsed, 1), best_mask)


# ---------------------------------------------------------------------------
# Per-detection worker (runs in parallel)
# ---------------------------------------------------------------------------


class _WorkerInput(NamedTuple):
    rank: int
    frame: str
    det_idx: int
    score: float
    size_um2: float
    contour: list[list[int]]
    bbox: list[int]
    frame_path: str
    um_per_px: float


def _process_detection(inp: _WorkerInput) -> DetectionResult:
    """Process one detection: load frame, reconstruct mask, run all 4 metrics."""
    img = cv2.imread(inp.frame_path)
    if img is None:
        empty = MetricResult(0.0, 0.0, np.array([]))
        return DetectionResult(inp.rank, inp.frame, inp.det_idx, inp.score, inp.size_um2, empty, empty, empty, empty)

    h, w = img.shape[:2]

    # Reconstruct binary mask from contour
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(inp.contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    mask_bool = mask > 0

    # Crop to bbox + padding for efficiency
    bx, by, bw, bh = inp.bbox
    pad = 20
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(w, bx + bw + pad)
    y1 = min(h, by + bh + pad)

    img_crop = img[y0:y1, x0:x1]
    mask_crop = mask_bool[y0:y1, x0:x1]

    tile = metric_tile_variance(img_crop, mask_crop, inp.um_per_px)
    entropy = metric_local_entropy(img_crop, mask_crop, inp.um_per_px)
    sliding = metric_flood_fill(img_crop, mask_crop, inp.um_per_px)
    subseg = metric_subseg(img_crop, mask_crop, inp.um_per_px)

    return DetectionResult(
        rank=inp.rank,
        frame=inp.frame,
        det_idx=inp.det_idx,
        score=inp.score,
        size_um2=inp.size_um2,
        tile=MetricResult(tile.area_um2, tile.time_ms, _uncrop(tile.mask, h, w, y0, x0)),
        entropy=MetricResult(entropy.area_um2, entropy.time_ms, _uncrop(entropy.mask, h, w, y0, x0)),
        sliding=MetricResult(sliding.area_um2, sliding.time_ms, _uncrop(sliding.mask, h, w, y0, x0)),
        subseg=MetricResult(subseg.area_um2, subseg.time_ms, _uncrop(subseg.mask, h, w, y0, x0)),
    )


def _uncrop(cropped_mask: np.ndarray, full_h: int, full_w: int, y0: int, x0: int) -> np.ndarray:
    """Place a cropped mask back into a full-frame-sized array."""
    if cropped_mask.size == 0:
        return np.zeros((full_h, full_w), dtype=bool)
    full = np.zeros((full_h, full_w), dtype=bool)
    ch, cw = cropped_mask.shape[:2]
    full[y0 : y0 + ch, x0 : x0 + cw] = cropped_mask
    return full


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_detection(
    result: DetectionResult,
    frame_path: Path,
    contour: list[list[int]],
    bbox: list[int],
    output_path: Path,
    um_per_px: float,
) -> None:
    import matplotlib.pyplot as plt

    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"Warning: could not load {frame_path}")
        return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop region for display
    bx, by, bw, bh = bbox
    pad = 40
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(w, bx + bw + pad)
    y1 = min(h, by + bh + pad)

    crop_rgb = img_rgb[y0:y1, x0:x1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1) Original with contour
    ax = axes[0, 0]
    ax.imshow(crop_rgb)
    pts = np.array(contour, dtype=np.int32)
    ax.plot(pts[:, 0] - x0, pts[:, 1] - y0, "g-", linewidth=1, alpha=0.8)
    title = (
        f"Detection #{result.rank}\n{result.frame} d{result.det_idx}\n"
        f"score={result.score:.3f}  size={result.size_um2:.0f} µm²"
    )
    ax.set_title(title)

    # 2) Tile variance
    ax = axes[0, 1]
    ax.imshow(crop_rgb, alpha=0.6)
    tile_crop = result.tile.mask[y0:y1, x0:x1]
    overlay = np.zeros((*tile_crop.shape, 4))
    overlay[tile_crop > 0] = [0, 1, 0, 0.4]
    ax.imshow(overlay)
    ax.set_title(f"Tile Variance\n{result.tile.area_um2:.0f} µm²  ({result.tile.time_ms:.0f} ms)")

    # 3) Local entropy
    ax = axes[0, 2]
    ax.imshow(crop_rgb, alpha=0.6)
    ent_crop = result.entropy.mask[y0:y1, x0:x1]
    overlay = np.zeros((*ent_crop.shape, 4))
    overlay[ent_crop > 0] = [0, 0.5, 1, 0.4]
    ax.imshow(overlay)
    ax.set_title(f"Local Entropy\n{result.entropy.area_um2:.0f} µm²  ({result.entropy.time_ms:.0f} ms)")

    # 4) Flood-fill uniform region
    ax = axes[1, 0]
    ax.imshow(crop_rgb, alpha=0.6)
    flood_crop = result.sliding.mask[y0:y1, x0:x1]
    overlay = np.zeros((*flood_crop.shape, 4))
    overlay[flood_crop > 0] = [1, 0.3, 0, 0.4]
    ax.imshow(overlay)
    ax.set_title(f"Flood Fill\n{result.sliding.area_um2:.0f} µm²  ({result.sliding.time_ms:.0f} ms)")

    # 5) Sub-segmentation
    ax = axes[1, 1]
    ax.imshow(crop_rgb, alpha=0.6)
    sub_crop = result.subseg.mask[y0:y1, x0:x1]
    overlay = np.zeros((*sub_crop.shape, 4))
    overlay[sub_crop > 0] = [0.8, 0, 0.8, 0.4]
    ax.imshow(overlay)
    ax.set_title(f"Sub-segmentation\n{result.subseg.area_um2:.0f} µm²  ({result.subseg.time_ms:.0f} ms)")

    # 6) Comparison bar chart
    ax = axes[1, 2]
    names = ["Tile\nVar", "Local\nEntropy", "Flood\nFill", "Sub-\nSeg"]
    values = [result.tile.area_um2, result.entropy.area_um2, result.sliding.area_um2, result.subseg.area_um2]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
    bars = ax.bar(names, values, color=colors, alpha=0.8)
    ax.axhline(result.size_um2, color="gray", linestyle="--", label=f"Total: {result.size_um2:.0f} µm²")
    ax.set_ylabel("Area (µm²)")
    ax.legend()
    for bar, v in zip(bars, values, strict=True):
        pct = v / result.size_um2 * 100 if result.size_um2 > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{pct:.0f}%", ha="center", fontsize=9)
    ax.set_title("Comparison")

    fig.suptitle(f"Homogeneous Region Analysis — Det #{result.rank}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare homogeneous-region metrics on detected flakes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="Run directory (e.g. scans/run_20260221_1651)")
    parser.add_argument("--chip", type=int, required=True, help="Chip index")
    parser.add_argument("--top", type=int, default=None, help="Limit to top N detections by score (default: all T1)")
    parser.add_argument(
        "--visualize",
        type=int,
        default=None,
        help="Produce diagnostic PNG for Nth-ranked detection (1-indexed)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Save JSON results to file")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Parallel worker count")
    args = parser.parse_args()

    run_dir = (REPO_DIR / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    chip_dir = run_dir / f"chip_{args.chip}"
    seg_dir = chip_dir / "seg"
    summary_path = seg_dir / "summary.json"

    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return 1

    # Find scan directory (scan_10x, scan_20x, etc.)
    scan_dirs = sorted(chip_dir.glob("scan_*x"))
    if not scan_dirs:
        print(f"Error: no scan_*x directory found in {chip_dir}")
        return 1
    scan_dir = scan_dirs[0]
    meta_path = scan_dir / "scan_meta.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found")
        return 1

    with open(meta_path) as f:
        scan_meta = json.load(f)
    um_per_px = scan_meta["optics"]["sample_pixel_x_um"]
    mag = scan_meta["optics"].get("objective_mag", "?")
    print(f"Scan: {scan_dir.name}  mag={mag}x  um_per_px={um_per_px:.4f}")

    # Load summary and extract detections
    with open(summary_path) as f:
        summary = json.load(f)

    material = summary.get("params", {}).get("material", "hbn_medium")
    print(f"Material: {material}  Stats: {summary['stats']}")

    # Collect all detections with tier/score from summary
    all_dets = []
    for frame_name, dets in summary["detections_by_frame"].items():
        for i, d in enumerate(dets):
            all_dets.append((frame_name, i, d))

    # Filter: T1 by default, or top N by score
    if args.top is not None:
        # Sort by (tier asc, score desc), take top N
        all_dets.sort(key=lambda x: (x[2].get("tier", 3), -x[2].get("score", 0)))
        selected = all_dets[: args.top]
    else:
        selected = [(f, i, d) for f, i, d in all_dets if d.get("tier") == 1]
        selected.sort(key=lambda x: -x[2].get("score", 0))

    if not selected:
        print("No detections matched the filter criteria.")
        return 0

    print(f"\nProcessing {len(selected)} detections (top={args.top}, T1={args.top is None})...")

    # For each selected detection, load the per-frame JSON for contour
    worker_inputs: list[_WorkerInput] = []
    contour_cache: dict[int, list[list[int]]] = {}  # rank -> contour (for viz)
    bbox_cache: dict[int, list[int]] = {}

    for rank, (frame_name, det_idx_in_summary, det_summary) in enumerate(selected, 1):
        # Load per-frame JSON to get contour (stripped from summary)
        frame_json_path = seg_dir / f"{frame_name}.json"
        if not frame_json_path.exists():
            print(f"  Warning: {frame_json_path} not found, skipping")
            continue

        with open(frame_json_path) as f:
            frame_data = json.load(f)

        # Match detection by det_id or bbox
        det_id = det_summary.get("det_id")
        matched_det = None
        if det_id is not None:
            for fd in frame_data["detections"]:
                # det_id in summary = index in the frame's detection list
                idx = frame_data["detections"].index(fd)
                if idx == det_id:
                    matched_det = fd
                    break
        if matched_det is None:
            # Fallback: match by bbox
            target_bbox = det_summary["bbox"]
            for fd in frame_data["detections"]:
                if fd["bbox"] == target_bbox:
                    matched_det = fd
                    break
        if matched_det is None:
            print(f"  Warning: could not match detection in {frame_name}, skipping")
            continue

        frame_path = scan_dir / f"{frame_name}.jpg"
        if not frame_path.exists():
            print(f"  Warning: {frame_path} not found, skipping")
            continue

        contour_cache[rank] = matched_det["contour"]
        bbox_cache[rank] = matched_det["bbox"]

        worker_inputs.append(
            _WorkerInput(
                rank=rank,
                frame=frame_name,
                det_idx=det_id if det_id is not None else det_idx_in_summary,
                score=det_summary.get("score", 0),
                size_um2=det_summary.get("size_um2", 0),
                contour=matched_det["contour"],
                bbox=matched_det["bbox"],
                frame_path=str(frame_path),
                um_per_px=um_per_px,
            )
        )

    if not worker_inputs:
        print("No detections could be loaded.")
        return 1

    # Run all detections in parallel
    t_total = time.perf_counter()
    results: list[DetectionResult] = []

    n_workers = min(args.jobs, len(worker_inputs))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_detection, inp): inp.rank for inp in worker_inputs}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            t_sum = result.tile.time_ms + result.entropy.time_ms + result.sliding.time_ms + result.subseg.time_ms
            print(f"  Done: #{result.rank} {result.frame} ({t_sum:.0f} ms total)")

    results.sort(key=lambda r: r.rank)
    total_elapsed = time.perf_counter() - t_total

    # Print summary table
    print(f"\n{'=' * 110}")
    hdr = (
        f"{'Det':<5} {'Frame':<14} {'Score':>6} {'Size(µm²)':>10} "
        f"{'Tile':>10} {'Entropy':>10} {'Flood':>10} {'SubSeg':>10}   {'Time(ms)':>20}"
    )
    print(hdr)
    print(f"{'-' * 110}")

    json_results = []
    for r in results:
        times = f"{r.tile.time_ms:.0f}/{r.entropy.time_ms:.0f}/{r.sliding.time_ms:.0f}/{r.subseg.time_ms:.0f}"
        print(
            f"#{r.rank:<4} {r.frame:<14} {r.score:>6.3f} {r.size_um2:>10.0f} "
            f"{r.tile.area_um2:>10.0f} {r.entropy.area_um2:>10.0f} "
            f"{r.sliding.area_um2:>10.0f} {r.subseg.area_um2:>10.0f}   {times:>20}"
        )

        json_results.append(
            {
                "rank": r.rank,
                "frame": r.frame,
                "det_idx": r.det_idx,
                "score": r.score,
                "size_um2": r.size_um2,
                "tile_area_um2": r.tile.area_um2,
                "tile_time_ms": r.tile.time_ms,
                "entropy_area_um2": r.entropy.area_um2,
                "entropy_time_ms": r.entropy.time_ms,
                "sliding_area_um2": r.sliding.area_um2,
                "sliding_time_ms": r.sliding.time_ms,
                "subseg_area_um2": r.subseg.area_um2,
                "subseg_time_ms": r.subseg.time_ms,
            }
        )

    print(f"{'=' * 110}")
    print(f"Total: {len(results)} detections in {total_elapsed:.1f}s ({n_workers} workers)")

    # Save JSON
    if args.output:
        out_path = REPO_DIR / args.output if not args.output.is_absolute() else args.output
        with open(out_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Saved JSON: {out_path}")

    # Visualization
    if args.visualize is not None:
        viz_rank = args.visualize
        viz_result = next((r for r in results if r.rank == viz_rank), None)
        if viz_result is None:
            print(f"Warning: detection rank {viz_rank} not found in results")
        else:
            frame_path = scan_dir / f"{viz_result.frame}.jpg"
            output_path = seg_dir / f"homo_region_det{viz_rank}.png"
            visualize_detection(
                viz_result,
                frame_path,
                contour_cache[viz_rank],
                bbox_cache[viz_rank],
                output_path,
                um_per_px,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
