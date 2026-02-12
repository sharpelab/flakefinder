"""Segment flakes using flatfield correction + contrast threshold.

Usage:
    python segment_flakes.py frame.jpg --flatfield flatfield.npy
    python segment_flakes.py frame.jpg --flatfield flatfield.npy --save-plot
    python segment_flakes.py frame.jpg --flatfield flatfield.npy -o output.jpg
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from flakefinder.scan_utils import apply_flatfield


def histogram_mode(image: np.ndarray, channel: int | None = None) -> float:
    """Find the histogram peak (mode) of an image or single channel."""
    if channel is not None:
        data = image[:, :, channel]
    else:
        data = image
    hist, _ = np.histogram(data.ravel(), bins=256, range=(0, 256))
    hist[:20] = 0
    hist[230:] = 0
    return float(np.argmax(hist))


def compute_dark_frac(image: np.ndarray, threshold: float = 30.0) -> float:
    """Fraction of pixels with mean brightness below threshold (off-chip indicator)."""
    return float((image.mean(axis=2) < threshold).mean())


def _make_detection(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
) -> dict:
    """Build a detection dict from a binary component mask."""
    ys, xs = np.where(component)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    region_pixels = image[component].astype(np.float32)
    mean_contrast = float(np.mean(region_pixels - bg_modes))

    ch_means = region_pixels.mean(axis=0)  # BGR
    norm_contrast_bgr = (ch_means - bg_modes) / np.maximum(bg_modes, 1.0)

    # Shape metrics via contour analysis
    comp_u8 = component.astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea) if contours else None
    if cnt is not None:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / max(hull_area, 1)
        circularity = (4 * np.pi * area) / max(perim**2, 1)
    else:
        solidity = 0.0
        circularity = 0.0

    # Color uniformity: std of per-pixel normalized R contrast within blob
    r_norm = (image[:, :, 2].astype(np.float32) - bg_modes[2]) / max(float(bg_modes[2]), 1.0)
    r_std = float(r_norm[component].std())

    return {
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "center": [round((x_min + x_max) / 2, 1), round((y_min + y_max) / 2, 1)],
        "size_px": int(component.sum()),
        "mean_contrast": round(mean_contrast, 1),
        "contrast_rgb": [
            round(float(norm_contrast_bgr[2]), 4),
            round(float(norm_contrast_bgr[1]), 4),
            round(float(norm_contrast_bgr[0]), 4),
        ],
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
        "r_std": round(r_std, 4),
    }


def _subsegment_by_contrast(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
    min_size_px: int,
) -> list[np.ndarray]:
    """Split a blob into sub-regions if internal R contrast is bimodal.

    Uses Otsu thresholding on normalized R contrast within the blob.
    Returns list of sub-component masks (may be just [component] if no split).
    """
    # Compute per-pixel normalized R contrast within blob
    r_contrast = (image[:, :, 2].astype(np.float32) - bg_modes[2]) / max(bg_modes[2], 1.0)
    blob_r = r_contrast[component]

    # Only attempt split if internal R contrast has high variance
    if blob_r.std() < 0.8:
        return [component]

    # Otsu on the R contrast values within the blob
    # Shift to 0-255 range for cv2.threshold
    r_min, r_max = blob_r.min(), blob_r.max()
    if r_max - r_min < 0.5:
        return [component]

    r_scaled = ((blob_r - r_min) / (r_max - r_min) * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(r_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert threshold back to R contrast space
    r_thresh = r_min + (thresh_val / 255) * (r_max - r_min)

    # Split into low-R and high-R sub-masks
    sub_components = []
    for is_high in [False, True]:
        if is_high:
            sub_mask = component & (r_contrast >= r_thresh)
        else:
            sub_mask = component & (r_contrast < r_thresh)

        # Clean up: morphological open to remove noise, then re-label
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sub_clean = cv2.morphologyEx(sub_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        sub_labels, n_sub = ndimage.label(sub_clean)

        for j in range(1, n_sub + 1):
            sub_comp = sub_labels == j
            if int(sub_comp.sum()) >= min_size_px // 2:
                sub_components.append(sub_comp)

    # Use sub-segmentation if we got pieces that together cover most of the blob
    if len(sub_components) >= 2:
        return sub_components
    # Even with one qualifying sub-component, use it if it purifies the contrast
    # (the other side was too small but removing it cleans the detection)
    if len(sub_components) == 1 and int(sub_components[0].sum()) < int(component.sum()) * 0.95:
        return sub_components
    return [component]


def segment_frame(
    image: np.ndarray,
    contrast_offset: float = 15.0,
    min_size_px: int = 1000,
    edge_margin_px: int = 50,
    min_solidity: float = 0.0,
) -> list[dict]:
    """Segment flakes by thresholding above background mode + offset.

    Returns list of detected regions with bbox, size, center, mean_contrast.
    Large blobs with bimodal R contrast are sub-segmented by thickness.
    Detections with solidity below *min_solidity* are filtered out (tape rejection).
    """
    bg_modes = np.array([histogram_mode(image, c) for c in range(3)])

    above = image.astype(np.float32) - bg_modes[np.newaxis, np.newaxis, :]
    mask = np.any(above > contrast_offset, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    labels, n_labels = ndimage.label(mask_clean)

    h, w = image.shape[:2]
    detections = []

    for i in range(1, n_labels + 1):
        component = labels == i
        size = int(component.sum())

        if size < min_size_px:
            continue

        ys, xs = np.where(component)
        cx = (int(xs.min()) + int(xs.max())) / 2
        cy = (int(ys.min()) + int(ys.max())) / 2

        if cx < edge_margin_px or cx > w - edge_margin_px:
            continue
        if cy < edge_margin_px or cy > h - edge_margin_px:
            continue

        # Solidity filter on parent blob (before sub-segmentation).
        # Sub-segments inherit the pass — their irregular Otsu-split
        # boundaries would falsely fail solidity checks.
        if min_solidity > 0:
            comp_u8 = component.astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                parent_solidity = cv2.contourArea(cnt) / max(hull_area, 1)
                if parent_solidity < min_solidity:
                    continue

        # Try contrast-based sub-segmentation for large blobs
        sub_components = _subsegment_by_contrast(image, component, bg_modes, min_size_px)

        for sub_comp in sub_components:
            det = _make_detection(image, sub_comp, bg_modes)
            # Re-check edge margin for sub-components
            scx, scy = det["center"]
            if scx < edge_margin_px or scx > w - edge_margin_px:
                continue
            if scy < edge_margin_px or scy > h - edge_margin_px:
                continue
            detections.append(det)

    detections.sort(key=lambda d: d["size_px"], reverse=True)
    return detections


def draw_scale_bar(image: np.ndarray, um_per_px: float) -> None:
    """Draw a scale bar in the bottom-right corner of *image* (mutates in place)."""
    h, w = image.shape[:2]
    for candidate_um in [500, 200, 100, 50, 20, 10]:
        candidate_px = int(candidate_um / um_per_px)
        if candidate_px <= w * 0.20:
            break
    bar_px = int(candidate_um / um_per_px)
    bar_h = max(4, h // 200)
    margin = max(15, h // 60)
    bx = w - margin - bar_px
    by = h - margin - bar_h
    cv2.rectangle(image, (bx - 1, by - 1), (bx + bar_px + 1, by + bar_h + 1), (0, 0, 0), -1)
    cv2.rectangle(image, (bx, by), (bx + bar_px, by + bar_h), (255, 255, 255), -1)
    label = f"{candidate_um} um"
    font_scale = max(0.4, h / 2000)
    thick = max(1, int(h / 800))
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    lx = bx + (bar_px - lw) // 2
    ly = by - max(4, int(h / 200))
    cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 2)
    cv2.putText(image, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)


def draw_detections(image: np.ndarray, detections: list[dict], um_per_px: float = 0.0) -> np.ndarray:
    """Draw bounding boxes on image. Returns a copy."""
    vis = image.copy()
    for i, d in enumerate(detections):
        bx, by, bw, bh = d["bbox"]
        s = d["size_px"]
        c = d.get("mean_contrast", 0)
        color = (0, 255, 0) if s > 1000 else (0, 255, 255) if s > 500 else (0, 0, 255)
        thickness = 2 if s > 1000 else 1
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, thickness)
        label = f"#{i} {s}px c={c:.0f}"
        cv2.putText(vis, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if um_per_px > 0:
        draw_scale_bar(vis, um_per_px)
    return vis


def save_plot(image: np.ndarray, detections: list[dict], title: str, output_path: Path):
    """Save matplotlib figure with image + threshold mask side by side."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for d in detections:
        bx, by, bw, bh = d["bbox"]
        s = d["size_px"]
        color = "lime" if s > 1000 else "yellow" if s > 500 else "red"
        lw = 2 if s > 1000 else 1
        ax1.add_patch(Rectangle((bx, by), bw, bh, linewidth=lw, edgecolor=color, facecolor="none"))
        ax1.text(bx, by - 4, f"{s}px", color=color, fontsize=7, fontweight="bold")
    ax1.set_title(f"{title}\n{len(detections)} detections")

    bg_modes = np.array([histogram_mode(image, c) for c in range(3)])
    above = image.astype(np.float32) - bg_modes[np.newaxis, np.newaxis, :]
    mask = np.any(above > 15, axis=2)
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"Threshold mask (bg_mode + 15)\nbg_modes BGR: {bg_modes.astype(int)}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment flakes via flatfield + contrast threshold")
    parser.add_argument("input", type=Path, help="Frame image")
    parser.add_argument("--flatfield", type=Path, default=None, help="Flatfield .npy file")
    parser.add_argument("--contrast-offset", type=float, default=15.0, help="Threshold above bg mode")
    parser.add_argument("--min-size", type=int, default=1000, help="Min detection size (px)")
    parser.add_argument("--edge-margin", type=int, default=50, help="Ignore detections near frame edge (px)")
    parser.add_argument(
        "--min-solidity",
        type=float,
        default=0.0,
        help="Filter detections below this solidity (tape rejection, try 0.7)",
    )
    parser.add_argument(
        "--dark-frac-cutoff",
        type=float,
        default=0.05,
        help="Skip frame if dark pixel fraction exceeds this (edge/off-chip filter)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output image path")
    parser.add_argument("--save-plot", action="store_true", help="Also save matplotlib analysis plot")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.36,
        help="µm per pixel (default: 0.36 for 20x bin3)",
    )
    args = parser.parse_args()

    raw = cv2.imread(str(args.input))
    if raw is None:
        print(f"Failed to read {args.input}")
        return 1

    if args.flatfield:
        ff = np.load(str(args.flatfield)).astype(np.float32)
        ff = ff[:, :, ::-1]  # RGB (numpy save convention) -> BGR (cv2)
        corrected = apply_flatfield(raw, ff)
    else:
        corrected = raw

    # Edge/off-chip detection
    dark_frac = compute_dark_frac(corrected)
    skipped = dark_frac > args.dark_frac_cutoff

    if skipped:
        dets = []
        print(f"{args.input.name}: SKIPPED (dark_frac={dark_frac:.3f} > {args.dark_frac_cutoff})")
    else:
        dets = segment_frame(corrected, args.contrast_offset, args.min_size, args.edge_margin, args.min_solidity)
        print(f"{args.input.name}: {len(dets)} detections (dark_frac={dark_frac:.3f})")
        for i, d in enumerate(dets):
            print(f"  #{i}: size={d['size_px']}px, center={d['center']}, contrast={d['mean_contrast']}")

    # Output image — draw on raw to preserve true colors
    out = args.output or args.input.with_suffix(".seg.jpg")
    vis = draw_detections(raw, dets, um_per_px=args.pixel_size)
    cv2.imwrite(str(out), vis)
    print(f"Saved: {out}")

    # Output metadata
    bg_modes = [histogram_mode(corrected, c) for c in range(3)]
    meta = {
        "timestamp": datetime.now().isoformat(),
        "command": ["segment_flakes.py"] + sys.argv[1:],
        "input": str(args.input),
        "output_image": str(out),
        "frame_shape": list(raw.shape),
        "flatfield": str(args.flatfield) if args.flatfield else None,
        "params": {
            "contrast_offset": args.contrast_offset,
            "min_size_px": args.min_size,
            "edge_margin_px": args.edge_margin,
            "dark_frac_cutoff": args.dark_frac_cutoff,
        },
        "dark_frac": round(dark_frac, 4),
        "skipped": skipped,
        "bg_modes_bgr": [round(m, 1) for m in bg_modes],
        "detection_count": len(dets),
        "detections": dets,
    }

    meta_path = Path(str(out).rsplit(".", 1)[0] + ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    if args.save_plot:
        plot_path = args.input.with_suffix(".seg.plot.png")
        save_plot(corrected, dets, args.input.name, plot_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
