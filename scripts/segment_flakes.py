"""Segment flakes using flatfield correction + contrast threshold.

Usage:
    python segment_flakes.py frame.jpg --flatfield flatfield.npy
    python segment_flakes.py frame.jpg --flatfield flatfield.npy --material graphene
    python segment_flakes.py frame.jpg --flatfield flatfield.npy --save-plot
    python segment_flakes.py frame.jpg --flatfield flatfield.npy -o output.jpg
"""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from detector_config import ContrastMode, DetectorConfig
from scipy import ndimage
from scipy.stats import kurtosis as scipy_kurtosis

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


def _analyze_component(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
    norm_contrast: np.ndarray,
    grad_mag: np.ndarray,
    config: DetectorConfig,
) -> dict:
    """Analyze a binary component mask and return detection metrics."""
    rows = np.any(component, axis=1)
    cols = np.any(component, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])

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
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_perim = cv2.arcLength(hull, True)
        solidity = area / max(hull_area, 1)
        circularity = (4 * np.pi * area) / max(perim**2, 1)
        perim_ratio = perim / max(hull_perim, 1.0)
        hull_pts = hull.reshape(-1, 2).tolist()
        contour_pts = cnt.reshape(-1, 2).tolist()
    else:
        solidity = 0.0
        circularity = 0.0
        perim_ratio = 0.0
        hull_pts = []
        contour_pts = []

    # Color uniformity: std of per-pixel normalized contrast within blob (BGR channel order)
    b_vals = norm_contrast[:, :, 0][component]
    b_std = float(b_vals.std())
    b_kurt = float(scipy_kurtosis(b_vals, fisher=True))
    g_vals = norm_contrast[:, :, 1][component]
    g_std = float(g_vals.std())
    g_kurt = float(scipy_kurtosis(g_vals, fisher=True))
    r_vals = norm_contrast[:, :, 2][component]
    r_std = float(r_vals.std())
    r_kurt = float(scipy_kurtosis(r_vals, fisher=True))

    # Internal gradient energy (Sobel on G channel, masked to blob)
    grad_energy = float(grad_mag[component].mean())

    # Histogram entropy of G normalized contrast within blob
    hist, _ = np.histogram(g_vals, bins=50)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    g_entropy = float(-np.sum(probs * np.log2(probs)))

    r_contrast = round(float(norm_contrast_bgr[2]), 4)
    g_contrast = round(float(norm_contrast_bgr[1]), 4)
    b_contrast = round(float(norm_contrast_bgr[0]), 4)

    return {
        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
        "center": [round((x_min + x_max) / 2, 1), round((y_min + y_max) / 2, 1)],
        "size_px": int(component.sum()),
        "mean_contrast": round(mean_contrast, 1),
        "contrast_rgb": [r_contrast, g_contrast, b_contrast],
        "cal_dist": round(config.cal_distance(r_contrast, g_contrast), 4),
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
        "perim_ratio": round(perim_ratio, 4),
        "r_std": round(r_std, 4),
        "r_kurt": round(r_kurt, 4),
        "g_std": round(g_std, 4),
        "g_kurt": round(g_kurt, 4),
        "b_std": round(b_std, 4),
        "b_kurt": round(b_kurt, 4),
        "grad_energy": round(grad_energy, 2),
        "g_entropy": round(g_entropy, 4),
        "hull": hull_pts,
        "contour": contour_pts,
    }


def _otsu_split(
    contrast_channels: np.ndarray,
    component: np.ndarray,
    min_size_px: int,
) -> list[np.ndarray] | None:
    """Try one Otsu split on the highest-variance channel. Returns sub-components or None."""
    # Crop to component bounding box — all ops run on the small ROI
    rows = np.any(component, axis=1)
    cols = np.any(component, axis=0)
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    sl = (slice(y_idx[0], y_idx[-1] + 1), slice(x_idx[0], x_idx[-1] + 1))
    roi_comp = component[sl]
    roi_cc = contrast_channels[sl]

    # Pick the channel with highest within-blob variance
    stds = [roi_cc[:, :, c][roi_comp].std() for c in range(3)]
    best_ch = int(np.argmax(stds))
    roi_ch = roi_cc[:, :, best_ch]
    blob_vals = roi_ch[roi_comp]

    if blob_vals.std() < 0.8:
        return None

    v_min, v_max = blob_vals.min(), blob_vals.max()
    if v_max - v_min < 0.5:
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
            roi_sub.astype(np.uint8), cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        roi_labels, n_sub = ndimage.label(roi_clean)

        for j in range(1, n_sub + 1):
            roi_piece = roi_labels == j
            if int(roi_piece.sum()) >= min_size_px // 2:
                # Expand back to full frame
                full = np.zeros(component.shape, dtype=bool)
                full[sl] = roi_piece
                sub_components.append(full)

    if len(sub_components) >= 2:
        return sub_components
    if len(sub_components) == 1 and int(sub_components[0].sum()) < comp_size * 0.95:
        return sub_components
    return None


def _subsegment_by_contrast(
    image: np.ndarray,
    component: np.ndarray,
    bg_modes: np.ndarray,
    min_size_px: int,
    norm_contrast: np.ndarray,
    max_depth: int = 3,
) -> list[np.ndarray]:
    """Iteratively split a blob using Otsu on the highest-variance color channel.

    Each split produces sub-components that are checked again for further
    splitting, up to *max_depth* levels. This handles cases where extreme
    outlier pixels pull the first Otsu threshold away from subtler gradients.
    """
    # Iterative: keep a work queue of components to try splitting
    final = []
    queue = [(component, 0)]

    while queue:
        comp, depth = queue.pop()
        if depth >= max_depth:
            final.append(comp)
            continue

        pieces = _otsu_split(norm_contrast, comp, min_size_px)
        if pieces is None:
            final.append(comp)
        else:
            for piece in pieces:
                queue.append((piece, depth + 1))

    return final if final else [component]


def segment_frame(
    image: np.ndarray,
    config: DetectorConfig,
    um_per_px: float,
    perim_ratio_thresh: float = 0.0,
) -> list[dict]:
    """Segment flakes by thresholding relative to background mode.

    For 'above' contrast mode (hBN), finds pixels brighter than background.
    For 'below' contrast mode (graphene), finds pixels darker than background.
    Returns list of detected regions with bbox, size, center, mean_contrast.
    """
    min_size_px = int(config.min_size_um2 / (um_per_px**2))

    bg_modes = np.array([histogram_mode(image, c) for c in range(3)])

    above = image.astype(np.float32) - bg_modes[np.newaxis, np.newaxis, :]
    if config.contrast_mode == ContrastMode.ABOVE:
        mask = np.any(above > config.contrast_offset, axis=2)
    else:
        mask = np.any(above < -config.contrast_offset, axis=2)
    # Reuse above buffer for normalized contrast (used by subsegment + analyze)
    norm_contrast = above
    norm_contrast /= np.maximum(bg_modes[np.newaxis, np.newaxis, :], 1.0)

    # Precompute gradient magnitude on G channel (used by analyze)
    gray_g = image[:, :, 1].astype(np.float32)
    sx = cv2.Sobel(gray_g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sx * sx + sy * sy)

    ks = config.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    labels, n_labels = ndimage.label(mask_clean)
    component_sizes = ndimage.sum(mask_clean, labels, range(1, n_labels + 1))
    component_slices = ndimage.find_objects(labels)

    h, w = image.shape[:2]
    detections = []

    for i in range(n_labels):
        if component_slices[i] is None or component_sizes[i] < min_size_px:
            continue

        sl = component_slices[i]
        cx = (sl[1].start + sl[1].stop) / 2
        cy = (sl[0].start + sl[0].stop) / 2

        if cx < config.edge_margin_px or cx > w - config.edge_margin_px:
            continue
        if cy < config.edge_margin_px or cy > h - config.edge_margin_px:
            continue

        component = labels == (i + 1)

        # Try contrast-based sub-segmentation for large blobs
        sub_components = _subsegment_by_contrast(image, component, bg_modes, min_size_px, norm_contrast)

        for sub_comp in sub_components:
            det = _analyze_component(image, sub_comp, bg_modes, norm_contrast, grad_mag, config)
            # Re-check edge margin for sub-components
            scx, scy = det["center"]
            if scx < config.edge_margin_px or scx > w - config.edge_margin_px:
                continue
            if scy < config.edge_margin_px or scy > h - config.edge_margin_px:
                continue
            detections.append(det)

    detections.sort(key=lambda d: d["size_px"], reverse=True)
    classify_detections(detections, config, perim_ratio_thresh=perim_ratio_thresh)
    return detections


def score_detections(detections: list[dict], config: DetectorConfig) -> None:
    """Compute tier and score for detections in place.

    Reads existing keys (perim_ratio, cal_dist, contrast_rgb, size_px).
    Adds keys: tier, score.
    """
    for det in detections:
        det["tier"], det["score"] = config.score_detection(det)


def classify_detections(
    detections: list[dict],
    config: DetectorConfig,
    perim_ratio_thresh: float = 0.0,
) -> None:
    """Classify detections and compute ranking scores in place.

    Adds keys: classification, tier, score.
    """
    if not detections:
        return

    # Tape classification
    for det in detections:
        if perim_ratio_thresh > 0 and det.get("perim_ratio", 0) >= perim_ratio_thresh:
            det["classification"] = "tape"
        else:
            det["classification"] = None

    score_detections(detections, config)


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


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    um_per_px: float = 0.0,
    draw_bbox: bool = True,
    draw_hull: bool = False,
    draw_contour: bool = True,
) -> np.ndarray:
    """Draw detection overlays on image. Returns a copy."""
    vis = image.copy()
    for i, d in enumerate(detections):
        s = d["size_px"]
        c = d.get("mean_contrast", 0)
        is_tape = d.get("classification") == "tape"
        if is_tape:
            color = (128, 128, 128)
            thickness = 1
        else:
            color = (0, 255, 0) if s > 1000 else (0, 255, 255) if s > 500 else (0, 0, 255)
            thickness = 2 if s > 1000 else 1

        bx, by, bw, bh = d["bbox"]
        if draw_bbox:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, thickness)

        hull = d.get("hull")
        if draw_hull and hull and len(hull) >= 3:
            pts = np.array(hull, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        contour = d.get("contour")
        if draw_contour and contour and len(contour) >= 3:
            pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

        label = f"#{i} {s}px c={c:.0f}"
        cv2.putText(vis, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, label, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if um_per_px > 0:
        draw_scale_bar(vis, um_per_px)
    return vis


def save_plot(
    image: np.ndarray,
    detections: list[dict],
    title: str,
    output_path: Path,
    config: DetectorConfig,
):
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
    if config.contrast_mode == ContrastMode.ABOVE:
        mask = np.any(above > config.contrast_offset, axis=2)
        mode_label = f"bg_mode + {config.contrast_offset}"
    else:
        mask = np.any(above < -config.contrast_offset, axis=2)
        mode_label = f"bg_mode - {config.contrast_offset}"
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"Threshold mask ({mode_label})\nbg_modes BGR: {bg_modes.astype(int)}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment flakes via flatfield + contrast threshold")
    parser.add_argument("input", type=Path, help="Frame image")
    parser.add_argument("--flatfield", type=Path, default=None, help="Flatfield .npy file")
    parser.add_argument("--material", default="hbn", choices=["hbn", "graphene"], help="Material preset")
    parser.add_argument("--contrast-offset", type=float, default=None, help="Override contrast offset from preset")
    parser.add_argument("--min-size-um", type=float, default=None, help="Override min detection area (µm²)")
    parser.add_argument("--edge-margin", type=int, default=None, help="Override edge margin (px)")
    parser.add_argument(
        "--dark-frac-cutoff",
        type=float,
        default=0.05,
        help="Skip frame if dark pixel fraction exceeds this (edge/off-chip filter)",
    )
    parser.add_argument(
        "--perim-ratio",
        type=float,
        default=0.0,
        help="Classify detections with perim_ratio >= this as tape (0 to disable)",
    )
    parser.add_argument(
        "--perim-ratio-kill",
        type=float,
        default=0.0,
        help="Remove detections with perim_ratio >= this entirely (0 to disable)",
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

    config = DetectorConfig.from_material(args.material)
    overrides = {}
    if args.contrast_offset is not None:
        overrides["contrast_offset"] = args.contrast_offset
    if args.min_size_um is not None:
        overrides["min_size_um2"] = args.min_size_um
    if args.edge_margin is not None:
        overrides["edge_margin_px"] = args.edge_margin
    if overrides:
        config = replace(config, **overrides)

    raw = cv2.imread(args.input)
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
        dets = segment_frame(corrected, config, args.pixel_size, args.perim_ratio)
        if args.perim_ratio_kill > 0:
            before = len(dets)
            dets = [d for d in dets if d["perim_ratio"] < args.perim_ratio_kill]
            killed = before - len(dets)
        else:
            killed = 0
        print(f"{args.input.name}: {len(dets)} detections (dark_frac={dark_frac:.3f}, killed={killed})")
        for i, d in enumerate(dets):
            R, G = d["contrast_rgb"][0], d["contrast_rgb"][1]
            print(
                f"  #{i}: {d['size_px']}px  R={R:+.2f} G={G:+.2f}  "
                f"grad={d['grad_energy']:.1f}  entropy={d['g_entropy']:.2f}  "
                f"pr={d['perim_ratio']:.2f}  r_std={d['r_std']:.3f}"
            )

    # Output image — draw on raw to preserve true colors
    out = args.output or args.input.with_suffix(".seg.jpg")
    vis = draw_detections(raw, dets, um_per_px=args.pixel_size)
    cv2.imwrite(out, vis)
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
            "material": args.material,
            "contrast_offset": config.contrast_offset,
            "min_size_um2": config.min_size_um2,
            "min_size_px": int(config.min_size_um2 / (args.pixel_size**2)),
            "edge_margin_px": config.edge_margin_px,
            "dark_frac_cutoff": args.dark_frac_cutoff,
            "perim_ratio_thresh": args.perim_ratio,
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
        save_plot(corrected, dets, args.input.name, plot_path, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
