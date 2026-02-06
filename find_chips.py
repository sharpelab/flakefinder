"""Detect chips in stitched microscope images using Otsu thresholding."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np


def load_stitch_meta(image_path: Path) -> dict:
    """Load stitch metadata from companion JSON file."""
    meta_path = image_path.with_name(image_path.stem + "_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Stitch metadata not found: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def px_to_stage(px_x: float, px_y: float, meta: dict) -> tuple[float, float]:
    """Convert pixel coordinates to stage coordinates."""
    bounds = meta["stage_bounds_um"]
    scale = meta["scale_um_per_px"]
    stage_x = bounds["x_min"] + px_x * scale
    stage_y = bounds["y_min"] + px_y * scale
    return stage_x, stage_y


def touches_edge(contour: np.ndarray, img_w: int, img_h: int, margin: int = 1) -> str | None:
    """Check if contour touches image edge. Returns edge name or None."""
    x, y, w, h = cv2.boundingRect(contour)

    edges = []
    if x <= margin:
        edges.append("left")
    if y <= margin:
        edges.append("top")
    if x + w >= img_w - margin:
        edges.append("right")
    if y + h >= img_h - margin:
        edges.append("bottom")

    return ",".join(edges) if edges else None


def extract_chip_geometry(contour: np.ndarray, meta: dict) -> dict:
    """Extract geometric properties from contour in both pixel and stage coords."""
    # Bounding box in pixels
    x, y, w, h = cv2.boundingRect(contour)
    bbox_px = [int(x), int(y), int(w), int(h)]

    # Bounding box in stage coordinates
    x_min_stage, y_min_stage = px_to_stage(x, y, meta)
    x_max_stage, y_max_stage = px_to_stage(x + w, y + h, meta)
    bbox_stage_um = {
        "x_min": x_min_stage,
        "y_min": y_min_stage,
        "x_max": x_max_stage,
        "y_max": y_max_stage,
    }

    # Centroid from moments
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx_px = M["m10"] / M["m00"]
        cy_px = M["m01"] / M["m00"]
    else:
        cx_px = x + w / 2
        cy_px = y + h / 2
    centroid_stage = px_to_stage(cx_px, cy_px, meta)

    # Convex hull in stage coordinates
    hull = cv2.convexHull(contour)
    hull_stage = []
    for point in hull:
        px_x, px_y = point[0]
        stage_x, stage_y = px_to_stage(float(px_x), float(px_y), meta)
        hull_stage.append([stage_x, stage_y])

    # Area
    area_px = cv2.contourArea(contour)
    scale = meta["scale_um_per_px"]
    area_um2 = area_px * (scale ** 2)

    return {
        "bbox_px": bbox_px,
        "bbox_stage_um": bbox_stage_um,
        "centroid_stage_um": list(centroid_stage),
        "convex_hull_stage_um": hull_stage,
        "area_um2": area_um2,
    }


def find_chips(
    image_path: Path,
    min_area_um2: float = 1e6,
    morph_kernel_um: float = 50,
) -> dict:
    """
    Detect chips in a stitched microscope image.

    Args:
        image_path: Path to stitched image
        min_area_um2: Minimum chip area in µm² (default 1mm²)
        morph_kernel_um: Morphological kernel size in µm

    Returns:
        Detection results dict
    """
    start_time = time.perf_counter()
    # Load image and metadata
    meta = load_stitch_meta(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_h, img_w = img.shape[:2]
    scale = meta["scale_um_per_px"]

    # Convert to grayscale (luminance via LAB for better perceptual uniformity)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = lab[:, :, 0]  # L channel

    # Apply Otsu's threshold
    # Chips are brighter than dark substrate, so we want bright regions
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel_px = max(3, int(morph_kernel_um / scale))
    if kernel_px % 2 == 0:
        kernel_px += 1  # Ensure odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))

    # Close to fill small holes, then open to remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and classify contours
    chips = []
    rejected = []
    chip_id = 0

    min_area_px = min_area_um2 / (scale ** 2)

    for contour in contours:
        area_px = cv2.contourArea(contour)
        area_um2 = area_px * (scale ** 2)

        # Check minimum area
        if area_px < min_area_px:
            continue

        # Check edge contact
        edge = touches_edge(contour, img_w, img_h)

        if edge:
            # Rejected due to edge clipping
            x, y, w, h = cv2.boundingRect(contour)
            rejected.append({
                "reason": "edge_clipped",
                "edge": edge,
                "bbox_px": [int(x), int(y), int(w), int(h)],
                "area_um2": area_um2,
            })
        else:
            # Valid chip
            geom = extract_chip_geometry(contour, meta)
            geom["id"] = chip_id
            chips.append(geom)
            chip_id += 1

    # Sort chips top-left to bottom-right
    # Bucket Y by 10mm so chips in same row sort left-to-right by X
    y_bucket_um = 10000
    chips.sort(key=lambda c: (c["centroid_stage_um"][1] // y_bucket_um, c["centroid_stage_um"][0]))
    for i, chip in enumerate(chips):
        chip["id"] = i

    # Build results
    duration_s = time.perf_counter() - start_time
    results = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(duration_s, 2),
        "source_stitch": image_path.name,
        "source_meta": image_path.stem + "_meta.json",
        "detection_params": {
            "otsu_threshold": int(threshold),
            "min_area_um2": min_area_um2,
            "morph_kernel_um": morph_kernel_um,
        },
        "chips": chips,
        "rejected": rejected,
    }

    # Detection visualization
    debug_img = img.copy()

    # Scale drawing parameters based on image size
    # Reference: 2000px image gets base values, scale proportionally
    ref_size = 2000
    img_scale = max(img_w, img_h) / ref_size

    line_thickness = max(2, int(4 * img_scale))
    bbox_thickness = max(2, int(4 * img_scale))
    font_scale = 4.5 * img_scale
    font_thickness = max(2, int(9 * img_scale))

    # Find contours again to draw them (we need the actual contour objects)
    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < min_area_px:
            continue

        edge = touches_edge(contour, img_w, img_h)
        x, y, w, h = cv2.boundingRect(contour)

        if edge:
            # Red contour for rejected (no bbox)
            cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), line_thickness)
        else:
            # Green contour, red bbox for valid
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), line_thickness)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), bbox_thickness)

    # Draw chip IDs at centroids
    for chip in chips:
        cx = chip["bbox_px"][0] + chip["bbox_px"][2] // 2
        cy = chip["bbox_px"][1] + chip["bbox_px"][3] // 2
        text = str(chip["id"])
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, font_thickness)
        cv2.putText(debug_img, text, (cx - text_w // 2, cy + text_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thickness)

    # Add scale bar
    # Choose largest round number that fits in 10-20% of image width
    img_width_um = img_w * scale
    bar_length_um = 1000  # default 1mm
    for candidate_um in [20000, 10000, 5000, 2000, 1000]:
        candidate_px = int(candidate_um / scale)
        if candidate_px <= img_w * 0.20:  # Pick largest that's ≤20% of width
            bar_length_um = candidate_um
            break
    bar_length_px = int(bar_length_um / scale)

    bar_height_px = max(5, int(10 * img_scale))
    bar_margin_px = max(30, int(60 * img_scale))
    bar_x = img_w - bar_margin_px - bar_length_px
    bar_y = img_h - bar_margin_px - bar_height_px

    # Draw scale bar with outline for visibility
    cv2.rectangle(debug_img, (bar_x - 2, bar_y - 2),
                 (bar_x + bar_length_px + 2, bar_y + bar_height_px + 2),
                 (0, 0, 0), -1)  # Black outline
    cv2.rectangle(debug_img, (bar_x, bar_y),
                 (bar_x + bar_length_px, bar_y + bar_height_px),
                 (255, 255, 255), -1)  # White bar

    # Scale bar label
    bar_label = f"{bar_length_um / 1000:.0f} mm" if bar_length_um >= 1000 else f"{bar_length_um} µm"
    label_font_scale = 1.0 * img_scale
    label_thickness = max(1, int(2 * img_scale))
    (label_w, label_h), _ = cv2.getTextSize(bar_label, cv2.FONT_HERSHEY_SIMPLEX,
                                             label_font_scale, label_thickness)
    label_x = bar_x + (bar_length_px - label_w) // 2
    label_y = bar_y - max(5, int(10 * img_scale))

    # Draw label with outline
    cv2.putText(debug_img, bar_label, (label_x, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (0, 0, 0), label_thickness + 2)
    cv2.putText(debug_img, bar_label, (label_x, label_y),
               cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), label_thickness)

    # Downscale to reasonable size for viewing
    max_debug_dim = 1500
    if max(debug_img.shape[:2]) > max_debug_dim:
        scale_factor = max_debug_dim / max(debug_img.shape[:2])
        new_w = int(debug_img.shape[1] * scale_factor)
        new_h = int(debug_img.shape[0] * scale_factor)
        debug_img = cv2.resize(debug_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    detected_path = image_path.with_name(image_path.stem + "_chips_detected.png")
    cv2.imwrite(str(detected_path), debug_img)
    print(f"Detection image saved to {detected_path} ({debug_img.shape[1]}x{debug_img.shape[0]})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect chips in stitched microscope images"
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to stitched image (PNG). Metadata JSON auto-discovered.",
    )
    parser.add_argument(
        "--min-area-um2",
        type=float,
        default=1e6,
        help="Minimum chip area in µm² (default: 1e6 = 1mm²)",
    )
    parser.add_argument(
        "--morph-kernel-um",
        type=float,
        default=50,
        help="Morphological kernel size in µm (default: 50)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    print(f"Processing {args.image}")
    results = find_chips(
        args.image,
        min_area_um2=args.min_area_um2,
        morph_kernel_um=args.morph_kernel_um,
    )

    # Save results
    out_path = args.image.with_name(args.image.stem + "_chips.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary
    print(f"\nDetection summary:")
    print(f"  Otsu threshold: {results['detection_params']['otsu_threshold']}")
    print(f"  Chips found: {len(results['chips'])}")
    print(f"  Rejected (edge-clipped): {len(results['rejected'])}")

    if results["chips"]:
        print(f"\nChips:")
        for chip in results["chips"]:
            cx, cy = chip["centroid_stage_um"]
            area_mm2 = chip["area_um2"] / 1e6
            print(f"  #{chip['id']}: center=({cx/1000:.1f}, {cy/1000:.1f}) mm, "
                  f"area={area_mm2:.2f} mm²")

    return 0


if __name__ == "__main__":
    exit(main())
