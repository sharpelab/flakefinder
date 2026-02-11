"""Quick contrast-threshold segmentation on any image.

Usage:
    python scripts/segment_test.py IMAGE [--threshold 15] [--min-size 200] [--edge 50]
    python scripts/segment_test.py img1.jpg img2.jpg --threshold 20
    python scripts/segment_test.py IMAGE --no-plot  # just print detections
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage, stats


def segment_image(img, threshold=15, min_size=200, edge_margin=50):
    """Threshold segmentation on a raw image.

    Returns list of dicts with keys: id, size, center, bbox, contrast, mask.
    """
    arr = np.array(img, dtype=float)
    bg = np.array(
        [stats.mode(arr[:, :, c], axis=None, keepdims=False).mode for c in range(3)]
    )

    mask = np.any(arr > bg + threshold, axis=2)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=2)
    labeled, n = ndimage.label(mask)

    h, w = mask.shape
    detections = []
    for i in range(1, n + 1):
        comp = labeled == i
        size = comp.sum()
        if size < min_size:
            continue
        ys, xs = np.where(comp)
        cx, cy = float(xs.mean()), float(ys.mean())
        if cx < edge_margin or cx > w - edge_margin or cy < edge_margin or cy > h - edge_margin:
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pixels = arr[comp]
        contrast = float(np.mean(np.abs(pixels - bg) / np.maximum(bg, 1)))
        detections.append(
            {
                "id": i,
                "size": size,
                "center": (int(cx), int(cy)),
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "contrast": contrast,
                "mask": comp,
            }
        )

    detections.sort(key=lambda d: d["size"], reverse=True)
    return detections, bg.astype(int), mask


def main():
    parser = argparse.ArgumentParser(description="Contrast-threshold flake segmentation")
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s)")
    parser.add_argument("--threshold", type=float, default=15, help="BG mode + threshold (default: 15)")
    parser.add_argument("--min-size", type=int, default=200, help="Min detection size in px (default: 200)")
    parser.add_argument("--edge", type=int, default=50, help="Edge margin in px (default: 50)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot, just print")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    parser.add_argument("--top", type=int, default=15, help="Print top N detections")
    args = parser.parse_args()

    for img_path in args.images:
        img = Image.open(img_path)
        arr = np.array(img)
        dets, bg, mask = segment_image(img, args.threshold, args.min_size, args.edge)

        print(f"\n{img_path.name}: {arr.shape}, bg_mode={bg}, {len(dets)} detections")
        for i, d in enumerate(dets[: args.top]):
            print(f"  #{i}: size={d['size']:6d}px, center={d['center']}, contrast={d['contrast']:.3f}")

    if args.no_plot:
        return

    n = len(args.images)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n), squeeze=False)
    for row, img_path in enumerate(args.images):
        img = Image.open(img_path)
        arr = np.array(img)
        dets, bg, mask = segment_image(img, args.threshold, args.min_size, args.edge)

        axes[row][0].imshow(arr)
        axes[row][0].set_title(f"{img_path.name} (bg={bg})")

        axes[row][1].imshow(mask, cmap="gray")
        axes[row][1].set_title(f"Threshold mask (bg+{args.threshold})")

        axes[row][2].imshow(arr)
        for d in dets:
            x, y, w, h = d["bbox"]
            color = "lime" if d["size"] > 1000 else "yellow" if d["size"] > 500 else "cyan"
            lw = 2 if d["size"] > 1000 else 1
            axes[row][2].add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor="none"))
        axes[row][2].set_title(f"{len(dets)} detections")

    fig.tight_layout()
    out = args.output or Path(f"downloads/segment_test_{args.images[0].stem}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
