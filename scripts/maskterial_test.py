"""Quick MaskTerial detection test with visualization.

Usage:
    python scripts/maskterial_test.py IMAGE [--model M2F-hBN_Thin] [--score 0.0] [--size 0] [--zoom x1,y1,x2,y2]

Examples:
    python scripts/maskterial_test.py downloads/maskterial_test_frame_1535_ff.jpg
    python scripts/maskterial_test.py downloads/maskterial_test_frame_1535_ff.jpg --model M2F-GrapheneH --size 500
    python scripts/maskterial_test.py downloads/maskterial_test_frame_1535_ff.jpg --zoom 1050,750,1550,1100
    python scripts/maskterial_test.py img1.jpg img2.jpg --model M2F-hBN_Thin M2F-GrapheneH
"""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import requests
from matplotlib.patches import Rectangle


def predict(image_path: Path, model: str, score_threshold: float, size_threshold: int):
    with open(image_path, "rb") as f:
        r = requests.post(
            "http://localhost:8000/predict",
            files={"files": (image_path.name, f, "image/jpeg")},
            data={
                "segmentation_model": model,
                "score_threshold": str(score_threshold),
                "size_threshold": str(size_threshold),
                "return_bbox": "true",
            },
            timeout=60,
        )
    r.raise_for_status()
    data = r.json()
    # Handle nested list (one result per file)
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data[0]
    return data


def plot_detections(ax, img, dets, title, zoom=None):
    ax.imshow(img)
    for d in dets:
        bx, by, bw, bh = d["bbox"]
        s = d["size"]
        color = "lime" if s > 1000 else "yellow" if s > 500 else "red"
        lw = 2 if s > 1000 else 1
        ax.add_patch(Rectangle((bx, by), bw, bh, linewidth=lw, edgecolor=color, facecolor="none"))
    if zoom:
        ax.set_xlim(zoom[0], zoom[2])
        ax.set_ylim(zoom[3], zoom[1])  # inverted y
    ax.set_title(title, fontsize=10)


def main():
    parser = argparse.ArgumentParser(description="Test MaskTerial detection")
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s)")
    parser.add_argument("--model", nargs="+", default=["M2F-hBN_Thin"], help="Segmentation model(s)")
    parser.add_argument("--score", type=float, default=0.0, help="Score threshold")
    parser.add_argument("--size", type=int, default=0, help="Size threshold (px)")
    parser.add_argument("--zoom", type=str, default=None, help="Zoom region: x1,y1,x2,y2")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path (default: auto)")
    parser.add_argument("--top", type=int, default=15, help="Print top N detections")
    args = parser.parse_args()

    zoom = None
    if args.zoom:
        zoom = [int(x) for x in args.zoom.split(",")]

    n_images = len(args.images)
    n_models = len(args.model)
    ncols = n_models
    nrows = n_images

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)

    for row, img_path in enumerate(args.images):
        img = mpimg.imread(img_path)
        for col, model in enumerate(args.model):
            dets = predict(img_path, model, args.score, args.size)
            dets_sorted = sorted(dets, key=lambda d: d["size"], reverse=True)

            print(f"\n{img_path.name} + {model}: {len(dets)} detections (score>{args.score}, size>{args.size})")
            for i, d in enumerate(dets_sorted[: args.top]):
                print(
                    f"  #{i:2d}: size={d['size']:6d}px, center={d['center']}, "
                    f"bbox={d['bbox']}, fp={d['false_positive_probability']:.3f}"
                )
            if len(dets) > args.top:
                print(f"  ... and {len(dets) - args.top} more")

            title = f"{img_path.stem} + {model}\n{len(dets)} dets, score>{args.score}, size>{args.size}"
            plot_detections(axes[row][col], img, dets, title, zoom)

    fig.tight_layout()
    out = args.output or Path(f"downloads/maskterial_test_{args.images[0].stem}.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
