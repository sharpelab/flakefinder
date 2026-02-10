#!/usr/bin/env python3
"""Simple image mosaic utility.

Tiles input images into a grid, maintaining aspect ratio.

Usage:
    python mosaic_util.py -o output.jpg image1.jpg image2.jpg image3.jpg
    python mosaic_util.py -o output.jpg --glob 'scans/chips/*.jpg'
    python mosaic_util.py -o output.jpg --glob 'scans/chips/*.jpg' --rows 2
    python mosaic_util.py -o output.jpg --glob 'scans/chips/*.jpg' --labels --label-size 16
"""

import argparse
import glob
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont


def _parse_color(s):
    """Parse 'R,G,B' or 'R,G,B,A' string into a tuple."""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(f"Expected R,G,B or R,G,B,A, got: {s}")
    return tuple(parts)


def _get_font(size):
    """Try to load a monospace TTF font, fall back to default."""
    for name in ["DejaVuSansMono.ttf", "DejaVuSans.ttf", "LiberationMono-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_mosaic(
    image_paths,
    rows,
    cols,
    max_dim,
    margin,
    bg_color,
    labels=False,
    label_size=None,
    label_color=(255, 255, 255),
    label_bg=None,
):
    """Tile images into a grid mosaic.

    All thumbnails use the aspect ratio of the first image.
    Returns the composited PIL Image.
    """
    if not image_paths:
        print("No images to mosaic.")
        sys.exit(1)

    n = len(image_paths)

    # Determine grid layout
    if cols is None:
        cols = math.ceil(n / rows)
    total_slots = rows * cols

    # Load first image to get aspect ratio
    first_img = Image.open(image_paths[0])
    aspect = first_img.width / first_img.height

    # Compute thumbnail size to fit within max_dim
    # canvas_width = cols * thumb_w + (cols - 1) * margin
    # canvas_height = rows * thumb_h + (rows - 1) * margin
    # thumb_h = thumb_w / aspect
    # Solve for thumb_w from width constraint:
    thumb_w_from_width = (max_dim - (cols - 1) * margin) / cols
    thumb_w_from_width / aspect

    # Solve for thumb_h from height constraint:
    thumb_h_from_height = (max_dim - (rows - 1) * margin) / rows
    thumb_w_from_height = thumb_h_from_height * aspect

    # Use whichever is more constraining
    if thumb_w_from_width <= thumb_w_from_height:
        thumb_w = int(thumb_w_from_width)
        thumb_h = int(thumb_w / aspect)
    else:
        thumb_h = int(thumb_h_from_height)
        thumb_w = int(thumb_h * aspect)

    if thumb_w < 1 or thumb_h < 1:
        print("Error: too many images or margin too large for max-dim.")
        sys.exit(1)

    # Font setup: label_size is in points. Default ~3% of thumb height, clamped.
    if labels:
        if label_size is None:
            label_size = max(10, min(36, int(thumb_h * 0.03)))
        font = _get_font(label_size)

    canvas_w = cols * thumb_w + (cols - 1) * margin
    canvas_h = rows * thumb_h + (rows - 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for i, path in enumerate(image_paths[:total_slots]):
        row = i // cols
        col = i % cols
        x = col * (thumb_w + margin)
        y = row * (thumb_h + margin)

        img = Image.open(path)
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(img, (x, y))

        if labels:
            label = os.path.basename(path)
            pad = 2
            tx = x + pad
            ty = y + thumb_h - label_size - pad
            if label_bg is not None:
                bbox = draw.textbbox((tx, ty), label, font=font)
                draw.rectangle(
                    (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                    fill=label_bg,
                )
            draw.text((tx, ty), label, fill=label_color, font=font)

    if n > total_slots:
        print(f"Warning: {n - total_slots} images did not fit in {rows}x{cols} grid.")

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Tile images into a grid mosaic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", nargs="*", help="Input image files")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern for input images")
    parser.add_argument("--max-dim", type=int, default=5000, help="Maximum canvas dimension in pixels")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows")
    parser.add_argument("--cols", type=int, default=None, help="Max columns (default: fit all images)")
    parser.add_argument("--margin", type=int, default=2, help="Gap between images in pixels")
    parser.add_argument("--bg-color", type=_parse_color, default=(30, 30, 30), help="Canvas background color as R,G,B")
    parser.add_argument("--labels", action="store_true", help="Label each image with its filename")
    parser.add_argument(
        "--label-size", type=int, default=None, help="Label font size in points (default: auto ~3%% of thumb height)"
    )
    parser.add_argument("--label-color", type=_parse_color, default=(255, 255, 255), help="Label text color as R,G,B")
    parser.add_argument(
        "--label-bg", type=_parse_color, default=None, help="Label background color as R,G,B or R,G,B,A (default: none)"
    )
    args = parser.parse_args()

    # Collect image paths
    paths = list(args.images)
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    if not paths:
        parser.error("No input images. Provide files as arguments or use --glob.")

    print(f"Mosaic: {len(paths)} images, {args.rows} row(s), max {args.max_dim}px")

    canvas = make_mosaic(
        paths,
        args.rows,
        args.cols,
        args.max_dim,
        args.margin,
        args.bg_color,
        labels=args.labels,
        label_size=args.label_size,
        label_color=args.label_color,
        label_bg=args.label_bg,
    )
    canvas.save(args.output, quality=95)
    print(f"Saved {canvas.width}x{canvas.height} -> {args.output}")


if __name__ == "__main__":
    main()
