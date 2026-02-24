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
import re
import sys

from PIL import Image, ImageDraw, ImageFont

from flakefinder.scan_utils import parse_area_rect_i
from flakefinder.types import AreaRectI


def _expand_braces(pattern):
    """Expand {a,b,c} brace patterns into multiple strings."""
    match = re.search(r"\{([^{}]+)\}", pattern)
    if not match:
        return [pattern]
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    results = []
    for alt in match.group(1).split(","):
        results.extend(_expand_braces(prefix + alt + suffix))
    return results


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


def _resolve_labels(paths, label_text=None):
    """Determine label strings for each image.

    If *label_text* is given, split by comma and validate count.
    Otherwise, if any basenames collide, use ``parent/filename``; else plain filename.
    """
    if label_text is not None:
        labels = [s.strip() for s in label_text.split(",")]
        if len(labels) != len(paths):
            print(f"Error: --label-text has {len(labels)} labels but {len(paths)} images.")
            sys.exit(1)
        return labels

    basenames = [os.path.basename(p) for p in paths]
    if len(set(basenames)) < len(basenames):
        # Duplicates exist – use parent/filename
        return [os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in paths]
    return basenames


def make_mosaic(
    image_paths,
    rows,
    cols,
    max_dim,
    margin,
    bg_color,
    labels=None,
    label_size=None,
    label_color=(255, 255, 255),
    label_bg=None,
    crop: AreaRectI | None = None,
):
    """Tile images into a grid mosaic.

    All thumbnails use the aspect ratio of the first image (after cropping).
    *labels* is an optional list of strings (one per image) to overlay.
    *crop* applies a uniform crop box to every image before thumbnailing.
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

    # Load first image to get aspect ratio (after crop)
    first_img = Image.open(image_paths[0])
    if crop is not None:
        first_img = first_img.crop((crop.x_min, crop.y_min, crop.x_max, crop.y_max))
    aspect = first_img.width / first_img.height

    # Compute thumbnail size to fit within max_dim
    # canvas_width = cols * thumb_w + (cols - 1) * margin
    # canvas_height = rows * thumb_h + (rows - 1) * margin
    # thumb_h = thumb_w / aspect
    # Solve for thumb_w from width constraint:
    thumb_w_from_width = (max_dim - (cols - 1) * margin) / cols

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

    # Cap to original image size — never upscale
    if thumb_w > first_img.width:
        thumb_w = first_img.width
        thumb_h = int(thumb_w / aspect)

    if thumb_w < 1 or thumb_h < 1:
        print("Error: too many images or margin too large for max-dim.")
        sys.exit(1)

    # Font setup: label_size is in points. Default ~3% of thumb height, clamped.
    if labels is not None:
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
        if crop is not None:
            img = img.crop((crop.x_min, crop.y_min, crop.x_max, crop.y_max))
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(img, (x, y))

        if labels is not None:
            label = labels[i]
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
    parser.add_argument(
        "--from-file", type=str, default=None, help="Read image paths from file (one per line, '-' for stdin)"
    )
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
    parser.add_argument(
        "--label-text", type=str, default=None, help="Comma-separated manual labels (overrides auto-detection)"
    )
    parser.add_argument(
        "--crop", type=parse_area_rect_i, default=None, help="Crop as x_min,x_max,y_min,y_max applied to all images"
    )
    args = parser.parse_args()

    # Collect image paths
    paths = list(args.images)
    if args.glob:
        for expanded in _expand_braces(args.glob):
            paths.extend(sorted(glob.glob(expanded)))
    if args.from_file:
        if args.from_file == "-":
            lines = sys.stdin.read().splitlines()
        else:
            with open(args.from_file) as f:
                lines = f.read().splitlines()
        paths.extend(line.strip() for line in lines if line.strip())
    if not paths:
        parser.error("No input images. Provide files as arguments or use --glob.")

    print(f"Mosaic: {len(paths)} images, {args.rows} row(s), max {args.max_dim}px")

    # Resolve labels
    resolved_labels = None
    if args.label_text is not None or args.labels:
        resolved_labels = _resolve_labels(paths, label_text=args.label_text)

    canvas = make_mosaic(
        paths,
        args.rows,
        args.cols,
        args.max_dim,
        args.margin,
        args.bg_color,
        labels=resolved_labels,
        label_size=args.label_size,
        label_color=args.label_color,
        label_bg=args.label_bg,
        crop=args.crop,
    )
    canvas.save(args.output, quality=95)
    print(f"Saved {canvas.width}x{canvas.height} -> {args.output}")


if __name__ == "__main__":
    main()
