"""Apply brightness/contrast/saturation adjustments to microscope images.

Matches CSS filter behavior: no intermediate clamping, CSS saturate() color
matrix, contrast pivots on 128.

Usage:
    python scripts/enhance_image.py input.png -o output.jpg
    python scripts/enhance_image.py input.png -o output.jpg --brightness 1.46 --contrast 4.47
    python scripts/enhance_image.py input.png -o output.jpg --saturation 2.0 --crop-preset center
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def crop_preset(image: np.ndarray, preset: str) -> np.ndarray:
    """Crop to 50% of each dimension at the given anchor position."""
    h, w = image.shape[:2]
    ch, cw = h // 2, w // 2
    origins = {
        "center": (h // 4, w // 4),
        "top-left": (0, 0),
        "top-right": (0, w - cw),
        "bottom-left": (h - ch, 0),
        "bottom-right": (h - ch, w - cw),
    }
    y0, x0 = origins[preset]
    return image[y0 : y0 + ch, x0 : x0 + cw]


def crop_rect(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop to pixel rectangle [x1, y1, x2, y2]."""
    return image[y1:y2, x1:x2]


def apply_brightness(img_f: np.ndarray, factor: float) -> np.ndarray:
    """Scale all pixel values by factor (float32 in/out, no clamping)."""
    return img_f * factor


def apply_contrast(img_f: np.ndarray, factor: float) -> np.ndarray:
    """CSS-style contrast: (pixel - 128) * factor + 128 (float32 in/out, no clamping)."""
    return (img_f - 128) * factor + 128


def apply_saturation(img_f: np.ndarray, factor: float) -> np.ndarray:
    """CSS-style saturate(): linear color matrix in RGB (float32 in/out, no clamping).

    Input is BGR (OpenCV convention), converted to RGB for the matrix, then back.
    """
    # CSS saturate matrix coefficients (ITU-R BT.601 luma)
    s = factor
    mat = np.array(
        [
            [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
            [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
            [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
        ],
        dtype=np.float32,
    )
    # BGR → RGB, apply matrix, RGB → BGR
    rgb = img_f[:, :, ::-1]
    h, w, _ = rgb.shape
    result_rgb = (rgb.reshape(-1, 3) @ mat.T).reshape(h, w, 3)
    return result_rgb[:, :, ::-1]


def main():
    parser = argparse.ArgumentParser(description="Enhance microscope images (brightness/contrast/saturation)")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness multiplier (default: 1.0)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier (default: 1.0)")
    parser.add_argument("--saturation", type=float, default=1.0, help="Saturation multiplier (default: 1.0)")

    parser.add_argument(
        "--order",
        type=str,
        default="bsc",
        help="Apply order for brightness/contrast/saturation (default: bsc)",
    )

    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop-preset",
        choices=["center", "top-left", "top-right", "bottom-left", "bottom-right"],
        help="Crop to 50%% of each dimension at the given anchor",
    )
    crop_group.add_argument(
        "--crop-rect",
        type=str,
        help="Crop to pixel rect: x1,y1,x2,y2",
    )

    args = parser.parse_args()

    if sorted(args.order) != ["b", "c", "s"]:
        print(f"--order must be a permutation of 'bcs', got '{args.order}'")
        return 1

    image = cv2.imread(str(args.input))
    if image is None:
        print(f"Failed to read {args.input}")
        return 1

    h, w = image.shape[:2]
    print(f"Input: {args.input} ({w}x{h})")

    # Crop first (always before enhancements)
    if args.crop_preset:
        image = crop_preset(image, args.crop_preset)
        print(f"Cropped ({args.crop_preset}): {image.shape[1]}x{image.shape[0]}")
    elif args.crop_rect:
        coords = [int(v) for v in args.crop_rect.split(",")]
        if len(coords) != 4:
            print("--crop-rect requires 4 comma-separated values: x1,y1,x2,y2")
            return 1
        image = crop_rect(image, *coords)
        print(f"Cropped (rect): {image.shape[1]}x{image.shape[0]}")

    # Apply enhancements in --order (float32 throughout, single clamp at end)
    ops = {
        "b": ("Brightness", args.brightness, apply_brightness),
        "c": ("Contrast", args.contrast, apply_contrast),
        "s": ("Saturation", args.saturation, apply_saturation),
    }
    img_f = image.astype(np.float32)
    for key in args.order:
        label, factor, fn = ops[key]
        if factor != 1.0:
            img_f = fn(img_f, factor)
            print(f"{label}: {factor}")

    image = np.clip(img_f, 0, 255).astype(np.uint8)

    cv2.imwrite(str(args.output), image)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
