"""Apply brightness/contrast/saturation adjustments to microscope images.

Usage:
    python scripts/enhance_image.py input.png -o output.jpg
    python scripts/enhance_image.py input.png -o output.jpg --brightness 2.75 --contrast 4.5
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


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale all pixel values by factor, clipped to 0-255."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply contrast: (pixel - mean) * factor + mean, per channel, clipped to 0-255."""
    img_f = image.astype(np.float32)
    for c in range(img_f.shape[2]):
        ch = img_f[:, :, c]
        mean = ch.mean()
        img_f[:, :, c] = (ch - mean) * factor + mean
    return np.clip(img_f, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    """Multiply S channel in HSV by factor, clipped."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main():
    parser = argparse.ArgumentParser(description="Enhance microscope images (brightness/contrast/saturation)")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness multiplier (default: 1.0)")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier (default: 1.0)")
    parser.add_argument("--saturation", type=float, default=1.0, help="Saturation multiplier (default: 1.0)")

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

    image = cv2.imread(str(args.input))
    if image is None:
        print(f"Failed to read {args.input}")
        return 1

    h, w = image.shape[:2]
    print(f"Input: {args.input} ({w}x{h})")

    # Apply order: crop → brightness → contrast → saturation
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

    if args.brightness != 1.0:
        image = adjust_brightness(image, args.brightness)
        print(f"Brightness: {args.brightness}")

    if args.contrast != 1.0:
        image = adjust_contrast(image, args.contrast)
        print(f"Contrast: {args.contrast}")

    if args.saturation != 1.0:
        image = adjust_saturation(image, args.saturation)
        print(f"Saturation: {args.saturation}")

    cv2.imwrite(str(args.output), image)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
