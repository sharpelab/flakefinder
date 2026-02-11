"""Print pixel statistics for one or more images. Optionally compare to a reference."""

import argparse
import sys

import numpy as np
from PIL import Image


def stats(path: str) -> dict:
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    return {
        "path": path,
        "size": f"{w}x{h}",
        "mean": img.mean(),
        "min": int(img.min()),
        "max": int(img.max()),
        "rgb": (img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()),
        "clipped_pct": (img > 254).mean() * 100,
    }


def print_stats(s: dict) -> None:
    r, g, b = s["rgb"]
    print(f"  {s['path']}")
    print(f"    Size: {s['size']}")
    print(f"    Mean: {s['mean']:.0f}  Min: {s['min']}  Max: {s['max']}")
    print(f"    Mean RGB: {r:.0f}, {g:.0f}, {b:.0f}")
    print(f"    Clipped (>254): {s['clipped_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image files to analyze")
    parser.add_argument("--ref", help="Reference image to compare against")
    args = parser.parse_args()

    for path in args.images:
        try:
            print_stats(stats(path))
        except Exception as e:
            print(f"  {path}: ERROR - {e}", file=sys.stderr)

    if args.ref:
        print()
        print("Reference:")
        try:
            print_stats(stats(args.ref))
        except Exception as e:
            print(f"  {args.ref}: ERROR - {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
