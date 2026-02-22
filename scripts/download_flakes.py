"""Download flakes (images + metadata) from flakes.sharpelab.science."""

import argparse
import json
import sys
from pathlib import Path

from flakefinder.flakes_api import BASE_URL, api_get, download_image

# Images to attempt downloading for each flake
FLAKE_IMAGES = ["eval_img.jpg", "raw_img.png", "flake_mask.png", "overview_marked.jpg"]
# Magnification images use the pattern {mag}x.png — derived from flake metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_id", type=int, help="Scan ID to download from")
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Target directory (default: downloads/flakes_scan<id>)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all flakes, not just favorites",
    )
    args = parser.parse_args()

    # Fetch scan metadata
    print(f"Fetching scan {args.scan_id} metadata...")
    scans = api_get("scans", {"scan_id": args.scan_id})
    if not scans:
        print(f"Error: scan {args.scan_id} not found", file=sys.stderr)
        sys.exit(1)
    scan_meta = scans[0]
    scan_name = scan_meta["scan_name"]
    print(f"  Scan: {scan_name} (user: {scan_meta['scan_user']})")

    # Fetch flakes
    params: dict = {"scan_id": args.scan_id}
    if not args.all:
        params["flake_favorite"] = 1
    label = "all" if args.all else "favorited"
    print(f"Fetching {label} flakes...")
    flakes = api_get("flakes", params)
    if not flakes:
        print(f"No {label} flakes found.", file=sys.stderr)
        sys.exit(1)
    n_fav = sum(1 for f in flakes if f.get("flake_favorite"))
    print(f"  Found {len(flakes)} flakes ({n_fav} favorites)")

    # Resolve target directory
    if args.target:
        target = Path(args.target)
    else:
        target = Path("downloads") / f"flakes_scan{args.scan_id}"
    target.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to: {target}")

    # Save metadata
    meta_path = target / "flakes_meta.json"
    meta_path.write_text(json.dumps({"scan": scan_meta, "flakes": flakes}, indent=2))
    print(f"  Wrote {meta_path}")

    # Download images
    total_bytes = 0
    total_files = 0

    skipped = 0

    for i, flake in enumerate(flakes):
        flake_path = flake["flake_path"]  # e.g. "SF118_ABCD_E13-16/Chip_4/Flake_7"
        flake_dir = target / flake_path
        flake_id = flake["flake_id"]

        # Skip if already downloaded (meta.json exists)
        if (flake_dir / "meta.json").exists():
            skipped += 1
            continue

        fav_marker = " *" if flake.get("flake_favorite") else ""
        print(f"  [{i + 1}/{len(flakes)}] Flake {flake_id} ({flake_path}){fav_marker}")

        # Save per-flake metadata
        flake_dir.mkdir(parents=True, exist_ok=True)
        (flake_dir / "meta.json").write_text(json.dumps(flake, indent=2))

        # Standard images
        for filename in FLAKE_IMAGES:
            url = f"{BASE_URL}/images/{flake_path}/{filename}"
            dest = flake_dir / filename
            size = download_image(url, dest)
            if size is not None:
                total_bytes += size
                total_files += 1

        # Magnification images
        for mag in flake.get("flake_available_magnifications", []):
            mag_str = f"{mag:g}x"  # e.g. "10x", "50x"
            filename = f"{mag_str}.png"
            url = f"{BASE_URL}/images/{flake_path}/{filename}"
            dest = flake_dir / filename
            size = download_image(url, dest)
            if size is not None:
                total_bytes += size
                total_files += 1

    if skipped:
        print(f"  ({skipped} flakes already downloaded, skipped)")
    print(f"\nDone: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB total")


if __name__ == "__main__":
    main()
