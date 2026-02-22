"""Download flakes (images + metadata) from flakes.sharpelab.science."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flakefinder.flakes_api import BASE_URL, api_get, download_image

# Images to attempt downloading for each flake
FLAKE_IMAGES = ["eval_img.jpg", "raw_img.png", "flake_mask.png", "overview_marked.jpg"]
# Magnification images use the pattern {mag}x.png — derived from flake metadata


def _download_flake(flake: dict, target: Path) -> tuple[int, int]:
    """Download all images for a single flake. Returns (files, bytes)."""
    flake_path = flake["flake_path"]
    flake_dir = target / flake_path

    flake_dir.mkdir(parents=True, exist_ok=True)
    (flake_dir / "meta.json").write_text(json.dumps(flake, indent=2))

    total_files = 0
    total_bytes = 0

    for filename in FLAKE_IMAGES:
        url = f"{BASE_URL}/images/{flake_path}/{filename}"
        dest = flake_dir / filename
        size = download_image(url, dest)
        if size is not None:
            total_bytes += size
            total_files += 1

    for mag in flake.get("flake_available_magnifications", []):
        mag_str = f"{mag:g}x"
        filename = f"{mag_str}.png"
        url = f"{BASE_URL}/images/{flake_path}/{filename}"
        dest = flake_dir / filename
        size = download_image(url, dest)
        if size is not None:
            total_bytes += size
            total_files += 1

    return total_files, total_bytes


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
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Parallel download workers (default: 8)",
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

    # Filter out already-downloaded flakes
    todo = []
    skipped = 0
    for flake in flakes:
        flake_dir = target / flake["flake_path"]
        if (flake_dir / "meta.json").exists():
            skipped += 1
        else:
            todo.append(flake)

    if skipped:
        print(f"  ({skipped} flakes already downloaded, skipping)")

    if not todo:
        print("Nothing to download.")
        return

    # Download in parallel
    total_bytes = 0
    total_files = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_download_flake, f, target): f for f in todo}
        for future in as_completed(futures):
            flake = futures[future]
            files, nbytes = future.result()
            total_files += files
            total_bytes += nbytes
            done += 1
            fav = " *" if flake.get("flake_favorite") else ""
            print(f"  [{done}/{len(todo)}] {flake['flake_path']}{fav}")

    print(f"\nDone: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB total")


if __name__ == "__main__":
    main()
