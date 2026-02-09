"""Detect flakes in scan frames using MaskTerial inference server."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import requests


def check_server(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url}/status", timeout=5)
        if r.status_code == 200 and "Ready" in r.text:
            return True
        print(f"Server not ready: {r.text}")
        return False
    except requests.ConnectionError:
        print(f"Cannot connect to MaskTerial server at {server_url}")
        print("Start it with:")
        print("  cd ~/code/MaskTerial && uvicorn server:app --host 0.0.0.0 --port 8000")
        return False


def predict_frame(server_url: str, frame_path: Path, seg_model: str, size_threshold: int) -> list[dict]:
    with open(frame_path, "rb") as f:
        r = requests.post(
            f"{server_url}/predict",
            files={"files": (frame_path.name, f, "image/jpeg")},
            data={
                "segmentation_model": seg_model,
                "size_threshold": size_threshold,
                "return_bbox": True,
            },
            timeout=30,
        )
    r.raise_for_status()
    return r.json()


def compute_stage_coords(flake: dict, frame: dict, um_per_px: float, frame_w: int, frame_h: int):
    cx, cy = flake["center"]
    frame_x_center = (frame["x_start"] + frame["x_end"]) / 2
    stage_x = frame_x_center + (cx - frame_w / 2) * um_per_px
    stage_y = frame["y_um"] + (cy - frame_h / 2) * um_per_px
    return stage_x, stage_y


def make_flake_map(flakes: list[dict], scan_meta: dict, output_path: Path):
    if not flakes:
        print("No flakes to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Chip bounding box if available
    if "chip_info" in scan_meta and "bbox_stage_um" in scan_meta["chip_info"]:
        bb = scan_meta["chip_info"]["bbox_stage_um"]
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (bb["x_min"], bb["y_min"]),
            bb["x_max"] - bb["x_min"],
            bb["y_max"] - bb["y_min"],
            linewidth=1,
            edgecolor="gray",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    # Color by thickness
    thicknesses = sorted(set(f["thickness"] for f in flakes))
    cmap = plt.cm.tab10
    color_map = {t: cmap(i / max(len(thicknesses), 1)) for i, t in enumerate(thicknesses)}

    for t in thicknesses:
        subset = [f for f in flakes if f["thickness"] == t]
        xs = [f["stage_x_um"] for f in subset]
        ys = [f["stage_y_um"] for f in subset]
        sizes = [max(f["size_px"] / 50, 5) for f in subset]
        ax.scatter(xs, ys, s=sizes, c=[color_map[t]], label=f"{t}L ({len(subset)})", alpha=0.7)

    ax.set_xlabel("Stage X (µm)")
    ax.set_ylabel("Stage Y (µm)")
    ax.set_title(f"Detected Flakes ({len(flakes)} total)")
    ax.legend(title="Thickness")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved flake map: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect flakes in scan frames using MaskTerial")
    parser.add_argument("scan_dir", type=Path, help="Path to scan directory")
    parser.add_argument("--server", default="http://localhost:8000", help="MaskTerial server URL")
    parser.add_argument("--seg-model", default="M2F-GrapheneH", help="Segmentation model (e.g. M2F-GrapheneH)")
    parser.add_argument("--size-threshold", type=int, default=200, help="Min flake size in pixels")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: <scan_dir>/detections/)")
    parser.add_argument("--start", type=int, default=0, help="Start at frame index N (for testing)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N frames (for testing)")
    args = parser.parse_args()

    # Load scan metadata
    meta_path = args.scan_dir / "scan_meta.json"
    if not meta_path.exists():
        print(f"No scan_meta.json in {args.scan_dir}")
        return 1

    with open(meta_path) as f:
        scan_meta = json.load(f)

    um_per_px = scan_meta["optics"]["sample_pixel_x_um"]
    frame_w = scan_meta["camera"]["frame_width_px"]
    frame_h = scan_meta["camera"]["frame_height_px"]
    frames = scan_meta["frames"]

    # Check server
    if not check_server(args.server):
        return 1

    # Resolve frame files
    all_frame_files = sorted(args.scan_dir.glob("frame_*.jpg"))
    if not all_frame_files:
        print(f"No frame_*.jpg files in {args.scan_dir}")
        return 1

    if args.start:
        all_frame_files = all_frame_files[args.start :]
    if args.limit:
        all_frame_files = all_frame_files[: args.limit]

    total = len(all_frame_files)
    print(f"Processing {total} frames ({um_per_px:.4f} µm/px, {frame_w}x{frame_h})")

    # Output dir
    out_dir = args.output or (args.scan_dir / "detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process frames
    all_flakes = []
    t_start = time.monotonic()

    for i, frame_path in enumerate(all_frame_files):
        frame_idx = int(frame_path.stem.split("_")[1])

        if frame_idx >= len(frames):
            print(f"  Warning: frame {frame_idx} has no metadata entry, skipping")
            continue

        frame_meta = frames[frame_idx]

        try:
            detections = predict_frame(args.server, frame_path, args.seg_model, args.size_threshold)
        except Exception as e:
            print(f"  Error on {frame_path.name}: {e}")
            continue

        for det in detections:
            stage_x, stage_y = compute_stage_coords(det, frame_meta, um_per_px, frame_w, frame_h)
            all_flakes.append(
                {
                    "frame_idx": frame_idx,
                    "frame_file": frame_path.name,
                    "stage_x_um": round(stage_x, 2),
                    "stage_y_um": round(stage_y, 2),
                    "thickness": det["thickness"],
                    "size_px": det["size"],
                    "size_um2": round(det["size"] * um_per_px**2, 2),
                    "center_px": det["center"],
                    "max_sidelength_px": det["max_sidelength"],
                    "min_sidelength_px": det["min_sidelength"],
                    "aspect_ratio": det["aspect_ratio"],
                    "false_positive_probability": det["false_positive_probability"],
                    "entropy": det["entropy"],
                    "bbox": det.get("bbox"),  # [x, y, w, h] in pixels
                }
            )

        elapsed = time.monotonic() - t_start
        per_frame = elapsed / (i + 1)
        eta = per_frame * (total - i - 1)
        n_flakes = len(all_flakes)
        n_this = len(detections)
        print(
            f"  [{i + 1}/{total}] {frame_path.name}: {n_this} flakes ({n_flakes} total) | {per_frame:.2f}s/frame | ETA {eta:.0f}s"  # noqa: E501
        )

    # Write results
    result = {
        "scan_dir": str(args.scan_dir),
        "timestamp": datetime.now().isoformat(),
        "um_per_px": um_per_px,
        "seg_model": args.seg_model,
        "size_threshold": args.size_threshold,
        "total_frames": total,
        "total_flakes": len(all_flakes),
        "flakes": all_flakes,
    }

    json_path = out_dir / "flakes.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {len(all_flakes)} flakes to {json_path}")

    # Flake map
    make_flake_map(all_flakes, scan_meta, out_dir / "flake_map.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
