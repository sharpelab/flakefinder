"""Build a flatfield from median of captures at bare-substrate positions.

Runs on the microscope. Captures at each position, validates brightness
consistency and uniformity, builds median flatfield from good frames.

To recalibrate all objectives, run once per objective with AF beforehand:
    sls autofocus --objective-mag X --x CX --y CY --z Z_REF -q
    sls build-flatfield --objective-mag X --chips-meta ... --chip N -o calibration/flatfield_Xx_bin3.npy

Usage:
    # Auto-generate grid from chip bbox (Z from current position)
    uv run python scripts/build_flatfield.py --objective-mag 2.5 \
        --chips-meta scans/overview_5x_stitch_chips.json \
        --chip 1 -o calibration/flatfield_2.5x_bin3.npy

    # Explicit Z
    uv run python scripts/build_flatfield.py --objective-mag 2.5 \
        --chips-meta scans/chips.json --chip 1 --z 24591 \
        -o calibration/ff.npy

    # Dry run — show positions without capturing
    uv run python scripts/build_flatfield.py --objective-mag 2.5 \
        --chips-meta scans/chips.json --chip 1 \
        -o calibration/ff.npy --dry-run

    # Manual positions
    uv run python scripts/build_flatfield.py --objective-mag 2.5 \
        --positions "57000,8000 60000,13000" \
        -o calibration/flatfield_2.5x_bin3.npy
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from flakefinder.data_utils import compute_frame_size_um, require_microscope_description
from flakefinder.leica import Microscope, wait_all
from flakefinder.scan_utils import parse_white_balance


def generate_chip_grid(
    chips_meta_path: str,
    chip_index: int,
    fov_um: tuple[float, float],
) -> list[tuple[float, float]]:
    """Generate a grid of positions inside a chip bbox, inset by 1.5 FOV.

    Caps at 4x3 (or 3x4 for tall chips) = 12 points max.
    """
    with open(chips_meta_path) as f:
        data = json.load(f)
    chip = data["chips"][chip_index]
    bb = chip["bbox_stage_um"]

    fov_w, fov_h = fov_um
    x_min = bb["x_min"] + 1.5 * fov_w
    x_max = bb["x_max"] - 1.5 * fov_w
    y_min = bb["y_min"] + 1.5 * fov_h
    y_max = bb["y_max"] - 1.5 * fov_h

    if x_max <= x_min or y_max <= y_min:
        cx, cy = chip["centroid_stage_um"]
        print(f"Warning: chip {chip_index} too small for 1.5-FOV inset, using centroid only")
        return [(cx, cy)]

    aspect = (x_max - x_min) / (y_max - y_min)
    if aspect >= 1:
        n_x = min(4, max(2, int((x_max - x_min) / fov_w) + 1))
        n_y = min(3, max(2, int((y_max - y_min) / fov_h) + 1))
    else:
        n_x = min(3, max(2, int((x_max - x_min) / fov_w) + 1))
        n_y = min(4, max(2, int((y_max - y_min) / fov_h) + 1))

    x_step = (x_max - x_min) / (n_x - 1) if n_x > 1 else 0
    y_step = (y_max - y_min) / (n_y - 1) if n_y > 1 else 0

    positions = []
    for iy in range(n_y):
        for ix in range(n_x):
            positions.append((x_min + ix * x_step, y_min + iy * y_step))
    return positions


def validate_frame(
    arr: np.ndarray,
    reference_mean: float | None,
) -> tuple[bool, str]:
    """Check that a frame looks like bare substrate."""
    mean = arr.mean()

    if mean < 20:
        return False, f"too dark (mean={mean:.1f})"

    if reference_mean is not None:
        ratio = mean / reference_mean
        if abs(ratio - 1.0) > 0.3:
            return False, (f"brightness mismatch (mean={mean:.1f}, ref={reference_mean:.1f}, ratio={ratio:.2f})")

    h, w = arr.shape[:2]
    center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
    ratio = center / mean if mean > 0 else 0
    if abs(ratio - 1.0) > 0.15:
        return False, (f"non-uniform (center={center:.1f}, full={mean:.1f}, ratio={ratio:.2f})")

    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build median flatfield from bare-substrate captures.",
    )

    parser.add_argument(
        "--objective-mag",
        type=str,
        required=True,
        help="Objective magnification (e.g. 2.5, 5x, 20)",
    )

    pos_group = parser.add_mutually_exclusive_group(required=True)
    pos_group.add_argument(
        "--positions",
        type=str,
        help="Space-separated X,Y pairs in µm",
    )
    pos_group.add_argument(
        "--chips-meta",
        type=str,
        help="Path to chips JSON (from find_chips.py)",
    )

    parser.add_argument(
        "--chip",
        type=int,
        default=0,
        help="Chip index to sample (with --chips-meta, default: 0)",
    )
    parser.add_argument("--z", type=float, default=None, help="Z position in µm (default: current microscope Z)")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "--white-balance",
        type=parse_white_balance,
        default="2.51,1.02,1.41",
    )
    parser.add_argument("--exposure-ms", type=float, default=1.0, help="Exposure time in ms")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--lamp", type=float, default=100, help="Lamp intensity (0-100%%)")
    parser.add_argument("--binning", type=int, default=2, choices=[0, 1, 2], help="Binning level (0=1x1, 1=2x2, 2=3x3)")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=0.1)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-capture output")
    parser.add_argument("--dry-run", action="store_true", help="Print positions and exit without capturing")
    parser.add_argument("--notes", type=str, default=None, help="Notes about substrate/conditions")
    args = parser.parse_args()

    # Resolve magnification for FOV computation
    desc = require_microscope_description()
    mag = float(args.objective_mag.lower().rstrip("x"))

    # Resolve positions
    if args.positions:
        positions = []
        for pair in args.positions.split():
            x, y = pair.split(",")
            positions.append((float(x), float(y)))
    else:
        fov = compute_frame_size_um(desc.camera, mag, binning_idx=args.binning)
        if fov is None:
            print(f"Error: could not compute FOV for binning level {args.binning}")
            return 1
        positions = generate_chip_grid(
            args.chips_meta,
            args.chip,
            (fov.x, fov.y),
        )
        print(f"Chip {args.chip} @ {mag}x: {len(positions)} positions (FOV {fov.x:.0f}x{fov.y:.0f} µm)")

    if args.dry_run:
        z_info = f" at Z={args.z:.1f}" if args.z is not None else " (Z will use current position)"
        print(f"\nDry run — {len(positions)} positions{z_info}:")
        for i, (x, y) in enumerate(positions):
            print(f"  [{i + 1}] ({x:.0f}, {y:.0f})")
        return 0

    all_frames = []
    with Microscope() as scope:
        # Switch objective
        scope.switch_objective_mag(args.objective_mag)

        # Resolve Z: explicit or current position
        z_um = args.z if args.z is not None else scope.z.position_um
        z_source = "from --z" if args.z is not None else "current position"
        print(f"Building flatfield from {len(positions)} positions at Z={z_um:.1f} µm ({z_source})")

        scope.light_on(args.lamp)
        scope.z.move_to(z_um)

        camera = scope.camera
        camera.binning = args.binning
        camera.exposure_time = args.exposure_ms / 1000.0
        camera.gain = args.gain
        camera.gain_rgb = args.white_balance
        camera.gamma = args.gamma

        for i, (x, y) in enumerate(positions):
            wait_all(list(scope.stage.move_to_async(x, y)))
            time.sleep(args.settle)

            camera.capture()  # discard (motion blur)
            image = camera.capture()

            arr = image.astype(np.float32)
            all_frames.append((x, y, arr))
            if not args.quiet:
                print(f"  [{i + 1}/{len(positions)}] ({x:.0f}, {y:.0f}) mean={arr.mean():.1f}")

    # Validate
    all_means = [arr.mean() for _, _, arr in all_frames]
    reference_mean = float(np.median(all_means))
    print(f"\nValidating ({len(all_frames)} frames, reference mean={reference_mean:.1f}):")

    good_frames = []
    for x, y, arr in all_frames:
        ok, reason = validate_frame(arr, reference_mean)
        status = "PASS" if ok else f"REJECT ({reason})"
        print(f"  ({x:.0f}, {y:.0f}) mean={arr.mean():.1f} — {status}")
        if ok:
            good_frames.append(arr)

    rejected = len(all_frames) - len(good_frames)
    if rejected:
        print(f"\n{rejected}/{len(all_frames)} frames rejected")

    if len(good_frames) < 3:
        print(f"Error: only {len(good_frames)} good frames, need >= 3")
        return 1

    stack = np.stack(good_frames, axis=0)
    raw_flatfield = np.median(stack, axis=0).astype(np.float32)

    # Vignetting sanity check (on raw values)
    h, w = raw_flatfield.shape[:2]
    center = raw_flatfield[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3].mean()
    corners = np.mean(
        [
            raw_flatfield[:100, :100].mean(),
            raw_flatfield[:100, -100:].mean(),
            raw_flatfield[-100:, :100].mean(),
            raw_flatfield[-100:, -100:].mean(),
        ]
    )
    vignette_pct = (1 - corners / center) * 100
    print(f"\nRaw flatfield: shape={raw_flatfield.shape}, mean={raw_flatfield.mean():.1f}")
    print(f"  Center={center:.1f}, corners={corners:.1f}, vignetting={vignette_pct:.1f}%")

    if vignette_pct < -5:
        print("WARNING: corners brighter than center — flatfield may be bad")

    # Convert to correction factors: correction = ch_mean / raw_pixel (~1.0)
    ch_means = raw_flatfield.mean(axis=(0, 1))  # shape (3,)
    raw_clamped = np.maximum(raw_flatfield, 1.0)
    correction = (ch_means / raw_clamped).astype(np.float32)
    print(f"\nCorrection factors: mean={correction.mean():.3f}, range=[{correction.min():.3f}, {correction.max():.3f}]")

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(outpath, correction)
    print(f"Saved to {outpath}")

    # Save .png preview (of raw flatfield for visual inspection)
    png_path = outpath.with_suffix(".png")
    ff_uint8 = np.clip(raw_flatfield, 0, 255).astype(np.uint8)
    Image.fromarray(ff_uint8).save(png_path)
    print(f"Saved preview: {png_path}")

    # Compute calibration values (mean BGR + reference white balance)
    mean_b = float(raw_flatfield[:, :, 0].mean())
    mean_g = float(raw_flatfield[:, :, 1].mean())
    mean_r = float(raw_flatfield[:, :, 2].mean())
    mean_bgr = [round(mean_b, 2), round(mean_g, 2), round(mean_r, 2)]

    wb_b = mean_g / mean_b if mean_b > 0 else 1.0
    wb_r = mean_g / mean_r if mean_r > 0 else 1.0
    reference_wb_bgr = [round(wb_b, 3), 1.0, round(wb_r, 3)]

    # Build source info
    if args.chips_meta:
        source = {
            "type": "chips_meta",
            "chips_meta_path": args.chips_meta,
            "chip_index": args.chip,
        }
    else:
        source = {"type": "manual"}

    # Save .json metadata
    meta = {
        "format": "correction_factors",
        "objective_mag": mag,
        "binning": int(np.array([1, 2, 3])[args.binning]),
        "lamp_pct": args.lamp,
        "exposure_s": args.exposure_ms / 1000.0,
        "gain": args.gain,
        "white_balance_rgb": [
            args.white_balance.red,
            args.white_balance.green,
            args.white_balance.blue,
        ],
        "gamma": args.gamma,
        "frame_size_px": [int(raw_flatfield.shape[1]), int(raw_flatfield.shape[0])],
        "z_um": z_um,
        "positions_um": [[round(x, 1), round(y, 1)] for x, y in positions],
        "num_positions": len(positions),
        "num_frames_kept": len(good_frames),
        "source": source,
        "mean_bgr": mean_bgr,
        "reference_wb_bgr": reference_wb_bgr,
        "flatfield_stats": {
            "shape": list(raw_flatfield.shape),
            "raw_mean": round(float(raw_flatfield.mean()), 1),
            "center_brightness": round(float(center), 1),
            "corner_brightness": round(float(corners), 1),
            "vignetting_pct": round(float(vignette_pct), 1),
            "correction_mean": round(float(correction.mean()), 3),
            "correction_range": [round(float(correction.min()), 3), round(float(correction.max()), 3)],
        },
        "captured_at": datetime.now().isoformat(),
        "notes": args.notes,
    }

    json_path = outpath.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
