"""Multi-row position-based stitch for snake scan patterns."""

import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np

DEFAULT_SCAN_DIR = Path(__file__).parent / "test_area_2"
CALIBRATION_DIR = Path(__file__).parent / "calibration"


def create_blend_alpha(width, height, blend_width_x, blend_width_y=0,
                       is_first_x=False, is_last_x=False,
                       is_first_y=False, is_last_y=False):
    """
    Create alpha mask with linear gradient edges for blending in both X and Y.
    """
    # Start with full opacity
    alpha = np.ones((height, width), dtype=np.float32)

    # X blending (horizontal edges)
    if blend_width_x > 0:
        if not is_first_x:
            for x in range(min(blend_width_x, width)):
                alpha[:, x] *= x / blend_width_x
        if not is_last_x:
            for x in range(max(0, width - blend_width_x), width):
                alpha[:, x] *= (width - 1 - x) / blend_width_x

    # Y blending (vertical edges)
    if blend_width_y > 0:
        if not is_first_y:
            for y in range(min(blend_width_y, height)):
                alpha[y, :] *= y / blend_width_y
        if not is_last_y:
            for y in range(max(0, height - blend_width_y), height):
                alpha[y, :] *= (height - 1 - y) / blend_width_y

    return Image.fromarray((alpha * 255).astype('uint8'), mode='L')


def deskew_image(img, shear_px):
    """Apply horizontal shear to correct rolling shutter skew.

    Positive shear_px: bottom of image shifts RIGHT
    Negative shear_px: bottom of image shifts LEFT

    Returns image with same dimensions. Void regions filled with transparent white.
    """
    width, height = img.size

    a, b, c = 1, shear_px / height, 0
    d, e, f = 0, 1, 0

    return img.transform(
        (width, height),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255, 0) if img.mode == 'RGBA' else (255, 255, 255)
    )


def find_constant_velocity_frames(frames):
    """
    Find frames in the constant-velocity portion of the scan.
    Returns (start_idx, end_idx, velocity) for the CV region.
    """
    if len(frames) < 3:
        return 0, len(frames), 0

    # Get frame-to-frame deltas
    deltas = [frames[i+1]["x_start"] - frames[i]["x_start"] for i in range(len(frames)-1)]

    # Find median delta (robust estimate of constant velocity spacing)
    sorted_deltas = sorted(deltas, key=abs)
    median_delta = sorted_deltas[len(sorted_deltas)//2]

    if abs(median_delta) < 1e-6:
        return 0, len(frames), 0

    # Frames are "constant velocity" if delta is within 20% of median
    tolerance = 0.2
    cv_mask = [abs(d - median_delta) / abs(median_delta) < tolerance for d in deltas]

    # Find first and last CV frame
    try:
        first_cv = next(i for i, m in enumerate(cv_mask) if m)
        last_cv = len(cv_mask) - 1 - next(i for i, m in enumerate(reversed(cv_mask)) if m)
    except StopIteration:
        return 0, len(frames), 0

    # Estimate velocity from median delta and frame timing
    avg_dt = sum(frames[i+1]["t_start"] - frames[i]["t_start"]
                 for i in range(first_cv, last_cv)) / max(1, last_cv - first_cv)
    velocity = median_delta / avg_dt if avg_dt > 0 else 0

    return first_cv, last_cv + 1, velocity


def fit_linear_positions(frames, cv_start, cv_end):
    """Fit linear model to CV frames and return smoothed positions."""
    cv_frames = frames[cv_start:cv_end]

    times = [(f["t_start"] + f["t_end"]) / 2 for f in cv_frames]
    positions = [f["x_start"] for f in cv_frames]

    n = len(times)
    sum_t = sum(times)
    sum_x = sum(positions)
    sum_tt = sum(t*t for t in times)
    sum_tx = sum(t*x for t, x in zip(times, positions))

    denom = n * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-10:
        return positions, 0

    fit_velocity = (n * sum_tx - sum_t * sum_x) / denom
    fit_intercept = (sum_x - fit_velocity * sum_t) / n

    smoothed = [fit_velocity * t + fit_intercept for t in times]
    return smoothed, fit_velocity


def stitch_row_to_global(meta, row, frame_w, frame_h, um_per_px, output_downsample,
                         scan_dir, global_x_min, global_x_max, blend=True, deskew=True,
                         hysteresis_um=0, flatfield=None, flatfield_mean=None):
    """
    Stitch a single row directly into global X coordinate space.

    Returns (row_image, cv_count) where row_image is sized to fit
    global_x_min to global_x_max + fov_width.
    """
    frames = meta["frames"]
    optics = meta["optics"]
    fov_width_um = optics["frame_width_um"]

    row_frames = frames[row["frame_start"]:row["frame_end"]]
    direction = row["direction"]

    # Find constant-velocity region for this row
    cv_start, cv_end, velocity = find_constant_velocity_frames(row_frames)
    cv_frames = row_frames[cv_start:cv_end]

    # Fit linear positions - keep in original order (global coordinates)
    smoothed_positions, fit_velocity = fit_linear_positions(row_frames, cv_start, cv_end)

    # Apply hysteresis correction to -X rows
    if direction < 0 and hysteresis_um != 0:
        smoothed_positions = [x + hysteresis_um for x in smoothed_positions]

    # Canvas width in global coordinates
    canvas_w_um = (global_x_max - global_x_min) + fov_width_um
    canvas_w = int(canvas_w_um / um_per_px)
    canvas_h = frame_h

    # Blend width for X
    if len(smoothed_positions) > 1:
        frame_spacing_um = abs(smoothed_positions[1] - smoothed_positions[0])
        frame_spacing_px = int(frame_spacing_um / um_per_px)
    else:
        frame_spacing_px = frame_w
    overlap_px = max(0, frame_w - frame_spacing_px)
    blend_width_x = overlap_px // 2 if blend else 0

    # Create canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Determine which frames fall within global range and stitch them
    num_frames = len(cv_frames)
    frames_placed = 0

    # For snake pattern, process frames in the order that places them left-to-right
    if direction < 0:
        # -X direction: reverse the order
        indices = list(range(num_frames - 1, -1, -1))
    else:
        # +X direction: normal order
        indices = list(range(num_frames))

    for seq_i, i in enumerate(indices):
        x_um = smoothed_positions[i]
        frame_idx = row["frame_start"] + cv_start + i

        # Check if this frame overlaps with global bounds
        # Frame center is at x_um, so frame covers [x_um, x_um + fov_width] in our placement
        # (left edge at x_um position for stitching purposes)
        frame_left = x_um
        frame_right = x_um + fov_width_um
        canvas_left = global_x_min
        canvas_right = global_x_max + fov_width_um

        # Skip if frame doesn't overlap with canvas at all
        if frame_right < canvas_left or frame_left > canvas_right:
            continue

        img = Image.open(scan_dir / f"frame_{frame_idx:04d}.jpg").convert("RGBA")

        if output_downsample > 1:
            img = img.resize((frame_w, frame_h), Image.LANCZOS)

        # Apply flatfield correction (after downsample, before blending)
        if flatfield is not None:
            img_arr = np.array(img, dtype=np.float32)
            # Correct RGB channels only (not alpha)
            for c in range(3):
                img_arr[:, :, c] = (img_arr[:, :, c] / flatfield[:, :, c]) * flatfield_mean
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="RGBA")

        if blend:
            is_first = (seq_i == 0)
            is_last = (seq_i == len(indices) - 1)
            alpha = create_blend_alpha(frame_w, frame_h, blend_width_x, 0,
                                       is_first_x=is_first, is_last_x=is_last)
        else:
            alpha = Image.new('L', img.size, 128)

        img.putalpha(alpha)

        # Place at global X position
        x_offset = int((x_um - global_x_min) / um_per_px)

        # Clamp to canvas bounds
        if x_offset < 0:
            # Crop left edge of frame
            crop_left = -x_offset
            img = img.crop((crop_left, 0, img.width, img.height))
            x_offset = 0
        if x_offset + img.width > canvas_w:
            # Crop right edge of frame
            crop_right = canvas_w - x_offset
            img = img.crop((0, 0, crop_right, img.height))

        if img.width > 0 and img.height > 0:
            canvas.alpha_composite(img, (x_offset, 0))
            frames_placed += 1

    # Apply deskew (no expansion - clips at edges)
    if deskew and fit_velocity != 0:
        readout_time = meta["camera"]["readout_time_s"]
        shear_um = abs(fit_velocity) * readout_time
        shear_px = shear_um / um_per_px
        correction_shear = -shear_px * direction
        canvas = deskew_image(canvas, correction_shear)

    return canvas, frames_placed


def main():
    parser = argparse.ArgumentParser(description="Stitch multi-row snake scan into 2D image")
    parser.add_argument("scan_dir", nargs="?", type=Path, default=DEFAULT_SCAN_DIR,
                        help="Scan directory (default: test_area_2)")
    parser.add_argument("--no-deskew", action="store_true", help="Disable rolling shutter deskew")
    parser.add_argument("--no-blend", action="store_true", help="Disable gradient blending")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Additional downsample factor (e.g., 2 = half resolution)")
    parser.add_argument("--rows", type=str, default=None,
                        help="Row range to process (e.g., '0-5' or '10')")
    # TODO: Investigate source of ~100 µm hysteresis between +X and -X scan directions.
    # Likely candidates: stage backlash, encoder offset, or position readout timing.
    parser.add_argument("--hysteresis", type=float, default=0,
                        help="Hysteresis correction in µm (applied to -X rows)")
    parser.add_argument("--flatfield", type=Path, default=None,
                        help="Path to flatfield .npy file (default: auto-load from calibration/)")
    parser.add_argument("--no-flatfield", action="store_true",
                        help="Disable flatfield correction")
    args = parser.parse_args()

    scan_dir = args.scan_dir

    # Load metadata
    with open(scan_dir / "scan_meta.json") as f:
        meta = json.load(f)

    if "rows" not in meta:
        print("Error: scan_meta.json has no 'rows' array. Use stitch_position.py for single-row scans.")
        return

    rows = meta["rows"]
    optics = meta["optics"]
    downsample = meta["downsample"]

    # Load flatfield for vignetting correction
    flatfield = None
    flatfield_mean = None
    flatfield_path = None

    if args.no_flatfield:
        print("Flatfield: disabled (--no-flatfield)")
    else:
        if args.flatfield:
            flatfield_path = args.flatfield
        else:
            # Auto-load based on objective and binning
            obj_mag = optics["objective_mag"]
            binning = meta["camera"]["binning"]
            flatfield_path = CALIBRATION_DIR / f"flatfield_{obj_mag}x_bin{binning}.npy"

        if not flatfield_path.exists():
            print(f"Error: Flatfield not found: {flatfield_path}")
            print("  Run capture_flatfield.py or specify --no-flatfield to skip correction.")
            return 1

        flatfield = np.load(flatfield_path).astype(np.float32)
        flatfield_mean = np.mean(flatfield)
        print(f"Flatfield: {flatfield_path}")

    # Parse row range if specified
    if args.rows:
        if '-' in args.rows:
            start, end = map(int, args.rows.split('-'))
            rows = [r for r in rows if start <= r["row_idx"] <= end]
        else:
            row_idx = int(args.rows)
            rows = [r for r in rows if r["row_idx"] == row_idx]

    # Calibration
    um_per_px = optics["sample_pixel_x_um"] * downsample * args.downsample
    fov_width_um = optics["frame_width_um"]
    fov_height_um = optics["frame_height_um"]
    y_step_um = meta["y_step_um"]
    y_overlap_um = fov_height_um - y_step_um

    # Get frame dimensions
    first_img = Image.open(scan_dir / "frame_0000.jpg")
    frame_w, frame_h = first_img.size
    frame_w //= args.downsample
    frame_h //= args.downsample

    # Resize flatfield to match working frame dimensions
    if flatfield is not None:
        ff_h, ff_w = flatfield.shape[:2]
        if ff_h != frame_h or ff_w != frame_w:
            # Resize each channel separately using LANCZOS
            ff_resized = np.zeros((frame_h, frame_w, 3), dtype=np.float32)
            for c in range(3):
                ff_channel = Image.fromarray(flatfield[:, :, c], mode='F')
                ff_channel = ff_channel.resize((frame_w, frame_h), Image.LANCZOS)
                ff_resized[:, :, c] = np.array(ff_channel, dtype=np.float32)
            print(f"  Resized flatfield {ff_w}x{ff_h} -> {frame_w}x{frame_h}")
            flatfield = ff_resized
            flatfield_mean = np.mean(flatfield)

    print(f"Scan: {len(meta['rows'])} rows, {meta['frame_count']} frames")
    print(f"Processing: {len(rows)} rows")
    print(f"Calibration: {um_per_px:.3f} µm/px (downsample {downsample}x{args.downsample})")
    print(f"Frame: {frame_w}x{frame_h} px = {fov_width_um:.0f}x{fov_height_um:.0f} µm")
    print(f"Y step: {y_step_um:.0f} µm, Y overlap: {y_overlap_um:.0f} µm ({y_overlap_um/fov_height_um*100:.0f}%)")

    # Process each row to find CV regions and global X bounds
    print(f"\nDetecting constant-velocity regions...")
    row_results = []

    for row in rows:
        frames = meta["frames"][row["frame_start"]:row["frame_end"]]
        cv_start, cv_end, velocity = find_constant_velocity_frames(frames)
        cv_count = cv_end - cv_start

        # Get smoothed X positions for CV region (in global coordinates)
        smoothed, fit_vel = fit_linear_positions(frames, cv_start, cv_end)
        x_min = min(smoothed)
        x_max = max(smoothed)

        row_results.append({
            "row": row,
            "cv_start": cv_start,
            "cv_end": cv_end,
            "cv_count": cv_count,
            "x_min": x_min,
            "x_max": x_max,
            "smoothed": smoothed,
            "velocity": velocity,
        })

        print(f"  Row {row['row_idx']:2d}: CV frames {cv_start}-{cv_end-1} "
              f"({cv_count} frames), X: {x_min:.0f} - {x_max:.0f} µm")

    # Find global X bounds: intersection of all rows
    global_x_min = max(r["x_min"] for r in row_results)
    global_x_max = min(r["x_max"] for r in row_results)
    global_x_range = global_x_max - global_x_min + fov_width_um

    print(f"\nGlobal X bounds: {global_x_min:.0f} - {global_x_max:.0f} µm ({global_x_range:.0f} µm total)")

    # Calculate final canvas dimensions
    n_rows = len(rows)
    total_height_um = y_step_um * (n_rows - 1) + fov_height_um

    canvas_w = int(global_x_range / um_per_px)
    canvas_h = int(total_height_um / um_per_px)

    print(f"Final canvas: {canvas_w}x{canvas_h} px ({global_x_range:.0f}x{total_height_um:.0f} µm)")

    # Y blend width
    y_overlap_px = int(y_overlap_um / um_per_px)
    blend_width_y = y_overlap_px // 2 if not args.no_blend else 0

    # Create final canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    print(f"\nStitching rows...")

    for i, result in enumerate(row_results):
        row = result["row"]
        row_idx = row["row_idx"]

        # Stitch this row in global coordinates
        row_img, frames_placed = stitch_row_to_global(
            meta, row, frame_w, frame_h, um_per_px, args.downsample,
            scan_dir, global_x_min, global_x_max,
            blend=not args.no_blend, deskew=not args.no_deskew,
            hysteresis_um=args.hysteresis,
            flatfield=flatfield, flatfield_mean=flatfield_mean
        )

        # Calculate Y position for this row
        # Use last row as reference (lowest Y = top of image)
        last_y = rows[-1]["y_um"]
        row_y = row["y_um"]
        y_offset_um = row_y - last_y
        y_offset_px = int(y_offset_um / um_per_px)

        # Apply Y blending
        # Note: rows are placed with highest row index at top (y=0), lowest at bottom
        # is_first_y = don't fade top edge, is_last_y = don't fade bottom edge
        if not args.no_blend:
            is_at_top = (i == n_rows - 1)  # Last in iteration = top of image
            is_at_bottom = (i == 0)         # First in iteration = bottom of image

            row_alpha = row_img.split()[3]
            y_blend = create_blend_alpha(row_img.width, row_img.height, 0, blend_width_y,
                                         is_first_y=is_at_top, is_last_y=is_at_bottom)

            existing = np.array(row_alpha, dtype=np.float32)
            new_blend = np.array(y_blend, dtype=np.float32)
            combined = (existing * new_blend / 255).astype('uint8')
            row_img.putalpha(Image.fromarray(combined, mode='L'))

        # Ensure row image fits canvas width (handle rounding)
        if row_img.width != canvas_w:
            if row_img.width > canvas_w:
                row_img = row_img.crop((0, 0, canvas_w, row_img.height))
            # If smaller, composite will just leave edges transparent

        # Composite onto canvas
        canvas.alpha_composite(row_img, (0, y_offset_px))

        print(f"  Row {row_idx}: {frames_placed} frames, y={y_offset_px} px")

    # Convert to RGB with white background
    background = Image.new("RGB", canvas.size, (255, 255, 255))
    background.paste(canvas, mask=canvas.split()[3])

    # Save image
    out_path = scan_dir.parent / f"{scan_dir.name}_stitch.png"
    background.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Final size: {background.width}x{background.height} px")

    # Save stitch metadata for downstream processing (chip detection, etc.)
    # Stage bounds: the actual area covered by the stitched image
    # Note: reported positions are frame CENTERS, so coverage extends ±FOV/2
    # - X: global_x_min - fov/2 to global_x_max + fov/2
    # - Y: last row's y - fov/2 to first row's y + fov/2
    stage_bounds_um = {
        "x_min": global_x_min - fov_width_um / 2,
        "x_max": global_x_max + fov_width_um / 2,
        "y_min": rows[-1]["y_um"] - fov_height_um / 2,
        "y_max": rows[0]["y_um"] + fov_height_um / 2,
    }

    stitch_meta = {
        "image_file": out_path.name,
        "image_size_px": [background.width, background.height],
        "stage_bounds_um": stage_bounds_um,
        "scale_um_per_px": um_per_px,
        "source_scan": f"{scan_dir.name}/scan_meta.json",
        "objective_mag": optics["objective_mag"],
        "downsample": downsample * args.downsample,
        "flatfield_correction": {
            "applied": flatfield is not None,
            "file": str(flatfield_path) if flatfield is not None else None,
        },
    }

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(stitch_meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
