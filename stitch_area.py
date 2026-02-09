"""Multi-row position-based stitch for snake scan patterns."""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy.signal import savgol_filter

DEFAULT_SCAN_DIR = Path(__file__).parent / "test_area_2"
CALIBRATION_DIR = Path(__file__).parent / "calibration"


def create_blend_alpha(
    width,
    height,
    blend_width_x,
    blend_width_y=0,
    is_first_x=False,
    is_last_x=False,
    is_first_y=False,
    is_last_y=False,
):
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

    return Image.fromarray((alpha * 255).astype("uint8"), mode="L")


def deskew_image(img, shear_px, bg_color=(0, 0, 0)):
    """Apply horizontal shear to correct rolling shutter skew.

    Positive shear_px: bottom of image shifts RIGHT
    Negative shear_px: bottom of image shifts LEFT

    Returns image with same dimensions. Void regions filled with bg_color.
    """
    width, height = img.size

    a, b, c = 1, shear_px / height, 0
    d, e, f = 0, 1, 0

    return img.transform(
        (width, height),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
        fillcolor=(bg_color + (0,)) if img.mode == "RGBA" else bg_color,
    )


def draw_grid(background, stage_bounds_um, um_per_px, spacing_um, line_color, line_width):
    """Draw stage-coordinate grid lines and mm labels on the stitched image."""
    w, h = background.size
    x_min = stage_bounds_um["x_min"]
    x_max = stage_bounds_um["x_max"]
    y_min = stage_bounds_um["y_min"]
    y_max = stage_bounds_um["y_max"]

    color = ImageColor.getrgb(line_color)
    if len(color) == 3:
        color = color + (80,)
    # Label color: same hue but higher opacity for readability
    label_alpha = min(255, color[3] * 3)
    label_color = color[:3] + (label_alpha,)
    shadow_color = (0, 0, 0, label_alpha)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(14, min(w, h) // 500)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    label_pad = 4

    # Vertical grid lines (constant X)
    x = math.ceil(x_min / spacing_um) * spacing_um
    while x <= x_max:
        px = int((x - x_min) / um_per_px)
        if 0 <= px < w:
            draw.line([(px, 0), (px, h - 1)], fill=color, width=line_width)
            label = f"{x / 1000:.1f}"
            # Shadow then text
            draw.text((px + label_pad + 1, label_pad + 1), label, fill=shadow_color, font=font)
            draw.text((px + label_pad, label_pad), label, fill=label_color, font=font)
        x += spacing_um

    # Horizontal grid lines (constant Y)
    y = math.ceil(y_min / spacing_um) * spacing_um
    while y <= y_max:
        py = int((y - y_min) / um_per_px)
        if 0 <= py < h:
            draw.line([(0, py), (w - 1, py)], fill=color, width=line_width)
            label = f"{y / 1000:.1f}"
            draw.text((label_pad + 1, py + label_pad + 1), label, fill=shadow_color, font=font)
            draw.text((label_pad, py + label_pad), label, fill=label_color, font=font)
        y += spacing_um

    result = Image.alpha_composite(background.convert("RGBA"), overlay)
    return result.convert("RGB")


def find_constant_velocity_frames(frames):
    """
    Find frames in the constant-velocity portion of the scan.
    Returns (start_idx, end_idx, velocity) for the CV region.
    """
    if len(frames) < 3:
        return 0, len(frames), 0

    # Get frame-to-frame deltas
    deltas = [frames[i + 1]["x_start"] - frames[i]["x_start"] for i in range(len(frames) - 1)]

    # Find median delta (robust estimate of constant velocity spacing)
    sorted_deltas = sorted(deltas, key=abs)
    median_delta = sorted_deltas[len(sorted_deltas) // 2]

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
    avg_dt = sum(frames[i + 1]["t_start"] - frames[i]["t_start"] for i in range(first_cv, last_cv)) / max(
        1, last_cv - first_cv
    )
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
    sum_tt = sum(t * t for t in times)
    sum_tx = sum(t * x for t, x in zip(times, positions, strict=True))

    denom = n * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-10:
        return positions, 0

    fit_velocity = (n * sum_tx - sum_t * sum_x) / denom
    fit_intercept = (sum_x - fit_velocity * sum_t) / n

    smoothed = [fit_velocity * t + fit_intercept for t in times]
    return smoothed, fit_velocity


def smooth_positions_savgol(frames, cv_start, cv_end, window=7, polyorder=2):
    """Smooth CV frame positions with Savitzky-Golay filter.

    Unlike fit_linear_positions which fits a single line across all CV frames,
    this uses a local polynomial fit that tracks the actual stage trajectory
    while reducing position jitter (~100 µm from SDK readout noise).

    Returns (smoothed_positions, fit_velocity) — fit_velocity is from a linear
    fit used only for deskew correction.
    """
    cv_frames = frames[cv_start:cv_end]
    raw_x = np.array([f["x_start"] for f in cv_frames])

    if len(raw_x) < window:
        return raw_x.tolist(), 0

    smoothed = savgol_filter(raw_x, window, polyorder)

    # Linear fit velocity still needed for rolling shutter deskew
    _, fit_velocity = fit_linear_positions(frames, cv_start, cv_end)

    return smoothed.tolist(), fit_velocity


def stitch_row_to_global(
    meta,
    row,
    frame_w,
    frame_h,
    um_per_px,
    output_downsample,
    scan_dir,
    global_x_min,
    global_x_max,
    blend=True,
    deskew=True,
    hysteresis_um=0,
    flatfield=None,
    flatfield_mean=None,
    num_threads=1,
    bg_color=(0, 0, 0),
    smoothing="linear",
):
    """
    Stitch a single row directly into global X coordinate space.

    Returns (row_image, cv_count) where row_image is sized to fit
    global_x_min to global_x_max + fov_width.

    smoothing: "linear" (global linear fit) or "savgol" (Savitzky-Golay local filter)
    """
    frames = meta["frames"]
    optics = meta["optics"]
    fov_width_um = optics["frame_width_um"]

    row_frames = frames[row["frame_start"] : row["frame_end"]]
    direction = row["direction"]

    # Find constant-velocity region for this row
    cv_start, cv_end, velocity = find_constant_velocity_frames(row_frames)
    cv_frames = row_frames[cv_start:cv_end]

    # Smooth positions
    if smoothing == "savgol":
        smoothed_positions, fit_velocity = smooth_positions_savgol(row_frames, cv_start, cv_end)
    else:
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

    # Determine which frames fall within global range
    num_frames = len(cv_frames)

    # For snake pattern, process frames in the order that places them left-to-right
    if direction < 0:
        indices = list(range(num_frames - 1, -1, -1))
    else:
        indices = list(range(num_frames))

    # Build work list: filter to frames that overlap canvas
    canvas_right_um = global_x_max + fov_width_um
    work_items = []
    for seq_i, i in enumerate(indices):
        x_um = smoothed_positions[i]
        if x_um + fov_width_um < global_x_min or x_um > canvas_right_um:
            continue
        frame_idx = row["frame_start"] + cv_start + i
        x_offset = int((x_um - global_x_min) / um_per_px)
        work_items.append((seq_i, frame_idx, x_offset))

    num_work = len(work_items)

    def load_frame(item):
        """Load, resize, flatfield-correct, and alpha-blend a single frame."""
        seq_i, frame_idx, _ = item
        path = scan_dir / f"frame_{frame_idx:04d}.jpg"

        img = Image.open(path)
        # JPEG draft() decodes at reduced resolution (1/2, 1/4, 1/8) during
        # DCT, avoiding a full-resolution decode. No-op for non-JPEG.
        if output_downsample > 1:
            img.draft("RGB", (frame_w, frame_h))
        img.load()
        if img.size[0] != frame_w or img.size[1] != frame_h:
            img = img.resize((frame_w, frame_h), Image.LANCZOS)
        img = img.convert("RGBA")

        if flatfield is not None:
            img_arr = np.array(img, dtype=np.float32)
            for c in range(3):
                img_arr[:, :, c] = (img_arr[:, :, c] / flatfield[:, :, c]) * flatfield_mean
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="RGBA")

        if blend:
            is_first = seq_i == 0
            is_last = seq_i == num_work - 1
            alpha = create_blend_alpha(frame_w, frame_h, blend_width_x, 0, is_first_x=is_first, is_last_x=is_last)
        else:
            alpha = Image.new("L", img.size, 128)
        img.putalpha(alpha)

        return img

    # Load frames in parallel, composite sequentially
    if num_threads > 1 and num_work > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            loaded = list(pool.map(load_frame, work_items))
    else:
        loaded = [load_frame(item) for item in work_items]

    frames_placed = 0
    for img, (_, _, x_offset) in zip(loaded, work_items, strict=True):
        # Clamp to canvas bounds
        if x_offset < 0:
            img = img.crop((-x_offset, 0, img.width, img.height))
            x_offset = 0
        if x_offset + img.width > canvas_w:
            img = img.crop((0, 0, canvas_w - x_offset, img.height))

        if img.width > 0 and img.height > 0:
            canvas.alpha_composite(img, (x_offset, 0))
            frames_placed += 1

    # Apply deskew (no expansion - clips at edges)
    if deskew and fit_velocity != 0:
        readout_time = meta["camera"]["readout_time_s"]
        shear_um = abs(fit_velocity) * readout_time
        shear_px = shear_um / um_per_px
        correction_shear = -shear_px * direction
        canvas = deskew_image(canvas, correction_shear, bg_color)

    return canvas, frames_placed


def main():
    parser = argparse.ArgumentParser(description="Stitch multi-row snake scan into 2D image")
    parser.add_argument(
        "scan_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SCAN_DIR,
        help="Scan directory (default: test_area_2)",
    )
    parser.add_argument("--no-deskew", action="store_true", help="Disable rolling shutter deskew")
    parser.add_argument("--no-blend", action="store_true", help="Disable gradient blending")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Additional downsample factor (e.g., 2 = half resolution)",
    )
    parser.add_argument("--rows", type=str, default=None, help="Row range to process (e.g., '0-5' or '10')")
    # TODO: Investigate source of ~100 µm hysteresis between +X and -X scan directions.
    # Likely candidates: stage backlash, encoder offset, or position readout timing.
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=0,
        help="Hysteresis correction in µm (applied to -X rows)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="0,0,0,0",
        help="Crop margins in µm: TOP,RIGHT,BOTTOM,LEFT (default: 0,0,0,0)",
    )
    parser.add_argument(
        "--flatfield",
        type=Path,
        default=None,
        help="Path to flatfield .npy file (default: auto-load from calibration/)",
    )
    parser.add_argument("--no-flatfield", action="store_true", help="Disable flatfield correction")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for parallel frame loading (default: 4)",
    )
    parser.add_argument("--bg", type=str, default="black", help="Background color: name or #RRGGBB (default: black)")
    parser.add_argument(
        "--grid-spacing-um",
        type=float,
        default=0,
        help="Grid line spacing in µm (0 = no grid, 1000 = 1mm)",
    )
    parser.add_argument(
        "--grid-line-color",
        type=str,
        default="#FFFFFF50",
        help="Grid line color: name or #RRGGBBAA (default: #FFFFFF50)",
    )
    parser.add_argument("--grid-line-width", type=int, default=1, help="Grid line width in pixels (default: 1)")
    parser.add_argument(
        "--smoothing",
        type=str,
        default="linear",
        choices=["linear", "savgol"],
        help="Position smoothing: linear (global fit, default) or savgol (local filter)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename (default: {scan_dir}_stitch.png)",
    )
    args = parser.parse_args()

    bg_color = ImageColor.getrgb(args.bg)

    start_time = time.perf_counter()
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
        if "-" in args.rows:
            start, end = map(int, args.rows.split("-"))
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
                ff_channel = Image.fromarray(flatfield[:, :, c], mode="F")
                ff_channel = ff_channel.resize((frame_w, frame_h), Image.LANCZOS)
                ff_resized[:, :, c] = np.array(ff_channel, dtype=np.float32)
            print(f"  Resized flatfield {ff_w}x{ff_h} -> {frame_w}x{frame_h}")
            flatfield = ff_resized
            flatfield_mean = np.mean(flatfield)

    print(f"Scan: {len(meta['rows'])} rows, {meta['frame_count']} frames")
    print(f"Processing: {len(rows)} rows")
    print(f"Calibration: {um_per_px:.3f} µm/px (downsample {downsample}x{args.downsample})")
    print(f"Frame: {frame_w}x{frame_h} px = {fov_width_um:.0f}x{fov_height_um:.0f} µm")
    print(f"Y step: {y_step_um:.0f} µm, Y overlap: {y_overlap_um:.0f} µm ({y_overlap_um / fov_height_um * 100:.0f}%)")

    # Process each row to find CV regions and global X bounds
    smooth_fn = smooth_positions_savgol if args.smoothing == "savgol" else fit_linear_positions
    print(f"\nDetecting constant-velocity regions (smoothing: {args.smoothing})...")
    row_results = []

    for row in rows:
        frames = meta["frames"][row["frame_start"] : row["frame_end"]]
        cv_start, cv_end, velocity = find_constant_velocity_frames(frames)
        cv_count = cv_end - cv_start

        # Get smoothed X positions for CV region (in global coordinates)
        smoothed, fit_vel = smooth_fn(frames, cv_start, cv_end)
        x_min = min(smoothed)
        x_max = max(smoothed)

        row_results.append(
            {
                "row": row,
                "cv_start": cv_start,
                "cv_end": cv_end,
                "cv_count": cv_count,
                "x_min": x_min,
                "x_max": x_max,
                "smoothed": smoothed,
                "velocity": velocity,
            }
        )

        print(
            f"  Row {row['row_idx']:2d}: CV frames {cv_start}-{cv_end - 1} "
            f"({cv_count} frames), X: {x_min:.0f} - {x_max:.0f} µm"
        )

    # Find global X bounds: union of all rows
    global_x_min = min(r["x_min"] for r in row_results)
    global_x_max = max(r["x_max"] for r in row_results)
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

    print("\nStitching rows...")

    for result in row_results:
        row = result["row"]
        row_idx = row["row_idx"]

        # Stitch this row in global coordinates
        row_img, frames_placed = stitch_row_to_global(
            meta,
            row,
            frame_w,
            frame_h,
            um_per_px,
            args.downsample,
            scan_dir,
            global_x_min,
            global_x_max,
            blend=not args.no_blend,
            deskew=not args.no_deskew,
            hysteresis_um=args.hysteresis,
            flatfield=flatfield,
            flatfield_mean=flatfield_mean,
            num_threads=args.threads,
            bg_color=bg_color,
            smoothing=args.smoothing,
        )

        # Calculate Y position for this row (min Y = top of image)
        min_y = min(r["y_um"] for r in rows)
        row_y = row["y_um"]
        y_offset_um = row_y - min_y
        y_offset_px = int(y_offset_um / um_per_px)

        # Apply Y blending
        # is_first_y = don't fade top edge, is_last_y = don't fade bottom edge
        if not args.no_blend:
            is_at_top = row_y == min(r["y_um"] for r in rows)
            is_at_bottom = row_y == max(r["y_um"] for r in rows)

            row_alpha = row_img.split()[3]
            y_blend = create_blend_alpha(
                row_img.width,
                row_img.height,
                0,
                blend_width_y,
                is_first_y=is_at_top,
                is_last_y=is_at_bottom,
            )

            existing = np.array(row_alpha, dtype=np.float32)
            new_blend = np.array(y_blend, dtype=np.float32)
            combined = (existing * new_blend / 255).astype("uint8")
            row_img.putalpha(Image.fromarray(combined, mode="L"))

        # Ensure row image fits canvas width (handle rounding)
        if row_img.width != canvas_w:
            if row_img.width > canvas_w:
                row_img = row_img.crop((0, 0, canvas_w, row_img.height))
            # If smaller, composite will just leave edges transparent

        # Composite onto canvas
        canvas.alpha_composite(row_img, (0, y_offset_px))

        print(f"  Row {row_idx}: {frames_placed} frames, y={y_offset_px} px")

    # Convert to RGB with background color
    background = Image.new("RGB", canvas.size, bg_color)
    background.paste(canvas, mask=canvas.split()[3])

    # Stage bounds before crop
    # Note: reported positions are frame CENTERS, so coverage extends ±FOV/2
    # - X: global_x_min - fov/2 to global_x_max + fov/2
    # - Y: last row's y - fov/2 to first row's y + fov/2
    stage_bounds_um = {
        "x_min": global_x_min - fov_width_um / 2,
        "x_max": global_x_max + fov_width_um / 2,
        "y_min": min(r["y_um"] for r in rows) - fov_height_um / 2,
        "y_max": max(r["y_um"] for r in rows) + fov_height_um / 2,
    }

    # Apply crop margins (top, right, bottom, left in µm)
    crop_parts = [float(x) for x in args.crop.split(",")]
    if len(crop_parts) != 4:
        print("Error: --crop must be TOP,RIGHT,BOTTOM,LEFT (e.g., '15000,0,7000,0')")
        return 1
    crop_top_um, crop_right_um, crop_bottom_um, crop_left_um = crop_parts

    if any(c != 0 for c in crop_parts):
        crop_top_px = int(crop_top_um / um_per_px)
        crop_right_px = int(crop_right_um / um_per_px)
        crop_bottom_px = int(crop_bottom_um / um_per_px)
        crop_left_px = int(crop_left_um / um_per_px)

        w, h = background.size
        box = (crop_left_px, crop_top_px, w - crop_right_px, h - crop_bottom_px)
        background = background.crop(box)

        # Image top = low Y, bottom = high Y
        stage_bounds_um["y_min"] += crop_top_um
        stage_bounds_um["y_max"] -= crop_bottom_um
        stage_bounds_um["x_min"] += crop_left_um
        stage_bounds_um["x_max"] -= crop_right_um

        print(
            f"\nCropped: top={crop_top_um:.0f} right={crop_right_um:.0f} "
            f"bottom={crop_bottom_um:.0f} left={crop_left_um:.0f} µm"
        )

    # Draw grid overlay
    if args.grid_spacing_um > 0:
        background = draw_grid(
            background,
            stage_bounds_um,
            um_per_px,
            args.grid_spacing_um,
            args.grid_line_color,
            args.grid_line_width,
        )
        print(f"Grid: {args.grid_spacing_um:.0f} µm spacing")

    # Save image
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = scan_dir.parent / f"{scan_dir.name}_stitch.jpg"
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        background.save(out_path, quality=95)
    else:
        background.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Final size: {background.width}x{background.height} px")

    # Compute average CV X overlap across all rows
    all_cv_steps = []
    for rr in row_results:
        smoothed = rr["smoothed"]
        if len(smoothed) >= 2:
            steps = [abs(smoothed[i + 1] - smoothed[i]) for i in range(len(smoothed) - 1)]
            all_cv_steps.extend(steps)

    if all_cv_steps:
        avg_cv_step = sum(all_cv_steps) / len(all_cv_steps)
        avg_cv_overlap_pct = (fov_width_um - avg_cv_step) / fov_width_um * 100
    else:
        avg_cv_step = None
        avg_cv_overlap_pct = None

    duration_s = time.perf_counter() - start_time
    stitch_meta = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(duration_s, 2),
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
        "cv_stats": {
            "avg_step_um": round(avg_cv_step, 1) if avg_cv_step else None,
            "avg_overlap_pct": round(avg_cv_overlap_pct, 1) if avg_cv_overlap_pct else None,
            "frame_width_um": round(fov_width_um, 1),
        },
    }

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(stitch_meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
