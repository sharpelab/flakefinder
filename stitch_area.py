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


def deskew_image(img, shear_px):
    """Apply horizontal shear to correct rolling shutter skew.

    Positive shear_px: bottom of image shifts RIGHT
    Negative shear_px: bottom of image shifts LEFT

    Returns image with same dimensions. Edge pixels are replicated to fill
    void regions (avoids transparent gaps at frame edges).
    """
    width, height = img.size
    pad = int(abs(shear_px)) + 1

    # Pad the side that will have void with edge-replicated pixels
    arr = np.array(img)
    if shear_px > 0:
        # Void at bottom-right → pad right side
        pad_width = [(0, 0), (0, pad)] + ([(0, 0)] if arr.ndim == 3 else [])
        x_crop = 0
    else:
        # Void at bottom-left → pad left side
        pad_width = [(0, 0), (pad, 0)] + ([(0, 0)] if arr.ndim == 3 else [])
        x_crop = pad

    padded_arr = np.pad(arr, pad_width, mode="edge")
    padded = Image.fromarray(padded_arr, mode=img.mode)
    pw, ph = padded.size

    a, b, c = 1, shear_px / ph, 0
    d, e, f = 0, 1, 0

    result = padded.transform(
        (pw, ph),
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0,) * len(img.getbands()),
    )

    # Crop back to original size
    return result.crop((x_crop, 0, x_crop + width, height))


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


def draw_row_labels(background, rows, um_per_px, stage_bounds_um):
    """Draw row index + direction labels (e.g., '0+', '1-') on the left edge."""
    w, h = background.size
    y_min_stage = stage_bounds_um["y_min"]

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, min(w, h) // 200)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    pad = 6
    for row in rows:
        row_idx = row["row_idx"]
        direction = row["direction"]
        label = f"{row_idx}{'+' if direction > 0 else '-'}"

        # Row center Y in image coords
        row_y_stage = row["y_um"]
        row_y_px = int((row_y_stage - y_min_stage) / um_per_px)

        # Center text vertically on the row
        bbox = font.getbbox(label)
        text_h = bbox[3] - bbox[1]
        ty = row_y_px - text_h // 2

        # Draw with shadow for readability
        draw.text((pad + 1, ty + 1), label, fill=(0, 0, 0, 200), font=font)
        draw.text((pad, ty), label, fill=(255, 255, 100, 220), font=font)

    result = Image.alpha_composite(background.convert("RGBA"), overlay)
    return result.convert("RGB")


def draw_frame_outlines(background, frame_rects, crop_offset=(0, 0)):
    """Draw outline rectangles for each placed frame."""
    w, h = background.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ox, oy = crop_offset

    for fx, fy, fw, fh in frame_rects:
        x0 = fx - ox
        y0 = fy - oy
        x1 = x0 + fw - 1
        y1 = y0 + fh - 1
        # Skip if fully outside
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 100), width=1)

    result = Image.alpha_composite(background.convert("RGBA"), overlay)
    return result.convert("RGB")


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
    savgol_window=15,
    flatfield=None,
    flatfield_mean=None,
    num_threads=1,
):
    """
    Stitch a single row directly into global X coordinate space.

    Returns (row_image, frames_placed) where row_image is sized to fit
    global_x_min to global_x_max + fov_width.

    All frame positions are savgol-smoothed. Stationary boundary frames
    (stage parked) are trimmed. Rolling shutter deskew uses per-frame
    velocity (not a single CV estimate).
    """
    frames = meta["frames"]
    optics = meta["optics"]
    fov_width_um = optics["frame_width_um"]

    all_row_frames = frames[row["frame_start"] : row["frame_end"]]
    direction = row["direction"]

    # Trim stationary frames at row boundaries (stage parked, camera still capturing)
    raw_x = np.array([f["x_start"] for f in all_row_frames])
    deltas = np.diff(raw_x)
    stationary_thresh = 2.0  # µm — frames closer than this are "stationary"

    # Find first/last moving frames
    moving = np.abs(deltas) > stationary_thresh
    if np.any(moving):
        first_moving = int(np.argmax(moving))
        last_moving = int(len(moving) - 1 - np.argmax(moving[::-1])) + 1
    else:
        first_moving = 0
        last_moving = len(all_row_frames)
    row_frames = all_row_frames[first_moving:last_moving]

    # Fix capture bubble positions before smoothing.
    # Bubbles: a delayed capture (large gap before) followed by an immediate
    # one (small gap after, frame was already in camera buffer). Both frames
    # have inaccurate interpolated positions because the actual exposure time
    # differs from t_start. Fix by replacing bubble positions with linear
    # interpolation from their clean neighbors.
    raw_positions = np.array([f["x_start"] for f in row_frames])
    raw_times = np.array([(f["t_start"] + f["t_end"]) / 2 for f in row_frames])
    n = len(raw_positions)

    if n >= 5:
        dt = np.diff(raw_times)
        median_dt = np.median(dt)
        # A bubble is a gap > 1.5x median followed by a gap < 0.7x median
        for i in range(len(dt) - 1):
            if dt[i] > 1.5 * median_dt and dt[i + 1] < 0.7 * median_dt:
                # Frames i+1 (delayed) and i+2 (immediate-after) have bad positions.
                # Interpolate from neighbors based on time.
                # Find clean neighbors: i (before bubble) and i+3 or later (after)
                left = i
                right = min(i + 3, n - 1)
                if right > left:
                    t_left, t_right = raw_times[left], raw_times[right]
                    x_left, x_right = raw_positions[left], raw_positions[right]
                    for j in range(left + 1, right):
                        frac = (raw_times[j] - t_left) / (t_right - t_left)
                        raw_positions[j] = x_left + frac * (x_right - x_left)

    # Savgol smooth positions. Absorbs residual position polling noise.
    if savgol_window > 0 and n >= savgol_window:
        window = min(savgol_window, n if n % 2 == 1 else n - 1)
        all_positions = savgol_filter(raw_positions, window, 2).tolist()
    else:
        all_positions = raw_positions.tolist()

    # Apply hysteresis correction to -X rows
    if direction < 0 and hysteresis_um != 0:
        all_positions = [x + hysteresis_um for x in all_positions]

    # Canvas width in global coordinates
    canvas_w_um = (global_x_max - global_x_min) + fov_width_um
    canvas_w = int(canvas_w_um / um_per_px)
    canvas_h = frame_h

    # Blend width for X (based on mid-row frame spacing, i.e. CV region)
    mid = len(all_positions) // 2
    mid_positions = all_positions[max(0, mid - 5) : mid + 5]
    if len(mid_positions) > 1:
        frame_spacing_um = abs(mid_positions[1] - mid_positions[0])
        frame_spacing_px = int(frame_spacing_um / um_per_px)
    else:
        frame_spacing_px = frame_w
    overlap_px = max(0, frame_w - frame_spacing_px)
    blend_width_x = overlap_px // 2 if blend else 0

    # Create canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Use all frames (including accel zones)
    num_frames = len(row_frames)

    # For snake pattern, process frames in the order that places them left-to-right
    if direction < 0:
        indices = list(range(num_frames - 1, -1, -1))
    else:
        indices = list(range(num_frames))

    # Compute per-frame velocity for deskew using savgol derivative.
    # Uses bubble-fixed positions (raw_positions) so timing bubbles don't
    # produce velocity spikes. A wider window (31 samples, ~450ms) gives
    # smooth estimates; the stage velocity is physically smooth.
    if deskew:
        readout_time = meta["camera"]["readout_time_s"]
        vel_window = min(31, n if n % 2 == 1 else n - 1)
        if vel_window >= 5:
            dt = np.mean(np.diff(raw_times))
            frame_velocities = savgol_filter(raw_positions, vel_window, 2, deriv=1, delta=dt)
        else:
            frame_velocities = np.gradient(np.array(all_positions), raw_times)
    else:
        frame_velocities = np.zeros(num_frames)

    # Build work list: filter to frames that overlap canvas
    canvas_right_um = global_x_max + fov_width_um
    work_items = []
    for seq_i, i in enumerate(indices):
        x_um = all_positions[i]
        if x_um + fov_width_um < global_x_min or x_um > canvas_right_um:
            continue
        frame_idx = row["frame_start"] + first_moving + i

        # Per-frame deskew from actual velocity at this frame
        vel = frame_velocities[i]
        shear_um = abs(vel) * readout_time if deskew else 0
        shear_px = shear_um / um_per_px
        frame_deskew = -shear_px * np.sign(vel)

        x_offset = int((x_um - global_x_min) / um_per_px)
        work_items.append((seq_i, frame_idx, x_offset, frame_deskew))

    num_work = len(work_items)

    def load_frame(item):
        """Load, resize, flatfield-correct, deskew, and alpha-blend a single frame."""
        seq_i, frame_idx, _, frame_deskew = item
        path = scan_dir / f"frame_{frame_idx:04d}.jpg"

        img = Image.open(path)
        # JPEG draft() decodes at reduced resolution (1/2, 1/4, 1/8) during
        # DCT, avoiding a full-resolution decode. No-op for non-JPEG.
        if output_downsample > 1:
            img.draft("RGB", (frame_w, frame_h))
        img.load()
        if img.size[0] != frame_w or img.size[1] != frame_h:
            img = img.resize((frame_w, frame_h), Image.Resampling.LANCZOS)
        img = img.convert("RGBA")

        if flatfield is not None:
            img_arr = np.array(img, dtype=np.float32)
            for c in range(3):
                img_arr[:, :, c] = (img_arr[:, :, c] / flatfield[:, :, c]) * flatfield_mean
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr, mode="RGBA")

        # Apply per-frame deskew (rolling shutter correction).
        if frame_deskew != 0:
            img = deskew_image(img, frame_deskew)

        if blend:
            is_first = seq_i == 0
            is_last = seq_i == num_work - 1
            alpha = create_blend_alpha(frame_w, frame_h, blend_width_x, 0, is_first_x=is_first, is_last_x=is_last)
        else:
            alpha = Image.new("L", img.size, 128)

        # Mask deskew void: the shear leaves a triangle of edge-replicated
        # pixels that grows from 0 px at top to |shear| px at bottom.
        # Zero alpha there so they don't contaminate blending.
        if frame_deskew != 0:
            alpha_arr = np.array(alpha)
            abs_shear = abs(frame_deskew)
            # Build column index array and compare against void boundary per row
            ys = np.arange(frame_h)
            void_widths = (abs_shear * ys / frame_h + 1).astype(int)
            cols = np.arange(frame_w)
            if frame_deskew < 0:  # +X motion: void at left
                mask = cols[np.newaxis, :] < void_widths[:, np.newaxis]
            else:  # -X motion: void at right
                mask = cols[np.newaxis, :] >= (frame_w - void_widths[:, np.newaxis])
            alpha_arr[mask] = 0
            alpha = Image.fromarray(alpha_arr, mode="L")

        img.putalpha(alpha)

        return img

    # Load frames in parallel, composite sequentially
    if num_threads > 1 and num_work > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            loaded = list(pool.map(load_frame, work_items))
    else:
        loaded = [load_frame(item) for item in work_items]

    frames_placed = 0
    frame_rects = []  # (x, y=0, w, h) of each placed frame in row-local coords
    for img, (_, _, x_offset, _) in zip(loaded, work_items, strict=True):
        # Clamp to canvas bounds
        actual_x = x_offset
        actual_w = img.width
        if x_offset < 0:
            img = img.crop((-x_offset, 0, img.width, img.height))
            actual_x = 0
            actual_w = img.width
        if actual_x + img.width > canvas_w:
            img = img.crop((0, 0, canvas_w - actual_x, img.height))
            actual_w = img.width

        if img.width > 0 and img.height > 0:
            canvas.alpha_composite(img, (actual_x, 0))
            frame_rects.append((actual_x, 0, actual_w, img.height))
            frames_placed += 1

    return canvas, frames_placed, frame_rects


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
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=0,
        help="Hysteresis correction in µm (applied to -X rows)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop to stage rect in µm: x_min,x_max,y_min,y_max (same as scan_area --area-rect)",
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
    parser.add_argument("--row-labels", action="store_true", help="Draw row index+direction labels (e.g., 0+, 1-, 2+)")
    parser.add_argument("--draw-frame-outlines", action="store_true", help="Draw outline of each placed frame (debug)")
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=15,
        help="Savgol smoothing window for positions (0=disabled, default: 15)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-row progress output")
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
            print("  Run scripts/build_flatfield.py or specify --no-flatfield to skip correction.")
            return 1

        flatfield = np.load(flatfield_path).astype(np.float32)
        flatfield_mean = np.mean(flatfield)
        if not args.quiet:
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
                ff_channel = ff_channel.resize((frame_w, frame_h), Image.Resampling.LANCZOS)
                ff_resized[:, :, c] = np.array(ff_channel, dtype=np.float32)
            if not args.quiet:
                print(f"  Resized flatfield {ff_w}x{ff_h} -> {frame_w}x{frame_h}")
            flatfield = ff_resized
            flatfield_mean = np.mean(flatfield)

    if not args.quiet:
        print(f"Scan: {len(meta['rows'])} rows, {meta['frame_count']} frames")
        print(f"Processing: {len(rows)} rows")
        print(f"Calibration: {um_per_px:.3f} µm/px (downsample {downsample}x{args.downsample})")
        print(f"Frame: {frame_w}x{frame_h} px = {fov_width_um:.0f}x{fov_height_um:.0f} µm")
        overlap_pct = y_overlap_um / fov_height_um * 100
        print(f"Y step: {y_step_um:.0f} µm, Y overlap: {y_overlap_um:.0f} µm ({overlap_pct:.0f}%)")

    # Process each row to find CV regions and global X bounds
    if not args.quiet:
        print("\nAnalyzing rows...")
    row_results = []

    for row in rows:
        row_frames = meta["frames"][row["frame_start"] : row["frame_end"]]

        all_x = [f["x_start"] for f in row_frames]
        x_min = min(all_x)
        x_max = max(all_x)

        row_results.append({"row": row, "x_min": x_min, "x_max": x_max})

        if not args.quiet:
            print(f"  Row {row['row_idx']:2d}: {len(row_frames)} frames, X: {x_min:.0f} - {x_max:.0f} µm")

    # Find global X bounds: union of all rows (including accel zones)
    global_x_min = min(r["x_min"] for r in row_results)
    global_x_max = max(r["x_max"] for r in row_results)
    global_x_range = global_x_max - global_x_min + fov_width_um

    if not args.quiet:
        print(f"\nGlobal X bounds: {global_x_min:.0f} - {global_x_max:.0f} µm ({global_x_range:.0f} µm total)")

    # Calculate final canvas dimensions
    n_rows = len(rows)
    total_height_um = y_step_um * (n_rows - 1) + fov_height_um

    canvas_w = int(global_x_range / um_per_px)
    canvas_h = int(total_height_um / um_per_px)

    if not args.quiet:
        print(f"Final canvas: {canvas_w}x{canvas_h} px ({global_x_range:.0f}x{total_height_um:.0f} µm)")

    # Y blend width
    y_overlap_px = int(y_overlap_um / um_per_px)
    blend_width_y = y_overlap_px // 2 if not args.no_blend else 0

    # Create final canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    if not args.quiet:
        print("\nStitching rows...")
    all_frame_rects = []  # (x, y, w, h) in final canvas coords

    min_y = min(r["y_um"] for r in rows)
    max_y = max(r["y_um"] for r in rows)

    for result in row_results:
        row = result["row"]
        row_idx = row["row_idx"]

        # Stitch this row in global coordinates
        row_img, frames_placed, frame_rects = stitch_row_to_global(
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
            savgol_window=args.savgol_window,
            flatfield=flatfield,
            flatfield_mean=flatfield_mean,
            num_threads=args.threads,
        )

        # Calculate Y position for this row (min Y = top of image)
        row_y = row["y_um"]
        y_offset_um = row_y - min_y
        y_offset_px = int(y_offset_um / um_per_px)

        # Apply Y blending
        # is_first_y = don't fade top edge, is_last_y = don't fade bottom edge
        if not args.no_blend:
            is_at_top = row_y == min_y
            is_at_bottom = row_y == max_y

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

        # Collect frame rects in global canvas coords
        for fx, fy, fw, fh in frame_rects:
            all_frame_rects.append((fx, fy + y_offset_px, fw, fh))

        if not args.quiet:
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
        "y_min": min_y - fov_height_um / 2,
        "y_max": max_y + fov_height_um / 2,
    }

    # Apply crop to stage coordinate rect (x_min,x_max,y_min,y_max in µm)
    crop_left_px = 0
    crop_top_px = 0
    if args.crop:
        crop_parts = [float(x) for x in args.crop.split(",")]
        if len(crop_parts) != 4:
            print("Error: --crop must be x_min,x_max,y_min,y_max in µm")
            return 1
        crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop_parts
        crop_x_min, crop_x_max = min(crop_x_min, crop_x_max), max(crop_x_min, crop_x_max)
        crop_y_min, crop_y_max = min(crop_y_min, crop_y_max), max(crop_y_min, crop_y_max)

        # Clamp to actual stage bounds
        crop_x_min = max(crop_x_min, stage_bounds_um["x_min"])
        crop_x_max = min(crop_x_max, stage_bounds_um["x_max"])
        crop_y_min = max(crop_y_min, stage_bounds_um["y_min"])
        crop_y_max = min(crop_y_max, stage_bounds_um["y_max"])

        crop_left_px = int((crop_x_min - stage_bounds_um["x_min"]) / um_per_px)
        crop_top_px = int((crop_y_min - stage_bounds_um["y_min"]) / um_per_px)
        crop_right_px = int((crop_x_max - stage_bounds_um["x_min"]) / um_per_px)
        crop_bottom_px = int((crop_y_max - stage_bounds_um["y_min"]) / um_per_px)

        background = background.crop((crop_left_px, crop_top_px, crop_right_px, crop_bottom_px))

        stage_bounds_um["x_min"] = crop_x_min
        stage_bounds_um["x_max"] = crop_x_max
        stage_bounds_um["y_min"] = crop_y_min
        stage_bounds_um["y_max"] = crop_y_max

        if not args.quiet:
            print(
                f"\nCropped to stage rect: X=[{crop_x_min:.0f}, {crop_x_max:.0f}] "
                f"Y=[{crop_y_min:.0f}, {crop_y_max:.0f}] µm"
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
        if not args.quiet:
            print(f"Grid: {args.grid_spacing_um:.0f} µm spacing")

    # Draw row labels
    if args.row_labels:
        background = draw_row_labels(background, rows, um_per_px, stage_bounds_um)

    # Draw frame outlines
    if args.draw_frame_outlines:
        background = draw_frame_outlines(background, all_frame_rects, (crop_left_px, crop_top_px))
        if not args.quiet:
            print(f"Frame outlines: {len(all_frame_rects)} frames")

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

    # Compute average frame step from raw positions (median across all rows)
    all_steps = []
    for rr in row_results:
        row = rr["row"]
        row_frames = meta["frames"][row["frame_start"] : row["frame_end"]]
        positions = [f["x_start"] for f in row_frames]
        if len(positions) >= 2:
            steps = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
            all_steps.extend(steps)

    if all_steps:
        avg_step = float(np.median(all_steps))
        avg_overlap_pct = (fov_width_um - avg_step) / fov_width_um * 100
    else:
        avg_step = None
        avg_overlap_pct = None

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
            "avg_step_um": round(avg_step, 1) if avg_step else None,
            "avg_overlap_pct": round(avg_overlap_pct, 1) if avg_overlap_pct else None,
            "frame_width_um": round(fov_width_um, 1),
        },
    }

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(stitch_meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
