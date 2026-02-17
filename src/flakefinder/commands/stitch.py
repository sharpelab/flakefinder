"""Multi-row position-based stitch for snake scan patterns.

Usage:
    uv run python commands/stitch.py scans/overview_5x
    uv run python commands/stitch.py scans/overview_5x --downsample 4
    uv run python commands/stitch.py scans/overview_5x --no-flatfield -q
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy.signal import savgol_filter

from flakefinder.data_utils import load_scan_meta
from flakefinder.scan_utils import CALIBRATION_DIR, apply_flatfield, parse_area_rect
from flakefinder.types import AreaRect, AreaRectI, Point2I, ScanLineMeta, ScanMeta


@dataclass
class StitchResult:
    """Output from stitch run()."""

    output_path: Path


@dataclass
class RowStitchResult:
    """Output from stitching a single row (raw accumulators, not normalized)."""

    color_sum: np.ndarray  # float32 (H, W, 3) — weighted pixel sums
    weight_sum: np.ndarray  # float32 (H, W) — weight sums
    frames_placed: int
    frame_rects: list[AreaRectI]


def create_blend_alpha(
    width: int,
    height: int,
    blend_width_x: int,
    blend_width_y: int = 0,
    is_first_x: bool = False,
    is_last_x: bool = False,
    is_first_y: bool = False,
    is_last_y: bool = False,
) -> Image.Image:
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


def deskew_image(
    img: Image.Image,
    shear_px: float,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
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
        resample=resample,
        fillcolor=(0,) * len(img.getbands()),
    )

    # Crop back to original size
    return result.crop((x_crop, 0, x_crop + width, height))


def draw_grid(
    background: Image.Image,
    stage_bounds_um: AreaRect,
    um_per_px: float,
    spacing_um: float,
    line_color: str,
    line_width: int,
) -> Image.Image:
    """Draw stage-coordinate grid lines and mm labels on the stitched image."""
    w, h = background.size
    x_min = stage_bounds_um.x_min
    x_max = stage_bounds_um.x_max
    y_min = stage_bounds_um.y_min
    y_max = stage_bounds_um.y_max

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


def draw_row_labels(
    background: Image.Image,
    rows: list[ScanLineMeta],
    um_per_px: float,
    stage_bounds_um: AreaRect,
) -> Image.Image:
    """Draw row index + direction labels (e.g., '0+', '1-') on the left edge."""
    w, h = background.size
    y_min_stage = stage_bounds_um.y_min

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, min(w, h) // 200)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    pad = 6
    for row in rows:
        row_idx = row["line_idx"]
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


def draw_frame_outlines(
    background: Image.Image,
    frame_rects: list[AreaRectI],
    crop_offset: Point2I | None = None,
) -> Image.Image:
    """Draw outline rectangles for each placed frame."""
    if crop_offset is None:
        crop_offset = Point2I(0, 0)
    w, h = background.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for rect in frame_rects:
        x0 = rect.x_min - crop_offset.x
        y0 = rect.y_min - crop_offset.y
        x1 = rect.x_max - crop_offset.x
        y1 = rect.y_max - crop_offset.y
        # Skip if fully outside
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 100), width=1)

    result = Image.alpha_composite(background.convert("RGBA"), overlay)
    return result.convert("RGB")


def stitch_row_to_global(
    meta: ScanMeta,
    row: ScanLineMeta,
    frame_w: int,
    frame_h: int,
    um_per_px: float,
    output_downsample: int,
    scan_dir: Path,
    global_x_min: float,
    global_x_max: float,
    *,
    blend: bool = True,
    deskew: bool = True,
    hysteresis_um: float = 0,
    savgol_window: int = 0,
    flatfield: np.ndarray | None = None,
    num_threads: int = 1,
    deskew_resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> RowStitchResult:
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
    assert fov_width_um is not None

    all_row_frames = frames[row["frame_start"] : row["frame_end"]]
    direction = row["direction"]

    # Trim stationary frames at row boundaries (stage parked, camera still capturing)
    raw_x = np.array([f["x_um"] for f in all_row_frames])
    deltas = np.diff(raw_x)
    stationary_thresh = 2.0  # µm — frames closer than this are "stationary"

    # Find first/last moving frames
    moving = np.abs(deltas) > stationary_thresh
    if np.any(moving):
        first_moving = int(np.argmax(moving))
        last_moving = int(len(moving) - 1 - np.argmax(moving[::-1])) + 2
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
    raw_positions = np.array([f["x_um"] for f in row_frames])
    raw_times = np.array([f["t_capture"] + f["capture_duration_s"] / 2 for f in row_frames])
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

    # Weighted-average accumulators (per-row X blending)
    color_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Use all frames (including accel zones)
    num_frames = len(row_frames)

    # Composite in capture order: later-captured frames go on top.
    # This is direction-independent — stale/early frames always end up
    # underneath regardless of +X/-X scan direction.
    indices = list(range(num_frames))

    # Per-frame velocity for deskew from scan_meta (spline derivative
    # on the full position stream — accurate even at row edges).
    frame_velocities = np.array([row_frames[i]["x_vel_um_s"] for i in range(n)])

    if deskew:
        readout_time = meta["camera"]["readout_time_s"]

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

    # Pre-compute blend weight masks (float32, 0-1).
    # "first_x" = spatially leftmost (no left ramp), "last_x" = rightmost (no right ramp).
    if blend and num_work > 0:

        def _wt(first: bool = False, last: bool = False) -> np.ndarray:
            return (
                np.array(
                    create_blend_alpha(frame_w, frame_h, blend_width_x, 0, is_first_x=first, is_last_x=last),
                    dtype=np.float32,
                )
                / 255.0
            )

        wt_first_x = _wt(first=True, last=(num_work == 1))
        wt_last_x = _wt(last=True)
        wt_interior = _wt()
    else:
        wt_first_x = wt_last_x = wt_interior = np.ones((frame_h, frame_w), dtype=np.float32)

    # Map capture-order sequence to spatial position for weight selection.
    # +X: capture order = spatial order; -X: reversed.
    if direction > 0:
        _leftmost_seq = 0
        _rightmost_seq = num_work - 1
    else:
        _leftmost_seq = num_work - 1
        _rightmost_seq = 0

    def load_frame(item):
        """Load, resize, flatfield-correct, and deskew a single frame.

        Returns (rgb, weight): float32 (H,W,3) and float32 (H,W).
        """
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

        if flatfield is not None:
            img_arr = np.array(img)
            img_arr = apply_flatfield(img_arr[:, :, :3], flatfield)
            img = Image.fromarray(img_arr, mode="RGB")

        if frame_deskew != 0:
            img = deskew_image(img, frame_deskew, resample=deskew_resample)

        rgb = np.array(img, dtype=np.float32)

        # Select blend weight by spatial position (not capture order)
        if seq_i == _leftmost_seq:
            weight = wt_first_x.copy()
        elif seq_i == _rightmost_seq:
            weight = wt_last_x.copy()
        else:
            weight = wt_interior.copy()

        # Zero weight in deskew void region
        if frame_deskew != 0:
            abs_shear = abs(frame_deskew)
            ys = np.arange(frame_h)
            void_widths = (abs_shear * ys / frame_h + 1).astype(int)
            cols = np.arange(frame_w)
            if frame_deskew < 0:  # +X motion: void at left
                void = cols[np.newaxis, :] < void_widths[:, np.newaxis]
            else:  # -X motion: void at right
                void = cols[np.newaxis, :] >= (frame_w - void_widths[:, np.newaxis])
            weight[void] = 0.0

        return rgb, weight

    # Load frames in parallel, accumulate sequentially
    if num_threads > 1 and num_work > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            loaded = list(pool.map(load_frame, work_items))
    else:
        loaded = [load_frame(item) for item in work_items]

    frames_placed = 0
    frame_rects: list[AreaRectI] = []
    for (rgb, weight), (_, _, x_offset, _) in zip(loaded, work_items, strict=True):
        # Clamp to canvas bounds
        src_x0 = max(0, -x_offset)
        dst_x0 = max(0, x_offset)
        src_x1 = min(frame_w, canvas_w - x_offset)

        if src_x1 > src_x0:
            w = src_x1 - src_x0
            wgt_s = weight[:, src_x0:src_x1]
            color_sum[:, dst_x0 : dst_x0 + w, :] += rgb[:, src_x0:src_x1, :] * wgt_s[:, :, np.newaxis]
            weight_sum[:, dst_x0 : dst_x0 + w] += wgt_s
            frame_rects.append(AreaRectI(dst_x0, dst_x0 + w - 1, 0, canvas_h - 1))
            frames_placed += 1

    return RowStitchResult(
        color_sum=color_sum, weight_sum=weight_sum, frames_placed=frames_placed, frame_rects=frame_rects
    )


def run(
    scan_dir: Path,
    *,
    deskew: bool = True,
    blend: bool = True,
    downsample: int = 1,
    rows_range: str | None = None,
    hysteresis: float = 0,
    crop: AreaRect | None = None,
    flatfield_path: Path | None = None,
    no_flatfield: bool = False,
    threads: int = 8,
    bg: str = "black",
    grid_spacing_um: float = 0,
    grid_line_color: str = "#FFFFFF50",
    grid_line_width: int = 1,
    row_labels: bool = False,
    show_frame_outlines: bool = False,
    savgol_window: int = 0,
    bicubic_deskew: bool = False,
    quiet: bool = False,
    output: str | None = None,
) -> StitchResult:
    """Stitch multi-row snake scan into 2D image.

    Returns:
        StitchResult with path to the saved stitch image.

    Raises:
        ValueError: On invalid input (missing rows, missing flatfield).
    """
    bg_color = ImageColor.getrgb(bg)

    start_time = time.perf_counter()

    # Load metadata
    meta = load_scan_meta(scan_dir)

    rows = meta["lines"]
    optics = meta["optics"]
    scan_downsample = meta["downsample"]

    # Load flatfield for vignetting correction
    flatfield = None

    if no_flatfield:
        if not quiet:
            print("Flatfield: disabled (--no-flatfield)")
    else:
        if not flatfield_path:
            # Auto-load based on objective and binning
            obj_mag = optics["objective_mag"]
            binning = meta["camera"]["binning"]
            flatfield_path = CALIBRATION_DIR / f"flatfield_{obj_mag}x_bin{binning}.npy"

        if not flatfield_path.exists():
            raise ValueError(
                f"Flatfield not found: {flatfield_path}\n"
                "  Run scripts/build_flatfield.py or specify --no-flatfield to skip correction."
            )

        flatfield = np.load(flatfield_path).astype(np.float32)
        if not quiet:
            print(f"Flatfield: {flatfield_path}")

    # Parse row range if specified
    if rows_range:
        if "-" in rows_range:
            start, end = map(int, rows_range.split("-"))
            rows = [r for r in rows if start <= r["line_idx"] <= end]
        else:
            row_idx = int(rows_range)
            rows = [r for r in rows if r["line_idx"] == row_idx]

    # Calibration — assert optics fields are populated (always true for scan output)
    assert optics["sample_pixel_x_um"] is not None
    assert optics["frame_width_um"] is not None
    assert optics["frame_height_um"] is not None
    um_per_px = optics["sample_pixel_x_um"] * scan_downsample * downsample
    fov_width_um = optics["frame_width_um"]
    fov_height_um = optics["frame_height_um"]
    y_step_um = meta["y_step_um"]
    y_overlap_um = fov_height_um - y_step_um

    # Get frame dimensions (on-disk frames are already scan-downsampled)
    first_img = Image.open(scan_dir / "frame_0000.jpg")
    frame_w, frame_h = first_img.size
    if downsample > 1:
        frame_w //= downsample
        frame_h //= downsample

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
            if not quiet:
                print(f"  Resized flatfield {ff_w}x{ff_h} -> {frame_w}x{frame_h}")
            flatfield = ff_resized

    if not quiet:
        print(f"Scan: {len(meta['lines'])} rows, {meta['frame_count']} frames")
        print(f"Processing: {len(rows)} rows")
        ds_total = scan_downsample * downsample
        print(f"Calibration: {um_per_px:.3f} µm/px (scan {scan_downsample}x, stitch {downsample}x, total {ds_total}x)")
        print(f"Frame: {frame_w}x{frame_h} px = {fov_width_um:.0f}x{fov_height_um:.0f} µm")
        overlap_pct = y_overlap_um / fov_height_um * 100
        print(f"Y step: {y_step_um:.0f} µm, Y overlap: {y_overlap_um:.0f} µm ({overlap_pct:.0f}%)")

    # Process each row to find CV regions and global X bounds
    if not quiet:
        print("\nAnalyzing rows...")
    row_results = []

    for row in rows:
        row_frames = meta["frames"][row["frame_start"] : row["frame_end"]]

        all_x = [f["x_um"] for f in row_frames]
        x_min = min(all_x)
        x_max = max(all_x)

        row_results.append({"line": row, "x_min": x_min, "x_max": x_max})

        if not quiet:
            print(f"  Row {row['line_idx']:2d}: {len(row_frames)} frames, X: {x_min:.0f} - {x_max:.0f} µm")

    # Find global X bounds: union of all rows (including accel zones)
    global_x_min = min(r["x_min"] for r in row_results)
    global_x_max = max(r["x_max"] for r in row_results)
    global_x_range = global_x_max - global_x_min + fov_width_um

    if not quiet:
        print(f"\nGlobal X bounds: {global_x_min:.0f} - {global_x_max:.0f} µm ({global_x_range:.0f} µm total)")

    # Calculate final canvas dimensions
    n_rows = len(rows)
    total_height_um = y_step_um * (n_rows - 1) + fov_height_um

    canvas_w = int(global_x_range / um_per_px)
    canvas_h = int(total_height_um / um_per_px)

    if not quiet:
        print(f"Final canvas: {canvas_w}x{canvas_h} px ({global_x_range:.0f}x{total_height_um:.0f} µm)")

    # Y blend width
    y_overlap_px = int(y_overlap_um / um_per_px)
    blend_width_y = y_overlap_px // 2 if blend else 0

    # Global weighted-average accumulators
    global_color = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    global_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    if not quiet:
        print("\nStitching rows...")
    all_frame_rects: list[AreaRectI] = []

    min_y = min(r["y_um"] for r in rows)
    max_y = max(r["y_um"] for r in rows)

    for result in row_results:
        row = result["line"]
        row_idx = row["line_idx"]

        # Stitch this row in global coordinates
        row_stitch = stitch_row_to_global(
            meta,
            row,
            frame_w,
            frame_h,
            um_per_px,
            downsample,
            scan_dir,
            global_x_min,
            global_x_max,
            blend=blend,
            deskew=deskew,
            hysteresis_um=hysteresis,
            savgol_window=savgol_window,
            flatfield=flatfield,
            num_threads=threads,
            deskew_resample=Image.Resampling.BICUBIC if bicubic_deskew else Image.Resampling.BILINEAR,
        )

        # Calculate Y position for this row (min Y = top of image)
        row_y = row["y_um"]
        y_offset_um = row_y - min_y
        y_offset_px = int(y_offset_um / um_per_px)
        row_h = row_stitch.color_sum.shape[0]
        row_w = min(row_stitch.color_sum.shape[1], canvas_w)

        # Compute Y blend weight (1D array, varies only along Y)
        if blend and blend_width_y > 0:
            is_at_top = row_y == min_y
            is_at_bottom = row_y == max_y
            y_weight = np.ones(row_h, dtype=np.float32)
            if not is_at_top:
                for y in range(min(blend_width_y, row_h)):
                    y_weight[y] = y / blend_width_y
            if not is_at_bottom:
                for y in range(max(0, row_h - blend_width_y), row_h):
                    y_weight[y] = (row_h - 1 - y) / blend_width_y
        else:
            y_weight = None

        # Accumulate into global arrays with Y weight
        y1 = min(y_offset_px + row_h, canvas_h)
        src_h = y1 - y_offset_px
        if y_weight is not None:
            yw = y_weight[:src_h, np.newaxis]  # (H, 1) for broadcasting
            global_color[y_offset_px:y1, :row_w, :] += row_stitch.color_sum[:src_h, :row_w, :] * yw[:, :, np.newaxis]
            global_weight[y_offset_px:y1, :row_w] += row_stitch.weight_sum[:src_h, :row_w] * yw
        else:
            global_color[y_offset_px:y1, :row_w, :] += row_stitch.color_sum[:src_h, :row_w, :]
            global_weight[y_offset_px:y1, :row_w] += row_stitch.weight_sum[:src_h, :row_w]

        # Collect frame rects in global canvas coords
        for rect in row_stitch.frame_rects:
            all_frame_rects.append(
                AreaRectI(rect.x_min, rect.x_max, rect.y_min + y_offset_px, rect.y_max + y_offset_px)
            )

        if not quiet:
            print(f"  Row {row_idx}: {row_stitch.frames_placed} frames, y={y_offset_px} px")

    # Single global normalization → RGB
    covered = global_weight > 0
    np.divide(global_color[:, :, 0], global_weight, out=global_color[:, :, 0], where=covered)
    np.divide(global_color[:, :, 1], global_weight, out=global_color[:, :, 1], where=covered)
    np.divide(global_color[:, :, 2], global_weight, out=global_color[:, :, 2], where=covered)
    np.clip(global_color, 0, 255, out=global_color)
    result_rgb = global_color.astype(np.uint8)
    result_rgb[~covered] = bg_color[:3] if len(bg_color) >= 3 else 0
    background = Image.fromarray(result_rgb, mode="RGB")

    # Stage bounds before crop
    # Note: reported positions are frame CENTERS, so coverage extends ±FOV/2
    # - X: global_x_min - fov/2 to global_x_max + fov/2
    # - Y: last row's y - fov/2 to first row's y + fov/2
    stage_bounds_um = AreaRect(
        x_min=global_x_min - fov_width_um / 2,
        x_max=global_x_max + fov_width_um / 2,
        y_min=min_y - fov_height_um / 2,
        y_max=max_y + fov_height_um / 2,
    )

    # Apply crop to stage coordinate rect (already parsed by argparse type=parse_area_rect)
    crop_offset = Point2I(0, 0)
    if crop:
        crop_x_min, crop_x_max, crop_y_min, crop_y_max = crop

        # Clamp to actual stage bounds
        crop_x_min = max(crop_x_min, stage_bounds_um.x_min)
        crop_x_max = min(crop_x_max, stage_bounds_um.x_max)
        crop_y_min = max(crop_y_min, stage_bounds_um.y_min)
        crop_y_max = min(crop_y_max, stage_bounds_um.y_max)

        crop_left_px = int((crop_x_min - stage_bounds_um.x_min) / um_per_px)
        crop_top_px = int((crop_y_min - stage_bounds_um.y_min) / um_per_px)
        crop_right_px = int((crop_x_max - stage_bounds_um.x_min) / um_per_px)
        crop_bottom_px = int((crop_y_max - stage_bounds_um.y_min) / um_per_px)

        background = background.crop((crop_left_px, crop_top_px, crop_right_px, crop_bottom_px))
        crop_offset = Point2I(crop_left_px, crop_top_px)

        stage_bounds_um = AreaRect(crop_x_min, crop_x_max, crop_y_min, crop_y_max)

        if not quiet:
            print(
                f"\nCropped to stage rect: X=[{crop_x_min:.0f}, {crop_x_max:.0f}] "
                f"Y=[{crop_y_min:.0f}, {crop_y_max:.0f}] µm"
            )

    # Draw grid overlay
    if grid_spacing_um > 0:
        background = draw_grid(
            background,
            stage_bounds_um,
            um_per_px,
            grid_spacing_um,
            grid_line_color,
            grid_line_width,
        )
        if not quiet:
            print(f"Grid: {grid_spacing_um:.0f} µm spacing")

    # Draw row labels
    if row_labels:
        background = draw_row_labels(background, rows, um_per_px, stage_bounds_um)

    # Draw frame outlines
    if show_frame_outlines:
        background = draw_frame_outlines(background, all_frame_rects, crop_offset)
        if not quiet:
            print(f"Frame outlines: {len(all_frame_rects)} frames")

    # Save image
    if output:
        out_path = Path(output)
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
        row = rr["line"]
        row_frames = meta["frames"][row["frame_start"] : row["frame_end"]]
        positions = [f["x_um"] for f in row_frames]
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
        "command": sys.argv,
        "duration_s": round(duration_s, 2),
        "image_file": out_path.name,
        "image_size_px": [background.width, background.height],
        "stage_bounds_um": stage_bounds_um._asdict(),
        "scale_um_per_px": um_per_px,
        "source_scan": f"{scan_dir.name}/scan_meta.json",
        "objective_mag": optics["objective_mag"],
        "downsample": scan_downsample * downsample,
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

    return StitchResult(output_path=out_path)


def _build_parser():
    parser = argparse.ArgumentParser(description="Stitch multi-row snake scan into 2D image")
    parser.add_argument("scan_dir", type=Path, help="Scan directory")
    parser.add_argument("--no-deskew", action="store_true", help="Disable rolling shutter deskew")
    parser.add_argument("--no-blend", action="store_true", help="Disable gradient blending")
    parser.add_argument("--downsample", type=int, default=1, help="Additional downsample factor")
    parser.add_argument("--rows", type=str, default=None, help="Row range (e.g., '0-5' or '10')")
    parser.add_argument("--hysteresis", type=float, default=0, help="Hysteresis correction in µm (-X rows)")
    parser.add_argument(
        "--crop", type=parse_area_rect, default=None, help="Crop to stage rect: x_min,x_max,y_min,y_max"
    )
    parser.add_argument("--flatfield", type=Path, default=None, help="Path to flatfield .npy file")
    parser.add_argument("--no-flatfield", action="store_true", help="Disable flatfield correction")
    parser.add_argument("--threads", type=int, default=8, help="Threads for parallel frame loading")
    parser.add_argument("--bg", type=str, default="black", help="Background color")
    parser.add_argument("--grid-spacing-um", type=float, default=0, help="Grid line spacing in µm")
    parser.add_argument("--grid-line-color", type=str, default="#FFFFFF50", help="Grid line color")
    parser.add_argument("--grid-line-width", type=int, default=1, help="Grid line width")
    parser.add_argument("--row-labels", action="store_true", help="Draw row labels")
    parser.add_argument(
        "--frame-outlines",
        action="store_true",
        dest="show_frame_outlines",
        help="Draw frame outlines",
    )
    parser.add_argument("--savgol-window", type=int, default=0, help="Savgol smoothing window (0=off)")
    parser.add_argument("--bicubic", action="store_true", help="Use bicubic deskew (slower, slightly sharper)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output filename")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        run(
            args.scan_dir,
            deskew=not args.no_deskew,
            blend=not args.no_blend,
            downsample=args.downsample,
            rows_range=args.rows,
            hysteresis=args.hysteresis,
            crop=args.crop,
            flatfield_path=args.flatfield,
            no_flatfield=args.no_flatfield,
            threads=args.threads,
            bg=args.bg,
            grid_spacing_um=args.grid_spacing_um,
            grid_line_color=args.grid_line_color,
            grid_line_width=args.grid_line_width,
            row_labels=args.row_labels,
            show_frame_outlines=args.show_frame_outlines,
            savgol_window=args.savgol_window,
            bicubic_deskew=args.bicubic,
            quiet=args.quiet,
            output=args.output,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
