"""Position-based stitch with 50% alpha overlap and linear position smoothing."""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCAN_DIR = Path(__file__).parent / "test_scan"


def deskew_image(img, shear_px):
    """
    Apply horizontal shear to correct rolling shutter skew.

    For +X scan with top-to-bottom readout:
    - Bottom of image was captured later, so it's shifted right
    - We need to shift bottom left to deskew (negative shear)

    shear_px: pixels to shift bottom row (positive = shift right)
    """
    width, height = img.size

    # Expand canvas to fit sheared image
    new_width = width + abs(int(shear_px))

    # PIL transform uses inverse mapping (output -> input)
    # x_src = a*x_dst + b*y_dst + c
    # y_src = d*x_dst + e*y_dst + f
    #
    # To shift bottom left by shear_px:
    # At y=0: x_src = x_dst + offset
    # At y=height: x_src = x_dst + offset + shear_px
    # So: b = shear_px / height

    # Offset to keep content in frame
    if shear_px > 0:
        # Bottom shifts right in source, so we offset to keep left edge
        x_offset = 0
    else:
        # Bottom shifts left in source, need offset to not clip
        x_offset = abs(shear_px)

    # Affine coefficients: (a, b, c, d, e, f)
    # x_src = a*x + b*y + c
    # y_src = d*x + e*y + f
    a, b, c = 1, shear_px / height, -x_offset
    d, e, f = 0, 1, 0

    return img.transform(
        (new_width, height),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255, 0) if img.mode == 'RGBA' else (255, 255, 255)
    )


def find_constant_velocity_frames(meta):
    """
    Find frames in the constant-velocity portion of the scan.
    Returns (start_idx, end_idx, velocity) for the CV region.
    """
    frames = meta["frames"]

    # Get frame-to-frame deltas
    deltas = [frames[i+1]["x_start"] - frames[i]["x_start"] for i in range(len(frames)-1)]

    # Find median delta (robust estimate of constant velocity spacing)
    sorted_deltas = sorted(deltas)
    median_delta = sorted_deltas[len(sorted_deltas)//2]

    # Frames are "constant velocity" if delta is within 20% of median
    tolerance = 0.2
    cv_mask = [abs(d - median_delta) / median_delta < tolerance for d in deltas]

    # Find first and last CV frame
    first_cv = next(i for i, m in enumerate(cv_mask) if m)
    last_cv = len(cv_mask) - 1 - next(i for i, m in enumerate(reversed(cv_mask)) if m)

    # Estimate velocity from median delta and frame timing
    avg_dt = sum(frames[i+1]["t_start"] - frames[i]["t_start"]
                 for i in range(first_cv, last_cv)) / (last_cv - first_cv)
    velocity = median_delta / avg_dt

    return first_cv, last_cv + 1, velocity


def main():
    parser = argparse.ArgumentParser(description="Stitch scan frames with position smoothing")
    parser.add_argument("--deskew", action="store_true", help="Apply rolling shutter deskew correction")
    args = parser.parse_args()

    # Load metadata
    with open(SCAN_DIR / "scan_meta.json") as f:
        meta = json.load(f)

    frames = meta["frames"]
    frame_count = meta["frame_count"]
    downsample = meta["downsample"]

    # Get calibration from metadata
    optics = meta["optics"]
    um_per_px = optics["sample_pixel_x_um"] * downsample
    fov_width_um = optics["frame_width_um"]
    fov_height_um = optics["frame_height_um"]

    # Get frame dimensions
    first_img = Image.open(SCAN_DIR / "frame_0000.jpg")
    frame_w, frame_h = first_img.size

    print(f"Calibration: {um_per_px:.3f} µm/px (from metadata)")
    print(f"Frame size: {frame_w}x{frame_h} px = {fov_width_um:.0f}x{fov_height_um:.0f} µm")

    # Find constant-velocity portion
    cv_start, cv_end, velocity = find_constant_velocity_frames(meta)
    cv_frames = frames[cv_start:cv_end]
    print(f"Constant velocity: frames {cv_start}-{cv_end-1} ({len(cv_frames)} frames), v={velocity/1000:.1f} mm/s")

    # Fit linear model to CV frames
    times = [(f["t_start"] + f["t_end"]) / 2 for f in cv_frames]
    positions = [f["x_start"] for f in cv_frames]

    n = len(times)
    sum_t = sum(times)
    sum_x = sum(positions)
    sum_tt = sum(t*t for t in times)
    sum_tx = sum(t*x for t, x in zip(times, positions))

    denom = n * sum_tt - sum_t * sum_t
    fit_velocity = (n * sum_tx - sum_t * sum_x) / denom
    fit_intercept = (sum_x - fit_velocity * sum_t) / n

    # Compute smoothed positions using linear fit
    smoothed_positions = [fit_velocity * t + fit_intercept for t in times]

    # Residual stats
    residuals = [x - sx for x, sx in zip(positions, smoothed_positions)]
    max_residual = max(abs(r) for r in residuals)
    rms_residual = (sum(r*r for r in residuals) / len(residuals)) ** 0.5
    print(f"Linear fit residuals: max={max_residual:.1f} µm, RMS={rms_residual:.1f} µm")

    # X range from smoothed positions
    x_min = min(smoothed_positions)
    x_max = max(smoothed_positions)

    print(f"X range: {x_min:.0f} - {x_max:.0f} µm")

    # Calculate canvas size
    # Canvas needs to fit from first frame start to last frame end (plus frame width)
    canvas_w_um = (x_max - x_min) + fov_width_um
    canvas_w = int(canvas_w_um / um_per_px)
    canvas_h = frame_h

    print(f"Canvas: {canvas_w}x{canvas_h} px ({canvas_w_um:.0f} µm wide)")
    print(f"Stitching {len(cv_frames)} frames with 50% alpha...")

    # Create RGBA canvas (transparent background)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    for i, (frame, x_um) in enumerate(zip(cv_frames, smoothed_positions)):
        frame_idx = cv_start + i
        img = Image.open(SCAN_DIR / f"frame_{frame_idx:04d}.jpg").convert("RGBA")

        # Set 50% alpha
        alpha = img.split()[3] if img.mode == 'RGBA' else Image.new('L', img.size, 255)
        alpha = alpha.point(lambda x: 128)  # 50% alpha
        img.putalpha(alpha)

        # Calculate X offset from smoothed position
        x_offset = int((x_um - x_min) / um_per_px)

        # Composite onto canvas
        canvas.alpha_composite(img, (x_offset, 0))

        # Draw frame number in center
        draw = ImageDraw.Draw(canvas)
        text = str(frame_idx)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x_offset + frame_w // 2 - tw // 2
        ty = frame_h // 2 - th // 2

        # White text with black outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + dx, ty + dy), text, fill=(0, 0, 0, 255), font=font)
        draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cv_frames)}...")

    # Apply deskew if requested
    if args.deskew:
        # Calculate shear from velocity and readout time
        # For +X scan: bottom row captured later = shifted right = positive shear in image
        # To correct: shift bottom left = negative shear_px
        readout_time = meta["camera"]["readout_time_s"]
        shear_um = fit_velocity * readout_time  # µm of travel during readout
        shear_px = shear_um / um_per_px  # pixels of skew

        print(f"\nDeskew: {shear_um:.1f} µm = {shear_px:.1f} px (v={fit_velocity/1000:.1f} mm/s, readout={readout_time*1000:.1f} ms)")

        # Negative because we're correcting the skew (shifting bottom left)
        canvas = deskew_image(canvas, -shear_px)
        print(f"Deskewed canvas: {canvas.size[0]}x{canvas.size[1]} px")

    # Convert to RGB for saving
    # Use white background
    background = Image.new("RGB", canvas.size, (255, 255, 255))
    background.paste(canvas, mask=canvas.split()[3])

    out_path = SCAN_DIR.parent / "position_stitch.png"
    background.save(out_path)
    print(f"Saved to {out_path}")

    # Stats
    avg_delta = sum(smoothed_positions[i+1] - smoothed_positions[i]
                    for i in range(len(smoothed_positions)-1)) / (len(smoothed_positions)-1)
    overlap_um = fov_width_um - avg_delta
    overlap_pct = overlap_um / fov_width_um * 100
    print(f"\nOverlap: {overlap_um:.0f} µm ({overlap_pct:.0f}%)")
    print(f"Frame spacing: {avg_delta:.1f} µm (from linear fit)")

    return out_path

if __name__ == "__main__":
    main()
