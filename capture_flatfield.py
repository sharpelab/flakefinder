"""Capture flatfield reference image for vignetting and white balance correction.

Captures from a blank substrate area and stores calibration data for use
by the scanner and stitcher.

Usage:
    # Interactive: position stage at blank area first
    python capture_flatfield.py --objective-mag 5x

    # With position: move to known blank spot
    python capture_flatfield.py --objective-mag 5x --position 50000,35000

    # Add notes about substrate
    python capture_flatfield.py --objective-mag 5x --notes "Blank SiO2/Si wafer edge"
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from flakefinder.leica.units import Nosepiece

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "calibration"


def parse_position(value: str) -> tuple[float, float]:
    """Parse X,Y position from string."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("Position must be X,Y (2 values)")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise ValueError("Position values must be numbers") from e


def resolve_objective_mag(objective_mag: str | None, objective_pos: int | None) -> float:
    """Resolve objective magnification from --objective-mag or --objective-pos.

    Uses Nosepiece.DEFAULT_MAGNIFICATIONS for position-to-mag lookup (no hardware needed).

    Returns the magnification as a float (e.g. 5.0, 20.0).
    """
    if objective_mag is not None:
        match = re.match(r"^(\d+(?:\.\d+)?)[xX]?$", objective_mag.strip())
        if not match:
            raise ValueError(f"Invalid objective format: {objective_mag}")
        mag = float(match.group(1))
        if mag not in Nosepiece.DEFAULT_MAGNIFICATIONS.values():
            valid = [f"{m}x" for m in sorted(Nosepiece.DEFAULT_MAGNIFICATIONS.values())]
            raise ValueError(f"Unknown magnification {mag}x. Available: {', '.join(valid)}")
        return mag

    if objective_pos is not None:
        mag = Nosepiece.DEFAULT_MAGNIFICATIONS.get(objective_pos)
        if mag is None:
            valid_pos = sorted(Nosepiece.DEFAULT_MAGNIFICATIONS.keys())
            raise ValueError(f"Position {objective_pos} not in magnification table. Valid: {valid_pos}")
        return mag

    raise ValueError("Either --objective-mag or --objective-pos is required")


def validate_flatfield(image: np.ndarray, max_cv: float = 0.15) -> tuple[bool, str]:
    """Validate that image looks like a blank substrate (uniform illumination).

    Checks that the coefficient of variation (std/mean) is reasonable.
    Vignetting causes some variation, but features/samples cause much more.

    Args:
        image: RGB image array (H, W, 3)
        max_cv: Maximum allowed coefficient of variation per channel

    Returns:
        (is_valid, message)
    """
    warnings = []

    for i, channel in enumerate(["Blue", "Green", "Red"]):
        ch_data = image[:, :, i].astype(np.float64)
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        cv = std / mean if mean > 0 else 0

        if cv > max_cv:
            warnings.append(f"{channel}: CV={cv:.3f} (>{max_cv:.2f})")

    # Check for very dark or saturated regions
    min_val = np.min(image)
    max_val = np.max(image)

    if min_val < 10:
        warnings.append(f"Very dark pixels detected (min={min_val})")
    if max_val > 250:
        warnings.append(f"Near-saturated pixels detected (max={max_val})")

    # Check for strong local gradients (edges)
    gray = np.mean(image, axis=2)
    # Simple gradient magnitude using Sobel-like kernel
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    max_gradient = max(np.max(gx), np.max(gy))

    if max_gradient > 30:
        warnings.append(f"Strong edges detected (max_gradient={max_gradient:.1f})")

    if warnings:
        return False, "Image may not be blank substrate:\n  - " + "\n  - ".join(warnings)
    return True, "Image appears to be uniform (suitable for flatfield)"


def compute_calibration(image: np.ndarray) -> dict:
    """Compute calibration values from flatfield image.

    Args:
        image: RGB image array (H, W, 3), uint8

    Returns:
        Dict with mean_bgr and reference_wb_bgr
    """
    # Compute mean of each channel (BGR order to match OpenCV convention)
    mean_b = float(np.mean(image[:, :, 0]))
    mean_g = float(np.mean(image[:, :, 1]))
    mean_r = float(np.mean(image[:, :, 2]))

    mean_bgr = [mean_b, mean_g, mean_r]

    # Compute white balance gains to neutralize (make R=G=B)
    # Normalize so green = 1.0 (green is typically reference)
    # Gains are what you multiply each channel by
    wb_b = mean_g / mean_b if mean_b > 0 else 1.0
    wb_g = 1.0
    wb_r = mean_g / mean_r if mean_r > 0 else 1.0

    reference_wb_bgr = [wb_b, wb_g, wb_r]

    return {
        "mean_bgr": mean_bgr,
        "reference_wb_bgr": reference_wb_bgr,
    }


def save_flatfield_16bit(image: np.ndarray, path: Path) -> None:
    """Save flatfield as 16-bit PNG, normalized so mean = 32768.

    This preserves the vignette pattern while using full 16-bit range.
    To apply: corrected = raw * (32768 / flatfield)

    Args:
        image: RGB image array (H, W, 3), uint8 or float
        path: Output path
    """
    # Convert to float for processing
    img_float = image.astype(np.float64)

    # Normalize so mean = 32768 (middle of 16-bit range)
    mean_val = np.mean(img_float)
    if mean_val > 0:
        img_normalized = img_float * (32768.0 / mean_val)
    else:
        img_normalized = img_float

    # Clip to valid 16-bit range
    img_16bit = np.clip(img_normalized, 0, 65535).astype(np.uint16)

    # Save as 16-bit PNG
    pil_img = Image.fromarray(img_16bit, mode="I;16")
    # For RGB, need to handle channels separately
    if len(img_16bit.shape) == 3:
        # PIL doesn't directly support 16-bit RGB, save as separate channels or use raw
        # Actually, let's save the per-channel normalized version
        # For simplicity, save as 3-channel 16-bit by scaling
        h, w, c = img_16bit.shape
        # Use mode 'I;16' for grayscale, for RGB we'll save raw numpy
        np.save(path.with_suffix(".npy"), img_16bit)
        # Also save a viewable 8-bit version
        img_8bit = (img_normalized / 256).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_8bit).save(path.with_suffix(".png"))
        return

    pil_img.save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Capture flatfield reference for vignetting/WB correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Position stage at blank area first, then capture
    python capture_flatfield.py --objective-mag 5x

    # Move to specific position first
    python capture_flatfield.py --objective-mag 5x --position 50000,35000

    # By turret position (position 3 = 20x)
    python capture_flatfield.py --objective-pos 3

    # Add notes about the substrate
    python capture_flatfield.py --objective-mag 5x --notes "Blank SiO2/Si edge"
""",
    )
    parser.add_argument(
        "--objective-mag", type=str, metavar="MAG", help="Objective by magnification (e.g., 5, 5x, 20, 2.5)"
    )
    parser.add_argument("--objective-pos", type=int, metavar="POS", help="Objective by turret position (1-6)")
    parser.add_argument(
        "--position", "-p", type=str, default=None, help="Stage position X,Y in µm (default: use current position)"
    )
    parser.add_argument("--frames", "-n", type=int, default=5, help="Number of frames to average (default: 5)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing calibration")
    parser.add_argument("--notes", type=str, default=None, help="Notes about substrate/conditions")
    parser.add_argument("--exposure", type=float, default=1.0, help="Exposure time in ms (default: 1.0)")
    parser.add_argument("--gain", type=float, default=None, help="Camera gain (default: not set)")
    parser.add_argument(
        "--wb", type=str, default="1.0,1.0,1.0", help="White balance as R,G,B gains (default: 1.0,1.0,1.0)"
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip uniformity validation (use with caution)")

    args = parser.parse_args()

    # Validate objective args
    if args.objective_mag is None and args.objective_pos is None:
        print("Error: Either --objective-mag or --objective-pos is required")
        return 1
    if args.objective_mag is not None and args.objective_pos is not None:
        print("Error: --objective-mag and --objective-pos are mutually exclusive")
        return 1

    # Parse inputs
    try:
        objective_mag = resolve_objective_mag(args.objective_mag, args.objective_pos)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    position = None
    if args.position:
        try:
            position = parse_position(args.position)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Check for existing calibration
    binning = 3  # Hardcoded for now
    flatfield_basename = f"flatfield_{int(objective_mag)}x_bin{binning}"
    flatfield_json = args.output / f"{flatfield_basename}.json"

    if flatfield_json.exists() and not args.force:
        print(f"Error: Calibration already exists: {flatfield_json}")
        print("  Use --force to overwrite.")
        return 1

    # Import microscope modules (slow, so do after arg validation)
    print("Connecting to microscope...")
    from flakefinder.leica import Microscope

    with Microscope() as scope:
        stage = scope.stage
        camera = scope.camera

        # Switch objective if needed
        if args.objective_mag is not None:
            if scope.switch_objective_mag(args.objective_mag):
                print(f"Switched objective to {scope.objective_mag}x")
                time.sleep(0.5)  # Let it settle
            else:
                print(f"Objective: already at {scope.objective_mag}x")
        elif args.objective_pos is not None:
            if scope.switch_objective_pos(args.objective_pos):
                print(f"Switched objective to position {args.objective_pos} ({scope.objective_mag}x)")
                time.sleep(0.5)
            else:
                print(f"Objective: already at {scope.objective_mag}x")

        # Lighting
        scope.light_on()
        lamp_intensity = scope.lamp.intensity
        print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({lamp_intensity}/{scope.lamp.max_intensity})")

        # Parse white balance
        wb_parts = args.wb.split(",")
        if len(wb_parts) != 3:
            print("Error: --wb must be R,G,B (e.g., '1.41,1.02,2.51')")
            return 1
        wb_r, wb_g, wb_b = float(wb_parts[0]), float(wb_parts[1]), float(wb_parts[2])

        # Configure camera for calibration capture
        camera.binning = 2  # 3x3 binning (index 2)
        camera.exposure_time = args.exposure / 1000.0
        camera.gamma = 1.0  # Linear
        camera.gain_rgb = (wb_r, wb_g, wb_b)
        camera.auto_brightness = False
        if args.gain is not None:
            camera.gain = args.gain

        frame_w, frame_h = camera.frame_size_px
        print(f"Camera: {camera.name}, {frame_w}x{frame_h} px, binning={binning}x{binning}")

        # TODO: Run autofocus before capturing flatfield to ensure sharp image

        # Move to position if specified
        if position:
            print(f"Moving to position ({position[0]:.0f}, {position[1]:.0f}) µm...")
            stage.x.move_to(position[0])
            stage.y.move_to(position[1])
        else:
            pos = stage.position_um
            print(f"Using current position: ({pos[0]:.0f}, {pos[1]:.0f}) µm")
            position = pos

        # Capture frames
        print(f"Capturing {args.frames} frames...")
        frames = []
        for i in range(args.frames):
            img = camera.capture()
            frames.append(img.astype(np.float64))
            print(f"  Frame {i + 1}/{args.frames}")
            time.sleep(0.05)  # Small delay between frames

        # Average frames
        averaged = np.mean(frames, axis=0).astype(np.uint8)
        print(f"Averaged {args.frames} frames: shape={averaged.shape}, dtype={averaged.dtype}")

        # Validate
        if not args.skip_validation:
            is_valid, msg = validate_flatfield(averaged)
            print(f"Validation: {msg}")
            if not is_valid:
                print("\nThe captured image may not be suitable for flatfield calibration.")
                print("Ensure the stage is positioned over a blank substrate area.")
                print("Use --skip-validation to bypass this check (not recommended).")
                return 1

        # Compute calibration values
        cal_values = compute_calibration(averaged)
        bgr = cal_values["mean_bgr"]
        wb = cal_values["reference_wb_bgr"]
        print(f"Mean BGR: [{bgr[0]:.1f}, {bgr[1]:.1f}, {bgr[2]:.1f}]")
        print(f"Reference WB (BGR): [{wb[0]:.3f}, {wb[1]:.3f}, {wb[2]:.3f}]")

        # Save flatfield image
        flatfield_path = args.output / flatfield_basename

        # Save as both .npy (for processing) and .png (for viewing)
        np.save(flatfield_path.with_suffix(".npy"), averaged)
        Image.fromarray(averaged).save(flatfield_path.with_suffix(".png"))
        print(f"Saved: {flatfield_path.with_suffix('.npy')}")
        print(f"Saved: {flatfield_path.with_suffix('.png')} (preview)")

        # Save per-flatfield metadata
        meta = {
            "objective_mag": objective_mag,
            "binning": binning,
            "lamp_intensity": lamp_intensity,
            "exposure_s": camera.exposure_time,
            "gain": args.gain,
            "white_balance_rgb": [wb_r, wb_g, wb_b],
            "frame_size_px": [frame_w, frame_h],
            "position_um": list(position),
            "mean_bgr": cal_values["mean_bgr"],
            "reference_wb_bgr": cal_values["reference_wb_bgr"],
            "frames_averaged": args.frames,
            "captured_at": datetime.now().isoformat(),
            "notes": args.notes,
        }

        with open(flatfield_json, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved: {flatfield_json}")

        print()
        print("=" * 50)
        print(f"Calibration complete for {flatfield_basename}")
        print()
        print("To use in scanner:")
        print(f"  python scan_area_v1.py --objective-mag {int(objective_mag)}x ...")
        print()
        print("The stitcher will automatically apply flatfield correction.")

        return 0


if __name__ == "__main__":
    exit(main())
