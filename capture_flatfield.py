"""Capture flatfield reference image for vignetting and white balance correction.

Captures from a blank substrate area and stores calibration data for use
by the scanner and stitcher.

Usage:
    # Interactive: position stage at blank area first
    python capture_flatfield.py --objective 5x

    # With position: move to known blank spot
    python capture_flatfield.py --objective 5x --position 50000,35000

    # Add notes about substrate
    python capture_flatfield.py --objective 5x --notes "Blank SiO2/Si wafer edge"
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "calibration"


def parse_position(value: str) -> tuple[float, float]:
    """Parse X,Y position from string."""
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("Position must be X,Y (2 values)")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError("Position values must be numbers")


def parse_objective(value: str) -> float:
    """Parse objective magnification from string like '5x', '20X', or '50'."""
    match = re.match(r"^(\d+(?:\.\d+)?)[xX]?$", value.strip())
    if not match:
        raise ValueError(f"Invalid objective format: {value}")
    return float(match.group(1))


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


def load_calibration_json(path: Path) -> dict:
    """Load existing calibration.json or return empty dict."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_calibration_json(path: Path, data: dict) -> None:
    """Save calibration.json with pretty formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Capture flatfield reference for vignetting/WB correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Position stage at blank area first, then capture
    python capture_flatfield.py --objective 5x

    # Move to specific position first
    python capture_flatfield.py --objective 5x --position 50000,35000

    # Add notes about the substrate
    python capture_flatfield.py --objective 5x --notes "Blank SiO2/Si edge"
"""
    )
    parser.add_argument("--objective", "-obj", required=True,
                        help="Objective magnification (e.g., 5x, 10, 20X)")
    parser.add_argument("--position", "-p", type=str, default=None,
                        help="Stage position X,Y in µm (default: use current position)")
    parser.add_argument("--frames", "-n", type=int, default=5,
                        help="Number of frames to average (default: 5)")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing calibration")
    parser.add_argument("--notes", type=str, default=None,
                        help="Notes about substrate/conditions")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip uniformity validation (use with caution)")

    args = parser.parse_args()

    # Parse inputs
    try:
        objective_mag = parse_objective(args.objective)
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
    cal_key = f"{int(objective_mag)}x_bin{binning}"
    cal_json_path = args.output / "calibration.json"
    existing_cal = load_calibration_json(cal_json_path)

    if cal_key in existing_cal and not args.force:
        print(f"Error: Calibration for {cal_key} already exists.")
        print(f"  Use --force to overwrite, or delete {cal_json_path}")
        return 1

    # Import microscope modules (slow, so do after arg validation)
    print("Connecting to microscope...")
    from flakefinder.leica import LeicaConnection, Stage, ZDrive, Lamp, Shutter, Nosepiece
    from flakefinder.leica.camera import Camera

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up components
        stage = Stage.from_connection(conn)
        camera = Camera.from_connection(conn)

        # Set up nosepiece and switch objective
        try:
            nosepiece = Nosepiece.from_connection(conn)
            current_mag = nosepiece.magnification

            # Find position for requested magnification
            target_pos = None
            for pos, mag in nosepiece.magnifications.items():
                if mag == objective_mag:
                    target_pos = pos
                    break

            if target_pos is None:
                print(f"Error: No objective with magnification {objective_mag}x found")
                print(f"  Available: {list(nosepiece.magnifications.values())}")
                return 1

            if nosepiece.position != target_pos:
                print(f"Switching objective: {current_mag}x -> {objective_mag}x...")
                nosepiece.position = target_pos
                time.sleep(0.5)  # Let it settle
        except LookupError:
            print("Warning: Nosepiece not found, assuming correct objective is in place")

        # Set up lighting
        try:
            lamp = Lamp.from_connection(conn)
            lamp.full()
            lamp_intensity = lamp.intensity
            print(f"Lamp: {lamp.name}, intensity={lamp_intensity}/{lamp.max_intensity}")
        except LookupError:
            lamp_intensity = None
            print("Warning: Lamp not found")

        try:
            shutter = Shutter.from_connection(conn)
            shutter.open()
        except LookupError:
            pass

        # Configure camera for calibration capture
        camera.binning = 2  # 3x3 binning (index 2)
        camera.exposure_time = 0.001  # 1ms
        camera.gamma = 1.0  # Linear
        camera.gain_rgb = (1.0, 1.0, 1.0)  # Neutral WB for calibration
        camera.auto_brightness = False

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
            print(f"  Frame {i+1}/{args.frames}")
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
        print(f"Mean BGR: [{cal_values['mean_bgr'][0]:.1f}, {cal_values['mean_bgr'][1]:.1f}, {cal_values['mean_bgr'][2]:.1f}]")
        print(f"Reference WB (BGR): [{cal_values['reference_wb_bgr'][0]:.3f}, {cal_values['reference_wb_bgr'][1]:.3f}, {cal_values['reference_wb_bgr'][2]:.3f}]")

        # Save flatfield image
        flatfield_filename = f"flatfield_{int(objective_mag)}x_bin{binning}"
        flatfield_path = args.output / flatfield_filename

        # Save as both .npy (for processing) and .png (for viewing)
        np.save(flatfield_path.with_suffix(".npy"), averaged)
        Image.fromarray(averaged).save(flatfield_path.with_suffix(".png"))
        print(f"Saved: {flatfield_path.with_suffix('.npy')}")
        print(f"Saved: {flatfield_path.with_suffix('.png')} (preview)")

        # Update calibration.json
        existing_cal[cal_key] = {
            "flatfield_file": f"{flatfield_filename}.npy",
            "preview_file": f"{flatfield_filename}.png",
            "objective_mag": objective_mag,
            "binning": binning,
            "lamp_intensity": lamp_intensity,
            "exposure_s": camera.exposure_time,
            "frame_size_px": [frame_w, frame_h],
            "position_um": list(position),
            "mean_bgr": cal_values["mean_bgr"],
            "reference_wb_bgr": cal_values["reference_wb_bgr"],
            "frames_averaged": args.frames,
            "captured_at": datetime.now().isoformat(),
            "notes": args.notes,
        }

        save_calibration_json(cal_json_path, existing_cal)
        print(f"Updated: {cal_json_path}")

        print()
        print("=" * 50)
        print(f"Calibration complete for {cal_key}")
        print()
        print("To use in scanner:")
        print(f"  python scan_area_v1.py --objective {int(objective_mag)}x ...")
        print()
        print("The stitcher will automatically apply flatfield correction.")

        return 0


if __name__ == "__main__":
    exit(main())
