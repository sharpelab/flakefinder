#!/usr/bin/env python3
"""Simple microscope image capture utility.

Usage:
    python capture_util.py output.jpg                    # Basic capture
    python capture_util.py output.png --binning 0        # Full resolution (1x1)
    python capture_util.py output.jpg --downsample 2     # Downsample 2x
    python capture_util.py output.jpg --lamp 80          # Set lamp to 80%
    python capture_util.py output.jpg --exposure-ms 50    # 50ms exposure
    python capture_util.py output.jpg --white-balance 2.51,1.02,1.41  # White balance (B,G,R)
    python capture_util.py output.jpg --x 5000 --y 14441 # Move then capture
    python capture_util.py output.jpg --objective-mag 20x  # Switch to 20x then capture
    python capture_util.py output.jpg --focus             # Autofocus then capture sharpest frame
    python capture_util.py output.jpg --focus --z-range 100 --z-speed 800  # Custom focus params
"""

import argparse
import sys

from PIL import Image as PILImage

from flakefinder.cli_utils import report_status
from flakefinder.data_utils import require_microscope_description
from flakefinder.leica import Microscope, wait_all
from flakefinder.leica.autofocus import focus_and_capture
from flakefinder.scan_utils import parse_white_balance


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture an image from the microscope")
    parser.add_argument("output", help="Output file path (jpg, png, tiff)")
    parser.add_argument("--lamp", type=float, default=100, help="Lamp intensity 0-100%% (default: 100)")
    parser.add_argument(
        "--binning",
        type=int,
        choices=[0, 1, 2],
        help="Binning level: 0=1x1, 1=2x2, 2=3x3 (default: 2)",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor after capture (default: 1)")
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=None,
        help="Exposure time in ms (default: from objective in --focus mode, else 1.0)",
    )
    parser.add_argument(
        "--white-balance",
        type=parse_white_balance,
        default="2.51,1.02,1.41",
        help="White balance as B,G,R gains (default: '2.51,1.02,1.41')",
    )
    parser.add_argument("--gain", type=float, help="Camera gain (e.g., 4.0)")
    parser.add_argument("--saturation", type=int, help="Color saturation (e.g., 100)")
    parser.add_argument("--gamma", type=float, help="Gamma correction (e.g., 1.0)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output; print only saved filename")
    parser.add_argument("--x", type=float, default=None, help="Move to X position in µm before capture")
    parser.add_argument("--y", type=float, default=None, help="Move to Y position in µm before capture")
    parser.add_argument("--z", type=float, help="Move to Z position in µm before capture")
    parser.add_argument("--aperture", type=int, metavar="VALUE", help="Set aperture diaphragm value")
    parser.add_argument("--objective-mag", type=str, help="Switch objective magnification (e.g. '20x', '5', '2.5x')")
    parser.add_argument("--focus", action="store_true", help="Autofocus: scan Z range and capture sharpest frame")
    parser.add_argument(
        "--z-range", type=float, default=None, help="Focus Z search range in µm (default: auto from objective)"
    )
    parser.add_argument("--z-speed", type=float, default=None, help="Focus Z speed in µm/s (default: from objective)")
    args = parser.parse_args()

    with Microscope() as scope:
        # Switch objective if specified (before any moves)
        if args.objective_mag is not None:
            switched = scope.switch_objective_mag(args.objective_mag)
            mag = scope.nosepiece.magnification
            if switched:
                print(f"Switched to {mag}x objective")
            else:
                print(f"Already at {mag}x objective")

        # Set lamp and open shutter
        scope.light_on(args.lamp)

        # Set aperture if specified
        if args.aperture is not None:
            scope.aperture.value = args.aperture

        # Move to XY position if specified
        if args.x is not None or args.y is not None:
            cur_x, cur_y = scope.stage.position_um
            target_x = args.x if args.x is not None else cur_x
            target_y = args.y if args.y is not None else cur_y
            print(f"Moving to X={target_x:.0f}, Y={target_y:.0f} µm...")
            hx, hy = scope.stage.move_to_async(target_x, target_y)
            wait_all([hx, hy])
            x, y = scope.stage.position_um
            print(f"Arrived at X={x:.1f}, Y={y:.1f} µm")

        # Move Z if specified (skip when --focus, since focus_and_capture handles z_center)
        if args.z is not None and not args.focus:
            print(f"Moving Z to {args.z:.1f} µm...")
            scope.z.move_to(args.z)
            print(f"Z at {scope.z.position_um:.1f} µm")

        # Configure camera
        camera = scope.camera

        if args.binning is not None:
            camera.binning = args.binning

        # In --focus mode, focus_and_capture handles exposure from objective defaults.
        # In regular mode, apply explicit value or fallback to 1.0 ms.
        if not args.focus:
            camera.exposure_time = (args.exposure_ms if args.exposure_ms is not None else 1.0) / 1000.0

        if args.gain is not None:
            camera.gain = args.gain

        if args.saturation is not None:
            camera.saturation = args.saturation

        if args.gamma is not None:
            camera.gamma = args.gamma

        camera.gain_rgb = args.white_balance

        # Report capture settings
        if not args.quiet:
            x, y = scope.stage.position_um
            mag = scope.nosepiece.magnification
            obj_str = f"{mag}x" if mag else f"position {scope.nosepiece.position}"
            print(f"Objective: {obj_str}")
            print(f"Position: X={x:.1f} Y={y:.1f} Z={scope.z.position_um:.1f} µm")
            print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")
            print(f"Shutter: {'open' if scope.shutter.is_open else 'closed'}")
            ap = scope.aperture
            ap_label = " (fully open)" if ap.value == ap.max_value else ""
            print(f"Aperture: {ap.value}/{ap.max_value}{ap_label}")

            desc = require_microscope_description()
            binning_str = desc.camera.binning_levels[camera.binning].name
            r, g, b = camera.gain_rgb
            print(f"Exposure: {camera.exposure_time * 1000:.2f} ms")
            print(f"Binning: {binning_str}")
            print(f"White balance: R={r:.2f} G={g:.2f} B={b:.2f}")
            print()

        # Capture
        if args.focus:
            if not args.quiet:
                print("Focus-and-capture...")
            z_center = args.z if args.z is not None else None
            result = focus_and_capture(
                scope,
                z_center_um=z_center,
                z_range_um=args.z_range,
                z_speed_um_s=args.z_speed,
                exposure_ms=args.exposure_ms,
                gain=args.gain,
            )
            image = result.image
            if not args.quiet:
                print(f"Best Z: {result.z_um:.1f} µm (sharpness: {result.sharpness:.1f})")
                print(f"Frames: {result.frame_count}, scan: {result.scan_duration_s:.2f}s")
                print(f"Z range: {result.z_range_um:.0f} µm")
                print(f"Focus quality: {result.focus_quality.value}")
            else:
                fq = result.focus_quality.value
                if fq != "ok":
                    print(f"focus_quality={fq}")
        else:
            if not args.quiet:
                print("Capturing...")
            image = camera.capture()

        # Convert to PIL
        img = PILImage.fromarray(image)

        # Downsample if requested
        if args.downsample > 1:
            new_size = (img.width // args.downsample, img.height // args.downsample)
            img = img.resize(new_size, PILImage.Resampling.LANCZOS)
            print(f"Downsampled: {image.shape[1]}x{image.shape[0]} -> {new_size[0]}x{new_size[1]}")

        # Save
        output_lower = args.output.lower()
        if output_lower.endswith(".jpg") or output_lower.endswith(".jpeg"):
            img.save(args.output, quality=args.quality)
        else:
            img.save(args.output)

        print(f"Saved: {args.output} ({img.width}x{img.height})")

        # Report full status
        if not args.quiet:
            print()
            report_status(scope)

    return 0


if __name__ == "__main__":
    sys.exit(main())
