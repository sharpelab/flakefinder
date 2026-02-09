#!/usr/bin/env python3
"""Simple microscope image capture utility.

Usage:
    python capture_util.py output.jpg                    # Basic capture
    python capture_util.py output.png --binning 0        # Full resolution (1x1)
    python capture_util.py output.jpg --downsample 2     # Downsample 2x
    python capture_util.py output.jpg --lamp 80          # Set lamp to 80%
    python capture_util.py output.jpg --exposure 0.05    # 50ms exposure
    python capture_util.py output.jpg --wb 1.2,1.0,0.9   # White balance (R,G,B)
"""

import argparse
import sys

from PIL import Image as PILImage

from flakefinder.leica import Microscope


def report_status(scope: Microscope) -> None:
    """Print current microscope status."""
    # Stage XY
    x, y = scope.stage.position_um
    print(f"Stage X: {x:.1f} µm ({scope.stage.x.min_um:.0f} - {scope.stage.x.max_um:.0f})")
    print(f"Stage Y: {y:.1f} µm ({scope.stage.y.min_um:.0f} - {scope.stage.y.max_um:.0f})")

    # Z axis
    print(f"Stage Z: {scope.z.position_um:.1f} µm ({scope.z.min_um:.0f} - {scope.z.max_um:.0f})")

    # Nosepiece/objective
    mag = scope.nosepiece.magnification
    if mag:
        print(f"Objective: position {scope.nosepiece.position} ({mag}x)")
    else:
        print(f"Objective: position {scope.nosepiece.position}")

    # Lamp
    print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")

    # Shutter
    state = "open" if scope.shutter.is_open else "closed"
    print(f"Shutter: {state}")

    # Camera settings
    camera = scope.camera
    binning_map = {0: "1x1", 1: "2x2", 2: "3x3"}
    binning_str = binning_map.get(camera.binning, str(camera.binning))
    w, h = camera.frame_size_px
    r, g, b = camera.gain_rgb
    print(f"Camera: {w}x{h} @ {binning_str} binning, {camera.exposure_time * 1000:.1f}ms exposure")
    print(f"White balance: R={r:.2f} G={g:.2f} B={b:.2f}")


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
        "--exposure",
        type=float,
        default=0.001,
        help="Exposure time in seconds (default: 0.001 = 1ms)",
    )
    parser.add_argument(
        "--wb",
        type=str,
        default="1.41,1.02,2.51",
        help="White balance as R,G,B (default: '1.41,1.02,2.51')",
    )
    parser.add_argument("--wb-red", type=float, help="Red channel gain")
    parser.add_argument("--wb-green", type=float, help="Green channel gain")
    parser.add_argument("--wb-blue", type=float, help="Blue channel gain")
    parser.add_argument("--gain", type=float, help="Camera gain (e.g., 4.0)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    parser.add_argument(
        "--xy",
        type=str,
        metavar="X,Y",
        help="Move to X,Y position in µm before capture (e.g., '5000,14441')",
    )
    parser.add_argument("--z", type=float, help="Move to Z position in µm before capture")
    args = parser.parse_args()

    # Parse white balance
    wb_r, wb_g, wb_b = None, None, None
    if args.wb:
        parts = args.wb.split(",")
        if len(parts) != 3:
            print("Error: --wb must be R,G,B (e.g., '1.2,1.0,0.9')")
            return 1
        wb_r, wb_g, wb_b = float(parts[0]), float(parts[1]), float(parts[2])
    if args.wb_red is not None:
        wb_r = args.wb_red
    if args.wb_green is not None:
        wb_g = args.wb_green
    if args.wb_blue is not None:
        wb_b = args.wb_blue

    with Microscope() as scope:
        # Set lamp and open shutter
        scope.light_on(args.lamp)

        # Move to XY position if specified
        if args.xy:
            parts = args.xy.split(",")
            if len(parts) != 2:
                print("Error: --xy must be X,Y (e.g., '5000,14441')")
                return 1
            target_x, target_y = float(parts[0]), float(parts[1])
            print(f"Moving to X={target_x:.0f}, Y={target_y:.0f} µm...")
            scope.stage.move_to(target_x, target_y)
            x, y = scope.stage.position_um
            print(f"Arrived at X={x:.1f}, Y={y:.1f} µm")

        # Move Z if specified
        if args.z is not None:
            print(f"Moving Z to {args.z:.1f} µm...")
            scope.z.move_to(args.z)
            print(f"Z at {scope.z.position_um:.1f} µm")

        # Configure camera
        camera = scope.camera

        if args.binning is not None:
            camera.binning = args.binning

        if args.exposure is not None:
            camera.exposure_time = args.exposure

        if args.gain is not None:
            camera.gain = args.gain

        if wb_r is not None or wb_g is not None or wb_b is not None:
            current_r, current_g, current_b = camera.gain_rgb
            camera.gain_rgb = (
                wb_r if wb_r is not None else current_r,
                wb_g if wb_g is not None else current_g,
                wb_b if wb_b is not None else current_b,
            )

        # Report capture settings
        x, y = scope.stage.position_um
        print(f"Position: X={x:.1f} Y={y:.1f} Z={scope.z.position_um:.1f} µm")
        print(f"Lamp: {scope.lamp.intensity_pct:.0f}% ({scope.lamp.intensity}/{scope.lamp.max_intensity})")
        print(f"Shutter: {'open' if scope.shutter.is_open else 'closed'}")

        binning_map = {0: "1x1", 1: "2x2", 2: "3x3"}
        binning_str = binning_map.get(camera.binning, str(camera.binning))
        r, g, b = camera.gain_rgb
        print(f"Exposure: {camera.exposure_time * 1000:.2f} ms")
        print(f"Binning: {binning_str}")
        print(f"White balance: R={r:.2f} G={g:.2f} B={b:.2f}")
        print()

        # Capture
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
        print()
        report_status(scope)

    return 0


if __name__ == "__main__":
    sys.exit(main())
