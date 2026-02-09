#!/usr/bin/env python3
"""Simple microscope image capture utility.

Usage:
    python capture_util.py output.jpg                    # Basic capture
    python capture_util.py output.png --binning 0        # Full resolution (1x1)
    python capture_util.py output.jpg --downsample 2     # Downsample 2x
    python capture_util.py output.jpg --lamp 80          # Set lamp first
    python capture_util.py output.jpg --exposure 0.05    # 50ms exposure
    python capture_util.py output.jpg --wb 1.2,1.0,0.9   # White balance (R,G,B)
"""

import argparse
import sys

from PIL import Image as PILImage

from flakefinder.leica import (
    Camera,
    Lamp,
    LeicaConnection,
    Nosepiece,
    Shutter,
    Stage,
    ZDrive,
)


def report_status(conn: LeicaConnection, camera: Camera) -> None:
    """Print current microscope status."""
    # Stage XY
    stage = Stage.from_connection(conn)
    x, y = stage.position_um
    print(f"Stage X: {x:.1f} µm ({stage.x.min_um:.0f} - {stage.x.max_um:.0f})")
    print(f"Stage Y: {y:.1f} µm ({stage.y.min_um:.0f} - {stage.y.max_um:.0f})")

    # Z axis
    z = ZDrive.from_connection(conn)
    print(f"Stage Z: {z.position_um:.1f} µm ({z.min_um:.0f} - {z.max_um:.0f})")

    # Nosepiece/objective
    try:
        nosepiece = Nosepiece.from_connection(conn)
        mag = nosepiece.magnification
        if mag:
            print(f"Objective: position {nosepiece.position} ({mag}x)")
        else:
            print(f"Objective: position {nosepiece.position}")
    except LookupError:
        pass

    # Lamp
    try:
        lamp = Lamp.from_connection(conn)
        print(f"Lamp: {lamp.intensity}/{lamp.max_intensity}")
    except LookupError:
        pass

    # Shutter
    try:
        shutter = Shutter.from_connection(conn)
        state = "open" if shutter.is_open else "closed"
        print(f"Shutter: {state}")
    except LookupError:
        pass

    # Camera settings
    binning_map = {0: "1x1", 1: "2x2", 2: "3x3"}
    binning_str = binning_map.get(camera.binning, str(camera.binning))
    w, h = camera.frame_size_px
    r, g, b = camera.gain_rgb
    print(f"Camera: {w}x{h} @ {binning_str} binning, {camera.exposure_time * 1000:.1f}ms exposure")
    print(f"White balance: R={r:.2f} G={g:.2f} B={b:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture an image from the microscope")
    parser.add_argument("output", help="Output file path (jpg, png, tiff)")
    parser.add_argument("--lamp", type=int, default=255, help="Lamp intensity (default: 255)")
    parser.add_argument(
        "--binning",
        type=int,
        choices=[0, 1, 2],
        help="Binning level: 0=1x1, 1=2x2, 2=3x3 (default: 2)",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, help="Downsample factor after capture (default: 1)"
    )
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

    with LeicaConnection() as conn:
        # Set lamp if specified
        if args.lamp is not None:
            try:
                lamp = Lamp.from_connection(conn)
                lamp.intensity = args.lamp
                print(f"Lamp: set to {lamp.intensity}")
            except LookupError:
                print("Warning: Lamp not available")

        # Ensure shutter is open
        try:
            shutter = Shutter.from_connection(conn)
            if not shutter.is_open:
                shutter.open()
                print("Shutter: opened")
        except LookupError:
            pass  # No shutter

        # Move to XY position if specified
        if args.xy:
            parts = args.xy.split(",")
            if len(parts) != 2:
                print("Error: --xy must be X,Y (e.g., '5000,14441')")
                return 1
            target_x, target_y = float(parts[0]), float(parts[1])
            stage = Stage.from_connection(conn)
            print(f"Moving to X={target_x:.0f}, Y={target_y:.0f} µm...")
            stage.move_to(target_x, target_y)
            x, y = stage.position_um
            print(f"Arrived at X={x:.1f}, Y={y:.1f} µm")

        # Move Z if specified
        if args.z is not None:
            z_drive = ZDrive.from_connection(conn)
            print(f"Moving Z to {args.z:.1f} µm...")
            z_drive.move_to(args.z)
            print(f"Z at {z_drive.position_um:.1f} µm")

        # Initialize camera
        camera = Camera.from_connection(conn)

        # Configure camera
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
        stage = Stage.from_connection(conn)
        x, y = stage.position_um
        z = ZDrive.from_connection(conn)
        print(f"Position: X={x:.1f} Y={y:.1f} Z={z.position_um:.1f} µm")

        try:
            lamp = Lamp.from_connection(conn)
            print(f"Lamp: {lamp.intensity}/{lamp.max_intensity}")
        except LookupError:
            print("Lamp: N/A")

        try:
            shutter = Shutter.from_connection(conn)
            print(f"Shutter: {'open' if shutter.is_open else 'closed'}")
        except LookupError:
            print("Shutter: N/A")

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

        # Dispose camera before reporting status (avoids issues)
        camera.dispose()

        # Report full status
        print()
        # Re-create camera just for status (read-only)
        camera2 = Camera.from_connection(conn)
        report_status(conn, camera2)
        camera2.dispose()

    return 0


if __name__ == "__main__":
    sys.exit(main())
