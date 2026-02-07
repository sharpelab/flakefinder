"""Continuous Z-scan autofocus demo.

Performs autofocus by scanning Z downward while capturing frames,
computing sharpness for each frame, and moving to the Z position
with maximum sharpness.

This demo script provides CLI interface, before/after photos, and debug
output. The core autofocus logic is in flakefinder.leica.autofocus.
"""

import argparse
import json
import os
import time

from PIL import Image as PILImage

from flakefinder.leica.autofocus import sharpness, continuous_autofocus, AutofocusResult


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Z-scan autofocus demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x", type=float, default=None, help="Stage X position (um), default: current")
    parser.add_argument("--y", type=float, default=None, help="Stage Y position (um), default: current")
    parser.add_argument("--z", type=float, default=None, help="Initial Z position (um), default: current")
    parser.add_argument("--range", type=float, default=None, help="Z scan range (um), default: auto from objective")
    parser.add_argument("--z-speed", type=float, default=None, help="Z axis speed (um/s), default: use current. Slower = more frames.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for photos (optional)")
    parser.add_argument("--debug-dir", type=str, default=None, help="Save all scan frames to this directory")
    parser.add_argument("--fine", action="store_true", help="Two-pass: coarse scan then fine 50um scan")
    parser.add_argument("--fine-speed-factor", type=float, default=0.25, help="Speed multiplier for fine pass (default: 0.25 = 1/4 speed)")
    parser.add_argument("--sharpness-method", choices=["tenengrad", "laplacian"], default="tenengrad", help="Sharpness metric (laplacian better for low-contrast areas)")
    parser.add_argument("--settle-time", type=float, default=0, help="Settle time in seconds after moves (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without moving")
    parser.add_argument("--clean", action="store_true", help="Remove existing output/debug directories before running")
    args = parser.parse_args()

    print("Continuous Autofocus Demo")
    print("=" * 50)

    # Handle dry-run mode (no hardware connection)
    if args.dry_run:
        if args.x is None or args.y is None or args.z is None:
            print("Error: --dry-run requires --x, --y, --z to be specified")
            print("       (cannot read current position without hardware)")
            return 1

        scan_range = args.range if args.range is not None else 500
        z_start = args.z + scan_range / 2
        z_end = args.z - scan_range / 2

        print(f"Target position: X={args.x:.1f} um, Y={args.y:.1f} um")
        print(f"Initial Z: {args.z:.1f} um")
        print(f"Scan range: {scan_range:.1f} um" + (" (auto)" if args.range is None else ""))
        print(f"Scan: Z={z_start:.1f} -> {z_end:.1f} um (downward)")
        if args.output:
            print(f"Output: {args.output}/")
        print()
        print("[DRY RUN] Would perform:")
        print(f"  1. Move stage to X={args.x:.1f}, Y={args.y:.1f}")
        print(f"  2. Move Z to {args.z:.1f} um")
        print(f"  3. Capture 'before' photo")
        print(f"  4. Scan Z from {z_start:.1f} to {z_end:.1f} um")
        print(f"     - Capture frames continuously")
        print(f"     - Poll Z position at ~64 Hz")
        print(f"     - Compute Sobel sharpness for each frame")
        print(f"  5. Find Z with maximum sharpness")
        print(f"  6. Move to best Z")
        print(f"  7. Capture 'after' photo")
        if args.output:
            print(f"  8. Save summary to {args.output}/")
        return 0

    # Clean existing directories if requested
    if args.clean:
        import shutil
        if args.output and os.path.exists(args.output):
            shutil.rmtree(args.output)
            print(f"Removed existing: {args.output}")
        if args.debug_dir and os.path.exists(args.debug_dir):
            shutil.rmtree(args.debug_dir)
            print(f"Removed existing: {args.debug_dir}")

    # Check directories BEFORE connecting to hardware
    if args.output and os.path.exists(args.output):
        print(f"Error: Output directory '{args.output}' already exists")
        return 1
    if args.debug_dir and os.path.exists(args.debug_dir):
        print(f"Error: Debug directory '{args.debug_dir}' already exists")
        return 1

    # Import hardware libraries
    from flakefinder.leica import LeicaConnection, Stage, Lamp, Shutter, ZDrive
    from flakefinder.leica.camera import Camera
    from flakefinder.leica.enums import UCAPI_IID
    from flakefinder.leica.core import get_interface_required

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up stage (X, Y) and Z
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        # Read current positions
        current_x, current_y = stage.position_um
        current_z = z_drive.position_um

        print(f"Current position: X={current_x:.1f} um, Y={current_y:.1f} um, Z={current_z:.1f} um")
        print()

        # Use current positions as defaults
        target_x = args.x if args.x is not None else current_x
        target_y = args.y if args.y is not None else current_y
        target_z = args.z if args.z is not None else current_z

        print(f"Target position: X={target_x:.1f} um, Y={target_y:.1f} um")
        print(f"Initial Z: {target_z:.1f} um")
        print(f"Scan range: {args.range:.1f} um" if args.range else "Scan range: auto (from objective)")
        if args.output:
            print(f"Output: {args.output}/")
        print()

        # Create output directory if specified
        if args.output:
            os.makedirs(args.output)

        # Set up lighting
        try:
            shutter = Shutter.from_connection(conn)
            shutter.open()
        except LookupError:
            pass

        lamp = None
        try:
            lamp = Lamp.from_connection(conn)
            lamp.full()
        except LookupError:
            pass

        # Set up camera
        try:
            camera = Camera.from_connection(conn)
        except LookupError:
            print("Error: Camera not found")
            return 1

        # Get acquisition interface for raw capture loop
        acquisition = get_interface_required(camera._unit, UCAPI_IID.IID_IMAGE_ACQUISITION)

        # Configure camera for fast capture
        camera.trigger_mode = 0  # CONTINUOUS
        camera.binning = 2       # 3x3 binning for speed
        camera.exposure_time = 0.001  # 1ms exposure

        print(f"Camera: {camera.name}")
        print(f"  Binning: 3x3, Exposure: 1ms")
        if lamp:
            print(f"Lamp: {lamp.name}, intensity={lamp.intensity}/{lamp.max_intensity}")
        print()

        # Set up acquisition context
        context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory

        # === Step 1: Move to XY position ===
        print(f"Moving to X={target_x:.1f}, Y={target_y:.1f}...")
        hx, hy = stage.move_to_async(target_x, target_y)
        Stage.wait_all([hx, hy])
        hx.dispose()
        hy.dispose()
        if args.settle_time > 0:
            time.sleep(args.settle_time)

        # === Step 2: Move to initial Z ===
        # Set Z speed for positioning (restores sane default if a previous crash left it slow)
        if z_drive.supports_velocity and z_drive.max_velocity_um_s:
            z_drive.set_velocity_um_s(args.z_speed if args.z_speed else z_drive.max_velocity_um_s)
        print(f"Moving Z to {target_z:.1f} um...")
        z_drive.move_to(target_z)

        # === Step 3: Capture 'before' photo ===
        print("Capturing 'before' photo...")
        before_img = camera.capture()
        if before_img is not None:
            before_sharpness = sharpness(before_img)
            if args.output:
                before_path = os.path.join(args.output, "before.jpg")
                PILImage.fromarray(before_img).save(before_path, quality=95)
                print(f"  Saved: {before_path}")
            print(f"  Sharpness: {before_sharpness:.2f}")
        else:
            print("  Warning: Failed to capture before image")
            before_sharpness = 0.0

        # === Step 4: Run autofocus using library ===
        print()
        print("Starting autofocus scan...")

        try:
            af_result = continuous_autofocus(
                conn=conn,
                camera=camera,
                acquisition=acquisition,
                context=context,
                z_range_um=args.range,
                z_start_um=target_z,
                z_speed_um_s=args.z_speed,
                fine_pass=args.fine,
                fine_speed_factor=args.fine_speed_factor,
                sharpness_method=args.sharpness_method,
                store_frames=bool(args.debug_dir),
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        print(f"  Scan completed in {af_result.scan_duration_s:.2f}s")
        print(f"  Frames captured: {af_result.frame_count}")
        print(f"  Z samples: {af_result.z_sample_count}")
        print()

        # Sharpness stats
        sharpness_values = [r["sharpness"] for r in af_result.sharpness_curve]
        min_sharpness = min(sharpness_values) if sharpness_values else 0
        max_sharpness = max(sharpness_values) if sharpness_values else 0
        mean_sharpness = sum(sharpness_values) / len(sharpness_values) if sharpness_values else 0

        print(f"  Sharpness range: {min_sharpness:.2f} - {max_sharpness:.2f}")
        print(f"  Mean sharpness: {mean_sharpness:.2f}")
        print(f"  Best Z: {af_result.best_z_um:.2f} um")
        print(f"  Best sharpness: {af_result.best_sharpness:.2f}")
        if af_result.stayed_at_initial:
            print(f"  ** Stayed at initial (scan found nothing better) **")

        # Find best frame for saving
        best_frame_idx = 0
        best_scan_path = None
        if af_result.sharpness_curve:
            best = max(af_result.sharpness_curve, key=lambda r: r["sharpness"])
            best_frame_idx = best["frame"]

        # Save best frame from scan (if we have frames stored and output specified)
        if args.output and af_result.frames and len(af_result.frames) > best_frame_idx:
            best_frame = af_result.frames[best_frame_idx]
            if best_frame.image is not None:
                best_scan_path = os.path.join(args.output, "best_scan_frame.jpg")
                PILImage.fromarray(best_frame.image).save(best_scan_path, quality=95)
                print(f"  Saved best scan frame: {best_scan_path}")

        # Save debug frames if requested
        if args.debug_dir and af_result.frames:
            os.makedirs(args.debug_dir)
            print(f"  Saving {len(af_result.frames)} frames to {args.debug_dir}/...")

            for i, frame in enumerate(af_result.frames):
                if frame.image is not None:
                    fname = f"frame_{i:03d}_z_{frame.z_um:.1f}_s_{frame.sharpness:.1f}.jpg"
                    PILImage.fromarray(frame.image).save(os.path.join(args.debug_dir, fname), quality=95)

            # Save sharpness curve CSV
            csv_path = os.path.join(args.debug_dir, "sharpness_curve.csv")
            with open(csv_path, "w") as f:
                f.write("frame,z_um,sharpness\n")
                for r in af_result.sharpness_curve:
                    f.write(f"{r['frame']},{r['z_um']:.2f},{r['sharpness']:.2f}\n")
            print(f"  Saved sharpness curve to {csv_path}")

        # === Step 5: Capture 'after' photo ===
        # Z is already at best position (library moved it there)
        print()
        print("Capturing 'after' photo...")
        if args.settle_time > 0:
            time.sleep(args.settle_time)
        after_img = camera.capture()
        after_path = None
        if after_img is not None:
            after_sharpness = sharpness(after_img)
            if args.output:
                after_path = os.path.join(args.output, "after.jpg")
            else:
                after_path = "autofocus_after.jpg"
            PILImage.fromarray(after_img).save(after_path, quality=95)
            print(f"  Saved: {after_path}")
            print(f"  Sharpness: {after_sharpness:.2f}")
            os.startfile(after_path)
        else:
            print("  Warning: Failed to capture after image")
            after_sharpness = af_result.final_sharpness

        # === Step 6: Save summary (if output specified) ===
        summary_path = None
        if args.output:
            summary = {
                "params": {
                    "x_um": target_x,
                    "y_um": target_y,
                    "z_initial_um": target_z,
                    "range_um": af_result.z_range_um,
                    "fine_pass": args.fine,
                    "objective_position": af_result.objective_position,
                },
                "scan": {
                    "duration_s": af_result.scan_duration_s,
                    "frame_count": af_result.frame_count,
                    "z_sample_count": af_result.z_sample_count,
                },
                "before": {
                    "z_um": target_z,
                    "sharpness": before_sharpness,
                },
                "best": {
                    "z_um": af_result.best_z_um,
                    "sharpness": af_result.best_sharpness,
                },
                "after": {
                    "z_um": af_result.best_z_um,
                    "sharpness": after_sharpness,
                },
                "sharpness_stats": {
                    "min": min_sharpness,
                    "max": max_sharpness,
                    "mean": mean_sharpness,
                },
                "sharpness_curve": af_result.sharpness_curve,
            }

            summary_path = os.path.join(args.output, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print()
            print(f"Saved summary: {summary_path}")

        # === Print final summary ===
        print()
        print("=" * 50)
        print("AUTOFOCUS SUMMARY")
        print("=" * 50)
        print(f"Initial Z:    {target_z:.2f} um (sharpness: {before_sharpness:.2f})")
        print(f"Best Z:       {af_result.best_z_um:.2f} um (sharpness: {af_result.best_sharpness:.2f})")
        print(f"After Z:      {af_result.best_z_um:.2f} um (sharpness: {after_sharpness:.2f})")
        print(f"Z adjustment: {af_result.best_z_um - target_z:+.2f} um")
        if before_sharpness > 0:
            improvement = after_sharpness - before_sharpness
            pct = (after_sharpness / before_sharpness - 1) * 100
            print(f"Sharpness improvement: {improvement:+.2f} ({pct:+.1f}%)")
        if args.output:
            print()
            print("Output files:")
            print(f"  {os.path.join(args.output, 'before.jpg')}")
            print(f"  {os.path.join(args.output, 'after.jpg')}")
            if best_scan_path:
                print(f"  {best_scan_path}")
            print(f"  {summary_path}")
        print()
        print("Done.")

        return 0


if __name__ == "__main__":
    exit(main())
