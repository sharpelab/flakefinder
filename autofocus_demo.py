"""Continuous Z-scan autofocus demo.

Performs autofocus by scanning Z downward while capturing frames,
computing sharpness for each frame, and moving to the Z position
with maximum sharpness.
"""

import argparse
import bisect
import json
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image as PILImage


def sharpness(image: np.ndarray) -> float:
    """Compute Tenengrad sharpness (Sobel gradient magnitude mean).

    Args:
        image: BGR or grayscale image as numpy array.

    Returns:
        Sharpness value (higher = sharper).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.mean(cv2.magnitude(sobel_x, sobel_y))[0]


def interpolate_position(t: float, samples: list[tuple[float, float, float]]) -> float | None:
    """Interpolate position at time t from (t_before, t_after, z_um) samples.

    Uses midpoint of t_before/t_after as the effective sample time.
    """
    if not samples:
        return None

    # Use midpoint of before/after as effective time
    times = [(s[0] + s[1]) / 2 for s in samples]

    # Find insertion point
    idx = bisect.bisect_left(times, t)

    if idx == 0:
        return samples[0][2]  # Before first sample
    if idx >= len(samples):
        return samples[-1][2]  # After last sample

    # Linear interpolate between samples[idx-1] and samples[idx]
    t0, z0 = times[idx - 1], samples[idx - 1][2]
    t1, z1 = times[idx], samples[idx][2]

    if t1 == t0:
        return z0

    alpha = (t - t0) / (t1 - t0)
    return z0 + alpha * (z1 - z0)


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Z-scan autofocus demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--x", type=float, default=None, help="Stage X position (um), default: current")
    parser.add_argument("--y", type=float, default=None, help="Stage Y position (um), default: current")
    parser.add_argument("--z", type=float, default=None, help="Initial Z position (um), default: current")
    parser.add_argument("--range", type=float, default=500, help="Z scan range (um)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for photos")
    parser.add_argument("--debug-dir", type=str, default=None, help="Save all scan frames to this directory")
    parser.add_argument("--fine", action="store_true", help="Two-pass: coarse scan then fine 50um scan")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without moving")
    args = parser.parse_args()

    print("Continuous Autofocus Demo")
    print("=" * 50)

    # Handle dry-run mode (no hardware connection)
    if args.dry_run:
        if args.x is None or args.y is None or args.z is None:
            print("Error: --dry-run requires --x, --y, --z to be specified")
            print("       (cannot read current position without hardware)")
            return 1

        z_start = args.z + args.range / 2
        z_end = args.z - args.range / 2

        print(f"Target position: X={args.x:.1f} um, Y={args.y:.1f} um")
        print(f"Initial Z: {args.z:.1f} um")
        print(f"Scan range: {args.range:.1f} um")
        print(f"Scan: Z={z_start:.1f} -> {z_end:.1f} um (downward)")
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
        print(f"  8. Save summary to {args.output}/")
        return 0

    # Check directories BEFORE connecting to hardware
    if os.path.exists(args.output):
        print(f"Error: Output directory '{args.output}' already exists")
        return 1
    if args.debug_dir and os.path.exists(args.debug_dir):
        print(f"Error: Debug directory '{args.debug_dir}' already exists")
        return 1

    # Import hardware libraries
    from flakefinder.leica import LeicaConnection, Stage, Lamp, Shutter, Axis, TID
    from flakefinder.leica.camera import Camera
    from flakefinder.leica.enums import UCAPI_IID
    from flakefinder.leica.core import get_interface_required, find_unit

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up stage (X, Y)
        stage = Stage.from_connection(conn)

        # Set up Z axis
        z_unit = find_unit(conn.root, TID.MICROSCOPE_ZDRIVE)
        if z_unit is None:
            print("Error: Z drive not found")
            return 1
        z_axis = Axis(z_unit)

        # Read current positions
        current_x, current_y = stage.position_um
        current_z = z_axis.position_um

        print(f"Current position: X={current_x:.1f} um, Y={current_y:.1f} um, Z={current_z:.1f} um")
        print()

        # Use current positions as defaults
        target_x = args.x if args.x is not None else current_x
        target_y = args.y if args.y is not None else current_y
        target_z = args.z if args.z is not None else current_z

        # Compute scan bounds (scan downward for safety)
        z_start = target_z + args.range / 2  # Start high
        z_end = target_z - args.range / 2    # End low (away from sample)

        print(f"Target position: X={target_x:.1f} um, Y={target_y:.1f} um")
        print(f"Initial Z: {target_z:.1f} um")
        print(f"Scan range: {args.range:.1f} um")
        print(f"Scan: Z={z_start:.1f} -> {z_end:.1f} um (downward)")
        print(f"Output: {args.output}/")
        print()

        # Create output directory (already validated above)
        os.makedirs(args.output)

        # Verify Z limits
        z_min_um = z_axis.min_um
        z_max_um = z_axis.max_um
        print(f"Z axis range: {z_min_um:.1f} - {z_max_um:.1f} um")

        if z_start > z_max_um or z_start < z_min_um:
            print(f"Error: Z start ({z_start:.1f}) outside axis limits")
            return 1
        if z_end > z_max_um or z_end < z_min_um:
            print(f"Error: Z end ({z_end:.1f}) outside axis limits")
            return 1

        # Set up lighting
        shutter = None
        lamp = None
        try:
            shutter = Shutter.from_connection(conn)
            shutter.open()
        except LookupError:
            pass

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
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # Get fast position reading interfaces
        z_bcv = z_axis.bcv
        z_converter = z_axis.converter

        # === Step 1: Move to XY position ===
        print(f"Moving to X={target_x:.1f}, Y={target_y:.1f}...")
        hx, hy = stage.move_to_async(target_x, target_y)
        Stage.wait_all([hx, hy])
        hx.dispose()
        hy.dispose()

        # === Step 2: Move to initial Z ===
        print(f"Moving Z to {target_z:.1f} um...")
        z_axis.move_to(target_z)

        # === Step 3: Capture 'before' photo ===
        print("Capturing 'before' photo...")
        current_image[0] = None
        acquisition.Acquire(context, None)
        if current_image[0] is not None:
            before_img = Camera._image_to_numpy(current_image[0])
            current_image[0].Dispose()
            before_sharpness = sharpness(before_img)
            before_path = os.path.join(args.output, "before.jpg")
            PILImage.fromarray(before_img).save(before_path, quality=95)
            print(f"  Saved: {before_path}")
            print(f"  Sharpness: {before_sharpness:.2f}")
        else:
            print("  Warning: Failed to capture before image")
            before_sharpness = 0

        # === Step 4: Continuous Z-scan autofocus ===
        print()
        print(f"Starting Z scan: {z_start:.1f} -> {z_end:.1f} um...")

        # Move to scan start position
        z_axis.move_to(z_start)
        time.sleep(0.1)  # Brief settle

        # Data collection
        z_samples = []        # (t_before, t_after, z_um)
        frame_data = []       # [(t_capture, image), ...]
        stop_polling = threading.Event()

        def z_poll_thread():
            """Poll Z position continuously during scan."""
            while not stop_polling.is_set():
                t_before = time.perf_counter()
                z_native = z_bcv.GetControlValue()
                t_after = time.perf_counter()
                z_um = z_converter.GetMetricsValue(z_native)
                z_samples.append((t_before, t_after, z_um))

        # Start Z polling
        z_thread = threading.Thread(target=z_poll_thread, daemon=True)
        z_thread.start()

        scan_start_time = time.perf_counter()

        # Start async Z move (downward)
        z_handle = z_axis.move_to_async(z_end)

        # Capture frames during move
        frame_count = 0
        while not z_handle.is_complete:
            t_capture = time.perf_counter()
            current_image[0] = None
            acquisition.Acquire(context, None)

            if current_image[0] is not None:
                # Convert immediately and dispose .NET object
                img_arr = Camera._image_to_numpy(current_image[0])
                current_image[0].Dispose()
                frame_data.append((t_capture, img_arr))
                frame_count += 1

        scan_end_time = time.perf_counter()
        z_handle.dispose()

        # Stop polling
        stop_polling.set()
        z_thread.join(timeout=1.0)

        scan_duration = scan_end_time - scan_start_time
        print(f"  Scan completed in {scan_duration:.2f}s")
        print(f"  Frames captured: {frame_count}")
        print(f"  Z samples: {len(z_samples)}")

        # === Step 5: Compute sharpness and find best Z ===
        print()
        print("Computing sharpness...")

        results = []
        for i, (t_capture, img) in enumerate(frame_data):
            z_interp = interpolate_position(t_capture, z_samples)
            s = sharpness(img)
            results.append({
                "frame": i,
                "t": t_capture - scan_start_time,
                "z_um": z_interp,
                "sharpness": s,
            })

        # Find best frame
        if results:
            best = max(results, key=lambda r: r["sharpness"])
            best_z = best["z_um"]
            best_sharpness = best["sharpness"]
            best_frame_idx = best["frame"]

            # Stats
            sharpness_values = [r["sharpness"] for r in results]
            min_sharpness = min(sharpness_values)
            max_sharpness = max(sharpness_values)
            mean_sharpness = sum(sharpness_values) / len(sharpness_values)

            print(f"  Sharpness range: {min_sharpness:.2f} - {max_sharpness:.2f}")
            print(f"  Mean sharpness: {mean_sharpness:.2f}")
            print(f"  Best frame: #{best_frame_idx}")
            print(f"  Best Z: {best_z:.2f} um")
            print(f"  Best sharpness: {best_sharpness:.2f}")
        else:
            print("  Error: No frames captured")
            return 1

        # Save best frame from scan
        best_scan_path = os.path.join(args.output, "best_scan_frame.jpg")
        PILImage.fromarray(frame_data[best_frame_idx][1]).save(best_scan_path, quality=95)

        # Save all frames to debug dir if requested
        if args.debug_dir:
            os.makedirs(args.debug_dir)  # Already validated above

            print(f"  Saving {len(frame_data)} frames to {args.debug_dir}/...")
            for i, (t_capture, img) in enumerate(frame_data):
                r = results[i]
                z_um = r["z_um"]
                s = r["sharpness"]
                # Filename: frame_NNN_z_ZZZZ.Z_s_SSS.S.jpg
                fname = f"frame_{i:03d}_z_{z_um:.1f}_s_{s:.1f}.jpg"
                PILImage.fromarray(img).save(os.path.join(args.debug_dir, fname), quality=95)

            # Also save a CSV of the sharpness curve
            csv_path = os.path.join(args.debug_dir, "sharpness_curve.csv")
            with open(csv_path, "w") as f:
                f.write("frame,t_s,z_um,sharpness\n")
                for r in results:
                    f.write(f"{r['frame']},{r['t']:.4f},{r['z_um']:.2f},{r['sharpness']:.2f}\n")
            print(f"  Saved sharpness curve to {csv_path}")
        print(f"  Saved best scan frame: {best_scan_path}")

        # === Optional: Fine pass ===
        if args.fine:
            fine_range = 50.0  # 50µm fine scan
            fine_z_start = best_z + fine_range / 2
            fine_z_end = best_z - fine_range / 2

            # Clamp to axis limits
            fine_z_start = min(fine_z_start, z_max_um)
            fine_z_end = max(fine_z_end, z_min_um)

            print()
            print(f"=== Fine pass: {fine_z_start:.1f} -> {fine_z_end:.1f} um (50um range) ===")

            # Move to fine scan start
            z_axis.move_to(fine_z_start)
            time.sleep(0.1)

            # Fine scan data collection
            fine_z_samples = []
            fine_frame_data = []
            fine_stop_polling = threading.Event()

            def fine_z_poll_thread():
                while not fine_stop_polling.is_set():
                    t_before = time.perf_counter()
                    z_native = z_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    z_um = z_converter.GetMetricsValue(z_native)
                    fine_z_samples.append((t_before, t_after, z_um))

            fine_z_thread = threading.Thread(target=fine_z_poll_thread, daemon=True)
            fine_z_thread.start()

            fine_scan_start = time.perf_counter()
            fine_z_handle = z_axis.move_to_async(fine_z_end)

            fine_frame_count = 0
            while not fine_z_handle.is_complete:
                t_capture = time.perf_counter()
                current_image[0] = None
                acquisition.Acquire(context, None)

                if current_image[0] is not None:
                    img_arr = Camera._image_to_numpy(current_image[0])
                    current_image[0].Dispose()
                    fine_frame_data.append((t_capture, img_arr))
                    fine_frame_count += 1

            fine_scan_end = time.perf_counter()
            fine_z_handle.dispose()

            fine_stop_polling.set()
            fine_z_thread.join(timeout=1.0)

            fine_duration = fine_scan_end - fine_scan_start
            print(f"  Fine scan: {fine_duration:.2f}s, {fine_frame_count} frames")

            # Compute fine sharpness
            fine_results = []
            for i, (t_capture, img) in enumerate(fine_frame_data):
                z_interp = interpolate_position(t_capture, fine_z_samples)
                s = sharpness(img)
                fine_results.append({
                    "frame": i,
                    "t": t_capture - fine_scan_start,
                    "z_um": z_interp,
                    "sharpness": s,
                })

            if fine_results:
                fine_best = max(fine_results, key=lambda r: r["sharpness"])
                print(f"  Fine best: Z={fine_best['z_um']:.2f} um, sharpness={fine_best['sharpness']:.2f}")
                print(f"  Improvement: {fine_best['sharpness'] - best_sharpness:+.2f} over coarse")

                # Update best Z and overwrite best scan frame
                best_z = fine_best["z_um"]
                best_sharpness = fine_best["sharpness"]
                fine_best_idx = fine_best["frame"]
                PILImage.fromarray(fine_frame_data[fine_best_idx][1]).save(best_scan_path, quality=95)
                print(f"  Updated best scan frame: {best_scan_path}")

                # Save fine debug frames if debug_dir specified
                if args.debug_dir:
                    fine_dir = os.path.join(args.debug_dir, "fine")
                    os.makedirs(fine_dir)
                    for i, (t_capture, img) in enumerate(fine_frame_data):
                        r = fine_results[i]
                        fname = f"frame_{i:03d}_z_{r['z_um']:.1f}_s_{r['sharpness']:.1f}.jpg"
                        PILImage.fromarray(img).save(os.path.join(fine_dir, fname), quality=95)
                    csv_path = os.path.join(fine_dir, "sharpness_curve.csv")
                    with open(csv_path, "w") as f:
                        f.write("frame,t_s,z_um,sharpness\n")
                        for r in fine_results:
                            f.write(f"{r['frame']},{r['t']:.4f},{r['z_um']:.2f},{r['sharpness']:.2f}\n")
                    print(f"  Saved fine frames to {fine_dir}/")

        # === Step 6: Move to best Z ===
        print()
        print(f"Moving to best Z: {best_z:.2f} um...")
        z_axis.move_to(best_z)

        # === Step 7: Capture 'after' photo ===
        # Wait for Z to settle and discard a frame to clear any buffered state
        time.sleep(0.1)
        current_image[0] = None
        acquisition.Acquire(context, None)  # Discard this frame
        if current_image[0] is not None:
            current_image[0].Dispose()

        print("Capturing 'after' photo...")
        time.sleep(0.05)
        current_image[0] = None
        acquisition.Acquire(context, None)
        if current_image[0] is not None:
            after_img = Camera._image_to_numpy(current_image[0])
            current_image[0].Dispose()
            after_sharpness = sharpness(after_img)
            after_path = os.path.join(args.output, "after.jpg")
            PILImage.fromarray(after_img).save(after_path, quality=95)
            print(f"  Saved: {after_path}")
            print(f"  Sharpness: {after_sharpness:.2f}")
        else:
            print("  Warning: Failed to capture after image")
            after_sharpness = 0

        # === Step 8: Save summary ===
        summary = {
            "params": {
                "x_um": target_x,
                "y_um": target_y,
                "z_initial_um": target_z,
                "range_um": args.range,
                "z_start_um": z_start,
                "z_end_um": z_end,
            },
            "scan": {
                "duration_s": scan_duration,
                "frame_count": frame_count,
                "z_sample_count": len(z_samples),
            },
            "before": {
                "z_um": target_z,
                "sharpness": before_sharpness,
            },
            "best": {
                "z_um": best_z,
                "sharpness": best_sharpness,
                "frame": best_frame_idx,
            },
            "after": {
                "z_um": best_z,
                "sharpness": after_sharpness,
            },
            "sharpness_stats": {
                "min": min_sharpness,
                "max": max_sharpness,
                "mean": mean_sharpness,
            },
            "sharpness_curve": results,
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
        print(f"Best Z:       {best_z:.2f} um (sharpness: {best_sharpness:.2f})")
        print(f"After Z:      {best_z:.2f} um (sharpness: {after_sharpness:.2f})")
        print(f"Z adjustment: {best_z - target_z:+.2f} um")
        print(f"Sharpness improvement: {after_sharpness - before_sharpness:+.2f} ({(after_sharpness/before_sharpness - 1)*100:+.1f}%)" if before_sharpness > 0 else "")
        print()
        print("Output files:")
        print(f"  {os.path.join(args.output, 'before.jpg')}")
        print(f"  {os.path.join(args.output, 'after.jpg')}")
        print(f"  {os.path.join(args.output, 'best_scan_frame.jpg')}")
        print(f"  {os.path.join(args.output, 'summary.json')}")
        print()
        print("Done.")

        return 0


if __name__ == "__main__":
    exit(main())
