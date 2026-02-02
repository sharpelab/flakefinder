"""Test parallel position reading and image capture with interpolation."""

import argparse
import bisect
import json
import os
import shutil
import threading
import time


def interpolate_position(t, samples):
    """Interpolate position at time t from (t, x_um) samples."""
    if not samples:
        return None

    times = [s[0] for s in samples]

    # Find insertion point
    idx = bisect.bisect_left(times, t)

    if idx == 0:
        return samples[0][1]  # Before first sample
    if idx >= len(samples):
        return samples[-1][1]  # After last sample

    # Linear interpolate between samples[idx-1] and samples[idx]
    t0, x0 = samples[idx - 1][0], samples[idx - 1][1]
    t1, x1 = samples[idx][0], samples[idx][1]

    if t1 == t0:
        return x0

    alpha = (t - t0) / (t1 - t0)
    return x0 + alpha * (x1 - x0)


def main():
    parser = argparse.ArgumentParser(description="Test parallel position reading and capture")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--margin", type=float, default=1000, help="Margin from edges in µm")
    parser.add_argument("--duration", type=float, default=10.0, help="Max duration in seconds")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (2 = half dims)")
    parser.add_argument("--compress", action="store_true", help="Create .zip of output directory")
    parser.add_argument("--clean", action="store_true", help="Wipe output directory if it exists")
    args = parser.parse_args()

    from PIL import Image as PILImage

    # Check/create output directory before connecting to hardware
    if os.path.exists(args.output):
        if args.clean:
            shutil.rmtree(args.output)
        else:
            print(f"Error: Output directory '{args.output}' already exists. Use --clean to wipe it.")
            return 1
    os.makedirs(args.output)

    from flakefinder.leica import LeicaConnection, Stage, Lamp, Shutter
    from flakefinder.leica.enums import UCAPI_TID, UCAPI_IID, UCAPI_PROP, IID
    from flakefinder.leica.core import find_unit, get_interface_required
    from flakefinder.leica.camera import Camera

    print("Row Scan v1")
    print("=" * 50)

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up stage
        stage = Stage.from_connection(conn)
        x_bcv = stage.x._bcv  # Native position reader
        x_converter = stage.x._converter  # For native -> um conversion

        # Set up lighting
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
        camera_unit = find_unit(conn.root, UCAPI_TID.UCAPI_CAMERA)
        if camera_unit is None:
            print("Camera not found!")
            return 1

        camera_unit.Init()
        acquisition = get_interface_required(camera_unit, UCAPI_IID.IID_IMAGE_ACQUISITION)
        properties = get_interface_required(camera_unit, IID.IID_PROPERTIES)

        # Configure camera
        # Set trigger mode to CONTINUOUS (index 0) for faster capture
        trigger_prop = properties.FindProperty(UCAPI_PROP.PROP_IMAGE_TRIGGER_MODE)
        if trigger_prop:
            trigger_prop.GetValue().SetIndex(0)

        binning_prop = properties.FindProperty(UCAPI_PROP.PROP_BINNING_LEVEL)
        if binning_prop:
            binning_prop.GetValue().SetIndex(2)
        exposure_prop = properties.FindProperty(UCAPI_PROP.PROP_EXPOSURE_TIME)
        if exposure_prop:
            exposure_prop.GetValue().SetValue(0.001)

        # Check readout time
        readout_prop = properties.FindProperty(UCAPI_PROP.PROP_IMAGE_READOUT_TIME)
        readout_fps = ""
        if readout_prop:
            readout_time = readout_prop.GetValue().GetValue()
            readout_fps = f", Readout: {readout_time*1000:.1f}ms ({1/readout_time:.0f} fps)"

        print(f"Camera: {camera_unit.GetName()}")
        print(f"  Trigger: CONTINUOUS, Binning: 3x3, Exposure: 1ms{readout_fps}")
        if args.downsample > 1:
            print(f"  Downsample: {args.downsample}x")

        # Set up acquisition context
        context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # Calculate scan range
        margin = args.margin
        x_min = stage.x.min_um + margin
        x_max = stage.x.max_um - margin
        x_center = (stage.x.min_um + stage.x.max_um) / 2
        y_pos = stage.y.position_um  # Record Y position

        print(f"Stage X range: {x_min:.0f} - {x_max:.0f} µm")
        print(f"Stage Y: {y_pos:.0f} µm")
        print(f"Output: {args.output}/")
        print()

        # Move to start
        print(f"Moving to start ({x_min:.0f} µm)...")
        stage.x.move_to(x_min)

        # Background thread state - store (t, x_um) directly
        x_samples = []  # (t, x_um)
        stop_polling = threading.Event()

        def x_poll_thread():
            """Background thread: busy-poll X position."""
            while not stop_polling.is_set():
                t = time.perf_counter()
                x_native = x_bcv.GetControlValue()
                x_um = x_converter.GetMetricsValue(x_native)
                x_samples.append((t, x_um))

        # Main thread state - keep .NET images
        frames = []  # (t_start, t_end, .NET image)

        print(f"Starting scan...")
        print()

        # Start background position polling
        x_thread = threading.Thread(target=x_poll_thread, daemon=True)
        x_thread.start()

        scan_start = time.perf_counter()

        # Start async move
        handle = stage.x.move_to_async(x_max)

        # Main thread: serial capture
        while not handle.is_complete and (time.perf_counter() - scan_start) < args.duration:
            t_start = time.perf_counter()
            current_image[0] = None
            acquisition.Acquire(context, None)
            t_end = time.perf_counter()

            if current_image[0] is not None:
                frames.append((t_start, t_end, current_image[0]))

                if len(frames) <= 5 or len(frames) % 50 == 0:
                    print(f"  Frame {len(frames)}: t={t_start - scan_start:.3f}s")

        scan_end = time.perf_counter()

        # Dispose scan handle and start async return to center
        handle.dispose()
        print()
        print(f"Returning to center (async)...")
        return_handle = stage.x.move_to_async(x_center)

        # Stop background thread
        stop_polling.set()
        x_thread.join(timeout=1.0)

        scan_duration = scan_end - scan_start

        print(f"Scan complete: {scan_duration:.2f}s, {len(frames)} frames, {len(x_samples)} position samples")
        print()

        # Filter position samples to scan period
        scan_x_samples = [(t, x) for t, x in x_samples if scan_start <= t <= scan_end]

        # Build metadata
        meta = {
            "scan_duration_s": scan_duration,
            "x_min_um": x_min,
            "x_max_um": x_max,
            "y_um": y_pos,
            "frame_count": len(frames),
            "position_sample_count": len(scan_x_samples),
            "downsample": args.downsample,
            "position_stream": [
                {"t": t - scan_start, "x_um": x_um}
                for t, x_um in scan_x_samples
            ],
            "frames": [],
        }

        # Convert and save frames with interpolated positions
        print("Converting and saving frames...")

        for i, (t_start, t_end, image) in enumerate(frames):
            # Convert to numpy
            arr = Camera._image_to_numpy(image)
            image.Dispose()

            # Downsample if requested
            img = PILImage.fromarray(arr)
            if args.downsample > 1:
                new_size = (img.width // args.downsample, img.height // args.downsample)
                img = img.resize(new_size, PILImage.Resampling.LANCZOS)

            # Save frame
            path = os.path.join(args.output, f"frame_{i:04d}.jpg")
            img.save(path, quality=95)

            # Interpolate position at start and end
            x_start = interpolate_position(t_start, scan_x_samples)
            x_end = interpolate_position(t_end, scan_x_samples)

            # Calculate velocity (µm/s)
            dt = t_end - t_start
            x_vel = (x_end - x_start) / dt if dt > 0 and x_start and x_end else 0

            meta["frames"].append({
                "n": i,
                "t_start": t_start - scan_start,
                "t_end": t_end - scan_start,
                "x_start": x_start,
                "x_end": x_end,
                "x_vel": x_vel,
            })

            if (i + 1) % 50 == 0:
                print(f"  Saved {i + 1}/{len(frames)}...")

        # Save metadata
        meta_path = os.path.join(args.output, "scan_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved metadata to {meta_path}")

        # Compress if requested
        if args.compress:
            print(f"  Creating {args.output}.zip...")
            shutil.make_archive(args.output, 'zip', args.output)
            print(f"  Created {args.output}.zip")

        print()

        # Stats
        print("=" * 50)
        print("CAPTURE:")
        print(f"  Frames: {len(frames)}")
        print(f"  FPS: {len(frames) / scan_duration:.1f}")

        if len(frames) > 1:
            acquire_times = [f[1] - f[0] for f in frames]
            intervals = [frames[i+1][0] - frames[i][0] for i in range(len(frames)-1)]
            print(f"  Acquire: avg={sum(acquire_times)/len(acquire_times)*1000:.1f}ms")
            print(f"  Interval: avg={sum(intervals)/len(intervals)*1000:.1f}ms, min={min(intervals)*1000:.1f}ms, max={max(intervals)*1000:.1f}ms")
            # Debug: show outlier intervals
            avg_interval = sum(intervals) / len(intervals)
            long_intervals = [(i, intervals[i]*1000) for i in range(len(intervals)) if intervals[i] > avg_interval * 1.5]
            if long_intervals:
                print(f"  Long intervals (>1.5x avg): {[(i, f'{ms:.1f}ms') for i, ms in long_intervals]}")

        print()
        print("POSITION:")
        print(f"  Samples: {len(scan_x_samples)}")

        if len(scan_x_samples) > 1:
            pos_duration = scan_x_samples[-1][0] - scan_x_samples[0][0]
            print(f"  FPS: {len(scan_x_samples) / pos_duration:.1f}")

            pos_intervals = [scan_x_samples[i+1][0] - scan_x_samples[i][0] for i in range(len(scan_x_samples)-1)]
            print(f"  Interval: avg={sum(pos_intervals)/len(pos_intervals)*1000:.2f}ms, min={min(pos_intervals)*1000:.2f}ms, max={max(pos_intervals)*1000:.2f}ms")

            x_vals = [x for t, x in scan_x_samples]
            print(f"  X range: {min(x_vals):.0f} - {max(x_vals):.0f} µm")

        # Interpolation accuracy check
        if len(meta["frames"]) > 1:
            x_starts = [f["x_start"] for f in meta["frames"] if f["x_start"]]
            velocities = [f["x_vel"] for f in meta["frames"] if f["x_vel"]]
            if len(x_starts) > 1:
                x_deltas = [x_starts[i+1] - x_starts[i] for i in range(len(x_starts)-1)]
                print()
                print("INTERPOLATED FRAME POSITIONS:")
                print(f"  X range: {min(x_starts):.0f} - {max(x_starts):.0f} µm")
                print(f"  X delta: avg={sum(x_deltas)/len(x_deltas):.1f}µm, min={min(x_deltas):.1f}µm, max={max(x_deltas):.1f}µm")
            if velocities:
                print(f"  Velocity: avg={sum(velocities)/len(velocities)/1000:.1f}mm/s, min={min(velocities)/1000:.1f}mm/s, max={max(velocities)/1000:.1f}mm/s")

        # Wait for return to center
        print()
        if not return_handle.is_complete:
            print("Waiting for stage to return to center...")
        return_handle.wait()
        return_handle.dispose()
        print("Done.")

        context.Dispose()
        return 0


if __name__ == "__main__":
    exit(main())
