"""Multi-row snake scan with parallel position reading and image capture."""

import argparse
import bisect
import json
import os
import queue
import shutil
import threading
import time


def interpolate_position(t, samples):
    """Interpolate position at time t from (t_before, t_after, x_um) samples.

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
    t0, x0 = times[idx - 1], samples[idx - 1][2]
    t1, x1 = times[idx], samples[idx][2]

    if t1 == t0:
        return x0

    alpha = (t - t0) / (t1 - t0)
    return x0 + alpha * (x1 - x0)


def main():
    parser = argparse.ArgumentParser(description="Multi-row snake scan")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--margin", type=float, default=1000, help="Margin from stage edges in µm")
    parser.add_argument("--y-overlap-percent", type=float, default=12, help="Y overlap between rows as %% of frame height")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (2 = half dims)")
    parser.add_argument("--compress", action="store_true", help="Create .zip of output directory")
    parser.add_argument("--clean", action="store_true", help="Wipe output directory if it exists")
    parser.add_argument("--write-threads", type=int, default=2, help="Number of image writer threads (default: 2)")
    parser.add_argument("--white-balance", type=str, default="2.51,1.02,1.41",
                        help="White balance as B,G,R gains (default: 2.51,1.02,1.41)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma level (default: 1.0)")
    args = parser.parse_args()

    # Parse white balance
    wb_parts = args.white_balance.split(",")
    if len(wb_parts) != 3:
        print("Error: --white-balance must be 3 comma-separated values (B,G,R)")
        return 1
    try:
        wb_blue, wb_green, wb_red = float(wb_parts[0]), float(wb_parts[1]), float(wb_parts[2])
    except ValueError:
        print("Error: --white-balance values must be numbers")
        return 1

    from PIL import Image as PILImage

    # Check/create output directory before connecting to hardware
    if os.path.exists(args.output):
        if args.clean:
            shutil.rmtree(args.output)
        else:
            print(f"Error: Output directory '{args.output}' already exists. Use --clean to wipe it.")
            return 1
    os.makedirs(args.output)

    from flakefinder.leica import LeicaConnection, Stage, Lamp, Shutter, TID
    from flakefinder.leica.enums import UCAPI_TID, UCAPI_IID, UCAPI_PROP, IID
    from flakefinder.leica.core import find_unit, get_interface_required, get_interface
    from flakefinder.leica.camera import Camera

    print("Area Scan v1 (Snake Pattern)")
    print("=" * 50)

    with LeicaConnection() as conn:
        from LeicaMicrosystems.HardwareModel import Extensions
        Extensions.ExUCAPI.Register()

        # Set up stage
        stage = Stage.from_connection(conn)
        x_bcv = stage.x._bcv  # Native position reader
        x_converter = stage.x._converter  # For native -> um conversion

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

        # Set white balance (per-channel gain)
        gain_blue_prop = properties.FindProperty(UCAPI_PROP.PROP_GAIN_BLUE)
        if gain_blue_prop:
            gain_blue_prop.GetValue().SetValue(wb_blue)
        gain_green_prop = properties.FindProperty(UCAPI_PROP.PROP_GAIN_GREEN)
        if gain_green_prop:
            gain_green_prop.GetValue().SetValue(wb_green)
        gain_red_prop = properties.FindProperty(UCAPI_PROP.PROP_GAIN_RED)
        if gain_red_prop:
            gain_red_prop.GetValue().SetValue(wb_red)

        # Set gamma
        gamma_prop = properties.FindProperty(UCAPI_PROP.PROP_GAMMA_LEVEL)
        if gamma_prop:
            gamma_prop.GetValue().SetValue(args.gamma)

        # Check readout time
        readout_prop = properties.FindProperty(UCAPI_PROP.PROP_IMAGE_READOUT_TIME)
        readout_fps = ""
        if readout_prop:
            readout_time = readout_prop.GetValue().GetValue()
            readout_fps = f", Readout: {readout_time*1000:.1f}ms ({1/readout_time:.0f} fps)"

        # Read camera properties for metadata
        def get_prop_value(prop_id):
            """Get a property value, returns None if not available."""
            prop = properties.FindProperty(prop_id)
            if prop:
                return prop.GetValue().GetValue()
            return None

        def get_prop_index(prop_id):
            """Get a property index value, returns None if not available."""
            prop = properties.FindProperty(prop_id)
            if prop:
                return prop.GetValue().GetIndex()
            return None

        # Frame dimensions in pixels (after binning)
        frame_width_px = get_prop_value(UCAPI_PROP.PROP_LOGICAL_XRESOLUTION)
        frame_height_px = get_prop_value(UCAPI_PROP.PROP_LOGICAL_YRESOLUTION)

        # Pixel size in µm (logical = already accounts for binning and objective)
        # SDK returns meters, convert to µm
        pixel_size_x_um = get_prop_value(UCAPI_PROP.PROP_LOGICAL_PIXEL_XSIZE)
        pixel_size_y_um = get_prop_value(UCAPI_PROP.PROP_LOGICAL_PIXEL_YSIZE)
        if pixel_size_x_um:
            pixel_size_x_um *= 1e6
        if pixel_size_y_um:
            pixel_size_y_um *= 1e6

        # Physical sensor properties (before binning)
        sensor_width_px = get_prop_value(UCAPI_PROP.PROP_SENSOR_XRESOLUTION)
        sensor_height_px = get_prop_value(UCAPI_PROP.PROP_SENSOR_YRESOLUTION)
        physical_pixel_x_um = get_prop_value(UCAPI_PROP.PROP_PHYSICAL_PIXEL_XSIZE)
        physical_pixel_y_um = get_prop_value(UCAPI_PROP.PROP_PHYSICAL_PIXEL_YSIZE)
        if physical_pixel_x_um:
            physical_pixel_x_um *= 1e6
        if physical_pixel_y_um:
            physical_pixel_y_um *= 1e6

        # Exposure and binning
        actual_exposure = get_prop_value(UCAPI_PROP.PROP_EXPOSURE_TIME)
        actual_binning_idx = get_prop_index(UCAPI_PROP.PROP_BINNING_LEVEL)
        binning_map = {0: 1, 1: 2, 2: 3}
        actual_binning = binning_map.get(actual_binning_idx, actual_binning_idx)

        # Readout time
        readout_time = get_prop_value(UCAPI_PROP.PROP_IMAGE_READOUT_TIME)

        # Compute frame size in µm
        # The SDK's "logical pixel size" doesn't account for objective magnification
        # Sample pixel size = physical_pixel × binning / magnification
        frame_width_um = None
        frame_height_um = None
        sample_pixel_x_um = None
        sample_pixel_y_um = None

        # Get current objective from nosepiece
        objective_mag = None
        objective_idx = None
        nosepiece_unit = find_unit(conn.root, TID.MICROSCOPE_NOSEPIECE)
        if nosepiece_unit:
            bcv_iface = get_interface(nosepiece_unit, IID.IID_BASIC_CONTROL_VALUE)
            if bcv_iface:
                objective_idx = bcv_iface.GetControlValue()
                # Map index to magnification
                obj_map = {1: 5, 2: 10, 3: 20, 4: 50, 5: 100, 6: 150}
                objective_mag = obj_map.get(objective_idx)

        # Compute sample-plane pixel size and frame size in µm
        # sample_pixel = physical_pixel × binning / magnification
        if physical_pixel_x_um and actual_binning and objective_mag:
            sample_pixel_x_um = physical_pixel_x_um * actual_binning / objective_mag
            if frame_width_px:
                frame_width_um = frame_width_px * sample_pixel_x_um
        if physical_pixel_y_um and actual_binning and objective_mag:
            sample_pixel_y_um = physical_pixel_y_um * actual_binning / objective_mag
            if frame_height_px:
                frame_height_um = frame_height_px * sample_pixel_y_um

        print(f"Camera: {camera_unit.GetName()}")
        exp_str = f"{actual_exposure*1000:.1f}ms" if actual_exposure else "?"
        print(f"  Trigger: CONTINUOUS, Binning: {actual_binning}x{actual_binning}, Exposure: {exp_str}{readout_fps}")
        print(f"  White balance (B,G,R): {wb_blue}, {wb_green}, {wb_red}")
        print(f"  Gamma: {args.gamma}")
        if frame_width_px and frame_height_px:
            print(f"  Frame: {frame_width_px}x{frame_height_px} px")
        if frame_width_um and frame_height_um:
            print(f"  FOV: {frame_width_um:.2f} x {frame_height_um:.2f} µm")
        if objective_mag:
            print(f"  Objective: {objective_mag}x")
        if args.downsample > 1:
            print(f"  Downsample: {args.downsample}x")

        # Print lighting info
        if lamp:
            print(f"Lamp: {lamp.name}, intensity={lamp.intensity}/{lamp.max_intensity}")
        if shutter:
            print(f"Shutter: {shutter.name}, {'open' if shutter.is_open else 'closed'}")

        # Set up acquisition context
        context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
        current_image = [None]

        def on_image(image):
            current_image[0] = image

        context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_image)

        # Calculate scan area bounds
        margin = args.margin
        x_min = stage.x.min_um + margin
        x_max = stage.x.max_um - margin
        y_min = stage.y.min_um + margin
        y_max = stage.y.max_um - margin
        x_center = (stage.x.min_um + stage.x.max_um) / 2
        y_center = (stage.y.min_um + stage.y.max_um) / 2

        # Calculate row Y positions (top to bottom, -Y direction)
        if not frame_height_um:
            print("Error: Could not determine frame height. Check objective/camera.")
            return 1

        y_step = frame_height_um * (1 - args.y_overlap_percent / 100)
        row_y_positions = []
        y = y_max
        while y >= y_min:
            row_y_positions.append(y)
            y -= y_step

        num_rows = len(row_y_positions)

        print(f"Scan area: X={x_min:.0f}-{x_max:.0f} µm, Y={y_min:.0f}-{y_max:.0f} µm")
        print(f"Frame: {frame_width_um:.1f} x {frame_height_um:.1f} µm")
        print(f"Y step: {y_step:.1f} µm ({args.y_overlap_percent:.0f}% overlap)")
        print(f"Rows: {num_rows}")
        print(f"Output: {args.output}/")
        print()

        # Move to start of first row
        print(f"Moving to start (X={x_min:.0f}, Y={row_y_positions[0]:.0f})...")
        stage.x.move_to(x_min)
        stage.y.move_to(row_y_positions[0])

        # Initialize global metadata
        total_scan_start = time.perf_counter()
        all_position_samples = []  # All position samples across all rows
        global_frame_idx = 0  # Global frame counter

        # Background saver thread
        save_queue = queue.Queue()
        saved_frames_meta = []

        def saver_thread():
            """Background thread: convert, resize, and save frames."""
            while True:
                item = save_queue.get()
                if item is None:  # Poison pill
                    break

                frame_idx, row_idx, t_start, t_end, image, row_y, row_x_samples, t0 = item

                # Convert to numpy and dispose .NET image
                arr = Camera._image_to_numpy(image)
                image.Dispose()

                # Resize if needed
                img = PILImage.fromarray(arr)
                if args.downsample > 1:
                    new_size = (img.width // args.downsample, img.height // args.downsample)
                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)

                # Save
                path = os.path.join(args.output, f"frame_{frame_idx:04d}.jpg")
                img.save(path, quality=95)

                # Compute metadata
                x_start_interp = interpolate_position(t_start, row_x_samples)
                x_end_interp = interpolate_position(t_end, row_x_samples)
                dt = t_end - t_start
                x_vel = (x_end_interp - x_start_interp) / dt if dt > 0 and x_start_interp and x_end_interp else 0

                saved_frames_meta.append({
                    "n": frame_idx,
                    "row": row_idx,
                    "t_start": t_start - t0,
                    "t_end": t_end - t0,
                    "x_start": x_start_interp,
                    "x_end": x_end_interp,
                    "x_vel": x_vel,
                    "y_um": row_y,
                })

                save_queue.task_done()

        savers = []
        for _ in range(args.write_threads):
            t = threading.Thread(target=saver_thread, daemon=True)
            t.start()
            savers.append(t)

        meta = {
            "x_min_um": x_min,
            "x_max_um": x_max,
            "y_min_um": y_min,
            "y_max_um": y_max,
            "y_step_um": y_step,
            "y_overlap_percent": args.y_overlap_percent,
            "downsample": args.downsample,
            # Camera and optics metadata
            "camera": {
                "name": camera_unit.GetName(),
                "exposure_s": actual_exposure,
                "binning": actual_binning,
                "readout_time_s": readout_time,
                "frame_width_px": frame_width_px,
                "frame_height_px": frame_height_px,
                "pixel_size_x_um": pixel_size_x_um,
                "pixel_size_y_um": pixel_size_y_um,
                "sensor_width_px": sensor_width_px,
                "sensor_height_px": sensor_height_px,
                "physical_pixel_x_um": physical_pixel_x_um,
                "physical_pixel_y_um": physical_pixel_y_um,
                "white_balance_bgr": [wb_blue, wb_green, wb_red],
                "gamma": args.gamma,
            },
            "optics": {
                "objective_mag": objective_mag,
                "objective_idx": objective_idx,
                "sample_pixel_x_um": sample_pixel_x_um,
                "sample_pixel_y_um": sample_pixel_y_um,
                "frame_width_um": frame_width_um,
                "frame_height_um": frame_height_um,
            },
            "lighting": {
                "lamp_name": lamp.name if lamp else None,
                "lamp_intensity": lamp.intensity if lamp else None,
                "lamp_max_intensity": lamp.max_intensity if lamp else None,
                "shutter_name": shutter.name if shutter else None,
                "shutter_open": shutter.is_open if shutter else None,
            },
            "rows": [],
        }

        # Scan each row
        for row_idx, row_y in enumerate(row_y_positions):
            # Snake pattern: even rows +X, odd rows -X
            direction = 1 if row_idx % 2 == 0 else -1
            if direction == 1:
                x_start_pos, x_end_pos = x_min, x_max
                dir_str = "+X"
            else:
                x_start_pos, x_end_pos = x_max, x_min
                dir_str = "-X"

            print(f"Row {row_idx}/{num_rows-1}: Y={row_y:.0f}µm, {dir_str}")

            # Move to row start if not already there
            if row_idx > 0:
                stage.y.move_to(row_y)
                stage.x.move_to(x_start_pos)

            # Set up position polling for this row
            x_samples = []
            stop_polling = threading.Event()

            def x_poll_thread():
                while not stop_polling.is_set():
                    t_before = time.perf_counter()
                    x_native = x_bcv.GetControlValue()
                    t_after = time.perf_counter()
                    x_um = x_converter.GetMetricsValue(x_native)
                    x_samples.append((t_before, t_after, x_um))

            # Start position polling
            x_thread = threading.Thread(target=x_poll_thread, daemon=True)
            x_thread.start()

            row_start = time.perf_counter()
            row_frame_start = global_frame_idx
            row_frame_count = 0

            # Start async X move
            handle = stage.x.move_to_async(x_end_pos)

            # Capture frames during move, queue to savers immediately
            while not handle.is_complete:
                t_start = time.perf_counter()
                current_image[0] = None
                acquisition.Acquire(context, None)
                t_end = time.perf_counter()

                if current_image[0] is not None:
                    # Queue frame immediately for background saving
                    save_queue.put((
                        global_frame_idx, row_idx, t_start, t_end, current_image[0],
                        row_y, x_samples, total_scan_start
                    ))
                    global_frame_idx += 1
                    row_frame_count += 1

            row_end = time.perf_counter()
            handle.dispose()

            # Stop position polling
            stop_polling.set()
            x_thread.join(timeout=1.0)

            row_duration = row_end - row_start

            # Filter position samples to row scan period
            row_x_samples = [(t_before, t_after, x) for t_before, t_after, x in x_samples
                            if row_start <= t_before <= row_end]

            print(f"  {row_frame_count} frames, {len(row_x_samples)} pos samples, {row_duration:.2f}s")

            # Add position samples to global list (with adjusted timestamps)
            for t_before, t_after, x_um in row_x_samples:
                all_position_samples.append({
                    "t_before": t_before - total_scan_start,
                    "t_after": t_after - total_scan_start,
                    "x_um": x_um,
                    "row": row_idx,
                })

            # Record row metadata
            meta["rows"].append({
                "row_idx": row_idx,
                "y_um": row_y,
                "direction": direction,
                "frame_start": row_frame_start,
                "frame_end": global_frame_idx,
                "duration_s": row_duration,
                "position_samples": len(row_x_samples),
            })

        total_scan_end = time.perf_counter()
        total_duration = total_scan_end - total_scan_start

        # Return to center while saver finishes
        print()
        print("Returning to center...")
        return_handle_x = stage.x.move_to_async(x_center)
        return_handle_y = stage.y.move_to_async(y_center)

        # Wait for savers to finish
        print(f"Waiting for savers ({save_queue.qsize()} frames queued, {len(savers)} threads)...")
        for _ in savers:
            save_queue.put(None)  # Poison pill for each thread
        for t in savers:
            t.join()
        print(f"Savers done ({len(saved_frames_meta)} frames saved)")

        # Finalize metadata
        # Sort saved_frames_meta by frame index (may be out of order due to threading)
        saved_frames_meta.sort(key=lambda f: f["n"])
        meta["frames"] = saved_frames_meta
        meta["scan_duration_s"] = total_duration
        meta["frame_count"] = global_frame_idx
        meta["position_sample_count"] = len(all_position_samples)
        meta["position_stream"] = all_position_samples

        # Save metadata
        meta_path = os.path.join(args.output, "scan_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")

        # Compress if requested
        if args.compress:
            print(f"Creating {args.output}.zip...")
            shutil.make_archive(args.output, 'zip', args.output)
            print(f"Created {args.output}.zip")

        # Summary stats
        print()
        print("=" * 50)
        print("SCAN SUMMARY:")
        print(f"  Total time: {total_duration:.1f}s")
        print(f"  Rows: {num_rows}")
        print(f"  Total frames: {global_frame_idx}")
        print(f"  Total position samples: {len(all_position_samples)}")
        print(f"  Avg FPS: {global_frame_idx / total_duration:.1f}")

        # Per-row stats
        if meta["rows"]:
            row_frame_counts = [r["frame_end"] - r["frame_start"] for r in meta["rows"]]
            row_durations = [r["duration_s"] for r in meta["rows"]]
            print(f"  Frames/row: avg={sum(row_frame_counts)/len(row_frame_counts):.0f}, "
                  f"min={min(row_frame_counts)}, max={max(row_frame_counts)}")
            print(f"  Row duration: avg={sum(row_durations)/len(row_durations):.2f}s")

        # Wait for return
        return_handle_x.wait()
        return_handle_y.wait()
        return_handle_x.dispose()
        return_handle_y.dispose()
        print()
        print("Done.")

        return 0


if __name__ == "__main__":
    exit(main())
