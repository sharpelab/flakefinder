"""FlakeFinder CLI - Microscope control and flake detection."""

from __future__ import annotations

import argparse
import sys


def cmd_connect(args: argparse.Namespace) -> int:
    """Attempt to connect to the microscope and report status."""
    print("FlakeFinder - Microscope Connection Test")
    print("=" * 40)

    try:
        from .leica import LeicaConnection, TID, print_unit_tree

        config_dir = args.config_dir
        print(f"Config directory: {config_dir or '(default)'}")
        print("Attempting to connect...")
        print()

        with LeicaConnection(config_dir) as conn:
            print(f"Connected: {conn.root.GetName()}")
            print()

            # Find key units
            units_to_find = [
                (TID.MICROSCOPE_STAGE, "Stage"),
                (TID.MICROSCOPE_ZDRIVE, "Z-Drive"),
                (TID.MICROSCOPE_NOSEPIECE, "Nosepiece"),
                (TID.MICROSCOPE_LAMP, "Lamp"),
                (TID.MICROSCOPE_IL_SHUTTER, "IL Shutter"),
            ]

            print("Units found:")
            for tid, label in units_to_find:
                unit = conn.find_unit(tid)
                if unit:
                    print(f"  {label}: {unit.GetName()}")
                else:
                    print(f"  {label}: (not found)")

            if args.tree:
                print()
                print("Unit tree:")
                print_unit_tree(conn.root)

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print()
        print("This usually means pythonnet or the Leica DLLs are not available.")
        print("See docs/setup.md for installation instructions.")
        return 1

    except Exception as e:
        print(f"Connection failed: {e}")
        print()
        print("This is expected if you're not on the microscope PC.")
        print("The driver requires the Leica hardware and DLLs to be present.")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show FlakeFinder version and configuration info."""
    print("FlakeFinder v0.1.0")
    print()
    print("API: flakefinder.leica")
    print("  - Async-capable microscope control")
    print("  - Event-based position monitoring")
    print()
    print("See: https://github.com/sharpelab/flakefinder")
    return 0


def cmd_test_axis(args: argparse.Namespace) -> int:
    """Test Axis class with Z-drive and optionally stage."""
    import time

    try:
        from .leica import LeicaConnection, TID, Axis, Stage, PositionMonitor, AxisEvents, EventQueue

        print("FlakeFinder - Axis Test")
        print("=" * 40)

        with LeicaConnection(args.config_dir) as conn:
            # Test Z-drive
            z_unit = conn.find_unit(TID.MICROSCOPE_ZDRIVE)
            if z_unit is None:
                print("Z-drive not found!")
                return 1

            z = Axis(z_unit)
            print(f"Z-Drive: {z.name}")
            print(f"  Position: {z.position_um:.2f} µm")
            print(f"  Range: {z.min_um:.2f} - {z.max_um:.2f} µm")
            print(f"  Supports async: {z.supports_async}")
            print(f"  Supports halt: {z.supports_halt}")
            print(f"  Supports velocity: {z.supports_velocity}")
            print(f"  Calibrated: {z.is_calibrated}")
            print()

            # Test Stage
            try:
                stage = Stage.from_connection(conn)
                print(f"Stage X: {stage.x.name}")
                print(f"  Position: {stage.x.position_um:.2f} µm")
                print(f"  Range: {stage.x.min_um:.2f} - {stage.x.max_um:.2f} µm")
                print(f"  Supports async: {stage.x.supports_async}")
                print()
                print(f"Stage Y: {stage.y.name}")
                print(f"  Position: {stage.y.position_um:.2f} µm")
                print(f"  Range: {stage.y.min_um:.2f} - {stage.y.max_um:.2f} µm")
                print(f"  Supports async: {stage.y.supports_async}")
                print()
            except LookupError as e:
                print(f"Stage not found: {e}")
                print()

            # Test async move if requested
            if args.test_move:
                # Safety margin from edges (µm)
                MARGIN = 1000

                # First, move Z to center (sync) for safety
                z_mid = (z.min_um + z.max_um) / 2
                print(f"Moving Z to center ({z_mid:.0f} µm) for safety...")
                z.move_to(z_mid)
                print(f"  Z now at: {z.position_um:.2f} µm")
                print()

                # Test async with X axis (safer than Z)
                if not stage.x.supports_async:
                    print("X axis doesn't support async moves")
                    return 1

                x_start = stage.x.position_um
                x_left = stage.x.min_um + MARGIN
                x_right = stage.x.max_um - MARGIN

                print("Testing async X moves...")
                print(f"  Start: {x_start:.2f} µm")
                print(f"  Left target: {x_left:.2f} µm")
                print(f"  Right target: {x_right:.2f} µm")
                print()

                if args.events:
                    # Use event-based position monitoring
                    print("  Using EVENT-BASED position monitoring (blocking on events)")
                    print()

                    events = AxisEvents(stage.x.unit)
                    eq = EventQueue()
                    converter = stage.x._converter

                    with events.subscribe_position(eq.handler):
                        # Move to left edge
                        print("  Moving to left edge (async)...")
                        handle = stage.x.move_to_async(x_left)
                        event_count = 0
                        while not handle.is_complete:
                            pos = eq.get(timeout=0.2)  # Block until event or timeout
                            if pos is not None:
                                event_count += 1
                                pos_um = converter.GetMetricsValue(pos)
                                print(f"    [event {event_count}] pos={pos_um:.0f}µm", flush=True)
                        # Drain any remaining events
                        for pos in eq.drain():
                            event_count += 1
                        print(f"  Final: {stage.x.position_um:.2f} µm ({event_count} events)")
                        print()

                        # Move to right edge
                        print("  Moving to right edge (async)...")
                        handle = stage.x.move_to_async(x_right)
                        event_count = 0
                        while not handle.is_complete:
                            pos = eq.get(timeout=0.2)
                            if pos is not None:
                                event_count += 1
                                pos_um = converter.GetMetricsValue(pos)
                                print(f"    [event {event_count}] pos={pos_um:.0f}µm", flush=True)
                        for pos in eq.drain():
                            event_count += 1
                        print(f"  Final: {stage.x.position_um:.2f} µm ({event_count} events)")
                        print()

                        # Return to start
                        print(f"  Returning to start ({x_start:.2f} µm)...")
                        handle = stage.x.move_to_async(x_start)
                        handle.wait()
                        print(f"  Final: {stage.x.position_um:.2f} µm")
                else:
                    # Use polling-based position monitoring
                    print("  Using POLLING-based position monitoring")
                    print()

                    # Move to left edge
                    print("  Moving to left edge (async)...", end="", flush=True)
                    handle = stage.x.move_to_async(x_left)
                    print(f" [raw state: {handle.state_raw}]")
                    while not handle.is_complete:
                        print(f"    pos={stage.x.position_um:.0f} state={handle.state_raw}", flush=True)
                        time.sleep(0.1)
                    print(f"  Position: {stage.x.position_um:.2f} µm")
                    print(f"  State: {handle.state.name} (raw={handle.state_raw})")
                    print()

                    # Move to right edge
                    print("  Moving to right edge (async)...", end="", flush=True)
                    handle = stage.x.move_to_async(x_right)
                    print(f" [raw state: {handle.state_raw}]")
                    while not handle.is_complete:
                        print(f"    pos={stage.x.position_um:.0f} state={handle.state_raw}", flush=True)
                        time.sleep(0.1)
                    print(f"  Position: {stage.x.position_um:.2f} µm")
                    print(f"  State: {handle.state.name} (raw={handle.state_raw})")
                    print()

                    # Return to start
                    print(f"  Returning to start ({x_start:.2f} µm)...")
                    handle = stage.x.move_to_async(x_start)
                    handle.wait()
                    print(f"  Final: {stage.x.position_um:.2f} µm")

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


def cmd_test_camera(args: argparse.Namespace) -> int:
    """Test Camera class with single-shot and optional streaming."""
    import time

    try:
        from .leica import LeicaConnection, Camera, Stage, Lamp, Shutter

        print("FlakeFinder - Camera Test")
        print("=" * 40)

        with LeicaConnection(args.config_dir) as conn:
            camera = Camera.from_connection(conn)

            # Set up shutter (open by default)
            try:
                shutter = Shutter.from_connection(conn)
                shutter.open()
                print(f"Shutter: {shutter.name} ({'open' if shutter.is_open else 'closed'})")
            except LookupError:
                print("Shutter: (not found)")

            # Set up lamp
            try:
                lamp = Lamp.from_connection(conn)
                if args.lamp is not None:
                    lamp.intensity = args.lamp
                else:
                    lamp.full()  # Default to full
                print(f"Lamp: {lamp.name}")
                print(f"  Intensity: {lamp.intensity}/{lamp.max_intensity}")
            except LookupError:
                print("Lamp: (not found)")

            # Apply settings from CLI
            if args.exposure_ms is not None:
                camera.exposure_time = args.exposure_ms / 1000.0
            if args.gain is not None:
                camera.gain = args.gain

            print(f"Camera: {camera.name}")
            print(f"  Exposure: {camera.exposure_time * 1000:.1f}ms")
            print(f"  Gain: {camera.gain}")
            print(f"  Binning: {camera.binning} (0=1x1, 1=2x2, 2=3x3)")
            print()

            # Single-shot test
            print("Taking single image...")
            start = time.monotonic()
            image = camera.capture()
            elapsed = time.monotonic() - start
            print(f"  Captured: {image.shape} in {elapsed:.3f}s")
            print()

            # Streaming test
            if args.stream:
                import os
                from PIL import Image as PILImage

                duration = args.stream_duration
                print(f"Streaming for {duration}s...")

                # Get stage for position tagging if available
                try:
                    stage = Stage.from_connection(conn)
                    print(f"  Position tagging enabled (stage found)")
                except LookupError:
                    stage = None
                    print(f"  No stage found, positions will be None")

                # Prepare output directory if saving frames
                save_frames = args.output is not None
                if save_frames:
                    base, ext = os.path.splitext(args.output)
                    if not ext:
                        ext = ".jpg"
                    print(f"  Saving frames to {base}_NNN{ext}")

                saved_count = 0
                first_shape = None
                with camera.stream(stage) as stream:
                    start = time.monotonic()
                    while (time.monotonic() - start) < duration:
                        frame = stream.get_frame(timeout=0.5)
                        if frame:
                            # Only log first frame and every 10th unless quiet
                            if not args.quiet and (frame.frame_number == 0 or frame.frame_number % 10 == 0):
                                pos_str = (
                                    f"({frame.position[0]:.0f}, {frame.position[1]:.0f})"
                                    if frame.position
                                    else "None"
                                )
                                print(
                                    f"  Frame {frame.frame_number}: "
                                    f"{frame.image.shape} @ {pos_str}"
                                )
                            if first_shape is None:
                                first_shape = frame.image.shape
                            # Save frame if output specified
                            if save_frames:
                                frame_path = f"{base}_{frame.frame_number:03d}{ext}"
                                PILImage.fromarray(frame.image).save(frame_path)
                                saved_count += 1

                print()
                if first_shape:
                    print(f"  Frame size: {first_shape}")
                print(f"  Captured: {stream.frames_captured} frames")
                print(f"  Dropped: {stream.frames_dropped} frames")
                print(f"  Rate: {stream.frame_rate:.1f} fps")
                if save_frames:
                    print(f"  Saved: {saved_count} frames")
                return 0  # Skip single-image save below

            # Save image if requested
            if args.output:
                try:
                    from PIL import Image as PILImage

                    pil_image = PILImage.fromarray(image)
                    pil_image.save(args.output)
                    print(f"Saved to: {args.output}")
                except ImportError:
                    print("Install PIL to save images: pip install Pillow")

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        import traceback

        print(f"Error: {e}")
        traceback.print_exc()
        return 1


def cmd_scan_line(args: argparse.Namespace) -> int:
    """Scan a line along X axis, capturing frames at max rate."""
    import json
    import os
    import time

    try:
        from PIL import Image as PILImage
        from .leica import LeicaConnection, Camera, Stage, Lamp, Shutter
        from .leica.camera import DeferredFrameStream

        print("FlakeFinder - Line Scan")
        print("=" * 40)

        with LeicaConnection(args.config_dir) as conn:
            # Set up hardware
            stage = Stage.from_connection(conn)
            camera = Camera.from_connection(conn)

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

            # Configure camera
            camera.exposure_time = args.exposure_ms / 1000.0
            camera.binning = 2  # 3x3 for speed

            print(f"Camera: {camera.name}")
            print(f"  Exposure: {args.exposure_ms}ms, Binning: 3x3")

            # Determine scan range
            margin = 1000  # µm from edges
            x_min = stage.x.min_um + margin
            x_max = stage.x.max_um - margin
            x_start = stage.x.position_um

            print(f"Stage X range: {x_min:.0f} - {x_max:.0f} µm")
            print(f"  Current: {x_start:.0f} µm")

            # Prepare output directory
            os.makedirs(args.output, exist_ok=True)

            # Move to start position (sync)
            print(f"\nMoving to start ({x_min:.0f} µm)...")
            stage.x.move_to(x_min)
            print(f"  At: {stage.x.position_um:.0f} µm")

            # Record metadata
            meta = {
                "start_pos_um": stage.x.position_um,
                "end_pos_um": x_max,
                "exposure_ms": args.exposure_ms,
                "binning": 2,
                "frames": [],
            }

            # Start deferred streaming (keeps images in .NET memory)
            print(f"\nScanning to {x_max:.0f} µm...")
            scan_start = time.monotonic()
            meta["start_time"] = scan_start

            with camera.deferred_stream() as stream:
                # Start async move
                handle = stage.x.move_to_async(x_max)

                # Wait for move to complete (images accumulate in .NET memory)
                handle.wait()

            scan_end = time.monotonic()
            frame_count = stream.frames_captured

            print(f"  Captured {frame_count} frames in {scan_end - scan_start:.2f}s")
            print(f"  Acquisition FPS: {stream.frame_rate:.1f}")
            print(f"\nConverting and saving frames...")

            # Now bulk convert and save
            save_start = time.monotonic()
            for i, (timestamp, image) in enumerate(stream.get_all_frames()):
                path = os.path.join(args.output, f"frame_{i:04d}.jpg")
                PILImage.fromarray(image).save(path, quality=95)
                meta["frames"].append({
                    "n": i,
                    "t": timestamp - scan_start,
                })
                if (i + 1) % 20 == 0:
                    print(f"  Saved {i + 1}/{frame_count}...")

            save_end = time.monotonic()

            meta["end_time"] = scan_end
            meta["end_pos_actual_um"] = stage.x.position_um
            meta["duration_s"] = scan_end - scan_start
            meta["frame_count"] = frame_count
            meta["fps"] = frame_count / (scan_end - scan_start) if scan_end > scan_start else 0

            # Save metadata
            meta_path = os.path.join(args.output, "scan_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"\nScan complete:")
            print(f"  Frames: {frame_count}")
            print(f"  Scan duration: {meta['duration_s']:.1f}s")
            print(f"  Acquisition FPS: {meta['fps']:.1f}")
            print(f"  Save time: {save_end - save_start:.1f}s")
            print(f"  Output: {args.output}/")

            # Return to start
            if args.return_home:
                print(f"\nReturning to {x_start:.0f} µm...")
                stage.x.move_to(x_start)

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


def cmd_center(args: argparse.Namespace) -> int:
    """Center the stage X/Y."""
    try:
        from .leica import LeicaConnection, Stage

        with LeicaConnection(args.config_dir) as conn:
            stage = Stage.from_connection(conn)

            x_center = (stage.x.min_um + stage.x.max_um) / 2
            y_center = (stage.y.min_um + stage.y.max_um) / 2

            print(f"Centering stage to ({x_center:.0f}, {y_center:.0f}) µm...")

            # Move both axes
            stage.x.move_to(x_center)
            stage.y.move_to(y_center)

            print(f"Done: ({stage.x.position_um:.0f}, {stage.y.position_um:.0f}) µm")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_raster_scan(args: argparse.Namespace) -> int:
    """Full snake raster scan of the stage area."""
    import json
    import os
    import signal
    import time

    # Track stage for Ctrl+C cleanup
    stage_ref = [None]
    x_center_ref = [None]
    y_center_ref = [None]
    interrupted = [False]

    def signal_handler(sig, frame):
        interrupted[0] = True
        print("\n\nInterrupted! Finishing current row then saving metadata...")

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        from PIL import Image as PILImage
        from .leica import LeicaConnection, Camera, Stage, Lamp, Shutter

        print("FlakeFinder - Raster Scan")
        print("=" * 40)

        with LeicaConnection(args.config_dir) as conn:
            # Set up hardware
            stage = Stage.from_connection(conn)
            stage_ref[0] = stage
            camera = Camera.from_connection(conn)

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

            # Configure camera
            camera.exposure_time = args.exposure_ms / 1000.0
            camera.binning = 2  # 3x3 for speed

            print(f"Camera: {camera.name}")
            print(f"  Exposure: {args.exposure_ms}ms, Binning: 3x3")
            if args.downsample > 1:
                print(f"  Downsample: {args.downsample}x, JPEG quality: {args.jpeg_quality}")
            if args.endpoints_only:
                print("  Mode: endpoints only (first/last frame per row)")

            # Determine scan range
            margin = args.margin
            x_min = stage.x.min_um + margin
            x_max = stage.x.max_um - margin
            y_min = stage.y.min_um + margin
            y_max = stage.y.max_um - margin

            # Calculate rows based on step size
            y_step = args.y_step
            num_rows = int((y_max - y_min) / y_step) + 1

            print(f"Stage X range: {x_min:.0f} - {x_max:.0f} µm")
            print(f"Stage Y range: {y_min:.0f} - {y_max:.0f} µm")
            print(f"Y step: {y_step:.0f} µm, Rows: {num_rows}")

            # Save center position for return
            x_center = (stage.x.min_um + stage.x.max_um) / 2
            y_center = (stage.y.min_um + stage.y.max_um) / 2
            x_center_ref[0] = x_center
            y_center_ref[0] = y_center

            # Prepare output directory
            os.makedirs(args.output, exist_ok=True)

            # Metadata
            meta = {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "y_step": y_step,
                "num_rows": num_rows,
                "exposure_ms": args.exposure_ms,
                "binning": 2,
                "endpoints_only": args.endpoints_only,
                "downsample": args.downsample,
                "jpeg_quality": args.jpeg_quality,
                "rows": [],
            }

            # Move to starting position
            print(f"\nMoving to start ({x_min:.0f}, {y_min:.0f}) µm...")
            stage.x.move_to(x_min)
            stage.y.move_to(y_min)

            total_frames = 0
            total_saved = 0
            scan_start = time.monotonic()

            # Snake raster scan
            for row in range(num_rows):
                if interrupted[0]:
                    break

                y_pos = y_min + row * y_step
                if y_pos > y_max:
                    break

                # Move to Y position
                stage.y.move_to(y_pos)

                # Determine X direction (snake pattern)
                if row % 2 == 0:
                    x_start, x_end = x_min, x_max
                    direction = "forward"
                else:
                    x_start, x_end = x_max, x_min
                    direction = "reverse"

                # Move to X start
                stage.x.move_to(x_start)

                # Record actual positions for velocity calculation
                x_actual_start = stage.x.position_um
                y_actual = stage.y.position_um

                row_meta = {
                    "row": row,
                    "y_target": y_pos,
                    "y_actual": y_actual,
                    "x_start_target": x_start,
                    "x_end_target": x_end,
                    "x_actual_start": x_actual_start,
                    "direction": direction,
                    "frames": [],
                }

                print(f"Row {row + 1}/{num_rows}: Y={y_pos:.0f}µm, {direction}...", end=" ", flush=True)

                row_start_time = time.monotonic()
                row_frames = 0
                first_frame = None
                last_frame = None

                # Stream while moving X
                with camera.stream() as stream:
                    handle = stage.x.move_to_async(x_end)

                    while not handle.is_complete and not interrupted[0]:
                        frame = stream.get_frame(timeout=0.1)
                        if frame:
                            if args.endpoints_only:
                                # Keep first and update last
                                if first_frame is None:
                                    pos = (stage.x.position_um, stage.y.position_um) if args.record_positions else None
                                    first_frame = (frame.image.copy(), frame.timestamp, pos)
                                pos = (stage.x.position_um, stage.y.position_um) if args.record_positions else None
                                last_frame = (frame.image.copy(), frame.timestamp, pos)
                            else:
                                # Save every frame
                                path = os.path.join(args.output, f"row_{row:03d}_frame_{row_frames:05d}.jpg")
                                img = PILImage.fromarray(frame.image)
                                if args.downsample > 1:
                                    new_size = (img.width // args.downsample, img.height // args.downsample)
                                    img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                                img.save(path, quality=args.jpeg_quality)
                                frame_meta = {
                                    "n": row_frames,
                                    "t": frame.timestamp - scan_start,
                                }
                                if args.record_positions:
                                    frame_meta["x"] = stage.x.position_um
                                    frame_meta["y"] = stage.y.position_um
                                row_meta["frames"].append(frame_meta)
                                total_saved += 1
                            total_frames += 1
                            row_frames += 1

                # Record actual end position and time
                row_end_time = time.monotonic()
                x_actual_end = stage.x.position_um

                # Save endpoints if in that mode
                if args.endpoints_only and first_frame and last_frame:
                    # Save first
                    path = os.path.join(args.output, f"row_{row:03d}_first.jpg")
                    img = PILImage.fromarray(first_frame[0])
                    if args.downsample > 1:
                        new_size = (img.width // args.downsample, img.height // args.downsample)
                        img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                    img.save(path, quality=args.jpeg_quality)
                    first_meta = {"type": "first", "t": first_frame[1] - scan_start}
                    if first_frame[2]:
                        first_meta["x"], first_meta["y"] = first_frame[2]
                    row_meta["frames"].append(first_meta)
                    total_saved += 1

                    # Save last
                    path = os.path.join(args.output, f"row_{row:03d}_last.jpg")
                    img = PILImage.fromarray(last_frame[0])
                    if args.downsample > 1:
                        new_size = (img.width // args.downsample, img.height // args.downsample)
                        img = img.resize(new_size, PILImage.Resampling.LANCZOS)
                    img.save(path, quality=args.jpeg_quality)
                    last_meta = {"type": "last", "t": last_frame[1] - scan_start}
                    if last_frame[2]:
                        last_meta["x"], last_meta["y"] = last_frame[2]
                    row_meta["frames"].append(last_meta)
                    total_saved += 1

                row_elapsed = row_end_time - row_start_time
                row_fps = row_frames / row_elapsed if row_elapsed > 0 else 0
                print(f"{row_frames} frames, {row_fps:.1f} fps")

                # Calculate velocity for this row
                row_velocity = abs(x_actual_end - x_actual_start) / row_elapsed if row_elapsed > 0 else 0

                row_meta["x_actual_end"] = x_actual_end
                row_meta["t_start"] = row_start_time - scan_start
                row_meta["t_end"] = row_end_time - scan_start
                row_meta["frame_count"] = row_frames
                row_meta["duration_s"] = row_elapsed
                row_meta["fps"] = row_fps
                row_meta["velocity_um_per_s"] = row_velocity
                meta["rows"].append(row_meta)

            scan_end = time.monotonic()
            total_duration = scan_end - scan_start
            avg_fps = total_frames / total_duration if total_duration > 0 else 0

            meta["total_frames"] = total_frames
            meta["total_saved"] = total_saved
            meta["total_duration_s"] = total_duration
            meta["avg_fps"] = avg_fps
            meta["interrupted"] = interrupted[0]

            # Save metadata
            meta_path = os.path.join(args.output, "scan_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            status = "interrupted" if interrupted[0] else "complete"
            print(f"\nScan {status}:")
            print(f"  Total frames: {total_frames}")
            print(f"  Saved: {total_saved}")
            print(f"  Total duration: {total_duration:.1f}s")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Output: {args.output}/")

            # Return to center
            print(f"\nReturning to center ({x_center:.0f}, {y_center:.0f}) µm...")
            stage.x.move_to(x_center)
            stage.y.move_to(y_center)
            print("Done.")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    finally:
        signal.signal(signal.SIGINT, original_handler)


def main() -> int:
    """Main entry point for the FlakeFinder CLI."""
    parser = argparse.ArgumentParser(
        prog="flakefinder",
        description="Automated 2D material flake detection system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # connect command
    connect_parser = subparsers.add_parser(
        "connect",
        help="Test microscope connection",
    )
    connect_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    connect_parser.add_argument(
        "--tree",
        action="store_true",
        help="Print full unit tree",
    )
    connect_parser.set_defaults(func=cmd_connect)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show version and configuration info",
    )
    info_parser.set_defaults(func=cmd_info)

    # test-axis command
    axis_parser = subparsers.add_parser(
        "test-axis",
        help="Test Axis/Stage classes (Phase 2)",
    )
    axis_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    axis_parser.add_argument(
        "--test-move",
        action="store_true",
        help="Actually move the X axis to test async",
    )
    axis_parser.add_argument(
        "--events",
        action="store_true",
        help="Use event-based position monitoring (requires --test-move)",
    )
    axis_parser.set_defaults(func=cmd_test_axis)

    # test-camera command
    camera_parser = subparsers.add_parser(
        "test-camera",
        help="Test Camera class (single-shot and streaming)",
    )
    camera_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    camera_parser.add_argument(
        "--stream",
        action="store_true",
        help="Test continuous streaming",
    )
    camera_parser.add_argument(
        "--stream-duration",
        type=float,
        default=3.0,
        help="Streaming test duration in seconds (default: 3.0)",
    )
    camera_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save captured image to file (e.g., test.png)",
    )
    camera_parser.add_argument(
        "--exposure-ms",
        type=float,
        default=10.0,
        help="Exposure time in milliseconds (default: 10)",
    )
    camera_parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Camera gain (default: 1.0)",
    )
    camera_parser.add_argument(
        "--lamp",
        type=int,
        default=None,
        help="Lamp intensity (default: max)",
    )
    camera_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-frame logging during streaming (for fps testing)",
    )
    camera_parser.set_defaults(func=cmd_test_camera)

    # scan-line command
    scan_parser = subparsers.add_parser(
        "scan-line",
        help="Scan a line along X axis, capturing frames",
    )
    scan_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    scan_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="scan_output",
        help="Output directory for frames (default: scan_output)",
    )
    scan_parser.add_argument(
        "--exposure-ms",
        type=float,
        default=1.0,
        help="Exposure time in milliseconds (default: 1.0)",
    )
    scan_parser.add_argument(
        "--return-home",
        action="store_true",
        help="Return to starting position after scan",
    )
    scan_parser.set_defaults(func=cmd_scan_line)

    # center command
    center_parser = subparsers.add_parser(
        "center",
        help="Center the stage X/Y",
    )
    center_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    center_parser.set_defaults(func=cmd_center)

    # raster-scan command
    raster_parser = subparsers.add_parser(
        "raster-scan",
        help="Full snake raster scan of the stage",
    )
    raster_parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to Leica hardware model config directory",
    )
    raster_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="raster_output",
        help="Output directory for frames (default: raster_output)",
    )
    raster_parser.add_argument(
        "--exposure-ms",
        type=float,
        default=1.0,
        help="Exposure time in milliseconds (default: 1.0)",
    )
    raster_parser.add_argument(
        "--y-step",
        type=float,
        default=1000.0,
        help="Y step between rows in µm (default: 1000)",
    )
    raster_parser.add_argument(
        "--margin",
        type=float,
        default=1000.0,
        help="Margin from stage edges in µm (default: 1000)",
    )
    raster_parser.add_argument(
        "--endpoints-only",
        action="store_true",
        help="Only save first and last frame from each row",
    )
    raster_parser.add_argument(
        "--record-positions",
        action="store_true",
        help="Record X position for each frame (reduces fps from ~20 to ~12)",
    )
    raster_parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor (e.g., 4 = quarter resolution, default: 1)",
    )
    raster_parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85)",
    )
    raster_parser.set_defaults(func=cmd_raster_scan)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
