#!/usr/bin/env python3
"""Test minimum exposure time capabilities of the Leica K5C camera.

Determines if sub-1ms exposures are viable for continuous scanning with
minimal motion blur. Tests various exposure times and compares mean brightness.

Usage:
    uv run python scripts/test_min_exposure.py
    uv run python scripts/test_min_exposure.py --exposures 0.001,0.0005,0.0001
    uv run python scripts/test_min_exposure.py --frames 10 --lamp 128
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from flakefinder.leica import LeicaConnection, Camera, Lamp, Shutter


def test_exposure(
    camera: Camera,
    exposure_s: float,
    num_frames: int = 5,
    save_dir: str | None = None,
) -> dict:
    """Test a specific exposure time and collect statistics.

    Args:
        camera: Initialized Camera instance.
        exposure_s: Exposure time in seconds.
        num_frames: Number of frames to capture.
        save_dir: Directory to save sample image (if provided).

    Returns:
        Dictionary with test results.
    """
    camera.exposure_time = exposure_s
    actual_exposure = camera.exposure_time

    # Let camera settle after exposure change
    time.sleep(0.05)

    print(f"  Requested: {exposure_s*1000:.3f} ms, Actual: {actual_exposure*1000:.3f} ms")

    brightness_values = []
    capture_times = []
    saved_path = None

    for i in range(num_frames):
        t0 = time.perf_counter()
        frame = camera.capture()
        t1 = time.perf_counter()

        # Calculate mean brightness (average across all channels)
        mean_brightness = float(np.mean(frame))
        brightness_values.append(mean_brightness)
        capture_times.append(t1 - t0)

        print(f"    Frame {i+1}: brightness={mean_brightness:.1f}, "
              f"capture_time={capture_times[-1]*1000:.1f}ms")

        # Save first frame as sample
        if i == 0 and save_dir is not None:
            filename = f"exposure_{exposure_s*1000:.3f}ms.jpg"
            saved_path = os.path.join(save_dir, filename)
            img = Image.fromarray(frame)
            img.save(saved_path, quality=95)
            print(f"    Saved: {saved_path}")

    return {
        "requested_exposure_s": exposure_s,
        "actual_exposure_s": actual_exposure,
        "num_frames": num_frames,
        "mean_brightness": float(np.mean(brightness_values)),
        "std_brightness": float(np.std(brightness_values)),
        "brightness_values": brightness_values,
        "mean_capture_time_s": float(np.mean(capture_times)),
        "capture_times_s": capture_times,
        "saved_path": saved_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test minimum camera exposure times"
    )
    parser.add_argument(
        "--exposures",
        type=str,
        default="0.002,0.0015,0.001,0.0005,0.0002,0.0001",
        help="Comma-separated exposure times in seconds (default: 2ms,1.5ms,1ms,0.5ms,0.2ms,0.1ms)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames per exposure setting (default: 5)"
    )
    parser.add_argument(
        "--lamp",
        type=int,
        default=255,
        help="Lamp intensity (default: 255 = 100%%)"
    )
    parser.add_argument(
        "--binning",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Binning level: 0=1x1, 1=2x2, 2=3x3 (default: 2)"
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Camera gain multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exposure_test_images",
        help="Directory to save sample images (default: exposure_test_images)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse exposure times
    exposure_times = [float(x.strip()) for x in args.exposures.split(",")]

    print("=" * 60)
    print("Leica K5C Minimum Exposure Time Test")
    print("=" * 60)
    print(f"Testing exposures: {[f'{e*1000:.3f}ms' for e in exposure_times]}")
    print(f"Frames per test: {args.frames}")
    print(f"Lamp intensity: {args.lamp}")
    print(f"Binning: {args.binning}")
    print(f"Gain: {args.gain}")
    print(f"Output directory: {args.output_dir}")
    print()

    results = []

    with LeicaConnection() as conn:
        # Set up lamp
        try:
            lamp = Lamp.from_connection(conn)
            lamp.intensity = args.lamp
            print(f"Lamp set to: {lamp.intensity}/{lamp.max_intensity}")
        except LookupError:
            print("Warning: Lamp not available")

        # Open shutter
        try:
            shutter = Shutter.from_connection(conn)
            if not shutter.is_open:
                shutter.open()
                print("Shutter opened")
        except LookupError:
            pass

        # Initialize camera
        camera = Camera.from_connection(conn)
        camera.binning = args.binning
        camera.auto_brightness = False
        camera.gain = args.gain

        # Report camera info
        print(f"Camera: {camera.name}")
        print(f"Frame size: {camera.frame_size_px}")
        print(f"Sensor size: {camera.sensor_size_px}")
        readout = camera.readout_time_s
        if readout:
            print(f"Readout time: {readout*1000:.2f} ms")
        print()

        # Run tests
        for exposure_s in exposure_times:
            print(f"Testing {exposure_s*1000:.3f} ms exposure:")
            result = test_exposure(camera, exposure_s, args.frames, save_dir=args.output_dir)
            results.append(result)
            print()

        # Clean up
        camera.dispose()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Requested':>12} {'Actual':>12} {'Mean Bright':>12} {'Std':>8} {'Capture':>10}")
    print(f"{'(ms)':>12} {'(ms)':>12} {'':>12} {'':>8} {'(ms)':>10}")
    print("-" * 60)

    for r in results:
        print(f"{r['requested_exposure_s']*1000:>12.3f} "
              f"{r['actual_exposure_s']*1000:>12.3f} "
              f"{r['mean_brightness']:>12.1f} "
              f"{r['std_brightness']:>8.2f} "
              f"{r['mean_capture_time_s']*1000:>10.1f}")

    print()

    # Analysis
    print("ANALYSIS:")

    # Check if actual exposure matches requested
    for r in results:
        if abs(r['actual_exposure_s'] - r['requested_exposure_s']) > 0.00001:
            print(f"  - {r['requested_exposure_s']*1000:.3f}ms was clamped to "
                  f"{r['actual_exposure_s']*1000:.3f}ms (camera minimum)")

    # Check brightness relationship (should scale roughly linearly with exposure)
    if len(results) >= 2:
        # Compare highest and lowest exposure
        high = max(results, key=lambda r: r['actual_exposure_s'])
        low = min(results, key=lambda r: r['actual_exposure_s'])

        if low['mean_brightness'] > 5:  # Not completely dark
            exposure_ratio = high['actual_exposure_s'] / low['actual_exposure_s']
            brightness_ratio = high['mean_brightness'] / low['mean_brightness']

            print(f"  - Exposure ratio (high/low): {exposure_ratio:.2f}x")
            print(f"  - Brightness ratio (high/low): {brightness_ratio:.2f}x")

            if brightness_ratio < exposure_ratio * 0.5:
                print("  - WARNING: Brightness doesn't scale linearly - "
                      "camera may have minimum exposure floor")
            elif 0.7 < brightness_ratio / exposure_ratio < 1.3:
                print("  - Brightness scales roughly linearly with exposure (good!)")
        else:
            print(f"  - Shortest exposure ({low['actual_exposure_s']*1000:.3f}ms) "
                  "produces very dark images - may be too fast")

    # Motion blur estimate
    print()
    print("MOTION BLUR ESTIMATES (at various scan speeds):")
    for speed_mm_s in [5, 10, 20, 40]:
        print(f"  At {speed_mm_s} mm/s scan speed:")
        for r in results:
            blur_um = speed_mm_s * 1000 * r['actual_exposure_s']
            print(f"    {r['actual_exposure_s']*1000:.3f}ms exposure -> {blur_um:.2f} µm blur")

    return 0


if __name__ == "__main__":
    sys.exit(main())
