#!/usr/bin/env python
"""Test if multiple threads polling position gives better temporal resolution.

Spawns N threads, each polling stage.x position as fast as possible.
Records thread_id, t_before, t_after, x_um for each sample.

With --move flag, moves stage from current X to max X during polling.
"""

import argparse
import ctypes
import json
import random
import threading
import time
from pathlib import Path

# Improve Windows timer resolution to ~1ms
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass  # Not on Windows or no permission


def main():
    parser = argparse.ArgumentParser(description="Test multi-threaded position polling")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-t", "--threads", type=int, default=2, help="Number of polling threads (default: 2)")
    parser.add_argument("-d", "--duration", type=float, default=5.0, help="Duration in seconds (default: 5, ignored with --move)")
    parser.add_argument("--move", action="store_true", help="Move stage from current X to max X during test")
    parser.add_argument("--speed-mm", type=float, default=40, help="Move speed in mm/s (default: 40)")
    args = parser.parse_args()

    from flakefinder.leica import LeicaConnection, Stage

    print(f"Multi-threaded Position Polling Test")
    print(f"  Threads: {args.threads}")
    if args.move:
        print(f"  Mode: Move to max X at {args.speed_mm} mm/s")
    else:
        print(f"  Mode: Static for {args.duration}s")
    print(f"  Output: {args.output}")
    print()

    with LeicaConnection() as conn:
        stage = Stage.from_connection(conn)
        x_bcv = stage.x.bcv
        x_converter = stage.x.converter

        start_x = stage.x.position_um
        start_y = stage.y.position_um
        target_x = stage.x.max_um - 5000  # Stop 5mm before end to avoid edge issues

        print(f"Stage X: {start_x:.1f} µm (target: {target_x:.1f} µm)")
        if args.move:
            distance = target_x - start_x
            est_time = distance / (args.speed_mm * 1000)
            print(f"Move distance: {distance:.0f} µm, estimated time: {est_time:.2f}s")
        print()

        # Set speed if moving
        if args.move and stage.x.supports_velocity:
            target_um_s = args.speed_mm * 1000
            max_vel = stage.x.max_velocity_um_s or target_um_s
            clamped = min(max_vel, target_um_s)
            stage.x.set_velocity_um_s(clamped)
            print(f"X velocity set to {stage.x.velocity_um_s / 1000:.1f} mm/s")

        # Shared state
        all_samples = []
        samples_lock = threading.Lock()
        stop_event = threading.Event()
        start_barrier = threading.Barrier(args.threads + 1)  # +1 for main thread

        def poll_thread(thread_id: int):
            """Poll position as fast as possible until stop_event is set."""
            local_samples = []

            # Wait for all threads to be ready
            start_barrier.wait()

            # Stagger thread start to desynchronize
            time.sleep(thread_id * 0.008)  # 8ms offset per thread

            while not stop_event.is_set():
                t_before = time.perf_counter()
                x_native = x_bcv.GetControlValue()
                t_after = time.perf_counter()
                x_um = x_converter.GetMetricsValue(x_native)

                local_samples.append({
                    "thread_id": thread_id,
                    "t_before": t_before,
                    "t_after": t_after,
                    "x_um": x_um,
                })

                # Random jitter to desynchronize threads
                time.sleep(random.uniform(0, 0.002))  # 0-2ms

            # Merge into global list
            with samples_lock:
                all_samples.extend(local_samples)

        # Start threads
        threads = []
        for i in range(args.threads):
            t = threading.Thread(target=poll_thread, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        if args.move:
            # Start async move BEFORE releasing polling threads
            # Otherwise polling can saturate the SDK and block the move
            print(f"Starting move to X={target_x:.0f} µm...")
            handle = stage.x.move_to_async(target_x)
            time.sleep(0.05)  # Let move command register

            # Now release polling threads
            print(f"Starting {args.threads} polling threads...")
            start_barrier.wait()
            t0 = time.perf_counter()

            # Busy loop checking handle completion
            # (works now that polling threads have 0.1ms yield)
            while not handle.is_complete:
                time.sleep(0.001)  # 1ms between checks

            handle.dispose()
        else:
            # Static test - just start threads and wait
            print(f"Starting {args.threads} polling threads...")
            start_barrier.wait()
            t0 = time.perf_counter()
            time.sleep(args.duration)

        # Stop threads
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

        t_end = time.perf_counter()
        actual_duration = t_end - t0
        end_x = stage.x.position_um

        print(f"Collected {len(all_samples)} total samples in {actual_duration:.2f}s")
        if args.move:
            print(f"X moved: {start_x:.1f} -> {end_x:.1f} µm")

        # Convert timestamps to relative (from t0)
        for sample in all_samples:
            sample["t_before"] -= t0
            sample["t_after"] -= t0

        # Sort by t_before for easier analysis
        all_samples.sort(key=lambda s: s["t_before"])

        # Build output
        output = {
            "config": {
                "threads": args.threads,
                "duration_requested_s": args.duration if not args.move else None,
                "duration_actual_s": actual_duration,
                "move": args.move,
                "speed_mm_s": args.speed_mm if args.move else None,
                "start_x_um": start_x,
                "end_x_um": end_x,
                "target_x_um": target_x,
            },
            "summary": {
                "total_samples": len(all_samples),
                "samples_per_thread": {},
            },
            "samples": all_samples,
        }

        # Count per thread
        for i in range(args.threads):
            count = sum(1 for s in all_samples if s["thread_id"] == i)
            output["summary"]["samples_per_thread"][str(i)] = count

        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Saved to {args.output}")

        # Quick summary
        print()
        print("Per-thread sample counts:")
        for tid, count in output["summary"]["samples_per_thread"].items():
            rate = count / actual_duration
            print(f"  Thread {tid}: {count} samples ({rate:.1f} Hz)")

        total_rate = len(all_samples) / actual_duration
        print(f"\nCombined rate: {total_rate:.1f} Hz")

        # Return to start if we moved
        if args.move:
            print(f"\nReturning to X={start_x:.0f} µm...")
            handle = stage.x.move_to_async(start_x)
            handle.wait()
            handle.dispose()
            print("Done.")

    # Restore Windows timer resolution
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    exit(main())
