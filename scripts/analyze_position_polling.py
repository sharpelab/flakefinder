#!/usr/bin/env python
"""Analyze multi-threaded position polling results.

Loads JSON output from test_position_polling.py and analyzes:
- Effective sample rates per thread vs combined
- Whether samples from different threads interleave
- Timing gaps and whether N threads improves over 1 thread at 60Hz
- Position resolution during motion (with --move data)
"""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description="Analyze position polling test results")
    parser.add_argument("input", help="Input JSON file from test_position_polling.py")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    config = data["config"]
    samples = data["samples"]
    num_threads = config["threads"]
    duration = config["duration_actual_s"]
    is_move_test = config.get("move", False)

    print(f"Position Polling Analysis")
    print(f"=" * 60)
    print(f"Threads: {num_threads}")
    print(f"Duration: {duration:.2f}s")
    print(f"Total samples: {len(samples)}")
    if is_move_test:
        start_x = config.get("start_x_um", 0)
        end_x = config.get("end_x_um", 0)
        speed = config.get("speed_mm_s", 0)
        print(f"Mode: Motion test")
        print(f"  X: {start_x:.0f} -> {end_x:.0f} µm ({abs(end_x - start_x):.0f} µm)")
        print(f"  Speed: {speed} mm/s")
    print()

    # Per-thread analysis
    print("Per-Thread Statistics:")
    print("-" * 60)

    thread_samples = {i: [] for i in range(num_threads)}
    for s in samples:
        thread_samples[s["thread_id"]].append(s)

    for tid in range(num_threads):
        ts = thread_samples[tid]
        if len(ts) < 2:
            print(f"  Thread {tid}: {len(ts)} samples (insufficient data)")
            continue

        rate = len(ts) / duration

        # Compute inter-sample gaps within this thread
        gaps = []
        for i in range(1, len(ts)):
            # Use midpoint of before/after as sample time
            t_prev = (ts[i-1]["t_before"] + ts[i-1]["t_after"]) / 2
            t_curr = (ts[i]["t_before"] + ts[i]["t_after"]) / 2
            gaps.append(t_curr - t_prev)

        avg_gap_ms = statistics.mean(gaps) * 1000
        std_gap_ms = statistics.stdev(gaps) * 1000 if len(gaps) > 1 else 0
        min_gap_ms = min(gaps) * 1000
        max_gap_ms = max(gaps) * 1000

        # SDK call duration
        call_durations = [(s["t_after"] - s["t_before"]) * 1000 for s in ts]
        avg_call_ms = statistics.mean(call_durations)

        print(f"  Thread {tid}: {len(ts)} samples, {rate:.1f} Hz")
        print(f"    Inter-sample gap: avg={avg_gap_ms:.2f}ms, std={std_gap_ms:.2f}ms, "
              f"min={min_gap_ms:.2f}ms, max={max_gap_ms:.2f}ms")
        print(f"    SDK call duration: avg={avg_call_ms:.2f}ms")

    # Combined analysis
    print()
    print("Combined Statistics (all threads merged):")
    print("-" * 60)

    combined_rate = len(samples) / duration
    print(f"  Combined sample rate: {combined_rate:.1f} Hz")
    print(f"  Expected if no serialization: {combined_rate / num_threads:.1f} Hz × {num_threads} threads")

    # Check interleaving - compute gaps in the combined stream
    combined_gaps = []
    for i in range(1, len(samples)):
        t_prev = (samples[i-1]["t_before"] + samples[i-1]["t_after"]) / 2
        t_curr = (samples[i]["t_before"] + samples[i]["t_after"]) / 2
        combined_gaps.append(t_curr - t_prev)

    if combined_gaps:
        avg_combined_gap_ms = statistics.mean(combined_gaps) * 1000
        std_combined_gap_ms = statistics.stdev(combined_gaps) * 1000 if len(combined_gaps) > 1 else 0
        min_combined_gap_ms = min(combined_gaps) * 1000
        max_combined_gap_ms = max(combined_gaps) * 1000
        median_combined_gap_ms = statistics.median(combined_gaps) * 1000

        print(f"  Combined inter-sample gap: avg={avg_combined_gap_ms:.2f}ms, "
              f"median={median_combined_gap_ms:.2f}ms")
        print(f"    std={std_combined_gap_ms:.2f}ms, min={min_combined_gap_ms:.2f}ms, "
              f"max={max_combined_gap_ms:.2f}ms")

    # Check for thread interleaving
    print()
    print("Interleaving Analysis:")
    print("-" * 60)

    # Count consecutive samples from same thread
    consecutive_runs = []
    current_run = 1
    for i in range(1, len(samples)):
        if samples[i]["thread_id"] == samples[i-1]["thread_id"]:
            current_run += 1
        else:
            consecutive_runs.append(current_run)
            current_run = 1
    consecutive_runs.append(current_run)

    avg_run_len = statistics.mean(consecutive_runs)
    max_run_len = max(consecutive_runs)

    print(f"  Consecutive same-thread runs: avg={avg_run_len:.1f}, max={max_run_len}")

    if avg_run_len > 5:
        print(f"  → Low interleaving: threads appear to be serialized by SDK")
    elif avg_run_len < 2:
        print(f"  → High interleaving: threads are running concurrently")
    else:
        print(f"  → Moderate interleaving")

    # Check if we're actually getting better resolution
    print()
    print("Effective Resolution Analysis:")
    print("-" * 60)

    single_thread_rate = len(thread_samples[0]) / duration if thread_samples[0] else 0

    print(f"  Single thread rate: {single_thread_rate:.1f} Hz")
    print(f"  Combined rate: {combined_rate:.1f} Hz")
    print(f"  Speedup factor: {combined_rate / single_thread_rate:.2f}x" if single_thread_rate > 0 else "  (no data)")

    theoretical_max = single_thread_rate * num_threads
    efficiency = (combined_rate / theoretical_max * 100) if theoretical_max > 0 else 0
    print(f"  Theoretical max (linear scaling): {theoretical_max:.1f} Hz")
    print(f"  Efficiency: {efficiency:.1f}%")

    # Motion-specific analysis
    if is_move_test:
        print()
        print("Position Resolution Analysis (Motion Test):")
        print("-" * 60)

        # Get all X positions
        x_positions = [s["x_um"] for s in samples]
        x_min, x_max = min(x_positions), max(x_positions)
        x_range = x_max - x_min

        print(f"  X range observed: {x_min:.1f} - {x_max:.1f} µm ({x_range:.1f} µm)")

        # Compute position deltas between consecutive samples (combined stream)
        x_deltas = []
        for i in range(1, len(samples)):
            dx = abs(samples[i]["x_um"] - samples[i-1]["x_um"])
            x_deltas.append(dx)

        if x_deltas:
            avg_dx = statistics.mean(x_deltas)
            median_dx = statistics.median(x_deltas)
            min_dx = min(x_deltas)
            max_dx = max(x_deltas)
            # Filter out zeros for meaningful stats
            nonzero_deltas = [d for d in x_deltas if d > 0.1]  # >0.1 µm

            print(f"  Position delta (combined): avg={avg_dx:.1f}µm, median={median_dx:.1f}µm")
            print(f"    min={min_dx:.2f}µm, max={max_dx:.1f}µm")

            if nonzero_deltas:
                avg_nonzero = statistics.mean(nonzero_deltas)
                print(f"    non-zero avg={avg_nonzero:.1f}µm ({len(nonzero_deltas)}/{len(x_deltas)} samples)")

        # Compare single-thread vs combined resolution
        print()
        print("  Single-thread position resolution:")
        for tid in range(min(2, num_threads)):  # Just show first 2 threads
            ts = thread_samples[tid]
            if len(ts) < 2:
                continue
            deltas = []
            for i in range(1, len(ts)):
                dx = abs(ts[i]["x_um"] - ts[i-1]["x_um"])
                deltas.append(dx)
            if deltas:
                avg_d = statistics.mean(deltas)
                print(f"    Thread {tid}: avg delta = {avg_d:.1f} µm")

        # Effective spatial resolution
        speed_um_s = config.get("speed_mm_s", 40) * 1000
        single_resolution = speed_um_s / single_thread_rate if single_thread_rate > 0 else 0
        combined_resolution = speed_um_s / combined_rate if combined_rate > 0 else 0

        print()
        print(f"  Theoretical spatial resolution at {speed_um_s/1000:.0f} mm/s:")
        print(f"    Single thread ({single_thread_rate:.0f} Hz): {single_resolution:.1f} µm between samples")
        print(f"    Combined ({combined_rate:.0f} Hz): {combined_resolution:.1f} µm between samples")
        print(f"    Improvement: {single_resolution / combined_resolution:.1f}x finer resolution")

    # Conclusion
    print()
    print("Conclusion:")
    print("-" * 60)

    if efficiency > 80:
        print(f"  ✓ Multi-threading DOES improve effective sample rate")
        print(f"    {num_threads} threads achieved {combined_rate:.1f} Hz vs {single_thread_rate:.1f} Hz single-thread")
    elif efficiency > 50:
        print(f"  ~ Partial improvement from multi-threading")
        print(f"    Some SDK serialization detected, but still gaining {combined_rate / single_thread_rate:.1f}x")
    else:
        print(f"  ✗ SDK appears to serialize access")
        print(f"    {num_threads} threads only achieved {combined_rate:.1f} Hz (~{efficiency:.0f}% of theoretical)")
        print(f"    Single thread at {single_thread_rate:.1f} Hz is sufficient")

    # Compare to 60Hz baseline
    print()
    if combined_rate > 60:
        print(f"  Combined rate ({combined_rate:.1f} Hz) exceeds 60Hz baseline ✓")
    else:
        print(f"  Combined rate ({combined_rate:.1f} Hz) does not exceed 60Hz")

    if is_move_test and efficiency > 80:
        print()
        print(f"  → Multi-threaded polling provides {combined_resolution:.1f}µm spatial resolution")
        print(f"    vs {single_resolution:.1f}µm with single thread ({single_resolution/combined_resolution:.1f}x improvement)")

    return 0


if __name__ == "__main__":
    exit(main())
