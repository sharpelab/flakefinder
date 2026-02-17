#!/usr/bin/env python
"""Compare Z position reads: regular vs hysteresis-corrected interface.

Tests:
1. Static: Compare values and timing when Z is stationary
2. Motion: Compare during Z movement (hysteresis should matter more here)
3. Direction change: Move up then down, check for hysteresis offset

Run on microscope PC.
"""

import statistics
import time

from flakefinder.leica import LeicaConnection, ZDrive
from flakefinder.leica.core import get_interface
from flakefinder.leica.enums import IID


def main():
    print("Z Hysteresis-Corrected Interface Test")
    print("=" * 60)

    with LeicaConnection() as conn:
        z_drive = ZDrive.from_connection(conn)

        # Get both interfaces
        bcv_regular = z_drive._bcv  # Regular BasicControlValue
        bcv_hysteresis = get_interface(z_drive._unit, IID.IID_BASIC_CONTROL_VALUE_HYSTERESIS_CORRECTED)

        if bcv_hysteresis is None:
            print("ERROR: Hysteresis-corrected interface not available on Z")
            return 1

        # Get converter for both (should be same)
        converter = z_drive._converter

        # Check if hysteresis interface has its own converter
        try:
            hyst_converters = bcv_hysteresis.GetMetricsConverters()
            hyst_converter = hyst_converters.FindMetricsConverter(1)  # METRICS_MICRONS
            print(f"Hysteresis interface has its own converter: {hyst_converter is not None}")
        except Exception as e:
            print(f"Hysteresis converter check failed: {e}")
            hyst_converter = converter  # Fall back to regular

        initial_z = z_drive.position_um
        print(f"Initial Z: {initial_z:.1f} um")
        print()

        # =================================================================
        # Test 1: Static comparison
        # =================================================================
        print("Test 1: Static Position Reads")
        print("-" * 60)

        n_samples = 100

        # Warm up
        for _ in range(10):
            bcv_regular.GetControlValue()
            bcv_hysteresis.GetControlValue()

        # Collect samples - alternating to minimize timing bias
        regular_samples = []
        hysteresis_samples = []
        regular_times = []
        hysteresis_times = []

        for _ in range(n_samples):
            # Regular read
            t0 = time.perf_counter()
            val_r = bcv_regular.GetControlValue()
            t1 = time.perf_counter()
            regular_samples.append(converter.GetMetricsValue(val_r))
            regular_times.append((t1 - t0) * 1000)

            # Hysteresis read
            t0 = time.perf_counter()
            val_h = bcv_hysteresis.GetControlValue()
            t1 = time.perf_counter()
            hysteresis_samples.append(converter.GetMetricsValue(val_h))
            hysteresis_times.append((t1 - t0) * 1000)

        # Analyze
        print(f"\n  Regular interface ({n_samples} reads):")
        print(f"    Position: {statistics.mean(regular_samples):.2f} ± {statistics.stdev(regular_samples):.3f} um")
        print(f"    Latency:  {statistics.mean(regular_times):.2f} ± {statistics.stdev(regular_times):.2f} ms")
        print(f"    Range:    {min(regular_samples):.2f} - {max(regular_samples):.2f} um")

        print(f"\n  Hysteresis-corrected interface ({n_samples} reads):")
        print(
            f"    Position: {statistics.mean(hysteresis_samples):.2f} ± {statistics.stdev(hysteresis_samples):.3f} um"
        )
        print(f"    Latency:  {statistics.mean(hysteresis_times):.2f} ± {statistics.stdev(hysteresis_times):.2f} ms")
        print(f"    Range:    {min(hysteresis_samples):.2f} - {max(hysteresis_samples):.2f} um")

        # Compute differences
        diffs = [h - r for r, h in zip(regular_samples, hysteresis_samples, strict=False)]
        print("\n  Difference (hysteresis - regular):")
        print(f"    Mean:  {statistics.mean(diffs):+.3f} um")
        print(f"    Std:   {statistics.stdev(diffs):.3f} um")
        print(f"    Range: {min(diffs):+.3f} to {max(diffs):+.3f} um")

        # =================================================================
        # Test 2: During motion
        # =================================================================
        print("\n\nTest 2: Position Reads During Motion")
        print("-" * 60)

        # Move parameters
        z_travel = 500  # um
        z_start = initial_z
        z_end = initial_z + z_travel

        print(f"  Moving Z: {z_start:.0f} -> {z_end:.0f} um ({z_travel} um)")

        # Collect samples during motion
        motion_regular = []
        motion_hysteresis = []
        motion_times = []

        # Start async move
        handle = z_drive.move_to_async(z_end)
        t_start = time.perf_counter()

        while not handle.is_complete:
            t_now = time.perf_counter()

            # Read both
            val_r = bcv_regular.GetControlValue()
            val_h = bcv_hysteresis.GetControlValue()

            motion_regular.append(converter.GetMetricsValue(val_r))
            motion_hysteresis.append(converter.GetMetricsValue(val_h))
            motion_times.append(t_now - t_start)

        handle.dispose()

        print(f"  Collected {len(motion_regular)} sample pairs during motion")

        if len(motion_regular) > 10:
            # Compute instantaneous differences
            motion_diffs = [h - r for r, h in zip(motion_regular, motion_hysteresis, strict=False)]

            print("\n  Difference during upward motion:")
            print(f"    Mean:  {statistics.mean(motion_diffs):+.3f} um")
            print(f"    Std:   {statistics.stdev(motion_diffs):.3f} um")
            print(f"    Range: {min(motion_diffs):+.3f} to {max(motion_diffs):+.3f} um")

        # =================================================================
        # Test 3: Direction change (hysteresis test)
        # =================================================================
        print("\n\nTest 3: Direction Change (Hysteresis Detection)")
        print("-" * 60)

        # Move back down
        print(f"  Moving Z back down: {z_end:.0f} -> {z_start:.0f} um")

        motion_down_regular = []
        motion_down_hysteresis = []

        handle = z_drive.move_to_async(z_start)

        while not handle.is_complete:
            val_r = bcv_regular.GetControlValue()
            val_h = bcv_hysteresis.GetControlValue()

            motion_down_regular.append(converter.GetMetricsValue(val_r))
            motion_down_hysteresis.append(converter.GetMetricsValue(val_h))

        handle.dispose()

        print(f"  Collected {len(motion_down_regular)} sample pairs during downward motion")

        if len(motion_down_regular) > 10:
            motion_down_diffs = [h - r for r, h in zip(motion_down_regular, motion_down_hysteresis, strict=False)]

            print("\n  Difference during downward motion:")
            print(f"    Mean:  {statistics.mean(motion_down_diffs):+.3f} um")
            print(f"    Std:   {statistics.stdev(motion_down_diffs):.3f} um")
            print(f"    Range: {min(motion_down_diffs):+.3f} to {max(motion_down_diffs):+.3f} um")

        # Compare up vs down
        if len(motion_diffs) > 10 and len(motion_down_diffs) > 10:
            up_mean = statistics.mean(motion_diffs)
            down_mean = statistics.mean(motion_down_diffs)
            direction_shift = down_mean - up_mean

            print("\n  Direction-dependent offset:")
            print(f"    Upward mean diff:   {up_mean:+.3f} um")
            print(f"    Downward mean diff: {down_mean:+.3f} um")
            print(f"    Shift on reversal:  {direction_shift:+.3f} um")

            if abs(direction_shift) > 0.5:
                print("    -> Hysteresis correction IS active (shift > 0.5 um)")
            else:
                print("    -> No significant hysteresis correction detected")

        # =================================================================
        # Test 4: Rapid alternating reads (SDK contention test)
        # =================================================================
        print("\n\nTest 4: Rapid Alternating Reads")
        print("-" * 60)

        n_pairs = 200

        t0 = time.perf_counter()
        for _ in range(n_pairs):
            bcv_regular.GetControlValue()
            bcv_hysteresis.GetControlValue()
        t1 = time.perf_counter()

        total_ms = (t1 - t0) * 1000
        per_pair_ms = total_ms / n_pairs
        effective_hz = n_pairs / (t1 - t0)

        print(f"  {n_pairs} pairs (regular + hysteresis) in {total_ms:.0f} ms")
        print(f"  Per pair: {per_pair_ms:.2f} ms")
        print(f"  Effective rate: {effective_hz:.1f} pairs/s")
        print(f"  -> Each interface call: ~{per_pair_ms / 2:.2f} ms (no extra overhead)")

        # =================================================================
        # Summary
        # =================================================================
        print("\n\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        static_jitter_regular = statistics.stdev(regular_samples)
        static_jitter_hysteresis = statistics.stdev(hysteresis_samples)

        print("\n  Static jitter:")
        print(f"    Regular:    {static_jitter_regular:.3f} um")
        print(f"    Hysteresis: {static_jitter_hysteresis:.3f} um")

        if static_jitter_hysteresis < static_jitter_regular * 0.9:
            print(
                f"    -> Hysteresis interface has {(1 - static_jitter_hysteresis / static_jitter_regular) * 100:.0f}% less jitter"  # noqa: E501
            )
        elif static_jitter_hysteresis > static_jitter_regular * 1.1:
            print("    -> Hysteresis interface has MORE jitter (unexpected)")
        else:
            print("    -> Similar jitter levels")

        print("\n  Latency:")
        print(f"    Regular:    {statistics.mean(regular_times):.2f} ms")
        print(f"    Hysteresis: {statistics.mean(hysteresis_times):.2f} ms")

        latency_diff = statistics.mean(hysteresis_times) - statistics.mean(regular_times)
        if abs(latency_diff) < 0.5:
            print("    -> Same latency (no extra overhead)")
        else:
            print(f"    -> Hysteresis is {latency_diff:+.1f} ms different")

        # Restore position
        final_z = z_drive.position_um
        if abs(final_z - initial_z) > 1:
            print(f"\n  Restoring Z to {initial_z:.0f} um...")
            z_drive.move_to(initial_z)

        print("\nDone.")
        return 0


if __name__ == "__main__":
    exit(main())
