#!/usr/bin/env python
"""Investigate lesser-known SDK interfaces for rate limiting workarounds.

Checks if X, Y, Z axes support these interfaces:
- IID_ADVANCED_EVENT_SOURCE (0x10F) - Extended event capabilities
- IID_SNAPSHOTS (0x105) - Bulk/atomic reads?
- IID_BASIC_CONTROL_VALUE_TIMING (0x117) - Timing info

Run on microscope PC.
"""

import time
from flakefinder.leica import LeicaConnection, Stage, ZDrive
from flakefinder.leica.core import get_interface, has_interface
from flakefinder.leica.enums import IID, TID


def inspect_interface_object(obj, name: str, max_depth: int = 1):
    """Inspect a COM object to discover its methods and properties."""
    print(f"\n  === {name} ===")
    print(f"  Type: {type(obj)}")

    # Get all attributes
    attrs = []
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obj, attr)
            if callable(val):
                attrs.append((attr, "method", None))
            else:
                attrs.append((attr, "property", repr(val)[:60]))
        except Exception as e:
            attrs.append((attr, "error", str(e)[:40]))

    # Print methods
    methods = [a for a in attrs if a[1] == "method"]
    if methods:
        print(f"\n  Methods ({len(methods)}):")
        for name, _, _ in sorted(methods):
            print(f"    {name}()")

    # Print properties
    props = [a for a in attrs if a[1] == "property"]
    if props:
        print(f"\n  Properties ({len(props)}):")
        for name, _, val in sorted(props):
            print(f"    {name} = {val}")

    # Print errors
    errors = [a for a in attrs if a[1] == "error"]
    if errors:
        print(f"\n  Inaccessible ({len(errors)}):")
        for name, _, err in sorted(errors):
            print(f"    {name}: {err}")


def test_snapshots_interface(unit, unit_name: str):
    """Test IID_SNAPSHOTS interface if available."""
    print(f"\n{'='*60}")
    print(f"IID_SNAPSHOTS (0x105) on {unit_name}")
    print('='*60)

    snapshots = get_interface(unit, IID.IID_SNAPSHOTS)
    if snapshots is None:
        print("  NOT AVAILABLE")
        return

    print("  AVAILABLE!")
    inspect_interface_object(snapshots, "Snapshots Interface")

    # Try common snapshot patterns
    print("\n  Attempting operations...")

    # Try to get a snapshot
    for method_name in ['GetSnapshot', 'TakeSnapshot', 'CreateSnapshot', 'Snap']:
        if hasattr(snapshots, method_name):
            try:
                method = getattr(snapshots, method_name)
                result = method()
                print(f"    {method_name}() returned: {result}")
            except Exception as e:
                print(f"    {method_name}() failed: {e}")

    # Try to get count
    for method_name in ['GetNumSnapshots', 'Count', 'NumSnapshots']:
        if hasattr(snapshots, method_name):
            try:
                method = getattr(snapshots, method_name)
                if callable(method):
                    result = method()
                else:
                    result = method
                print(f"    {method_name}: {result}")
            except Exception as e:
                print(f"    {method_name} failed: {e}")


def test_advanced_event_source(unit, unit_name: str):
    """Test IID_ADVANCED_EVENT_SOURCE interface if available."""
    print(f"\n{'='*60}")
    print(f"IID_ADVANCED_EVENT_SOURCE (0x10F) on {unit_name}")
    print('='*60)

    adv_events = get_interface(unit, IID.IID_ADVANCED_EVENT_SOURCE)
    if adv_events is None:
        print("  NOT AVAILABLE")
        return

    print("  AVAILABLE!")
    inspect_interface_object(adv_events, "Advanced Event Source")

    # Also check regular event source for comparison
    print("\n  Checking regular IID_EVENT_SOURCE for comparison...")
    events = get_interface(unit, IID.IID_EVENT_SOURCE)
    if events:
        inspect_interface_object(events, "Regular Event Source")


def test_timing_interface(unit, unit_name: str):
    """Test IID_BASIC_CONTROL_VALUE_TIMING interface if available."""
    print(f"\n{'='*60}")
    print(f"IID_BASIC_CONTROL_VALUE_TIMING (0x117) on {unit_name}")
    print('='*60)

    timing = get_interface(unit, IID.IID_BASIC_CONTROL_VALUE_TIMING)
    if timing is None:
        print("  NOT AVAILABLE")
        return

    print("  AVAILABLE!")
    inspect_interface_object(timing, "Timing Interface")

    # Try to get timing info
    print("\n  Attempting operations...")
    for method_name in ['GetTiming', 'GetTimestamp', 'GetLastUpdateTime',
                        'GetUpdateRate', 'GetSampleRate', 'GetPeriod']:
        if hasattr(timing, method_name):
            try:
                method = getattr(timing, method_name)
                result = method()
                print(f"    {method_name}() returned: {result}")
            except Exception as e:
                print(f"    {method_name}() failed: {e}")


def test_other_interfaces(unit, unit_name: str):
    """Check for any other potentially useful interfaces."""
    print(f"\n{'='*60}")
    print(f"Other interfaces on {unit_name}")
    print('='*60)

    # List of potentially interesting interfaces from enums.py
    interesting = [
        (IID.IID_DELTA_CONTROL_VALUE, "Delta Control Value"),
        (IID.IID_LIMIT_CONTROL_VALUE, "Limit Control Value"),
        (IID.IID_DEFINED_CONTROL_VALUES, "Defined Control Values"),
        (IID.IID_BASIC_CONTROL_VALUE_SPECIAL, "Basic Control Value Special"),
        (IID.IID_LIMIT_SWITCHES, "Limit Switches"),
        (IID.IID_DELTA_CONTROL_VALUE_ASYNC, "Delta Control Value Async"),
        (IID.IID_BASIC_CONTROL_VALUE_ACCELERATION, "Acceleration"),
        (IID.IID_BASIC_CONTROL_VALUE_HYSTERESIS_CORRECTED, "Hysteresis Corrected"),
    ]

    found = []
    for iid, name in interesting:
        if has_interface(unit, iid):
            found.append((iid, name))

    if found:
        print(f"  Found {len(found)} additional interfaces:")
        for iid, name in found:
            print(f"    0x{int(iid):03X}: {name}")
    else:
        print("  No additional interfaces found")

    return found


def benchmark_interface_latency(unit, unit_name: str):
    """Measure latency of different read methods."""
    print(f"\n{'='*60}")
    print(f"Latency benchmark on {unit_name}")
    print('='*60)

    bcv = get_interface(unit, IID.IID_BASIC_CONTROL_VALUE)
    if not bcv:
        print("  No BasicControlValue interface!")
        return

    # Warm up
    for _ in range(10):
        bcv.GetControlValue()

    # Benchmark GetControlValue
    n_samples = 100
    times = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        bcv.GetControlValue()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    import statistics
    print(f"\n  GetControlValue() over {n_samples} calls:")
    print(f"    Mean: {statistics.mean(times):.2f} ms")
    print(f"    Std:  {statistics.stdev(times):.2f} ms")
    print(f"    Min:  {min(times):.2f} ms")
    print(f"    Max:  {max(times):.2f} ms")
    print(f"    Rate: {1000 / statistics.mean(times):.1f} Hz (single thread)")


def main():
    print("SDK Interface Investigation")
    print("=" * 60)

    with LeicaConnection() as conn:
        stage = Stage.from_connection(conn)
        z_drive = ZDrive.from_connection(conn)

        # Get raw units for interface probing
        units = [
            (stage.x._unit, "X Axis"),
            (stage.y._unit, "Y Axis"),
            (z_drive._unit, "Z Drive"),
        ]

        # Test each interface on each unit
        for unit, name in units:
            print(f"\n\n{'#'*60}")
            print(f"# {name}")
            print('#'*60)

            test_snapshots_interface(unit, name)
            test_advanced_event_source(unit, name)
            test_timing_interface(unit, name)
            other = test_other_interfaces(unit, name)

            # If we found interesting interfaces, inspect them
            for iid, iid_name in other:
                iface = get_interface(unit, iid)
                if iface:
                    inspect_interface_object(iface, f"{iid_name} (0x{int(iid):03X})")

        # Benchmark on X axis
        benchmark_interface_latency(stage.x._unit, "X Axis")

        print("\n" + "=" * 60)
        print("Investigation complete")


if __name__ == "__main__":
    main()
