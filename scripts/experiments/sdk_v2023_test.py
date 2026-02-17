"""Test V2023 SDK features: EVENT_VALUESET_CHANGED and PROP_IMAGE_TIMESTAMP.

Two quick experiments to check whether V2023 SDK additions are usable on DM6M/K5C:

  1. Event audit — subscribe to raw events on X axis for 5 seconds, log all
     event types and rates. Looking for EVENT_VALUESET_CHANGED (0x105).
  2. Camera timestamp — check if PROP_IMAGE_TIMESTAMP is supported on K5C,
     acquire a frame and try to read hardware timestamp from image metadata.

Usage:
    uv run python scripts/sdk_v2023_test.py
    uv run python scripts/sdk_v2023_test.py --experiments 1    # events only
    uv run python scripts/sdk_v2023_test.py --experiments 2    # camera only
"""

import argparse
import contextlib
import threading
import time
from collections import defaultdict

from flakefinder.leica import Microscope
from flakefinder.leica.core import find_unit, get_interface, has_interface
from flakefinder.leica.enums import IID, UCAPI_PROP, UCAPI_TID

# New event type from V2023 SDK (ahwevents.h)
# After __RESERVED_EVENT0 (0x103) and __RESERVED_EVENT1 (0x104)
EVENT_VALUESET_CHANGED = 0x105

# Known event type names
EVENT_NAMES = {
    0x11: "INTERFACE_STATE_CHANGED",
    0x100: "VALUE_CHANGED",
    0x101: "PROPERTY_CHANGED",
    0x102: "VALUE_CHANGED_STRING",
    0x103: "PROPERTY_INFO_CHANGED / RESERVED_0",
    0x104: "PROPERTY_ENUM_CHANGED / RESERVED_1",
    EVENT_VALUESET_CHANGED: "VALUESET_CHANGED (V2023)",
}


def experiment_1_events(scope: Microscope, duration_s: float = 5.0) -> None:
    """Audit all events on X axis for duration_s seconds."""
    print(f"\n{'=' * 60}")
    print("Experiment 1: Event audit on X axis")
    print(f"{'=' * 60}")

    x_unit = scope.stage.x.unit

    # Check which event interfaces exist
    has_event_source = has_interface(x_unit, IID.IID_EVENT_SOURCE)
    has_advanced = has_interface(x_unit, IID.IID_ADVANCED_EVENT_SOURCE)
    print(f"  EventSource (0x106): {'YES' if has_event_source else 'NO'}")
    print(f"  AdvancedEventSource (0x10F): {'YES' if has_advanced else 'NO'}")

    if not has_event_source:
        print("  SKIP: No EventSource on X axis")
        return

    # Subscribe to raw events
    event_log: list[tuple[float, int, int]] = []  # (time, iid, value)
    lock = threading.Lock()

    from flakefinder.leica.events import AxisEvents

    events = AxisEvents(x_unit)

    def on_event(sender, iid: int, value: int) -> None:
        with lock:
            event_log.append((time.perf_counter(), iid, value))

    print(f"\n  Subscribing to events for {duration_s}s...")
    sub = events.subscribe_raw(on_event)

    start = time.perf_counter()
    time.sleep(duration_s)

    sub.unsubscribe()
    elapsed = time.perf_counter() - start

    # Analyze
    with lock:
        total = len(event_log)

    print(f"\n  Total events received: {total} in {elapsed:.1f}s ({total / elapsed:.1f} Hz)")

    if total == 0:
        print("  No events received (axis stationary — this is expected)")
        print("  Note: VALUE_CHANGED events only fire during motion.")
        print("  subscribe_raw uses ValueChangedEventHandler, which only")
        print("  delivers VALUE_CHANGED events. VALUESET_CHANGED would need")
        print("  a different subscription mechanism (addSink/EventSink).")
        return

    # Count by IID
    iid_counts: dict[int, int] = defaultdict(int)
    for _, iid, _ in event_log:
        iid_counts[iid] += 1

    print("\n  Events by IID:")
    for iid, count in sorted(iid_counts.items()):
        hz = count / elapsed
        name = EVENT_NAMES.get(iid, "unknown")
        print(f"    0x{iid:04X} ({name}): {count} events ({hz:.1f} Hz)")

    # Check for VALUESET_CHANGED specifically
    if EVENT_VALUESET_CHANGED in iid_counts:
        print(f"\n  ** VALUESET_CHANGED detected! {iid_counts[EVENT_VALUESET_CHANGED]} events **")
    else:
        print(f"\n  VALUESET_CHANGED (0x{EVENT_VALUESET_CHANGED:04X}): not seen")

    # Gap analysis on most frequent event type
    if total > 1:
        # Sort by time
        sorted_events = sorted(event_log, key=lambda x: x[0])
        gaps = [(sorted_events[i + 1][0] - sorted_events[i][0]) * 1000 for i in range(len(sorted_events) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        max_gap = max(gaps)
        print(f"\n  Inter-event gaps: avg={avg_gap:.1f}ms, min={min_gap:.1f}ms, max={max_gap:.1f}ms")


def experiment_1b_events_during_move(scope: Microscope, duration_s: float = 3.0) -> None:
    """Same as experiment 1, but trigger a move to generate events."""
    print(f"\n{'=' * 60}")
    print("Experiment 1b: Event audit during X axis move")
    print(f"{'=' * 60}")

    x_unit = scope.stage.x.unit

    if not has_interface(x_unit, IID.IID_EVENT_SOURCE):
        print("  SKIP: No EventSource on X axis")
        return

    from flakefinder.leica.events import AxisEvents

    events = AxisEvents(x_unit)
    event_log: list[tuple[float, int, int]] = []
    lock = threading.Lock()

    def on_event(sender, iid: int, value: int) -> None:
        with lock:
            event_log.append((time.perf_counter(), iid, value))

    # Get current position, plan a small move
    x_pos = scope.stage.x.position_um
    move_dist = 2000  # 2mm
    target = x_pos + move_dist

    print(f"  Current X: {x_pos:.0f} µm, moving to {target:.0f} µm")

    sub = events.subscribe_raw(on_event)
    start = time.perf_counter()

    # Start move
    handle = scope.stage.x.move_to_async(target)
    handle.wait(timeout=10.0)

    # Continue listening briefly after move completes
    time.sleep(0.5)
    elapsed = time.perf_counter() - start

    sub.unsubscribe()

    # Move back
    scope.stage.x.move_to_async(x_pos).wait(timeout=10.0)

    with lock:
        total = len(event_log)

    print(f"\n  Total events: {total} in {elapsed:.1f}s ({total / elapsed:.1f} Hz)")

    if total == 0:
        print("  No events received even during move!")
        return

    # Count by IID
    iid_counts: dict[int, int] = defaultdict(int)
    for _, iid, _ in event_log:
        iid_counts[iid] += 1

    print("\n  Events by IID:")
    for iid, count in sorted(iid_counts.items()):
        hz = count / elapsed
        iid_name = f"0x{iid:04X}"
        # Try to name known IIDs
        for e in IID:
            if int(e) == iid:
                iid_name = f"{e.name} (0x{iid:04X})"
                break
        print(f"    {iid_name}: {count} events ({hz:.1f} Hz)")

    # Check for VALUESET_CHANGED
    if EVENT_VALUESET_CHANGED in iid_counts:
        print(f"\n  ** VALUESET_CHANGED detected! {iid_counts[EVENT_VALUESET_CHANGED]} events **")
    else:
        print(f"\n  VALUESET_CHANGED (0x{EVENT_VALUESET_CHANGED:04X}): not seen")

    # Gap analysis
    if total > 1:
        sorted_events = sorted(event_log, key=lambda x: x[0])
        gaps = [(sorted_events[i + 1][0] - sorted_events[i][0]) * 1000 for i in range(len(sorted_events) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        max_gap = max(gaps)
        print(f"\n  Inter-event gaps: avg={avg_gap:.1f}ms, min={min_gap:.1f}ms, max={max_gap:.1f}ms")


def experiment_2_camera_timestamp(scope: Microscope) -> None:
    """Check PROP_IMAGE_TIMESTAMP support on K5C."""
    print(f"\n{'=' * 60}")
    print("Experiment 2: Camera PROP_IMAGE_TIMESTAMP")
    print(f"{'=' * 60}")

    conn = scope._conn
    assert conn is not None
    camera_unit = find_unit(conn.root, UCAPI_TID.UCAPI_CAMERA)
    if camera_unit is None:
        print("  SKIP: No camera unit found")
        return

    # Check if PROP_IMAGE_TIMESTAMP is a known property
    props_iface = get_interface(camera_unit, IID.IID_PROPERTIES)
    if props_iface is None:
        print("  SKIP: No Properties interface on camera")
        return

    prop_ts = props_iface.FindProperty(int(UCAPI_PROP.PROP_IMAGE_TIMESTAMP))
    print(f"  PROP_IMAGE_TIMESTAMP (16495) on camera properties: {'FOUND' if prop_ts else 'NOT FOUND'}")

    # Also check on camera sensor
    sensor_unit = find_unit(conn.root, UCAPI_TID.UCAPI_CAMERA_SENSOR)
    if sensor_unit:
        sensor_props = get_interface(sensor_unit, IID.IID_PROPERTIES)
        if sensor_props:
            sensor_ts = sensor_props.FindProperty(int(UCAPI_PROP.PROP_IMAGE_TIMESTAMP))
            print(f"  PROP_IMAGE_TIMESTAMP on sensor properties: {'FOUND' if sensor_ts else 'NOT FOUND'}")

    # Check new framerate properties while we're here
    for prop_name, prop_id in [
        ("PROP_ACQUISITION_FRAMERATE", UCAPI_PROP.PROP_ACQUISITION_FRAMERATE),
        ("PROP_ACQUISITION_FRAMERATE_ENABLED", UCAPI_PROP.PROP_ACQUISITION_FRAMERATE_ENABLED),
    ]:
        p = props_iface.FindProperty(int(prop_id))
        if p:
            val = p.GetValue()
            print(f"  {prop_name}: FOUND, value={val.GetValue()}")
        else:
            print(f"  {prop_name}: NOT FOUND")

    # Try acquiring a frame and inspecting metadata
    print("\n  Acquiring test frame to inspect metadata...")
    from flakefinder.leica.camera import Camera

    cam = Camera.from_connection(conn)
    cam.binning = 2
    cam.exposure_time = 0.01

    # Acquire with metadata inspection
    from LeicaMicrosystems.HardwareModel import Extensions

    with contextlib.suppress(Exception):
        Extensions.ExUCAPI.Register()

    context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory

    image_ref = [None]
    image_ready = threading.Event()

    def on_acquired(image):
        image_ref[0] = image
        image_ready.set()

    context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_acquired)

    # Try passing timestamp prop ID to Acquire
    # Build an IdList with the timestamp property
    try:
        from System import Array, Int32  # noqa: N811

        prop_ids = Array.CreateInstance(Int32, 1)
        prop_ids[0] = int(UCAPI_PROP.PROP_IMAGE_TIMESTAMP)
    except Exception:
        prop_ids = None

    t_before = time.perf_counter()
    cam._acquisition.Acquire(context, prop_ids)
    t_after = time.perf_counter()

    if image_ready.wait(timeout=5.0) and image_ref[0] is not None:
        image = image_ref[0]
        print(f"  Frame acquired in {(t_after - t_before) * 1000:.1f}ms")

        # Inspect image object for metadata methods
        print("\n  Inspecting image object for timestamp/metadata:")
        img_methods = [m for m in dir(image) if not m.startswith("_")]
        metadata_methods = [
            m
            for m in img_methods
            if any(kw in m.lower() for kw in ["meta", "property", "bag", "time", "stamp", "setting"])
        ]
        if metadata_methods:
            print(f"  Relevant methods: {metadata_methods}")
        else:
            print(f"  All methods: {img_methods}")

        # Try common .NET metadata accessors
        for method_name in [
            "MetaData",
            "GetMetaData",
            "PropertyBag",
            "GetPropertyBag",
            "Settings",
            "GetSettings",
            "Properties",
            "GetProperties",
        ]:
            accessor = getattr(image, method_name, None)
            if accessor is not None:
                try:
                    result = accessor() if callable(accessor) else accessor
                    print(f"  image.{method_name}: {result}")
                    # Try to enumerate if it's a collection
                    if result is not None:
                        for sub_method in ["NumSettings", "NumValues", "Count", "Length"]:
                            counter = getattr(result, sub_method, None)
                            if counter is not None:
                                try:
                                    n = counter() if callable(counter) else counter
                                    print(f"    {sub_method}: {n}")
                                    # Try to read entries
                                    getter = getattr(result, "GetSetting", None) or getattr(result, "Get", None)
                                    if getter and n > 0:
                                        for i in range(min(n, 5)):
                                            try:
                                                entry = getter(i)
                                                entry_id = getattr(entry, "GetId", lambda: "?")()
                                                entry_val = getattr(entry, "GetValue", lambda: "?")()
                                                print(f"    [{i}] id={entry_id}, value={entry_val}")
                                            except Exception as e:
                                                print(f"    [{i}] error: {e}")
                                except Exception as e:
                                    print(f"    {sub_method}: error {e}")
                except Exception as e:
                    print(f"  image.{method_name}: error {e}")

        image.Dispose()
    else:
        print("  Frame acquisition failed/timed out")

    with contextlib.suppress(Exception):
        context.Dispose()
    cam.dispose()


def main():
    parser = argparse.ArgumentParser(description="Test V2023 SDK features")
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,1b,2",
        help="Comma-separated experiment list (default: 1,1b,2)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Event listening duration in seconds (default: 5)",
    )
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]

    print("SDK V2023 Feature Test")
    print("=" * 60)

    with Microscope() as scope:
        print(f"Connected. Stage at {scope.stage.position_um}")

        if "1" in experiments:
            experiment_1_events(scope, args.duration)

        if "1b" in experiments:
            experiment_1b_events_during_move(scope, args.duration)

        if "2" in experiments:
            experiment_2_camera_timestamp(scope)

    print("\nDone.")


if __name__ == "__main__":
    main()
