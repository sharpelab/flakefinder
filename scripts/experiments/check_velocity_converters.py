"""Check what converters are available for directed velocity interface."""

from flakefinder.leica import LeicaConnection, Stage, ZDrive
from flakefinder.leica.core import get_interface
from flakefinder.leica.enums import IID, EMetricsId

with LeicaConnection() as conn:
    stage = Stage.from_connection(conn)
    z_drive = ZDrive.from_connection(conn)

    for name, axis in [("X", stage.x), ("Z", z_drive)]:
        dv = get_interface(axis._unit, IID.IID_DIRECTED_CONTROL_VALUE_ASYNC_VELOCITY)
        print(f"\n{name} axis directed velocity interface:")
        if dv is None:
            print("  Not available")
            continue

        print(f"  Interface: {dv}")
        print(f"  Methods: {[m for m in dir(dv) if not m.startswith('_')]}")

        try:
            converters = dv.GetMetricsConverters()
            print(f"  Converters object: {converters}")
            print(f"  Converters dir: {[m for m in dir(converters) if not m.startswith('_')]}")

            # Try to iterate
            try:
                for i, conv in enumerate(converters):
                    print(f"    [{i}] {conv}, MetricsId={getattr(conv, 'MetricsId', 'N/A')}")
            except Exception as e:
                print(f"  Can't iterate: {e}")

        except Exception as e:
            print(f"  Error getting converters: {e}")

        # Try specific metrics IDs
        print("  Looking for µm/s converter:")
        try:
            conv = converters.FindMetricsConverter(int(EMetricsId.METRICS_MICRONS_PER_SECOND))
            print(f"    METRICS_MICRONS_PER_SECOND: {conv}")
        except Exception as e:
            print(f"    METRICS_MICRONS_PER_SECOND: Error - {e}")

        print("  Looking for µm converter:")
        try:
            conv = converters.FindMetricsConverter(int(EMetricsId.METRICS_MICRONS))
            print(f"    METRICS_MICRONS: {conv}")
        except Exception as e:
            print(f"    METRICS_MICRONS: Error - {e}")
