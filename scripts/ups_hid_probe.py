"""Probe APC UPS via HID to discover available data."""

import hid

APC_VID = 0x051D
APC_PID = 0x0002

# List all matching devices
print("=== HID devices matching APC ===")
for dev in hid.enumerate(APC_VID, APC_PID):
    for k, v in dev.items():
        print(f"  {k}: {v}")
    print()

# Open device
print("=== Opening UPS ===")
h = hid.device()
h.open(APC_VID, APC_PID)
print(f"Manufacturer: {h.get_manufacturer_string()}")
print(f"Product: {h.get_product_string()}")
print(f"Serial: {h.get_serial_number_string()}")

# Probe feature reports (report IDs 0x00 through 0x3F)
print("\n=== Feature Reports ===")
for report_id in range(64):
    try:
        data = h.get_feature_report(report_id, 64)
        if data:
            hex_str = " ".join(f"{b:02X}" for b in data)
            print(f"Report 0x{report_id:02X} ({len(data)} bytes): {hex_str}")
    except Exception:
        pass

# Try reading input reports (non-blocking)
print("\n=== Input Reports (non-blocking) ===")
h.set_nonblocking(1)
for _ in range(10):
    data = h.read(64)
    if data:
        hex_str = " ".join(f"{b:02X}" for b in data)
        print(f"Input ({len(data)} bytes): {hex_str}")
    else:
        break

h.close()
print("\nDone.")
