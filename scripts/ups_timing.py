"""Benchmark UPS HID query timing."""

import time

import hid

APC_VID = 0x051D
APC_PID = 0x0002

dev = hid.device()
dev.open(APC_VID, APC_PID)
print(f"Device: {dev.get_product_string()}")

# Benchmark individual report reads
REPORTS = [0x22, 0x23, 0x26, 0x25, 0x31, 0x30, 0x32, 0x33, 0x16, 0x36, 0x35, 0x21]
LABELS = [
    "charge",
    "runtime",
    "batt_v",
    "batt_nom",
    "input_v",
    "input_nom",
    "low_xfer",
    "high_xfer",
    "status",
    "last_xfer",
    "sensitivity",
    "self_test",
]

print("\n--- Individual report timing (100 reads each) ---")
for report_id, label in zip(REPORTS, LABELS, strict=True):
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        dev.get_feature_report(report_id, 8)
        times.append(time.perf_counter() - t0)
    avg_us = sum(times) / len(times) * 1e6
    min_us = min(times) * 1e6
    max_us = max(times) * 1e6
    print(f"  0x{report_id:02X} ({label:12s}): avg {avg_us:7.1f} µs  min {min_us:6.1f}  max {max_us:7.1f}")

# Benchmark full status read (all 12 reports)
print("\n--- Full status read timing (100 iterations) ---")
times = []
for _ in range(100):
    t0 = time.perf_counter()
    for report_id in REPORTS:
        dev.get_feature_report(report_id, 8)
    times.append(time.perf_counter() - t0)

avg_ms = sum(times) / len(times) * 1e3
min_ms = min(times) * 1e3
max_ms = max(times) * 1e3
print(f"  12 reports: avg {avg_ms:.2f} ms  min {min_ms:.2f}  max {max_ms:.2f}")

dev.close()
