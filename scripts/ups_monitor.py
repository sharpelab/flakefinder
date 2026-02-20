"""APC Back-UPS HID monitor prototype.

Reads UPS status via USB HID feature reports. Designed for APC Back-UPS RS 1500G
but report IDs likely work across the Back-UPS USB family.

Usage:
    python ups_monitor.py              # Single status read
    python ups_monitor.py --poll 5     # Poll every 5 seconds
    python ups_monitor.py --json       # JSON output (for integration)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import hid

APC_VID = 0x051D
APC_PID = 0x0002


@dataclass
class UPSStatus:
    """Decoded UPS status."""

    timestamp: str
    # Power state
    ac_present: bool
    on_battery: bool
    # Battery
    charge_pct: int
    runtime_sec: int
    battery_voltage: float
    battery_nominal_voltage: float
    # Input
    input_voltage: int
    input_nominal_voltage: int
    low_transfer_voltage: int
    high_transfer_voltage: int
    # Status
    status_raw: int
    last_transfer_cause: int
    sensitivity: int
    self_test_result: int

    @property
    def runtime_min(self) -> float:
        return self.runtime_sec / 60.0

    @property
    def status_str(self) -> str:
        if not self.ac_present:
            return "ON BATTERY"
        if self.charge_pct >= 100:
            return "ONLINE"
        return "ONLINE (charging)"

    def summary(self) -> str:
        lines = [
            f"Status:    {self.status_str}",
            f"Charge:    {self.charge_pct}%",
            f"Runtime:   {self.runtime_min:.1f} min ({self.runtime_sec} sec)",
            f"Batt V:    {self.battery_voltage:.2f}V (nominal {self.battery_nominal_voltage:.2f}V)",
            f"Input V:   {self.input_voltage}V (nominal {self.input_nominal_voltage}V)",
            f"Transfer:  {self.low_transfer_voltage}V low / {self.high_transfer_voltage}V high",
            f"Raw status: 0x{self.status_raw:02X}  AC={'yes' if self.ac_present else 'NO'}",
        ]
        return "\n".join(lines)


# HID feature report definitions: (report_id, num_bytes, decode_func)
# Byte counts exclude the report ID byte (hidapi includes it in the response)


def _u8(data: list[int]) -> int:
    """Single byte value."""
    return data[1] if len(data) > 1 else 0


def _u16le(data: list[int]) -> int:
    """Little-endian uint16."""
    if len(data) < 3:
        return 0
    return data[1] | (data[2] << 8)


TRANSFER_CAUSES = {
    0: "No transfer",
    1: "High line voltage",
    2: "Brownout",
    3: "Blackout",
    4: "Small sag",
    5: "Large sag",
    6: "Small spike",
    7: "Large spike",
    8: "Self test",
    9: "Rate of voltage change",
}

SENSITIVITY = {0: "Low", 1: "High", 2: "Medium"}

SELF_TEST = {1: "Passed", 2: "Warning", 4: "Failed", 6: "No test"}


def read_status(dev: hid.device) -> UPSStatus:
    """Read all relevant feature reports and return decoded status."""

    def feat(report_id: int, length: int = 8) -> list[int]:
        return dev.get_feature_report(report_id, length)

    # Battery
    charge = _u8(feat(0x22))
    runtime = _u16le(feat(0x23))
    batt_v = _u16le(feat(0x26)) / 100.0
    batt_nom_v = _u16le(feat(0x25)) / 100.0

    # Input
    input_v = _u16le(feat(0x31))
    input_nom_v = _u8(feat(0x30))
    low_xfer = _u16le(feat(0x32))
    high_xfer = _u16le(feat(0x33))

    # Status
    status_data = feat(0x16)
    status_raw = status_data[1] if len(status_data) > 1 else 0
    ac_present = bool(status_raw & 0x01)

    last_xfer = _u8(feat(0x36))
    sensitivity = _u8(feat(0x35))
    self_test = _u8(feat(0x21))

    return UPSStatus(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ac_present=ac_present,
        on_battery=not ac_present,
        charge_pct=charge,
        runtime_sec=runtime,
        battery_voltage=batt_v,
        battery_nominal_voltage=batt_nom_v,
        input_voltage=input_v,
        input_nominal_voltage=input_nom_v,
        low_transfer_voltage=low_xfer,
        high_transfer_voltage=high_xfer,
        status_raw=status_raw,
        last_transfer_cause=last_xfer,
        sensitivity=sensitivity,
        self_test_result=self_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="APC UPS HID Monitor")
    parser.add_argument("--poll", type=float, metavar="SEC", help="Poll interval in seconds")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    dev = hid.device()
    try:
        dev.open(APC_VID, APC_PID)
    except OSError as e:
        print(f"Cannot open UPS (VID={APC_VID:#06x} PID={APC_PID:#06x}): {e}")
        print("Is the UPS connected via USB?")
        sys.exit(1)

    product = dev.get_product_string()
    serial = dev.get_serial_number_string()
    print(f"Connected: {product} (S/N: {serial})")
    print()

    try:
        while True:
            status = read_status(dev)

            if args.json:
                d = asdict(status)
                d["runtime_min"] = status.runtime_min
                d["status_str"] = status.status_str
                print(json.dumps(d))
            else:
                print(f"--- {status.timestamp} ---")
                print(status.summary())
                print()

            if not args.poll:
                break
            time.sleep(args.poll)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        dev.close()


if __name__ == "__main__":
    main()
