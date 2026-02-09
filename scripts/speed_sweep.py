#!/usr/bin/env python3
"""Scan the same row at multiple speeds for comparison."""

import subprocess
import sys

AREA_RECT = "9987,38924,14439,14441"
OBJECTIVE = "20x"
SPEEDS = [5, 10, 15, 20, 25, 30, 35, 40]
OUTPUT_BASE = "scans/speed_sweep"


def main():
    for speed in SPEEDS:
        output = f"{OUTPUT_BASE}_{speed}mms"
        print(f"\n{'=' * 60}")
        print(f"Scanning at {speed} mm/s -> {output}")
        print(f"{'=' * 60}\n")

        cmd = [
            sys.executable,
            "scan_area_v1.py",
            "-o",
            output,
            "--objective-mag",
            OBJECTIVE,
            "--area-rect",
            AREA_RECT,
            "--speed-mm",
            str(speed),
            "--compress",
            "--clean",
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: scan at {speed} mm/s returned error (continuing...)")

    print(f"\n{'=' * 60}")
    print("All scans complete!")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
