"""Restore + characterize the IL illumination path (2026-08-13 incident).

HARDWARE SCRIPT — run only on explicit operator go. Requires a blank chip
under the 10x objective (rough focus fine; bg mode ratios are defocus-
insensitive). NO stage/Z motion; only IL turret, IL field diaphragm, DIC
turret, and IL aperture rotate/step, one element at a time, restoring the
initial state between sweeps and at exit.

For every condition, captures in TWO colour spaces:
  scan: WB 2.51/1.02/1.41 + CCM 5800K  -> comparable to historical
        bg target R/G ~= 1.00, B/G ~= 2.07 (pre-Aug-11 baseline)
  raw:  unity WB + identity CCM        -> per-element 3-channel transfer
        for the physics/illumination model

Self-contained against pre-sync scope master: illumination units are
accessed via find_unit + BasicControlValue directly (the ControlUnit
facade is not deployed yet).

Output: experiments/restore_illum_20260813/ (results.json + centre crops)

Usage (on the microscope PC):
    uv run python scripts/experiments/restore_illum_state.py
    uv run python scripts/experiments/restore_illum_state.py --leave-best
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from flakefinder.leica import Microscope
from flakefinder.leica.core import find_unit, get_interface_required
from flakefinder.leica.enums import IID, TID
from flakefinder.types import ColourMatrix, GainRGB

# Historical (pre-Aug-11) background target in scan space at 10x.
TARGET_RG = 1.00
TARGET_BG = 2.07

SCAN_WB = GainRGB(red=1.41, green=1.02, blue=2.51)
UNITY_WB = GainRGB(red=1.0, green=1.0, blue=1.0)
EXPOSURE_S = 0.00025
GAIN = 4.0

SETTLE_S = 1.0
FRAMES_PER_CONDITION = 3

# Element sweeps: (label, TID, positions to visit). Initial values are
# read live and restored; sweeps run one element at a time.
SWEEPS = [
    ("il_turret", TID.MICROSCOPE_IL_TURRET, [1, 2, 3, 4]),
    ("il_field_diaphragm", TID.MICROSCOPE_IL_FIELD_DIAPHRAGM, [1, 2, 3, 4, 5, 6]),
    ("dic_turret", TID.MICROSCOPE_DIC_TURRET, [1, 2, 3, 4]),
    ("il_aperture", TID.MICROSCOPE_IL_APERTURE_DIAPHRAGM, [11, 8, 5, 3, 1]),
]


class Bcv:
    """Minimal BasicControlValue wrapper for a unit found by TID."""

    def __init__(self, scope: Microscope, tid: TID):
        unit = find_unit(scope.conn.root, tid)
        if unit is None:
            raise LookupError(f"Unit not found: {tid.name}")
        self.name = unit.GetName()
        self._bcv = get_interface_required(unit, IID.IID_BASIC_CONTROL_VALUE)

    @property
    def value(self) -> int:
        return self._bcv.GetControlValue()

    @value.setter
    def value(self, v: int) -> None:
        self._bcv.SetControlValue(int(v))


def bg_mode_rgb(img: np.ndarray) -> list[float]:
    """Background mode per channel on the centre-third crop.

    Histogram mode, refined as the mean of pixels within +/-5 counts of
    the mode (same approach as segmentation bg extraction).
    """
    h, w = img.shape[:2]
    crop = img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    out = []
    for ch in range(3):
        vals = crop[:, :, ch].ravel()
        mode = int(np.bincount(vals, minlength=256).argmax())
        sel = vals[(vals >= mode - 5) & (vals <= mode + 5)]
        out.append(float(sel.mean()) if len(sel) else float(mode))
    return out


def centre_crop(img: np.ndarray, size: int = 768) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = max(0, h // 2 - size // 2)
    x0 = max(0, w // 2 - size // 2)
    return img[y0 : y0 + size, x0 : x0 + size]


def _set_element(scope: Microscope, unit: Bcv, pos: int) -> bool:
    """Set an element position, closing the IL shutter around the move.

    The stand refuses some element moves with the shutter open (cube-change
    interlock). Returns False if the stand still refuses the move.
    """
    if unit.value == pos:
        return True
    scope.shutter.close()
    time.sleep(0.2)
    try:
        unit.value = pos
        return True
    except Exception:
        return False
    finally:
        scope.shutter.open()
        time.sleep(0.2)


def set_space(scope: Microscope, space: str) -> None:
    cam = scope.camera
    if space == "scan":
        cam.gain_rgb = SCAN_WB
        cam.colour_matrix = ColourMatrix.CCM_5800K
    else:
        cam.gain_rgb = UNITY_WB
        cam.colour_matrix = ColourMatrix.IDENTITY
    time.sleep(0.2)


def measure(scope: Microscope, out_dir: Path, cond: str) -> dict:
    """Capture both spaces for the current hardware state."""
    rec: dict = {"condition": cond}
    for space in ("scan", "raw"):
        set_space(scope, space)
        modes = []
        first = None
        for _ in range(FRAMES_PER_CONDITION):
            img = scope.camera.capture()
            if first is None:
                first = img
            modes.append(bg_mode_rgb(img))
        med = np.median(np.array(modes), axis=0)
        rec[space] = {
            "bg_rgb": [round(float(v), 2) for v in med],
            "rg": round(float(med[0] / med[1]), 4),
            "bg_ratio": round(float(med[2] / med[1]), 4),
        }
        if space == "scan" and first is not None:
            cv2.imwrite(str(out_dir / f"{cond}_scan.png"), centre_crop(first)[:, :, ::-1])
    s = rec["scan"]
    rec["target_dist"] = round(abs(s["rg"] - TARGET_RG) + abs(s["bg_ratio"] - TARGET_BG), 4)
    print(
        f"  {cond:<24} scan R/G={s['rg']:.3f} B/G={s['bg_ratio']:.3f} "
        f"dist={rec['target_dist']:.3f}  raw RGB={rec['raw']['bg_rgb']}"
    )
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore + characterize IL illumination path")
    parser.add_argument(
        "--leave-best",
        action="store_true",
        help="Leave each swept element at its best (min target-distance) position instead of restoring",
    )
    parser.add_argument("--out", type=str, default="experiments/restore_illum_20260813")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"target": {"rg": TARGET_RG, "bg_ratio": TARGET_BG}, "sweeps": {}}

    with Microscope() as scope:
        mag = scope.nosepiece.magnification
        if mag != 10:
            print(f"ABORT: objective is {mag}x, expected 10x. No changes made.")
            return 1

        units = {label: Bcv(scope, tid) for label, tid, _ in SWEEPS}
        initial = {label: u.value for label, u in units.items()}
        initial_lamp = scope.lamp.intensity_pct
        initial_shutter_open = scope.shutter.is_open
        results["initial_state"] = dict(initial)
        shutter_str = "open" if initial_shutter_open else "closed"
        print(f"Initial state: {initial} (lamp {initial_lamp:.0f}%, shutter {shutter_str})")

        # Camera baseline settings (binning/gamma pinned by camera init)
        cam = scope.camera
        cam.exposure_time = EXPOSURE_S
        cam.gain = GAIN
        scope.light_on()
        time.sleep(0.5)

        try:
            print("\nBaseline (current state):")
            results["baseline"] = measure(scope, out_dir, "baseline")

            for label, _tid, positions in SWEEPS:
                unit = units[label]
                print(f"\nSweep {label} ({unit.name}), initial={initial[label]}:")
                recs = []
                for pos in positions:
                    if not _set_element(scope, unit, pos):
                        print(f"  {label}_{pos:<18} REFUSED by stand (interlock or empty position)")
                        recs.append({"condition": f"{label}_{pos}", "set": pos, "refused": True, "target_dist": 99})
                        continue
                    time.sleep(SETTLE_S)
                    actual = unit.value
                    rec = measure(scope, out_dir, f"{label}_{pos}")
                    rec["set"] = pos
                    rec["readback"] = actual
                    if actual != pos:
                        print(f"  WARNING: set {pos} but readback {actual}")
                    recs.append(rec)
                results["sweeps"][label] = recs
                # restore this element before sweeping the next
                _set_element(scope, unit, initial[label])
                time.sleep(SETTLE_S)

            # Optionally leave each element at its best position
            if args.leave_best:
                print("\n--leave-best: applying per-element best positions:")
                for label, _tid, _pos in SWEEPS:
                    best = min(results["sweeps"][label], key=lambda r: r["target_dist"])
                    _set_element(scope, units[label], best["set"])
                    time.sleep(SETTLE_S)
                    print(f"  {label} -> {best['set']} (dist {best['target_dist']:.3f})")
                time.sleep(SETTLE_S)
                results["final"] = measure(scope, out_dir, "final_leave_best")
        finally:
            if not args.leave_best:
                for label, _tid, _pos in SWEEPS:
                    _set_element(scope, units[label], initial[label])
                time.sleep(SETTLE_S)
                restored = {label: u.value for label, u in units.items()}
                print(f"\nRestored initial state: {restored}")
                results["restored_state"] = restored
            # restore light state
            scope.lamp.intensity_pct = initial_lamp
            if not initial_shutter_open:
                scope.shutter.close()

        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'condition':<24} {'R/G':>7} {'B/G':>7} {'dist':>7}")
    print("-" * 50)
    rows = [results["baseline"]] + [r for recs in results["sweeps"].values() for r in recs]
    for r in sorted(rows, key=lambda r: r["target_dist"]):
        if r.get("refused"):
            print(f"{r['condition']:<24} {'REFUSED':>23}")
            continue
        s = r["scan"]
        print(f"{r['condition']:<24} {s['rg']:>7.3f} {s['bg_ratio']:>7.3f} {r['target_dist']:>7.3f}")
    print(f"\nTarget: R/G={TARGET_RG:.2f} B/G={TARGET_BG:.2f}")
    print(f"Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
