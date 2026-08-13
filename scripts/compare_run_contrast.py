"""Pairwise between-scan contrast comparison for two runs of the same chips.

Matches detections across two scan runs by absolute stage coordinates
(mutual nearest neighbour within --match-radius, sizes within --size-ratio)
and compares their stored seg contrast_rgb pairwise.  Isolates between-scan
drift (lamp / focus / flatfield) from extraction and AFM questions: a
uniform per-channel offset across many matched flakes is instrument drift,
not model or flake physics.

Usage:
    uv run python scripts/compare_run_contrast.py RUN_A RUN_B

Deltas are reported as B - A.  Run dirs must contain chip_*/seg/*.json and
chip_*/scan_*/scan_meta.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from flakefinder.data_utils import add_stage_coords


class Det(NamedTuple):
    chip: int
    stage_x: float
    stage_y: float
    size_um2: float
    contrast_rgb: tuple[float, float, float]


def load_run(run_dir: Path) -> list[Det]:
    dets: list[Det] = []
    for chip_dir in sorted(run_dir.glob("chip_*")):
        chip = int(chip_dir.name.split("_")[1])
        meta_paths = list(chip_dir.glob("scan_*/scan_meta.json"))
        if not meta_paths:
            continue
        scan_meta = json.loads(meta_paths[0].read_text())
        chip_dets = []
        for frame_json in sorted((chip_dir / "seg").glob("frame_*.json")):
            data = json.loads(frame_json.read_text())
            for d in data["detections"]:
                d["frame"] = data["frame"]
                chip_dets.append(d)
        add_stage_coords(chip_dets, scan_meta)
        for d in chip_dets:
            if "stage_x" not in d:
                continue
            dets.append(Det(chip, d["stage_x"], d["stage_y"], d["size_um2"], tuple(d["contrast_rgb"])))
    return dets


def match(a: list[Det], b: list[Det], radius_um: float, size_ratio: float) -> list[tuple[Det, Det]]:
    """Mutual nearest neighbour within radius, same chip, similar size."""
    pairs: list[tuple[Det, Det]] = []
    b_xy = np.array([(d.stage_x, d.stage_y) for d in b])
    a_xy = np.array([(d.stage_x, d.stage_y) for d in a])
    for i, da in enumerate(a):
        db_d2 = np.sum((b_xy - a_xy[i]) ** 2, axis=1)
        j = int(np.argmin(db_d2))
        if np.sqrt(db_d2[j]) > radius_um:
            continue
        # mutual: is a_i also b_j's nearest?
        da_d2 = np.sum((a_xy - b_xy[j]) ** 2, axis=1)
        if int(np.argmin(da_d2)) != i:
            continue
        db = b[j]
        if da.chip != db.chip:
            continue
        ratio = max(da.size_um2, db.size_um2) / max(min(da.size_um2, db.size_um2), 1e-6)
        if ratio > size_ratio:
            continue
        pairs.append((da, db))
    return pairs


def report(pairs: list[tuple[Det, Det]], label: str) -> None:
    if not pairs:
        print(f"  {label}: no matches")
        return
    deltas = np.array([[b.contrast_rgb[c] - a.contrast_rgb[c] for c in range(3)] for a, b in pairs])
    print(f"  {label}: N={len(pairs)}")
    for c, ch in enumerate("RGB"):
        d = deltas[:, c]
        print(
            f"    d{ch} = {np.mean(d):+.3f} ± {np.std(d):.3f}  (median {np.median(d):+.3f},"
            f" p10/p90 {np.percentile(d, 10):+.3f}/{np.percentile(d, 90):+.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--match-radius", type=float, default=15.0, help="max stage distance, µm (default 15)")
    parser.add_argument("--size-ratio", type=float, default=1.5, help="max size ratio between matches (default 1.5)")
    parser.add_argument("--min-size", type=float, default=0.0, help="min size_um2 for the filtered breakdown")
    args = parser.parse_args()

    a = load_run(args.run_a)
    b = load_run(args.run_b)
    print(f"{args.run_a.name}: {len(a)} detections   {args.run_b.name}: {len(b)} detections")
    pairs = match(a, b, args.match_radius, args.size_ratio)
    print(f"\ndeltas = {args.run_b.name} - {args.run_a.name}")
    report(pairs, "all matches")
    big = [(x, y) for x, y in pairs if min(x.size_um2, y.size_um2) >= 1000.0]
    report(big, "size >= 1000 um2")
    print("\nper chip:")
    for chip in sorted({x.chip for x, _ in pairs}):
        report([(x, y) for x, y in pairs if x.chip == chip], f"chip {chip}")


if __name__ == "__main__":
    main()
