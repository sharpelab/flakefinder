"""Backfill det_id into summary.json and fix revisit filenames for old runs.

Old runs stored d0 in revisit labels (because det_id was never persisted).
This script adds det_id to summary.json detections, re-derives the ranked
list to recover the rank→det_id mapping, and renames revisit image files.

Usage:
    uv run python scripts/repair_det_id.py scans/run_20260220_2344
    uv run python scripts/repair_det_id.py scans/run_20260220_2344 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from flakefinder.segmentation import dedup_detections


def repair_run(run_dir: Path, *, dry_run: bool = False) -> None:
    """Add det_id to summary.json and fix revisit labels/filenames."""
    label_re = re.compile(r"^(rank(\d+))_(frame_\d+)_d(\d+)$")

    for chip_dir in sorted(run_dir.glob("chip_*")):
        if not chip_dir.is_dir():
            continue
        seg_dir = chip_dir / "seg"
        summary_path = seg_dir / "summary.json"
        if not summary_path.exists():
            continue

        # Phase 1: Load and fix summary in memory
        with open(summary_path) as f:
            summary = json.load(f)

        summary_changed = False
        for _frame_name, dets in summary.get("detections_by_frame", {}).items():
            for i, d in enumerate(dets):
                if "det_id" not in d:
                    d["det_id"] = i
                    summary_changed = True

        if summary_changed:
            if dry_run:
                print(f"  Would fix: {summary_path}")
            else:
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"  Fixed: {summary_path}")

        # Phase 2: Re-derive ranked list to recover rank→det_id mapping.
        # This reproduces the logic from _generate_revisits in find_flakes.py.
        all_flat = []
        for frame_name, dets in summary.get("detections_by_frame", {}).items():
            for d in dets:
                d.setdefault("frame", frame_name)
                all_flat.append(d)

        ranked = sorted(all_flat, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        ranked = dedup_detections(ranked)
        ranked = [d for d in ranked if d.get("tier") == 1]

        # Build rank→(frame, det_id) mapping (1-indexed)
        rank_map: dict[int, tuple[str, int]] = {}
        for i, d in enumerate(ranked):
            rank_map[i + 1] = (d["frame"], d["det_id"])

        # Phase 3: Fix revisit JSON labels and rename images
        for revisit_json_path in sorted(seg_dir.glob("revisit_*.json")):
            mag_match = re.search(r"revisit_(\d+x)", revisit_json_path.name)
            if not mag_match:
                continue
            mag_str = mag_match.group(1)
            revisit_dir = chip_dir / f"revisit_{mag_str}"

            with open(revisit_json_path) as f:
                revisit_data = json.load(f)

            renames: list[tuple[Path, Path]] = []
            json_changed = False

            for point in revisit_data.get("points", []):
                label = point.get("label", "")
                m = label_re.match(label)
                if not m:
                    continue

                rank_prefix = m.group(1)  # e.g. rank01
                rank_num = int(m.group(2))
                old_frame = m.group(3)
                old_d = int(m.group(4))

                mapping = rank_map.get(rank_num)
                if not mapping:
                    continue

                real_frame, real_det_id = mapping
                if old_frame != real_frame:
                    print(f"  WARNING: rank {rank_num} frame mismatch: label={old_frame} vs derived={real_frame}")
                    continue

                if old_d != real_det_id:
                    new_label = f"{rank_prefix}_{real_frame}_d{real_det_id}"
                    point["label"] = new_label
                    json_changed = True

                    if revisit_dir.is_dir():
                        old_img = revisit_dir / f"{label}_{mag_str}.png"
                        new_img = revisit_dir / f"{new_label}_{mag_str}.png"
                        if old_img.exists():
                            renames.append((old_img, new_img))

            if json_changed:
                if dry_run:
                    print(f"  Would fix labels: {revisit_json_path}")
                    for old, new in renames:
                        print(f"    rename: {old.name} → {new.name}")
                else:
                    with open(revisit_json_path, "w") as f:
                        json.dump(revisit_data, f, indent=2)
                    print(f"  Fixed labels: {revisit_json_path}")
                    for old, new in renames:
                        old.rename(new)
                        print(f"    renamed: {old.name} → {new.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill det_id and fix revisit filenames")
    parser.add_argument("run_dir", type=Path, help="Run directory to repair")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: {args.run_dir} is not a directory")
        return 1

    repair_run(args.run_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
