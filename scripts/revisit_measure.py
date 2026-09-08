"""Measure revisit captures (20x/50x) of ranked detections and join them to the 10x seg metrics.

Revisit captures are raw PNGs with no analysis attached. For each capture this
records the background (integer histogram mode and the ±5 DN refined mode, per
channel) and the median per-channel contrast of the dark component nearest the
frame centre, which is where the revisited detection sits after the parcentric
correction. Rows are joined to the 10x detection by frame and det_id from the
capture's label, so 10x and higher-mag contrasts of the same flake sit side by side.

Usage:
    uv run python scripts/revisit_measure.py scans/run_20260827_1550 -o revisit_1550.csv
    uv run python scripts/revisit_measure.py scans/run_* -o revisits.csv --mags 50x
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from flakefinder.scan_utils import apply_flatfield
from flakefinder.segmentation import _refined_mode, histogram_mode

PX_UM = {"20x": 0.3603515625, "50x": 0.144140625}
LABEL_RE = re.compile(r"rank(\d+)_frame_(\d+)_d(\d+)")
# Component must be darker than background in both R and G by this much to count as the flake
DARK_CONTRAST = -0.03
MIN_COMPONENT_PX = 40
MAX_CENTRE_OFFSET_FRAC = 0.25


def load_flatfield(path: Path) -> np.ndarray:
    """Correction factors in cv2 (BGR) channel order — the .npy is stored RGB."""
    return np.load(path).astype(np.float32)[:, :, ::-1]


def measure_capture(png: Path, mag: str, flatfield: np.ndarray) -> dict:
    raw = cv2.imread(str(png))
    if raw is None:
        raise FileNotFoundError(png)
    im = apply_flatfield(raw, flatfield).astype(np.float32)
    h, w, _ = im.shape
    modes = np.array([histogram_mode(im, c) for c in range(3)])
    refined = np.array([_refined_mode(im[:, :, c].ravel()) for c in range(3)])
    out = {
        "bgB": modes[0],
        "bgG": modes[1],
        "bgR": modes[2],
        "rbgB": refined[0],
        "rbgG": refined[1],
        "rbgR": refined[2],
        "clip": float((raw.max(axis=2) >= 250).mean()),
        "found": 0,
    }
    contrast = im / np.maximum(modes, 1.0) - 1.0
    dark = (contrast[:, :, 1] < DARK_CONTRAST) & (contrast[:, :, 2] < DARK_CONTRAST)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    best: tuple[float, int] | None = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_PX:
            continue
        d = float(np.hypot(centroids[i][0] - w / 2, centroids[i][1] - h / 2))
        if d > MAX_CENTRE_OFFSET_FRAC * w:
            continue
        if best is None or d < best[0]:
            best = (d, i)
    if best is None:
        return out
    comp = (labels == best[1]).astype(np.uint8)
    # Erode so the median samples the flake interior, not its anti-aliased rim
    core = cv2.erode(comp, np.ones((5, 5), np.uint8)) if stats[best[1], cv2.CC_STAT_AREA] > 400 else comp
    m = core.astype(bool)
    out.update(
        found=1,
        dist_px=best[0],
        area_um2=float(stats[best[1], cv2.CC_STAT_AREA]) * PX_UM[mag] ** 2,
        npx=int(m.sum()),
        R=float(np.median(contrast[:, :, 2][m])),
        G=float(np.median(contrast[:, :, 1][m])),
        B=float(np.median(contrast[:, :, 0][m])),
        rR=float(np.median(im[:, :, 2][m]) / max(refined[2], 1.0) - 1.0),
        rG=float(np.median(im[:, :, 1][m]) / max(refined[1], 1.0) - 1.0),
        rB=float(np.median(im[:, :, 0][m]) / max(refined[0], 1.0) - 1.0),
    )
    return out


def measure_run(run_dir: Path, mags: list[str], flatfields: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    for chip_dir in sorted(run_dir.glob("chip_*")):
        summary_path = chip_dir / "seg" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        modes = np.array([v for v in summary["bg_mode_by_frame"].values() if v], float)
        chip_rg = modes[:, 0].mean() / modes[:, 1].mean() if len(modes) else float("nan")
        chip_bg = modes[:, 2].mean() / modes[:, 1].mean() if len(modes) else float("nan")
        dets = {(d["frame"], d["det_id"]): d for v in summary["detections_by_frame"].values() for d in v}
        for mag in mags:
            for png in sorted((chip_dir / f"revisit_{mag}").glob("*.png")):
                m = LABEL_RE.match(png.stem)
                if not m:
                    continue
                row = {
                    "run": run_dir.name,
                    "chip": chip_dir.name,
                    "mag": mag,
                    "rank": int(m[1]),
                    "label": png.stem,
                    "chip_rg10": chip_rg,
                    "chip_bg10": chip_bg,
                }
                d10 = dets.get((f"frame_{m[2]}", int(m[3])))
                if d10:
                    row.update(
                        R10=d10["contrast_rgb"][0],
                        G10=d10["contrast_rgb"][1],
                        B10=d10["contrast_rgb"][2],
                        size10=d10["size_um2"],
                        tier=d10["tier"],
                        cls=d10["classification"],
                        cal_dist=d10["cal_dist"],
                        ent=d10["entropy"],
                        sol=d10["solidity"],
                    )
                row.update(measure_capture(png, mag, flatfields[mag]))
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True, help="CSV to write (overwritten)")
    ap.add_argument("--mags", default="20x,50x", help="Revisit mags to measure (default 20x,50x)")
    ap.add_argument(
        "--flatfield-dir", type=Path, default=Path("calibration"), help="Dir holding flatfield_{mag}_5800K_bin3.npy"
    )
    args = ap.parse_args()
    mags = args.mags.split(",")
    flatfields = {m: load_flatfield(args.flatfield_dir / f"flatfield_{m}_5800K_bin3.npy") for m in mags}
    rows: list[dict] = []
    for run_dir in args.run_dirs:
        rows += measure_run(run_dir, mags, flatfields)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("run", "chip", "mag", "rank", "label"), k))
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
