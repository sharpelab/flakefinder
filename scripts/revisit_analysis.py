"""Use 20x/50x revisit measurements to test the film model and confirm 10x detections.

Reads the CSV from revisit_measure.py and, for one run:

1. Cross-objective background test — each chip's effective oxide Δd is fitted
   from its 10x background alone, then the film model predicts that chip's 20x
   and 50x backgrounds for comparison with the revisit captures.
2. 50x classification — each revisited detection is assigned the nearest
   model layer count at 50x and gated as graphene (near the arc, B negative,
   unclipped); tabulated against the 10x preset class and per chip.
3. Thickness-controlled chip test — for confirmed flakes, the 10x signed offset
   from the preset locus is regressed on chip Δd with layer dummies.

Usage:
    uv run python scripts/revisit_analysis.py revisits.csv run_20260827_1550 -m graphene_thin_90nm_loose
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bg_residue_audit import signed_perp  # noqa: E402
from film_model import OXIDE_NM, bg_ratios, fit_oxide_delta, graphene_contrast  # noqa: E402

from flakefinder.segmentation import DetectorConfig, RGPointDetectorConfig  # noqa: E402

# Golden blank refs (calibration/blank_refs_20260813.json), scan space, per objective. Thin/hBN WB only.
GOLDEN = {
    "thin": {"10x": (1.0098, 2.0440), "20x": (1.0318, 1.9027), "50x": (1.0915, 1.5407)},
    "thick": {"10x": (1.0799, 1.1375)},
}
ARC_GATE = 0.06  # max (R,G) distance from the nearest 50x model arc point
B_GATE = -0.02  # graphene is darker in B; residue is B-bright
CLIP_GATE = 0.02
MIN_CAPTURES = 5


def wb_for(material: str) -> str:
    return "thick" if "thick" in material else "thin"


def nearest_layer(arc: dict[int, np.ndarray], r: float, g: float) -> tuple[int, float]:
    n = min(arc, key=lambda k: (arc[k][0] - r) ** 2 + (arc[k][1] - g) ** 2)
    return n, float(np.hypot(arc[n][0] - r, arc[n][1] - g))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="Output of revisit_measure.py")
    ap.add_argument("run", help="Run name to analyse (e.g. run_20260827_1550)")
    ap.add_argument("-m", "--material", required=True, choices=DetectorConfig.material_names())
    args = ap.parse_args()
    wb = wb_for(args.material)
    gold = GOLDEN[wb]
    f = lambda r, k: float(r[k])  # noqa: E731

    with open(args.csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r["run"] == args.run and r["found"] == "1"]
    if not rows:
        print(f"no measured captures for {args.run} in {args.csv}")
        return 1
    chips = sorted({r["chip"] for r in rows})
    r0 = {o: bg_ratios(o, OXIDE_NM, wb) for o in ("10x", "20x", "50x")}

    # ---- 1. per-chip background: 10x frame modes -> dd -> predicted 20x/50x vs refined capture modes
    print(f"=== {args.run} ({wb} WB): per-chip background, measured vs film model from the 10x-derived effective oxide")
    print(
        f"{'chip':>7} {'n50':>3} {'10x R/G':>8} {'10x B/G':>8} {'dd nm':>6} | "
        f"{'50x R/G':>8} {'pred':>7} {'50x B/G':>8} {'pred':>7} | {'20x R/G':>8} {'pred':>7} {'20x B/G':>8} {'pred':>7}"
    )
    info: dict[str, dict] = {}
    for c in chips:
        rc = [r for r in rows if r["chip"] == c]
        rg10, bg10 = f(rc[0], "chip_rg10"), f(rc[0], "chip_bg10")
        dd = fit_oxide_delta(np.log(rg10 / gold["10x"][0]), np.log(bg10 / gold["10x"][1]), wb)
        meas, pred = {}, {}
        for o in ("20x", "50x"):
            rr = [r for r in rc if r["mag"] == o]
            meas[o] = (
                float(np.median([f(r, "rbgR") / f(r, "rbgG") for r in rr])) if rr else np.nan,
                float(np.median([f(r, "rbgB") / f(r, "rbgG") for r in rr])) if rr else np.nan,
                len(rr),
            )
            if o in gold:
                r1 = bg_ratios(o, OXIDE_NM + dd, wb)
                pred[o] = (gold[o][0] * r1[0] / r0[o][0], gold[o][1] * r1[1] / r0[o][1])
            else:
                pred[o] = (np.nan, np.nan)
        info[c] = {"dd": dd, "meas": meas, "pred": pred}
        print(
            f"{c:>7} {meas['50x'][2]:3d} {rg10:8.3f} {bg10:8.3f} {dd:+6.2f} | "
            f"{meas['50x'][0]:8.4f} {pred['50x'][0]:7.4f} {meas['50x'][1]:8.4f} {pred['50x'][1]:7.4f} | "
            f"{meas['20x'][0]:8.4f} {pred['20x'][0]:7.4f} {meas['20x'][1]:8.4f} {pred['20x'][1]:7.4f}"
        )
    good = [c for c in chips if info[c]["meas"]["50x"][2] >= MIN_CAPTURES]
    if len(good) >= 3 and "50x" in gold:
        x = np.array([info[c]["dd"] for c in good])
        for o in ("50x", "20x"):
            for j, name in ((0, "R/G"), (1, "B/G")):
                y = np.log([info[c]["meas"][o][j] for c in good])
                p = np.log([info[c]["pred"][o][j] for c in good])
                print(
                    f"   {o} {name}: chip-to-chip slope dln/dd measured {np.polyfit(x, y, 1)[0]:+.4f} "
                    f"model {np.polyfit(x, p, 1)[0]:+.4f}  (n={len(good)} chips, corr {np.corrcoef(y, p)[0, 1]:+.2f})"
                )

    # ---- 2. 50x classification vs 10x class
    arc50 = {n: graphene_contrast("50x", n, OXIDE_NM, wb) for n in range(1, 9)}
    r50 = [r for r in rows if r["mag"] == "50x"]
    n50: dict[str, int] = {}
    d50s: dict[str, float] = {}
    ok50: dict[str, bool] = {}
    for r in r50:
        n, d = nearest_layer(arc50, f(r, "rR"), f(r, "rG"))
        n50[r["label"]], d50s[r["label"]] = n, d
        ok50[r["label"]] = d < ARC_GATE and f(r, "rB") < B_GATE and f(r, "clip") < CLIP_GATE
    print(f"\n=== 50x classification (nearest arc, d<{ARC_GATE}, B<{B_GATE}, unclipped) vs 10x class, n={len(r50)}")
    hdr = " ".join(f"N50={n}" for n in range(1, 9))
    print(f"{'10x class':>14} {'n':>4} {'confirmed':>9}  {hdr}   median d50")
    for cls in sorted({r.get("cls", "?") for r in r50}, key=lambda s: (not s[:1].isdigit(), s)):
        m = [r for r in r50 if r.get("cls", "?") == cls]
        ok = [r for r in m if ok50[r["label"]]]
        cnt = [sum(1 for r in ok if n50[r["label"]] == n) for n in range(1, 9)]
        d50 = float(np.median([d50s[r["label"]] for r in m]))
        print(f"{cls:>14} {len(m):4d} {len(ok):9d}  " + " ".join(f"{k:5d}" for k in cnt) + f"   {d50:.3f}")
    print("\nper chip: revisited, 50x-confirmed graphene, precision, N50 median of confirmed")
    for c in chips:
        m = [r for r in r50 if r["chip"] == c]
        ok = [r for r in m if ok50[r["label"]]]
        med = f"{float(np.median([n50[r['label']] for r in ok])):.1f}" if ok else "-"
        print(f"{c:>7} dd {info[c]['dd']:+5.2f}  {len(m):4d} {len(ok):4d} {len(ok) / max(len(m), 1):5.2f}  {med:>5}")

    # ---- 3. thickness-controlled chip test with confirmed flakes
    cfg = DetectorConfig.from_material(args.material)
    if not isinstance(cfg, RGPointDetectorConfig):
        print(f"\n{args.material} has no RG point locus — skipping the locus-offset test")
        return 0
    pts = cfg.cal_reference_points
    print("\n=== confirmed flakes: 10x signed perp offset from the preset locus by 50x layer count and chip (n/median)")
    print(f"{'chip':>7} {'dd':>6} " + " ".join(f"{'N50=' + str(n):>13}" for n in range(1, 6)) + "   all (n/median)")
    xs, ys, ns = [], [], []
    for c in chips:
        ok = [r for r in r50 if r["chip"] == c and ok50[r["label"]]]
        cells = []
        for n in range(1, 6):
            v = [signed_perp(pts, f(r, "R10"), f(r, "G10")) for r in ok if n50[r["label"]] == n and "R10" in r]
            cells.append(f"{len(v):3d}/{np.median(v):+7.3f}" if v else f"{0:3d}/{'-':>7}")
            xs += [info[c]["dd"]] * len(v)
            ys += v
            ns += [n] * len(v)
        allv = [signed_perp(pts, f(r, "R10"), f(r, "G10")) for r in ok if "R10" in r]
        med_all = np.median(allv) if allv else float("nan")
        cells_s = " ".join(f"{s:>13}" for s in cells)
        print(f"{c:>7} {info[c]['dd']:+6.2f} {cells_s}   {len(allv):3d}/{med_all:+7.3f}")
    if len(ys) > 10:
        x, y, n_arr = np.array(xs), np.array(ys), np.array(ns)
        design = np.c_[np.ones(len(x)), x]
        for n in range(2, 6):
            design = np.c_[design, (n_arr == n).astype(float)]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        se = float(np.sqrt(np.sum(resid**2) / (len(y) - design.shape[1]) * np.linalg.inv(design.T @ design)[1, 1]))
        c0, c1 = graphene_contrast("10x", 3, OXIDE_NM, wb), graphene_contrast("10x", 3, OXIDE_NM + 1.0, wb)
        model = ((c1[0] - c1[1]) - (c0[0] - c0[1])) / np.sqrt(2)
        print(
            f"\n   perp ~ dd + layer dummies over {len(y)} confirmed flakes: "
            f"d(perp)/d(dd) = {coef[1]:+.4f} +/- {se:.4f} per nm"
            f"   (film-under-flake model at N=3: {model:+.4f} per nm)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
