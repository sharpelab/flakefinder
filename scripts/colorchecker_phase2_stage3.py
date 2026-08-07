#!/usr/bin/env python
"""Analyze ColorChecker Phase 2 Stage 3 per-patch capture series.

Reads patch_<name>/ dirs produced by capture-series with stage3_patch_spec.json
(exposure ladder 0.2/0.5/2/8 ms x configs: unity idx2, scan WB idx2, unity idx0,
M-cancellation WB trio under idx2).

Per patch/config the brightest unclipped rung (ROI max < 250 in every channel —
blooming rule) is converted to a pedestal-corrected rate: (mean + c)/ms.

Analyses (calibration/colorchecker_phase2/PLAN.md Stage 3):
  A. Raw (idx0) patch rates + ratios to white.
  B. CCM validation: predict unity_idx2 = C @ raw per patch (out-of-sample).
  C. Scan-config validation: predict scanwb_idx2 = C @ (w_scan * raw).
  D. M-cancellation raw extraction (CCM-free) vs direct idx0 rates.
  E. Spectral model comparison: BabelColor reflectance x lamp x T^2 chain
     vs measured raw — lamp SPD validation on spectrally distinct targets.
  F. Black-patch glare floors, raw + output space, vs pre-registered values.

Usage:
    uv run python scripts/colorchecker_phase2_stage3.py \
        calibration/colorchecker_phase2_20260806
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import colorchecker_cal as cal
from colorchecker_phase2_stage1 import CH, fmt, load_conditions

PEDESTAL = 1.02
RUNGS = [("e8", 8.0), ("e2", 2.0), ("e05", 0.5), ("e02", 0.2)]  # brightest first
CONFIGS = ["unity_idx2", "scanwb_idx2", "unity_idx0"]
TRIO = [("wbr2_idx2", 0, 1.0), ("wbg13_idx2", 1, 0.3), ("wbb2_idx2", 2, 1.0)]
W_SCAN = np.array([1.41, 1.02, 2.51])
PREREG_F_RAW = np.array([0.031, 0.014, 0.045])
PREREG_F_OUT = np.array([0.042, 0.009, 0.100])


def best_rate(cond: dict[str, dict], prefix: str) -> tuple[np.ndarray, str] | None:
    """Brightest unclipped rung as pedestal-corrected counts/ms."""
    for tag, ms in RUNGS:
        c = cond.get(f"{prefix}_{tag}")
        if c is None or c["max"].max() >= 250:
            continue
        return (c["mean"] + PEDESTAL) / ms, tag
    return None


def trio_response(cond: dict[str, dict], j: int, name: str, delta: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(K[:,j], unity_mean) with K = (out_wb - out_unity)/delta in rate units, best shared rung."""
    for tag, ms in RUNGS:
        wb, un = cond.get(f"{name}_{tag}"), cond.get(f"unity_idx2_{tag}")
        if wb is None or un is None or wb["max"].max() >= 250 or un["max"].max() >= 250:
            continue
        return (wb["mean"] - un["mean"]) / delta / ms, un["mean"]
    return None


def load_reflectances(csv: Path, lamb: np.ndarray) -> dict[int, np.ndarray]:
    rows = csv.read_text().strip().splitlines()
    wl = np.array([float(x) for x in rows[0].split(",")[2:]])
    out = {}
    for row in rows[1:]:
        parts = row.split(",")
        out[int(parts[0])] = np.interp(lamb, wl, np.array([float(x) for x in parts[2:]]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir", type=Path, help="phase2 dir containing patch_*/ and ccm_models.json")
    ap.add_argument("--roi", type=int, default=300)
    args = ap.parse_args()

    ccm = np.array(json.loads((args.base_dir / "ccm_models.json").read_text())["idx2_real_units"])
    patches: dict[str, dict] = {}
    for d in sorted(args.base_dir.glob("patch_*")):
        patches[d.name.removeprefix("patch_")] = load_conditions(d, args.roi)
    if "white19" not in patches:
        print("need patch_white19 as reference", file=sys.stderr)
        return 1
    names = ["white19"] + [n for n in patches if n != "white19"]
    analysis: dict = {"roi_px": args.roi, "pedestal": PEDESTAL, "patches": names}

    # Rates per patch/config
    rates: dict[str, dict[str, np.ndarray]] = {}
    rungs_used: dict[str, dict[str, str]] = {}
    print("=== Rates (counts/ms, pedestal-corrected; rung in parens) ===")
    print(f"{'patch':<11} " + " ".join(f"{c:<26}" for c in CONFIGS))
    for n in names:
        rates[n], rungs_used[n] = {}, {}
        cells = []
        for cfg in CONFIGS:
            r = best_rate(patches[n], cfg)
            if r is None:
                cells.append(f"{'CLIPPED/MISSING':<26}")
                continue
            rates[n][cfg], rungs_used[n][cfg] = r
            cells.append(f"{fmt(r[0], 2) + ' (' + r[1] + ')':<26}")
        print(f"{n:<11} " + " ".join(cells))
    analysis["rates_counts_per_ms"] = {
        n: {c: v[0].tolist() for c, v in [(c2, (rates[n][c2], 0)) for c2 in rates[n]]} for n in names
    }
    analysis["rungs_used"] = rungs_used

    raw_w = rates["white19"]["unity_idx0"]

    # A. Raw ratios to white
    print("\n=== A. Raw (idx0) patch/white ratios ===")
    raw_ratio = {}
    for n in names:
        rr = rates[n]["unity_idx0"] / raw_w
        raw_ratio[n] = rr
        print(f"{n:<11} {fmt(rr, 4)}")
    analysis["raw_ratio_to_white"] = {n: v.tolist() for n, v in raw_ratio.items()}

    # B/C. CCM + scan-config validation
    print("\n=== B. unity_idx2 predicted from raw via CCM (err % of white-channel rate) ===")
    print("=== C. scanwb_idx2 predicted from CCM @ (w_scan * raw) ===")
    ref_out = rates["white19"]["unity_idx2"]
    errs_b, errs_c = {}, {}
    clamped_b: dict[str, list[str]] = {}
    for n in names:
        raw = rates[n]["unity_idx0"]
        pred_b = ccm @ raw
        err_b = (pred_b - rates[n]["unity_idx2"]) / ref_out * 100
        # Output floor clamp: a negative CCM prediction cannot be observed —
        # the pipeline clamps at 0. Flag instead of scoring as model error.
        clamp = pred_b < 0
        clamped_b[n] = [CH[i] for i in np.where(clamp)[0]]
        err_b = np.where(clamp, np.nan, err_b)
        pred_c = ccm @ (W_SCAN * raw)
        meas_c = rates[n].get("scanwb_idx2")
        err_c = (pred_c - meas_c) / (ccm @ (W_SCAN * raw_w)) * 100 if meas_c is not None else None
        errs_b[n], errs_c[n] = err_b, err_c
        note = f"  [pred<0 -> clamped: {','.join(clamped_b[n])}]" if clamped_b[n] else ""
        cc = fmt(err_c, 2) if err_c is not None else "n/a"
        print(f"{n:<11} B err {fmt(err_b, 2):<20} C err {cc}{note}")
    analysis["ccm_validation_err_pct_of_white"] = {n: v.tolist() for n, v in errs_b.items()}
    analysis["scanwb_validation_err_pct_of_white"] = {
        n: (v.tolist() if v is not None else None) for n, v in errs_c.items()
    }

    # D. M-cancellation raw extraction (CCM-free)
    print("\n=== D. M-cancellation raw ratios vs direct idx0 (per WB axis) ===")
    mc = {}
    for n in names:
        if n == "white19":
            continue
        row = {}
        for name, j, delta in TRIO:
            rp = trio_response(patches[n], j, name, delta)
            rw = trio_response(patches["white19"], j, name, delta)
            if rp is None or rw is None:
                continue
            (kp, un_p), (kw, _) = rp, rw
            # A floor-clamped unity reference channel corrupts K on that row;
            # exclude clamped output rows from the row choice.
            c_best = int(np.argmax(np.abs(kw) * (un_p > 0.5)))
            row[CH[j]] = {
                "ratio_best_row": float(kp[c_best] / kw[c_best]),
                "ratio_all_rows": (kp / kw).tolist(),
                "direct_idx0": float(raw_ratio[n][j]),
            }
        mc[n] = row
        cells = [f"{ax}: {row[ax]['ratio_best_row']:.4f} vs {row[ax]['direct_idx0']:.4f}" for ax in row]
        print(f"{n:<11} " + "   ".join(cells))
    analysis["m_cancellation"] = mc

    # E. Spectral model comparison (BabelColor x lamp x T^2 chain)
    lamb = cal._IMX183_WAVELENGTHS
    model = cal.derive_pipeline(lamb)
    chain = model.objectives["10x"].chain_spd
    refl = load_reflectances(Path("calibration/reference/colorchecker_spectra_babelcolor.csv"), lamb)
    patch_no = {n: int("".join(ch for ch in n if ch.isdigit())) for n in names}
    print("\n=== E. Spectral model: predicted vs measured raw, patch/white per channel ===")
    v_w = cal._channel_integrals(chain * refl[19], lamb)
    preds = {n: cal._channel_integrals(chain * refl[patch_no[n]], lamb) / v_w for n in names}
    # Glare-correct with the black-patch raw floor before judging the lamp
    # model (black itself then matches by construction — excluded from verdict).
    f_raw_glare = raw_ratio["black24"] - preds["black24"] if "black24" in preds else np.zeros(3)
    spect = {}
    for n in names:
        pred, meas = preds[n], raw_ratio[n]
        corr = meas - f_raw_glare
        spect[n] = {
            "pred": pred.tolist(),
            "meas": meas.tolist(),
            "meas_over_pred": (meas / pred).tolist(),
            "glare_corrected_over_pred": (corr / pred).tolist(),
        }
        tag = "  [defines glare floor]" if n == "black24" else ""
        print(f"{n:<11} pred {fmt(pred, 4):<24} meas {fmt(meas, 4):<24} corr/pred {fmt(corr / pred, 3)}{tag}")
    analysis["spectral_model"] = spect

    # F. Black glare floors
    if "black24" in patches:
        pred_black = np.array(spect["black24"]["pred"])
        f_raw = raw_ratio["black24"] - pred_black
        out_ratio = rates["black24"]["unity_idx2"] / rates["white19"]["unity_idx2"]
        v_out_pred = ccm @ (np.array(spect["black24"]["pred"]) * raw_w)
        f_out = out_ratio - v_out_pred / (ccm @ raw_w)
        print("\n=== F. Black #24 glare floors (measured - spectral expectation) ===")
        print(f"raw:    f = {fmt(f_raw, 4)}   (pre-registered {fmt(PREREG_F_RAW, 3)})")
        print(f"output: f = {fmt(f_out, 4)}   (pre-registered {fmt(PREREG_F_OUT, 3)})")
        analysis["glare_floors"] = {
            "f_raw": f_raw.tolist(),
            "f_out": f_out.tolist(),
            "prereg_raw": PREREG_F_RAW.tolist(),
            "prereg_out": PREREG_F_OUT.tolist(),
        }

    (args.base_dir / "stage3_analysis.json").write_text(json.dumps(analysis, indent=1))
    print(f"\nwrote {args.base_dir}/stage3_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
