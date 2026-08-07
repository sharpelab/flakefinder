#!/usr/bin/env python
"""Analyze ColorChecker Phase 2 Stage 1 capture series (CCM + WB decoupling).

Reads a stage1/ directory of captures named <condition>_f<N>.png produced by
capture-series with stage1_ccm_wb_spec.json (white patch #19, 10x, motion-free).

Analysis (calibration/colorchecker_phase2/PLAN.md):
  1. M_idx2 WB-coupling columns vs session-1 wb_model.json — method valid today?
  2. M_idx0 WB-coupling columns — decision: diagonal => idx0 is the clean CCM.
  3. unity_idx0 white fingerprint vs pre-registered raw prediction (G/R, G/B).
  4. Real-unit CCM response columns K[:,j] = out_wbj - out_base (= C[:,j]*v_j),
     column-normalized to compare with the pre-registered 5800K extraction.
  5. Cap check: R effective mult at requests 4/6/8 (0.15 ms block).
  6. sat0 luma weights from three WB points.
  7. Drift: drift_idx2 vs base_idx2.

Outputs ccm_models.json + stage1_analysis.json in the parent calibration dir.

Usage:
    uv run python scripts/colorchecker_phase2_stage1.py \
        calibration/colorchecker_phase2_20260806/stage1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CH = "RGB"


def roi_stats(path: Path, roi: int) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    rgb = img[:, :, ::-1].astype(float)
    h, w = rgb.shape[:2]
    core = rgb[h // 2 - roi // 2 : h // 2 + roi // 2, w // 2 - roi // 2 : w // 2 + roi // 2]
    return {
        "mean": core.mean(axis=(0, 1)),
        "std": core.std(axis=(0, 1)),
        "max": core.max(axis=(0, 1)),
        "clip_frac": (core >= 250).mean(axis=(0, 1)),
    }


def load_conditions(stage_dir: Path, roi: int) -> dict[str, dict]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(stage_dir.glob("*.png")):
        m = re.match(r"(.+)_f(\d+)$", p.stem)
        if m:
            groups[m.group(1)].append(p)
    out = {}
    for name, paths in groups.items():
        stats = [roi_stats(p, roi) for p in paths]
        means = np.array([s["mean"] for s in stats])
        out[name] = {
            "n_frames": len(paths),
            "mean": means.mean(axis=0),
            "frame_std": means.std(axis=0),
            "pixel_std": np.array([s["std"] for s in stats]).mean(axis=0),
            "max": np.array([s["max"] for s in stats]).max(axis=0),
            "clip_frac": np.array([s["clip_frac"] for s in stats]).mean(axis=0),
        }
    return out


def fmt(v: np.ndarray, prec: int = 2) -> str:
    return "/".join(f"{x:.{prec}f}" for x in v)


def wb_columns(cond: dict[str, dict], base: str, trio: list[str]) -> np.ndarray:
    """M columns from single-channel WB doublings: M[:,j] = out_wbj/out_base - 1."""
    b = cond[base]["mean"]
    return np.stack([cond[t]["mean"] / b - 1.0 for t in trio], axis=1)


def response_columns(cond: dict[str, dict], base: str, trio: list[str]) -> np.ndarray:
    """Real-unit CCM response columns K[:,j] = out_wbj - out_base = C[:,j]*v_j."""
    b = cond[base]["mean"]
    return np.stack([cond[t]["mean"] - b for t in trio], axis=1)


def col_normalize(k: np.ndarray) -> np.ndarray:
    """Divide each column by its diagonal element -> diag = 1 (v_j cancels)."""
    return k / np.diag(k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage_dir", type=Path, help="stage1 capture directory")
    ap.add_argument("--roi", type=int, default=300, help="central ROI size in px (default 300)")
    ap.add_argument(
        "--wb-model",
        type=Path,
        default=Path("calibration/colorchecker_dm6m_20260806/wb_model.json"),
        help="session-1 wb_model.json for comparison",
    )
    ap.add_argument(
        "--fillin",
        type=Path,
        default=None,
        help="stage1_fillin dir (sub-clamp wbg steps) to measure clamped M_idx2 entries",
    )
    args = ap.parse_args()

    cond = load_conditions(args.stage_dir, args.roi)
    out_dir = args.stage_dir.parent
    analysis: dict = {"roi_px": args.roi, "source": str(args.stage_dir)}

    print(f"=== Conditions ({len(cond)}) ===")
    print(f"{'name':<20} {'mean RGB':<22} {'frame_std':<16} {'max':<14} clip%")
    for name, c in cond.items():
        clip = fmt(c["clip_frac"] * 100, 2)
        print(f"{name:<20} {fmt(c['mean']):<22} {fmt(c['frame_std'], 3):<16} {fmt(c['max'], 0):<14} {clip}")
    analysis["conditions"] = {
        n: {k: np.asarray(v).tolist() for k, v in c.items() if k != "n_frames"} | {"n_frames": c["n_frames"]}
        for n, c in cond.items()
    }

    # 1. idx2 WB coupling vs session-1 wb_model
    trio2 = ["wbr2_idx2", "wbg2_idx2", "wbb2_idx2"]
    m_idx2 = wb_columns(cond, "base_idx2", trio2)
    # Floor-clamp detection: output channel driven to ~0 makes the entry read
    # -1.0 exactly (the clamp), not the true coupling. Substitute session-1 M
    # (measured with sub-clamp WB steps) for those entries.
    base2 = cond["base_idx2"]["mean"]
    clamped = np.stack([cond[t]["mean"] < 0.5 for t in trio2], axis=1) & (base2[:, None] > 5)
    print("\n=== 1. M_idx2 (WB coupling under 5800K CCM), cols = req R/G/B ===")
    print(np.array_str(m_idx2, precision=3, suppress_small=True))
    wbm = None
    if args.wb_model.exists():
        wbm = json.loads(args.wb_model.read_text())
        m_s1 = np.array(wbm["M_rows_RGB_cols_reqRGB"])
        print("session-1 M:")
        print(np.array_str(m_s1, precision=3, suppress_small=True))
        if clamped.any():
            idx = [f"[{CH[i]},req{CH[j]}]" for i, j in zip(*np.where(clamped), strict=True)]
            if args.fillin and args.fillin.exists():
                # Re-measure the G column from sub-clamp steps: M[:,G] = (out/base - 1)/delta
                fc = load_conditions(args.fillin, args.roi)
                fb = fc["base_idx2"]["mean"]
                g13 = (fc["wbg13_idx2"]["mean"] / fb - 1.0) / 0.3
                g15 = (fc["wbg15_idx2"]["mean"] / fb - 1.0) / 0.5
                print(f"floor-clamped entries {idx}: G column re-measured from fill-in")
                print(f"  wbg1.3 col: {fmt(g13, 3)}   wbg1.5 col: {fmt(g15, 3)}")
                m_idx2[:, 1] = g13  # wbg1.3: B stays furthest from the floor clamp
                analysis["m_idx2_g_column_fillin"] = {"wbg13": g13.tolist(), "wbg15": g15.tolist()}
            else:
                print(f"floor-clamped entries {idx}: substituting session-1 values")
                m_idx2 = np.where(clamped, m_s1, m_idx2)
        valid = ~clamped
        print(f"max |diff| (unclamped entries): {np.abs(m_idx2 - m_s1)[valid].max():.3f}")
        analysis["m_idx2_vs_session1_max_abs_diff"] = float(np.abs(m_idx2 - m_s1)[valid].max())
        analysis["m_idx2_clamped_entries_substituted"] = clamped.tolist()
    analysis["M_idx2"] = m_idx2.tolist()

    # 2. idx0 WB coupling — the decision measurement
    print("\n=== 2. M_idx0 (WB coupling under UserDefinedMatrix) ===")
    for base, trio, tag in [
        ("unity_idx0", ["wbr2_idx0", "wbg2_idx0", "wbb2_idx0"], "0.5ms"),
        ("unity_idx0_e02", ["wbr2_idx0_e02", "wbg2_idx0_e02", "wbb2_idx0_e02"], "0.2ms"),
    ]:
        m = wb_columns(cond, base, trio)
        off = m[~np.eye(3, dtype=bool)]
        print(f"[{tag}]")
        print(np.array_str(m, precision=3, suppress_small=True))
        print(f"  max |off-diag|: {np.abs(off).max():.3f}  (idx2 reference max: 1.62)")
        analysis[f"M_idx0_{tag}"] = m.tolist()
        analysis[f"M_idx0_{tag}_max_abs_offdiag"] = float(np.abs(off).max())

    # 3. unity_idx0 fingerprint vs pre-registered raw prediction
    u0, u2 = cond["unity_idx0"]["mean"], cond["base_idx2"]["mean"]
    fp0 = {"G/R": u0[1] / u0[0], "G/B": u0[1] / u0[2], "R/B": u0[0] / u0[2]}
    fp2 = {"G/R": u2[1] / u2[0], "G/B": u2[1] / u2[2], "R/B": u2[0] / u2[2]}
    print("\n=== 3. White fingerprints (unity WB) ===")
    print(f"idx0: G/R {fp0['G/R']:.3f}  G/B {fp0['G/B']:.3f}  R/B {fp0['R/B']:.3f}   (pre-reg raw: 1.68 / 2.63)")
    print(f"idx2: G/R {fp2['G/R']:.3f}  G/B {fp2['G/B']:.3f}  R/B {fp2['R/B']:.3f}   (session-1: 1.916 / 4.987)")
    analysis["fingerprint_idx0"] = {k: float(v) for k, v in fp0.items()}
    analysis["fingerprint_idx2"] = {k: float(v) for k, v in fp2.items()}

    # 4. CCM response columns, column-normalized (diag = 1)
    # K[:,j] = base * M[:,j] uses the clamp-corrected M for idx2.
    k2 = base2[:, None] * m_idx2
    k0 = response_columns(cond, "unity_idx0", ["wbr2_idx0", "wbg2_idx0", "wbb2_idx0"])
    c2n, c0n = col_normalize(k2), col_normalize(k0)
    prereg = np.array([[1.0, -0.245, -0.140], [-0.281, 1.0, -0.127], [-0.075, -0.206, 1.0]])
    print("\n=== 4. CCM ===")
    print("idx2 measured (column-normalized):")
    print(np.array_str(c2n, precision=3, suppress_small=True))
    print("idx0 measured (column-normalized):")
    print(np.array_str(c0n, precision=3, suppress_small=True))
    # Real units: idx0 is identity (section 7b), so unity_idx0 + pedestal IS the
    # raw white vector v. C2_real[:,j] = K2[:,j] / v_j, no conventions assumed.
    # The pre-registered table is ROW-normalized (ccm / diag per row) — compare
    # like with like.
    ped = 1.02
    v_meas = cond["unity_idx0"]["mean"] + ped
    c2_real = k2 / v_meas[None, :]
    c2_rownorm = c2_real / np.diag(c2_real)[:, None]
    print("idx2 measured (real units, v from unity_idx0 + pedestal):")
    print(np.array_str(c2_real, precision=3, suppress_small=True))
    print("idx2 measured (row-normalized) vs pre-registered extraction:")
    print(np.array_str(c2_rownorm, precision=3, suppress_small=True))
    print(np.array_str(prereg, precision=3, suppress_small=True))
    print(f"max |diff| row-norm vs pre-reg: {np.abs(c2_rownorm - prereg).max():.3f}")
    analysis["ccm_idx2_colnorm"] = c2n.tolist()
    analysis["ccm_idx0_colnorm"] = c0n.tolist()
    analysis["ccm_idx2_real"] = c2_real.tolist()
    analysis["ccm_idx2_rownorm"] = c2_rownorm.tolist()
    analysis["ccm_idx2_vs_prereg_max_abs_diff"] = float(np.abs(c2_rownorm - prereg).max())

    # White signatures of all four CCM indices (unity WB)
    sigs = {
        "idx0": u0,
        "idx1": cond["ccm_idx1"]["mean"],
        "idx2": u2,
        "idx3": cond["ccm_idx3"]["mean"],
    }
    print("\nWhite signatures (unity WB): " + "  ".join(f"{k}={fmt(v, 1)}" for k, v in sigs.items()))

    # 5. Cap check (0.15 ms block, idx0) + pedestal reconciliation.
    # Over-unity effective mults are consistent with out = g*(raw) - c, c ~ 1
    # count over-subtracted black level (session-1 intercepts -1.1..-1.4):
    # solve c = b*(eff - g)/(g - 1) at every idx0 gain point.
    b015 = cond["unity_idx0_e015"]["mean"]
    print("\n=== 5. WB cap check (R requests 4/6/8 @ 0.15 ms, idx0) ===")
    cap = {}
    ped_pts = []
    for req, name in [(4, "wbr4_idx0_e015"), (6, "wbr6_idx0_e015"), (8, "wbr8_idx0_e015")]:
        eff = cond[name]["mean"][0] / b015[0]
        mx, cf = cond[name]["max"][0], cond[name]["clip_frac"][0] * 100
        c = b015[0] * (eff - req) / (req - 1)
        ped_pts.append(("e015", f"R{req}", c))
        print(f"  req {req}: effective x{eff:.3f}  (R max {mx:.0f}, clip {cf:.2f}%, pedestal c={c:.2f})")
        cap[str(req)] = {"effective_mult": float(eff), "r_max": float(mx), "clip_pct": float(cf)}
    for base, trio, tag in [
        ("unity_idx0", ["wbr2_idx0", "wbg2_idx0", "wbb2_idx0"], "0.5ms"),
        ("unity_idx0_e02", ["wbr2_idx0_e02", "wbg2_idx0_e02", "wbb2_idx0_e02"], "0.2ms"),
    ]:
        for ch, t in enumerate(trio):
            b, eff = cond[base]["mean"][ch], cond[t]["mean"][ch] / cond[base]["mean"][ch]
            ped_pts.append((tag, CH[ch] + "2", b * (eff - 2.0)))
    peds = np.array([c for _, _, c in ped_pts])
    print(f"pedestal c across {len(peds)} idx0 gain points: {peds.mean():.2f} +/- {peds.std():.2f} counts")
    analysis["cap_check"] = cap
    analysis["pedestal_counts"] = {"mean": float(peds.mean()), "std": float(peds.std()), "n": len(peds)}

    # 6. sat0 luma weights: solve w from L_i = w . x_i
    m_cols = m_idx2  # today's coupling for the scan-WB prediction
    delta_scan = np.array([0.41, 0.02, 1.51])
    x_scan = (1.0 + m_cols @ delta_scan) * cond["base_idx2"]["mean"]
    pairs = [
        (cond["sat0_unity_idx2"]["mean"], cond["base_idx2"]["mean"]),
        (cond["sat0_wbr2_idx2"]["mean"], cond["wbr2_idx2"]["mean"]),
        (cond["sat0_scanwb_idx2"]["mean"], x_scan),
    ]
    grayness = [float(np.ptp(luma)) for luma, _ in pairs]
    x_mat = np.stack([x for _, x in pairs])
    l_vec = np.array([luma.mean() for luma, _ in pairs])
    w = np.linalg.solve(x_mat, l_vec)
    print("\n=== 6. Luma weights (sat0) ===")
    print(f"weights RGB: {fmt(w, 4)}  (sum {w.sum():.4f}; provisional was 0.29/0.52/0.18)")
    print(f"sat0 channel spread (should be ~0): {[f'{g:.2f}' for g in grayness]}")
    analysis["luma_weights_rgb"] = w.tolist()
    analysis["sat0_channel_spread"] = grayness

    # 7b. idx0 diagonal estimate under the row-sum-1 convention.
    # WB-preserving CCMs conventionally have rows summing to 1. If C2_real =
    # colnorm @ diag(s) with rows summing to 1, s solves colnorm_rowsum
    # equations; then v_raw = C2_real^-1 @ base_idx2 and D = unity_idx0 / v_raw.
    # D proportional to (1,1,1) => idx0 is identity (up to exposure scale).
    s = np.linalg.solve(c2n, np.ones(3))
    c2_real = c2n * s
    v_raw = np.linalg.solve(c2_real, cond["base_idx2"]["mean"])
    d_diag = cond["unity_idx0"]["mean"] / v_raw
    print("\n=== 7b. idx0 diagonal (row-sum-1 assumption on idx2 CCM) ===")
    print(f"column scales s: {fmt(s, 3)}")
    ratios = f"G/R {v_raw[1] / v_raw[0]:.3f} G/B {v_raw[1] / v_raw[2]:.3f}"
    print(f"v_raw (implied raw white): {fmt(v_raw)}   ratios {ratios}")
    print(f"D = unity_idx0 / v_raw: {fmt(d_diag, 3)}   normalized: {fmt(d_diag / d_diag[1], 3)}")
    analysis["idx0_diag_rowsum1"] = {
        "column_scales": s.tolist(),
        "v_raw": v_raw.tolist(),
        "D": d_diag.tolist(),
        "D_over_Dg": (d_diag / d_diag[1]).tolist(),
    }

    # 7. Drift / state restore
    d = cond["drift_idx2"]["mean"] / cond["base_idx2"]["mean"] - 1.0
    print(f"\n=== 7. Drift: drift_idx2 vs base_idx2: {fmt(d * 100, 2)} % ===")
    analysis["drift_pct_rgb"] = (d * 100).tolist()

    meta_path = args.stage_dir / "capture_series_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ccm_models = {
        "scope_id": "leica_dm6m_sharpelab",
        "camera": "Leica K5C",
        "date": meta.get("timestamp", ""),
        "measured_at": {"objective": "10x", "target": "colorchecker_white_19"},
        "notes": (
            "Column-normalized CCMs (diag=1) from single-channel WB doublings on white #19: "
            "K[:,j] = out_wbj - out_base = C[:,j]*v_j; column-normalizing cancels v_j. "
            "idx1/idx3: white signature only (one vector, full matrix unrecoverable). "
            "M_* are WB-coupling matrices in the wb_model.json sense."
        ),
        "idx0": c0n.tolist(),
        "idx2": c2n.tolist(),
        "idx2_real_units": c2_real.tolist(),
        "idx2_rownorm": c2_rownorm.tolist(),
        "raw_white_v_10x_counts_per_0p5ms": v_meas.tolist(),
        "pedestal_counts": ped,
        "response_columns_idx0": k0.tolist(),
        "response_columns_idx2": k2.tolist(),
        "white_signatures_unity_wb": {k: v.tolist() for k, v in sigs.items()},
        "M_idx0": analysis["M_idx0_0.5ms"],
        "M_idx2": m_idx2.tolist(),
        "luma_weights_rgb": w.tolist(),
    }
    (out_dir / "ccm_models.json").write_text(json.dumps(ccm_models, indent=1))
    (out_dir / "stage1_analysis.json").write_text(json.dumps(analysis, indent=1))
    print(f"\nwrote {out_dir}/ccm_models.json, {out_dir}/stage1_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
