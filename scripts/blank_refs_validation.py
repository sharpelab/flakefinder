"""Validate the golden blank-chip references against the rest of the calibration stack.

Cross-checks calibration/blank_refs_20260813.json (scan space) and
blank_refs_raw_20260813.json (identity-CCM raw space) against the measured
5800K CCM (ccm_models.json) and the 2026-08-06 ColorChecker pipeline model
(colorchecker_cal.derive_pipeline), and derives/validates the common-path
correction ε(λ) (colorchecker_cal.derive_blank_ref_tilt).

Sections:
  1. stage-2 gain/exposure invariance claim
  2. cross-space closure: recover WB gains from scan-space presets via CCM
  3. per-objective raw ratios vs pipeline model, without and with ε
  4. scan-space closure per objective via measured CCM + hBN WB
  5. intra-session repeatability (repeated 10x scan-WB conditions)

Findings and interpretation: docs/blank_refs_20260813.md.

Usage:
    uv run python scripts/blank_refs_validation.py [--plot-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colorchecker_cal import (
    _CAL_DIR,
    OBJECTIVE_NA,
    _channel_integrals,
    derive_blank_ref_tilt,
    derive_pipeline,
)
from hbn_contrast import _IMX183_WAVELENGTHS, reflectance

WB_HBN = np.array([1.41, 1.02, 2.51])
# Nominal WB gains (R, G, B) per stage-1 preset; wse2_legacy's nominal
# definition is undocumented — recovered below, not asserted.
PRESET_WB_NOMINAL: dict[str, tuple[float, float, float] | None] = {
    "graphene_thin/hbn": (1.41, 1.02, 2.51),
    "graphene_thick": (1.40, 1.00, 1.70),
    "wse2_monolayer": (1.60, 1.00, 1.20),
    "wse2_legacy": None,
    "unity_raw": None,  # identity-CCM capture, not a scan-space preset
}
OBJECTIVES = ("2.5x", "5x", "10x", "20x", "50x")


def _ratios(rgb: np.ndarray) -> tuple[float, float]:
    return rgb[0] / rgb[1], rgb[2] / rgb[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the golden blank-chip references against the calibration stack"
    )
    parser.add_argument("--plot-dir", type=Path, help="write validation figures to this directory")
    args = parser.parse_args()

    scan = json.loads((_CAL_DIR / "blank_refs_20260813.json").read_text())
    raw = json.loads((_CAL_DIR / "blank_refs_raw_20260813.json").read_text())
    ccm_models = json.loads((_CAL_DIR / "colorchecker_phase2_20260806" / "ccm_models.json").read_text())
    ccm = np.array(ccm_models["idx2_real_units"])
    pedestal = float(ccm_models["pedestal_counts"])

    lamb = _IMX183_WAVELENGTHS
    model = derive_pipeline(lamb)
    eps = derive_blank_ref_tilt(model, lamb)
    r0 = {o: reflectance(lamb, None, 0, 90.0, na=OBJECTIVE_NA[o]) for o in OBJECTIVES}

    def fwd_ccm(wb_rgb: np.ndarray) -> np.ndarray:
        return (wb_rgb + pedestal) @ ccm.T - pedestal

    def inv_ccm(out_rgb: np.ndarray) -> np.ndarray:
        return (out_rgb + pedestal) @ np.linalg.inv(ccm).T - pedestal

    print("=== 1. stage-2 invariance (claim: ratios stable to ~0.005 over 4x gain + 4x exposure)")
    s2 = scan["stage2_invariance"]
    (ka, va), (kb, vb) = s2.items()
    print(f"  {ka}: rg={va['rg']:.4f} bg={va['bg']:.4f}")
    print(f"  {kb}: rg={vb['rg']:.4f} bg={vb['bg']:.4f}")
    print(f"  |d rg|={abs(va['rg'] - vb['rg']):.4f}  |d bg|={abs(va['bg'] - vb['bg']):.4f}")

    print()
    print("=== 2. cross-space closure: WB gains recovered from stage-1 presets via measured CCM")
    raw10 = np.array(raw["10x"]["rgb"])
    for name, v in scan["stage1_wb_presets"].items():
        meas = np.array(v["rgb"])
        if name == "unity_raw":
            # identity-CCM capture: compare directly against the raw file
            drift = meas / raw10
            print(f"  {name:20s} identity-CCM; vs raw 10x per channel: {np.round(drift, 4)} (brightness drift)")
            continue
        gains = inv_ccm(meas) / raw10
        nom = PRESET_WB_NOMINAL.get(name)
        nom_s = f"nominal ({nom[0]:.2f},{nom[1]:.2f},{nom[2]:.2f})" if nom else "nominal undocumented"
        print(f"  {name:20s} recovered ({gains[0]:.3f},{gains[1]:.3f},{gains[2]:.3f})  {nom_s}")

    print()
    print("=== 3. per-objective raw ratios vs ColorChecker pipeline model (x SiO2 90nm at each NA)")
    print(f"  {'obj':5s} {'meas rg':>8s} {'model':>7s} {'model+e':>8s}   {'meas bg':>8s} {'model':>7s} {'model+e':>8s}")
    pred_eps: dict[str, np.ndarray] = {}
    for o in OBJECTIVES:
        chain = model.objectives[o].chain_spd * r0[o]
        v0 = _channel_integrals(chain, lamb)
        v1 = _channel_integrals(chain * eps, lamb)
        pred_eps[o] = v1
        print(
            f"  {o:5s} {raw[o]['rg']:8.4f} {v0[0] / v0[1]:7.4f} {v1[0] / v1[1]:8.4f}"
            f"   {raw[o]['bg']:8.4f} {v0[2] / v0[1]:7.4f} {v1[2] / v1[1]:8.4f}"
        )
    print("  (e fit only at 10x; other rows are predictions with no free parameters)")

    print()
    print("=== 4. scan-space closure per objective: measured raw x hBN WB -> CCM vs stage-3")
    print(f"  {'obj':5s} {'meas rg':>8s} {'pred rg':>8s} {'d':>7s}   {'meas bg':>8s} {'pred bg':>8s} {'d':>7s}")
    for o in OBJECTIVES:
        m = scan["stage3_objectives"][o]
        pred = fwd_ccm(np.array(raw[o]["rgb"]) * WB_HBN)
        prg, pbg = _ratios(pred)
        print(
            f"  {o:5s} {m['rg']:8.4f} {prg:8.4f} {prg - m['rg']:+7.4f}"
            f"   {m['bg']:8.4f} {pbg:8.4f} {pbg - m['bg']:+7.4f}"
        )

    print()
    print("=== 5. intra-session repeatability: repeated 10x scan-WB conditions")
    reps = {
        "stage1 hbn preset": scan["stage1_wb_presets"]["graphene_thin/hbn"],
        "stage2 scan g4/0.25ms": s2[ka],
        "stage2 focusmap g1/1ms": s2[kb],
        "stage3 10x": scan["stage3_objectives"]["10x"],
    }
    for k, v in reps.items():
        print(f"  {k:24s} rg={v['rg']:.4f} bg={v['bg']:.4f}")
    rgs = [v["rg"] for v in reps.values()]
    bgs = [v["bg"] for v in reps.values()]
    print(f"  spread: rg {max(rgs) - min(rgs):.4f}, bg {max(bgs) - min(bgs):.4f}")

    if args.plot_dir:
        _plot(args.plot_dir, raw, model, eps, r0, lamb)


def _plot(plot_dir: Path, raw: dict, model, eps: np.ndarray, r0: dict, lamb: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    nas = [OBJECTIVE_NA[o] for o in OBJECTIVES]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, idx, lbl in ((axes[0], 2, "B/G"), (axes[1], 0, "R/G")):
        meas = [raw[o]["rgb"][idx] / raw[o]["rgb"][1] for o in OBJECTIVES]
        pred0, pred1 = [], []
        for o in OBJECTIVES:
            chain = model.objectives[o].chain_spd * r0[o]
            v0 = _channel_integrals(chain, lamb)
            v1 = _channel_integrals(chain * eps, lamb)
            pred0.append(v0[idx] / v0[1])
            pred1.append(v1[idx] / v1[1])
        ax.plot(nas, meas, "ko-", label="measured (golden raw)")
        ax.plot(nas, pred0, "s--", label="pipeline model")
        ax.plot(nas, pred1, "^--", label="pipeline model x eps")
        ax.set_xlabel("NA")
        ax.set_ylabel(f"raw {lbl}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        for x, o in zip(nas, OBJECTIVES, strict=True):
            ax.annotate(o, (x, ax.get_ylim()[0]), xytext=(0, 3), textcoords="offset points", fontsize=7, ha="center")
    fig.suptitle("Golden blank-chip raw refs vs ColorChecker pipeline model (eps fit at 10x only)")
    fig.tight_layout()
    out = plot_dir / "blank_refs_vs_pipeline_model.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    i540 = np.argmin(abs(lamb - 540))
    ax.plot(lamb, eps / eps[i540], "k-")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("eps (rel. 540 nm)")
    ax.set_title("Common-path correction eps(lambda)\n(lumped: cube RxT, collector, tube lens, lamp-trace error)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = plot_dir / "common_path_correction.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
