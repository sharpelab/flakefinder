"""ColorChecker 2026-08-06 session: K5C pipeline model for the DM6M.

Models the camera pipeline confirmed by the SDK investigation (2026-08-06):
the K5C streams raw Bayer; the host ISP applies WB gains per channel, then
a 3x3 color-correction matrix (CCM), then identity stages for our configs
(gamma 1.0, saturation 100, levels 0/255, sharpening off):

    out = CCM @ diag(wb_requests) @ V_raw
    V_raw_c = integral of bandpass_c(lambda) x lamp(lambda) x T_obj^2(lambda) x R_target(lambda)

Ingredients and provenance:

- lamp SPD: Elijah's sister-scope spectrometer trace, used UNCHANGED.
  Validated 2026-08-06: with the CCM extracted below, this chain predicts
  the Toghrul 10x AFM anchors with n as the only free parameter
  (R rms 0.108 / G rms 0.142 vs 0.47/0.51 without mixing).
- T^2 per objective: 10x anchored on the DataThief'd transmission curve
  (wavelength axis speculative, see load_objective_transmission_10x);
  other objectives are smooth tilts solved against their raw fingerprints.
- CCM: extracted from the measured WB request-coupling matrix W
  (wb_model.json) + the unity-WB white fingerprint.  For a pipeline
  out = M diag(g) V, the coupling is W_cj = M_cj V_j / (M V)_c, so
  M_cj = W_cj out0_c / V_j given the raw white signals V.  W was measured
  at 10x; M is assumed camera-level (firmware 5800K matrix, default).
  SWAPPABLE: replace with the exact matrix once Probe A
  (GetColorMatrixDescription) or a color-patch session delivers it.
- glare floors: measured in OUTPUT space at the black patch; raw-space
  floors are recovered via M^-1.  Diffuse-chart values — specular-chip
  floors pending the empty-substrate session.

The channel-space fingerprints (counts/ms slopes) are direct measurements.
Everything spectral is inference constrained by them; the CCM is inference
constrained by the WB sweep.  Leica confirmed (2026-08-06) that no lamp
SPD exists from the factory.

Run directly for a summary table; --plot for the model figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import fsolve

sys.path.insert(0, str(Path(__file__).parent))
from hbn_contrast import (
    _IMX183_BLUE,
    _IMX183_GREEN,
    _IMX183_RED,
    _IMX183_WAVELENGTHS,
)

_CAL_DIR = Path(__file__).resolve().parents[1] / "calibration"
_SESSION_DIR = _CAL_DIR / "colorchecker_dm6m_20260806"

# Nominal NA per objective.  microscope_description.json only carries
# magnification/name; 10x = 0.25 is established repo-wide, the rest are
# nominal Leica N PLAN values (the widget presets these but leaves the
# slider draggable).
OBJECTIVE_NA: dict[str, float] = {"2.5x": 0.07, "5x": 0.12, "10x": 0.25, "20x": 0.40, "50x": 0.75}

# Selector/derivation order: modern objectives first (10/20/50x cluster),
# then the older outliers.
OBJECTIVE_ORDER = ("10x", "20x", "50x", "5x", "2.5x")

# WB requests used for hBN scanning and the Toghrul AFM anchor captures
# (gains applied per channel upstream of the CCM).
WB_REQUESTS_HBN_SCAN = np.array([1.41, 1.02, 2.51])


class ChannelRGB(NamedTuple):
    red: float
    green: float
    blue: float


class ObjectivePipeline(NamedTuple):
    """Measured + derived pipeline quantities for one objective."""

    name: str
    fingerprint_out: ChannelRGB  # white slopes, counts/ms at unity WB (measured)
    fingerprint_raw: ChannelRGB  # sensor-space white signals, CCM inverted (derived)
    glare_f_out: ChannelRGB | None  # veiling glare floor, output space (measured)
    glare_f_raw: ChannelRGB | None  # veiling glare floor, raw space via CCM^-1 (derived)
    chain_spd: np.ndarray  # lamp x T^2 on the working grid, peak-normalized (derived)
    transmission_sq_rel: np.ndarray  # relative T^2, peak-normalized (derived)


class PipelineModel(NamedTuple):
    lamp_spd: np.ndarray  # Elijah trace on the working grid (measured, sister scope)
    ccm: np.ndarray  # 3x3, rows scaled so ccm @ V_raw_white_10x = fingerprint_out_10x
    ccm_norm: np.ndarray  # diag-normalized copy for display/comparison
    objectives: dict[str, ObjectivePipeline]


def load_elijah_led(target_lamb: np.ndarray) -> np.ndarray:
    """Load Elijah's LED spectrum and resample onto target_lamb (nm).

    Source: calibration/elijah_led_spectrum.csv — 3648-pt spectrometer
    trace, 200–1011 nm.  Two columns (wavelength_nm, intensity, no header)."""
    raw = np.loadtxt(_CAL_DIR / "elijah_led_spectrum.csv", delimiter=",")
    return np.interp(target_lamb, raw[:, 0], raw[:, 1])


def load_objective_transmission_10x(target_lamb: np.ndarray) -> np.ndarray:
    """Load 10x objective transmission and resample onto target_lamb (nm).

    Source: calibration/objective_10x_transmission.csv — DataThief'd from
    a chart whose X-axis annotation didn't survive digitization, so column-0
    values are pixel coordinates spanning [101.34, 297.35].  We remap
    linearly to [400, 700] nm under the assumption that the chart spanned
    the visible band; the right edge looks slightly sharp under that
    assumption (cuts hard near 700 nm) so the chart may have actually been
    400–680 or 400–690.  Treat this axis as speculative until verified.
    """
    raw = np.loadtxt(_CAL_DIR / "objective_10x_transmission.csv", delimiter=",")
    raw = raw[raw[:, 0].argsort()]
    x_min, x_max = float(raw[0, 0]), float(raw[-1, 0])
    lamb_remapped = 400.0 + (raw[:, 0] - x_min) / (x_max - x_min) * 300.0
    return np.interp(target_lamb, lamb_remapped, raw[:, 1])


def _bandpasses(lamb: np.ndarray) -> np.ndarray:
    """Raw IMX183 R/G/B response resampled onto lamb — (3, len(lamb))."""
    return np.vstack([np.interp(lamb, _IMX183_WAVELENGTHS, bp) for bp in (_IMX183_RED, _IMX183_GREEN, _IMX183_BLUE)])


def _channel_integrals(spd: np.ndarray, lamb: np.ndarray) -> np.ndarray:
    """Raw-sensor channel signals (R, G, B) for a full-chain SPD."""
    return np.trapezoid(_bandpasses(lamb) * spd, lamb, axis=1)


def _tilted(base: np.ndarray, lamb: np.ndarray, b: float, c: float) -> np.ndarray:
    x = (lamb - 550.0) / 100.0
    return base * np.exp(b * x + c * x * x)


def _solve_tilt(
    base: np.ndarray,
    lamb: np.ndarray,
    target_gr: float,
    target_gb: float,
    chain: np.ndarray,
) -> np.ndarray:
    """Tilt `base` so raw-channel ratios of (tilted x chain) match targets.

    Returns the tilted base only (chain not multiplied in) so the caller
    can factor lamp and transmission separately."""

    def resid(p: np.ndarray) -> list[float]:
        v = _channel_integrals(_tilted(base, lamb, p[0], p[1]) * chain, lamb)
        return [v[1] / v[0] - target_gr, v[1] / v[2] - target_gb]

    sol, _info, ier, msg = fsolve(resid, [0.0, 0.0], full_output=True)
    if ier != 1:
        raise RuntimeError(f"tilt solve failed: {msg}")
    return _tilted(base, lamb, sol[0], sol[1])


def derive_pipeline(lamb: np.ndarray = _IMX183_WAVELENGTHS) -> PipelineModel:
    """Derive the CCM and per-objective raw-space light models.

    Chain: fix the lamp to Elijah's trace and T^2_10x to the DataThief
    curve; extract the CCM from the measured WB coupling W + 10x unity-WB
    fingerprint; invert each objective's output fingerprint through the
    CCM to raw space; solve each non-10x T^2 as a smooth tilt against its
    raw fingerprint.  Raw glare floors come from output floors via CCM^-1.
    """
    session = json.loads((_SESSION_DIR / "session_results.json").read_text())
    wb_model = json.loads((_SESSION_DIR / "wb_model.json").read_text())
    W = np.array(wb_model["M_rows_RGB_cols_reqRGB"])
    slopes = session["white_fingerprint_slopes_per_ms"]
    glare = session["glare_floor_f_from_black_RGB"]

    lamp = load_elijah_led(lamb)
    t2_10x = load_objective_transmission_10x(lamb) ** 2

    out0 = np.array(slopes["10x"])
    v_white_10x = _channel_integrals(lamp * t2_10x, lamb)
    ccm = W * out0[:, None] / v_white_10x[None, :]
    ccm_inv = np.linalg.inv(ccm)

    objectives: dict[str, ObjectivePipeline] = {}
    for name in OBJECTIVE_ORDER:
        out_fp = np.array(slopes[name])
        raw_fp = ccm_inv @ out_fp
        if name == "10x":
            t2 = t2_10x.copy()
        else:
            t2 = _solve_tilt(np.ones_like(lamb), lamb, raw_fp[1] / raw_fp[0], raw_fp[1] / raw_fp[2], chain=lamp)
        t2 /= t2.max()
        chain = lamp * t2
        if name in glare:
            f_out = np.array(glare[name])
            raw_black = ccm_inv @ (f_out * out_fp)
            f_raw = raw_black / raw_fp
            glare_out, glare_raw = ChannelRGB(*f_out), ChannelRGB(*f_raw)
        else:
            glare_out = glare_raw = None
        objectives[name] = ObjectivePipeline(
            name=name,
            fingerprint_out=ChannelRGB(*out_fp),
            fingerprint_raw=ChannelRGB(*raw_fp),
            glare_f_out=glare_out,
            glare_f_raw=glare_raw,
            chain_spd=chain / chain.max(),
            transmission_sq_rel=t2,
        )
    return PipelineModel(
        lamp_spd=lamp / lamp.max(),
        ccm=ccm,
        ccm_norm=ccm / np.diag(ccm)[:, None],
        objectives=objectives,
    )


def _print_summary(model: PipelineModel) -> None:
    print(f"K5C pipeline model — {_SESSION_DIR.name}")
    print()
    print("Extracted CCM (diag-normalized; swap for exact matrix when probed):")
    for row in model.ccm_norm:
        print("   [{:+.3f}  {:+.3f}  {:+.3f}]".format(*row))
    print()
    header = f"{'obj':>5} | {'out R':>7} {'out G':>7} {'out B':>7} | {'raw R':>7} {'raw G':>7} {'raw B':>7}"
    print(header + " | glare f_out (R/G/B) -> f_raw")
    for op in model.objectives.values():
        fo, fr = op.fingerprint_out, op.fingerprint_raw
        if op.glare_f_out is not None and op.glare_f_raw is not None:
            g = "{:.3f}/{:.3f}/{:.3f} -> {:.3f}/{:.3f}/{:.3f}".format(*op.glare_f_out, *op.glare_f_raw)
        else:
            g = "-"
        print(
            f"{op.name:>5} | {fo.red:7.2f} {fo.green:7.2f} {fo.blue:7.2f}"
            f" | {fr.red:7.2f} {fr.green:7.2f} {fr.blue:7.2f} | {g}"
        )
    print()
    print("out = measured unity-WB white slopes (counts/ms).  raw = CCM^-1 @ out.")
    print("lamp = Elijah sister-scope trace (unchanged); T^2: 10x DataThief anchor,")
    print("others smooth tilts vs raw fingerprints.  Glare: diffuse-chart values.")


def _plot(model: PipelineModel, lamb: np.ndarray, output: str | None) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_chain, ax_t, ax_resp) = plt.subplots(1, 3, figsize=(16, 5))

    for op in model.objectives.values():
        style = "-" if op.name in ("10x", "20x", "50x") else "--"
        ax_chain.plot(lamb, op.chain_spd, style, label=op.name)
        t_label = op.name + (" (DataThief anchor)" if op.name == "10x" else "")
        ax_t.plot(lamb, op.transmission_sq_rel, style, label=t_label)
    ax_chain.plot(lamb, model.lamp_spd, "k:", lw=1.5, label="lamp (Elijah trace)")
    ax_chain.set_title("Raw-space illumination chain (lamp × T²)")
    ax_chain.legend(fontsize=8)
    ax_t.set_title("Relative T² per objective (smooth-only, arb. scale)")
    ax_t.legend(fontsize=8)

    bp = _bandpasses(lamb)
    eff = model.ccm_norm @ bp
    for i, (ch, color) in enumerate(zip("RGB", ("tab:red", "tab:green", "tab:blue"), strict=True)):
        ax_resp.plot(lamb, bp[i] / bp[i].max(), color=color, ls="--", alpha=0.5, label=f"{ch} raw")
        ax_resp.plot(lamb, eff[i] / np.abs(eff[i]).max(), color=color, ls="-", label=f"{ch} × CCM")
    ax_resp.axhline(0, color="k", lw=0.5)
    ax_resp.set_title("Channel response: raw sensor vs CCM-sharpened")
    ax_resp.legend(fontsize=8)

    for ax in (ax_chain, ax_t, ax_resp):
        ax.set_xlabel("wavelength (nm)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=120)
        print(f"saved {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="K5C pipeline model from the ColorChecker 2026-08-06 session")
    parser.add_argument("--plot", action="store_true", help="plot chain spectra, T², channel responses")
    parser.add_argument("--output", help="save plot to file instead of showing")
    args = parser.parse_args()

    lamb = np.arange(400.0, 701.0, 2.0)  # finer grid than the 10 nm camera table, for plots
    model = derive_pipeline(lamb)
    _print_summary(model)
    if args.plot or args.output:
        _plot(model, lamb, args.output)


if __name__ == "__main__":
    main()
