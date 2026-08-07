"""Quantify what 12-bit capture buys over 8-bit, at the raw-config operating point.

Consumes existing data only — no hardware. Three sources:

  1. `experiments/raw12_recon/`  (Stage 2 probe, depth-8 + depth-12 frame pair)
     Fixes the 8<->12 mapping and the black-level pedestal.
  2. `calibration/colorchecker_phase2_20260806/stage1/`  (Stage 1, lossless PNG,
     3-5 repeat frames per condition)
     Repeat frames differenced give per-pixel TEMPORAL noise free of scene
     structure and fixed-pattern.  Includes the `unity_idx0` conditions, i.e.
     the raw config (identity CCM + unity WB) measured directly at three
     exposures, so raw-config noise needs no CCM inversion.
  3. A chip scan directory — real substrate levels under the scan config.

The question: at the raw config's operating levels, how much of the measured
noise is 8-bit quantization, and what does removing it buy for thin-flake
contrast and hBN thickness discrimination?

Two subtleties the analysis has to respect:

  * Sub-LSB dither.  At the raw config's short exposures the analog noise is
    well under 1 LSB, so the textbook `var_quant = 1/12` is invalid and
    `sqrt(sigma_meas^2 - 1/12)` can even go imaginary.  Instead the rounding
    response `sigma_measured(sigma_analog)` is simulated and inverted
    numerically (`_rounding_response`).
  * WB gain does not move noise along a single curve.  Empirically the output
    noise of the WB-doubled idx0 conditions scales as neither wb^1 nor wb^2, so
    only the three unity-WB idx0 conditions (0.15 / 0.20 / 0.50 ms) define the
    noise-vs-level curve.  The raw-config substrate levels land inside that
    bracket, so the operating-point number is an INTERPOLATION.

Scope limit, deliberate: the budget covers the RAW config only.  The scan
config (idx2 + scan WB) would need the WB scaling law, which does not close
against the measured WB-doubled conditions, so no scan-config noise number is
produced here — only its measured levels, for context.  The CCM propagation law
`Sigma_out = C Sigma_raw C^T` IS validated against a held-out condition and is
reported as a diagnostic.

Usage:
    uv run python scripts/experiments/analyze_raw12_benefit.py \
        --scan-dir scans/run_20260804_1303/run_20260804_1303/chip_0/scan_10x --plot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colorchecker_cal import OBJECTIVE_NA  # noqa: E402
from hbn_contrast import (  # noqa: E402
    _HBN_LAYER_THICKNESS_NM,
    camera_channel_contrast,
    n_hbn_constant,
)

CHANNELS = ("red", "green", "blue")
_LSB_12BIT_IN_8BIT = 1.0 / 16.0  # 12-bit LSB expressed in 8-bit counts


class RGB(NamedTuple):
    """A per-channel triple in RGB order."""

    red: float
    green: float
    blue: float

    def to_dict(self) -> dict[str, float]:
        return {"red": self.red, "green": self.green, "blue": self.blue}

    @classmethod
    def from_array(cls, a) -> RGB:
        return cls(float(a[0]), float(a[1]), float(a[2]))

    def as_array(self) -> np.ndarray:
        return np.array(self, dtype=float)


class ProbeRecon(NamedTuple):
    """8<->12 bit mapping and 12-bit validity checks from the Stage 2 probe."""

    slope: RGB  # 12-bit counts per 8-bit count
    offset: RGB  # 12-bit counts
    pedestal_counts: RGB  # offset / slope, in 8-bit counts
    nibble_max_dev: RGB  # max |p(nibble) - 1/16|, uniform => genuine 12-bit
    missing_codes: RGB  # unoccupied codes inside the p5..p95 range
    unique_8bit: RGB
    unique_12bit: RGB

    def to_dict(self) -> dict:
        return {k: getattr(self, k).to_dict() for k in self._fields}


class NoisePoint(NamedTuple):
    """Temporal noise for one capture-series condition."""

    condition: str
    colour_temperature: int
    white_balance: RGB
    exposure_ms: float
    level: RGB  # 8-bit counts, ROI mean
    sigma_measured: RGB  # 8-bit counts, includes quantization
    sigma_analog: RGB  # rounding response inverted

    @property
    def eight_bit_penalty(self) -> RGB:
        """sigma_measured / sigma_analog — what 8-bit rounding costs here.

        sigma_measured IS the 8-bit performance; sigma_analog is what 12-bit
        would deliver (its own LSB being 16x finer is negligible).
        """
        return RGB(*(self.sigma_measured[c] / self.sigma_analog[c] for c in range(3)))

    def to_dict(self) -> dict:
        d = {"condition": self.condition, "colour_temperature": self.colour_temperature,
             "exposure_ms": self.exposure_ms}  # fmt: skip
        for k in ("white_balance", "level", "sigma_measured", "sigma_analog"):
            d[k] = getattr(self, k).to_dict()
        d["eight_bit_penalty"] = self.eight_bit_penalty.to_dict()
        return d


class BudgetRow(NamedTuple):
    """Noise budget and derived precision for one config x depth x channel."""

    config: str
    depth_bits: int
    channel: str
    level: float  # 8-bit counts
    sigma_analog: float
    sigma_quant: float
    sigma_total: float
    contrast_precision_px: float  # delta-C, single pixel
    contrast_precision_flake: float  # delta-C, averaged over flake_px pixels

    def to_dict(self) -> dict:
        return dict(self._asdict())


# ---------------------------------------------------------------------------
# Stage 2 probe: 8<->12 mapping, pedestal, 12-bit validity
# ---------------------------------------------------------------------------


def load_probe(probe_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the depth-8/depth-12 buffer pair as (H, W, 3) RGB arrays."""
    meta = json.loads((probe_dir / "recon.json").read_text())
    w, h = meta["frame_px"]
    buf8 = np.load(probe_dir / "depth8_raw.npy")
    buf12 = np.load(probe_dir / "depth12_raw.npy")
    # SDK delivers BGR; [..., ::-1] -> RGB (project convention)
    img8 = buf8.reshape(h, w, 3)[..., ::-1].astype(np.float64)
    img12 = buf12.view(np.uint16).reshape(h, w, 3)[..., ::-1]
    return img8, img12, meta


def probe_recon(img8: np.ndarray, img12: np.ndarray, block: int = 32) -> ProbeRecon:
    """Fit the 8->12 affine map and run the 12-bit validity checks.

    The affine fit uses block means so that per-pixel noise in BOTH frames is
    averaged away; regressing raw pixels would attenuate the slope
    (errors-in-variables) since the two frames are independent acquisitions.
    """
    h, w, _ = img8.shape
    hb, wb = (h // block) * block, (w // block) * block

    def block_mean(a: np.ndarray) -> np.ndarray:
        return a[:hb, :wb].reshape(hb // block, block, wb // block, block).mean(axis=(1, 3)).ravel()

    slope, offset, ped, dev, missing, u8n, u12n = [], [], [], [], [], [], []
    for c in range(3):
        x, y = block_mean(img8[..., c]), block_mean(img12[..., c].astype(np.float64))
        m, b = np.polyfit(x, y, 1)
        slope.append(m)
        offset.append(b)
        ped.append(b / m)

        ch12 = img12[..., c]
        hist = np.bincount((ch12 & 0xF).ravel(), minlength=16) / ch12.size
        dev.append(float(np.abs(hist - 1 / 16).max()))

        lo, hi = np.percentile(ch12, [5, 95]).astype(int)
        occupied = np.bincount(ch12.ravel(), minlength=hi + 1)[lo : hi + 1]
        missing.append(int((occupied == 0).sum()))

        u8n.append(len(np.unique(img8[..., c])))
        u12n.append(len(np.unique(ch12)))

    return ProbeRecon(
        slope=RGB.from_array(slope),
        offset=RGB.from_array(offset),
        pedestal_counts=RGB.from_array(ped),
        nibble_max_dev=RGB.from_array(dev),
        missing_codes=RGB.from_array(missing),
        unique_8bit=RGB.from_array(u8n),
        unique_12bit=RGB.from_array(u12n),
    )


# ---------------------------------------------------------------------------
# Rounding response: recover analog noise from noise measured after rounding
# ---------------------------------------------------------------------------


def _rounding_response(sigma_analog: np.ndarray, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """Simulate sigma_measured for each analog sigma, in LSB units.

    Two independent noisy reads of the same level are rounded and differenced,
    matching how `measure_noise_series` estimates sigma from repeat frames.
    Sub-LSB level offsets are taken uniform over [0, 1): across an ROI spanning
    many LSBs the fractional parts are effectively uniform.
    """
    rng = np.random.default_rng(seed)
    frac = rng.uniform(0.0, 1.0, n)
    out = np.empty_like(sigma_analog)
    for i, s in enumerate(sigma_analog):
        a = np.rint(frac + rng.normal(0.0, s, n))
        b = np.rint(frac + rng.normal(0.0, s, n))
        out[i] = np.sqrt((a - b).var() / 2.0)
    return out


def build_rounding_inverse(max_sigma: float = 4.0, steps: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma_measured_grid, sigma_analog_grid) for interpolation."""
    analog = np.linspace(0.0, max_sigma, steps)
    measured = _rounding_response(analog)
    # enforce monotonicity so np.interp is well-defined
    measured = np.maximum.accumulate(measured)
    return measured, analog


def invert_rounding(sigma_measured, measured_grid: np.ndarray, analog_grid: np.ndarray):
    """Map measured (post-rounding) sigma back to analog sigma, in LSB units."""
    return np.interp(sigma_measured, measured_grid, analog_grid)


# ---------------------------------------------------------------------------
# Stage 1 repeat frames: temporal noise per condition
# ---------------------------------------------------------------------------


def _imread_rgb(path: str) -> np.ndarray:
    """Read an image as float64 RGB, raising if the decode failed."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"could not read image: {path}")
    return img[:, :, ::-1].astype(np.float64)


def _repeat_frames(stage1_dir: Path, condition: str) -> list[np.ndarray]:
    paths = sorted(glob.glob(str(stage1_dir / f"{condition}_f*.png")))
    return [_imread_rgb(p) for p in paths]


def _roi_level_and_cov(frames: list[np.ndarray], roi_half: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """ROI mean level and the 3x3 temporal noise covariance from repeat frames.

    Differencing cancels the scene and fixed-pattern noise; var(diff)/2 is the
    per-frame temporal covariance (still including quantization).
    """
    h, w, _ = frames[0].shape
    dy, dx = roi_half
    roi = (slice(h // 2 - dy, h // 2 + dy), slice(w // 2 - dx, w // 2 + dx))
    level = np.mean([f[roi].reshape(-1, 3).mean(axis=0) for f in frames], axis=0)
    diffs = [(frames[i + 1][roi] - frames[i][roi]).reshape(-1, 3) for i in range(len(frames) - 1)]
    cov = np.mean([np.cov(d.T) for d in diffs], axis=0) / 2.0
    return level, cov


class PropagationCheck(NamedTuple):
    """Held-out validation of `Sigma_out = C Sigma_raw C^T`."""

    source_condition: str
    target_condition: str
    predicted_sigma: RGB
    measured_sigma: RGB
    ratio: RGB

    def to_dict(self) -> dict:
        return {
            "source_condition": self.source_condition,
            "target_condition": self.target_condition,
            "predicted_sigma": self.predicted_sigma.to_dict(),
            "measured_sigma": self.measured_sigma.to_dict(),
            "ratio": self.ratio.to_dict(),
        }


def validate_ccm_propagation(
    stage1_dir: Path,
    ccm: np.ndarray,
    source: str = "unity_idx0",
    target: str = "base_idx2",
    roi_half: tuple[int, int] = (300, 400),
) -> PropagationCheck:
    """Predict the CCM-mixed condition's noise from the identity-CCM one.

    Both conditions are unity WB at the same illumination, so the only
    difference is the colour matrix.  A close match confirms the CCM acts as a
    plain linear map on the noise, and quantifies what the CCM costs in noise.
    Quantization is stripped before the map and re-added after.
    """
    quant = np.eye(3) / 12.0
    _, cov_src = _roi_level_and_cov(_repeat_frames(stage1_dir, source), roi_half)
    _, cov_tgt = _roi_level_and_cov(_repeat_frames(stage1_dir, target), roi_half)
    predicted = ccm @ (cov_src - quant) @ ccm.T + quant
    pred_sigma = np.sqrt(np.abs(np.diag(predicted)))
    meas_sigma = np.sqrt(np.abs(np.diag(cov_tgt)))
    return PropagationCheck(
        source_condition=source,
        target_condition=target,
        predicted_sigma=RGB.from_array(pred_sigma),
        measured_sigma=RGB.from_array(meas_sigma),
        ratio=RGB.from_array(pred_sigma / meas_sigma),
    )


def _condition_settings(spec_path: Path) -> dict[str, dict]:
    """Per-condition capture settings, defaults merged with overrides."""
    spec = json.loads(spec_path.read_text())
    defaults = spec["defaults"]
    out = {}
    for cond in spec["conditions"]:
        merged = dict(defaults)
        merged.update({k: v for k, v in cond.items() if k != "name"})
        out[cond["name"]] = merged
    return out


def measure_noise_series(
    stage1_dir: Path,
    spec_path: Path,
    measured_grid: np.ndarray,
    analog_grid: np.ndarray,
    roi_half: tuple[int, int] = (300, 400),
) -> list[NoisePoint]:
    """Temporal noise per condition from differenced repeat frames.

    Differencing repeat frames cancels the scene and any fixed-pattern
    component, leaving only temporal noise.  Conditions with a clipped channel
    (level at 0 or 255) are dropped — their noise is not meaningful.
    """
    settings = _condition_settings(spec_path)
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(glob.glob(str(stage1_dir / "*.png"))):
        m = re.match(r"(.+)_f(\d+)\.png$", os.path.basename(path))
        if m:
            groups[m.group(1)].append(path)

    points: list[NoisePoint] = []
    for name in sorted(groups):
        paths = groups[name]
        if len(paths) < 2:
            continue
        frames = [_imread_rgb(p) for p in paths]
        level, cov = _roi_level_and_cov(frames, roi_half)
        if level.min() <= 0.5 or level.max() >= 254.5:
            continue
        sigma_meas = np.sqrt(np.diag(cov))
        sigma_analog = invert_rounding(sigma_meas, measured_grid, analog_grid)

        cfg = settings.get(name, {})
        wbd = cfg.get("white_balance", {"r": 1.0, "g": 1.0, "b": 1.0})
        points.append(
            NoisePoint(
                condition=name,
                colour_temperature=int(cfg.get("colour_temperature", 2)),
                white_balance=RGB(wbd["r"], wbd["g"], wbd["b"]),
                exposure_ms=float(cfg.get("exposure_ms", 0.5)),
                level=RGB.from_array(level),
                sigma_measured=RGB.from_array(sigma_meas),
                sigma_analog=RGB.from_array(sigma_analog),
            )
        )
    return points


def interpolate_sigma_analog(points: list[NoisePoint], target: RGB) -> tuple[RGB, list[str]]:
    """Analog sigma at the target levels, from the unity-WB idx0 conditions.

    Only unity-WB identity-CCM conditions are used: WB-doubled conditions do
    not fall on the same noise-vs-level curve (the output noise scales as
    neither wb^1 nor wb^2), and CCM-mixed conditions are in a different space.
    Fits var = a*level + b per channel, which is exact for photon + read noise.
    """
    ref = [p for p in points if p.colour_temperature == 0 and p.white_balance == RGB(1.0, 1.0, 1.0)]
    if len(ref) < 2:
        raise RuntimeError("need >=2 unity-WB idx0 conditions to interpolate noise")

    sigmas = []
    for c in range(len(CHANNELS)):
        levels = np.array([p.level[c] for p in ref])
        variances = np.array([p.sigma_analog[c] ** 2 for p in ref])
        a, b = np.polyfit(levels, variances, 1)
        sigmas.append(float(np.sqrt(max(a * target[c] + b, 0.0))))
    return RGB.from_array(sigmas), [p.condition for p in ref]


# ---------------------------------------------------------------------------
# Anchor: real substrate levels, and the raw-config levels they imply
# ---------------------------------------------------------------------------


def scan_substrate_levels(scan_dir: Path, n_frames: int = 30, seed: int = 0) -> tuple[RGB, dict]:
    """Median substrate level per channel across a sample of scan frames.

    The median over the frame is substrate-dominated; flakes and chip edges are
    a small minority of pixels.
    """
    meta = json.loads((scan_dir / "scan_meta.json").read_text())
    paths = sorted(glob.glob(str(scan_dir / "frame_*.jpg")))
    rng = np.random.default_rng(seed)
    pick = [paths[i] for i in rng.choice(len(paths), size=min(n_frames, len(paths)), replace=False)]

    medians = []
    for p in pick:
        img = _imread_rgb(p)
        medians.append(np.median(img[::4, ::4].reshape(-1, 3), axis=0))
    return RGB.from_array(np.median(medians, axis=0)), meta["camera"]


def raw_config_levels(out_level: RGB, white_balance: RGB, ccm: np.ndarray, pedestal: float) -> RGB:
    """Invert `out = CCM @ diag(wb) @ V_raw - pedestal` to the raw-config output.

    Under the raw config the CCM is identity and WB is unity, so its output is
    just `V_raw - pedestal`.
    """
    v_raw = np.linalg.inv(np.diag(white_balance.as_array())) @ np.linalg.inv(ccm) @ (out_level.as_array() + pedestal)
    return RGB.from_array(v_raw - pedestal)


# ---------------------------------------------------------------------------
# Budget and thickness translation
# ---------------------------------------------------------------------------


def build_budget(config: str, level: RGB, sigma_analog: RGB, flake_px: int) -> list[BudgetRow]:
    """Noise budget for one config at 8- and 12-bit.

    The delivered noise is the analog noise put through the rounding response
    at that depth's LSB — NOT quadrature with LSB/sqrt(12), which only holds
    once the analog noise well exceeds an LSB.  At the raw config's levels the
    analog noise is ~0.3 LSB, squarely in the regime where quadrature fails.
    `sigma_quant` is reported as the implied quadrature residual so the budget
    still decomposes additively for plotting.
    """
    rows: list[BudgetRow] = []
    for depth, lsb in ((8, 1.0), (12, _LSB_12BIT_IN_8BIT)):
        for c, ch in enumerate(CHANNELS):
            s_a = sigma_analog[c]
            # rounding response operates in LSB units, so scale in and out
            s_t = float(_rounding_response(np.array([s_a / lsb]))[0] * lsb)
            s_q = float(np.sqrt(max(s_t**2 - s_a**2, 0.0)))
            rows.append(
                BudgetRow(
                    config=config,
                    depth_bits=depth,
                    channel=ch,
                    level=level[c],
                    sigma_analog=s_a,
                    sigma_quant=s_q,
                    sigma_total=s_t,
                    contrast_precision_px=s_t / level[c],
                    contrast_precision_flake=s_t / (level[c] * np.sqrt(flake_px)),
                )
            )
    return rows


class ThicknessSensitivity(NamedTuple):
    """|dC/dt| per channel versus hBN thickness, at one oxide thickness."""

    oxide_nm: float
    thickness_nm: list[float]
    dc_dt: dict[str, list[float]]  # per channel, contrast per nm

    def to_dict(self) -> dict:
        return {"oxide_nm": self.oxide_nm, "thickness_nm": self.thickness_nm, "dc_dt": self.dc_dt}


def thickness_sensitivity(oxide_nm: float, na: float, max_layers: int = 60) -> ThicknessSensitivity:
    """Contrast sensitivity |dC/dt| for hBN on SiO2/Si, per camera channel.

    `camera_channel_contrast` integrates reflectance against the IMX183 channel
    responses, i.e. it predicts RAW (unmixed) camera contrast — which is what
    the raw config measures directly.
    """
    n_fn = n_hbn_constant(np.sqrt(3.0))
    layers = np.arange(0, max_layers + 1)
    thickness = layers * _HBN_LAYER_THICKNESS_NM
    contrast = np.array([camera_channel_contrast(n_fn, int(nl), oxide_nm, None, na=na) for nl in layers])
    grad = np.abs(np.gradient(contrast, thickness, axis=0))
    return ThicknessSensitivity(
        oxide_nm=oxide_nm,
        thickness_nm=[float(t) for t in thickness],
        dc_dt={ch: [float(v) for v in grad[:, c]] for c, ch in enumerate(CHANNELS)},
    )


def thickness_precision(rows: list[BudgetRow], sens: ThicknessSensitivity, at_nm: float) -> dict[str, dict[str, float]]:
    """delta-t (nm) implied by each row's contrast precision, at one thickness."""
    idx = int(np.argmin(np.abs(np.array(sens.thickness_nm) - at_nm)))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        slope = sens.dc_dt[row.channel][idx]
        key = f"{row.config}_{row.depth_bits}bit"
        out.setdefault(key, {})[row.channel] = (
            float(row.contrast_precision_flake / slope) if slope > 0 else float("inf")
        )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_rgb(triple: RGB, width: int = 6, prec: int = 3) -> str:
    """Render an RGB triple as 'R<v> G<v> B<v>' for report tables."""
    return " ".join(f"{k}{triple[i]:{width}.{prec}f}" for i, k in enumerate("RGB"))


def print_report(
    recon: ProbeRecon,
    points: list[NoisePoint],
    check: PropagationCheck,
    ref_conditions: list[str],
    scan_level: RGB,
    raw_level: RGB,
    budget: list[BudgetRow],
    delta_t: dict,
    flake_px: int,
) -> None:
    print("=" * 78)
    print("STAGE 2 PROBE — 8<->12 bit mapping")
    print("=" * 78)
    header = f"{'channel':8s} {'slope':>8s} {'offset':>8s} {'pedestal':>9s}"
    print(f"{header} {'nibble dev':>11s} {'missing':>8s} {'uniq 8/12':>12s}")
    for c, ch in enumerate(CHANNELS):
        print(
            f"{ch:8s} {recon.slope[c]:8.4f} {recon.offset[c]:8.3f} {recon.pedestal_counts[c]:9.3f}"
            f" {recon.nibble_max_dev[c]:11.5f} {int(recon.missing_codes[c]):8d}"
            f" {int(recon.unique_8bit[c]):5d}/{int(recon.unique_12bit[c]):<6d}"
        )
    print("  slope ~16 and offset/slope ~1 count => 8-bit = (12-bit - pedestal)/16")
    print("  nibble dev ~0 and 0 missing codes  => genuine 12-bit, not 8-bit shifted")

    print()
    print("=" * 78)
    print("STAGE 1 REPEAT FRAMES — temporal noise (8-bit counts)")
    print("=" * 78)
    header = f"{'condition':22s} {'ccm':>4s} {'ms':>5s} {'level R/G/B':>20s}"
    print(f"{header} {'sigma_meas':>18s} {'sigma_analog':>18s} {'8-bit pen':>18s}")
    for p in points:
        lv = " ".join(f"{p.level[c]:6.1f}" for c in range(3))
        sm = " ".join(f"{p.sigma_measured[c]:5.3f}" for c in range(3))
        sa = " ".join(f"{p.sigma_analog[c]:5.3f}" for c in range(3))
        pen = " ".join(f"{p.eight_bit_penalty[c]:5.2f}" for c in range(3))
        lead = f"{p.condition:22s} {p.colour_temperature:4d} {p.exposure_ms:5.2f}"
        print(f"{lead} {lv:>20s} {sm:>18s} {sa:>18s} {pen:>18s}")
    print("  ccm 0 = identity (raw config), 2 = 5800K (legacy scan space)")

    print()
    print("=" * 78)
    print("CCM NOISE PROPAGATION — held-out validation")
    print("=" * 78)
    print(f"  predict {check.target_condition} from {check.source_condition}: Sigma_out = C Sigma_raw C^T")
    print(f"    predicted sigma: {_fmt_rgb(check.predicted_sigma)}")
    print(f"    measured  sigma: {_fmt_rgb(check.measured_sigma)}")
    print(f"    ratio          : {_fmt_rgb(check.ratio)}")
    print("  => the CCM acts as a plain linear map on noise; it INFLATES it.")

    print()
    print("=" * 78)
    print("OPERATING POINT")
    print("=" * 78)
    print(f"  scan config substrate (idx2, scan WB, measured): {_fmt_rgb(scan_level, prec=2)}")
    print(f"  raw config substrate  (idx0, unity WB, implied): {_fmt_rgb(raw_level, prec=2)}")
    print(f"  noise interpolated over unity-WB idx0 conditions: {', '.join(ref_conditions)}")
    print("  (raw levels land inside that bracket => interpolation, not extrapolation)")

    print()
    print("=" * 78)
    print(f"NOISE BUDGET (8-bit counts; delta-C over a {flake_px}-pixel flake)")
    print("=" * 78)
    header = f"{'config':10s} {'bits':>5s} {'chan':6s} {'level':>7s} {'analog':>8s}"
    print(f"{header} {'quant':>7s} {'total':>7s} {'dC/px':>9s} {'dC/flake':>9s}")
    for row in budget:
        print(
            f"{row.config:10s} {row.depth_bits:5d} {row.channel:6s} {row.level:7.2f}"
            f" {row.sigma_analog:8.4f} {row.sigma_quant:7.4f} {row.sigma_total:7.4f}"
            f" {row.contrast_precision_px:9.5f} {row.contrast_precision_flake:9.5f}"
        )

    print()
    print("  8-bit penalty (sigma_total ratio 8bit/12bit):")
    by_key = {(r.config, r.depth_bits, r.channel): r for r in budget}
    for config in dict.fromkeys(r.config for r in budget):
        parts = []
        for ch in CHANNELS:
            r8, r12 = by_key[(config, 8, ch)], by_key[(config, 12, ch)]
            parts.append(f"{ch[0].upper()} {r8.sigma_total / r12.sigma_total:5.2f}x")
        print(f"    {config:10s} " + "  ".join(parts))

    print()
    print("  LIMITATION: no scan-config (idx2 + scan WB) noise number is produced.")
    print("    WB gain does not propagate as a digital multiplier (predicting")
    print("    wbr2_idx0 from unity_idx0 overshoots by ~22%), so the scan config's")
    print("    noise cannot be derived from these conditions. One capture-series")
    print("    condition at idx2 with the scan WB would close that gap.")

    print()
    print("=" * 78)
    print(f"hBN THICKNESS PRECISION (nm, 1-sigma, over a {flake_px}-pixel flake)")
    print("=" * 78)
    for oxide, at_nm_map in delta_t.items():
        for at_nm, per_config in at_nm_map.items():
            print(f"  oxide {oxide} nm, at t = {at_nm} nm:")
            for key, per_ch in per_config.items():
                vals = "  ".join(f"{ch[0].upper()} {per_ch[ch]:7.3f}" for ch in CHANNELS)
                print(f"    {key:22s} {vals}")


def make_plot(budget: list[BudgetRow], sens_list: list[ThicknessSensitivity], out_path: Path, flake_px: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels, analog, quant = [], [], []
    for row in budget:
        labels.append(f"{row.config}\n{row.depth_bits}b {row.channel[0].upper()}")
        analog.append(row.sigma_analog)
        quant.append(row.sigma_total - row.sigma_analog)
    x = np.arange(len(labels))
    ax1.bar(x, analog, color="#4878a8", label="analog noise")
    ax1.bar(x, quant, bottom=analog, color="#c44e52", label="quantization (added)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7, rotation=90)
    ax1.set_ylabel("sigma (8-bit counts)")
    ax1.set_title("Noise budget at the substrate operating point")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    by_key = {(r.config, r.depth_bits, r.channel): r for r in budget}
    for sens in sens_list:
        for depth, style in ((8, "-"), (12, "--")):
            row = by_key[("raw", depth, "green")]
            slopes = np.array(sens.dc_dt["green"])
            with np.errstate(divide="ignore"):
                dt = row.contrast_precision_flake / slopes
            ax2.plot(
                sens.thickness_nm,
                dt,
                style,
                label=f"{sens.oxide_nm:.0f} nm oxide, {depth}-bit",
            )
    ax2.set_yscale("log")
    ax2.set_xlabel("hBN thickness (nm)")
    ax2.set_ylabel(f"thickness precision, 1-sigma (nm) over {flake_px} px")
    ax2.set_title("Raw config, green channel")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nWrote {out_path}")


def main() -> int:
    repo = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, default=repo / "experiments/raw12_recon")
    parser.add_argument("--stage1", type=Path, default=repo / "calibration/colorchecker_phase2_20260806/stage1")
    parser.add_argument("--spec", type=Path, default=repo / "calibration/colorchecker_phase2/stage1_ccm_wb_spec.json")
    parser.add_argument("--ccm", type=Path, default=repo / "calibration/colorchecker_phase2_20260806/ccm_models.json")
    parser.add_argument("--scan-dir", type=Path, required=True, help="Chip scan dir (frame_*.jpg + scan_meta.json)")
    parser.add_argument("--objective", default="10x", help="Objective key for NA lookup (default: 10x)")
    parser.add_argument("--oxide", default="90,285", help="Oxide thicknesses in nm (default: 90,285)")
    parser.add_argument("--at-nm", default="1,5,20", help="hBN thicknesses to report precision at (default: 1,5,20)")
    parser.add_argument("--flake-px", type=int, default=25, help="Pixels averaged over a flake (default: 25)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="JSON output (default: <probe>/benefit.json)")
    parser.add_argument("--plot", action="store_true", help="Write a summary figure next to the JSON")
    args = parser.parse_args()

    out_path = args.output or (args.probe / "benefit.json")
    oxides = [float(x) for x in args.oxide.split(",")]
    at_nms = [float(x) for x in args.at_nm.split(",")]
    na = OBJECTIVE_NA[args.objective]

    img8, img12, _ = load_probe(args.probe)
    recon = probe_recon(img8, img12)

    measured_grid, analog_grid = build_rounding_inverse()
    points = measure_noise_series(args.stage1, args.spec, measured_grid, analog_grid)

    ccm_models = json.loads(args.ccm.read_text())
    ccm = np.array(ccm_models["idx2_real_units"])
    pedestal = ccm_models["pedestal_counts"]

    scan_level, cam_meta = scan_substrate_levels(args.scan_dir)
    wb_bgr = cam_meta["white_balance_bgr"]
    scan_wb = RGB(wb_bgr[2], wb_bgr[1], wb_bgr[0])
    raw_level = raw_config_levels(scan_level, scan_wb, ccm, pedestal)

    raw_sigma, ref_conditions = interpolate_sigma_analog(points, raw_level)
    budget = build_budget("raw", raw_level, raw_sigma, args.flake_px)
    check = validate_ccm_propagation(args.stage1, ccm)

    sens_list = [thickness_sensitivity(ox, na) for ox in oxides]
    delta_t = {
        f"{sens.oxide_nm:.0f}": {f"{at:.0f}": thickness_precision(budget, sens, at) for at in at_nms}
        for sens in sens_list
    }

    print_report(recon, points, check, ref_conditions, scan_level, raw_level, budget, delta_t, args.flake_px)

    payload = {
        "probe_recon": recon.to_dict(),
        "noise_series": [p.to_dict() for p in points],
        "ccm_propagation_check": check.to_dict(),
        "noise_reference_conditions": ref_conditions,
        "operating_point": {
            "scan_config_level": scan_level.to_dict(),
            "raw_config_level": raw_level.to_dict(),
            "scan_white_balance": scan_wb.to_dict(),
            "exposure_ms": cam_meta["exposure_s"] * 1000.0,
            "gain": cam_meta["gain"],
            "objective": args.objective,
            "na": na,
        },
        "budget": [r.to_dict() for r in budget],
        "thickness_precision_nm": delta_t,
        "thickness_sensitivity": [s.to_dict() for s in sens_list],
        "flake_px": args.flake_px,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")

    if args.plot:
        make_plot(budget, sens_list, out_path.with_suffix(".png"), args.flake_px)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
