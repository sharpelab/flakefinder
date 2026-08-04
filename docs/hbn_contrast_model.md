# hBN Optical Contrast Model

Transfer matrix thin-film optics model for predicting hBN flake contrast in R/G space on different SiO₂ substrates. Two scripts: a CLI tool and an interactive widget.

## Quick Start

```bash
# Interactive widget — best way to explore
uv run python scripts/hbn_contrast_widget.py

# CLI with current best params
uv run python scripts/hbn_contrast.py --oxide 91,294 --na 0.25 --n-hbn 1.91 --r-offset 0.5 --g-offset -0.2
```

## Scripts

### `scripts/hbn_contrast.py` — CLI tool

Transfer matrix for air/hBN/SiO₂/Si stack, convolved with IMX183 (K5C) Bayer filter response. Plots R/G contrast curves with empirical data overlay.

**Flags:**
- `--n-hbn MODEL` — refractive index: `lee` (ordinary ~2.13), `zotev-e` (extraordinary ~1.6), `sqrt3`, or any float
- `--oxide 91,294` — comma-separated oxide thicknesses (nm)
- `--na 0.25` — objective NA for angle averaging (0 = normal incidence)
- `--r-offset 0.5` — additive R offset applied to empirical data
- `--g-offset -0.2` — additive G offset applied to empirical data
- `--lamp TEMP_K` — blackbody illumination spectrum
- `--fit` — fit ε_r to 90nm empirical data (with `--fit-oxide`, `--fit-lamp`)
- `--output FILE` — save plot to file

### `scripts/hbn_contrast_widget.py` — Interactive explorer

Two-panel matplotlib widget (left: ~90nm, right: ~285nm) with 6 sliders:

| Slider | Range | Controls |
|--------|-------|----------|
| n_hBN | 1.4–2.8 | Refractive index |
| t_oxide | 70–120 nm | Left panel oxide |
| t_285 | 250–310 nm | Right panel oxide |
| NA | 0–0.9 | Objective numerical aperture |
| R offset | -1.0–1.5 | Shift empirical R data |
| G offset | -1.0–1.0 | Shift empirical G data |

**Export button** (bottom-right) dumps current slider values to `/tmp/hbn_contrast_params.json`.

Physics params (n, oxide, NA) recompute the theory curve. Offset sliders are instant (just shift the data points).

## Adding Empirical Data Points

Both scripts share calibration data defined in `scripts/hbn_contrast.py`.

### 90nm substrate (AFM-verified)

Edit `_CAL_DATA` — columns are `[thickness_nm, R_contrast, G_contrast]`:

```python
_CAL_DATA = np.array([
    [4.6,  -0.600, 0.286],
    [6.3,  -0.597, 0.219],
    ...
])
```

### 285nm substrate (estimated thickness)

Edit `_CAL_DATA_285` — same format:

```python
_CAL_DATA_285 = np.array([
    [1.0,  -0.29, 0.15],
    [1.5,  -0.46, 0.19],
    ...
])
```

To add points, append rows. Thickness values for 285nm are guesses — use the widget to slide points along the theory curve and find the best-fit thickness. The R/G contrast values are measured as `(flake_pixel - substrate_pixel) / substrate_pixel` per channel.

After editing, relaunch the widget to see the new points.

## Current Best-Fit Parameters (2026-02-27)

```json
{
  "n_hbn": 1.91,
  "t_oxide_90": 91,
  "t_oxide_285": 294,
  "na": 0.25,
  "r_offset": 0.50,
  "g_offset": -0.20
}
```

- **n = 1.91** — between Lee ordinary (2.13) and extraordinary (1.6)
- **Oxide 90nm wafer** — actually ~91nm
- **Oxide 285nm wafer** — actually ~294nm (confirmed independently via substrate B/G color ratio)
- **NA = 0.25** — 10x objective (what chip scans use for flake detection)
- **R/G offsets** — empirical correction for scope-specific systematics. The theory under-predicts R contrast by ~0.5 and over-predicts G by ~0.2. Offsets are applied to the empirical data to align with theory. The offsets are the same for both substrates, confirming they're scope-level (not substrate-level).

## Known Caveats

- **NA mismatch in the fit**: the AFM cal data is 50x (NA≈0.75–0.9) but the
  best-fit params above were fit with NA=0.25. At 50x, NA is a large lever
  (NA 0.25→0.75 shifts theory R by up to −0.9 at the cal thicknesses), so the
  fitted n=1.91 (vs literature ~2.15) partly compensates for the wrong NA and
  the params should not be trusted to extrapolate beyond the cal range —
  notably into the 50–100nm band, where n=1.91 vs n=2.15 differ by up to
  ΔR≈1.1. Refit at the correct NA, or recalibrate at 10x (where NA
  sensitivity is ≤0.08 R), before relying on extrapolated thickness.
- **n=1.91 fails its own thick-end anchors**: residuals of the n=1.91/NA=0.25
  fit against the AFM-verified 90 nm cal set (camera space, model − offset)
  grow rapidly with thickness, exactly where thick-band presets need the
  model. The widget's checked-in defaults (n=2.152, NA=0.25) fit the entire
  set to ≤0.17 in R and ≤0.53 in G:

  | t (nm) | n=1.91, NA=0.25 (this doc's fit) | n=2.152, NA=0.25 (widget defaults) |
  |--------|----------------------------------|-------------------------------------|
  | 4.6  | ΔR −0.08, ΔG −0.09 | ΔR −0.13, ΔG −0.09 |
  | 10.2 | ΔR −0.07, ΔG −0.14 | ΔR −0.08, ΔG +0.01 |
  | 18.0 | ΔR −0.07, ΔG −0.70 | ΔR +0.15, ΔG −0.24 |
  | 26.0 | **ΔR −0.51, ΔG −1.28** | ΔR +0.04, ΔG −0.53 |
  | 46.0 | **ΔR −1.03, ΔG −1.39** | ΔR −0.05, ΔG −0.44 |

  Use n=2.152 for anything at or beyond the thick end of the cal range
  (the `hbn_thick_50_100_90nm` cal table is generated with it — see
  `docs/hbn_thick_50_100_calibration.md`).
- **`_hist_entropy` zeroes out for high-contrast flakes**: the segmentation
  entropy metric histograms per-pixel contrast over a fixed (−1, 1) range.
  Flakes whose every pixel contrast exceeds +1 (e.g. the whole 50–100 nm
  hBN band on 90 nm SiO₂) produce an empty histogram and entropy evaluates
  to exactly 0.0. Entropy gates must be disabled, not tuned, for such
  presets — any nonzero threshold there is accidental behavior.
- **Params drift between tools**: this doc's best-fit, the widget's slider
  defaults, and the CLI defaults are not currently reconciled (e.g. widget
  inits n=2.152 / oxide 90,285; CLI graphite n=2.4−1.0j vs widget
  2.65−1.3j). The same material can produce different curves depending on
  entry point. Reconciliation pending.
- **285nm panel RMS readout**: the widget's 285 panel computes its R/G rms
  against `_CAL_DATA_285` sequential indices (1–60) as if they were nm —
  the displayed number is meaningless for slider tuning.
- The transfer matrix core itself is verified against the independent `tmm`
  package (≤2e-16 across polarizations, oblique incidence, absorbing films),
  and the 10nm spectral grid contributes ≤0.008 contrast error even at
  300nm hBN on 294nm oxide.

## The R Offset Problem

The transfer matrix model consistently under-predicts R-channel contrast by ~0.5 (additive, not multiplicative). The curve *shape* in R/G space matches well, but the absolute R position is off. This persists across all tested n values, oxide thicknesses, NA values, and illumination spectra.

The per-wavelength spectral contrast in the red band (600-700nm) maxes out at ~-0.31, so no spectral reweighting (optics filtering, lamp spectrum) can produce the observed -0.65. The cause is unknown — possibly related to the AFM calibration data being taken on a different microscope, or an unmodeled optical effect.

The pragmatic fix is the additive R offset, which works for both substrates.

## Physics

The model computes reflectance of a 4-medium stack (air/hBN/SiO₂/Si) using the transfer matrix method:

1. Fresnel coefficients at each interface (s and p polarization)
2. Phase accumulation through each layer: β = 2πn·cos(θ)·t/λ
3. 3-interface analytic formula for total reflectance
4. Angle averaging over objective NA cone (Gauss-Legendre quadrature, sin(θ)cos(θ) weighting)
5. Convolution with IMX183 RGB Bayer filter response (10nm intervals, 400-700nm)
6. Per-channel contrast: (V_flake - V_substrate) / V_substrate

### Data sources

- **Si, SiO₂ refractive indices** — CSVs from `~/sharpelab/graphene_optics/` (refractiveindex.info)
- **hBN refractive index** — Lee et al. 2019 Sellmeier (ordinary), Zotev et al. 2023 (extraordinary)
- **Camera spectral response** — Sony IMX183CQJ-J from Basler acA5472-5gc docs
- **90nm AFM calibration** — `docs/bn_thickness_calibration.md`
