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
