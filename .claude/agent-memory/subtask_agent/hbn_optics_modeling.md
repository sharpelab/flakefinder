# hBN Optical Contrast Modeling

## Context
Goal: predict R/G contrast curves for hBN flakes on different SiO₂ thicknesses using transfer matrix optics, validated against AFM-calibrated empirical data on "90nm" wafer, then predict for 285nm wafer.

## Script
`scripts/hbn_contrast.py` — transfer matrix method for air/hBN/SiO₂/Si stack, convolved with IMX183 (K5C) Bayer filter response. Plots in R/G contrast space with AFM data overlay.

### CLI flags
- `--n-hbn`: refractive index model — `lee` (ordinary, ~2.13), `zotev-e` (extraordinary, ~1.6), `sqrt3` (√3≈1.73), or any float
- `--max-layers`: max hBN layers (default 200, each 0.333nm)
- `--oxide`: comma-separated oxide thicknesses (default `90,285`)
- `--lamp TEMP_K`: blackbody illumination spectrum
- `--wb R,G,B`: derive illumination from WB gains (1/gain weighting)
- `--fit`: fit ε_r to 90nm empirical data
- `--fit-oxide`: also fit oxide thickness
- `--fit-lamp`: also fit lamp color temperature
- `--output`: save plot to file

### Data dependencies
- Si/SiO₂ refractive index CSVs from `~/sharpelab/graphene_optics/` (Aaron's repo, cloned there this session)
- IMX183 RGB spectral response (hardcoded from Basler docs)
- Empirical cal data from `docs/bn_thickness_calibration.md` (hardcoded, minus 5.6nm outlier)

## Key Findings

### What works
- Transfer matrix physics is correct (matches Aaron's graphene_optics notebook)
- Camera spectral response data found: Sony IMX183CQJ-J, from Basler acA5472-5gc docs
- R/G contrast space is WB-invariant: `(flake-sub)/sub` cancels per-channel WB gains
- **Illumination spectrum does NOT cancel in contrast** — it changes within-channel spectral weighting. I was wrong about this initially.
- Qualitative curve shape in R/G space matches data (parabolic trajectory from thin → thick)

### What doesn't work
- **Absolute contrast magnitudes are off by ~2x** — theory under-predicts contrast at all n, oxide, and lamp values
- 3-parameter fit (ε_r=5.58, t_ox=82nm, T_lamp=5600K) gives RSS=2.59, still bad
- Thin flakes (5-10nm): empirical R≈-0.65, G≈0.35 vs theory R≈-0.35, G≈0.05

### Root cause: high-NA objective
- 50x objective NA=0.75 (from `docs/microscope_reference.md`) → half-angle 49°
- Normal-incidence transfer matrix misses oblique angle contributions
- At 49° incidence, effective optical path is 1/cos(49°) ≈ 1.52× longer
- This increases contrast magnitude — exactly the discrepancy pattern
- **Next step: implement angle-averaged reflectance integrating over the NA cone**

### Refractive index
- Aaron suggested n=√3 (≈1.73) — between published ordinary (2.13) and extraordinary (1.6)
- Aaron says out-of-plane (extraordinary) index matters, not in-plane — contradicts standard normal-incidence analysis but may be correct for high-NA
- Published data: Lee et al. 2019 Sellmeier (ordinary), Zotev et al. 2023 (extraordinary) — both on refractiveindex.info
- Fit to empirical data (without NA correction) gives n≈2.37, ε_r≈5.6 — but this absorbs the missing NA physics
- hBN k=0 across visible (bandgap 5.955eV = 208nm UV)
- Layer thickness: 0.333nm

### Microscope illumination
- DM6M can have halogen (3200K) or LED — Leica spec page confirms
- WB BGR=2.51,1.02,1.41 strongly suggests halogen (big blue boost needed)
- LED5000 MCI is 5700K — wouldn't need B=2.51
- SDK has `PROP_IL_LED_WAVELENGTH` etc but actual values only readable at runtime
- Blackbody lamp model didn't significantly improve fit (lamp+optics ≠ pure blackbody)

### Oxide thickness
- "90nm" wafer may actually be ~82nm based on contrast curve fit
- 285nm wafer is likely ~290nm (robust across all lamp temperature assumptions)
- Substrate color (R/G, B/G pixel ratios) can estimate oxide but requires known lamp spectrum
- WB-corrected substrate ratios: 90nm wafer R/G=0.62, B/G=0.94; 285nm wafer R/G=0.88, B/G=1.00

### Camera sensor
- Sony IMX183CQJ-J confirmed (pixel size, resolution, sensor format all match K5C)
- RGB response at 10nm intervals 400-700nm from Basler acA5472-5gc docs
- Peak responses: B at 455nm, G at 530nm, R at 600nm
- Absolute QE ~69% at green peak (from FLIR EMVA 1288 data on same sensor)

## Status
Script is functional but fit is poor due to missing NA integration. Two paths forward:
1. **Angle-averaged reflectance** — integrate transfer matrix over 0→arcsin(NA) cone, weighted by apodization. This is the physics-correct fix.
2. **Empirical scaling** — fit R/G scale factors on 90nm data, apply to 285nm prediction. Pragmatic but less principled.

Aaron and Zack are actively discussing — pick up from here next session.
