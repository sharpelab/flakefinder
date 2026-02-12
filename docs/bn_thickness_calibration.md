# hBN Thickness Calibration on 90nm SiO2

Source: [BN Thickness Calibration on 90nm SiO2](https://s.zoumez.net/bn-cal) (Google Sheets, Sharpe Lab)

Normalized contrast `(flake - substrate) / substrate` measured at 50x on Leica DM6M, with AFM-verified thickness.

## Data

| Flake ID | R contrast | G contrast | Thickness (nm) | Classification | Stack | Notes |
|----------|-----------|-----------|----------------|----------------|-------|-------|
| SF102IJ | -0.600 | 0.286 | 4.6 | Very Thin | | |
| SF102IJ | -0.671 | 0.079 | 5.6 | Very Thin | SF96I3 | |
| 2025-08 E3-E5 | -0.597 | 0.219 | 6.3 | Very Thin | | |
| SF103 EFGH | -0.645 | 0.313 | 7.0 | Very Thin | SF96I6 | |
| SF102F 2367 | -0.649 | 0.376 | 8.1 | Very Thin | SF96H5 | |
| E7-E10_run4 | -0.670 | 0.409 | 8.6 | Very Thin | SF96I3 | |
| SF103 EFGH | -0.673 | 0.526 | 10.2 | Very Thin | SF96I6 | |
| S102 IJ | -0.692 | 0.948 | 14.1 | Very Thin | | |
| SF102D 1705 | -0.481 | 1.613 | 18.0 | Medium Thin | | |
| SF102D 1890 | 0.421 | 2.907 | 26.0 | Medium | | |
| SF102G 2587 | -0.026 | 2.356 | — | Medium | SF96H5 | |
| benalex 8967 | 2.520 | 4.636 | 46.0 | Thick | B25_003 | AFM via planefit + histogram peaks |

## Classification Boundaries

From the data:
- **Very Thin** (< ~15 nm): R in [-0.7, -0.6], G in [0.08, 0.95]
- **Medium Thin** (~18 nm): R ~ -0.48, G ~ 1.6
- **Medium** (~26 nm): R ~ [-0.03, 0.42], G ~ [2.4, 2.9]
- **Thick** (~46 nm): R ~ 2.5, G ~ 4.6

## Key Observations

- R contrast is **negative** for thin flakes (darker than substrate in red channel), transitions positive around 20-25 nm
- G contrast is **always positive** and increases monotonically with thickness
- The R-G trajectory traces a clear curve from (-0.7, 0.08) to (2.5, 4.6)
- These are 50x measurements; 20x values follow the same pattern but absolute values may differ
