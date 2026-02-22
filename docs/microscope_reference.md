# Microscope Reference

Empirical measurements and reference data for the Sharpe Lab Leica DM6M.

## Objectives

| Pos | Mag | Name | NA | DOF (µm) | Working Dist (µm) | AF Range (µm) |
|-----|-----|------|----|----------|-------------------|----------------|
| 6 | 2.5x | 2.5x | 0.07 | 153 | 15,000 | 500 (auto) |
| 1 | 5x | 5x N PLAN | 0.12 | 50 | 12,700 | 500 (auto) |
| 2 | 10x | 10x | 0.25 | 12 | 11,000 | 500 (auto) |
| 3 | 20x | 20x | 0.40 | 4.3 | 1,900 | 500 (auto) |
| 4 | 50x | 50x | 0.75 | 1.2 | 380 | 127 (auto) |
| 5 | 150x | 150x | 0.90 | 0.7 | 210 | 70 (auto) |

DOF = λ/NA² + pixel/(M·NA) with λ=0.55 µm, 3×3 binning (effective pixel 7.2 µm). 20x and 50x validated empirically.

Autofocus auto range = min(working_distance / 3, 500).

### Autofocus Parameters by Objective

Measured 2026-02-14 on silicon substrate (Elijah's Chip Rejects).

| Mag | Offset from 5x | Range | Z Speed | Exposure |
|-----|----------------|-------|---------|----------|
| 2.5x | -87 µm | auto (500) | 2500 µm/s | 1 ms |
| 5x | — | auto (500) | 2500 µm/s | 1 ms |
| 10x | -3 µm | auto (500) | 1250 µm/s | 1 ms |
| 20x | +11 µm | auto (500) | 1250 µm/s | 2 ms |
| 50x | +13 µm | 100 µm | 500 µm/s | 2 ms |

Parfocal offsets are sample-dependent. 2.5x is a notable outlier (~87 µm below 5x).

**SDK parfocal correction is unreliable.** The Leica SDK adjusts Z on objective swap, but the delta is not a fixed offset — it varies with starting Z. Tested 2.5x→5x: -225 µm from Z=24483, -10.3 from Z=24670. Always supply explicit `--z`.

### Parcentric Offsets (relative to 10x)

Measured 2026-02-21 on SF119 A-H sample (chip 6, rank01_frame_0327_d0). Eyeballed from crosshair-annotated captures at same stage XY. Apply these deltas when switching from 10x to center the same feature.

| Mag | ΔX (µm) | ΔY (µm) |
|-----|---------|---------|
| 2.5x | -25 | 0 |
| 5x | +5 | +20 |
| 10x | 0 | 0 |
| 20x | +30 | +25 |
| 50x | +17 | +30 |

Accuracy ~±5 µm. +X = move stage right, +Y = move stage down (toward higher Y).

**SDK parcentric correction exists but is redundant.** The SDK applies its own XY correction on objective swap (e.g. 2.5x: ΔX=-36, ΔY=+167 from 10x). Using `capture --objective-mag` with explicit `--x/--y` overrides this — the move happens after the swap. Use our measured offsets instead.

### Focus-and-Capture Parameters by Objective

Single-pass Z scan, best frame saved. Validated 2026-02-14 at chip 2 rank 18.

| Mag | Range | Speed | Exposure | Frames | Time |
|-----|-------|-------|----------|--------|------|
| 2.5x | 200 µm | 1000 µm/s | 1 ms | 6 | 0.33s |
| 5x | 100 µm | 500 µm/s | 1 ms | 6 | 0.33s |
| 10x | 50 µm | 250 µm/s | 1 ms | 6 | 0.34s |
| 20x | 30 µm | 50 µm/s | 2 ms | 22 | 1.11s |
| 50x | 30 µm | 25 µm/s | 2 ms | ~25 | ~1.3s |

## Reference Z Positions

Objectives are parfocal — Z focus is similar across magnifications.

### Fixed Landmarks

| Landmark | Z (µm) | X (µm) | Y (µm) | Objective | Date |
|----------|---------|--------|--------|-----------|------|
| Metal sheath at origin | ~24903 | 0 | 0 | 5x | 2026-02-06 |
| Calibration ruler | ~24776 | ~38946 | ~6669 | 5x | 2026-02-06 |

### Sample Z Focus History

| Sample Set | Objective | Z (µm) | Source | Date |
|------------|-----------|--------|--------|------|
| Set 1 | 5x (pos 1) | ~24690 | z_ramp data, focus scan v2 | 2026-02-05 |
| Set 1 | 20x (pos 3) | ~24690 | autofocus runs (af_g01: 24699, af_g01_v6: 24690) | 2026-02-05 |
| Set 2 | 20x (pos 3) | ~24699 | af_20x_chip0_center_slow, chip 0 centroid | 2026-02-06 |
| Set 2 | 2.5x (pos 6) | ~24591 | af_2.5x_chip0, chip 0 centroid | 2026-02-11 |
| Set 2 | 5x (pos 1) | ~24707 | af_5x_chip1_ff, chip 1 centroid | 2026-02-11 |
| Set 2 | 10x (pos 2) | ~24704 | af_10x_chip1_ff, chip 1 centroid | 2026-02-11 |
| Set 2 | 20x (pos 3) | ~24718 | af_20x_chip1_ff, chip 1 centroid | 2026-02-11 |

Z is sample-dependent — different substrates shift focus. The ±250 µm autofocus range covers typical variation. Use as starting `--z` for `autofocus_demo.py`.

## Stage Limits

| Axis | Min (µm) | Max (µm) | Max Speed |
|------|----------|----------|-----------|
| X | 0 | 95,172 | 40 mm/s |
| Y | 0 | 85,103 | 40 mm/s |
| Z | 0 | 25,837 | 5 mm/s |

**Z convention:** +Z = closer to sample (crash risk at high Z). Retract by decreasing Z.

**Default scan rect:** `8000,95000,0,78000` (x_min, x_max, y_min, y_max in µm). Inset from stage edges to avoid metal sheath at origin and edge clipping.

## Camera (Leica K5C)

| Binning | Factor | Frame Size (px) |
|---------|--------|-----------------|
| 0 (1x1) | 1 | 5472 × 3648 |
| 1 (2x2) | 2 | 2736 × 1824 |
| 2 (3x3) | 3 | 1824 × 1216 |

Physical pixel: 2.40 µm. Sample pixel = physical_pixel × binning_factor / magnification.

### Field of View (3x3 binning)

| Objective | Pixel Size (µm) | FOV Width (µm) | FOV Height (µm) |
|-----------|-----------------|-----------------|------------------|
| 2.5x | 2.88 | 5,253 | 3,502 |
| 5x | 1.44 | 2,627 | 1,751 |
| 10x | 0.72 | 1,313 | 876 |
| 20x | 0.36 | 657 | 438 |
| 50x | 0.144 | 263 | 175 |
| 150x | 0.048 | 88 | 58 |

### Default Capture Settings

- Exposure: 1 ms
- White balance (BGR): 2.51, 1.02, 1.41
- Readout time: ~15 ms (rolling shutter)
