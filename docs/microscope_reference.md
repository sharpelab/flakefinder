# Microscope Reference

Empirical measurements and reference data for the Sharpe Lab Leica DM6M.

## Objectives

| Pos | Mag | Name | Working Dist (µm) | Autofocus Range (µm) |
|-----|-----|------|-------------------|----------------------|
| 6 | 2.5x | 2.5x | 15,000 | 500 (auto) |
| 1 | 5x | 5x N PLAN | 12,700 | 500 (auto) |
| 2 | 10x | 10x | 11,000 | 500 (auto) |
| 3 | 20x | 20x | 1,900 | 500 (auto) |
| 4 | 50x | 50x | 380 | 127 (auto) |
| 5 | 150x | 150x | 210 | 70 (auto) |

Autofocus auto range = min(working_distance / 3, 500).

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

Z is sample-dependent — different substrates shift focus. The ±250 µm autofocus range covers typical variation. Use as starting `--z` for `autofocus_demo.py`.

## Stage Limits

| Axis | Min (µm) | Max (µm) | Max Speed |
|------|----------|----------|-----------|
| X | 0 | 95,172 | 40 mm/s |
| Y | 0 | 85,103 | 40 mm/s |
| Z | 0 | 25,837 | 5 mm/s |

**Z convention:** +Z = closer to sample (crash risk at high Z). Retract by decreasing Z.

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
