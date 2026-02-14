# Objective-Aware Autofocus Defaults

## Problem

`continuous_autofocus()` uses the same z_speed and z_range defaults regardless of objective. At 50x (DOF ~1.2 µm), the default 1250 µm/s gives ~20 µm between samples — 16x the DOF. Fine/super-fine fixed defaults also break down at high magnification.

## DOF Calculation

Using DOF = λ/NA² + pixel/(M·NA) with λ=0.55 µm and 3×3 binning (effective pixel 7.2 µm):

| Obj | Pos | NA   | DOF µm | WD µm |
|-----|-----|------|--------|-------|
| 2.5x | 6 | 0.07 | 153  | 15000 |
| 5x   | 1 | 0.12 | 50   | 12700 |
| 10x  | 2 | 0.25 | 12   | 11000 |
| 20x  | 3 | 0.40 | 4.3  | 1900  |
| 50x  | 4 | 0.75 | 1.2  | 380   |
| 150x | 5 | 0.90 | 0.7  | 210   |

20x and 50x DOFs validated against empirical data.

## Sampling constraint

At 16ms polling interval, sample spacing = speed × 0.016s. Target: ≥3 samples/DOF for high-mag coarse pass.

## Current defaults vs proposed

### Current behavior
- z_range: `min(working_distance/3, 500)` — objective-aware but coarse
- z_speed: whatever caller passes (revisit hardcodes 1250)
- fine: fixed 50 µm range, 0.25× coarse speed
- super-fine: fixed 10 µm range, 20 µm/s absolute speed
- settle: 0.2s everywhere

### Problems at high magnification

| Issue | 50x | 150x |
|-------|-----|------|
| Fine range = coarse range | 50 = 50 µm | 50 > 25 µm (!) |
| SF faster than fine | 20 vs 6.25 µm/s | 20 vs 2.5 µm/s |
| SF coarser than fine | 0.32 vs 0.1 µm/sample | 0.32 vs 0.04 µm/sample |

The pass hierarchy (coarse → fine → super-fine) breaks down at 50x+.

### Proposed per-objective defaults

| Obj | DOF | Coarse |  | Fine |  | Super-fine |  | Settle |
|-----|-----|--------|-------|------|-------|------------|-------|--------|
|     |     | Speed  | Range | Speed| Range | Speed      | Range |        |
| 2.5x | 153 | 2500 | 500 | 625 | 50 | 20 | 10 | 0.2 |
| 5x   | 50  | 1250 | 500 | 312 | 50 | 20 | 10 | 0.2 |
| 10x  | 12  | 1250 | 500 | 312 | 50 | 20 | 10 | 0.2 |
| 20x  | 4.3 | 1250 | 500 | 312 | 50 | 20 | 10 | 0.2 |
| 50x  | 1.2 | 25   | 50  | 6   | 12 | 2  | 4  | 0.2 |
| 150x | 0.7 | 10   | 25  | 2.5 | 7  | 1  | 2  | 0.2 |

Validated data points: 20x coarse (1250/500) and 50x coarse (25/50).

### Sampling check

| Obj | Coarse samp/DOF | Fine samp/DOF | SF samp/DOF |
|-----|----------------|---------------|-------------|
| 2.5x | 3.8 | 15 | 478 |
| 5x   | 2.5 | 10 | 156 |
| 10x  | 0.6 | 2.4 | 37 |
| 20x  | 0.2 | 0.86 | 13 |
| 50x  | 3.0 | 12 | 37 |
| 150x | 4.4 | 17 | 44 |

≤20x coarse is below 3/DOF but works in practice — sharpness peaks are broad at low magnification. Fine/super-fine passes refine when needed.

## Implementation plan

### autofocus.py

1. Add `AFDefaults` NamedTuple with 7 fields: `z_speed`, `z_range`, `fine_speed`, `fine_range`, `sf_speed`, `sf_range`, `settle_time`
2. Add `_AF_DEFAULTS: dict[int, AFDefaults]` keyed by objective position
3. Replace `_get_safe_range()` with `_get_af_defaults()` returning resolved defaults + objective position. Caller-specified values override per-objective defaults; None = use default. Still validates z_range against WD safety limit.
4. Update `continuous_autofocus()`: use defaults for all 7 params. Fine speed changes from factor-based to absolute (from table). Callers can still override any individual param.

### commands/revisit.py

1. Change `z_speed` default from 1250 to None (auto from objective)
2. Change `settle_time` default from 0.2 to None (auto from objective)

### autofocus_demo.py

Already defaults z_speed to None. May need minor settle_time tweak.
