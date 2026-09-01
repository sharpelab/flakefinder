# hBN Detection on 285nm SiO₂ Substrates

## Background

The existing hBN presets (`hbn_thin`, `hbn_medium`) use a calibration curve built from
AFM-verified flakes on **90nm SiO₂**. On 2026-02-26, we ran the first hBN scan on
**285nm SiO₂** substrates (run_20260226_1657, 4 chips, 10x, hbn_medium preset,
WB 2.51/1.02/1.41, gain=2, 0.25ms exposure, operator: Ben Alexander).

Result: **913 T1 detections — all tape residue, zero real flakes.**

## Why the 90nm Preset Fails on 285nm

### Substrate color is drastically different

On 285nm SiO₂ with the hBN white balance (B=2.51, G=1.02, R=1.41):
- **10x scan BG modes**: B≈151, G≈39, R≈20
- Compare to 90nm SiO₂: B≈130, G≈100, R≈80

The 285nm substrate reflects almost no red or green light — the R and G channels
are starved (20 and 39 out of 255). This means:
- `contrast_offset=15` represents **75% of R background** (vs ~19% on 90nm)
- Normalized contrast is hypersensitive: tiny absolute differences → large relative contrast
- No clipping occurs (max pixels ~170), but dynamic range is terrible in R/G
- `contrast_offset=15` misses thin hBN entirely (needs G≥54, but thin hBN is G≈42-50)

### The calibration curve passes through tape territory

The 90nm cal curve happens to run through the R-G region where tape residue falls
on 285nm substrates. Tape residue T1 median: R=−0.63, G=+0.81, cal_dist=0.075.

### Real hBN sits below the 90nm curve

Manual measurements of confirmed hBN flakes on 285nm show real hBN is consistently
**~0.15–0.25 more R-negative** than the 90nm curve at the same G contrast.

## Measured Contrast Data

### 10x scan data (directly from scan frames)

Source: frame_0371 chip_0 (confirmed dense hBN frame), BG modes BGR=(151, 39, 20)

| Description | R contrast | G contrast | B contrast | Notes |
|-------------|-----------|-----------|-----------|-------|
| Very thin (large) | −0.895 | +0.421 | — | Found with offset=7; 1396 µm², invisible at offset=15 |
| Very thin | −0.827 | +0.458 | — | 1114 µm², also invisible at offset=15 |
| Thin candidate | −0.889 | +0.536 | +0.009 | 716 µm², sol=0.72, ent=3.39 |
| Thin candidate | −0.913 | +0.563 | +0.018 | 488 µm², sol=0.85, ent=3.26 |
| Small thin (below size gate) | −0.736 | +0.820 | — | 260 µm², cal_dist=0.063, fits curve |
| Small thin (below size gate) | −0.883 | +0.966 | — | 259 µm², cal_dist=0.121, fits curve |
| Medium | −0.780 | +1.438 | — | 2038 µm², ent=4.27, sol=0.67 |
| Medium | −0.560 | +1.390 | −0.076 | 4117 µm², sol=0.75 |
| Medium | −0.499 | +1.554 | −0.090 | 584 µm², sol=0.56 |
| Medium | −0.526 | +1.618 | −0.076 | 736 µm², sol=0.84 |
| Medium | −0.725 | +1.296 | −0.053 | 420 µm², sol=0.90, ent=3.89 |
| Medium | −0.759 | +1.546 | −0.020 | 1914 µm², sol=0.69 |
| Thick (orange) | +1 to +9 | +1.5 to +4 | −0.2 to −0.7 | Way off curve, very R-positive |

### Manual crops (user-identified flakes)

| Source | Magnification | Substrate BGR | R contrast | G contrast | B contrast |
|--------|--------------|---------------|-----------|-----------|-----------|
| Crop 1 (thin) | ~10x | (163, 42, 14) | −0.821 | +0.651 | +0.023 |
| Crop 4 thinner | 20x | (170, 46, 21) | −0.737 | +1.132 | +0.055 |
| Crop 4 middle | 20x | (170, 46, 21) | −0.965 | +1.534 | +0.088 |
| Crop 4 thicker | 20x | (170, 46, 21) | −0.776 | +1.699 | +0.039 |
| Crop 2 thinner | 50x | (163, 51, 64) | −0.933 | +0.709 | +0.378 |
| Crop 2 thicker | 50x | (163, 51, 64) | −0.952 | +0.935 | +0.371 |
| Crop 3 thinnest | 50x | (160, 50, 57) | −0.557 | +1.779 | +0.245 |
| Crop 3 bulk | 50x | (160, 50, 57) | −0.661 | +2.588 | +0.414 |
| Crop 3 thickest | 50x | (160, 50, 57) | −0.291 | +2.826 | +0.312 |

### Tape residue profile (from 913 T1 false positives with 90nm preset)

| Metric | Median | p25 | p75 |
|--------|--------|-----|-----|
| R contrast | −0.630 | −0.713 | −0.534 |
| G contrast | +0.807 | +0.678 | +1.390 |
| B contrast | −0.078 | −0.107 | −0.057 |
| Entropy | 4.106 | 3.886 | 4.342 |
| Solidity | 0.734 | 0.631 | 0.832 |
| Perim ratio | 1.258 | 1.155 | 1.367 |
| Circularity | 0.351 | 0.256 | 0.475 |
| Size (µm²) | 734 | 596 | 1061 |

## Key Discriminators (hBN vs Tape on 285nm)

1. **R contrast**: Real hBN is more R-negative (−0.82 to −0.95 for thin) vs tape (−0.63 median)
2. **B contrast**: Real hBN tends B-positive (+0.02 to +0.09 at 10x) vs tape B-negative (−0.08)
3. **Entropy**: Real thin hBN has lower entropy (3.2–3.5) vs tape (4.1 median)
4. **Shape**: Real flakes have higher solidity (0.85+) and lower perim_ratio (<1.15)
5. **Small pieces on curve**: Sub-400µm² pieces with cal_dist < 0.15 and R < −0.7 are likely real

## 285nm Calibration Curve

Built from 10x measurements + 20x manual crops. Thickness values are rough estimates (no AFM).

```python
HBN_285NM_CAL_POINTS = (
    # (R_contrast, G_contrast, est_thickness_nm)
    (-0.90,  0.42,   3.0),   # very thin — large flake below d35, found at offset=7
    (-0.90,  0.55,   5.0),   # thin — frame_0371 detections
    (-0.82,  0.65,   7.0),   # thin — crop1 manual ID
    (-0.74,  1.13,  12.0),   # medium-thin — crop4 20x
    (-0.56,  1.39,  16.0),   # medium — frame_0371 detection
    (-0.53,  1.62,  18.0),   # medium — frame_0371 detection
)
```

Curve fit: R = poly(G), quadratic. The curve sits ~0.2 below the 90nm curve in R.

## `hbn_medium_285nm` Preset (Current State)

Key differences from `hbn_medium`:
- **`contrast_offset=7`** (was 15) — 285nm has low G bg (~39), so offset=15 misses thin hBN
- **`cal_points=HBN_285NM_CAL_POINTS`** — 285nm-specific calibration curve
- **`cal_g_range=(-0.5, 3.0)`** — narrower since thick hBN on 285nm goes wildly R-positive
- **Scoring `_score_hbn_medium_285nm`**:
  - T1: `r < -0.70` (was `r < 0.0`) — rejects tape at R≈−0.63
  - T1: `cd < 0.20` (was `cd < 0.15`) — slightly wider for the new curve
  - T1: `ent < 4.5` (was `ent < 99`) — rejects high-entropy tape
  - T1: `min_size_um2 = 400` (was 500)
  - T2: `r < -0.45`, `cd < 0.25` — catches medium pieces that fail R gate

### Test results on frame_0371 (v2, offset=7 + extended curve)

| Preset | Total | T1 | T2 | T3 |
|--------|-------|----|----|-----|
| hbn_medium (90nm) | 59 | 3 (all tape) | 0 | 56 |
| hbn_medium_285nm v1 (offset=15) | 59 | 4 (thin hBN!) | 6 | 49 |
| hbn_medium_285nm v2 (offset=7) | 78 | 7 | 4 | 67 |

v2 T1s are all real hBN candidates. The large thin flake (1396 µm², R=−0.90, G=+0.42)
was completely invisible at offset=15 and is now the highest-confidence T1.

### Known issues / next steps

1. **Curve needs more thin-end data** — only 2 points below G=0.65. User will share more
   thin flakes from a new frame to anchor this end of the curve.
2. **Thick hBN (R >> 0) isn't on the curve** — on 285nm, thick pieces have R contrast
   of +1 to +9 due to the R bg being only 20. The curve only covers thin/medium range.
   Thick flakes would need a separate detection strategy or extended curve.
3. **Size gate (400 µm²) misses small real flakes** — confirmed pieces at 259-265 µm²
   fit the curve well (cal_dist 0.06–0.12) but are below the gate. Acceptable at 10x;
   these would be caught at higher-mag revisit.
4. **Not yet tested on full run** — only evaluated on frame_0371 so far.
5. **Tape at R≈−0.71 could sneak past** — the R < −0.70 gate has tape's p25 at −0.71.
   May need to tighten to −0.75 if false positives appear.

## Run Data

- **Run**: `scans/run_20260226_1657/`
- **Key frame**: `chip_0/scan_10x/frame_0371.jpg` — dense with real hBN flakes
- **Flatfield**: `calibration/flatfield_10x_5800K_bin3.npy`
- **pixel_size_um**: 0.720703125
- **Symlinked into worktree** at `.worktrees/hbn_detector/scans/run_20260226_1657/`

## Reference Images

All in `scans/run_20260226_1657/rerank/`:
- `frame_0371_285nm_ids.jpg` — v1 overlay with detection IDs
- `frame_0371_285nm_v2.jpg` — v2 overlay (offset=7)
- `all_tiers_scatter.png` — full R-G scatter, all tiers, all chips
- `hbn_vs_tape_285nm.png` — real hBN vs tape in R-G space
- `frame_0371_rg.png` — single-frame R-G scatter with manual reference points
