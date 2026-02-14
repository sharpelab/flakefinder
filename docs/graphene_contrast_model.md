# Graphene Contrast-to-Thickness Model

Research into 2DMatGMM's thin film optics model for graphene layer discrimination.
Based on Uslu et al. 2024 (Mach. Learn.: Sci. Technol. 5 015027) and the
[2DMatGMM codebase](https://github.com/Jaluus/2DMatGMM).

## Q1: How does 2DMatGMM compute expected contrast?

**Short answer: The production system does NOT use transfer matrix / Fresnel
calculations. Contrast values are empirically measured, not computed from optics.**

The paper describes the thin-film optics simulation (Section 4.2, Figure 6) as a
**design justification tool** -- used to choose the optimal SiO2 thickness (90nm
for graphene) -- NOT as part of the detection pipeline. The simulation uses:

- **Solcore** library (ref [35]) for transfer matrix reflectivity calculations
- Inputs: refractive indices of graphene (ref [36]: Weber 2010), SiO2 (ref [38]:
  Malitson 1965), Si (ref [37]: Green & Keevers 1995)
- Output: reflectivity R(lambda) for substrate vs substrate+N-layer-graphene,
  integrated against camera spectral sensitivity and illumination spectrum to get
  per-channel contrast (Equation 3 in the paper)

The **production detection pipeline** is:

1. Flatfield correction (divide by flatfield, multiply by mean)
2. Median blur (5x5)
3. Background extraction: histogram mode per channel (mean within +/-5 counts of mode)
4. **Contrast computation: `C = pixel / bg - 1`** (identical to our `(pixel - bg) / bg`)
5. GMM classification via Mahalanobis distance to pre-trained cluster centers

The GMM parameters (means + covariance matrices per layer) are trained from ~5
manually-identified flakes per layer count, stored in JSON files. They are NOT
computed from physics.

## Q2: Expected normalized contrast values for graphene on 90nm SiO2

Three different parameter sets exist in the codebase, each trained on different data:

**Graphene_90nm_10x** (the standard thin-graphene config):

| Layer | R | G | B |
|-------|--------|--------|--------|
| 1L | -0.128 | -0.160 | -0.109 |
| 2L | -0.225 | -0.288 | -0.202 |
| 3L | -0.312 | -0.406 | -0.287 |
| 4L | -0.408 | -0.510 | -0.335 |

**GrapheneThick_90nm_10x** (extended to 5 layers):

| Layer | R | G | B |
|-------|--------|--------|--------|
| 1L | -0.146 | -0.150 | -0.067 |
| 2L | -0.269 | -0.275 | -0.121 |
| 3L | -0.388 | -0.392 | -0.169 |
| 4L | -0.490 | -0.492 | -0.215 |
| 5L | -0.577 | -0.576 | -0.250 |

**Graphene_GMM.json** (GMMDetector library default):

| Layer | R | G | B |
|-------|--------|--------|--------|
| 1L | -0.137 | -0.140 | -0.071 |
| 2L | -0.248 | -0.264 | -0.152 |
| 3L | -0.347 | -0.368 | -0.216 |
| 4L | -0.436 | -0.465 | -0.272 |

Key observations:

- **All three datasets agree on the structure**: approximately linear, well-separated clusters
- **Absolute values differ by up to ~25%** between datasets -- this is from
  different wafer oxide batches, different cameras, different training data
- **G channel has the largest contrast per layer** (~0.12-0.14 step per layer)
- **B channel is unreliable** -- the paper explicitly states that at 90nm oxide,
  B has a large derivative w.r.t. oxide thickness, making it unstable across
  wafer batches (Figure 6a). This is why the "GrapheneThick" config uses only
  R+G channels for discrimination

## Q3: R-G contrast space trajectory

The graphene thickness trajectory is **approximately linear through the origin**
in R-G space, much simpler than hBN's polynomial curve:

| Dataset | R/G ratio range | Character |
|---------|----------------|-----------|
| Graphene_90nm_10x | 0.77-0.80 | R ~ 0.79*G |
| GrapheneThick_90nm_10x | 0.97-1.00 | R ~ G (diagonal) |
| Graphene_GMM.json | 0.94-0.98 | R ~ 0.95*G |

The R/G ratio is approximately constant across layer counts within each dataset,
meaning the trajectory is nearly a straight line from the origin into the third
quadrant. The variation in slope (0.78-1.0) between datasets is due to oxide
thickness variation and camera differences.

**This is fundamentally different from hBN**, which follows a curved polynomial
in R-G space. Graphene is a straight line (or very nearly so). A linear
`cal_poly` would work fine.

## Q4: Thresholds for "interesting" vs "too thick"

Averaging across all three datasets:

| Category | G contrast range | Layer count |
|----------|-----------------|-------------|
| **Thin (most valuable)** | -0.30 to -0.10 | 1-2L |
| **Medium (useful)** | -0.60 to -0.30 | 3-5L |
| **Thick (less interesting)** | < -0.60 | >5L |
| **Background/noise** | > -0.10 | substrate/dust |

Per-layer G contrast steps are roughly ~0.13 per layer, remarkably uniform.

**Important caveat**: These exact thresholds are from their camera (Imaging
Source DFK 33UX174) on their microscope (Nikon Eclipse). Our Leica K5C will have
different spectral sensitivity, so the absolute values will differ. The structure
(linear, well-separated clusters) should transfer, but the exact numbers need
calibration from our own data.

## Q5: Can we reuse `cal_poly` / `cal_distance`?

**Yes, with minor adjustments.** The infrastructure is a good fit:

1. **`cal_poly`**: For graphene, a degree-1 polynomial (linear) is sufficient:
   `cal_poly = (slope, 0.0)` where slope represents the R/G ratio. Placeholder
   starting point: `cal_poly = (0.9, 0.0)` meaning R = 0.9*G. We must calibrate
   from our own measured flakes -- the 2DMatGMM numbers are from a different camera.

2. **`cal_distance`**: Works perfectly. Perpendicular distance from the linear
   trajectory discriminates graphene from non-graphene.

3. **`cal_g_range`**: Should be `(-0.7, 0.0)` for graphene (negative G values,
   opposite sign from hBN).

4. **Classification by G**: Current `g_thin_max` / `g_medium_max` infrastructure
   works. Since graphene contrasts are negative, "thin" means G closest to 0
   (least negative). Suggested starting points:
   - `g_thin_max = -0.30` (1-2L)
   - `g_medium_max = -0.60` (3-5L)
   - Everything more negative = "thick"

5. **`cal_dist_match` / `cal_dist_possible`**: Same concept, similar scale. The
   covariance matrices from 2DMatGMM show sigma ~ 0.01-0.03 in R and G, so
   `cal_dist_match=0.05` and `cal_dist_possible=0.10` would be reasonable
   starting points (tighter than hBN's 0.5/1.0 because graphene clusters are
   tighter).

**One thing needs changing**: The current graphene `score_detection()` path
bypasses cal_distance entirely (line 80-88 of detector_config.py -- "no cal
curve yet"). Once we have calibrated coefficients, we should switch to the
hBN-style scoring that uses `cal_dist`.

## Q6: Is the model WB-dependent?

**No, in the ideal case.** The contrast normalization
`C = (pixel - bg) / bg = pixel/bg - 1` cancels multiplicative WB gain:

```
C = (WB * I_true - WB * I_bg) / (WB * I_bg) = (I_true - I_bg) / I_bg
```

The WB factor divides out. This is confirmed by:

- 2DMatGMM's camera_parameters.json sets **gamma=1** (linear response), which is
  essential for the cancellation to work
- Their WB is [1.7, 1.0, 1.4] BGR; ours is [2.51, 1.02, 1.41] BGR -- the 47%
  difference in blue gain should cancel
- The paper's Equation (1) explicitly uses division by background, making it
  WB-independent
- Different training datasets produce slightly different contrast values, but
  this is attributable to oxide thickness variation, not WB

**Caveat**: This only holds if:

1. Camera response is linear (gamma=1) -- both systems use gamma=1
2. No nonlinear processing (tone mapping, auto-contrast)
3. No channel crosstalk (negligible for modern sensors)

Our WB difference should NOT affect contrast values. The B channel difference
(2.51 vs 1.7) would have the largest impact if there were any nonlinearity, but
since B is the unreliable channel for graphene/90nm anyway, this doesn't matter.

## Q7: Physical constants from the code/paper

**Refractive indices** (from paper references, NOT in code):

- **Graphene**: n = 2.6 - 1.3i at 550nm (Weber et al. 2010, ref [36]).
  Wavelength-dependent; the imaginary part (absorption) is what produces the
  contrast
- **SiO2**: n ~ 1.46 at visible wavelengths (Malitson 1965, ref [38]).
  Sellmeier equation, essentially real-valued (no absorption)
- **Si**: Complex, wavelength-dependent (Green & Keevers 1995, ref [37]).
  Strongly absorbing. n ~ 4.15 - 0.044i at 550nm; n ~ 3.88 - 0.018i at 630nm

**Graphene monolayer thickness**: 0.335 nm (standard graphite interlayer spacing).
N-layer graphene thickness = N * 0.335 nm.

**Oxide thickness**: 90nm SiO2 (dry oxidation on Si). Chosen because R and G
channels have small derivative dC/d(t_ox) at 90nm, making detection robust to
+/-few nm thickness variation. B channel is NOT stable at 90nm (large derivative,
unreliable).

**Camera spectral sensitivity**: Not in the code; paper uses "an approximate
spectrum of the illumination" and the DFK 33UX174 camera response curve. These
are microscope-specific.

**None of the thin-film optics constants appear in the production code.** The
code only stores empirically trained GMM parameters (mean contrast vectors +
covariance matrices per layer). The transfer matrix simulation that generated
Figure 6 appears to have been done in a separate analysis script using Solcore,
not included in the published codebase.

## Summary / Key Takeaways

1. **No physics-based computation in production** -- 2DMatGMM's detection is
   purely empirical (GMM on measured contrast), using the same normalization we
   use. The thin-film optics simulation was only for design decisions (choosing
   90nm oxide).

2. **Our `cal_poly` infrastructure works** -- graphene's R-G trajectory is
   approximately linear through the origin, even simpler than hBN's polynomial
   curve. A linear `cal_poly = (slope, 0.0)` with slope ~0.85-0.95 is a good
   starting structure.

3. **We need our own calibration data** -- the 2DMatGMM contrast values are from
   a different camera/microscope, so the exact numbers won't transfer. But the
   *structure* (linear trajectory, ~0.13 G-contrast per layer, well-separated
   clusters) should hold for any microscope on 90nm SiO2.

4. **B channel is unreliable** for graphene on 90nm SiO2 -- only use R and G for
   thickness discrimination. This is a physics limitation (thin-film interference
   at 90nm oxide), not a camera issue.

5. **WB is irrelevant** -- our contrast normalization cancels it out, confirmed
   by both theory and 2DMatGMM's design.
