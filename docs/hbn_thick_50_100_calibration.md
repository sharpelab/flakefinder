# Thick hBN (50–100 nm) Detection on 90 nm SiO₂

Preset: `hbn_thick_50_100_90nm` (display "hBN 50-100nm"), cal table
`HBN_THICK_50_100_90NM_CAL_POINTS` in `src/flakefinder/segmentation.py`.
Target: QPC-quality flakes in the 50–100 nm band; strongest recall in the
band middle (~60–80 nm), weaker performance at the 90–100 nm edge accepted.

## Calibration table generation

26 reference points at 2 nm spacing over 50–100 nm, generated headlessly
from the transfer-matrix code path the widget's **Export all** button uses
(`scripts/hbn_contrast_widget.py` `compute_rg`), with parameters pinned to
the widget's checked-in `init` defaults:

```
n_hBN                  = 2.152 (constant, real)
t_oxide                = 90.0 nm
NA                     = 0.25
illuminant             = halogen_3200K
objective transmission = off
r_offset               = +0.54
g_offset               = -0.20
convention             = camera space = model − offset
hBN layer thickness    = 0.333 nm
sampling               = 2 nm, 50 → 100 nm
```

n=2.152/NA=0.25 was chosen over the doc's historical n=1.91 fit, which fails
its own thick-end anchors (see caveats in `docs/hbn_contrast_model.md`), and
over session-specific widget exports (differences ≤0.13 in R/G across the
band — immaterial, and the checked-in defaults are reproducible).

## Why point-based (RGPointDetectorConfig), not a curve

The band is an arc that doubles back in G: G peaks at +4.81 near 68 nm, so
`R = poly(G)` is multivalued over G ∈ [4.41, 4.81] and
`CurveDetectorConfig`'s G-parameterized thickness interpolation would
produce garbage. The arc does not self-intersect in 2D, so nearest-point
distance over (R, G) recovers thickness correctly. Quantization error from
the 2 nm spacing is ≤0.08 (chord half-length), negligible against the 0.30
tier-1 gate. `layers = round(t_nm / 0.333)` is literally the hBN layer
count; classification reports nm (`classify_nm=True`).

## Two-regime scoring (`_score_hbn_thick_50_100`)

The clip ceiling per channel is exactly `(255 − bg) / bg`, so it moves with
capture gain: at chip-scan gain 2 the G ceiling (~6.1) clears the band's
G max of 4.81 and flakes are measured unclipped; at gain 4 the G ceiling
(~3.6) pins every in-band flake to the ceiling corner. Two independent
tier-1 paths handle this:

1. **Curve path** (unclipped / gain ~2): `cal_dist < 0.30`, R ∈ [2.3, 4.0],
   G ∈ [3.0, 5.1], plus shape gates (perim_ratio < 1.5, aspect < 6,
   size ≥ 500 µm²).
2. **Ceiling path** (pinned / gain ~4): R ∈ [3.8, 5.2], G ∈ [2.8, 4.2],
   same shape gates, **no cal_dist requirement** — a pinned detection is
   distorted off the model locus by construction; requiring curve proximity
   would punish exactly the distortion that identifies it.
   - The R ≤ 5.2 upper bound: a pixel-saturated blob cannot exceed its own
     clip ceiling (≈5.0 in R at gain 4), so R > 5.2 only occurs in
     unclipped captures — far above the band's R ceiling of 3.70 and
     therefore not band hBN. Validation: 0 of 4590 gain-4 ceiling-path T1s
     exceed 5.2 (max 5.03 = the clip ceiling), while 55–64% of gain-2
     ceiling-box admits did and were above-band chunks.

Tier 2: `cal_dist < 0.60` + shape/size. Entropy gates are disabled (99.0):
band flakes have every pixel contrast above +1, so the fixed-range (−1, 1)
`_hist_entropy` histogram is empty and entropy evaluates to exactly 0.0.

Score: `log2(size)² × cal_factor × aspect_penalty` with
`cal_factor = exp(−2·cal_dist)` on the curve path (gentle: the band spans
~1.0 in R and ~1.5 in G, so an `exp(−8·cd)` weight would rank by model-fit
noise, not flake quality) and a constant `exp(−2·0.30)` on the ceiling path
(clipped flakes rank purely by size).

## Gain invariance of contrast

The two-regime design assumes per-channel contrast is gain-invariant while
unclipped. Verified empirically on hbn_medium tier-1 populations
(unclipped by their G < 3.0 gate):

| Run (gain) | n | R median | G median |
|---|---|---|---|
| run_20260409_1721 (2.0) | 606 | −0.555 | +0.601 |
| run_20260221_1651 (2.0) | 267 | −0.558 | +0.643 |
| run_20260505_1616 (4.0) | 101 | −0.546 | +0.687 |

The gain-2↔gain-4 gap is within the run-to-run scatter of the two gain-2
runs. Invariance holds.

## Offline validation (scan-wide rerank, dedup 50 µm)

| Run (gain) | detections | T1 | notes |
|---|---|---|---|
| run_20260409_1721 (2.0) | 23 109 | 830 | top-50 dominated by clean curve-path hits, cd < 0.2 |
| run_20260221_1651 (2.0) | 61 400 | 863 | chunk-rich chips; residual ceiling admits at R 3.8–5.2 |
| run_20260505_1616 (4.0) | 69 332 | 3 510 | all T1 via ceiling path; curve path fires 0 times (everything pinned) |

Mosaics: `<run>/rerank/hbn50100_v2/hbn50100_v2.jpg`.

## Impostor posture

150–260 nm hBN genuinely wraps back into the band's (R, G) region (e.g.
230 nm sits at R +3.52, G +3.11) and cannot be separated by contrast. On
gain-2 data, unclipped chunks with R ∈ [3.8, 5.2] also share the ceiling
box with nothing to distinguish them. Both are accepted: the preset is
tuned for low false negatives; the human review + AFM screen rejects
impostors downstream.
