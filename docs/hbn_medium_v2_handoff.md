# hbn_medium_v2 — identity-CCM cutover handoff

Handoff from the 2026-08-07 deccm_eval session (paired legacy/identity scans of
Ben's hBN, `run_20260807_1106` vs `run_20260807_1106_identity`). Goal per Zack:
get the model curves into production **in identity colour space at minimal
quality loss**, so all pipelines can run on identity captures. This doc carries
every parameter and finding an integration session needs; the analysis
machinery lives in `scripts/deccm_paired_eval.py` (see `build_hbn_medium_v2()`
for a working config) but this doc is self-contained.

## 1. Integration directive: per-config colour matrix

Add a `colour_matrix` field to `DetectorConfig`:

- **No default** — every preset must declare its space explicitly.
- All existing presets → `"5800K"` (the space their calibrations were measured in).
- `hbn_medium_v2` → `"identity"`.
- The capture path (find-flakes / chip-scan / revisit) reads the material's
  `colour_matrix` instead of the global camera default, so legacy presets keep
  scanning legacy and v2 scans identity. No global default flip.

## 2. hbn_medium_v2 specification

### Calibration curve (theory-in-pipeline, no hand offsets)

Raw-space (R, G)(t) from the measured pipeline model:

- `derive_pipeline(_IMX183_WAVELENGTHS)`, objective `10x`
- `compute_rg(complex(2.060), oxide=90.0, NA=0.25, max_layers=200,
  red/green/blue_lit = _IMX183_* × op.chain_spd, glare_f = op.glare_f_raw)`
- n = 2.060 (v4 AFM-anchor fit), oxide 90 nm, zero offsets.
- Model B(t) (for the B floor / future use): same call with
  `mix = perm @ np.eye(3)`, `perm = [[0,0,1],[1,0,0],[0,1,0]]` — B returns in
  slot 0.

Validated centering against the good-flake population of this dataset:
median residual **R +0.020, G −0.006** — the model earns zero offsets in raw
space.

Representation: `RGPointDetectorConfig` with points every **0.1 nm** from 3–50 nm
(471 points), `layer_spacing_nm = 0.1` as pseudo-layers (`layers = round(t/0.1)`),
`classify_nm = True`. Rationale: distance quantization 0.003 at 0.1 nm spacing;
a degree-2 `R = f(G)` curve fit misses the model curve by up to 0.10 —
unusable against a 0.065 gate. If integration prefers a curve class, it needs
higher-order fitting; points are simpler.

### Parameters (provenance in brackets)

| param | value | provenance |
|---|---|---|
| capture | 0.25 ms, gain 4, bin 3, WB B,G,R = 2.51,1.02,1.41, colour_matrix identity | same as legacy hbn_medium scans, CCM removed |
| contrast_offset | 14.0 counts | legacy 15 × measured G count-delta scale 0.92 |
| subseg_min_std / min_range | 0.69 / 0.43 | legacy 0.8/0.5 × band arc-length scale 0.862 |
| cal_dist_match / possible | 0.10 / 0.20 | good-population 2D residuals (median 0.027) |
| tier1_cal_dist | 0.065 | 95% recall on labeled good set |
| tier2_cal_dist | 0.10 | — |
| tier1_thickness_window_nm | (4.5, 26.0) | replaces legacy G∈[0,3)/R<0 box (band-limiting was its covert job) |
| tier1_perim_ratio / aspect / min_size | 1.50 / 6.0 / 500 µm² | legacy values; note ~14 of 19 legacy-T1 losses in validation were 433–493 µm² size jitter — consider 450 |
| grad_energy gate (T1) | ≤ 80 | labeled-data crap gate (disordered flakes) |
| **B floor (T1)** | ≥ 0.41 | labeled-data crap gate — tape residue is B-deficient in identity space |
| R/G box gates, entropy, bg_ratio | disabled (±99) | mixed-space semantics do not migrate; see §4 |

### Scorer

`_score_hbn_thick_50_100` has the right structure (cal_dist + thickness window +
shape gates, no colour demotions) and was used for validation with the two crap
gates applied as a post-tier filter. Production wants a small
`_score_hbn_medium_v2` = that structure **+ `grad_energy ≤ tier1_grad_energy_max`
+ B ≥ B-floor** natively. Do NOT reuse `_score_hbn_medium`: its hardcoded
`b/g > 1.2` purple demotion misfires in identity space (raw hBN B/G is ~2.4
mid-band).

## 3. Why the gates look like this (knowledge transfer)

- **Mixed-space 2D cal_dist and the r<0 gate did covert 3-channel work.** The
  5800K CCM mixes B into R/G, so B-anomalous junk fell off the mixed empirical
  curve and outside r<0. In raw space the 2D projection loses that; naive
  arc-scaled gate migration inflated T1 3.5× with (Zack-verified) tape and
  disordered flakes. Curve+gates always shared the rejection job.
- **Scalar gate scaling is wrong in principle**: distance normal to the curve
  compresses ~0.38× under de-CCM while band arc length compresses 0.862× —
  and it is direction/band-position dependent. Hence from-scratch gates, not
  transformed ones.
- **2D beats 3D**: on eyeball-labeled sets (good n=129 / crap n=316; unclipped
  subsets too) a 3D (R,G,B) curve distance adds nothing over 2D + separate
  gates (crap leak identical within noise). Decision: 2D cal_dist.
- **B is a healthy channel in identity space** (this reverses the historical
  "noisy / low-contrast" verdict, which was a mixed-space artifact): per-scan
  per-detection repeatability σ ≈ 0.045 (same as R/G), model dynamic range
  0.27→2.35 over 4–30 nm (G-class). Its production role for now is the simple
  ≥0.41 floor.
- **Oxide tolerance (±1 nm) is absorbed by thickness, not cal_dist**: the curve
  displacement under oxide change is ~90% along-curve (≈0.7 nm thickness
  relabel per nm oxide; normal component only ~0.02). B is oxide-insensitive
  in-band (dB/d(ox) ≈ 0.01/nm).

## 4. Known edges / follow-ups

1. **Thin-end B floor**: the fixed 0.41 floor demotes clean in-band 4.6–6 nm
   flakes (measured B runs ~0.9× model; model B(4.6–6) ≈ 0.44–0.55 sits on the
   boundary). Three such flakes on this dataset (e.g. c1 frame_0195#1) landed
   T2. Recommended refinement once the B recentering is settled (lambda_b_fit
   task arbitrating offset −0.06 vs scale ~0.92 vs Zack's widget fit 0.80):
   thickness-conditioned floor `B ≥ k·model_B(t)`.
2. **B clipping**: identity-space B substrate ≈ 98 counts → contrast ceiling
   ≈ 1.60, reached by ~19 nm at 0.25 ms/gain 4 — and **partial in-region
   saturation is widespread** (69% of good-labeled dets have some B=255
   pixels). Any future B-dependent scoring beyond the floor needs per-detection
   saturation masking or a capture change. Blooming rule: clipped dets' R/G are
   suspect too.
3. **Entropy gates are disabled**, not migrated — entropy is computed on
   contrast values, which compress under de-CCM; the legacy 4.65 threshold
   under-fires in raw space. grad_energy took over the disorder-rejection job;
   revisit if disorder leaks through.
4. **Overfit caveat**: grad_energy 80 / B 0.41 were tuned on 209 labeled
   detections from this one paired dataset (~90% joint recall, crap leak
   26%→8% inside T1). Validate on the next identity scan before trusting the
   exact values.
5. **Flatfields — new captures required at cutover.** The mixed-space
   flatfield is a CCM-mixture of the raw per-channel vignette profiles; the
   native identity vignette is the correct object. Capture fresh flatfields
   with `--colour-matrix identity` (same build_flatfield procedure) per
   objective/binning as presets migrate. `deccm_frames.transform_flatfield`
   stays analysis-only: it applies M⁻¹ to correction *factors* as if they were
   a signal (≈0.5% rms / 2% max corner error; exact fix needs per-channel
   capture means, which the .json doesn't store — store `channel_means` going
   forward if the transform is kept).
6. **Legacy 50x revisit archive**: 3.0% mean / 19.7% max clip, unrecoverable by
   de-CCM (feature-pixel B off by −12 counts mean). Identity 50x still clips
   2.2% — 50x revisit capture settings deserve a review.
7. **Ultrathin opportunity**: <5 nm hBN is G-invisible (G ≈ 0) but B-carried
   (B ≈ +0.26–0.31) in identity space — a future thin preset would gate on B,
   with a thickness-conditioned floor from day one.

## 5. Validation record (acceptance vs Jordan's legacy scan)

Floor = `run_20260807_1106` (legacy hbn_medium, operator Jordan). v2 numbers on
`run_20260807_1106_identity` (identical chips/hulls/planes/camera, CCM only):

| | legacy floor | v2 + gates |
|---|---|---|
| T1 (chip0+chip1) | 77 + 94 | 88 + 102 |
| labeled crap in T1 | ≥15% of legacy T1 B-deviant/junk | 16/190 (8%) |
| labeled good demoted | — | 14/149 (all land T2) |
| legacy-T1 kept at T1 | — | ~85% (losses: size jitter + the 3 thin-end B-floor cases) |
| thickness vs legacy | — | median \|Δ\| 3.3 nm (cal-shape difference; within-config repeatability 0.7 nm) |

Zack eyeball verdicts: v2 T1 top-20 ≈ legacy quality; demotions mostly tape
(correct) except the thin-end trio. De-CCM tooling itself validated at full
scan scale: detection parity 95%/93%, contrast round-trip to 0.009, pixel
parity at the two-capture noise floor.

Re-validation protocol for the next identity scan: segment with v2, spot-check
T1 mosaic + demoted mosaic (`scripts/crop_mosaic.py --seg-name <seg>`), and if
a paired legacy scan exists, run the floor check
(`scripts/deccm_paired_eval.py --v2 --v2-gates`).

## 6. Migration recipe for other materials

Per preset: (1) model curve in raw space from the pipeline model (material n
refit in raw space — graphene: n = 2.094, mixed-space 2.060 carried ~+3 nm
lamp-chain bias, per graphene_pipeline_effects); (2) 2D cal_dist with gates set
from a labeled paired dataset, NOT scalar-scaled from legacy; (3) thickness
window replaces R/G box gates; (4) shape gates carry over as-is; (5) crap gates
(grad_energy, B floor) re-derived per material; (6) fresh identity flatfield
for the capture objective. Note `hbn_medium_285nm` has hardcoded B gates in its
scorer (B < −0.08 etc.) — mixed-space values, must be re-derived, not copied.

## 7. Data of record

- Paired scans: `scans/run_20260807_1106{,_identity,_deccm}` (local)
- Labels + per-detection records: `/tmp/deccm_eval/labels_good_crap.json`
  (incl. `b_sat_frac` saturation flags)
- Session log: Scan Notebook 2026-08-07 (deccm_eval entries incl. B-offset
  correction); mosaics/plots: zeeref session `default` (A–I)
- Analysis tooling: `scripts/deccm_paired_eval.py` (stages: --pair-only,
  --pixel, --segment/--det-compare, --b3d, --plot-b, --v2/--v2-gates,
  --revisit)
