# Thick hBN (50–100 nm) Detection on 90 nm SiO₂

Preset: `hbn_thick_50_100_90nm` (display "hBN 50-100nm"), cal table
`HBN_THICK_50_100_90NM_CAL_POINTS` in `src/flakefinder/segmentation.py`.
Target: QPC-quality flakes in the 50–100 nm band; strongest recall in the
band middle (~60–80 nm), weaker performance at the 90–100 nm edge accepted.

## Calibration table generation

36 reference points at 2 nm spacing over 40–110 nm, generated from the
transfer-matrix code path the widget uses (`compute_rg`), via
`scripts/fit_hbn_thick_anchors.py --export --table`:

```
n_hBN                  = 2.165 (constant, real; see below)
t_oxide                = 90.0 nm
NA                     = 0.25
illuminant             = halogen_3200K
objective transmission = 10x T²
glare                  = off
r_offset               = +0.60
g_offset               = -0.30
convention             = camera space = model − offset
hBN layer thickness    = 0.333 nm
sampling               = 2 nm, 40 → 110 nm
```

The table deliberately extends past the 50–100 nm target band: nearest-point
assignment clamps out-of-band flakes onto whichever endpoint the table
happens to stop at, so table extent must not double as band policy. Band
membership is enforced by `tier1_thickness_window_nm` (below); the 40/110
extent covers the tier-2 `cal_dist < 0.60` acceptance region beyond the
window so out-of-band flakes get honest thickness estimates.

## AFM-anchored calibration (10x)

Two AFM anchor rounds by Toghrul, both 10x / gain 2.0 / flatfielded frames
/ hBN WB 1.41/1.02/2.51, recorded in `scripts/hbn_contrast.py`
(`_CAL_DATA_HBN_10X_THICK`, `_CAL_DATA_HBN_10X_THICK_R2` and companions):

- **Round 1** (scan 287, `run_20260804_1354`): 9 flakes, 45.5–57.5 nm.
- **Round 2** (scan 288, `run_20260804_1303`): 12 flakes AFM'd from the
  v3-rerank favorites; 8 in-band anchors spanning 55–85 nm, plus two
  past-fold flakes and one exclusion (below).

The parameter set was chosen interactively in the widget against the full
overlay — the combined thick anchors *and* the thin-band (2–46 nm,
prev-scope 50x) reference points, which share the offset sliders. A
least-squares alternative fitted to the thick anchors alone (n=2.354,
offsets +1.03/+0.18) predicts the thick anchors better (combined
re-prediction +1.3 ± 3.9 nm) but pushes the thin-band points far off the
curve, and was rejected for whole-dataset consistency.
`scripts/fit_hbn_thick_anchors.py` reproduces both fits and scores any
widget export against the anchor sets.

Re-prediction of the anchors through the shipped curve (nearest-point
projection, the segmentation readout):

| round | AFM nm | pred nm | Δ |
|---|---|---|---|
| 1 | 45.5 | 48.6 | +3.1 |
| 1 | 46.5 | 46.0 | −0.5 |
| 1 | 48.5 | 53.3 | +4.8 |
| 1 | 49.0 | 50.9 | +1.9 |
| 1 | 50.5 | 53.3 | +2.8 |
| 1 | 54.0 | 56.9 | +2.9 |
| 1 | 54.5 | 60.3 | +5.8 |
| 1 | 57.5 | 61.3 | +3.8 |
| 2 | 55.5 | 63.6 | +8.1 |
| 2 | 55.5 | 70.3 | +14.8 |
| 2 | 56.5 | 69.3 | +12.8 |
| 2 | 59.5 | 74.6 | +15.1 |
| 2 | 66.0 | 75.9 | +9.9 |
| 2 | 69.5 | 79.6 | +10.1 |
| 2 | 82.0 | 95.9 | +13.9 |
| 2 | 85.0 | 90.6 | +5.6 |

Round 1: **+3.1 ± 1.8 nm**; round 2: **+11.3 ± 3.2 nm**. The upper-band
over-read is a known, accepted property of this calibration: reported
thicknesses in the 60–100 nm range run high by roughly 5–15 nm, and AFM
remains the authority for any flake that matters.

### Anchor exclusions and flags (revisit on Toghrul's recheck)

- **98982** (round 1, AFM 56.5): R sits 0.39 below the neighboring
  anchors' trend (~4× the ±0.1 repeatability); site under AFM re-check.
- **99280** (round 2, AFM 45–47): AFM note reads "bottom blue part" —
  likely a different region of a multi-region flake was measured (the
  flake's contrast reads 81.9 nm through this table). Excluded pending
  recheck.
- **99236** (round 2 anchor at 85.0): AFM profile is concave, 87–81–89 nm
  over a 9.5 µm span; the anchor uses 85.0, the midpoint of the 81–89
  extremes. A treatment choice, not a measurement.
- **Interior-stability flags**: an erosion-sweep analysis (fold_degeneracy
  subtask, 2026-08-11) found the interior mean contrast of 99234, 99285
  and 99236 does not converge under mask erosion, and 99242 is marginal.
  The flags are recorded as masks next to the anchor data but are **not
  applied** to this calibration: dropping them discards every anchor above
  60 nm, and the calibration deliberately keeps the new-regime data.

### Past-fold flakes (aliasing)

Round 2 included two flakes far past the model's G-fold: 99226 (AFM
927–930 nm) and 99227 (AFM 626–628 nm). Their measured contrast projects
onto this table at **100.2 nm and 103.6 nm** — the 103.6 read falls
outside the (50, 100) tier-1 window, the 100.2 read does not. Very thick
hBN can therefore land in tier 1 with a ~100 nm label. Policy: RG contrast
cannot resolve this; the human review step checks for the visible shadow a
multi-hundred-nm flake casts, and Toghrul screens AFM candidates for it.

## Tier-1 thickness window

`tier1_thickness_window_nm = (50.0, 100.0)` on the preset (generic
`DetectorConfig` field, `None` = no window). Tier 1 requires the projected
`thickness_nm` inside the window (inclusive); out-of-window flakes keep
their honest thickness and fall through to the tier-2 check. Band edges are
explicit policy, not table-extent accidents. Table quantization at the
edges behaves: the 50 nm point rounds to 50.0 (in), 102 nm → 101.9 (out).

## Why point-based (RGPointDetectorConfig), not a curve

The band is an arc that doubles back in G: G peaks at +5.09 near 64 nm, so
`R = poly(G)` is multivalued over the band and `CurveDetectorConfig`'s
G-parameterized thickness interpolation would produce garbage. The arc does
not self-intersect in 2D, so nearest-point distance over (R, G) recovers
thickness correctly. Quantization error from the 2 nm spacing is ≤0.08
(chord half-length), negligible against the 0.30 tier-1 gate.
`layers = round(t_nm / 0.333)` is literally the hBN layer count;
classification reports nm (`classify_nm=True`).

## Capture settings live on the material

The material config carries its own capture settings; the pipeline resolves
**CLI flag > material > ScanPreset** for the chip scan and
**CLI flag > material `revisit_capture[mag]` > FC_DEFAULTS** for revisits.
The `find-flakes` startup banner reports which source won each value, so a
GUI-triggered run (which passes no gain flags) picks these up automatically.

| Step | Setting | Why |
|---|---|---|
| Chip scan | gain **2.0**, 0.25 ms | Band must be measured unclipped (below) |
| 50x revisit | gain **1.0**, 1 ms | FC default (1.5 / 2 ms, bg ~82) clips flake cores to solid white; at 1.0 / 1 ms bg ~27, 0% clipped, interior terraces resolved (`revisit_50x_hbn50100_v3`, 2026-08-04) |

Revisit JSONs written by the pipeline embed the material name
(`"material"` key); standalone `sls revisit --points` resolves the same
per-mag capture settings from it, with explicit `--gain`/`--exposure-ms`
always winning. Mags without a `revisit_capture` entry use `FC_DEFAULTS`
(`leica/autofocus.py`), resolved per objective at capture time.

### Why chip-scan gain 2.0

The per-channel clip ceiling is exactly `(255 − bg) / bg`, where bg is the
per-frame background mode — so it moves with gain **and with the ±10%
per-chip background spread**. A 5-gain revisit sweep (gains 1.0–3.0,
0.25 ms, 9 characterized points, 2026-08-04) established:

- At gain 2.0 the G ceiling is 7.6–8.4 across the observed chip background
  spread; the entire measured population (R up to 6.6, G up to 5.8) stays
  unclipped on every chip with margin.
- Measurement repeatability is ±0.1 in R/G — small against the 0.30 cal
  gate.
- Gain 2.5 is knife-edge (G ceiling ~5.9 on bright-background chips vs
  G 5.8 objects); gain 3.0 demonstrably pins them. Gain 4.0 (the ScanPreset
  default, G ceiling ~3.6) pins every in-band flake into a
  background-dependent ceiling corner that a fixed detector box cannot
  track — on a 6-chip run the fixed corner was reachable on only 1 of 6
  chips.
- The February gain-2 runs (`run_20260221_1651`, `run_20260409_1721`)
  validated the regime end-to-end.

Saturation/ceiling detection is deliberately not attempted: clipped
captures are fixed by capture settings, not by scoring.

### Gain couplings in segmentation (absolute counts)

Two segmentation parameters are absolute pixel counts, not contrast, so
their effective behavior shifts with capture gain:

- **Dark-frame guard**: `dark_frac_cutoff` skips a frame when >5% of
  pixels have mean brightness < 30 counts. At gain ≲1.5 (0.25 ms) a normal
  on-chip frame reads as ~98% "dark" and is skipped wholesale. At gain 2.0
  backgrounds sit well above the threshold — proven fine (Feb runs + the
  gain-2 sweep segment normally).
- **`contrast_offset = 15`** is counts above the background mode, so the
  effective contrast floor is `15 / bg`: ~0.27 at gain 4 backgrounds but
  ~0.55 at gain 2. Band flakes have contrast ≥ ~2.7 and clear it with a
  5× margin; materials with faint contrast would not. Check both couplings
  before running any material below gain ~2.

Per-frame background modes are recorded in the seg `summary.json`
(`bg_mode_by_frame`, R/G/B counts per frame) — giving downstream consumers
the local clip ceiling, a substrate/oxide fingerprint, and lamp-drift
visibility.

## Scoring (`_score_hbn_thick_50_100`)

Single tier-1 path for the unclipped gain-2.0 regime:

- **Tier 1**: `cal_dist < 0.30` to the arc, projected thickness inside the
  50–100 nm window, R ∈ [2.9, 4.3), G ∈ [2.7, 5.4), perim_ratio < 1.5,
  aspect < 6, size ≥ 500 µm². The R/G box predates this table's arc, whose
  G-peak (5.32) + 0.30 gate reaches past the box's G upper bound: on
  run_20260804_1303, 25 of 575 otherwise-T1 detections (dedup) fall in
  G ∈ [5.4, 5.65) and demote to tier 2 on the box alone.
- **Tier 2**: `cal_dist < 0.60` + shape/size (includes out-of-window flakes
  that otherwise pass tier-1 gates).
- Entropy gates disabled (99.0): band flakes have every pixel contrast
  above +1, so the fixed-range (−1, 1) `_hist_entropy` histogram is empty
  and entropy evaluates to exactly 0.0.

Score: `log2(size)² × exp(−2·cal_dist) × aspect_penalty`. The cal weight is
gentle (`−2`, not the `−8` used by thinner presets) because the band spans
~1.0 in R and ~1.5 in G — an `exp(−8·cd)` weight would rank by model-fit
noise, not flake quality.

Gate placement evidence: the cal_dist distribution inside the R/G+shape box
peaks at 0.05–0.20 and declines smoothly through 0.30 with no cliff — 0.30
sits comfortably above quantization (≤0.08) and repeatability (±0.1)
without a natural boundary to prefer.

## Gain invariance of contrast

Per-channel contrast is gain-invariant while unclipped. Verified
empirically on hbn_medium tier-1 populations (unclipped by their G < 3.0
gate):

| Run (gain) | n | R median | G median |
|---|---|---|---|
| run_20260409_1721 (2.0) | 606 | −0.555 | +0.601 |
| run_20260221_1651 (2.0) | 267 | −0.558 | +0.643 |
| run_20260505_1616 (4.0) | 101 | −0.546 | +0.687 |

The gain-2↔gain-4 gap is within the run-to-run scatter of the two gain-2
runs.

## Offline validation (scan-wide rerank, dedup 50 µm)

**Cross-material reranking requires `--reclassify`**: stored
`cal_dist`/`thickness_nm`/`layers` in seg output are computed against the
seg-time material (hbn_medium for these runs), and
`rerank_detections.py --reclassify` recomputes the cal projection against
the target material before scoring. Without it, curve gates score stale
distances to the wrong calibration.

| Run (gain) | detections (dedup) | T1 |
|---|---|---|
| run_20260804_1303 (2.0) | 9 408 | 550 |

T1 thickness distribution (nearest-table-point projection, 10 nm bins):
183 / 55 / 144 / 72 / 71 / 25 across the 50s / 60s / 70s / 80s / 90s /
100 nm bands. Top ranks are large, low-cal_dist (< 0.10) flakes on the
G-peak segment of the arc. Rerank outputs under
`<run>/rerank/v4_zackfit/`.

## Impostor posture

Thick hBN genuinely wraps back near the band's (R, G) arc and cannot be
separated by contrast. Under this calibration the model's ~200 nm point
sits 0.30 from the arc (reading ~52 nm), ~230 nm at 0.37 (~101 nm), and
~500 nm at 0.08 (~102 nm); the two measured past-fold flakes (627/928 nm)
read 103.6/100.2 nm (above). Impostors are accepted: the preset is tuned
for low false negatives; the human review (shadow check) + AFM screen
rejects them downstream.
