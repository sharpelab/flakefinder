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

- **Tier 1**: `cal_dist < 0.30` to the arc, R ∈ [2.3, 4.0), G ∈ [3.0, 5.1),
  perim_ratio < 1.5, aspect < 6, size ≥ 500 µm².
- **Tier 2**: `cal_dist < 0.60` + shape/size.
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

| Run (gain) | detections (dedup) | T1 | per-chip T1 spread |
|---|---|---|---|
| run_20260409_1721 (2.0) | 23 094 | 1 003 | 86–246 (all 8 chips) |
| run_20260221_1651 (2.0) | 61 259 | 1 173 | 107–243 (all 8 chips) |

Top ranks in both runs are large (10⁴–10⁵ px), low-cal_dist (mostly < 0.2)
flakes on both limbs of the arc (G ~3.3–3.5 thick end and G ~4.6–4.9 band
middle). Mosaics: `<run>/rerank/hbn50100_v3/hbn50100_v3.jpg`.

## Impostor posture

150–260 nm hBN genuinely wraps back into the band's (R, G) region (e.g.
230 nm sits at R +3.52, G +3.11) and cannot be separated by contrast —
the thick limb of the arc (G ~3.3–3.5) is where they concentrate. They are
accepted: the preset is tuned for low false negatives; the human review +
AFM screen rejects impostors downstream.
