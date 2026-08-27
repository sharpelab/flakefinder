# Background Sanity Check

Catches illumination-path deviations that the SDK cannot see (the class
behind the [2026-08-13 incident](illum_incident_20260813.md)) by
checking measured substrate background colour against golden references
at scan time. Complements `pin_illumination()`: pins enforce every
*readable* state; this check validates the *optical result*.

## Design

Three layers, all comparing background mode ratios (R/G, B/G) against
the expected pair for the active (WB preset × objective × substrate):

**A. Overview gate — after stitch + chip detection, before any per-chip
work.** Refined background mode over the union of detected chip bboxes
(`bbox_px`) in the stitched overview, compared at the overview
magnification. Earliest detection point: a deviant light path is caught
before any focus map or chip scan runs.

**B. Chip gate — DISABLED.**
Median refined-mode ratios across the focus map's saved best-AF images,
one per sample point. Contour points sample the chip *perimeter* and
several land off the chip, returning near-black frames (median RGB
~15/15/18). On run_20260826_1158 chip 3, four of five contour points
were dark, so the median was taken over darkness and B/G collapsed to
0.32 against a 2.03 reference — while the on-chip frames from the same
focus map read 0.914 R/G / 2.207 B/G, matching that chip's scan frames
(0.929 / 2.232). The failure is independent of the light path and shows
up on known-good pre-incident runs. Removed from the pipeline and from
`check_bg.py`; reinstating it means rejecting dark/off-chip frames
before taking the median, then re-deriving bands.

**B (historical spec) — after each chip's focus map, before its chip scan.**
Median refined-mode ratios across the focus map's saved best-AF images
(one per sample point; the median is robust to a partially covered
point). Basis: bg ratios are gain/exposure-invariant (measured ≤0.005
across the 4× gain and 4× exposure difference between focus-map and
scan settings — `blank_refs_20260813.json`, stage 2), and the focus map
uses the same WB + CCM as the scan.

**C. Monitor — in segmentation.** Per-chip median ratios over seg's
per-frame `bg_mode_by_frame`, written as a `bg_check` verdict to
`seg/summary.json`, with a loud marker on the `[seg chip N]` console
line when a chip leaves the warn band. Catches mid-run drift; costs
nothing.

Verdicts: within warn band → proceed silently (ratios recorded); beyond
warn band → loud console/pipeline.log warning; beyond deviant band →
a loud console/pipeline.log warning naming the band. The check never
halts a run; operators decide what to do with the
run. Every measurement is recorded (`bg_check` in `checkpoint.json` /
`seg/summary.json`) regardless of verdict. Checks are skipped, with the
reason recorded, when the run uses a custom CLI white balance or the
material×magnification has no reference.

## Thresholds

Bands (|Δ| of measured ratio vs reference, `segmentation.py`):

- **warn**: |ΔR/G| > 0.16 or |ΔB/G| > 0.29
- **deviant**: |ΔR/G| > 0.17 or |ΔB/G| > 0.30

The warn band clears observed good-path chip-to-chip substrate wobble
(worst measured good chip: Δ −0.065/+0.16 on the monitor, −0.082/+0.19
on the gate — run 290 chip 1, which can still trip a marginal warn);
the deviant band sits well inside incident-scale deviation
(Δ −0.19…−0.24 R/G, +0.34…+0.44 B/G across layers).

Anchors (`DetectorConfig.bg_reference`, keyed by objective mag; values
R/G / B/G from `calibration/blank_refs_20260813.json` — blank 90 nm
SiO₂ on the validated light path):

| Preset (WB, B/G/R) | 2.5x | 5x | 10x | 20x |
|---|---|---|---|---|
| hbn_medium, hbn_thick_90nm, hbn_thick_50_100_90nm, graphene_thin_90nm (2.51,1.02,1.41) | 1.0333 / 2.1153 | 1.0717 / 2.0656 | 1.0117 / 2.0319 | 1.0318 / 1.9027 |
| graphene_thick_90nm (1.7,1.0,1.4) | — | — | 1.0799 / 1.1375 | — |
| hbn_medium_285nm, wse2_monolayer_285nm, wse2_monolayer_300nm | — | — | — | — |

An anchor is never borrowed across magnifications (B/G is NA-dependent:
2.12 at 2.5x → 1.54 at 50x in scan space) or across substrates
(background colour is set by oxide interference). The wse2 WB pair in
the golden refs was measured on a 90 nm blank, so it does **not** apply
to the 285/300 nm substrates those presets scan — wse2 and the 285 nm
hBN preset are unchecked until refs exist on their substrates (see
Missing calibration data).

## Implementation

- `segmentation.py`: `BGReference` / `BGRatios` / `BGCheckResult`
  types, band constants, `evaluate_bg_check`, refined-mode extraction
  (histogram mode + mean within ±5 counts — matches the golden-ref
  extraction), and the three measurement helpers `measure_bg_stitch`,
  `measure_bg_images`, `measure_bg_frame_modes`. Per-preset anchors in
  `DetectorConfig.bg_reference`.
- `find_flakes.py`: overview gate after Detect Chips; chip gate after
  each focus-map analyze step. Every layer is reporting-only.
  All outcomes land under `bg_check` in
  `checkpoint.json`.
- Seg monitor: `_run_chip_seg` writes `bg_check` into
  `seg/summary.json` and appends a warn/deviant marker to the seg
  console summary line.
- `scripts/check_bg.py`: offline replay of all layers over completed
  run dirs (validation + post-hoc audit; no hardware).

## Offline validation (2026-08-14)

`check_bg.py` over run 290 (`run_20260807_1630`, validated light path)
and run 296 (`run_20260813_1125`, deviant light path), both
graphene_thin_90nm at 2.5x/10x:

| Layer | Run 290 (good) | Run 296 (deviant) |
|---|---|---|
| overview 2.5x | 0.972 / 2.127 → OK | 0.789 / 2.474 → DEVIANT |
| chip 0 gate | 0.976 / 2.051 → OK | 0.800 / 2.376 → DEVIANT |
| chip 0 monitor | 1.000 / 2.069 → OK | 0.821 / 2.429 → DEVIANT |
| chip 1 gate | 0.929 / 2.223 → WARN | 0.797 / 2.375 → DEVIANT |
| chip 1 monitor | 0.947 / 2.193 → OK | 0.818 / 2.473 → DEVIANT |

The deviant run trips the deviant band on every layer. The good run passes
everywhere except a marginal warn on chip 1's gate — that chip's
substrate reads off-baseline against all 8/04–8/11 history, which is
exactly the substrate-vs-light ambiguity the warn tier exists to
surface.

## Missing calibration data

The anchor set covers the default WB on 90 nm SiO₂ (2.5x–20x) and
graphene_thick at 10x only. Blank-chip captures that complete the set
(same procedure as `blank_refs_20260813.json` stages 1/3: capture per
WB × objective on a blank chip, refined-mode extraction;
`capture-series` can batch each grid in minutes):

1. **graphene_thick WB (1.7,1.0,1.4) on 90 nm blank at 2.5x, 5x, 20x**
   — enables the overview gate and 20x chip scans for
   graphene_thick_90nm (10x-only today).
2. **Blank 285 nm SiO₂: default WB (2.51,1.02,1.41) and wse2 WB
   (1.2,1.0,1.6) at 2.5x, 5x, 10x, 20x** — enables the guard for
   hbn_medium_285nm and wse2_monolayer_285nm (fully unchecked today).
3. **Blank 300 nm SiO₂: wse2 WB at 2.5x, 5x, 10x, 20x** — enables
   wse2_monolayer_300nm.
4. Optional: a 50x default-WB anchor already exists (1.0915 / 1.5407)
   if a revisit-time check is ever added; other WBs at 50x would need
   capture.

Requires one blank chip per substrate thickness (285/300 nm blanks not
yet in hand at the scope).

## At-scope validation (pending)

(1) rerun a freshly scanned known-good chip — gates must pass and
record ratios; (2) re-measure one of Elijah's 8/13 chips — doubles as
the incident's outstanding substrate-vs-light discriminator.

Bands are derived from `check_bg.py` replayed over every 2026-08 run.
The golden refs are blank-chip; working chips carry flakes and tape
residue that bias the background mode, so every real run sits offset
from the refs in the same direction — pre-incident runs on a known-good
path included. The bands clear that standing offset:

| layer | normal max R/G / B/G | 8/13 incident min R/G / B/G |
|---|---|---|
| overview 2.5x | 0.155 / 0.212 | 0.234 / 0.310 |
| seg monitor | 0.127 / 0.277 | 0.172 / 0.379 |

Under warn 0.16/0.29 and deviant 0.17/0.30: 118 normal measurements all
OK, and all 6 incident measurements flag deviant — the overview layer
included, so the earliest layer catches it.
