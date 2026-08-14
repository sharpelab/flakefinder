# Background Sanity Check — Plan

Catch illumination-path deviations that the SDK cannot see (the class
behind the [2026-08-13 incident](illum_incident_20260813.md)) by
checking measured substrate background colour against golden references
at scan time. Complements `pin_illumination()`: pins enforce every
*readable* state; this check validates the *optical result*.

## Design

Two layers, both comparing background mode ratios (R/G, B/G) against
expected values for the active (WB preset × objective):

**A. Gate — after each chip's focus map, before its chip scan.**
The focus map ends focused on the chip and already saves its best AF
image. Compute bg mode ratios from that image and compare. Within warn
band: proceed silently (ratios recorded). Beyond warn band: loud
console/pipeline.log warning. Beyond abort band: abort the run before
the chip scan starts.

Basis: bg ratios are gain/exposure-invariant (measured ≤0.005 across
the 4× gain and 4× exposure difference between focus-map and scan
settings — `blank_refs_20260813.json`, stage 2), and the focus map uses
the same WB + CCM as the scan.

**B. Monitor — in segmentation.**
Seg already computes `bg_mode_by_frame`. Add per-chip median ratios +
a `bg_check` verdict to `seg/summary.json`, with a console warning line
when a chip leaves the warn band. Catches mid-run drift; costs nothing.

## Thresholds

Anchors (10x, blank 90 nm SiO₂, validated light path,
`calibration/blank_refs_20260813.json`):

| WB preset | R/G | B/G |
|---|---|---|
| graphene_thin / hbn (2.51,1.02,1.41) | 1.012 | 2.032 |
| graphene_thick (1.7,1.0,1.4) | 1.080 | 1.137 |
| wse2_monolayer (1.2,1.0,1.6) | 1.434 | 0.638 |

Observed run-to-run wobble (8/04–8/11 history): ±0.04 R/G, ±0.08 B/G.
Incident-scale deviation: −0.18 R/G, +0.40 B/G.

Proposed bands (10x):

- **warn**: |ΔR/G| > 0.06 or |ΔB/G| > 0.15
- **abort**: |ΔR/G| > 0.12 or |ΔB/G| > 0.30

Other objectives get their own expected pairs (B/G is NA-dependent:
2.12 at 2.5x → 1.54 at 50x in scan space); chip scans run at 10x/20x
today, so 10x + 20x anchors cover the gate. The check must never apply
one magnification's anchor to another.

## Implementation sketch

1. Expected ratios + bands: per-material entries in `DetectorConfig`
   (keyed by objective mag), values from the calibration anchors.
2. Shared helper: reuse segmentation's bg-mode extraction (histogram
   mode, mean within ±5 counts) — one implementation, both layers.
3. Gate wiring: in `find_flakes` after the focus-map/analyze step,
   read the saved best-AF image, evaluate, act (warn/abort per bands).
4. Monitor wiring: per-chip verdict in `_run_chip_seg` summary +
   console line.
5. Everything measured is recorded in the run metadata regardless of
   verdict.

## Open decisions

- Abort enabled from day one vs warn-only burn-in period. (Leaning
  abort-capable immediately: incident-scale backgrounds produce garbage
  classifications, so the scan is wasted anyway.)
- Whether the substrate must be bare for the gate: focus-map best
  images are substrate-dominated on normal chips, but a heavily covered
  chip could bias the mode. Mitigation: mode extraction already ignores
  minority pixels; monitor layer (per-frame medians over the whole
  chip) cross-checks the gate.

## Validation plan

After sync: (1) rerun a freshly scanned known-good chip — gate must
pass and record ratios; (2) re-measure one of Elijah's 8/13 chips —
doubles as the incident's outstanding substrate-vs-light discriminator;
(3) synthetic trip test: temporarily tighten bands and confirm the
abort path parks the microscope cleanly.
