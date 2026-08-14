# Golden Blank-Chip References — 2026-08-13

Background references captured on a blank 90 nm SiO₂ chip immediately after the
[2026-08-13 illumination incident](illum_incident_20260813.md) light path was
verified restored. They anchor the scan-time
[background sanity check](illum_sanity_check_plan.md) and constrain the
illumination side of the contrast forward model.

Validation tool: `scripts/blank_refs_validation.py` (tables below;
`--plot-dir` for figures).

## The data

| File | Space | Contents |
|------|-------|----------|
| `calibration/blank_refs_20260813.json` | scan (WB → 5800K CCM) | stage 1: 5 WB presets at 10x · stage 2: gain/exposure invariance pair · stage 3: per-objective (2.5x–50x) at hBN scan WB |
| `calibration/blank_refs_raw_20260813.json` | raw (unity WB, identity CCM) | per-objective (2.5x–50x), gain/exp recorded |

Key values (10x): scan-WB rg 1.010, bg 2.044; raw rg 0.738, bg 0.628.
Raw R/G is objective-invariant (0.738–0.759); raw B/G falls with NA
(0.640 at 2.5x → 0.523 at 50x).

Known limitations of the capture (single chip, single session, one shot per
condition; stage-1/2 entries carry no gain/exp/objective/lamp metadata;
`wse2_legacy`'s nominal WB gains are undocumented; capture script lives in the
uncommitted `experiments/restore_illum_20260813/`).

## Validation results (2026-08-14)

- **Gain/exposure invariance confirmed**: ratios move ≤0.005 across the 4×
  gain + 4× exposure difference between scan and focus-map settings.
- **Cross-space closure at the 1–2% level**: inverting the measured 5800K CCM
  (`ccm_models.json`) recovers each stage-1 preset's nominal WB gains to 1–3%,
  and pushing measured raw values forward through hBN WB + CCM reproduces the
  stage-3 scan-space table to ~0.01 (R/G) / 0.03–0.06 (B/G) with a consistent
  sign — the CCM fit is the residual's likely source.
- **Repeatability**: the 10x scan-WB condition repeats 4× across stages with
  spread rg 0.003, bg 0.012 (includes inter-stage lamp drift, measured at
  0.4–0.6% uniform across channels).
- **`wse2_legacy` is a near-duplicate of `graphene_thick`** (same WB up to a
  G gain of ~1.02); redundant as a calibration anchor.
- **Sanity-check bands**: on scan 290 (known-good), per-chip
  `bg_mode_by_frame` medians sit at Δ(−0.010, +0.025) for chip 0 and
  Δ(−0.062, +0.149) for chip 1 vs the golden 10x anchor — chip-to-chip
  (substrate) variation is the dominant benign term. The landed warn bands
  (0.08 / 0.20, `segmentation.py`) contain it with margin; the originally
  drafted 0.06 / 0.15 would have warned on chip 1.

## Common-path correction ε(λ)

The ColorChecker pipeline chains (sister-scope lamp trace × T²_obj,
`colorchecker_cal.derive_pipeline`) miss the golden raw ratios by a large
offset (rg −0.12, bg −0.08 at 10x) that is **objective-independent**: the
relative cross-objective structure agrees to ≤3% (5x–50x) with no fitting.
The model error therefore lives in the **common path** — filter cube (R·T,
double pass), collector optics, tube lens, and/or lamp-trace error — not in
the per-objective T² curves.

`colorchecker_cal.derive_blank_ref_tilt` solves a smooth 2-parameter tilt
ε(λ) on the lamp trace so the 10x chain × SiO₂(90 nm, NA 0.25) reproduces the
golden raw 10x ratios exactly (ε rel. 540 nm: ×1.17 at 460 nm, ×1.34 at
620 nm — the real chain is redder *and* bluer than lamp×T² predicts). Fit
only at 10x, it transfers to 5x/20x/50x within 0.003–0.025 in raw ratios.
2.5x is the outlier (+0.10 in R/G): the ColorChecker 2.5x fingerprint is
suspect, not the blank data.

**Status: derived, not integrated** (decision 2026-08-14). A smooth
common-path tilt largely cancels in contrast ratios, so the contrast arcs
barely move; ε matters for absolute backgrounds and cross-objective
consistency. Callers opt in via `chain_spd * derive_blank_ref_tilt(...)`.

Caveats: only 2 dof of the common chain are constrained (channel-ratio
projections — any spectrum matching both ratios is admissible); the fit is
anchored through the 90 nm SiO₂ transfer-matrix reflectance, so
oxide-thickness error aliases into ε.

## Filter cube status

The IL turret's cube inventory is not in software: the stand config
(`DM6M-Z-saved.xml` / LAS X component data) lists positions 1–4 as free-text
`ICR` / `BF` / `DF` / `EMP` with every spectral field (filterset,
beamsplitter, EX/DC/EM bands) empty — no part numbers, no catalog data.
Identifying the BF reflector requires the engraved label on the physical
cassette or a Leica inquiry with the stand serial. Until then the cube is
characterizable only inside the lumped ε; the blank-ref fingerprint monitors
it (the incident signature — R −21%, B +13% at constant G — was exactly a
common-path tilt) without attribution.

## Follow-ups

Next time a blank chip is on the stage (cheap then, impossible later):

- **Bare-Si raw capture** (same identity-CCM per-objective set) — removes the
  SiO₂ reflectance model from the ε anchor, decoupling oxide thickness from
  the illumination chain.
- **Second/third blank 90 nm chip** — chip-to-chip spread is the number the
  sanity-check bands actually need.
- Repeat frames (3–5) per condition with std; full metadata in the JSON
  (objective, gain, exposure, lamp, WB as-applied, CCM, chip ID, timestamp).
- Non-hBN presets at 20x (currently 10x-only) if those materials ever scan
  at 20x; second lamp level within the no-PWM-aliasing regime.

Light-path classification (shared with the ColorChecker campaign's open
items — see `docs/leica_light_path.md`):

- **Identify the BF reflector** (physical label / Leica inquiry); if
  identified, enter the cube inventory into the stand config so it becomes
  SDK-visible for future forensics.
- **Spectrometer at the sample plane** (Elijah's instrument): measures
  lamp × collector × cube-reflection in one pass on *this* scope — the only
  route to separating ε into its parts, and it would replace the sister-scope
  lamp trace.
- **2.5x characterization**: both the ColorChecker fingerprint gap and the
  ε-validation outlier point at 2.5x; re-measure its white fingerprint.
- Lamp-SPD spectral modeling still has ±10–28% structured residuals (chart
  formulation vintage vs LED-trace accuracy not yet separable); raw-space
  card fingerprints are directly measured only at 10x.
- **CCM refinement**: after ε, the measured 5800K CCM is the next-weakest
  link (~0.01 rg / 0.05 bg systematic in scan-space closure).
