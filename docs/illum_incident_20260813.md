# Illumination Incident — 2026-08-13

Two graphene scans with byte-identical recorded settings produced very
different substrate backgrounds: run_20260807_1630 (scan 290, operator
Jordan) vs run_20260813_1125 (scan 296, operator Elijah). Investigated
2026-08-13; root cause narrowed to an illumination-path spectral change
but the specific mechanism was not identified.

## Timeline

| When | Event |
|---|---|
| 8/04–8/11 | All runs at shared WB (2.51,1.02,1.41) sit on a stable background baseline: R/G 0.96–1.00, B/G 2.07–2.15 |
| 8/11 ~13:15 | Ben's runs end — last pre-change data |
| 8/11 15:46–15:52 | Microscope PC reboots twice (Windows Update, automated; no mechanism to touch the stand) |
| 8/13 10:56, 11:25 | Elijah's hBN + graphene runs — first post-change data: R/G 0.82–0.84, B/G 2.43–2.47 |
| 8/13 afternoon | SDK probe: all element positions normal (IL turret 2, FD 5, DIC 4, AP 11, method IL-BF) while the deviant colour state presumably persisted |
| 8/13 evening | LAS X launched (first time since Jul 29); blank chip placed on stage; first measurement reads R/G 1.00, B/G 2.07 — deviation gone |

## Evidence the change was in the illumination path

1. **Fixed-settings background shift is a spectral tilt.** At identical
   WB/gain/exposure/lamp: bg G ~constant (58→55 DN), R −21%, B +13% —
   "bluer and less red light, same green flux".
2. **Flake contrast excludes substrate thickness.** Per-layer graphene
   contrast in R and G is unchanged between runs 290/296 while blue
   contrast grew ×2–3 at every layer count. Transfer-matrix modelling:
   no SiO₂ thickness (70–115, 285, 300 nm checked) can raise bg B/G and
   blue contrast together; 300 nm additionally predicts +63% G
   brightness (not observed) and near-zero blue contrast (opposite of
   observed). Unchanged R/G contrasts also imply the interference
   condition — i.e. the oxide — did not change.
3. **Camera side exonerated.** WB/gain/exposure/gamma logged identical;
   CCM pinned at init and read back live (5800K both runs); an
   identity-CCM flip moves B/G the opposite direction (measured in the
   v2eval identity rescore).

## Exclusions

- **LAS X**: zero launches Aug 11–13 until the evening of 8/13 (all
  Leica logs empty in the window).
- **PC reboots**: Windows Update–initiated; no Leica process runs at
  boot (shared stand driver log untouched Jul 29 → 8/13 evening).
- **SDK-visible element positions**: identical between the deviant
  state and the restored state; element sweeps (IL turret, IL FD 1–6,
  DIC 1–4, IL AP) reproduce nothing like the signature.
- **IL-DF**: pitch black on flat substrate — deviant scans had normal
  brightness.
- **Oxide thickness / 285 / 300 nm stock**: see evidence 2.

## Open questions

- **Mechanism unidentified.** Surviving candidates: (a) a manual element
  in the IL path (slider/filter/analyzer — invisible to the SDK, no
  logs), removed before the evening measurements; (b) per-method stored
  stand state not readable via BasicControlValue, rewritten by the
  LAS X launch.
- **Elijah's chips not formally cleared**: the restoration was verified
  on a different blank chip. Re-measuring one of the 8/13 chips on the
  known-good light path is the decisive remaining test (expected
  ≈1.00/2.07 if chips are normal).
- Report that Mihir used the scope for darkfield in the window is
  unverified; IL-DF itself is excluded, but a DIC/DF session is a
  plausible occasion for a manual element to be disturbed.

## Durable facts learned

- The stand **interlocks the IL turret to the active contrasting
  method** — direct BCV moves are refused; the method is the correct
  control handle.
- **Methods store per-method element settings** (observed: field
  diaphragm — IL-BF keeps FD 5, IL-DF keeps FD 6).
- `SetContrastingMethod` (PascalCase) works via the SDK; **IL-BF = 200,
  IL-DF = 202**; a single call restores turret + FD as a group.
- IL turret inventory: 1 = ICR (DIC reflector), 2 = BF, 3 = DF,
  4 = empty.
- Lamp dimming is PWM: below 100% drive, 0.25 ms exposures alias
  (75% ⇒ ~12× dimmer, ≤50% ⇒ black). Lamp 100% is the only usable scan
  setting at scan exposure.
- Background mode ratios are **invariant across gain/exposure**
  (≤0.005 over 4× changes) and **objective-dependent in B/G only**
  (NA effect: scan-space B/G 2.12 at 2.5x → 1.54 at 50x; raw-space R/G
  objective-invariant at 0.74–0.76).
- LAS X colour controls (saturation 33, its own RGB gains) are
  app-side; our camera init repins saturation/CCM/gamma, so LAS X
  residue cannot leak into scans.

## Defenses landed

- Full illumination-path logging in every `scan_meta.json` /
  `pipeline.log` / `checkpoint.json` (turrets, diaphragms, lamp switch,
  tube port, contrasting method).
- `scope.pin_illumination()` at every scan start: tube-port assert,
  method pin to IL-BF, aperture/FD pins, post-pin state verification
  (hardware-validated end-to-end).
- Golden background references on blank 90 nm SiO₂:
  `calibration/blank_refs_20260813.json` (per-WB-preset and
  per-objective, scan space) and `calibration/blank_refs_raw_20260813.json`
  (identity-CCM raw space).
- Background sanity check at scan time: see
  [illum_sanity_check_plan.md](illum_sanity_check_plan.md).

## Data artifacts

- `calibration/illum_units_probe_20260813.json` — full AHM unit-tree
  snapshot (probe: `scripts/experiments/probe_illum_units.py`)
- `calibration/blank_refs_20260813.json`, `calibration/blank_refs_raw_20260813.json`
  — analysis and validation: [blank_refs_20260813.md](blank_refs_20260813.md)
- `experiments/restore_illum_20260813/` (scope + local, uncommitted) —
  element-sweep frames and results (`scripts/experiments/restore_illum_state.py`)
