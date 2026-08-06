# ColorChecker Phase 2 — Color Pipeline Resolution Plan

Continues `calibration/colorchecker_dm6m_20260806/` (session 1) using the camera-raw
investigation results (`calibration/camera_probe_20260806/camera_pipeline_probe.json`).
Camera-side facts now known: WB gains are per-channel, floor 1.0 / max 8.0 / step 0.1;
the 3×3 WB coupling comes from the host CCM selected by `colour_temperature`
(options [UserDefinedMatrix, 4500K, 5800K, 6600K] = indices 0–3, default 2);
PIXEL_DEPTH offers 8|12; PIXEL_TYPE offers BGR|MONO only (no raw Bayer).

Setup per stage: ColorChecker on stage, 10x, focused on the target patch, LAS X closed.
All captures via `sls capture-series` (no motion). Operator drives positioning.

## Stage 1 — CCM identification + WB decoupling (one series, white patch #19)

**Run**: position/focus white patch #19 at 10x (session-1 wb_model conditions:
exposure 0.5 ms, gain 2.0, lamp 100), then:

```
sls capture-series -o calibration/colorchecker_phase2_<date>/stage1 --spec calibration/colorchecker_phase2/stage1_ccm_wb_spec.json
```

20 conditions × 3 frames (+warmups), ≈2 min. Blocks within the spec:

| Block | Conditions | Resolves |
|---|---|---|
| idx2 baseline + WB trio | `base_idx2`, `wbr2/wbg2/wbb2_idx2` | Reproduces session-1 M columns under 5800K CCM — method/drift validation against wb_model.json (expect R×2 → effective 2.05/−0.218/−0.352 pattern) |
| sat-0 riders | `sat0_unity/sat0_scanwb/sat0_wbr2_idx2` | Precise luma weights (session-1 pending #3): sat0 output = luma; three WB points give three independent weight equations. Provisional (0.29, 0.52, 0.18) → measured |
| CCM sweep | `ccm_idx1`, `ccm_idx3` (unity WB) | White-patch signature of each matrix; with the idx0/idx2 blocks gives all four CCMs' effect on a known input |
| idx0 WB trio | `unity/wbr2/wbg2/wbb2_idx0` (+ 0.2 ms riders) | **The decision measurement**: under UserDefinedMatrix, does each WB knob scale only its own channel? Decoupled ⇒ off-diagonal responses ≈ 0 (vs −1.62 max under idx2). Also: is idx0 identity (unity-WB ratios match raw sensor fingerprint) or some stored user matrix? |
| Cap check | `unity/wbr4/wbr6/wbr8_idx0_e015` (0.15 ms) | Empirically confirms max 8.0 usable and linear (Probe A property range vs session-1 "cap in (2.51,4]" inference — that inference is now suspect; find what actually happened) |
| State restore | `drift_idx2` (last) | Confirms colour_temperature returns cleanly to index 2 and matches `base_idx2` (no hysteresis/drift) |

**Analysis** (immediately after, center-ROI channel means per condition — `roi_stats`
pattern from `scripts/colorchecker_analyze.py`):
1. `wb*_idx2` vs wb_model.json predictions → method valid today?
2. `wb*_idx0`: build M_idx0. **Decision: if M_idx0 ≈ diag, idx0 is the clean-capture CCM.**
3. `unity_idx0` ratios vs session-1 unity fingerprint (G/R 1.916, G/B 4.987 measured *under idx2*):
   back out the 5800K CCM numerically; solve 3×3 per matrix from the sweep block.
4. Cap block: R-channel response vs request 4/6/8 → linear to 8.0?
5. sat0 block: solve luma weights.

**Outputs**: `ccm_models.json` (per-index measured matrices + M_idx0), updated
`wb_model.json` validity note, luma weights. If idx0 decouples → new recommended
capture config: `colour_temperature: 0`, WB unity, per-channel exposure control.

## Stage 2 — 12-bit format recon (one capture, needs light)

**Run** (after Stage 1, same white patch; script sets lamp/shutter itself):

```
uv run python scripts/experiments/probe_pixel_depth12.py --lamp 100 --exposure-ms 0.5 --gain 2
```

Sets PIXEL_DEPTH=12, acquires ONE frame with a raw buffer handler (bypasses the
uint8 converter), dumps Format() metadata + buffer as .npy + uint16 interpretation
stats, restores depth 8.

**Resolves**: delivered byte layout (16-bit container? alignment? BGR order?),
whether values span 12 bits and stay linear, effective noise floor vs 8-bit.
**Gates**: `image_utils` uint16 support + 16-bit PNG save → then 12-bit becomes a
capture-series/scan option. (Session-1's 0.25 ms short-exposure regime and thin-flake
G-contrast precision are the beneficiaries.)

## Stage 3 — chart sweep under the raw config (motion between patches, operator-driven)

Once Stage 1 picks the config: revisit-style per-patch captures (like session 1's
gray-row matrix) of the 6 primary/secondary patches + gray row at 10x, each under
(a) scan config (idx2, scan WB) and (b) raw config (idx0, unity WB).

**Resolves**:
- Measured CCM in real units → post-hoc conversion between raw and legacy spaces
  (existing scan data stays interpretable; no rescan needed).
- Physics-clean per-channel inputs for the hbn_contrast widget (replaces hand-fit
  r_off/g_off with measured glare + known channel response — session-1 pending #4).
- Chart-vintage question (session-1 pending #5) benefits from unmixed channels when
  comparing to BabelColor reference spectra.

## Stage 4 (opportunistic, session-1 leftovers needing the chart)

- 2.5x gray/black cells + anything in the 150x turret sector (assisted moves; the
  turret-collision deferral from session 1).
- Empty-substrate specular glare floors happen on chips, not the chart — separate session.

## Decision tree summary

- M_idx0 diagonal → adopt idx0 config; scan WB becomes pure per-channel exposure balance.
- M_idx0 = stored junk matrix (not identity, not diagonal) → keep idx2, invert measured
  CCM in post (Stage 3 gives the matrix either way).
- 12-bit layout sane → implement uint16 path; else record dead end.
- Cap linear to 8 → delete the "(2.51, 4]" caveat from wb_model.json validity note.
