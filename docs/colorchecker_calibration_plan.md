# ColorChecker Calibration Plan

Plan for using a ColorChecker chart to derive per-channel WB / effective
illumination on the Leica DM6 M (and, eventually, cross-compare against
the lab's other scopes).

## Problem

The DM6 M's incident-light lamp is **Leica article 11504242** ("Lamp
housing LED DM6, cable long") — a phosphor-converted white LED. Leica
does not publish its SPD or CCT. Inferred CCT from the sister DM2700 M
spec is **~4500 K**, but unverified. The contrast widget
(`scripts/hbn_contrast_widget.py:47`) currently uses a 3200 K halogen
blackbody as a placeholder, which is wrong for this lamp. See
`docs/contrast_widget_todos.md` for the broader illumination-spectrum
TODO.

We don't actually need the SPD for the contrast model — we only need
the three integrals `∫ R(λ) S_c(λ) L(λ) dλ` for `c ∈ {R,G,B}`. A
ColorChecker imaged on a known-flat / known-reflectance target gives
those scalars directly via WB. `scripts/hbn_contrast.py:620` already
accepts this as `--wb-gains R,G,B`.

## Approach

Image an X-Rite ColorChecker (or equivalent) under a *strictly linear*
camera pipeline, on each scope and at multiple lamp intensities. Derive:

1. **WB gains** — single triplet that replaces the halogen placeholder.
2. **Linearity check** — gray-ramp response curve, confirms the pipeline
   is actually linear (gamma off, AGC off, saturation neutral).
3. **Per-channel effective lamp** — `(R, G, B)` channel integrals stored
   alongside the conditions they were measured under.
4. **Cross-scope spectral fingerprint** — same patch, same camera, on
   each scope → ratio of `(R/G, R/B)` is a direct comparison of the
   lamps without needing any spectrometer.

## Hardware checklist (bring to lab)

- ColorChecker chart (Zack has one)
- Flat dark surface (or matte black backdrop) to set chart on
- Lab notebook entry — date, scope, chart serial / patches imaged

## Capture protocol

Pin the camera pipeline linear and reproducible. From
`src/flakefinder/leica/camera.py:96-103`:

```python
camera.gamma = 1.0                  # SDK default 0.45 — must override
camera.gain_rgb = (1.0, 1.0, 1.0)   # SDK default R1.93/G1.00/B1.94
camera.auto_brightness = False
# saturation: neutral if SDK exposes it; otherwise note and proceed
```

Lamp / illumination:

- Köhler-align before capture
- Leave field/aperture diaphragms at the configured BF settings for the
  objective
- Use the **lowest-mag objective** that fits useful patch area. 1.25×
  panorama would fit the whole chart; 2.5× fits ~6–9 patches; 5× and
  above is one patch at a time

Sweep:

- Fixed `gain` (master), fixed `exposure_ms`
- Lamp intensities: **10 / 30 / 50 / 70 / 90 %** of full
- 3–5 frames per condition, average for noise
- Re-do at least one mid-intensity point at the start and end as a
  drift check

Things to avoid:

- Ambient room light (turn it off)
- Specular glints (chart flat, normal incidence)
- Saturated pixels in the white patch — pick exposure so #19 sits at
  ~70–80% of full scale at 50% lamp

Per-frame metadata to record:

- `objective`, `gain`, `exposure_ms`, `lamp_pct`, `gamma`, `gain_rgb`,
  `auto_brightness`, `timestamp`, `chart_id`, `scope_id`

## Analysis

For each frame, segment the patches (auto by chart geometry, or
hand-pick ROI per patch). For each patch:

- Mean `(R, G, B)` over the central 60% of the patch (avoid edges)
- Std as a noise / uniformity check

Outputs per scope:

| Quantity | Computation |
|---|---|
| `wb_gains` | `target/mean_white` per channel; `target = max(mean_white)` so the largest channel becomes 1.0 |
| `linearity_curve` | for the gray ramp (#19–#24): `(lamp_pct, mean_R)`, same for G/B, per intensity |
| `linearity_residuals` | fit a line, report RMS deviation per channel as a fraction of full scale |
| `effective_lamp_RGB` | the `(R,G,B)` triplet of the white patch at the reference intensity, used as input to `hbn_contrast.py --wb-gains` |
| `cross_channel_ratios` | `(R/G, R/B, G/B)` of #19 — the lamp fingerprint |

Optional (not required for the immediate hBN contrast use case):

- 3×3 sRGB → camera-native CCM fitted from the 18 colored patches via
  least squares. Useful for any color-reproduction work later, doesn't
  affect contrast modeling.

## Outputs / file layout

```
calibration/
  colorchecker_<scope_id>_<YYYYMMDD>/
    manifest.json              # protocol, gains, exposure, etc.
    frames/
      lamp10_001.tif           # raw frames
      lamp10_002.tif
      ...
    patches.json               # per-patch mean RGB at each lamp setting
    summary.json               # wb_gains, effective_lamp_RGB,
                               #   linearity residuals, fingerprint ratios
```

`manifest.json` must capture every camera/scope/lamp setting needed to
re-derive the same numbers from the same chart later.

## Integration with existing code

- `scripts/hbn_contrast.py` already takes `--wb-gains R,G,B` — the
  output of this calibration is the input to that flag.
- `scripts/hbn_contrast_widget.py:44-49` should grow an
  "illumination mode" selector (already in
  `docs/contrast_widget_todos.md`); add **"measured (WB-derived)"** as
  one of the modes, sourcing from `summary.json:effective_lamp_RGB`.
- `src/flakefinder/microscope_description.json` is a sensible place to
  record the canonical `wb_gains` / `effective_lamp_RGB` for the DM6 M
  once measured, so other tools (segmentation, scan-time WB) can pick
  it up by default.

## Cross-scope comparison

For each scope in the lab (DM6 M, plus whichever others Sharpe Lab uses
for flake imaging):

1. Repeat the protocol with the *same physical chart* and, ideally, the
   same camera. If the cameras differ, compare ratios rather than
   absolute counts.
2. Tabulate the fingerprint ratios `(R/G, R/B)` — large differences
   imply meaningfully different lamp spectra.
3. Plot the gray-ramp linearity curves overlaid; differences in slope
   ratio = WB difference, differences in shape = ISP / nonlinearity
   difference.

Useful sanity question: do scopes nominally documented at the same CCT
actually agree to within a few %? If not, the per-scope WB calibration
matters more than any "correct" CCT figure.

## Open questions / future work

- **Spectrometer measurement** — a fiber spectrometer (Thorlabs CCS200
  or similar) trace at the specimen plane is still the gold standard,
  and would let us cross-check the WB-derived effective lamp against a
  measured SPD × camera-response convolution. Worth doing once if
  borrowable.
- **Email Leica sales** — request the SPD / CCT for article 11504242
  directly. They sometimes share on request.
- **Chart aging / fluorescence** — ColorChecker patches can fluoresce
  under blue-rich LED illumination (especially the white patch). If the
  cross-channel ratios look weird, check this with a UV cut-off
  comparison.
- **Lamp warm-up drift** — record if the LED's output drifts in the
  first few minutes after power-on. Repeat the start-vs-end frame check
  after various warm-up times.
