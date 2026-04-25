# Contrast Widget TODOs

Scratchpad for follow-ups on `scripts/hbn_contrast_widget.py`.

## Illumination spectrum

Currently hard-coded to a 3200 K blackbody (matches the halogen lamp the
AFM-referenced hBN cal points were measured under). The Leica's actual
lamp is an LED with a non-Planck spectrum.

- [ ] **Add an illumination mode toggle** — uniform / 3200 K / Leica-LED.
      Keep 3200 K as the default since the existing cal data is anchored
      there, but make it easy to switch and see the deltas.
- [ ] **Color-temperature slider** — e.g. 2500–6500 K, useful for
      sanity-checking how sensitive the predictions are and for fitting
      against measured points (parallels the `--fit-lamp` path in
      `scripts/hbn_contrast.py`).
- [ ] **Measure & integrate the Leica LED spectrum.** Need a
      spectrometer trace through the same optical path (or vendor data
      sheet). Once we have it, ship as a static array alongside the
      IMX183 sensor curves and add it as a selectable mode.
- [ ] **Understand the magnitude of each illuminant change.** The 3200 K
      shift produced ΔR≈−0.09, ΔG≈−0.17 at 30L hBN (~10 nm) but only
      −0.014 / −0.027 at 5L. Build intuition for which (oxide, n,
      thickness) regions are illumination-sensitive vs not.

## Other follow-ups

- [ ] Verify the plotted calibration points are *not* being
      illumination-shifted. They should be raw measured contrasts; if
      the widget transforms them through the illuminant, that double-
      counts.
- [ ] Confirm consistency with the CLI (`scripts/hbn_contrast.py`)
      — the CLI already supports `--lamp T` and `--fit-lamp`, but the
      widget historically ran with no illuminant. Once the illumination
      mode is wired up, the two should produce identical numbers for the
      same params.
- [ ] Consider exposing the dispersion model (n(λ)) for the materials
      in the UI so the user can see how much of the predicted curve
      shape comes from dispersion vs from oxide thickness.
