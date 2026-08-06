# ColorChecker reference data

`colorchecker_spectra_babelcolor.csv` — per-patch spectral reflectance
(380–730 nm, 10 nm step), BabelColor 30-chart average (2012,
`ColorChecker_RGB_and_spectra.xls`, sheet `spectral_data`).

**Vintage caveat**: this averages pre-Nov-2014 X-Rite charts. Our chart is a
Calibrite ColorChecker Passport Photo 2 (post-2014 formulation). Deltas are
small, smallest on the neutral row. For quantitative color work on the
colored patches, prefer Calibrite's published L*a*b* for the current
formulation.

Gray-row (400–700 nm mean) reflectance ratios vs white #19:
#20 0.651, #21 0.399, #22 0.212, #23 0.100, #24 0.0365.
White patch dips to ~0.42 reflectance at 400 nm (plateau ~0.93) — do not
assume spectral flatness for B-channel predictions; convolve.
