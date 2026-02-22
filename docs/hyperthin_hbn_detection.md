# Hyper-Thin hBN Detection

Working doc for detecting ultra-thin hBN flakes. Aaron is interested in these for his next project.

## Background

Discovered during 50x revisit review of run_20260221_1651, chip 6, frame_0212 d13. These flakes are nearly invisible at 20x but show up at 50x as large, highly homogeneous regions with very low contrast against the substrate.

## Reference Sample

- **Run**: `scans/run_20260221_1651/` (SF119 A-H, 90nm wafer)
- **Chip**: 6 (website Chip_7)
- **Detection**: frame_0212 d13 (website flake ~43283, scan 133)
- **Stage position**: X≈28122, Y≈65049
- **50x revisit**: `chip_6/revisit_50x/rank04_frame_0212_d13_50x.png`
- **Enhancement**: brightness 316%, contrast 209%, saturation 28% makes features visible

## Challenges

1. **Invisible at lower mags**: Almost impossible to see at 20x with any filter/enhancement. Current pipeline scans at 10x — these flakes are undetectable at that magnification.

2. **Requires 50x scanning**: Would need a 50x scan pass to detect. This is ~25x slower per unit area than 10x (FOV ratio: 263×175 µm vs 1313×876 µm). Only practical on targeted regions or small chips.

3. **Tape residue confusion**: At 50x, hyper-thin hBN appears as large homogeneous regions — visually similar to tape residue. Need to understand the distinguishing characteristics:
   - Color/contrast signature on 90nm SiO2?
   - Edge morphology differences?
   - Spectral (RGB channel) separation?

4. **Substrate matters**: Ultra-thin hBN is typically searched on **70nm SiO2** (not 90nm) for increased contrast via thin-film interference (per Sandesh). Detection at lower magnifications may be feasible on the correct substrate.

## Detection Strategy Ideas

- **Two-pass approach**: Run normal 10x scan for medium/thick, then targeted 50x scan on promising regions for thin
- **Contrast enhancement in detection**: Apply brightness/contrast boost before segmentation at 50x
- **Substrate-relative color**: Hyper-thin hBN on 90nm may have a specific B/G or R/G signature vs tape

## Next Steps

- [ ] Capture more examples at 50x on this sample
- [ ] Compare hyper-thin hBN vs tape residue at 50x — identify distinguishing features
- [ ] Discuss with Aaron what thickness range he's targeting
- [ ] Prototype 50x scan mode (much smaller FOV, need different scan parameters)
