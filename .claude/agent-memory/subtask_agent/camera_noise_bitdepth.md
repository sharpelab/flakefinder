# K5C Bit Depth & Noise (2026-08-07)

Tooling: `scripts/experiments/analyze_raw12_benefit.py` (reruns the whole thing).
Probe capture: `scripts/experiments/probe_pixel_depth12.py`.

## 12-bit delivery — resolved

- `PIXEL_FORMAT_BGR36`: 3 x uint16 LE, **B,G,R** order, stride = W*6, **no row padding**.
  Decode is the 8-bit path with a dtype swap: `buf.view(np.uint16).reshape(H,W,3)[..., ::-1]`.
- **Genuine 12-bit**, not 8-bit shifted: low nibble uniform to 0.0004, zero missing codes,
  1374 distinct values vs 91.
- Mapping is exact: `u12 = 16.00*u8 + 15.2`. Slope 16.000 to 0.01%, residual flat vs level
  (no gamma/knee). 12-bit is the **same host pipeline rendered wider**, NOT sensor-raw Bayer.
- Offset/16 = **0.953/0.977/0.943 counts** = the black-level pedestal. Independent check on
  `ccm_models.json` pedestal_counts 1.02.
- **No dynamic-range gain at the top.** Fixed 16x scaling means 4095 <-> 255.9, so both
  depths clip at the same scene brightness. All four extra bits are at the bottom.

## What 12-bit buys

At the raw config (idx0 + unity WB) on real chip substrate, 0.25 ms:
levels R21.3/G29.6/B19.0 of 255; **8-bit costs ~1.4x in sigma** (R 1.41 G 1.40 B 1.36).
Roughly half the noise variance at that operating point is the quantizer.
hBN thickness precision (G, 90 nm oxide, t=5 nm, 25 px): 0.186 -> 0.133 nm.

**It is a raw-config win, not a B-channel win** — 12-bit helps all channels about equally.
B is merely worst in absolute fractional precision because it sits lowest.

## Noise-analysis methodology (reusable)

- **Never estimate sensor noise from spatial variance** — that is scene texture. Difference
  repeat frames of a static target; that cancels scene AND fixed-pattern, leaving temporal
  noise only. `sigma = sqrt(var(f2-f1)/2)`.
- Phase 2 `stage1/` is the good noise dataset: **lossless PNG, 3-5 repeats x 22 conditions**,
  levels 0-177, including `unity_idx0*` (the raw config) at 0.15/0.20/0.50 ms.
- **Sub-LSB regime breaks quadrature.** When analog sigma < ~0.5 LSB, `var_quant = 1/12` is
  invalid — `sqrt(sigma^2 - 1/12)` overstates the penalty and on covariances yields
  correlations > 1 (saw 1.023). Simulate the rounding response and invert it numerically.
  Naive quadrature gave 1.70/1.58/1.87 where the correct answer was 1.57/1.52/1.65.
- **CCM propagates noise linearly**: `Sigma_out = C Sigma_raw C^T` validated to 1.7% against a
  held-out condition, correlation structure included. The CCM *inflates* noise — the raw
  config is already ~10-25% quieter than the 5800K space before bit depth enters.
- **WB gain does NOT propagate as a digital multiplier** (predicting wbr2_idx0 from
  unity_idx0 overshoots 22%). Do not derive scan-config noise from unity-WB conditions.
  Closing this needs one capture-series condition at idx2 with the scan WB.
- **Raw-space noise is channel-correlated (~0.65)** and no CCM inversion removes it. Ruled
  out misregistration (flattest 10% of pixels identical) and lamp flicker (removing the
  multiplicative common-mode changes nothing). It is demosaic + 3x3 binning sharing sensor
  sites — upstream of the CCM.

## Storage (punted 2026-08-07, revisit before any 12-bit scan)

- **PNG has no 12-bit mode** (8 or 16 only), and **Pillow cannot write 48-bit RGB PNG at
  all** — only 16-bit grayscale. The scan save path is PIL, so 16-bit needs cv2/pypng.
- Per 1824x1216 frame: JPEG q95 0.24 MB / 0.005 s -> 16-bit PNG 5.41 MB / 0.129 s.
  **23x storage, 26x encode.** Per chip (755 frames): 0.21 GB -> 4.09 GB, 4 s -> 98 s.
- Zeroing the low 4 bits drops 5.41 -> 2.18 MB: **over half the file is the incompressible
  low nibble.** The bits carrying the win are exactly the bits that will not compress.
- Scan frames are currently JPEG q95 with **4:2:0 chroma subsampling** — chroma decimated
  2x2 before disk. Verified from the file's sampling factors.

## Open hardware asks (none run)

1. One capture-series condition at **idx2 + scan WB** — closes the scan-config noise gap.
2. Shutter-closed dark frame at both depths — is the +15.2 pedestal real, or does 8-bit
   crush ~1 count of black?
3. Deliberate over-exposure — confirm the 4095<->255.9 ceiling and blooming parity.
4. 12-bit frame-rate/readout timing — gates whether 12-bit is usable in `chip-scan`.
