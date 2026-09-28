
## Graphene Classification Follow-ups (from scans 290 vs 296, 2026-08-13)

1. **Upload git revision to flakes server.** Done on the server side (read-only
   `scan.meta` field, deployed, migration run) and committed in flakefinder as
   `fdbff27` (scan + upload git revisions, pipeline args, operator → `meta`; comment
   keeps operator words only). **Remaining: push local master + microscope `git pull`**
   — until then, scope uploads still write provenance into the comment with empty
   `meta`, and the already-run migration won't retro-clean them.
2. **Illumination pin + log — landed on local master (1e88652..a28dffa), NOT synced.**
   Full illumination logging in every scan_meta/pipeline.log/checkpoint;
   `pin_illumination()` (SetContrastingMethod(200) = IL-BF ensemble) wired into
   scan/chip_scan/focus_map/revisit and hardware-validated against the calibrated bg
   target (R/G 1.00, B/G 2.07); golden blank-chip references in
   `calibration/blank_refs{,_raw}_20260813.json`. Incident record:
   `docs/illum_incident_20260813.md` (mechanism unidentified — SDK-actuatable
   elements exonerated; manual IL-path element or per-method stand state; deviation
   self-resolved by evening). The bg guard is BUILT (`79a4b7c`: overview/chip gates +
   seg monitor, `scripts/check_bg.py` offline audit) — summary review pending.
   Remaining:
   - review bg-guard and golden-refs subtask summaries (neither written yet at
     session close; commits `79a4b7c`, `b501338`)
   - push + scope pull — first post-sync scan doubles as the logging/pin smoke test
   - re-measure one of Elijah's 8/13 chips on the good light path (decisive
     chips-vs-light discriminator; expected ≈1.00/2.07 if chips normal)
   - continue light-path investigation per Zack: filter cubes, and more blank
     substrate shots to fill out whatever the tilt/model work says is missing
3. **Revisit reclassification (the big gun).** Re-measure contrast from the 50x revisit
   captures, where flakes segment cleanly from bg. Fixes 100700 (2L read as 1L via 10x
   contrast dilution), 100704 (fiber defocus halo), torn/debris flakes, and Toghrul's
   graphene layer-count complaints. Improved bg estimation/subtraction rides along.
4. **Decide: tighten 10x tier gates vs leave junk to revisit reclassification.** Open
   question — do stricter gates (e.g. entropy, both bad uploads were within 1% of the
   3.7 gate) eat real flakes? Answer with data from 290's confirmed-good tier-1 set.
5. **Chip-edge margin (100699 fix).** Exclude detections within a margin of the
   detected chip boundary in stage coords — chip bboxes already exist from overview
   detection. Root cause: detection sat in a frame corner darkened by the adjacent
   off-chip region, and contrast vs the frame-global bg mode turns locally darker
   substrate into a phantom 1L. Tile/per-scanline local bg remains the fallback if a
   margin proves insufficient.
6. **B-channel sanity gate — re-examine, skeptically.** Historically never works out;
   B moved the most between 290/296 (bg B/G 2.07 → 2.47), so substrate/illumination
   shifts bite hardest here. Maybe better post identity-CCM. Evaluate offline against
   labeled data before wiring anything into gates.
7. **Refit graphene cal points under the current camera pipeline.** The scan-154
   (old-system) points are off-center even for Jordan's good run (1L cluster residual
   dG ≈ +0.06). Eyeball layer labels suffice for 1-3L (no AFM needed); AFM only for
   thicker material.
8. **Oxide-thickness sensitivity analysis.** Cal assumes one ±1% 90nm wafer batch.
   Quantify d(R,G)/dt_oxide for 1–5L graphene at 89/90/91nm with the transfer-matrix
   model and compare against the 0.06/0.08 cal_dist gates — know in advance whether a
   1–2nm batch shift silently breaks classification.
9. **Interior-anchor refit — RESTART the subtask (killed 2026-08-14 before starting).**
   Zack's eyeball interior picks for the scan-288 AFM flakes (3 picks/flake on
   FF-corrected frames, contrast vs frame bg mode) are preserved with provenance in
   `calibration/eyeball_interior_anchors_20260814.json`; the ready-to-use subtask
   prompt is `calibration/eyeball_interior_anchors_20260814.refit-prompt.md`
   (respawn: `tools/subtask-launch interior_refit -f <that file>`; worktree/branch
   already cleaned). Scope: land the anchor set in repo + widget overlay (own space),
   interior-space refit (n, offsets), verdict on the +11nm bias and >70nm fold shape;
   production preset out of scope; includes the widget fix to render erosion-unstable
   r2 stars hollow (the solid stars misled the v4 widget fit). Key findings feeding
   it: interiors sit systematically above the v4 arc even for uniform flakes;
   99285/99236/99242 DB means are heavily edge-diluted; 99234 has two thickness
   domains (~50nm + ~72nm, AFM 65-67 between — subseg averaged them); interior-space
   B channel carries usable thickness signal (3.1@55nm → 0.74@82nm); cross-mag
   ratio (R50/R10 ≈ 0.47-0.58 in-band, 0.32 for the 627nm past-fold alias 99227) may
   discriminate past-fold aliases → candidate revisit-reclassification feature.
   Extend with round-1 anchors: Zack plans the same eyeball pass on the prior Toghrul
   run (scan 287 / run_20260804_1354, local copy exists) — then widget work.
   Open question for Toghrul: 99280's blue region reads optically ~55-65nm vs AFM
   45-47 — did the AFM step land on bare SiO₂?
10. **Illumination path as an explicit stage in the forward model.** The pipeline
   *state* side is handled (categorization pinnable / log-assert / SDK-invisible is
   in the incident doc; pin + logging landed; sanity check guards the SDK-invisible
   class). Still open on the *model* side: `hbn_contrast.py`'s light chain is
   `lamp × objective T²` with no path-response term — the IL cube / path spectral
   response has historically been absorbed into r/g offsets and anchor refits, which
   is why path changes de-calibrate silently. Add a path term keyed to the pinned
   configuration; pragmatic form is a low-order spectral tilt fit to blank-chip bg
   ratios (golden refs are the anchor data). `b501338` ("derive common-path tilt")
   may already cover part of this — confirm when reviewing that subtask's summary.
11. **Toghrul: email notifications.** Requested 2026-08-14 (presumably scan
    completion/upload notifications — confirm what he wants notified about).
12. **Toghrul: combined hBN preset, 1–100nm.** Single scan covering the thin through
    thick bands instead of separate presets.
13. **Segmenter mixture-blob investigation.** Contours that enclose large background/halo
    fractions make the blob mean a contrast no pixel population has (mixture means can
    land anywhere, including on the cal locus). Exemplar: run_20260826_1158 chip_3
    frame_0230 det7 (stage 67092,38484) — 44% of enclosed pixels at background level,
    solidity 0.49, r_kurt −1.2, yet entropy 3.61 squeaks under the 3.7 gate; Otsu subseg
    did NOT split it despite r_std 0.157 ≫ subseg_min_std 0.05 — find out why. Mixture
    signature r_std>0.10 & r_kurt<−0.5 covers ~23% of current tier 1 (58/251 on that
    run), so a naive gate has real false-negative cost (genuine two-thickness flakes
    share the signature). Candidates: fix subseg triggering, dominant-mode contrast
    instead of blob mean, halo-aware contouring. Evidence: scan-nb 2026-08-27
    ("Polyline cal_dist subtask — measurements").
14. **Focus-map internal AF converges ~130-150 µm low.** 2026-09-01, 285nm chip 1:
    `sls focus-map` produced a self-consistent plane (R²=0.985, resid 1.9 µm) sitting
    130-150 µm below true focus at every point, with a correct z_start seed — while the
    standalone `sls autofocus` CLI found true focus at the same points. Every plane the
    tool has produced is suspect until root-caused; symptom implies a systematic bias in
    focus-map's internal AF path (not a seed or plane-fit problem). Evidence: scan-nb
    2026-09-01 (defocused verify capture at plane Z; three standalone AFs at
    24723-24732 vs plane prediction ~24600).

## Camera Streaming Performance

- Position tagging in frame callback causes fps drop: 20.5fps → 12.5fps
- Fix: Move stage.position_um calls out of _on_frame callback
- Options: cache positions, sample less frequently, or use async position events

## Camera Upgrade (IMX392 global shutter)

Replace the K5C with a machine-vision camera: global shutter (zero intra-frame skew vs
K5C's ~15ms rolling readout), µs hardware frame timestamps + external trigger (kills
capture-timing jitter), ~6-20µs exposure floor (sub-pixel motion blur at full stage speed,
gated by light budget — measure first). Full comparison:
https://wiki.sharpelab.science/doc/flake-finder-camera-options-cBW9BOI018

Candidates (both Sony IMX392, 1936×1216-class — same frame as K5C 3×3 binning):
- **FLIR Blackfly S BFS-U3-23S3C** — $421, USB3, 163fps, PySpin. First buy / test article.
  Mono variant (23S3M) for NIR experiments.
- **Allied Vision Alvium G5-240** — ~$600-700 (quote), 5GigE+PoE, 191fps, Vimba X.
- Mount: C-mount; needs adapter check on the DM6M photo port (1/2.3" sensor → 0.35x adapter).

### Optical flow as a velocity stream

With trustworthy frame Δt (hardware timestamps) and no rolling-shutter skew, phase
correlation between consecutive frames measures stage displacement directly: at 163fps and
10mm/s, ~61µm travel per frame against ~500µm+ FOV = ~90% overlap → sub-pixel displacement
at 163Hz. Frames become the position sensor; Leica polls (124Hz, ~16ms latency) demote to
absolute anchoring — drift correction and coverage over featureless wafer where flow goes
blind. Stitching becomes self-registering.


## Stage Specifications (verified in LAS X)

- **Max velocity: 40 mm/s** (measured ~43 mm/s, at max setting)
- **Acceleration: ~62 mm/s²** (derived from scan data)
- **Accel/decel distance: ~12-13mm each**
- **Accel/decel time: ~690ms**

Stage is already at max speed. Acceleration zones need to be discarded in post-processing.

## Color Calibration

Images appear green vs blue in LAS X. Need to:
- Check LAS X white balance / color settings
- Match gain_rgb values to LAS X defaults
- Add CLI options: --gain-rgb "1.0,0.8,1.3" or --white-balance
- Consider auto white balance option (PROP_AUTO_BRIGHTNESS_ENABLED or similar)

## Low-Contrast Flake Segmentation (Elijah's metal substrates)

Sample: graphene on 5nm Pt / 100nm Cu / 40nm Au — flakes are extremely low contrast against the metal background. Try to segment individual flakes without the segmentation collapsing into "everything is foreground".

- Test data: `~/sharpelab/archive/elijah_forzack/04_09_26_EC_1/` (P18–P52 overlay JPGs, stitched.jpg, flake_positions.csv with stage coords for ground truth)
- Things to try: per-channel contrast (R/G separately), local-window normalization, edge-based rather than threshold-based, comparison against the existing SiO₂ flake pipeline to see what breaks
