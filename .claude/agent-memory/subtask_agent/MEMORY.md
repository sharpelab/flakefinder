# Subtask Agent Memory

## Python Version Constraints
- **pythonnet only supports up to Python 3.13** (TypeOffset table goes 37→313). Python 3.14 crashes at `import clr`.
- Always verify hardware-touching deps before bumping Python versions. Dry-run (no hardware) can pass while wet-run fails.

## Code Sync Rules
- Subtasks run locally → local code is already current after they push. Never `git pull` locally after a local subtask.
- Only the **microscope** needs `git pull` after subtask pushes.
- If microscope has local changes from previous version, `git stash && git pull` is fine.

## Leica SDK Gotchas
- **Don't call `GetObject()` on the same SDK interface twice.** Redundant `get_interface_required` calls can interfere with existing sessions. Delegate to the object that already owns the interface.
- `Camera.__init__` handles UCAPI registration internally — callers don't need `Extensions.ExUCAPI.Register()`.
- Leica `ucapi logfile*.log` files look binary to grep (mixed line endings); they're ASCII — use `grep -a`. Don't waste time on UTF-16 decoding.
- **Scope-side probe/experiment outputs go in `experiments/`** (scope repo top level), not `tmp/`. Committed artifacts still land under `calibration/` in git.
- .NET property-info API: `prop.GetInfo()` → `MinValue()/MaxValue()/StepSize()` (ranges) or `NumOptions()/GetOption(i)` (enums). `scripts/experiments/probe_camera_pipeline.py` dumps the whole camera property space read-only (no light/motion; does camera `Init()`).

## K5C Color Pipeline (2026-08-06 investigation)
See `calibration/camera_probe_20260806/camera_pipeline_probe.json` + `/tmp/camera_raw_summary.md`.
- Camera streams raw BayerRG8 to host; debayer/CCM/saturation/gamma are host-side in ucBgapi.dll (deployed = 1.15.0.12509, the 2023.3 build).
- WB coupling matrix (wb_model.json) = per-channel camera gains (floor 1.0, **max 8.0**, step 0.1) passing through the CCM selected by `camera.colour_temperature` (options [UserDefinedMatrix, 4500, 5800, 6600]K, default idx 2).
- PIXEL_TYPE options are only BGR|MONO — **no raw Bayer via UCAPI** (dead end). PIXEL_DEPTH offers 8|12 — 12-bit needs uint16 converter work.
- K5C does NOT implement: black/white level, hue/sat correction, shading, noise reduction, continuous WB, hot-pixel props.
- SDK evidence on scope: `SDK_Tests/ARCHIVE/UCAPI_SDK_V2023.3.0.12509/` (ucapi.h, Programmer's Guide, `docs/Genstruct Files/K5C.html`, real driver logs in `bin/`).

## Microscope Facade API
- `with Microscope() as scope:` — owns connection, all subsystems, acquisition context
- `scope.camera` / `scope.acquisition` — lazy init on first access
- `scope.context` — shared acquisition context, lazily created, auto-disposed in `__exit__`. For non-streaming use.
- `scope.create_acquisition_context()` — creates a NEW context caller must dispose. For streaming/per-thread use.
- `scope.light_on()` / `scope.light_off()` — shutter + lamp combined
- `scope.switch_objective_pos(pos)` / `switch_objective_mag("20x")` — handles z-speed maxing internally
- `wait_all(handles)` — module-level function in `units.py`

## Scope Discipline
- **NEVER commit without explicit user approval.** Always propose the commit and wait for "go"/"commit"/"sync". Premature commits ship suboptimal code — review catches issues that need fixing first.
- **Only do what was explicitly approved.** "go for X" means X only — do NOT batch in additional scripts/files without asking. Stop after each approved unit and check in.
- **Every task needs its own approval cycle.** A "go" for task N does NOT carry to task N+1. Each `# Operator Task` continuation resets: propose → wait for go → implement → ask "commit/push/pull?" → wait for go → sync → write summary.

## New Script Review Checklist
- Check against patterns in existing scripts (commands/scan.py, commands/chip_scan.py) for conventions
- Arg naming: `--exposure-ms` not `--exposure`, `--objective-mag` as `type=str` not float
- Camera setup: explicit binning, exposure (/1000), gain, gain_rgb, gamma
- Stage moves: `wait_all(list(scope.stage.move_to_async(x, y)))` for parallel XY
- FOV: use `compute_frame_size_um()` from `data_utils`, not hand-rolled
- Magnification: use `desc.objectives` from `require_microscope_description()`, not hardcoded dicts
- Objective switching: `scope.switch_objective_mag(str)` / `scope.switch_objective_pos(int)` on Microscope

## Summary Files
- Don't restate the problem — the operator already knows it. Just describe the fix/change, any new CLI flags, and sync status.
- Keep summaries to 2-3 sentences max. No technical details like method signatures or try/finally mechanics.

## Flatfield Correction
- **One flatfield per objective/binning.** Vignetting is optical; exposure, gain, WB, substrate color are uniform multipliers that cancel in the ratio.
- **Per-channel means, not global mean.** `apply_flatfield()` in `scan_utils.py` is the single source of truth. Global mean shifts color; per-channel means preserve it.
- **Channel order matters.** Helper is order-agnostic but caller must ensure image and flatfield match. PIL loads RGB, cv2 loads BGR, numpy flatfields are saved as RGB. `segment_flakes.py` flips with `ff[:,:,::-1]`.
- Flatfield metadata (.json) lives alongside .npy — see `build_flatfield.py`.

## Image Convention (2026-02-10)
- All images in flakefinder are RGB uint8 numpy arrays (`RGBImage` type alias in `types.py`)
- SDK returns BGR; `sdk_image_to_numpy()` converts at the boundary
- Autofocus sharpness functions use `cv2.COLOR_RGB2GRAY` (correct for RGB)
- Analysis scripts load from disk with `cv2.imread()` (returns BGR independently)
- `commands/stitch.py` loads with PIL `Image.open()` (returns RGB)

## Linting & Type Checking
- Pre-commit hook: ruff check + format, then `uv run ty check` on staged files
- Line length limit: 120 chars. Break long f-strings into multi-line or intermediate vars.
- Ruff catches unused variables — don't create lookups/dicts you never reference.
- `type X = ...` (PEP 695) preferred over `X: TypeAlias = ...` (UP040 rule)
- **Imports**: Just add at the top of the block, let ruff isort handle placement.
- **ty** (Astral's type checker): `uv run ty check src/flakefinder/` — config in pyproject.toml
  - `[tool.ty.src] exclude` only works on directory scans, NOT direct file args
  - ty narrows through `if x is not None:` but NOT through intermediate bool variables
  - For type narrowing past None guards: assign to local var, check, then assign to self
  - Prefer union types (`IID | UCAPI_IID`) over `int` when widening — preserves `.name`

## SDK Protocol Notes
- BasicControlValueVelocity is an empty subclass of BasicControlValue (same methods, different IID)
- .NET wrapper uses PascalCase: GetControlValue, SetControlValue, MinControlValue, MaxControlValue
- All DM6M axes (X, Y, Z) have velocity — required in Axis.__init__
- SDK headers at ~/sharpelab/leica_sdk/AHM_SDK_V2020.3.3.10693/C++/include/

## X/Z Velocity & Measurement
- X motor runs ~0.27% slower than commanded (4987 vs 5000 µm/s). Z is accurate (+0.05%).
- `commands/chip_scan.py` measures actual X cruise speed via regression during lead-in, uses it for Z velocity.
- Per-frame velocity deltas (`np.diff(x)/np.diff(t)`) are very noisy — always use `np.polyfit(t, x, 1)` for cruise speed.
- Velocity converter native resolution: X ~0.0015 µm/s/step, Z ~0.0006 µm/s/step — quantization is negligible.
- `BasicControlValueVelocity` (IID 0x108) and `DirectedControlValueAsyncVelocity` (IID 0x114) share same velocity converter per axis.

## Deskew Void Contamination (2026-02-10)

At high scan speeds, `deskew_image` creates a large triangular void filled with edge-replicated pixels. These have full alpha and contaminate blending at high-contrast edges. Fix: zero blend alpha in the void triangle in `load_frame`, not inside `deskew_image` (because `putalpha` replaces the alpha channel after deskew). The void grows linearly: `|shear_px| * y / frame_h` pixels from the affected edge.

## Stitching Artifact Debugging

Generate 4 variants (±deskew × ±blend) and compare. This immediately isolates which subsystem causes the artifact. The `--no-deskew` and `--no-blend` flags exist for this.

## Position Smoothing (2026-02-16)

**Committed fix (c504bb2):** `smooth_frame_positions()` in stitch.py applies savgol (win=15, poly=3) to the ~63 Hz raw position poll stream per row, evaluates at frame t_start. Eliminates ~49 µm direction-dependent position bias (at 10 mm/s) from per-poll interpolation timing jitter. Default savgol_window changed to 0 (frame-level savgol now redundant). `--no-pos-smooth` flag to disable.

**Remaining issues:**
- Savgol edge effect at stationary→moving boundary creates artifacts in first few frames of approach side. Trimming stationary samples before smoothing doesn't help (just moves the edge effect). The committed version smooths over the full sample stream including stationary — this is the least-bad option so far but still has minor dark-rectangle artifacts on -X rows' right edge.
- Right chip edge deskew: +X departure frames have ~4400 µm/s velocity (decel) vs -X approach ~5050 (accel) → 1.7 px deskew difference at the chip boundary. Clamping velocity to cruise didn't visibly help. Root cause is likely the velocity estimation, not the clamp approach.
- X-blend alpha masks (`alpha_first`/`alpha_last`) are position-independent but processing order flips for -X rows. Investigation was inconclusive — swapping them made chip edge worse, not better. Needs more study.

## Frame Removal Test Script

`scripts/test_frame_removal.py` — creates symlinked temp directories with frame subsets and stitches each. Strategies: sim_99pct, uniform_thin, no_zero_dx, beat_freq. Uses symlinks to avoid copying JPEGs. Key finding: artificially removing frames from 100% data does NOT reproduce 99% scan degradation — the issue was position noise, not frame selection.

## Don't Read Images — Use `show`

Use `show <path>` to display images to the user. Do NOT use the Read tool on images — it wastes tokens.

## Git Discipline — Pre-Commit Checklist
1. **Check if MEMORY.md is dirty** (`git status`). If so, stage it alongside task changes. Every commit. No exceptions.
2. When handing off to another subtask for commit, explicitly mention MEMORY.md in the blurb if it's dirty.

## Root Script Reorg (2026-02-11)
- Phase 0 (AreaRect type) done in aab788f. Phase 1 (file moves) done in 2efbce9. Phase 2 (in-process pipeline) done (pending commit).
- All 7 pipeline scripts now in `commands/`: scan.py, stitch.py, find_chips.py, focus_map.py, analyze_focus_map.py, chip_scan.py, stage.py
- Each command exposes `run()` with typed kwargs; `main()` is CLI wrapper calling `run()`
- Hardware commands use `scope: Microscope | None = None` + `nullcontext(scope) if scope else Microscope()` pattern
- `find_flakes.py` calls `run()` in-process with shared Microscope connection (no more subprocesses)
- `archive/` exists for superseded scripts; ty overrides exclude `archive/**`
- `cli_utils.py` lives at `src/flakefinder/cli_utils.py` — import as `from flakefinder.cli_utils import ...`
- `analyze_scan.py` renamed to `scripts/analyze_chip_scan.py`
- When moving files with ty overrides, update the include path in pyproject.toml

## Quick Scan GUI
See [quick_scan.md](quick_scan.md) for full context. PySide6 stage viewer at `src/quick_scan/`. Phase 1 (canvas) and Phase 2 (live camera, controls, frame stamping) done. Phase 3 next (load existing scan data). Architecture doc at `docs/quick_scan.md`.

## Microscope
- Do NOT run hardware commands (scans, autofocus, stage moves) — only the main scan session does that.
- SSH reads, git pulls, and file checks are fine.

## Batch Capture (2026-08-06)

- `sls capture-series -o DIR --spec spec.json` — many captures in one connection,
  driven by a JSON spec (`defaults` + per-condition overrides). Use this for
  calibration sweeps instead of bash loops over `sls capture` (~3.5 s/frame of
  ssh + venv + SDK-connect overhead). `--dry-run` renders the plan, no hardware.
- **It never moves anything** (no stage/Z/objective) and rejects motion keys
  (`x/y/z/objective_mag/focus/...`) at parse time. Position the scope first.
- **Warmup frame after any settings change.** Batched single-shot `capture()`
  can hand back a frame acquired under the *previous* exposure/lamp — a fresh
  connection per capture masked this. Default `warmup_frames: 1` discards it;
  corruption would otherwise land on frame 1 of every condition.
- Spec `white_balance` is `{"r","g","b"}` — the codebase already has three
  orderings in play (CLI takes B,G,R; `DEFAULT_WB` is R,G,B; metadata writes
  `white_balance_bgr`), so an object is the only unambiguous form.
- Metadata records requested **and** hardware-read-back applied settings per
  condition. Read-back is what makes numbers re-derivable ("asked 10 ms" vs
  "camera held 10.02 ms").

## Rerank / mosaic workflows (2026-08-04)

- `scripts/rerank_detections.py` scan-wide mode (run dir with chip_*/seg/) is read-only;
  **single-chip mode rewrites summary.json in place** — use scan-wide for non-destructive eval.
- Downscaled mosaic label text is unreliable when Read as an image — verify R/G values
  against the printed top-N table, not the rendered labels.
- Detector gates are data-only and can't know capture gain: a "clipped-regime" gate box
  fires on unclipped runs too. Bound such boxes by what's physically reachable when
  actually clipped (e.g. contrast ceiling (255-bg)/bg) to avoid cross-regime pollution.
- **Stored cal_dist is the seg-time material's** — cross-material rerank must use
  `--reclassify`, which recomputes the cal projection against `--material`. Without it,
  curve gates score distances to the wrong calibration.
- Scan-wide rerank without `--name` writes revisit_t1.json (and mosaic) at `rerank/`
  root, overwriting prior files. Always pass `--name <tag>` to sandbox outputs.
- Per-material capture settings live on DetectorConfig (`chip_scan_gain/exposure_ms`,
  `revisit_capture` keyed by mag). Resolution: CLI > material > ScanPreset/FC_DEFAULTS;
  revisit JSONs embed the material name for standalone re-runs.
- **Reranked upload flow**: `sls upload` selects by STORED tier/score in chip
  summaries, so a rerank-then-upload needs single-chip rerank (`chip_N/seg
  --reclassify --no-mosaic --no-scatter`, rewrites summary.json in place) per chip
  first, then `uv run upload <run> --tier 1 --top N --material "<label>" --substrate
  90nm --name <distinct_name>`. Upload defaults substrate to 285nm — always pass it.
  Upload runs fine locally (.env auth); revisit images attach by (frame, det_id, mag)
  match, so old-rank revisit PNGs carry over automatically.
