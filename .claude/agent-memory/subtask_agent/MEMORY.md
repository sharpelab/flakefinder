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

## Microscope Facade API
- `with Microscope() as scope:` — owns connection, all subsystems, acquisition context
- `scope.camera` / `scope.acquisition` — lazy init on first access
- `scope.context` — shared acquisition context, lazily created, auto-disposed in `__exit__`. For non-streaming use.
- `scope.create_acquisition_context()` — creates a NEW context caller must dispose. For streaming/per-thread use.
- `scope.light_on()` / `scope.light_off()` — shutter + lamp combined
- `scope.switch_objective_pos(pos)` / `switch_objective_mag("20x")` — handles z-speed maxing internally
- `wait_all(handles)` — module-level function in `units.py` (not on `Stage`)
- `require_microscope_description(path)` — raises on failure (not `| None`)

## Scope Discipline
- **Only do what was explicitly approved.** "go for X" means X only — do NOT batch in additional scripts/files without asking. Stop after each approved unit and check in.
- **Every task needs its own approval cycle.** A "go" for task N does NOT carry to task N+1. Each `# Operator Task` continuation resets: propose → wait for go → implement → ask "commit/push/pull?" → wait for go → sync → write summary.

## Summary Files
- Don't restate the problem — the operator already knows it. Just describe the fix/change, any new CLI flags, and sync status.
- Keep summaries to 2-3 sentences max. No technical details like method signatures or try/finally mechanics.

## Image Convention (2026-02-10)
- All images in flakefinder are RGB uint8 numpy arrays (`RGBImage` type alias in `types.py`)
- SDK returns BGR; `sdk_image_to_numpy()` converts at the boundary
- Autofocus sharpness functions use `cv2.COLOR_RGB2GRAY` (correct for RGB)
- Analysis scripts load from disk with `cv2.imread()` (returns BGR independently)
- `stitch_area.py` loads with PIL `Image.open()` (returns RGB)

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
- `scan_chip_v2.py` measures actual X cruise speed via regression during lead-in, uses it for Z velocity.
- Per-frame velocity deltas (`np.diff(x)/np.diff(t)`) are very noisy — always use `np.polyfit(t, x, 1)` for cruise speed.
- Velocity converter native resolution: X ~0.0015 µm/s/step, Z ~0.0006 µm/s/step — quantization is negligible.
- `BasicControlValueVelocity` (IID 0x108) and `DirectedControlValueAsyncVelocity` (IID 0x114) share same velocity converter per axis.

## Deskew Void Contamination (2026-02-10)

At high scan speeds, `deskew_image` creates a large triangular void filled with edge-replicated pixels. These have full alpha and contaminate blending at high-contrast edges. Fix: zero blend alpha in the void triangle in `load_frame`, not inside `deskew_image` (because `putalpha` replaces the alpha channel after deskew). The void grows linearly: `|shear_px| * y / frame_h` pixels from the affected edge.

## Stitching Artifact Debugging

Generate 4 variants (±deskew × ±blend) and compare. This immediately isolates which subsystem causes the artifact. The `--no-deskew` and `--no-blend` flags exist for this.

## Don't Read Images — Use `show`

Use `show <path>` to display images to the user. Do NOT use the Read tool on images — it wastes tokens.

## Git Discipline
- **Always include memory file changes in commits.** If MEMORY.md is dirty, stage it alongside the task changes. Don't leave it as perpetually uncommitted.

## Microscope
- Do NOT run hardware commands (scans, autofocus, stage moves) — only the main scan session does that.
- SSH reads, git pulls, and file checks are fine.
