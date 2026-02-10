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

## Summary Files
- Keep summaries to 2-3 sentences max. Root cause, fix approach, sync status. No technical details like method signatures or try/finally mechanics.

## Linting & Type Checking
- Pre-commit hook: ruff check + format, then `uv run ty check src/flakefinder/`
- Line length limit: 120 chars. Break long f-strings into multi-line or intermediate vars.
- Ruff catches unused variables — don't create lookups/dicts you never reference.
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

## Microscope
- Do NOT run hardware commands (scans, autofocus, stage moves) — only the main scan session does that.
- SSH reads, git pulls, and file checks are fine.
