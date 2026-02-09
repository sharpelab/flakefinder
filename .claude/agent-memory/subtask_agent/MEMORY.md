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

## Facade Migration Pattern
- Replace `LeicaConnection` + manual subsystem init with `Microscope() as scope`
- `scope.light_on()` replaces separate shutter.open() + lamp.full()
- `scope.camera` triggers lazy init; `scope.acquisition` delegates to camera's own interface
- `scope.create_acquisition_context()` replaces raw `Extensions.UCAPI...SystemMemoryFactory`
- `continuous_autofocus` still takes raw `(conn, camera, acquisition, context)` — pass `scope.conn`, `scope.camera`, `scope.acquisition` until Step 3
- Prefer explicit keyword params over passing opaque `args` namespace to library functions

## Nosepiece Switching
- Use `Microscope.switch_objective_pos(pos)` or `switch_objective_mag("20x")` — handles z-speed maxing internally.
- `Nosepiece.set_position(pos, z=z)` is deprecated. Legacy scan scripts still use it (Step 2 migration).

## Scope Discipline
- **Only do what was explicitly approved.** "go for X" means X only — do NOT batch in additional scripts/files without asking. Stop after each approved unit and check in.

## Summary Files
- Keep summaries to 2-3 sentences max. Root cause, fix approach, sync status. No technical details like method signatures or try/finally mechanics.

## Linting
- Pre-commit hook handles ruff check + format (`git config core.hooksPath hooks/`)
- Line length limit: 120 chars. Break long f-strings into multi-line or intermediate vars.
- Ruff catches unused variables — don't create lookups/dicts you never reference.

## Microscope
- Do NOT run hardware commands (scans, autofocus, stage moves) — only the main scan session does that.
- SSH reads, git pulls, and file checks are fine.
