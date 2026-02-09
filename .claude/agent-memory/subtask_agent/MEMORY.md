# Subtask Agent Memory

## Python Version Constraints
- **pythonnet only supports up to Python 3.13** (TypeOffset table goes 37→313). Python 3.14 crashes at `import clr`.
- Always verify hardware-touching deps before bumping Python versions. Dry-run (no hardware) can pass while wet-run fails.

## Code Sync Rules
- Subtasks run locally → local code is already current after they push. Never `git pull` locally after a local subtask.
- Only the **microscope** needs `git pull` after subtask pushes.
- If microscope has local changes from previous version, `git stash && git pull` is fine.

## Nosepiece Switching
- Always use `Nosepiece.set_position(pos, z=z)` (not `.position = pos`) — it maxes Z speed to avoid SDK timeout during the internal z-hop.

## Summary Files
- Keep summaries to 2-3 sentences max. Root cause, fix approach, sync status. No technical details like method signatures or try/finally mechanics.

## Linting
- Pre-commit hook handles ruff check + format (`git config core.hooksPath hooks/`)

## Microscope
- Do NOT run hardware commands (scans, autofocus, stage moves) — only the main scan session does that.
- SSH reads, git pulls, and file checks are fine.
