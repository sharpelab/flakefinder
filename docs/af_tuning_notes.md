# Autofocus Defaults — Operator Tuning Notes

Ad-hoc tuning observations from the Quick Scan AF Debug panel. Current
`AF_DEFAULTS` in `src/flakefinder/leica/autofocus.py` has **not** been
updated with these; this doc is a scratchpad for validation before baking
tuned values into code.

## 5x (position 1) — 2026-04-14

| Phase      | Current default | Proposed     | Notes |
|------------|-----------------|--------------|-------|
| Coarse     | 1000 µm @ 2000  | 1000 / 2000  | same  |
| Fine       |  100 µm @  500  |  250 /  400  | wider, slower |
| Super-fine | (none)          |  100 /   50  | add a pass |

Observation: fine range was too narrow; adding a super-fine pass helps.

## 20x (position 3) — TBD
