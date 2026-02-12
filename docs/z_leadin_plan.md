# Z Lead-In Ramp Plan

## Goals

1. **Improve Z slope tracking** — open-loop feedforward drift averages 0.26µm (max 0.56µm), eating into the 1.7µm DOF at 20x
2. **Improve scan speed** — reduce per-row overhead

## Problem

Z tracking during chip scan rows is pure open-loop feedforward: velocity is set once at the chip edge and runs uncontrolled for the entire row. From the 42-row production baseline (run_20260211_1756):

- Drift is systematic and directional: +X rows drift -0.25µm, -X rows +0.24µm
- Error grows monotonically along rows (e.g. +0.045 → +0.494µm over one row)
- Velocity mismatch: ~0.2µm/s on ~5µm/s tracking velocity (~4%)
- Overall tracking: mean=-0.008µm, std=0.219, max=0.680µm
