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

The biggest scan speed issue on many chips is the X-only row constraint. Chips with slanted or triangular edges produce very short rows at the top/bottom that still incur full preposition overhead (~450ms). A scan that could also run Y-direction rows or adapt row orientation to chip geometry would avoid this entirely.

## Current State (2026-02-12)

Removed the X speed measurement correction for Z velocity and fixed the Z preposition target (chip edge, not lead-in start). Halved lead-in from 2mm to 1mm. Results vs baseline: Z std 0.07µm (was 0.21), frame 1 error 0.05µm (was 0.07), drift 0.11µm (was 0.26), scan time 39.4s (was 42.6s). Good enough for now.

Future ideas to revisit:
- Replace directed velocity (`start_towards_max/min` + halt) with `move_to_async(z_end)` — eliminates halt overhead (36-84ms/row) and may improve tracking
- Active Z control during lead-in for steeper planes
- X/Y row orientation to avoid short rows on irregular chips
