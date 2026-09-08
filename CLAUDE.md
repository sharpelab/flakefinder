# CLAUDE.md - FlakeFinder

## Overview

FlakeFinder is a microscope automation system for the Sharpe Lab's Leica DM6M microscope. It performs continuous-motion scanning to detect 2D material flakes on silicon chips.

## Commands (`src/flakefinder/commands/`)

All commands are registered as pyproject.toml entry points and invokable via `sls`:
`sls <command> [args]` (e.g. `sls capture --wb ...`, `sls find-flakes --dry-run`).

| Command | Entry point | Purpose |
|---------|-------------|---------|
| `find-flakes` | `find_flakes.py` | Full pipeline orchestrator (overview → chips → focus → chip scan) |
| `autofocus` | `autofocus.py` | Z-scan autofocus CLI with debug output |
| `capture` | `capture.py` | Single image capture utility |
| `capture-series` | `capture_series.py` | Batch capture from a JSON spec, one connection, no motion |
| `scan` | `scan.py` | Multi-row snake scan with continuous motion capture |
| `stitch` | `stitch.py` | Stitch scan frames into 2D overview image |
| `find-chips` | `find_chips.py` | Detect chips in stitched image using Otsu thresholding |
| `focus-map` | `focus_map.py` | Autofocus grid sampling across a chip |
| `analyze-focus-map` | `analyze_focus_map.py` | Analyze focus map, fit tilt plane |
| `chip-scan` | `chip_scan.py` | Chip scan with continuous Z tracking |
| `revisit` | `revisit.py` | Revisit stage points with autofocus and capture |
| `stage` | `stage.py` | Stage/objective control utility |

### Analysis Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `scripts/process_overview.py` | Overview post-processing pipeline (rsync + stitch + detect chips) |
| `scripts/process_chip_scan.py` | Chip scan analysis pipeline (rsync + analyze_chip_scan) |
| `scripts/download_flakes.py` | Download flake images + metadata from flakes.sharpelab.science |
| `scripts/check_bg.py` | Offline background sanity-check audit (replays illum bg-check layers over run dirs) |
| `scripts/flake_density.py` | Flake yield per mm² of chip area, joined from the flakes server (or seg summaries via `--counts seg`); `--by user/material/week`, `--plot`, `--plot-tiers` |
| `scripts/explain_tier.py` | Explain why detections missed tier 1, replaying the preset's gates over a chip's seg summary |
| `scripts/hbn_contrast.py` | Transfer matrix hBN contrast model (CLI: `--oxide`, `--na`, `--n-hbn`, `--r-offset`, `--g-offset`, `--fit`) |
| `scripts/hbn_contrast_widget.py` | Interactive R/G contrast explorer with sliders (n, oxide×2, NA, offsets). Export button → `/tmp/hbn_contrast_params.json` |
| `scripts/bg_residue_audit.py` | Substrate-background audit per run: per-chip R/G and B/G vs the golden blank, within-frame colour structure, locus-offset regression against background |
| `scripts/film_model.py` | Effective-film model: substrate background (R/G, B/G) and graphene contrast per objective vs effective oxide thickness, scan space; CLI sweeps and fits one Δd to measured background shifts |
| `scripts/revisit_measure.py` | Measure 20x/50x revisit captures (background modes, central flake contrast per channel) and join to the 10x seg metrics → CSV |
| `scripts/revisit_analysis.py` | From that CSV: cross-objective film-model test, 50x layer classification of revisited detections vs the 10x class, thickness-controlled locus-offset test per chip |

Hardware characterization experiments (SDK probes, Z-tracking tests, speed sweeps) are in `scripts/experiments/`.

## Tools

| Tool | Purpose |
|------|---------|
| `tools/scan-nb` | Append timestamped entries to scan notebooks. Setup: `ln -sf ~/sharpelab/flakefinder/tools/scan-nb ~/.local/bin/scan-nb` |

## Library (`src/flakefinder/leica/`)

| Module | Purpose |
|--------|---------|
| `units.py` | Axis, Stage, ZDrive, Nosepiece, Shutter, Lamp classes |
| `camera.py` | Camera control and image capture |
| `core.py` | LeicaConnection context manager |
| `enums.py` | SDK interface IDs and constants |

## Hardware Specs (Leica DM6M)

From `microscope_description.json`:

| Axis | Range | Max Speed |
|------|-------|-----------|
| X | 0 - 95,172 µm | 40 mm/s |
| Y | 0 - 85,103 µm | 40 mm/s |
| Z | 0 - 25,837 µm | 5 mm/s |

**Camera:** Leica K5C, 5472×3648 sensor, 2.4 µm physical pixel

**Objectives:** 2.5x, 5x, 10x, 20x, 50x, 150x (positions 6, 1, 2, 3, 4, 5)

## Data Formats

- **Scan metadata:** `scan_meta.json` - see [docs/scan_metadata.md](docs/scan_metadata.md)
- **Stitch metadata:** `*_stitch_meta.json` - coordinate mapping for stitched images
- **Chip detection:** `*_chips.json` - detected chip bounding boxes in stage coordinates

## Linting & Type Checking

Run before committing:
```bash
uv run ruff check --fix . && uv run ruff format . && uv run ty check
```

Pre-commit hook setup (one-time per clone):
```bash
git config core.hooksPath hooks/
```

- **Ruff** config: `pyproject.toml` under `[tool.ruff]`. Rules: E, F, I, UP, B, SIM.
- **ty** config: `pyproject.toml` under `[tool.ty]`. Excludes `driver/`; suppresses `unresolved-reference` (forward-ref string annotations) and .NET SDK imports.
- **ty LSP**: The `ty-lsp@zack-local` plugin provides `ty server` as the Python LSP (Pyright is disabled). Plugin source: `~/.claude/plugins/ty-lsp/`. Binary: `.venv/bin/ty`.

## Microscope Operations

**NEVER run scripts that touch microscope hardware without explicit user approval.**

Workflow:
1. Show the command you intend to run
2. Ask "Go for microscope?" (or similar)
3. Wait for "go", "go for microscope", or explicit approval
4. Then execute

Requires approval:
- Running scans (`sls scan`)
- Capturing images (`sls capture`)
- Moving stage (`sls stage`)
- Any script that connects to the Leica hardware

Does NOT require approval:
- SCP files from microscope
- SSH to read files or check status
- Editing code on the microscope
- Local operations (stitching, analysis)

## Grabbing Files from Microscope

```bash
scp 'sharpelab-microscope:flakefinder/scans/chips_5x.zip' scans/
```

## Axis Conventions

- **+Z = closer to sample** (crash risk at high Z values)
- Retract for safety by **decreasing Z**
- Snake scan: alternating +X/-X rows, top to bottom (-Y)

## Documentation

| Doc | Content |
|-----|---------|
| `docs/architecture.md` | System overview, stage speeds |
| `docs/scan_metadata.md` | Scan output format reference |

## Subtask Context Files

See `.claude/commands/start_subtask.md` for the context file menu used when spawning subtasks.
