# CLAUDE.md - FlakeFinder

## Overview

FlakeFinder is a microscope automation system for the Sharpe Lab's Leica DM6M microscope. It performs continuous-motion scanning to detect 2D material flakes on silicon chips.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scan_area_v1.py` | Multi-row snake scan with continuous motion capture |
| `stitch_area.py` | Stitch scan frames into 2D overview image |
| `find_chips.py` | Detect chips in stitched image using Otsu thresholding |
| `stage_util.py` | Stage/objective control utility |
| `capture_util.py` | Single image capture utility |

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
uv run ruff check --fix . && uv run ruff format .
uvx ty check src/flakefinder/
```

Pre-commit hook setup (one-time per clone):
```bash
git config core.hooksPath hooks/
```

- **Ruff** config: `pyproject.toml` under `[tool.ruff]`. Rules: E, F, I, UP, B, SIM.
- **ty** config: `pyproject.toml` under `[tool.ty]`. Excludes `driver/` and `cli.py`; suppresses `unresolved-reference` (forward-ref string annotations) and .NET SDK imports.

## Microscope Operations

**NEVER run scripts that touch microscope hardware without explicit user approval.**

Workflow:
1. Show the command you intend to run
2. Ask "Go for microscope?" (or similar)
3. Wait for "go", "go for microscope", or explicit approval
4. Then execute

Requires approval:
- Running scans (`scan_area_v1.py`)
- Capturing images (`capture_util.py`)
- Moving stage (`stage_util.py`)
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
| `docs/maskterial_integration.md` | MaskTerial deep learning detector integration |
| `docs/continuous_autofocus_plan.md` | Z-scan autofocus design |
| `docs/chip_detection_plan.md` | Chip finding algorithm |
