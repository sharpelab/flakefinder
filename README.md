# FlakeFinder

Automated 2D material flake detection system for the Sharpe Lab's Leica DM6M microscope. Scans silicon wafers, detects flakes (hBN, graphene, WSe₂), and uploads results to [flakes.sharpelab.science](https://flakes.sharpelab.science).

## Pipeline

The `find-flakes` command orchestrates the full scan-to-upload pipeline:

1. **Overview scan** (2.5x) — continuous-motion snake scan of the full stage area
2. **Stitch + chip detection** — assemble frames into panoramic image, find chip boundaries via Otsu thresholding
3. **Per-chip focus map** — autofocus at grid points, fit tilt plane for Z tracking
4. **Chip scan** (10x or 20x) — continuous-motion scan with real-time Z tracking along the focus plane
5. **Segmentation** — per-frame flake detection, classification, and tiered scoring (runs in background)
6. **Revisit** (50x) — re-image top-ranked flakes at higher magnification
7. **Upload** — package results and POST to flakes.sharpelab.science

Two scan presets:

| Preset | Overview | Chip Scan | Speed | Use Case |
|--------|----------|-----------|-------|----------|
| `2.5_10` | 2.5x | 10x | 10 mm/s | Fast screening |
| `2.5_20` | 2.5x | 20x | 5 mm/s | Higher resolution |

Supports checkpointing — re-run with `--resume <run_dir>` to pick up where a previous run left off.

See [docs/architecture.md](docs/architecture.md) for detailed system design and hardware specs.

## Material Presets

Segmentation is configured per-material via `DetectorConfig` presets. Each preset defines contrast thresholds, calibration curves (R/G contrast → thickness), classification gates, and scoring functions.

| Preset | Material | Substrate | Calibration | Notes |
|--------|----------|-----------|-------------|-------|
| `hbn_medium` | hBN | 90nm SiO₂ | AFM-verified | Primary hBN preset |
| `hbn_thick_90nm` | hBN | 90nm SiO₂ | AFM-verified | 30-40nm hBN |
| `hbn_medium_285nm` | hBN | 285nm SiO₂ | Transfer matrix model | Tighter R gate for tape rejection |
| `graphene_thin_90nm` | Graphene | 90nm SiO₂ | Per-layer contrast | Layer counting (0.335 nm/layer) |
| `wse2_monolayer_90nm` | WSe₂ | 90nm SiO₂ | Point calibration | Single-layer reference |

Detections are classified by proximity to the calibration curve (thin/medium/thick) and assigned a tier (T1 = high confidence, T2 = possible, T3 = unlikely) plus a continuous score based on size, shape, and calibration distance.

## Installation

**Requirements:** Python 3.13+, Windows (for Leica SDK), uv package manager.

```bash
git clone git@github.com:sharpelab/flakefinder.git
cd flakefinder
uv sync
git config core.hooksPath hooks/
```

Leica SDK DLLs must be in `src/flakefinder/dlls/` (already in place on the microscope PC). See [docs/setup.md](docs/setup.md) for details.

## Usage

### On the microscope PC

```bash
# Full pipeline with default preset (2.5x overview + 10x chip scan)
uv run find-flakes

# Higher resolution preset
uv run find-flakes --preset 2.5_20

# Graphene detection
uv run find-flakes --material graphene_thin_90nm

# Only scan specific chips
uv run find-flakes --chips 0,2,5

# Resume a previous run
uv run find-flakes --resume scans/run_20260301_1430

# Preview without running
uv run find-flakes --dry-run
```

### Remotely via sls

The `sls` tool runs commands on the microscope PC over SSH:

```bash
sls find-flakes --dry-run          # run a command
sls stage                          # check stage position
sls pull scans/run_20260301_1430/  # download scan data
sls push calibration/flatfield.npy # upload a file
sls git status                     # run git on the microscope
```

### GUIs

| Command | Framework | Purpose |
|---------|-----------|---------|
| `uv run find-flakes-gui` | Tkinter | Form-based launcher for find-flakes (for collaborators) |
| `uv run run-viewer` | Tkinter | Browse completed runs, view detections with filters |
| `uv run quick-scan` | PySide6 | Interactive stage viewer with live camera feed |

## Commands

All commands are registered as pyproject.toml entry points. Run with `uv run <command>` on the microscope or `sls <command>` remotely.

| Command | Purpose |
|---------|---------|
| `find-flakes` | Full pipeline orchestrator |
| `scan` | Multi-row snake scan with continuous motion |
| `stitch` | Stitch scan frames into 2D overview image |
| `find-chips` | Detect chips in stitched image via Otsu thresholding |
| `focus-map` | Autofocus grid sampling across a chip |
| `analyze-focus-map` | Analyze focus map, fit tilt plane |
| `chip-scan` | Chip scan with continuous Z tracking |
| `autofocus` | Single-point Z-scan autofocus |
| `capture` | Single image capture |
| `revisit` | Revisit stage points with autofocus and capture |
| `stage` | Stage position, objective, and lamp control |
| `upload` | Upload run results to flakes.sharpelab.science |
| `run-viewer` | GUI: browse completed runs |
| `find-flakes-gui` | GUI: form-based pipeline launcher |
| `quick-scan` | GUI: interactive stage viewer |

## Analysis Scripts

Post-processing and analysis tools in `scripts/`. Run with `uv run python scripts/<script>.py`.

| Script | Purpose |
|--------|---------|
| `process_overview.py` | Overview post-processing (rsync + stitch + chip detection) |
| `process_chip_scan.py` | Chip scan analysis (rsync + segmentation) |
| `segment_chip_scan.py` | Run segmentation on a completed chip scan |
| `segment_flakes.py` | Segment individual frames |
| `crop_mosaic.py` | Build detection mosaic grids with filtering (`--tier`, `--where`, `--top`) |
| `eval_detections.py` | Evaluate detection quality across runs |
| `rerank_detections.py` | Re-score detections with updated config |
| `build_flatfield.py` | Build flatfield calibration from blank frames |
| `analyze_chip_scan.py` | Scan quality analysis (Z tracking, frame pacing) |
| `analyze_autofocus.py` | Autofocus quality analysis |
| `hbn_contrast.py` | Transfer matrix hBN contrast model |
| `hbn_contrast_widget.py` | Interactive R/G contrast explorer with sliders |
| `download_flakes.py` | Download flake images from flakes.sharpelab.science |

## Tools

| Tool | Purpose |
|------|---------|
| `tools/sls` | Run commands on the microscope via SSH. Setup: `ln -sf $(pwd)/tools/sls ~/.local/bin/sls` |
| `tools/scan-nb` | Append timestamped entries to scan notebooks. Setup: `ln -sf $(pwd)/tools/scan-nb ~/.local/bin/scan-nb` |

## Directory Structure

```
├── src/flakefinder/
│   ├── leica/              # Hardware library: Stage, Camera, ZDrive, Lamp, etc.
│   ├── commands/           # CLI entry points (see Commands table)
│   ├── segmentation.py     # Flake detection: presets, scoring, classification
│   ├── scan_utils.py       # Geometry, interpolation, flatfield correction
│   ├── flakes_api.py       # Client for flakes.sharpelab.science
│   ├── cli_utils.py        # Argparse helpers, metadata builders
│   ├── data_utils.py       # Scan data loading
│   └── types.py            # NamedTuples (ScanMeta, FrameMeta, etc.)
├── src/quick_scan/         # PySide6 stage viewer GUI
├── scripts/                # Analysis and post-processing scripts
│   └── experiments/        # Hardware characterization experiments
├── tools/                  # sls, scan-nb, subtask-launch
├── docs/                   # Architecture, scan format, microscope reference, setup
├── calibration/            # Flatfield calibration images
├── scans/                  # Scan output data (gitignored)
└── archive/                # Superseded scripts kept for reference
```

## Development

```bash
uv sync                                                  # install all deps
uv run ruff check --fix . && uv run ruff format .        # lint + format
uv run ty check                                          # type check
uv run pytest                                            # tests
```

Pre-commit hooks run ruff and ty automatically. Enable with `git config core.hooksPath hooks/`.

## Links

- [Production Site](https://flakes.sharpelab.science)
- [Wiki: FlakeFinder](https://wiki.sharpelab.science/doc/2dmatgmm-flake-detection-system-fcMf1L3yXM)
- [Original System](https://github.com/dgglab/2DMatGMM-System)
- [Website Repo](https://github.com/dgglab/2DMatGMM-Website-GGG)
