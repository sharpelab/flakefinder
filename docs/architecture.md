# Architecture

## System Overview

FlakeFinder is a microscope automation system for detecting 2D material flakes
(graphene, hBN, etc.) on silicon wafers.

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────┐
│     Microscope PC (Windows)         │     │  Server (flakes.sharpelab.science) │
│                                     │     │                                 │
│  ┌─────────────────────────────┐   │     │  ┌───────────────────────────┐ │
│  │       FlakeFinder           │   │     │  │   2DMatGMM-Website-GGG    │ │
│  │                             │   │     │  │                           │ │
│  │  ┌───────────┐ ┌─────────┐ │   │     │  │  ┌─────────┐ ┌─────────┐ │ │
│  │  │   Leica   │ │Commands │ │   │     │  │  │  Flask  │ │  React  │ │ │
│  │  │  Library  │ │  (CLI)  │ │   │     │  │  │   API   │ │   UI    │ │ │
│  │  └─────┬─────┘ └────┬────┘ │   │     │  │  └────┬────┘ └────┬────┘ │ │
│  │        │            │      │   │     │  │       │           │      │ │
│  │  ┌─────┴────────────┴────┐ │   │     │  │  ┌────┴───────────┴────┐ │ │
│  │  │   Segmentation +      │ │   │     │  │  │       MySQL         │ │ │
│  │  │   Detection Pipeline  │ │   │     │  │  │                     │ │ │
│  │  └───────────┬───────────┘ │   │     │  │  └─────────────────────┘ │ │
│  │              │             │   │────────▶│                         │ │
│  │  ┌───────────┴───────────┐ │   │ HTTP │  │  Per-scan upload API    │ │
│  │  │  Upload + Run Viewer  │ │   │     │  │                         │ │
│  │  └───────────────────────┘ │   │     │  └───────────────────────────┘ │
│  └─────────────────────────────┘   │     └─────────────────────────────────┘
│                                     │
│  ┌─────────────────────────────┐   │
│  │      Leica DM6M             │   │
│  │  ┌───────┐ ┌───────────┐   │   │
│  │  │ Stage │ │  Camera   │   │   │
│  │  │100x100│ │  Leica K5C│   │   │
│  │  └───────┘ └───────────┘   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Pipeline

The `find-flakes` command orchestrates the full pipeline:

1. **Overview scan** (2.5x or 5x) — continuous-motion snake scan of the full stage
2. **Stitch** — assemble frames into panoramic overview image
3. **Chip detection** — Otsu thresholding on luminance to find chip boundaries
4. Per chip:
   a. **Focus map** — autofocus at grid points, fit tilt plane
   b. **Chip scan** (10x or 20x) — continuous-motion scan with Z tracking along focus plane
   c. **Segmentation** — per-frame flake detection and classification
   d. **Revisit** (50x) — re-image top-ranked flakes at higher magnification
5. **Upload** — package results and POST to flakes.sharpelab.science

Two presets: 2.5x overview + 10x scan, or 5x overview + 20x scan.

## Hardware

### Microscope: Leica DM6M

- Motorized stage: Märzhäuser SCAN 100x100
- Objectives: 2.5x, 5x, 10x, 20x, 50x, 150x (on motorized nosepiece)
- Camera: Leica K5C (rolling shutter, ~15ms readout, ~66 fps at 3×3 binning)
- Illumination: LED with adjustable intensity
- Focus: Motorized Z-drive with continuous autofocus

See `docs/microscope_reference.md` for detailed specs (FOV, DOF, AF parameters).

#### Axis Conventions

| Axis | Direction | Notes |
|------|-----------|-------|
| X | +X = right | Origin at top-left of stage |
| Y | +Y = down | Origin at top-left of stage |
| **Z** | **+Z = CLOSER to sample** | ⚠️ Higher Z values move objective toward sample (crash risk) |

**Z Safety**: Lower Z values = safer (more clearance). When retracting for safety, DECREASE Z.

#### Stage Speeds

| Axis | Max Speed | Notes |
|------|-----------|-------|
| X | 40 mm/s | Continuous scanning at 5–40 mm/s |
| Y | 40 mm/s | Row repositioning |
| Z | 5 mm/s | Focus tracking at ~2–60 µm/s typical |

## Library Structure

```
src/flakefinder/
├── leica/
│   ├── core.py          # LeicaConnection context manager, SDK init
│   ├── microscope.py    # Microscope facade (subsystem init)
│   ├── units.py         # Axis, Stage, ZDrive, Nosepiece, Shutter, Lamp
│   ├── camera.py        # Camera control, frame acquisition
│   ├── autofocus.py     # Continuous Z-scan autofocus
│   ├── polling.py       # Rate-limited position polling
│   ├── enums.py         # SDK constants (TID, IID, UCAPI)
│   └── types.py         # .NET type stubs
├── commands/            # CLI entry points (see CLAUDE.md for full list)
├── segmentation.py      # Flake detection pipeline
├── scan_utils.py        # Geometry, interpolation, flatfield
├── cli_utils.py         # Argparse helpers, metadata builders
├── data_utils.py        # Scan data loading
└── types.py             # NamedTuples (ScanMeta, FrameMeta, etc.)
```

### Key Classes (`leica/`)

| Class | Module | Purpose |
|-------|--------|---------|
| `LeicaConnection` | `core.py` | Context manager for SDK connection lifecycle |
| `Microscope` | `microscope.py` | Facade — initializes all subsystems from connection |
| `Stage` | `units.py` | X/Y stage control |
| `ZDrive` | `units.py` | Z-axis with hysteresis-corrected positioning |
| `Axis` | `units.py` | Generic axis (position, velocity, async moves) |
| `Camera` | `camera.py` | Image acquisition, binning, white balance |
| `Nosepiece` | `units.py` | Objective switching |
| `Lamp` | `units.py` | Illumination control |
| `Shutter` | `units.py` | Light shutter |

## Links

- [Wiki Doc](https://wiki.sharpelab.science/doc/2dmatgmm-flake-detection-system-fcMf1L3yXM)
- [Original 2DMatGMM-System](https://github.com/dgglab/2DMatGMM-System)
- [Website Repo](https://github.com/dgglab/2DMatGMM-Website-GGG)
