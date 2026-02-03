# Architecture

## System Overview

FlakeFinder is a microscope automation system for detecting 2D material flakes (graphene, hBN, etc.) on silicon wafers.

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────┐
│     Microscope PC (Windows)         │     │   Server (flakes.sharpelab.science) │
│                                     │     │                                 │
│  ┌─────────────────────────────┐   │     │  ┌───────────────────────────┐ │
│  │       FlakeFinder           │   │     │  │   2DMatGMM-Website-GGG    │ │
│  │                             │   │     │  │                           │ │
│  │  ┌───────────┐ ┌─────────┐ │   │     │  │  ┌─────────┐ ┌─────────┐ │ │
│  │  │  Driver   │ │ Scanner │ │   │     │  │  │  Flask  │ │  React  │ │ │
│  │  │ (Phase 0) │ │(Phase 1)│ │   │     │  │  │   API   │ │   UI    │ │ │
│  │  └─────┬─────┘ └────┬────┘ │   │     │  │  └────┬────┘ └────┬────┘ │ │
│  │        │            │      │   │     │  │       │           │      │ │
│  │  ┌─────┴────────────┴────┐ │   │     │  │  ┌────┴───────────┴────┐ │ │
│  │  │    Processing Queue   │ │   │     │  │  │       MySQL         │ │ │
│  │  │      (Phase 3)        │ │   │     │  │  │                     │ │ │
│  │  └───────────┬───────────┘ │   │     │  │  └─────────────────────┘ │ │
│  │              │             │   │────────▶│                         │ │
│  │  ┌───────────┴───────────┐ │   │ HTTP │  │  Per-flake upload API   │ │
│  │  │   Detectors (Phase 5) │ │   │     │  │                         │ │
│  │  │  GMM | Maskterial | ? │ │   │     │  └───────────────────────────┘ │
│  │  └───────────────────────┘ │   │     │                                 │
│  └─────────────────────────────┘   │     └─────────────────────────────────┘
│                                     │
│  ┌─────────────────────────────┐   │
│  │      Leica DM6M             │   │
│  │  ┌───────┐ ┌───────────┐   │   │
│  │  │ Stage │ │  Camera   │   │   │
│  │  │100x100│ │DFK 33UX174│   │   │
│  │  └───────┘ └───────────┘   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Three Scan Types

| Scan | Magnification | Purpose | Output |
|------|---------------|---------|--------|
| **Overview** | 5x (or 2.5x) | Find chips on wafer | Stitched image + chip boundary mask |
| **Chip scan** | 10x/20x | Find flakes on chips | Flake positions + masks + detection images |
| **Deep scan** | 10x + 50x | Human review | High-quality multi-mag images per flake |

## Hardware

### Microscope: Leica DM6M

- Motorized stage: Märzhäuser SCAN 100x100
- Objectives: 5x, 10x, 20x, 50x, 100x (on nosepiece)
- Camera: Leica K5C (rolling shutter, ~15ms readout, ~20fps at 3x3 binning)
- Illumination: LED with adjustable intensity
- Focus: Motorized Z-drive with autofocus capability

#### Axis Conventions

| Axis | Direction | Notes |
|------|-----------|-------|
| X | +X = ? | TBD |
| Y | +Y = ? | TBD |
| **Z** | **+Z = CLOSER to sample** | ⚠️ Higher Z values move objective toward sample (crash risk) |

**Z Safety**: Lower Z values = safer (more clearance). When retracting for safety, DECREASE Z.

### Stage Speeds (Measured from SDK)

| Axis | Max Speed | Notes |
|------|-----------|-------|
| X | 40 mm/s | From SDK velocity converter |
| Y | 40 mm/s | From SDK velocity converter |
| Z | 5 mm/s | Much slower than X/Y |

Use `stage.x.max_velocity_um_s` to get max speed in µm/s from SDK.

## Phased Development

### Phase 0: Bootstrap (Current)

- New repo, clean structure
- Copied Leica SDK wrapper from 2DMatGMM-System
- Basic CLI with connection test
- Documentation

### Phase 1: Continuous Overview

- Implement continuous motion scanning (vs. stop-and-shoot)
- Target: < 1 minute for full overview (currently ~15 min)
- Output: Fast overview image + chip mask

### Phase 2: Chip Scan

- Detect chip boundaries from overview
- Plan efficient scan path for chip areas
- Implement chip-area raster with pipelining

**Milestone: Google Maps-style viewer**
- Zoomable/pannable overview
- Click to explore regions

### Phase 2.5: Evaluate SDK Rewrite

Decision point: evaluate whether the current driver needs modernization before proceeding.

**Consider rewriting if:**
- Driver bugs are causing real problems
- Need features the current structure doesn't support well (e.g., async, context managers)
- Code is actively blocking development

**Keep as-is if:**
- It works and we're not touching it much
- Time is better spent on scanning/detection logic

If rewriting, target:
- Thin SDK wrapper (just exposes .NET objects)
- Higher-level `Microscope` facade
- Context manager for connection lifecycle
- Proper `enum.IntEnum` types
- Error handling and logging

### Phase 3: Processing Pipeline

- Async queue architecture
- Pluggable processor interface
- Frame → Queue → Process → Queue → Output flow

### Phase 4: Deep Scan Optimization

- Revisit detected flakes at multiple magnifications
- Pipelined capture (stage never waits)
- Z-map or continuous autofocus

### Phase 5: Real Processors

- Hook up GMM detector
- Hook up Maskterial
- Interface for future ML models
- Per-flake upload to streaming API

## Driver Structure

```
src/flakefinder/driver/
├── microscope.py      # Main Microscope class and subunits
├── interfaces.py      # Type stubs for .NET SDK objects
├── helpers.py         # Unit traversal helpers
├── enums/             # SDK constants (TID, IID, etc.)
│   ├── TID.py         # Type IDs for microscope units
│   ├── IID.py         # Interface IDs
│   └── ...
└── dlls/              # Leica SDK binaries (not in git)
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `Microscope` | Root class, initializes all subsystems |
| `Stage` | X/Y stage control |
| `Axis` | Generic axis with position control |
| `Camera` | Image acquisition |
| `Lamp` | Illumination control |
| `Nosepiece` | Objective switching |
| `Aperture` | Aperture diaphragm |
| `Shutter` | Light shutter |

## Links

- [Wiki Doc](https://wiki.sharpelab.science/doc/2dmatgmm-flake-detection-system-fcMf1L3yXM)
- [Original 2DMatGMM-System](https://github.com/dgglab/2DMatGMM-System)
- [Website Repo](https://github.com/dgglab/2DMatGMM-Website-GGG)
