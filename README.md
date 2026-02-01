# FlakeFinder

Automated 2D material flake detection system for the Leica DM6M microscope.

## Status: Phase 0 Complete

This is a rewrite of the [2DMatGMM-System](https://github.com/dgglab/2DMatGMM-System). Hardware connection validated 2026-02-01.

**Confirmed hardware:**
- Stage: STAGE (Märzhäuser SCAN 100x100)
- Camera: K5C
- Nosepiece: 6-position
- Lamp, Z-Drive, Shutter, Aperture

**What works:**
- Microscope driver classes (Stage, Camera, Lamp, Nosepiece, etc.)
- Basic CLI with connection test (`flakefinder connect`)

**Next up:** Phase 1 (Continuous Overview)

## Installation

```bash
# Clone the repo
git clone git@github.com:sharpelab/flakefinder.git
cd flakefinder

# Install with uv
uv sync

# Run CLI
uv run flakefinder info
uv run flakefinder connect
```

## Requirements

- Python 3.11+
- Windows (for Leica SDK)
- Leica DM6M microscope with SDK DLLs

See [docs/setup.md](docs/setup.md) for detailed setup instructions.

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Bootstrap - minimal driver, validate hardware connection | Done |
| 1 | Continuous Overview - fast wafer overview scanning | Planned |
| 2 | Chip Scan - detect chips, scan chips, Google Maps-style viewer | Planned |
| 2.5 | Evaluate SDK rewrite - modernize driver if needed | Planned |
| 3 | Processing Pipeline - async queue architecture | Planned |
| 4 | Deep Scan - multi-magnification flake imaging | Planned |
| 5 | Real Processors - GMM detector, Maskterial integration | Planned |

## Architecture

```
[Microscope PC - Windows]              [Server - flakes.sharpelab.science]
FlakeFinder                            2DMatGMM-Website-GGG
    │                                       │
    ├─ controls Leica DM6M microscope       ├─ Flask API backend
    ├─ captures images via raster scan      ├─ MySQL database
    ├─ detects flakes (pluggable)           ├─ React frontend
    └─ uploads per-flake ──────────────────►└─ nginx reverse proxy
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest
```

## Links

- [Wiki: 2DMatGMM - Flake Detection System](https://wiki.sharpelab.science/doc/2dmatgmm-flake-detection-system-fcMf1L3yXM)
- [Original System Repo](https://github.com/dgglab/2DMatGMM-System)
- [Website Repo](https://github.com/dgglab/2DMatGMM-Website-GGG)
- [Production Site](https://flakes.sharpelab.science)
