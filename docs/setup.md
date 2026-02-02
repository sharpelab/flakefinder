# Setup Guide

## Prerequisites

- **Windows 10/11** - The Leica SDK only works on Windows
- **Python 3.11+** - pythonnet requires 3.11 or higher
- **uv** - Package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **Leica DM6M** - Physical microscope connected to the PC

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:sharpelab/flakefinder.git
cd flakefinder
```

### 2. Install Python Dependencies

```bash
uv sync
```

### 3. Install Leica SDK DLLs

Copy the required DLLs to the shared DLLs directory. On the microscope PC, they're already in place. For a fresh setup, copy from the Leica SDK installation:

```bash
# Copy DLLs to the shared location
cp /path/to/leica-sdk-dlls/* src/flakefinder/dlls/
```

See `src/flakefinder/dlls/README.md` for the full list of required files.

### 4. Verify Installation

```bash
uv run flakefinder connect
```

If successful, you should see output like:

```
FlakeFinder - Microscope Connection Test
========================================
Config directory: ./
Attempting to initialize microscope...
Connected successfully!

Subsystems:
  Stage: Name: 'SCAN 100x100', Type ID (TID): ...
  Lamp: Name: 'LED', Type ID (TID): ...
  ...
```

## Troubleshooting

### "Import error: No module named 'clr'"

pythonnet is not installed or not compatible with your Python version. Try:

```bash
uv pip install pythonnet --force-reinstall
```

### "Failed to load hwmodel2.dll"

The Leica DLLs are missing. Copy them to `src/flakefinder/dlls/`.

### "Connection failed: ..."

The microscope is not connected or powered on. Check:
- USB/serial connection
- Microscope power
- Any other software using the microscope (close LAS X, etc.)

## Development Setup

For development, install with dev dependencies:

```bash
uv sync --dev
```

This adds:
- `ruff` - Linter and formatter
- `mypy` - Type checker
- `pytest` - Test runner

### Running Checks

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run mypy src/

# Tests
uv run pytest
```
