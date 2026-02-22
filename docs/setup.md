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

### 2. Install Dependencies

```bash
uv sync
```

### 3. Install Leica SDK DLLs

Copy the required DLLs to `src/flakefinder/dlls/`. On the microscope PC they're
already in place. See `src/flakefinder/dlls/README.md` for the full list.

### 4. Pre-commit Hooks

```bash
git config core.hooksPath hooks/
```

### 5. Verify

There is no offline verify command — the SDK requires a live microscope connection.
On the microscope PC, test with:

```bash
uv run stage
```

This should print the current stage position, objective, and lamp state.

## Running Commands

Commands are registered as `[project.scripts]` entry points. On the microscope PC,
run directly with `uv run`:

```bash
uv run find-flakes --dry-run
uv run stage
uv run capture -o test.jpg
```

From a dev machine, use `sls` to run commands remotely via SSH:

```bash
sls find-flakes --dry-run
sls stage
sls capture -o test.jpg
```

See CLAUDE.md for the full command table.

## Development

Install with dev dependencies:

```bash
uv sync --dev
```

### Linting & Type Checking

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ty check
```

## Troubleshooting

### "Failed to load hwmodel2.dll"

The Leica DLLs are missing. Copy them to `src/flakefinder/dlls/`.

### "Connection failed: ..."

- Microscope not connected or powered on
- Another application holding the connection (close LAS X)
- USB/serial cable issue
