# Per-Chip Metadata Plan

## Problem

All chips in a scan currently share the same metadata (material, substrate). In practice, different chips within a run may have different exfoliation methods, exfoliators, and preparation details. We need per-chip metadata to track this so we can correlate preparation methods with yield and optimize the exfoliation step.

## Data Model

Per-chip metadata is an evolving record that starts sparse and gets richer over time:

1. **At scan time (GUI/CLI):** operator sets method, exfoliator, and comments for chip groups or individual chips
2. **After segmentation (automatic):** detection counts (T1/T2/T3), detection density per mm², quality flake density
3. **After manual review (website, future):** tape residue flags, notes
4. **Downstream pipeline (future):** extracted count, device count, yield metrics

### Fields

Starting fields (not exhaustive — the system must support adding fields later):

| Field | Source | Description |
|-------|--------|-------------|
| `method` | operator input | Exfoliation technique (free text, tighten vocabulary over time) |
| `method_comments` | operator input | Free-text notes on method |
| `exfoliator` | operator input | Person who exfoliated (distinct from scan operator) |
| `area_mm2` | automatic (chips.json) | Chip area |
| `detection_count` | automatic (seg summary) | Total detections |
| `detection_density_per_mm2` | automatic | Total detections / area |
| `quality_flake_count` | automatic | T1 count |
| `quality_flake_density_per_mm2` | automatic | T1 / area |
| `t1` / `t2` / `t3` | automatic | Per-tier counts |

Future fields (not implemented now, but the schema must accommodate):

- `method_tape`: controlled vocabulary (scotch, duct, nitto, ...)
- `tape_residue`: bool or severity rating
- `extracted_count`: flakes successfully extracted
- `device_count`: devices fabricated
- Focus map quality (R², tilt)
- Z tracking error stats

### Key Design Decisions

- **Chip assignment is arbitrary**, not necessarily contiguous ranges. Chip 0, 3, 5 might share a method while 1, 2, 4 have another.
- **Exfoliator ≠ scan operator.** Sandesh may exfoliate chips that Chaitrali scans. Both are recorded.
- **Free text for now.** Method and exfoliator start as free text. As patterns emerge, we add controlled vocabulary fields (e.g. `method_tape`) alongside the free text, not replacing it.
- **Storage format TBD.** Could be a section in `checkpoint.json`, a separate file, or something else — decide at implementation time based on what feels cleanest.

## GUI

The GUI supports labeling chips at any point during or after a scan:

- **Bulk set:** apply method/exfoliator to all chips in the run (common case: uniform preparation)
- **Per-chip set:** click a chip in the detection overview image to set or override its metadata (for runs with mixed preparation)
- **Editable mid-run:** operator can label chips before, during, or after scanning
- Metadata persists immediately on input (written to checkpoint/file)

## Upload

The upload pipeline reads per-chip metadata and includes it in the scan-level `flakefinder` object, merging operator-provided fields with automatic stats (detection counts, area, densities).

## Website

### Storage

Schema approach TBD — options include adding columns to the Chip table, a JSON blob column, or a separate key-value table. Decision deferred until website work begins, as the schema may change.

### Editing

The website should eventually allow editing per-chip metadata (method, exfoliator, notes) after upload. This is a future feature.

### Reporting View

An "Exfoliation Stats" page (or similar) that aggregates per-chip metadata across scans:

- **Dimensions:** method, exfoliator, date range
- **Metrics:** chips scanned, total area, T1/mm², T1 absolute count, (future: extraction yield, device yield)
- **Goal:** answer "which exfoliation method produces the most quality flakes per mm²?" and "who is getting the best yields?"
