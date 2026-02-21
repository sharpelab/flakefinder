You are spawning or continuing a subtask during a scanning session. Your job is to either construct a prompt and spawn a new instance, or write a continuation blurb for an existing one.

## Task Request

$ARGUMENTS

## Procedure

### 1. Classify the request

Determine whether this is:

- **New subtask**: User says "new subtask", "spawn a subtask for...", or there's no existing subtask that fits the work.
- **Continuation**: User says "send this to the X subtask", "tell the nosepiece subtask to also...", or references a recently-used subtask by name or topic.

If ambiguous, ask.

---

## New Subtask Flow

### 2. Understand the task

Parse what the user wants. Identify:
- What needs to be built or modified
- Which existing files are relevant context
- Whether this is a new script, a modification, or analysis work
- Any session-specific details to pass along (current Z values, chip numbers, scan prefixes, etc.)

### 3. Select context files

Pick from this menu based on relevance to the task. The subtask will read these before starting.

**Do NOT read these files yourself.** The subtask reads them — you're just picking which ones to list. Only read a file if YOU need information from it that isn't already in your session context (notebook, previous findings, etc.).

| File | When to include |
|------|----------------|
| `src/flakefinder/leica/units.py` | Hardware control, axis/stage/Z classes, SDK interfaces |
| `src/flakefinder/leica/camera.py` | Camera capture, image acquisition |
| `src/flakefinder/leica/core.py` | Connection management, LeicaConnection |
| `src/flakefinder/leica/enums.py` | SDK constants, interface IDs |
| `src/flakefinder/commands/autofocus.py` | Autofocus logic, Z scanning patterns |
| `src/flakefinder/commands/scan.py` | Scanning patterns, position polling, frame capture |
| `src/flakefinder/commands/focus_map.py` | Focus map grid autofocus |
| `src/flakefinder/commands/stitch.py` | Image stitching |
| `src/flakefinder/commands/find_chips.py` | Chip detection |
| `src/flakefinder/commands/chip_scan.py` | Chip scan with Z tracking |
| `src/flakefinder/commands/stage.py` | Stage/objective control |
| `src/flakefinder/commands/analyze_focus_map.py` | Focus map analysis, plane fitting |
| `docs/architecture.md` | System overview, stage speeds, design |
| `docs/continuous_autofocus_plan.md` | Focus system design |
| `docs/scan_metadata.md` | Scan data format reference |
| `docs/microscope_reference.md` | Reference Z values, hardware specs |
| `scans/*_stitch_chips_detected.png` | Chip detection results (1500px thumbnail with boxes) — use instead of raw stitch JPGs |

Also include any files the user specifically mentions, plus files you know are relevant from the current session.

CLAUDE.md is auto-loaded by the subtask — don't list it.

### 4. Construct the prompt

Write a prompt file to `/tmp/subtask_<descriptive_slug>.prompt.md`. Structure:

```
# Operator Task

<one-line summary>

## Context

<2-3 sentences of what this task is about and why>

## Read these files first

<bulleted list of context files to read>

## Requirements

<what the user asked for — pass through their requirements, not your interpretation of the implementation>
```

**Requirements, not implementation.** Describe *what* is needed and *why*, not *how* to build it. The subtask reads the context files and figures out the approach — that's its job. Don't specify CLI flag names, function signatures, data formats, or step-by-step implementation plans unless the user explicitly dictated them.

Include session-specific values when relevant (Z positions, file paths, measurements, error messages) — these are requirements context, not implementation detail. Also include test plans where applicable (e.g. dry-run verification, local unit tests, CLI smoke tests) — but NOT microscope hardware tests, those are operator-managed.

**NEVER include "commit", "push", or sync instructions in the prompt.** The subtask's default behavior is to propose a plan and wait for approval. Don't override that — the user reviews before anything is committed.

### 5. Spawn

```bash
tools/subtask-launch <slug> -f /tmp/subtask_<slug>.prompt.md
```

Tell the user: the subtask is running in a new terminal. It will propose its plan before writing any code, and write a summary to `/tmp/<name>_summary.md` when done.

### 6. After spawning

- Log the spawn in the scan notebook: what task, what prompt file
- When the user says the subtask is done (or you read its summary), review the output
- Subtasks land their own code (rebase + ff-merge into master, push, microscope pull) — you don't need to sync

---

## Continuation Flow

For sending a follow-up task to an already-running subtask.

### 2c. Pick a slug for the continuation

Choose a short descriptive slug for this follow-up task (e.g. `fix_dr_threshold`, `add_settle_param`). This determines the new summary file name: `/tmp/subtask_<slug>_summary.md`.

### 3c. Write the continuation blurb

Write a blurb that the user will paste into the existing subtask's terminal.

**Do NOT read source files to write the blurb.** You already have session context from the notebook. The subtask has the code open. Only include problem description and session-specific values (measurements, file paths, error messages).

Format:

```
# Operator Task

Summary file: `/tmp/subtask_<slug>_summary.md`

<problem description — what's wrong or what's needed, with any relevant values/context from the current session>

Before proceeding, recite your subtask_agent Rules.
```

Rules for the blurb:
- **Problem description only** — don't propose solutions. The subtask agent will figure out the approach.
- Include session-specific values (Z positions, file paths, error messages, etc.) that the subtask needs.
- Keep it concise — 1-2 paragraphs max.

### 4c. Present to the user

Show the blurb and tell the user which subtask terminal to paste it in. Example:

> Paste this into the [name] subtask terminal:

### 5c. Log it

Append to the scan notebook: what continuation was sent, to which subtask.
