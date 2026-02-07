You are spawning a subtask Claude instance to handle a side task during a scanning session. Your job is to construct the right prompt and spawn the instance.

## Task Request

$ARGUMENTS

## Procedure

### 1. Understand the task

Parse what the user wants. Identify:
- What needs to be built or modified
- Which existing files are relevant context
- Whether this is a new script, a modification, or analysis work
- Any session-specific details to pass along (current Z values, chip numbers, scan prefixes, etc.)

### 2. Select context files

Pick from this menu based on relevance to the task. The subtask will read these before starting.

| File | When to include |
|------|----------------|
| `src/flakefinder/leica/units.py` | Hardware control, axis/stage/Z classes, SDK interfaces |
| `src/flakefinder/leica/camera.py` | Camera capture, image acquisition |
| `src/flakefinder/leica/core.py` | Connection management, LeicaConnection |
| `src/flakefinder/leica/enums.py` | SDK constants, interface IDs |
| `autofocus_demo.py` | Autofocus logic, Z scanning patterns |
| `scan_area_v1.py` | Scanning patterns, position polling, frame capture |
| `focus_map.py` | Focus map grid autofocus |
| `stitch_area.py` | Image stitching |
| `find_chips.py` | Chip detection |
| `docs/architecture.md` | System overview, stage speeds, design |
| `docs/continuous_autofocus_plan.md` | Focus system design |
| `docs/scan_metadata.md` | Scan data format reference |
| `docs/microscope_reference.md` | Reference Z values, hardware specs |

Also include any files the user specifically mentions, plus files you know are relevant from the current session.

CLAUDE.md is auto-loaded by the subtask — don't list it.

### 3. Construct the prompt

Write a prompt file to `/tmp/subtask_<descriptive_slug>.prompt.md`. Structure:

```
# Task: <one-line summary>

## Context

<2-3 sentences of what this task is about and why>

## Read these files first

<bulleted list of context files to read>

## Task details

<full description of what to build/modify, including any specific requirements, code snippets, or values from the current session>

## Rules

**DO NOT sync to the microscope.** You are a subtask instance — the main scan session handles all `scp` to the microscope. Work locally only. Do not SSH to run commands on the microscope.

**Propose before implementing.** Before writing any code, present your plan to the user:
- What you'll build or change, and why
- For new scripts: CLI arguments (names, types, defaults), expected output files/formats, example usage
- For modifications: which files you'll touch and what the changes look like
Wait for the user to approve before writing code.

**Write a summary when done.** When finished, write `/tmp/<descriptive_name>_summary.md` with a concise handoff summary — what you found/built, the key decisions and rationale, and anything that needs action. Write it like a message to a coworker, not a reference document. No exhaustive tables or line-number inventories. If there are concrete artifacts (scripts, files, commands), mention them. Tell the user the path.

## Conventions

- `uv run python` to run scripts
- Plots: +Y axis points down (stage coordinate convention)
- Follow patterns in existing codebase
- Minimal implementation — don't over-engineer
```

### 4. Inject session-specific details

If relevant, pull details from the current session into the prompt:
- Current Z reference values (from notebook)
- Chip numbers and coordinates
- Scan prefixes and file paths
- Code snippets or function signatures the subtask needs
- Specific decisions made during this session

### 5. Spawn

```bash
newclaude ~/sharpelab/flakefinder -f /tmp/subtask_<slug>.prompt.md
```

Tell the user: the subtask is running in a new terminal. It will propose its plan before writing any code, and write a summary to `/tmp/<name>_summary.md` when done.

## After spawning

- Log the spawn in the scan notebook: what task, what prompt file
- When the user says the subtask is done (or you read its summary), review the output
- If the subtask produced code changes, you handle syncing to the microscope if needed
- If the subtask has open questions, relay them or spawn a follow-up
