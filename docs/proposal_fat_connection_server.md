# Microscope Connection Server — Fat Server Proposal (Draft)

> **Status: Draft/Proposal** — not yet approved for implementation.
> Supersedes `initial_proposal_connection_server.md` (thin proxy approach).

## Core Idea

A long-lived server process holds the `Microscope` connection and runs
commands in-process. CLI scripts become thin argument-parsing wrappers
that forward to the server over localhost RPC. The server calls the
same `run(scope, **kwargs)` functions the scripts call today — scan,
chip_scan, capture, stage, autofocus, focus_map — on worker threads
with a lock to prevent concurrent hardware access.

No operations are "too complex to proxy" because nothing is proxied at
the SDK level. The real code runs server-side with full in-process
access to position polling, streaming, directed velocity, etc.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    leica_server.py                           │
│                                                              │
│  Owns:                                                      │
│  - Microscope() context (persistent connection)             │
│  - All subsystems (stage, z, nosepiece, camera, etc.)       │
│                                                              │
│  RPC methods map 1:1 to existing commands:                   │
│  - "scan"         → scan.run(scope, **params)                │
│  - "chip_scan"    → chip_scan.run(scope, **params)           │
│  - "capture"      → capture logic (scope, **params)          │
│  - "stage"        → stage.run(scope, **params)               │
│  - "autofocus"    → continuous_autofocus(scope, **params)    │
│  - "focus_map"    → focus_map.run(scope, **params)           │
│  - "status"       → read-only position/state query           │
│                                                              │
│  Each command runs on a new thread with scope lock held.     │
│  stdout/stderr streamed back to client over the connection.  │
│                                                              │
│  Lifecycle:                                                  │
│  - Started manually, or auto-started by first client         │
│  - Idle timeout: exits after N minutes of no commands        │
│  - Ctrl-C: clean shutdown (halt axes, close shutter, exit)   │
│  - Crash in a command: catch, halt axes, report to client,   │
│    server stays alive for next command                        │
└────────────────────────────────────────────────────────────┘
          │
          │  JSON-RPC over localhost TCP (:19876)
          │  (args in, stdout stream + result back)
          │
┌─────────┴──────────────────────────────────────────────────┐
│                    Thin CLI scripts                          │
│                                                              │
│  capture_util.py, stage.py, autofocus_demo.py, etc:          │
│    1. Parse CLI args                                         │
│    2. Try connect to server                                  │
│       → Success: send command + args, stream output, exit    │
│       → ConnectionRefused: start server, retry               │
│       → Still refused: fall back to Microscope() directly    │
│                                                              │
│  find_flakes.py:                                             │
│    Could either run through the server (sending per-step     │
│    commands) or keep its in-process Microscope as today.     │
│    TBD — probably starts the server itself and issues         │
│    commands through it, or IS the server for that session.   │
└────────────────────────────────────────────────────────────┘
```

## Why Fat Server > Thin Proxy

The initial proposal had the server exposing low-level SDK calls
(`stage.move_to`, `camera.capture`, `z.position`) and scripts
orchestrating those over RPC. Problems with that:

- **API surface explosion**: every `Axis`, `Camera`, `Nosepiece`
  method needs an RPC wrapper. Dozens of methods.
- **Latency-sensitive ops can't be proxied**: position polling at
  124 Hz, directed velocity Z tracking, streaming capture loops —
  these need in-process SDK access with sub-ms latency.
- **Lossy abstraction**: you'd end up with "simple commands go through
  server, complex ones go direct" — two paths, more complexity.

The fat server avoids all of this. Commands run in-process with the
full SDK. The RPC boundary is at the *command* level, not the *SDK
call* level. There are ~6-8 commands, not dozens of SDK methods.

## Command Interface

Each command maps to an existing `run()` function. The RPC sends
the function's keyword arguments as JSON:

```json
// capture_util
{"method": "capture", "params": {"output": "captures/test.jpg", "lamp": 80, "exposure_ms": 1.0}, "id": 1}

// stage
{"method": "stage", "params": {"x": 50000, "y": 40000}, "id": 2}

// scan (long-running — stdout streams back)
{"method": "scan", "params": {"output": "scans/test_5x", "objective_mag": "5x", "clean": true}, "id": 3}

// status (read-only, no lock needed)
{"method": "status", "id": 4}
```

Responses:
```json
// Success
{"result": {"path": "captures/test.jpg", "width": 1824, "height": 1216}, "id": 1}

// Error (command failed but server is fine)
{"error": {"message": "Output directory exists. Use clean=true."}, "id": 3}

// Busy (another command is running)
{"error": {"message": "busy", "detail": "scan is running (started 45s ago)"}, "id": 5}
```

## Output Streaming

Long-running commands (scan, chip_scan, autofocus) print progress to
stdout. The server captures this and streams it back to the client
over the TCP connection so the user sees the same output they'd see
running the script directly. Implementation options:

- Redirect `sys.stdout` to a `TeeWriter` that writes both to the
  server's console and sends lines to the client socket
- Or: capture in a buffer, client polls for output (simpler but
  laggy)

Recommend the tee approach — `find_flakes.py` already has `TeeWriter`
for exactly this pattern.

## Concurrency Model

- **One command at a time.** A threading `Lock` guards command
  execution. If a command is running, new command requests get an
  immediate "busy" response.
- **Status queries bypass the lock.** Reading position/objective/lamp
  state is safe during a scan (the scan only uses its own polling
  threads, status reads from different SDK interfaces).
- **Each command runs on a new thread.** This keeps the server's
  main loop responsive for new connections and status queries.
- **Emergency halt.** A special `halt` command bypasses the lock and
  calls `stage.halt()` + `z.halt()` immediately. Always available,
  even during a scan.

## Lifecycle

### Startup

```bash
# Manual start
uv run python leica_server.py

# With idle timeout (exits after 30 min of no commands)
uv run python leica_server.py --idle-timeout 1800
```

- Creates `Microscope()`, enters context
- Binds to `localhost:19876`
- Writes lockfile (`.leica_server.lock`) with PID, port, start time
- Prints status, listens for connections

### Auto-start by Client

When a thin script can't connect to the server:

1. Check lockfile — if stale (PID not running), delete it
2. Spawn server process in background (`subprocess.Popen`, detached)
3. Poll for server readiness (retry connect with backoff, ~3s max)
4. If server starts, send command
5. If server fails to start, fall back to direct `Microscope()`

### Shutdown

- `Ctrl-C`: halt axes, close shutter, turn off lamp, disconnect, exit
- Idle timeout: same cleanup sequence
- Client can send `shutdown` command
- Crash: lockfile left behind, next client detects stale PID, cleans up

### Crash Recovery

**Command crash** (exception in `scan.run()`, etc.):
- Server catches exception, halts all axes as safety measure
- Reports error to client
- Server stays alive for next command
- `Microscope` object should still be valid (it's the command that
  failed, not the connection)

**Server crash** (segfault in .NET, unhandled exception):
- Server process dies, lockfile becomes stale
- Next thin script detects stale lockfile, auto-starts a new server
- Or falls back to direct `Microscope()` if auto-start fails
- No worse than today where every script creates its own connection

**SDK connection loss** (microscope powered off, USB disconnect):
- SDK calls will throw. Server should detect this pattern and shut
  down cleanly rather than accepting commands that will all fail.
- Open question: can we detect this proactively (health-check poll)?

## What This Solves

- **CLI latency for interactive sessions.** `capture_util.py` goes
  from ~3-5s (connection overhead) to <1s (RPC call). Major QoL win
  for the operator workflow of move→capture→move→capture.
- **Shared state between invocations.** Camera settings, lamp state,
  last Z position persist across commands without re-initialization.
- **Full SDK access for all operations.** Scans, autofocus, focus maps
  run in-process with position polling, streaming, directed velocity —
  no degradation from proxying.

## What This Doesn't Solve

- **SDK singleton limitation.** Still one consumer at a time. If the
  server is running, direct `Microscope()` from another process will
  conflict. The lockfile makes this explicit.
- **SDK threading constraints.** There's an uncharacterized limitation
  around per-thread vs global polling in the SDK. The server doesn't
  inherently fix this — need to characterize the constraint first.
  The server *could* help if the optimal pattern turns out to be
  "one process owns all SDK interaction" but that's speculative.

## Open Questions

1. **`find_flakes.py` integration.** Does it become a server client
   (sending per-step commands) or does it remain standalone? It could
   also *be* the server for its session — start server, run pipeline
   as a sequence of RPC calls, shut down.

2. **SDK threading constraints.** Need to characterize what the actual
   limitation is (global lock? per-thread affinity? COM apartment
   model?) before knowing if the server architecture helps or is
   irrelevant.

3. **Output streaming fidelity.** Progress bars, `\r` overwrites,
   ANSI codes — how much of the terminal experience do we preserve
   when tunneling stdout through a socket? Might need to keep it
   simple (line-buffered text only).

4. **Auto-start reliability on Windows.** Spawning a background
   process that outlives the parent, detecting readiness, handling
   permission issues. Doable but needs testing on the microscope PC.

5. **`timeBeginPeriod(1)`.** Currently each script sets this on
   connect. Server would set it once on startup and hold it for its
   lifetime. Tradeoff: slightly higher system-wide power consumption
   while server is running. Probably fine — the microscope PC isn't
   a laptop.

## Complexity Estimate

- `leica_server.py`: ~300-400 lines (TCP listener, JSON-RPC dispatch,
  command routing, thread management, lifecycle)
- `leica_client.py` (library): ~100-150 lines (connect, send command,
  stream output, auto-start logic, fallback)
- Per-script changes: ~10-20 lines each (try client, fallback to
  direct). Mostly in the `main()` functions.
- Total: ~500-600 lines of new code, minimal changes to existing code.

## Comparison with Initial Proposal

| Aspect | Thin proxy (v1) | Fat server (v2) |
|--------|----------------|-----------------|
| Server complexity | Simple (method dispatch) | Moderate (thread mgmt, output streaming) |
| API surface | Large (every SDK method) | Small (~8 commands) |
| Scan/AF support | No (too latency-sensitive) | Yes (runs in-process) |
| Client complexity | Moderate (orchestration logic) | Minimal (send args, print output) |
| Benefit scope | Only simple commands | All commands including scans |
| Risk | Two code paths (server vs direct) for complex ops | Crash in long-running command |
