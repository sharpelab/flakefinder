# Microscope Connection Server — Initial Proposal

## Current Architecture Summary

Every script creates its own `Microscope` → `LeicaConnection` → loads DLLs, walks the SDK device tree, and initializes subsystems. The `find_flakes.py` pipeline already optimizes this by calling command modules in-process with a **shared** `Microscope` instance — so the pipeline doesn't pay per-step connection costs. But standalone script invocations (`capture_util.py`, `autofocus_demo.py`, `stage.py`) each connect/disconnect independently.

Key observations from the code:

1. **`HardwareModel.TheHardwareModel()` is a singleton** — the SDK itself only allows one logical connection. Two processes can't both hold a connection.
2. **Connection setup cost is moderate** — DLL loading (`_init_sdk`), `TheHardwareModel()`, device tree traversal, UCAPI registration for camera. Probably 2-5 seconds total (untimed, but the DLL load + unit tree walk + camera init are non-trivial).
3. **`find_flakes.py` already solved this internally** — its `run()` function takes a `Microscope` and calls `scan.run(scope=scope, ...)`, `chip_scan.run(scope=scope, ...)`, etc. in-process. The `main()` entry points are only used for standalone CLI invocations.
4. **SDK objects are .NET COM interop** — they're in-process .NET objects accessed via pythonnet. They can't be serialized or shared across process boundaries.
5. **Timing-sensitive operations** (position polling at 124 Hz, directed velocity Z tracking, capture loops) all happen in tight in-process loops with direct SDK calls. Any IPC layer would add latency that matters here.

## What Problem Would a Server Solve?

The main pipeline (`find_flakes.py`) already has a persistent connection. The in-process `Microscope` context manager lives for the entire run. A server would only help:

- **Standalone script invocations** during interactive sessions — `capture_util.py`, `stage.py`, `autofocus_demo.py` each pay ~3-5s connect/disconnect. Running several in quick succession (e.g., move stage, capture, move again, capture) wastes time reconnecting.
- **Multi-tool workflows from the scan operator** — the Claude scan session currently shells out to standalone scripts, each reconnecting.
- **Persistent position monitoring** — a long-lived process could continuously track position, detect drift, maintain state between operations.

## Proposed Architecture: Thin Server with Fat Clients

```
┌──────────────────────────────────────────────────┐
│                 leica_server.py                    │
│                                                    │
│  Owns:                                            │
│  - Microscope() context (persistent connection)   │
│  - Subsystem objects (stage, z, nosepiece, etc)   │
│  - Acquisition context (shared, reusable)         │
│                                                    │
│  Exposes via JSON-RPC over localhost TCP:          │
│  - stage.position, stage.move_to, stage.halt      │
│  - z.position, z.move_to, z.halt                  │
│  - nosepiece.position, nosepiece.switch            │
│  - shutter.open/close, lamp.set                   │
│  - camera.capture (returns path to saved image)    │
│  - scope.status                                    │
│  - scope.light_on/off                              │
│                                                    │
│  Does NOT expose:                                  │
│  - Streaming/continuous acquisition                │
│  - Position polling at 124 Hz                      │
│  - Directed velocity control                       │
│  - Raw SDK interfaces                             │
│                                                    │
│  Lifecycle:                                        │
│  - Started manually: uv run python leica_server.py │
│  - Runs until killed (or --timeout for auto-exit)  │
│  - Single client at a time for commands            │
│  - Lock-based concurrency for safety               │
└──────────────────────────────────────────────────┘
          │
          │  JSON-RPC over TCP (localhost:19876)
          │
┌─────────┴──────────────────────────────────────┐
│              Client scripts                      │
│                                                  │
│  capture_util.py:                               │
│    - Checks for server → uses it (fast)          │
│    - No server → falls back to Microscope()      │
│                                                  │
│  stage.py, autofocus_demo.py:                   │
│    - Same pattern: try server, fallback direct   │
│                                                  │
│  find_flakes.py, scan.py, chip_scan.py:          │
│    - UNCHANGED. Continue using in-process        │
│      Microscope(). No server interaction.        │
│    - These need direct SDK access for tight      │
│      loops, streaming, directed velocity, etc.   │
└──────────────────────────────────────────────────┘
```

## What the Server Owns vs What Stays in Clients

**Server owns** (simple, latency-tolerant operations):
- Connection lifecycle (connect on start, disconnect on shutdown)
- Point queries: current position (X, Y, Z), objective, lamp, shutter
- Simple commands: move to position, switch objective, set lamp, open/close shutter
- Single-shot capture: acquire one frame, save to disk, return path
- Status reporting: what `stage.py` with no args does today

**Clients keep** (latency-sensitive, streaming, complex):
- Everything in `scan.py` and `chip_scan.py` — position polling threads, capture loops, directed velocity Z tracking, frame queuing. These need sub-millisecond SDK access.
- Autofocus — continuous Z-scan with frame capture at ~20 fps and position interpolation.
- `find_flakes.py` — already has a persistent in-process connection.
- Anything involving `FrameStream`, `DeferredFrameStream`, or the acquisition loop.

**The key insight: scan operations need direct in-process SDK access. The server is for convenience commands, not performance-critical paths.**

## Protocol: JSON-RPC over TCP

Why JSON-RPC over localhost TCP:
- Simple, well-understood protocol
- Python stdlib (`json`, `socket`) — no new dependencies
- Synchronous request/response fits the usage pattern (move, wait, done)
- Easy to debug (telnet/netcat to the port, send JSON)
- No HTTP overhead needed for localhost-only

Why not alternatives:
- Named pipes: Windows-native but more complex API
- HTTP/REST: overkill for single-client localhost
- gRPC: dependency, code generation, overkill
- Unix sockets: not available on Windows

Example request/response:
```json
// Request
{"method": "stage.move_to", "params": {"x": 50000, "y": 40000}, "id": 1}
// Response
{"result": {"x": 50000.1, "y": 39999.8}, "id": 1}

// Request
{"method": "camera.capture", "params": {"path": "captures/test.jpg"}, "id": 2}
// Response
{"result": {"path": "captures/test.jpg", "width": 1824, "height": 1216}, "id": 2}

// Request
{"method": "scope.status", "id": 3}
// Response
{"result": {"x": 50000.1, "y": 39999.8, "z": 24699.0, "objective": "20x", "lamp_pct": 100, "shutter": "open"}, "id": 3}
```

## Connection Lifecycle

**Startup:**
```bash
# Start server (blocks, holds connection)
uv run python leica_server.py

# With auto-timeout (exits after 30 min idle)
uv run python leica_server.py --idle-timeout 1800
```
- Creates `Microscope()`, enters context, binds to `localhost:19876`
- Writes a lockfile (e.g., `.leica_server.lock`) with PID and port
- Prints status and listens for connections

**Client connection:**
```python
# In capture_util.py, stage.py, etc.
def get_microscope():
    """Try server first, fall back to direct connection."""
    client = LeicaClient.try_connect()  # Returns None if no server
    if client is not None:
        return client  # Implements same interface subset
    return Microscope()  # Direct connection (existing behavior)
```

The key: **no server required.** Every script still works standalone. The server is an optional accelerator for interactive sessions.

**Crash recovery:**
- If the server crashes, the SDK connection dies with it. Next client attempt gets `ConnectionRefused`, falls back to direct `Microscope()`.
- If a client disconnects mid-operation (e.g., ctrl-C during move), the server should halt all axes as a safety measure.
- Server should catch and log SDK errors, but not try to "reconnect" — SDK singleton semantics make that unreliable. Kill and restart.

**Multiple clients:**
- Commands are serialized through a lock. One operation at a time.
- Status queries can be concurrent (read-only), but simplest to just serialize everything. The bottleneck is the microscope, not the server.
- If a scan is running in-process (via `find_flakes.py`), the server shouldn't be running — they'd conflict over the SDK singleton. The lockfile can warn about this.

## What This Solves and What It Doesn't

**Solves:**
- **Interactive session latency**: `capture_util.py` goes from ~5s (connect + capture + disconnect) to ~0.5s (send request, get result). Big win for the operator workflow where you're doing move→capture→move→capture repeatedly.
- **`stage.py` as a quick command**: Position queries and moves become instant instead of paying connection overhead each time.
- **Shared state**: Server can maintain "last known good focus", camera settings, etc. between invocations. Currently each script re-defaults everything.

**Doesn't solve:**
- **Scan performance**: Scans already use in-process `Microscope()`. No change.
- **Pipeline orchestration**: `find_flakes.py` already chains operations in-process. No change.
- **SDK single-instance limitation**: You still can't run two things simultaneously. The server makes this more explicit (server holds the lock, standalone scripts fall back or fail fast).
- **Position polling at 124 Hz**: Too latency-sensitive for IPC. Stays in-process.

## Complexity Costs and Trade-offs

**Where it's genuinely useful:**
- Interactive microscope sessions (operator workflow). The scan operator currently spawns `capture_util.py` and `stage.py` as subprocesses. If a server is running, these become near-instant.
- Quick status checks. `stage.py` with no args just reports position — shouldn't need 5s of connection setup.

**Where it's overkill:**
- Automated pipeline runs. `find_flakes.py` already has persistent in-process connections.
- One-off scripts. If you're running one `autofocus_demo.py` invocation, the 5s overhead doesn't matter.

**Complexity budget:**
- ~200-300 lines for the server (JSON-RPC handler, Microscope lifecycle, method dispatch)
- ~50-100 lines for the client library (connection, request/response, fallback logic)
- ~20 lines per script to add the "try server, fallback direct" pattern
- One new concept (server process) that needs to be understood and managed

**Risks:**
- SDK singleton conflicts: if someone starts the server then tries to run `find_flakes.py` separately, one of them fails. Need clear error messages.
- Stale server: if the microscope is physically disconnected/rebooted while the server is running, SDK calls will fail. Server needs to detect this and exit cleanly.
- Maintenance burden: another thing to keep working. But it's optional — nothing breaks if it's not running.

## Alternative: Don't Build a Server

The honest alternative is: **just make scripts accept an existing `Microscope` when called programmatically, and only pay connection cost for CLI invocations.** This is what `find_flakes.py` already does with the `run(scope, ...)` pattern.

The remaining pain is the operator workflow (Claude session running `capture_util.py` etc. as subprocesses). One option: instead of a server, have the scan operator session maintain a persistent Python REPL with a `Microscope()` and dispatch commands to it. But that's basically reinventing the server with worse ergonomics.

## Recommendation

Build it as a small, optional convenience tool. The design should be:
1. **Simple**: JSON-RPC, one file for server, one small module for client
2. **Optional**: Every script works without it
3. **Conservative**: Only expose simple commands; scan/autofocus/streaming stay in-process
4. **Killable**: `Ctrl-C` shuts it down cleanly, cleans up lockfile
5. **Discoverable**: Client checks for lockfile/port, falls back silently

Don't try to proxy the full `Microscope` API. The server is a convenience layer for move/capture/status, not a general-purpose middleware.
