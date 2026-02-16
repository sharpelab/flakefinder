# SDK Threading Investigation

## Goal

100 Hz position polling on all three axes (X, Y, Z) simultaneously,
full ~60 fps camera capture, and no broken SDK functionality (async
move waits must still complete). Currently chip_scan runs 2 poll
threads (X + Z) + capture at ~20 fps. Adding Y polling and pushing
camera to 60 fps increases SDK contention — need to find the limits.

## Why

Chip scan currently tracks X and Z but assumes Y is static per row.
Supporting X+Y stage moves (diagonal or curved paths) requires Y
position data at the same rate as X. And 60 fps capture at 3x3
binning is the camera's hardware limit — we want to use all of it.

## Known Constraints

- SDK `GetControlValue()` call takes ~16 ms (measured 2026-02-05)
- 2 threads polling: 124 Hz aggregate, ~62 Hz each, 100% efficient
- `timeBeginPeriod(1)` required for good timer resolution on Windows
- **Starvation bug**: saturating the SDK with many polling threads
  prevents `AsyncResult.GetState()` from returning COMPLETED — async
  move waits hang forever. Threshold unknown.
- Camera `Acquire()` is a blocking SDK call that also competes for
  whatever internal resource the polling uses.

## Characterization Plan

### Experiment 1: Polling throughput vs thread count

N threads all calling `GetControlValue()` in tight loops on X axis.
Measure aggregate rate and per-thread rate for N = 1, 2, 4, 8.
Determines: does throughput plateau (shared resource) or scale?

### Experiment 2: Move completion under polling load

Start N polling threads, then `move_to_async` + `wait(timeout)`.
Find the N where moves stop completing. Tests same-axis (poll X,
move X) first.

### Experiment 3: Cross-axis interference

Poll axis A while moving axis B. Does polling X prevent Y moves
from completing? Does polling Z interfere with X moves?

### Experiment 4: Sleep between polls (starvation mitigation)

If move completion breaks at N threads, does adding `time.sleep(K)`
between polls fix it? Find minimum K. This distinguishes pure
starvation (fixable with rate limiting) from fundamental limits.

### Experiment 5: Camera + polling combined

The real workload: N position poll threads + camera `Acquire()` in a
loop. Does capture interfere with polling? Does polling prevent
capture? What's the max sustainable fps with 3 poll threads running?

### Experiment 6: Combined full workload

3 poll threads (X, Y, Z) + continuous capture + async move.
The actual chip_scan scenario. Does it work? What are the rates?

### Experiment 7: Poll stagger analysis

With 2 threads per axis at 8ms sleep, do both threads fire at the same
time (clustered) or interleave? Test with no stagger vs explicit
phase offset. Measure inter-sample gap distribution.

### Test Script

`scripts/sdk_threading_test.py` — runs experiments 1-7.

```
uv run python scripts/sdk_threading_test.py                         # experiments 1-4
uv run python scripts/sdk_threading_test.py --experiments 5,6,7     # camera + stagger
uv run python scripts/sdk_threading_test.py --experiments 5 --camera-threads 0,1,3
```

```
uv run python scripts/sdk_threading_test.py
uv run python scripts/sdk_threading_test.py --experiments 1,2 --threads 0,1,2,4
```

---

## Log

*(append-only — newest entries at bottom)*

### 2026-02-15: Initial characterization (experiments 1-4)

Data: `results/sdk_threading_20260215_2014.json` (exp 1-2),
`results/sdk_threading_20260215_2016.json` (exp 3-4).

**Experiment 1 — Polling throughput (same axis, X):**

| Threads | Aggregate Hz | Per-thread Hz | Call latency |
|---------|-------------|---------------|--------------|
| 1       | 63          | 63            | 15,983 µs    |
| 2       | 125         | 63            | 15,985 µs    |
| 4       | 228         | 57            | 17,265 µs    |

Throughput scales nearly linearly up to 4 threads. Per-call latency
stays ~16 ms at 1-2 threads, degrades slightly at 4 (17 ms). The SDK
does not have a global lock — multiple threads can poll concurrently.

**Experiment 2 — Same-axis move completion under polling load:**

| Poll threads | Move completed | Move time | Poll Hz |
|-------------|----------------|-----------|---------|
| 0           | YES            | 0.461s    | —       |
| 1           | YES            | 0.535s    | 74      |
| 2           | **NO (hung)**  | —         | 127     |
| 4           | **NO (hung)**  | —         | 234     |

**Starvation threshold: 2 threads polling the same axis as the move.**
1 thread is safe. 2+ threads on the same axis starve the async
completion callback.

**Experiment 3 — Cross-axis interference:**

All combinations tested (poll X/move Y, poll Y/move X, poll Z/move X,
poll X/move Z) at 0, 2, 4 threads. **Every move completed.** No
starvation when polling a different axis than the one moving.

| Poll | Move | Threads | Move time | Poll Hz |
|------|------|---------|-----------|---------|
| X    | Y    | 4       | 0.613s    | 247     |
| Y    | X    | 4       | 0.586s    | 248     |
| Z    | X    | 4       | 0.594s    | 251     |
| X    | Z    | 4       | 0.441s    | 263     |

Move times ~0.1s slower than baseline (0.46s) at 4 poll threads, but
all complete. Cross-axis polling is completely safe.

**Experiment 4 — Sleep between polls (4 threads, same axis):**

| Sleep (ms) | Move completed | Poll Hz |
|------------|----------------|---------|
| 0          | NO             | 233     |
| 0.5        | NO             | 224     |
| 1          | NO             | 219     |
| 2          | NO             | 213     |
| 5          | NO             | 190     |
| 10         | **YES (0.57s)**| 151     |

10 ms sleep per poll call fixes starvation at 4 threads. 5 ms is not
enough. At 10 ms sleep, aggregate rate drops to 151 Hz (still
37 Hz/thread).

**Conclusions:**

The SDK serializes `GetControlValue()` and `AsyncResult.GetState()`
per axis (not globally). Starvation occurs when polling threads
monopolize the per-axis resource, preventing the completion callback
from firing. Cross-axis polling uses independent resources.

**Implications for chip_scan X+Y+Z polling:**

- 3 threads polling X, Y, Z independently = **safe**, no starvation
  risk. Each axis gets ~63 Hz — meets the 100 Hz aggregate goal
  (63×3 = 189 Hz total) though per-axis is 63 Hz, not 100 Hz.
- Current chip_scan (poll X + poll Z + move X) = safe, since Z
  polling doesn't interfere with X moves.
- Adding Y polling thread to chip_scan = safe, since Y polling
  doesn't interfere with X moves.
- Camera `Acquire()` = separate resource, expected safe (not yet
  tested in combination — experiment 5 TBD).
- **Do not** poll the same axis being moved with 2+ threads unless
  adding ≥10 ms sleep between calls.

### Follow-up: 100 Hz per-axis with 2 threads + smart sleep

Single-thread ceiling is 63 Hz (16 ms call latency). To hit 100 Hz
per axis, we need 2 threads per axis — but 2 threads on the same axis
with no sleep causes move starvation (experiment 2). Experiment 4
showed 10 ms sleep fixes starvation at 4 threads (151 Hz aggregate).

Next experiment: 2 threads polling the same axis with calibrated
sleep to land at ~100 Hz aggregate while keeping moves alive. The
math: 100 Hz from 2 threads = 50 Hz/thread = 20 ms period. With
16 ms call latency, that's ~4 ms sleep. But experiment 4 showed 5 ms
wasn't enough at 4 threads — need to test whether 2 threads has a
lower sleep threshold.

Test matrix:
- 2 threads on X, sleep = 5, 6, 7, 8, 9, 10 ms
- For each: measure aggregate Hz + does move_to_async complete?
- Goal: find minimum sleep that keeps moves alive at 2 threads,
  then verify aggregate rate is ≥100 Hz

If 2 threads + sleep can't hit 100 Hz without starvation, the
alternative is 1 thread per axis at 63 Hz (good enough for position
interpolation, just not the stated goal).

### Handoff notes

**What exists:**
- `scripts/sdk_threading_test.py` — experiments 1-4, writes JSON to
  `results/`. Run on microscope only (needs SDK).
- `results/sdk_threading_20260215_2014.json` — experiments 1-2 data
- `results/sdk_threading_20260215_2016.json` — experiments 3-4 data
- `docs/sdk_threading_investigation.md` — this file (goals, plan, log)
- `docs/proposal_fat_connection_server.md` — server design (draft)

**What's proven:**
- Per-axis SDK resource, not global. Cross-axis polling is safe.
- 1 poll thread + same-axis move = safe (63 Hz)
- 2+ poll threads + same-axis move = hangs (no sleep) or needs ≥10 ms
  sleep (at 4 threads)
- Current chip_scan (X poll + Z poll + X move) = safe

**What's not yet tested:**
- 2-thread same-axis polling with sleep values between 5-10 ms
  (the 100 Hz follow-up above)
- Camera `Acquire()` combined with polling (experiment 5 in plan).
  Expected safe (different SDK resource) but unverified.
- Full combined workload: 3 poll threads + capture + async move
  (experiment 6 in plan)

**To run the follow-up experiment**, add sleep sweep support for
2-thread same-axis to `sdk_threading_test.py` (experiment 4 currently
hardcodes `--sleep-threads` count but the sleep values are
configurable). Could run as-is with:
```
uv run python scripts/sdk_threading_test.py --experiments 4 \
    --sleep-threads 2 --sleep-values 5,6,7,8,9,10 --move-timeout 5
```

**To add camera experiments**, extend the test script with an
experiment 5 that runs `Acquire()` in a loop alongside poll threads
and measures fps + poll rate + move completion.

### 2026-02-15: 2-thread sleep sweep (experiment 4 follow-up)

Data: `results/sdk_threading_20260215_2049.json`

2 threads polling same axis (X), sleep 5–10 ms, with concurrent move:

| Sleep (ms) | Move completed | Move time | Poll Hz |
|------------|----------------|-----------|---------|
| 5          | YES            | 1.39s     | 129     |
| 6          | YES            | 1.45s     | 126     |
| 7          | YES            | 1.03s     | 120     |
| 8          | YES            | **0.57s** | **106** |
| 9          | YES            | 0.57s     | 101     |
| 10         | YES            | 0.56s     | 92      |

Unlike 4 threads, 2 threads never actually hung — all sleep values
completed the move. But move performance degrades sharply below 8 ms:
5–6 ms gives 3× slower moves (1.4s vs 0.46s baseline), 7 ms is 2×
slower (1.0s). At 8 ms sleep, moves complete at normal speed (0.57s)
with 106 Hz aggregate polling.

**Sweet spot: 8 ms sleep, 2 threads = 106 Hz per axis** with healthy
move performance. This exceeds the 100 Hz target.

**Updated architecture recommendation:**
- 2 threads per axis × 3 axes = 6 poll threads, each with 8 ms sleep
- Expected: ~106 Hz per axis, ~318 Hz aggregate
- Moves on any axis complete normally (cross-axis is safe; same-axis
  with 8 ms sleep is safe)
- Camera `Acquire()` still untested in combination (experiment 5 TBD)

### 2026-02-15: Camera + full workload (experiments 5-7)

Data: `results/sdk_threading_20260215_2132.json`

**Experiment 5 — Camera + polling combined:**

Camera `Acquire()` is completely independent of position polling.
65-66 fps regardless of poll thread count (0 through 6). Polling rate
unaffected by camera. With concurrent move, all complete up to 3 poll
threads; 6 threads slows move to 1.8s but still completes.

| Poll threads | Cam fps | Poll Hz | Move | Move time |
|-------------|---------|---------|------|-----------|
| 0           | 65.7    | —       | YES  | 0.470s    |
| 1           | 65.8    | 61      | YES  | 0.486s    |
| 2           | 66.0    | 93      | YES  | 0.597s    |
| 3           | 65.9    | 125     | YES  | 0.595s    |
| 6           | 66.2    | 212     | YES  | 1.776s    |

**Experiment 6 — Full combined workload (3-axis poll + camera + move):**

| Config | Cam fps | X Hz | Y Hz | Z Hz | Total Hz | Move time |
|--------|---------|------|------|------|----------|-----------|
| 1/axis no sleep   | 63.6 | 74 | 74 | 73 | 221 | 0.795s |
| 2/axis 8ms sleep  | 63.2 | 82 | 85 | 91 | 257 | 0.625s |

Both configurations work. Camera holds 63+ fps in both cases.

**Experiment 7 — Poll stagger analysis (2 threads, same axis):**

| Config | Hz | Gap mean±std | Clustered |
|--------|-----|-------------|-----------|
| 2T no stagger  | 94 | 10.7±7.5 ms | 33% |
| 2T stagger 12ms | 94 | 10.6±7.5 ms | 33% |
| 1T reference   | 62 | 16.2±1.7 ms | 0%  |

Staggering had no effect — 33% of samples clustered (gap < 1ms)
regardless of phase offset. The SDK likely serializes access
internally, so both threads end up waiting on the same lock and
firing back-to-back. 1 thread: perfectly even 16ms gaps, zero waste.

### 2026-02-15: Coordinated 2-thread polling (experiment 8)

Data: `results/sdk_threading_20260215_2158.json`

Attempted drift correction: each thread checks when its peer last
completed and waits until `target_gap` has elapsed before polling.

| Gap (ms) | Hz | Gap mean±std | Clustered | Move | Move time |
|----------|-----|-------------|-----------|------|-----------|
| 6        | 122 | 8.2±7.9 ms | **49%**   | YES  | 4.759s    |
| 8        | 123 | 8.2±7.9 ms | **49%**   | YES  | 2.412s    |
| 10       | 124 | 8.1±7.9 ms | **49%**   | NO   | —         |

Coordination made clustering worse (49% vs 33% naive, vs 0%
single-thread). P10=0.0 ms, P50=15.8 ms — bimodal distribution
unchanged. The SDK's internal per-axis lock defeats any client-side
timing strategy: both threads queue behind the lock and fire
back-to-back regardless of coordination.

Move performance also degraded: 6 ms gap = 4.8s (10× baseline),
10 ms gap hung entirely.

**Conclusion:** 2 threads per axis is not viable for improving
temporal resolution. The SDK serializes access per-axis, and no
client-side strategy can prevent clustering or avoid move
starvation at useful poll rates. **1 thread per axis (63 Hz, 0%
clustering) is the ceiling for this SDK.**

### 2026-02-16: USB driver investigation

The 63 Hz ceiling from experiments 1-8 was caused by the SDK's
communication driver, not SDK-level serialization. The DM6M connects
via an FTDI USB-to-serial bridge (COM4). The SDK supports two
transport drivers for this link:

- **`cmserial2.dll`** — opens COM4 via Windows serial API (VCP).
  Subject to the FTDI Virtual COM Port driver's latency timer.
- **`cmusb.dll`** — talks to the FTDI chip via D2XX direct USB,
  bypassing the VCP layer entirely.

The default `ahmconfig.xml` (from the 2DMatGMM project) used
`cmserial2.dll`. The FTDI VCP driver's default 16 ms latency timer
was the bottleneck — it buffers received data for up to 16 ms
before delivering to the host application.

**Driver comparison:**

| Transport driver | FTDI path | Per-call | Per-axis Hz |
|------------------|-----------|----------|-------------|
| `cmserial2.dll` + 16 ms latency (default) | VCP | ~16 ms | 63 Hz |
| `cmserial2.dll` + 1 ms latency | VCP | ~5.8 ms | 171 Hz |
| **`cmusb.dll`** | **D2XX** | **~4.8 ms** | **~206 Hz** |

With `cmusb.dll`, the FTDI latency timer setting is irrelevant —
D2XX bypasses the VCP stack. Confirmed by resetting the latency
timer to 16 ms with no change in polling rate.

**Baud rate investigation (COM port path only):**

- `mode COM4` while connected via `cmserial2.dll` shows 19200 baud
- Baud rate is set by the SDK driver internally, not configurable
  via `ahmconfig.xml` `baudrate` parameter or Device Manager
- At 19200 baud, ~11-byte round-trip takes ~5.7 ms (matches 171 Hz)
- `baudrate="115200"` in ahmconfig → connection fails (firmware
  doesn't support it)
- With `cmusb.dll`, `mode COM4` shows 1200/7-bit — stale defaults,
  confirming D2XX doesn't touch the VCP at all

**ahmconfig.xml setup (for `cmusb.dll`):**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configurations>
  <unit name="microscope" id="MICROSCOPE" detect="*" driver="valentine.dll">
    <parameters>
      <unit name="microscope-comm" id="CM-SERIAL-COMMUNICATION-II"
            detect="*" driver="cmusb.dll">
        <parameters>
          <usb-config usbport="usb:Leica DM|usb*" lgm-stand="DM6" />
        </parameters>
      </unit>
    </parameters>
    <unit name="camera" detect="*" driver="ucBgapi.dll" />
  </unit>
</configurations>
```

Config lives in `src/flakefinder/dlls/ahmconfig.xml` and is loaded
via `TheHardwareModelInDirectory()`. No `<driver-config>` needed —
removing it gave slightly better results (~213 Hz peak).

**SDK internals (from SDK source investigation):**

- `EventSource` (IID 0x106) available on all DM6B axis units
  — push-based position events, but observed at only ~10 Hz
- `BasicControlValueSets` (batch read, IID 0x118) exists in SDK
  headers but is NOT exposed on DM6B hardware
- No threading documentation or configuration knobs in SDK
- Events fire on SDK-managed threads (need CriticalSection)

**DLL provenance:**

Working DLLs are from **AHM SDK V2023.3.0.12509** and
**UCAPI SDK V2023.3.0.12509** (build date May 12, 2025), found
on the microscope at `SDK_Tests/ARCHIVE/`. The older V2020.3.3
SDK and the LAS X installation have different (incompatible)
builds. Backup of working DLL set at
`SDK_Tests/ARCHIVE/flakefinder_working_dlls_backup/`.

**Conclusions:**

- **~206 Hz per axis, 1 thread per axis** — 3.3× the original
- `cmusb.dll` (D2XX) is strictly better than `cmserial2.dll` (VCP)
- No Device Manager changes needed
- Multi-thread approaches remain non-viable (same-axis starvation)
- Events too slow (10 Hz) to replace polling
- Per-axis independence confirmed: 3 axes × 206 Hz = 618 Hz aggregate
- Remaining ~4.8 ms/call is USB round-trip overhead
