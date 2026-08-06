"""Phase 2 Stage 2: recon the K5C 12-bit delivery format.

Probe A showed PROP_PIXEL_DEPTH offers [8, 12] (index 0/1). Our converter
(image_utils.sdk_image_to_numpy) assumes uint8 BGR, so this script captures
ONE 12-bit frame with a raw buffer handler and reports the actual layout:
Format() metadata, buffer size vs pixel count, uint16 interpretation stats
(range, bit usage, per-channel means), saving the raw buffer as .npy for
offline analysis. A depth-8 reference frame is captured first for
comparison. Pixel depth is restored to 8-bit afterward.

Resolves: byte layout / container size / value range of 12-bit mode →
gates the image_utils uint16 implementation.

Hardware footprint: lamp + shutter via scope.light_on, two single-frame
captures. NO stage/Z/objective motion. Run with operator approval, focused
on a bright target (e.g. ColorChecker white patch, 10x).

Usage (on the microscope PC):
    uv run python scripts/experiments/probe_pixel_depth12.py --lamp 100 --exposure-ms 0.5 --gain 2
"""

import argparse
import contextlib
import json
import threading
from pathlib import Path

import numpy as np

from flakefinder.leica import Microscope


def _dump_format(fmt) -> dict:
    """Call every no-arg method on the SDK ImageFormat and record results."""
    skip = {
        "Equals", "Finalize", "GetHashCode", "GetType", "MemberwiseClone",
        "Overloads", "ReferenceEquals", "ToString", "Dispose",
    }  # fmt: skip
    out = {}
    for name in dir(fmt):
        if name.startswith("_") or name in skip:
            continue
        fn = getattr(fmt, name)
        with contextlib.suppress(BaseException):
            v = fn()
            out[name] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return out


def capture_raw(camera, timeout_s: float = 10.0) -> tuple[dict, np.ndarray]:
    """Acquire one frame; return (format_info, raw_buffer_bytes as uint8 array)."""
    import System
    from LeicaMicrosystems.HardwareModel import Extensions

    result: dict = {}
    ready = threading.Event()

    def on_acquired(image):
        try:
            image.LockPixelData()
            try:
                fmt = image.Format()
                info = _dump_format(fmt)
                size = fmt.PixelBufferSize()
                arr = System.Array[System.Byte](size)
                System.Runtime.InteropServices.Marshal.Copy(image.PixelData(), arr, 0, size)
                result["info"] = info
                result["buf"] = np.frombuffer(bytes(arr), dtype=np.uint8).copy()
            finally:
                image.UnlockPixelData()
        finally:
            image.Dispose()
            ready.set()

    context = Extensions.UCAPI.CancellableImageAcquisitionContext.SystemMemoryFactory
    context.ImageAcquiredHandler = Extensions.UCAPI.DelegateOnImageAcquired(on_acquired)
    try:
        camera._acquisition.Acquire(context, None)
        if not ready.wait(timeout=timeout_s):
            raise RuntimeError("acquisition timed out")
    finally:
        context.IsCancelled = True
        with contextlib.suppress(Exception):
            context.Dispose()
    return result["info"], result["buf"]


def report_buffer(buf: np.ndarray, width: int, height: int) -> dict:
    """Interpret the raw buffer and report stats for plausible layouts."""
    n = buf.size
    px = width * height
    rec: dict = {"buffer_bytes": int(n), "pixels": int(px), "bytes_per_pixel": n / px if px else None}

    if px and n == px * 6:  # 3 channels x 16-bit container
        u16 = buf.view(np.uint16).reshape(height, width, 3)
        rec["layout"] = "3ch x uint16 (assumed BGR)"
        rec["u16_min"] = int(u16.min())
        rec["u16_max"] = int(u16.max())
        rec["bits_used"] = int(u16.max()).bit_length()
        rec["channel_means_bgr"] = [float(u16[..., c].mean()) for c in range(3)]
    elif px and n == px * 3:
        u8 = buf.reshape(height, width, 3)
        rec["layout"] = "3ch x uint8 (still 8-bit?)"
        rec["channel_means_bgr"] = [float(u8[..., c].mean()) for c in range(3)]
        rec["u8_max"] = int(u8.max())
    else:
        rec["layout"] = f"unexpected ({n / px if px else '?'} bytes/px) — inspect .npy"
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="Recon K5C 12-bit pixel depth delivery (single captures)")
    parser.add_argument("--lamp", type=float, default=100.0, help="Lamp intensity %% (default 100)")
    parser.add_argument("--exposure-ms", type=float, default=0.5, help="Exposure in ms (default 0.5)")
    parser.add_argument("--gain", type=float, default=2.0, help="Camera gain (default 2.0)")
    parser.add_argument("-o", "--output", default="experiments/raw12_recon", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with Microscope() as scope:
        scope.light_on(args.lamp)
        try:
            camera = scope.camera
            camera.exposure_time = args.exposure_ms / 1000.0
            camera.gain = args.gain
            camera.gain_rgb = (1.0, 1.0, 1.0)

            w, h = camera.frame_size_px
            print(f"Frame size: {w}x{h}, pixel_depth index: {camera.pixel_depth}")

            print("\n--- depth 8 reference ---")
            info8, buf8 = capture_raw(camera)
            rec8 = report_buffer(buf8, w, h)
            print(json.dumps({"format": info8, "buffer": rec8}, indent=2))
            np.save(out / "depth8_raw.npy", buf8)

            print("\n--- switching to 12-bit (index 1) ---")
            camera.pixel_depth = 1
            print(f"pixel_depth index now: {camera.pixel_depth}")
            info12, buf12 = capture_raw(camera)
            rec12 = report_buffer(buf12, w, h)
            print(json.dumps({"format": info12, "buffer": rec12}, indent=2))
            np.save(out / "depth12_raw.npy", buf12)

            with open(out / "recon.json", "w") as f:
                json.dump(
                    {
                        "exposure_ms": args.exposure_ms,
                        "gain": args.gain,
                        "lamp": args.lamp,
                        "frame_px": [w, h],
                        "depth8": {"format": info8, "buffer": rec8},
                        "depth12": {"format": info12, "buffer": rec12},
                    },
                    f,
                    indent=2,
                )
            print(f"\nWrote {out}/recon.json, depth8_raw.npy, depth12_raw.npy")
        finally:
            with contextlib.suppress(Exception):
                camera.pixel_depth = 0
                print(f"pixel_depth restored to index: {camera.pixel_depth}")
            scope.light_off()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
