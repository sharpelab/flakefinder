"""De-CCM a chip-scan directory: invert the measured 5800K colour matrix per pixel.

Transforms legacy mixed-space frames (colour_matrix "5800K") into
identity-space-with-WB frames: out = M @ (wb * raw), so raw_wb = M^-1 @ out.
WB gains stay in — contrast is WB-invariant once the matrix is gone. The
measured M (idx2_real_units, ccm_models.json) has unit row sums and an
all-positive inverse, so brightness is preserved and the inversion adds no
new clipping.

Pixels with any channel at 0 or 255 in the source are unrecoverable
(information destroyed by the camera's floor/ceiling before we ever saw
them); they are transformed anyway and reported as the clip fraction.
The pedestal (black-level over-subtraction, +1.02 counts) is compensated
around the inversion and re-applied so outputs remain comparable to what
an identity-mode capture would deliver.

Usage:
    uv run python scripts/deccm_frames.py scans/run_X/chip_5/scan_10x -o scans/run_X_deccm/chip_5/scan_10x
    uv run python scripts/deccm_frames.py --control ...   # M = identity: JPEG-loss control
    uv run python scripts/deccm_frames.py --flatfield calibration/flatfield_10x_bin3.npy -o OUT ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

_CCM_MODELS = Path(__file__).resolve().parents[1] / "calibration" / "colorchecker_phase2_20260806" / "ccm_models.json"


def load_inverse_ccm() -> tuple[np.ndarray, float]:
    """(M^-1 for idx2_real_units, pedestal_counts)."""
    models = json.loads(_CCM_MODELS.read_text())
    m = np.array(models["idx2_real_units"])
    return np.linalg.inv(m), float(models["pedestal_counts"])


def deccm_image(img_rgb: np.ndarray, m_inv: np.ndarray, pedestal: float) -> tuple[np.ndarray, float]:
    """Apply M^-1 to an RGB uint8 image; returns (transformed, clip_fraction).

    clip_fraction counts source pixels with any channel at 0 or 255 —
    unrecoverable before the transform ever runs.
    """
    clipped = np.any((img_rgb == 0) | (img_rgb == 255), axis=-1)
    x = img_rgb.astype(np.float64) + pedestal
    y = x @ m_inv.T - pedestal
    return np.clip(np.rint(y), 0, 255).astype(np.uint8), float(clipped.mean())


def transform_scan_dir(
    scan_dir: Path, out_dir: Path, m_inv: np.ndarray, pedestal: float, quality: int, quiet: bool
) -> None:
    frames = sorted(scan_dir.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"no frame_*.jpg in {scan_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_fracs: list[float] = []
    for i, fp in enumerate(frames, 1):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            raise SystemExit(f"unreadable frame: {fp}")
        rgb = bgr[:, :, ::-1]
        out_rgb, clip_frac = deccm_image(rgb, m_inv, pedestal)
        clip_fracs.append(clip_frac)
        cv2.imwrite(str(out_dir / fp.name), out_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not quiet and i % 100 == 0:
            print(f"  [{i}/{len(frames)}]")

    # scan_meta: rewrite the colour space + provenance; everything else verbatim.
    meta_path = scan_dir / "scan_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["camera"]["colour_matrix"] = "identity"
        meta["deccm"] = {
            "source": str(scan_dir),
            "ccm_models": str(_CCM_MODELS),
            "matrix": "idx2_real_units^-1",
            "pedestal_counts": pedestal,
            "jpeg_quality": quality,
            "clip_fraction_mean": round(float(np.mean(clip_fracs)), 6),
            "clip_fraction_max": round(float(np.max(clip_fracs)), 6),
        }
        (out_dir / "scan_meta.json").write_text(json.dumps(meta, indent=2))
    for extra in scan_dir.glob("*.json"):
        if extra.name != "scan_meta.json" and not (out_dir / extra.name).exists():
            shutil.copy2(extra, out_dir / extra.name)

    print(f"{len(frames)} frames -> {out_dir}")
    print(f"clip fraction: mean {np.mean(clip_fracs):.4%}, max {np.max(clip_fracs):.4%}")


def transform_flatfield(ff_path: Path, out_path: Path, m_inv: np.ndarray) -> None:
    """Flatfield is a mixed-space mean image (RGB float); pedestal cancels in the ratio."""
    ff = np.load(ff_path)
    out = ff @ m_inv.T
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    meta_src = ff_path.with_suffix(".json")
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta["deccm"] = {"source": str(ff_path), "matrix": "idx2_real_units^-1"}
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"flatfield -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert the measured 5800K CCM on chip-scan frames")
    parser.add_argument("scan_dir", type=Path, nargs="?", help="Directory with frame_*.jpg + scan_meta.json")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory (or .npy for --flatfield)")
    parser.add_argument("--flatfield", type=Path, help="Transform this flatfield .npy instead of a scan dir")
    parser.add_argument("--control", action="store_true", help="Use M=identity (JPEG re-encode loss control)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for output frames (default: 95)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    m_inv, pedestal = load_inverse_ccm()
    if args.control:
        m_inv = np.eye(3)
        pedestal = 0.0

    if args.flatfield:
        transform_flatfield(args.flatfield, args.output, m_inv)
        return
    if args.scan_dir is None:
        parser.error("scan_dir required unless --flatfield is given")
    transform_scan_dir(args.scan_dir, args.output, m_inv, pedestal, args.quality, args.quiet)


if __name__ == "__main__":
    main()
