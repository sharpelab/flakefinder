"""Tkinter GUI for browsing completed find-flakes runs.

Provides a two-panel interface: run list on the left, run detail on the right.
Collaborators can browse runs, select chips, and view detection results without
CLI knowledge. Runs on the microscope PC (Windows 11).

Usage:
    uv run run-viewer
    uv run run-viewer --scans-dir path/to/scans
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import sys
import tkinter as tk
import tkinter.font as tkfont
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

from PIL import Image, ImageDraw, ImageTk

from flakefinder.segmentation import Detection, DetectorConfig, dedup_detections

# ── Data loading ─────────────────────────────────────────────────────


class RunInfo(NamedTuple):
    """Summary of a single run for the list view."""

    path: Path
    name: str
    timestamp: datetime | None
    preset: str
    material: str
    n_chips: int
    total_detections: int
    duration_s: float
    notes: str
    scan_name: str
    operator: str


def _parse_run_timestamp(name: str) -> datetime | None:
    """Parse datetime from run directory name like run_20260218_1430."""
    try:
        return datetime.strptime(name, "run_%Y%m%d_%H%M")
    except ValueError:
        return None


def _load_run_info(run_dir: Path) -> RunInfo | None:
    """Load summary info from a run directory. Returns None if not a valid run."""
    cp_path = run_dir / "checkpoint.json"
    if not cp_path.exists():
        return None

    try:
        with open(cp_path) as f:
            cp = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    name = run_dir.name
    timestamp = _parse_run_timestamp(name)
    args = cp.get("args", {})
    preset = args.get("preset", "?")
    material = args.get("material", "")
    notes = cp.get("notes", "") or ""
    scan_name = cp.get("name", "") or ""
    operator = cp.get("operator", "") or ""
    n_chips = cp.get("n_chips") or 0

    # Sum detections across all chip seg summaries
    total_detections = 0
    for i in range(n_chips):
        seg_summary = run_dir / f"chip_{i}" / "seg" / "summary.json"
        if seg_summary.exists():
            try:
                with open(seg_summary) as f:
                    ss = json.load(f)
                total_detections += ss.get("stats", {}).get("total_detections", 0)
            except (json.JSONDecodeError, OSError):
                pass

    # Total duration from step_timing
    timing = cp.get("step_timing", {})
    duration_s = sum(timing.values()) if timing else 0

    return RunInfo(
        path=run_dir,
        name=name,
        timestamp=timestamp,
        preset=preset,
        material=material,
        n_chips=n_chips,
        total_detections=total_detections,
        duration_s=duration_s,
        notes=notes,
        scan_name=scan_name,
        operator=operator,
    )


def _discover_runs(scans_dir: Path) -> list[RunInfo]:
    """Find all valid run directories, sorted newest-first."""
    runs = []
    if not scans_dir.is_dir():
        return runs
    for d in scans_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            info = _load_run_info(d)
            if info is not None:
                runs.append(info)
    runs.sort(key=lambda r: r.name, reverse=True)
    return runs


def _load_all_detections(run_dir: Path, n_chips: int) -> tuple[list[Detection], dict[int, Path | None], float]:
    """Load detections from all chips, tag each with chip_idx.

    Returns (all_detections sorted by tier/score, chip_scan_dirs, um_per_px).
    """
    all_dets: list[Detection] = []
    scan_dirs: dict[int, Path | None] = {}
    um_per_px = 0.36  # default for 20x

    for chip_idx in range(n_chips):
        chip_dir = run_dir / f"chip_{chip_idx}"
        seg_dir = chip_dir / "seg"

        # Find scan directory
        scan_dir = None
        if chip_dir.is_dir():
            for d in chip_dir.iterdir():
                if d.is_dir() and d.name.startswith("scan_"):
                    scan_dir = d
                    break
        scan_dirs[chip_idx] = scan_dir

        # Load seg summary
        seg_summary = seg_dir / "summary.json"
        if not seg_summary.exists():
            continue

        try:
            with open(seg_summary) as f:
                ss = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        um_per_px = ss.get("params", {}).get("pixel_size_um", um_per_px)

        for frame_name, dets in ss.get("detections_by_frame", {}).items():
            for d in dets:
                d.setdefault("frame", frame_name)
                d["chip_idx"] = chip_idx
            all_dets.extend(dets)

    all_dets.sort(key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
    return all_dets, scan_dirs, um_per_px


# Type alias for revisit lookup: (chip_idx, frame_name, det_id) → {mag_str: image_path}
RevisitLookup = dict[tuple[int, str, int], dict[str, Path]]

_REVISIT_DIR_RE = re.compile(r"revisit_(\d+)x$")
_REVISIT_FILE_RE = re.compile(r"frame_(\d+)_d(\d+)")


def _build_revisit_lookup(run_dir: Path, n_chips: int) -> RevisitLookup:
    """Build lookup from (chip_idx, frame_name, det_id) to {mag: image_path}.

    Scans pipeline revisit dirs: chip_N/revisit_{mag}x/ (e.g. revisit_20x, revisit_50x).
    """
    lookup: RevisitLookup = {}
    for chip_idx in range(n_chips):
        chip_dir = run_dir / f"chip_{chip_idx}"
        if not chip_dir.is_dir():
            continue
        for revisit_dir in chip_dir.iterdir():
            if not revisit_dir.is_dir():
                continue
            m = _REVISIT_DIR_RE.match(revisit_dir.name)
            if not m:
                continue
            mag = m.group(1) + "x"
            for img_path in revisit_dir.glob("*.png"):
                fm = _REVISIT_FILE_RE.search(img_path.stem)
                if not fm:
                    continue
                frame_name = f"frame_{fm.group(1)}"
                det_id = int(fm.group(2))
                key = (chip_idx, frame_name, det_id)
                if key not in lookup:
                    lookup[key] = {}
                lookup[key][mag] = img_path
    return lookup


# ── Formatting helpers ───────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


OVERVIEW_MAX_WIDTH = 600
CROP_THUMB_SIZE = 440
FILTER_DEBOUNCE_MS = 200
FRAME_CACHE_MAX = 30
DEFAULT_FILTER_TOP_N = 100
BBOX_PAD_PX = 5
CONTEXT_PANE_WIDTH = 350
OVERVIEW_THUMB_MAX = 400
SLIDER_MIN_G = -2.0
SLIDER_MAX_G = 6.0
SLIDER_MIN_R = -3.0
SLIDER_MAX_ENTROPY = 8.0
SLIDER_MAX_GRAD_ENERGY = 120.0
SLIDER_MAX_ASPECT_RATIO = 6.0
SLIDER_MAX_KURTOSIS = 50.0
CROP_THUMB_HEIGHT = int(CROP_THUMB_SIZE * 2 / 3)  # 3:2 camera aspect ratio


# ── Zoomable canvas ─────────────────────────────────────────────────

ZOOM_FACTOR = 1.3
MIN_ZOOM = 0.05
MAX_ZOOM = 5.0


class ZoomableCanvas(tk.Canvas):
    """Canvas with mouse-wheel zoom, drag-to-pan, and double-click reset.

    Optionally accepts an ``on_overlay`` callback that receives
    ``(draw, crop_x0, crop_y0, zoom)`` and can draw on the display image
    before it's blitted to the canvas.
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_overlay: Callable[[ImageDraw.ImageDraw, float, float, float], None] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("bg", "black")
        super().__init__(parent, **kwargs)

        self._src: Image.Image | None = None
        self._pyramid: list[Image.Image] = []
        self._photo: ImageTk.PhotoImage | None = None
        self._on_overlay = on_overlay

        # Zoom/pan state
        self._zoom = 1.0
        self._cx = 0.0
        self._cy = 0.0
        self._fit_mode = False  # True = re-fit on resize
        self._drag_x = 0
        self._drag_y = 0

        # Bindings
        self.bind("<MouseWheel>", self._on_scroll)
        self.bind("<Button-4>", self._on_scroll_linux)
        self.bind("<Button-5>", self._on_scroll_linux)
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Double-Button-1>", self._on_reset)
        self.bind("<Configure>", self._on_configure)

    # ── Public API ──

    def set_image(self, img: Image.Image, center: tuple[float, float] | None = None):
        """Set or replace the source image. Resets zoom to fit unless *center* is given."""
        self._src = img
        self._pyramid = [img]
        level_img = img
        while min(level_img.size) > 256:
            level_img = level_img.reduce(2)
            self._pyramid.append(level_img)
        if center is not None:
            self._fit_mode = False
            self._cx, self._cy = center
        else:
            self._fit_mode = True
            self._reset_view()
        self.render()

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, value))

    @property
    def center(self) -> tuple[float, float]:
        return (self._cx, self._cy)

    @center.setter
    def center(self, value: tuple[float, float]):
        self._cx, self._cy = value

    # ── Rendering ──

    def render(self):
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 2 or ch < 2 or self._src is None:
            return
        z = self._zoom

        # Pick mipmap level: largest k where z * 2^k >= 1 (never downscale > 2x)
        level = 0
        if z < 1.0 and len(self._pyramid) > 1:
            level = min(int(math.log2(1.0 / z)), len(self._pyramid) - 1)
        src = self._pyramid[level]
        f = 2.0**level  # coordinate scale factor (original → this level)
        lz = z * f  # effective zoom at this level (typically 1–2)

        sw, sh = src.size
        lcx = self._cx / f
        lcy = self._cy / f
        half_w = cw / (2.0 * lz)
        half_h = ch / (2.0 * lz)
        src_x0 = lcx - half_w
        src_y0 = lcy - half_h
        src_x1 = lcx + half_w
        src_y1 = lcy + half_h

        crop_x0 = max(0.0, src_x0)
        crop_y0 = max(0.0, src_y0)
        crop_x1 = min(float(sw), src_x1)
        crop_y1 = min(float(sh), src_y1)

        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            return

        crop = src.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
        disp_w = max(1, int((crop_x1 - crop_x0) * lz))
        disp_h = max(1, int((crop_y1 - crop_y0) * lz))
        resample = Image.Resampling.LANCZOS if lz < 1.0 else Image.Resampling.NEAREST
        display = crop.resize((disp_w, disp_h), resample)

        if self._on_overlay:
            draw = ImageDraw.Draw(display)
            # Overlay coords are in original image space
            self._on_overlay(draw, crop_x0 * f, crop_y0 * f, z)

        self._photo = ImageTk.PhotoImage(display)
        canvas_x = int((crop_x0 - src_x0) * lz)
        canvas_y = int((crop_y0 - src_y0) * lz)
        self.delete("all")
        self.create_image(canvas_x, canvas_y, anchor="nw", image=self._photo)

    # ── Internal ──

    def _reset_view(self):
        if self._src is None:
            return
        sw, sh = self._src.size
        cw = max(1, self.winfo_width())
        ch = max(1, self.winfo_height())
        self._zoom = min(cw / sw, ch / sh)
        self._cx = sw / 2.0
        self._cy = sh / 2.0

    def _on_configure(self, event):
        if event.widget is self:
            if self._fit_mode:
                self._reset_view()
            self.render()

    def _on_scroll(self, event):
        factor = ZOOM_FACTOR if event.delta > 0 else 1.0 / ZOOM_FACTOR
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _on_scroll_linux(self, event):
        factor = ZOOM_FACTOR if event.num == 4 else 1.0 / ZOOM_FACTOR
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _zoom_at(self, mx: int, my: int, factor: float):
        self._fit_mode = False
        cw = self.winfo_width()
        ch = self.winfo_height()
        src_x = self._cx + (mx - cw / 2.0) / self._zoom
        src_y = self._cy + (my - ch / 2.0) / self._zoom
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        self._cx = src_x - (mx - cw / 2.0) / new_zoom
        self._cy = src_y - (my - ch / 2.0) / new_zoom
        self._zoom = new_zoom
        self.render()

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        self._fit_mode = False
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x = event.x
        self._drag_y = event.y
        self._cx -= dx / self._zoom
        self._cy -= dy / self._zoom
        self.render()

    def _on_reset(self, _event):
        self._fit_mode = True
        self._reset_view()
        self.render()


# ── Image popup viewer ──────────────────────────────────────────────


class ImagePopup(tk.Toplevel):
    """Resizable image popup with zoom/pan. Dismiss with Escape."""

    def __init__(self, parent: tk.Tk | tk.Toplevel, img_path: Path):
        super().__init__(parent)
        self.withdraw()  # hide until fully rendered
        self.title(img_path.name)

        src = Image.open(img_path)
        src.load()

        self._canvas = ZoomableCanvas(self)
        self._canvas.pack(fill="both", expand=True)

        # Initial size: contain to 80% of screen
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = int(screen_w * 0.8)
        max_h = int(screen_h * 0.8)
        w, h = src.size
        scale = min(max_w / w, max_h / h)
        init_w, init_h = int(w * scale), int(h * scale)
        self.geometry(f"{init_w}x{init_h}")

        self.update_idletasks()
        self._canvas.set_image(src)

        self.bind("<Escape>", lambda _: self.destroy())
        self.deiconify()
        self.focus_set()


class FlakeInspectorContext(NamedTuple):
    """Shared context passed from RunViewerGUI to FlakeInspector."""

    overview_thumb: Image.Image | None
    overview_size_px: tuple[int, int] | None
    stitch_meta: dict | None
    run_dir: Path
    annotations: dict[str, str]
    on_annotation_change: Callable[[str, str | None], None]
    um_per_px: float
    total_count: int
    revisit_lookup: RevisitLookup


class FlakeInspector(tk.Toplevel):
    """Full-featured flake inspection popup with zoomable frame, overview locator, metrics, and annotations."""

    def __init__(
        self,
        parent: tk.Tk,
        src_image: Image.Image,
        det: Detection,
        title: str = "",
        contour: list[list[int]] | None = None,
        on_navigate: Callable[[int], tuple[Image.Image, Detection, str, list[list[int]] | None] | None] | None = None,
        context: FlakeInspectorContext | None = None,
        grid_idx: int = 0,
    ):
        super().__init__(parent)
        self.withdraw()

        self._det = det
        self._contour = contour
        self._on_navigate = on_navigate
        self._ctx = context
        self._grid_idx = grid_idx
        self._overview_photo: ImageTk.PhotoImage | None = None
        self._revisit_photos: dict[str, ImageTk.PhotoImage] = {}
        self._revisit_tabs: dict[str, ttk.Frame] = {}  # mag -> tab widget

        # Window sizing
        sw, sh = src_image.size
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = min(1400, int(screen_w * 0.7))
        max_h = min(1000, int(screen_h * 0.8))
        init_zoom = min((max_w - CONTEXT_PANE_WIDTH) / sw, max_h / sh)
        frame_w = max(400, int(sw * init_zoom))
        frame_h = max(300, int(sh * init_zoom))
        total_w = frame_w + CONTEXT_PANE_WIDTH
        self.geometry(f"{total_w}x{frame_h + 60}")

        self._build_layout()
        self._update_title_bar()
        self._update_annotation_buttons()
        self._update_overview_dot()
        self._update_metrics()
        self._update_revisit_tabs()

        # Bindings
        self.bind("<Escape>", lambda _: self.destroy())
        self.bind("<Left>", lambda _: self._navigate(-1))
        self.bind("<Right>", lambda _: self._navigate(1))
        self.bind("<g>", lambda _: self._mark("good"))
        self.bind("<b>", lambda _: self._mark("bad"))
        self.bind("<u>", lambda _: self._mark(None))

        # Set image centered on detection bbox
        bx, by, bw, bh = det["bbox"]
        self.update_idletasks()
        self._frame_canvas.zoom = init_zoom
        self._frame_canvas.set_image(src_image, center=(bx + bw / 2.0, by + bh / 2.0))
        self.deiconify()
        self.focus_set()

    def _build_layout(self):
        """Build the inspector layout: top bar, paned center, bottom bar."""
        # ── Top bar ──
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", padx=4, pady=(4, 0))

        self._title_label = ttk.Label(top_bar, text="", font=("TkDefaultFont", 10, "bold"))
        self._title_label.pack(side="left", padx=4)

        self._btn_bad = tk.Button(top_bar, text="✗ Bad", command=lambda: self._mark("bad"), padx=8, pady=2)
        self._btn_bad.pack(side="right", padx=2)

        self._btn_good = tk.Button(top_bar, text="✓ Good", command=lambda: self._mark("good"), padx=8, pady=2)
        self._btn_good.pack(side="right", padx=2)

        # ── Center: horizontal PanedWindow ──
        self._paned = ttk.PanedWindow(self, orient="horizontal")
        self._paned.pack(fill="both", expand=True, padx=4, pady=2)

        # Left: zoomable frame canvas
        self._frame_canvas = ZoomableCanvas(self._paned, on_overlay=self._draw_overlay)
        self._paned.add(self._frame_canvas, weight=3)

        # Right: context pane
        right_pane = ttk.Frame(self._paned, width=CONTEXT_PANE_WIDTH)
        self._paned.add(right_pane, weight=0)

        # Notebook for overview + revisit tabs
        self._notebook = ttk.Notebook(right_pane, takefocus=False)
        self._notebook.pack(fill="both", expand=True, padx=2, pady=2)
        # Refocus toplevel after tab clicks so arrow keys navigate detections, not tabs
        self._notebook.bind("<<NotebookTabChanged>>", lambda _: self.focus_set())

        # Overview tab
        overview_frame = ttk.Frame(self._notebook)
        self._notebook.add(overview_frame, text="Overview")

        self._overview_canvas = tk.Canvas(overview_frame, highlightthickness=0, bg="#333333")
        self._overview_canvas.pack(fill="both", expand=True)

        if self._ctx and self._ctx.overview_thumb is None:
            ttk.Label(overview_frame, text="No overview image", foreground="gray").pack(pady=10)

        # Metrics panel (below notebook)
        metrics_frame = ttk.LabelFrame(right_pane, text="Metrics", padding=4)
        metrics_frame.pack(fill="x", padx=2, pady=(2, 4))

        self._metrics_labels: dict[str, ttk.Label] = {}
        metric_rows = [
            ("Score", "score"),
            ("Tier", "tier"),
            ("R", "r_val"),
            ("G", "g_val"),
            ("B", "b_val"),
            ("Size", "size"),
            ("CalDist", "cal_dist"),
            ("PerimRatio", "perim_ratio"),
            ("AspectRatio", "aspect_ratio"),
            ("Entropy", "entropy"),
            ("GradEnergy", "grad_energy"),
            ("Kurtosis", "kurtosis"),
            ("Stage", "stage"),
        ]
        for label_text, key in metric_rows:
            row_frame = ttk.Frame(metrics_frame)
            row_frame.pack(fill="x", pady=1)
            ttk.Label(row_frame, text=f"{label_text}:", width=11, anchor="e").pack(side="left")
            val_label = ttk.Label(row_frame, text="—", anchor="w")
            val_label.pack(side="left", padx=(4, 0))
            self._metrics_labels[key] = val_label

        # ── Bottom bar ──
        bottom_bar = ttk.Frame(self)
        bottom_bar.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(
            bottom_bar,
            text="← / → navigate    G good    B bad    U unmark    Scroll zoom    Drag pan    Esc close",
            foreground="gray",
            font=("TkDefaultFont", 8),
        ).pack(side="left")

    def _annotation_key(self) -> str:
        """Generate annotation key for the current detection."""
        chip_idx = self._det["chip_idx"]
        frame = self._det["frame"]
        det_id = self._det["det_id"]
        return f"chip{chip_idx}_{frame}:{det_id}"

    def _update_title_bar(self):
        """Update the title label with current position info."""
        total = self._ctx.total_count if self._ctx else 0
        chip_idx = self._det["chip_idx"]
        frame = self._det["frame"]
        det_id = self._det["det_id"]
        text = f"#{self._grid_idx + 1}/{total}  C{chip_idx} {frame} d{det_id}"
        self._title_label.configure(text=text)
        self.title(f"Inspector — {text}")

    def _update_annotation_buttons(self):
        """Update Good/Bad button relief to reflect current annotation."""
        key = self._annotation_key()
        annotation = self._ctx.annotations.get(key) if self._ctx else None
        self._btn_good.configure(relief=tk.SUNKEN if annotation == "good" else tk.RAISED)
        self._btn_bad.configure(relief=tk.SUNKEN if annotation == "bad" else tk.RAISED)

    def _mark(self, label: str | None):
        """Mark the current detection as good/bad/unmarked."""
        if not self._ctx:
            return
        key = self._annotation_key()
        current = self._ctx.annotations.get(key)
        if label is not None and current == label:
            label = None  # toggle off if already set
        self._ctx.on_annotation_change(key, label)
        self._update_annotation_buttons()

    def _update_overview_dot(self):
        """Draw overview thumbnail with red dot at detection's stage position."""
        if not self._ctx or not self._ctx.overview_thumb:
            return
        thumb = self._ctx.overview_thumb.copy()
        meta = self._ctx.stitch_meta

        stage_x = self._det.get("stage_x")
        stage_y = self._det.get("stage_y")

        if meta and stage_x is not None and stage_y is not None:
            bounds = meta.get("stage_bounds_um", {})
            scale = meta.get("scale_um_per_px", 1.0)
            full_w, full_h = self._ctx.overview_size_px or thumb.size

            # Stage → full overview pixel coords
            px_x = (stage_x - bounds.get("x_min", 0)) / scale
            px_y = (stage_y - bounds.get("y_min", 0)) / scale

            # Full overview → thumbnail coords
            thumb_w, thumb_h = thumb.size
            tx = px_x * thumb_w / full_w
            ty = px_y * thumb_h / full_h

            draw = ImageDraw.Draw(thumb)
            r = 5
            draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill="red", outline="white")

        self._overview_photo = ImageTk.PhotoImage(thumb)
        self._overview_canvas.delete("all")
        self._overview_canvas.create_image(0, 0, anchor="nw", image=self._overview_photo)

    def _update_metrics(self):
        """Update the metrics panel labels with current detection values."""
        d = self._det
        um2 = self._ctx.um_per_px if self._ctx else 0.36

        r_val, g_val, b_val = d["contrast_rgb"]

        vals = {
            "score": f"{d.get('score', 0):.2f}",
            "tier": f"T{d.get('tier', '?')}",
            "r_val": f"{r_val:+.3f}",
            "g_val": f"{g_val:+.3f}",
            "b_val": f"{b_val:+.3f}",
            "size": f"{d['size_px'] * um2**2:.0f} µm²",
            "cal_dist": f"{d['cal_dist']:.3f}",
            "perim_ratio": f"{d['perim_ratio']:.2f}",
            "aspect_ratio": f"{d['aspect_ratio']:.2f}",
            "entropy": f"{d.get('entropy', d.get('g_entropy', 0)):.2f}",
            "grad_energy": f"{d['grad_energy']:.1f}",
            "kurtosis": f"{max(d['r_kurt'], d['g_kurt'], d['b_kurt']):.1f}",
        }

        sx = d.get("stage_x")
        sy = d.get("stage_y")
        vals["stage"] = f"({sx:.0f}, {sy:.0f}) µm" if sx is not None and sy is not None else "N/A"

        for key, text in vals.items():
            if key in self._metrics_labels:
                self._metrics_labels[key].configure(text=text)

    def _revisit_key(self) -> tuple[int, str, int]:
        """Key into the revisit lookup for the current detection."""
        return (
            self._det["chip_idx"],
            self._det["frame"],
            self._det["det_id"],
        )

    def _update_revisit_tabs(self):
        """Add/remove revisit image tabs based on current detection."""
        # Remember which tab is selected so we can restore it
        selected_text = None
        with contextlib.suppress(tk.TclError, KeyError):
            selected_text = self._notebook.tab(self._notebook.select(), "text")

        # Remove old revisit tabs
        for tab_widget in self._revisit_tabs.values():
            self._notebook.forget(tab_widget)
            tab_widget.destroy()
        self._revisit_tabs.clear()
        self._revisit_photos.clear()

        if not self._ctx:
            return
        mags = self._ctx.revisit_lookup.get(self._revisit_key(), {})
        for mag in sorted(mags, key=lambda m: int(m.rstrip("x"))):
            img_path = mags[mag]
            tab_frame = ttk.Frame(self._notebook)
            self._notebook.add(tab_frame, text=mag)
            self._revisit_tabs[mag] = tab_frame

            try:
                img = Image.open(img_path)
                img.load()
                w, h = img.size
                max_dim = CONTEXT_PANE_WIDTH - 10
                scale = min(max_dim / w, max_dim / h)
                if scale < 1.0:
                    thumb = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                else:
                    thumb = img
                photo = ImageTk.PhotoImage(thumb)
                self._revisit_photos[mag] = photo
                lbl = ttk.Label(tab_frame, image=photo, cursor="hand2")
                lbl.pack(fill="both", expand=True)
                lbl.bind("<Button-1>", lambda _e, p=img_path: ImagePopup(self, p))
            except Exception:
                ttk.Label(tab_frame, text=f"Could not load {mag}", foreground="red").pack(pady=10)

        # Restore previously selected tab if still present
        if selected_text and selected_text in self._revisit_tabs:
            self._notebook.select(self._revisit_tabs[selected_text])

    def _draw_overlay(self, draw: ImageDraw.ImageDraw, crop_x0: float, crop_y0: float, z: float):
        """Overlay callback for ZoomableCanvas: draw contour and bbox."""
        if self._contour and len(self._contour) >= 3:
            pts = [(int((x - crop_x0) * z), int((y - crop_y0) * z)) for x, y in self._contour]
            draw.polygon(pts, outline="cyan")
        bx, by, bw, bh = self._det["bbox"]
        rx0 = int((bx - BBOX_PAD_PX - crop_x0) * z)
        ry0 = int((by - BBOX_PAD_PX - crop_y0) * z)
        rx1 = int((bx + bw + BBOX_PAD_PX - crop_x0) * z)
        ry1 = int((by + bh + BBOX_PAD_PX - crop_y0) * z)
        draw.rectangle([rx0, ry0, rx1, ry1], outline="lime", width=2)

    def _navigate(self, delta: int):
        if not self._on_navigate:
            return
        result = self._on_navigate(delta)
        if result is None:
            return
        src_image, det, title, contour = result
        self._det = det
        self._contour = contour
        self._grid_idx += delta

        # Re-center on new detection
        bx, by, bw, bh = det["bbox"]
        self._frame_canvas.set_image(src_image, center=(bx + bw / 2.0, by + bh / 2.0))

        self._update_title_bar()
        self._update_annotation_buttons()
        self._update_overview_dot()
        self._update_metrics()
        self._update_revisit_tabs()


# ── GUI ──────────────────────────────────────────────────────────────


class RunViewerGUI:
    def __init__(self, root: tk.Tk, scans_dir: Path):
        self.root = root
        self.scans_dir = scans_dir
        self.root.title("FlakeFinder Run Viewer")
        self.root.minsize(1000, 650)

        # Image references (prevent GC)
        self._overview_photo: ImageTk.PhotoImage | None = None
        self._crop_photos: list[ImageTk.PhotoImage] = []

        # Current state
        self._runs: list[RunInfo] = []
        self._selected_run: RunInfo | None = None
        self._run_list_visible = True

        # Filter state
        self._all_detections: list[Detection] = []
        self._chip_scan_dirs: dict[int, Path | None] = {}
        self._frame_cache: dict[tuple[int, str], Image.Image] = {}
        self._filter_debounce_id: str | None = None
        self._filter_val_labels: list[tuple[ttk.Label, tk.DoubleVar, str]] = []
        self._filtered_crops_frame: ttk.LabelFrame | None = None
        self._filtered_table_frame: ttk.LabelFrame | None = None
        self._filter_count_var = tk.StringVar(value="")
        self._filter_preset_config: DetectorConfig | None = None
        self._um_per_px: float = 0.36
        self._filter_gen: int = 0  # generation counter for worker thread staleness
        self._thumb_pool = ThreadPoolExecutor(max_workers=4)
        self._thumb_cell_refs: list[tk.Frame] = []  # border frames indexed by grid_idx
        self._det_table: ttk.Treeview | None = None  # persistent treeview widget
        self._grid_frame: ttk.Frame | None = None  # persistent grid container
        self._grid_cells: list[dict] = []  # reusable cell widgets

        # Chip toggle state
        self._active_chips: set[int] = set()  # empty = all shown
        self._chip_buttons: dict[int, tk.Button] = {}

        # Tier toggle state
        self._active_tiers: set[int] = set()  # empty = all shown
        self._tier_buttons: dict[int, tk.Button] = {}
        self._fv_sliders_enabled = tk.BooleanVar(value=True)
        self._sliders_frame: ttk.Frame | None = None
        self._n_chips: int = 0
        self._last_top: list[Detection] = []
        self._last_thumb_cols: int = 0
        self._last_canvas_w: int = 0

        # Overview stitch for inspector locator
        self._overview_stitch_thumb: Image.Image | None = None
        self._overview_stitch_size_px: tuple[int, int] | None = None
        self._overview_stitch_meta: dict | None = None

        # Revisit images
        self._revisit_lookup: RevisitLookup = {}

        # Annotations
        self._annotations: dict[str, str] = {}
        self._thumb_borders: dict[str, tk.Frame] = {}

        self._state_path = scans_dir / ".viewer_state.json"

        self._build_ui()
        self._restore_geometry()
        self._refresh_runs()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _restore_geometry(self):
        """Restore window geometry from saved state."""
        if self._state_path.exists():
            with contextlib.suppress(Exception):
                with open(self._state_path) as f:
                    state = json.load(f)
                if geo := state.get("geometry"):
                    self.root.geometry(geo)

    def _save_geometry(self):
        """Save window geometry to state file."""
        with contextlib.suppress(Exception):
            state = {}
            if self._state_path.exists():
                with open(self._state_path) as f:
                    state = json.load(f)
            state["geometry"] = self.root.geometry()
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)

    def _on_close(self):
        """Save state and exit."""
        self._save_geometry()
        self.root.destroy()

    def _build_ui(self):
        # Main horizontal paned window
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Left panel: Run list ─────────────────────────────────
        self._left_frame = ttk.Frame(self.paned)
        self.paned.add(self._left_frame, weight=1)

        # Refresh button
        btn_row = ttk.Frame(self._left_frame)
        btn_row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Button(btn_row, text="↻ Refresh", command=self._refresh_runs).pack(side="left")
        self._run_count_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self._run_count_var, foreground="gray").pack(side="right")

        # Treeview
        columns = ("date", "scan_name", "operator", "preset", "chips", "detections", "duration", "notes")
        self.tree = ttk.Treeview(self._left_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("date", text="Date/Time")
        self.tree.heading("scan_name", text="Scan Name")
        self.tree.heading("operator", text="Operator")
        self.tree.heading("preset", text="Preset")
        self.tree.heading("chips", text="Chips")
        self.tree.heading("detections", text="Det")
        self.tree.heading("duration", text="Dur")
        self.tree.heading("notes", text="Notes")

        self.tree.column("date", width=120, minwidth=90)
        self.tree.column("scan_name", width=130, minwidth=80)
        self.tree.column("operator", width=70, minwidth=50)
        self.tree.column("preset", width=55, minwidth=40)
        self.tree.column("chips", width=35, minwidth=30, anchor="center")
        self.tree.column("detections", width=45, minwidth=35, anchor="center")
        self.tree.column("duration", width=50, minwidth=40, anchor="center")
        self.tree.column("notes", width=140, minwidth=80)

        tree_scroll = ttk.Scrollbar(self._left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y", padx=(0, 4), pady=(0, 4))
        self.tree.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 4))

        self.tree.bind("<<TreeviewSelect>>", self._on_run_selected)

        # ── Right panel: Run detail (scrollable) ─────────────────
        right_frame = ttk.Frame(self.paned)
        self.paned.add(right_frame, weight=3)

        # Scrollable canvas
        self._detail_canvas = tk.Canvas(right_frame, highlightthickness=0)
        detail_scroll = ttk.Scrollbar(right_frame, orient="vertical", command=self._detail_canvas.yview)
        self._detail_canvas.configure(yscrollcommand=detail_scroll.set)
        detail_scroll.pack(side="right", fill="y")
        self._detail_canvas.pack(fill="both", expand=True)

        self._detail_frame = ttk.Frame(self._detail_canvas)
        self._detail_window = self._detail_canvas.create_window((0, 0), window=self._detail_frame, anchor="nw")

        self._detail_frame.bind("<Configure>", self._on_detail_configure)
        self._detail_canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling — bind once, gate with hover flag
        self._scroll_active = False
        self._detail_canvas.bind("<Enter>", lambda _: self._set_scroll_active(True))
        self._detail_canvas.bind("<Leave>", lambda _: self._set_scroll_active(False))
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux)

        # Placeholder
        self._placeholder = ttk.Label(self._detail_frame, text="Select a run to view details", foreground="gray")
        self._placeholder.pack(pady=40)

    def _set_scroll_active(self, active: bool):
        self._scroll_active = active

    def _on_mousewheel(self, event):
        if self._scroll_active:
            self._detail_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_mousewheel_linux(self, event):
        if not self._scroll_active:
            return
        if event.num == 4:
            self._detail_canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self._detail_canvas.yview_scroll(3, "units")

    def _on_detail_configure(self, _event):
        self._detail_canvas.configure(scrollregion=self._detail_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        if event.width != self._last_canvas_w:
            self._last_canvas_w = event.width
            self._detail_canvas.itemconfig(self._detail_window, width=event.width)
            # Re-layout thumbnails if column count changed
            new_cols = max(1, (event.width - 32) // (CROP_THUMB_SIZE + 16))
            if new_cols != self._last_thumb_cols and self._last_top:
                self._last_thumb_cols = new_cols
                self._apply_filters()

    # ── Run list ─────────────────────────────────────────────────

    def _refresh_runs(self):
        self._runs = _discover_runs(self.scans_dir)
        self.tree.delete(*self.tree.get_children())
        for i, run in enumerate(self._runs):
            date_str = run.timestamp.strftime("%Y-%m-%d %H:%M") if run.timestamp else run.name
            det_str = str(run.total_detections) if run.total_detections > 0 else "—"
            dur_str = _fmt_duration(run.duration_s) if run.duration_s > 0 else "—"
            # Show first line of notes, truncated
            notes_short = run.notes.split("\n")[0][:30] if run.notes else ""
            self.tree.insert(
                "",
                "end",
                iid=str(i),
                values=(date_str, run.scan_name, run.operator, run.preset, run.n_chips, det_str, dur_str, notes_short),
            )
        self._run_count_var.set(f"{len(self._runs)} runs")

    def _on_run_selected(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        run = self._runs[idx]
        if run is self._selected_run:
            return
        self._selected_run = run
        self._hide_run_list()
        self._show_run_detail(run)

    def _hide_run_list(self):
        """Hide the left panel to give detail view full width."""
        with contextlib.suppress(tk.TclError):
            self.paned.forget(self._left_frame)
        self._run_list_visible = False

    def _show_run_list(self):
        """Restore the left panel."""
        with contextlib.suppress(tk.TclError):
            self.paned.insert(0, self._left_frame, weight=1)
        self._run_list_visible = True

    def _toggle_run_list(self):
        """Toggle the run list sidebar."""
        if self._run_list_visible:
            self._hide_run_list()
        else:
            self._show_run_list()

    # ── Run detail ───────────────────────────────────────────────

    def _clear_detail(self):
        if self._filter_debounce_id is not None:
            self.root.after_cancel(self._filter_debounce_id)
            self._filter_debounce_id = None
        for w in self._detail_frame.winfo_children():
            w.destroy()
        self._overview_photo = None
        self._crop_photos = []
        self._all_detections = []
        self._chip_scan_dirs = {}
        self._frame_cache = {}
        self._filter_val_labels = []
        self._active_chips = set()
        self._chip_buttons = {}
        self._active_tiers = set()
        self._tier_buttons = {}
        self._last_top = []
        self._last_thumb_cols = 0
        self._overview_stitch_thumb = None
        self._overview_stitch_size_px = None
        self._overview_stitch_meta = None
        self._revisit_lookup = {}
        self._annotations = {}
        self._thumb_borders = {}
        self._thumb_cell_refs = []
        self._det_table = None
        self._grid_frame = None
        self._grid_cells = []
        self._filter_gen += 1  # invalidate any in-flight worker

    def _show_run_detail(self, run: RunInfo):
        self._clear_detail()
        self._detail_canvas.yview_moveto(0)
        self._n_chips = run.n_chips

        # ── Phase 1: lightweight skeleton (renders immediately) ──

        # Back button
        btn = ttk.Button(self._detail_frame, text="← Runs", command=self._toggle_run_list)
        btn.pack(anchor="w", padx=8, pady=(4, 0))

        # Run header
        header = ttk.LabelFrame(self._detail_frame, text=run.name, padding=8)
        header.pack(fill="x", padx=8, pady=(8, 4))

        info_lines = []
        if run.scan_name or run.operator:
            meta_parts = []
            if run.scan_name:
                meta_parts.append(f"Scan: {run.scan_name}")
            if run.operator:
                meta_parts.append(f"Operator: {run.operator}")
            info_lines.append("    ".join(meta_parts))
        if run.timestamp:
            info_lines.append(f"Date: {run.timestamp.strftime('%Y-%m-%d %H:%M')}")
        info_lines.append(f"Preset: {run.preset}    Chips: {run.n_chips}    Detections: {run.total_detections}")
        if run.duration_s > 0:
            info_lines.append(f"Duration: {_fmt_duration(run.duration_s)}")
        if run.notes:
            info_lines.append(f"Notes: {run.notes}")
        ttk.Label(header, text="\n".join(info_lines), justify="left").pack(anchor="w")

        # Overview placeholder
        self._overview_container = ttk.LabelFrame(self._detail_frame, text="Overview", padding=4)
        overview_path, overview_stitch_path = self._find_overview_image(run.path)
        if overview_path:
            self._overview_container.pack(fill="x", padx=8, pady=4)
            ttk.Label(self._overview_container, text="Loading…", foreground="gray").pack()

        # Filter panel (cheap — just slider widgets)
        self._build_filter_panel(run.material)

        # Chip toggle buttons
        if run.n_chips > 0:
            chips_frame = ttk.LabelFrame(self._detail_frame, text="Chips", padding=4)
            chips_frame.pack(fill="x", padx=8, pady=4)

            btn_row = ttk.Frame(chips_frame)
            btn_row.pack(fill="x")
            for i in range(run.n_chips):
                btn = tk.Button(
                    btn_row,
                    text=f"Chip {i}",
                    relief=tk.RAISED,
                    command=lambda ci=i: self._toggle_chip(ci),
                    padx=6,
                    pady=2,
                )
                btn.pack(side="left", padx=2, pady=2)
                self._chip_buttons[i] = btn

        # Detection frames with loading indicator
        self._filtered_crops_frame = ttk.LabelFrame(self._detail_frame, text="Top Detections", padding=4)
        self._filtered_crops_frame.pack(fill="x", padx=8, pady=4)
        self._loading_bar = ttk.Progressbar(self._filtered_crops_frame, mode="indeterminate", length=200)
        self._loading_bar.pack(pady=8)
        self._loading_bar.start(15)

        self._filtered_table_frame = ttk.LabelFrame(self._detail_frame, text="Scoring Data", padding=4)

        # ── Phase 2: deferred heavy I/O (chained to let event loop paint) ──
        self.root.after(1, self._load_overview, run, overview_path, overview_stitch_path)

    def _load_overview(self, run: RunInfo, overview_path: Path | None, stitch_path: Path | None = None):
        """Phase 2a: load overview image, then chain to detection loading."""
        if self._selected_run is not run:
            return

        if overview_path and self._overview_container.winfo_exists():
            for w in self._overview_container.winfo_children():
                w.destroy()
            # Thumbnail shows detected (with chip boxes); popup opens high-res stitch
            self._load_image_into_label(
                overview_path,
                self._overview_container,
                OVERVIEW_MAX_WIDTH,
                "_overview_photo",
                popup_path=stitch_path or overview_path,
            )

        self.root.after(1, self._load_detections, run)

    def _load_detections(self, run: RunInfo):
        """Phase 2b: load detection data, overview stitch, annotations, and apply filters."""
        if self._selected_run is not run:
            return

        self._all_detections, self._chip_scan_dirs, self._um_per_px = _load_all_detections(run.path, run.n_chips)
        self._revisit_lookup = _build_revisit_lookup(run.path, run.n_chips)
        self._load_overview_stitch(run.path)
        self._load_annotations(run.path)

        # Apply filters (replaces loading bar with thumbnails)
        if self._all_detections and self._filtered_crops_frame and self._filtered_table_frame:
            self._filtered_table_frame.pack(fill="x", padx=8, pady=4)
            self._apply_filters()
        elif self._filtered_crops_frame:
            # No detections — replace loading bar with message
            for w in self._filtered_crops_frame.winfo_children():
                w.destroy()
            ttk.Label(self._filtered_crops_frame, text="No detections", foreground="gray").pack(anchor="w")

    def _find_overview_image(self, run_dir: Path) -> tuple[Path | None, Path | None]:
        """Find the overview detection image and high-res stitch.

        Returns (detected_path, stitch_path).
        """
        detected = None
        for p in run_dir.glob("overview_*_stitch_chips_detected.png"):
            detected = p
            break
        stitch = None
        for p in run_dir.glob("overview_*_stitch.jpg"):
            stitch = p
            break
        return detected, stitch

    # ── Overview stitch + annotations ────────────────────────────

    def _load_overview_stitch(self, run_dir: Path):
        """Load overview stitch image + meta for the inspector's locator dot."""
        self._overview_stitch_thumb = None
        self._overview_stitch_size_px = None
        self._overview_stitch_meta = None

        # Find stitch image (prefer the plain stitch, not _chips_detected)
        stitch_path = None
        for p in run_dir.glob("overview_*_stitch.jpg"):
            stitch_path = p
            break
        if stitch_path is None:
            return

        # Load companion meta
        meta_name = stitch_path.stem + "_meta.json"
        meta_path = run_dir / meta_name
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    self._overview_stitch_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        try:
            img = Image.open(stitch_path)
            img.load()
            self._overview_stitch_size_px = img.size
            # Downscale to thumbnail
            w, h = img.size
            scale = min(OVERVIEW_THUMB_MAX / w, OVERVIEW_THUMB_MAX / h)
            if scale < 1.0:
                thumb = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            else:
                thumb = img
            self._overview_stitch_thumb = thumb
        except Exception:
            pass

    def _load_annotations(self, run_dir: Path):
        """Load annotations from annotations.json in the run directory."""
        self._annotations = {}
        ann_path = run_dir / "annotations.json"
        if ann_path.exists():
            try:
                with open(ann_path) as f:
                    self._annotations = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    def _save_annotations(self):
        """Save annotations to annotations.json in the current run directory."""
        if not self._selected_run:
            return
        ann_path = self._selected_run.path / "annotations.json"
        try:
            with open(ann_path, "w") as f:
                json.dump(self._annotations, f, indent=2)
        except OSError:
            pass

    def _on_annotation_change(self, key: str, label: str | None):
        """Handle annotation change from FlakeInspector."""
        if label is None:
            self._annotations.pop(key, None)
        else:
            self._annotations[key] = label
        self._save_annotations()
        self._update_thumb_border(key)

    def _update_thumb_border(self, key: str):
        """Update a single thumbnail's border color based on annotation."""
        border_frame = self._thumb_borders.get(key)
        if border_frame is None:
            return
        annotation = self._annotations.get(key)
        if annotation == "good":
            border_frame.configure(highlightbackground="green", highlightthickness=3)
        elif annotation == "bad":
            border_frame.configure(highlightbackground="red", highlightthickness=3)
        else:
            border_frame.configure(highlightthickness=0)

    def _detection_annotation_key(self, d: Detection) -> str:
        """Generate annotation key for a detection dict."""
        chip_idx = d["chip_idx"]
        frame = d["frame"]
        det_id = d["det_id"]
        return f"chip{chip_idx}_{frame}:{det_id}"

    # ── Chip toggles ─────────────────────────────────────────────

    def _toggle_chip(self, chip_idx: int):
        """Toggle a chip filter. Empty active set = show all."""
        if chip_idx in self._active_chips:
            self._active_chips.discard(chip_idx)
        else:
            self._active_chips.add(chip_idx)
        self._update_chip_button_visuals()
        self._on_filter_change()

    def _update_chip_button_visuals(self):
        """Update chip button relief to reflect active state."""
        for idx, btn in self._chip_buttons.items():
            if idx in self._active_chips:
                btn.configure(relief=tk.SUNKEN)
            else:
                btn.configure(relief=tk.RAISED)

    def _get_visible_chips(self) -> set[int] | None:
        """Return set of visible chip indices, or None for 'show all'."""
        if not self._active_chips:
            return None  # empty = all
        return self._active_chips

    # ── Tier toggles ─────────────────────────────────────────────

    def _toggle_tier(self, tier: int):
        """Toggle a tier filter. Exclusive: only one tier active at a time."""
        if tier in self._active_tiers:
            self._active_tiers.discard(tier)
        else:
            self._active_tiers = {tier}
            # Auto-disable sliders when a tier is selected
            if self._fv_sliders_enabled.get():
                self._fv_sliders_enabled.set(False)
                self._on_sliders_toggle()
        self._update_tier_button_visuals()
        self._on_filter_change()

    def _update_tier_button_visuals(self):
        """Update tier button relief to reflect active state."""
        for tier, btn in self._tier_buttons.items():
            btn.configure(relief=tk.SUNKEN if tier in self._active_tiers else tk.RAISED)

    def _on_sliders_toggle(self):
        """Enable/disable slider filtering and dim the slider widgets."""
        enabled = self._fv_sliders_enabled.get()
        if self._sliders_frame:
            state = "normal" if enabled else "disabled"
            for child in self._sliders_frame.winfo_children():
                for widget in child.winfo_children():
                    with contextlib.suppress(tk.TclError):
                        widget.configure({"state": state})
        self._on_filter_change()

    # ── Filter panel ─────────────────────────────────────────────

    def _build_filter_panel(self, preset: str):
        """Create the filter panel with sliders for interactive detection tuning."""
        try:
            config = DetectorConfig.from_material(preset)
        except ValueError:
            config = DetectorConfig.hbn_thin()
        self._filter_preset_config = config

        filter_frame = ttk.LabelFrame(self._detail_frame, text="Filters", padding=6)
        filter_frame.pack(fill="x", padx=8, pady=4)

        # Tier toggle buttons + slider enable checkbox
        tier_row = ttk.Frame(filter_frame)
        tier_row.pack(fill="x", pady=(0, 4))
        ttk.Label(tier_row, text="Tier:").pack(side="left", padx=(0, 4))
        for tier in (1, 2):
            btn = tk.Button(
                tier_row,
                text=f"T{tier}",
                relief=tk.RAISED,
                command=lambda t=tier: self._toggle_tier(t),
                padx=8,
                pady=2,
            )
            btn.pack(side="left", padx=2)
            self._tier_buttons[tier] = btn

        self._fv_sliders_enabled.set(True)
        ttk.Checkbutton(
            tier_row,
            text="Sliders",
            variable=self._fv_sliders_enabled,
            command=self._on_sliders_toggle,
        ).pack(side="left", padx=(12, 0))

        # Compute clamped defaults from preset (reused by _reset_filters)
        self._filter_defaults = {
            "perim_ratio": config.tier1_perim_ratio,
            "cal_dist": config.tier1_cal_dist,
            "g_min": max(config.tier1_g_min, SLIDER_MIN_G),
            "g_max": min(config.tier1_g_max, SLIDER_MAX_G),
            "r_max": max(config.tier1_r_max, SLIDER_MIN_R),
            "entropy": min(config.tier1_entropy_max, SLIDER_MAX_ENTROPY),
            "min_size": config.tier1_min_size_um2,
            "grad_energy": SLIDER_MAX_GRAD_ENERGY,
            "aspect_ratio": SLIDER_MAX_ASPECT_RATIO,
            "kurtosis": SLIDER_MAX_KURTOSIS,
        }
        d = self._filter_defaults

        # Filter DoubleVars
        self._fv_perim_ratio = tk.DoubleVar(value=d["perim_ratio"])
        self._fv_cal_dist = tk.DoubleVar(value=d["cal_dist"])
        self._fv_g_min = tk.DoubleVar(value=d["g_min"])
        self._fv_g_max = tk.DoubleVar(value=d["g_max"])
        self._fv_r_max = tk.DoubleVar(value=d["r_max"])
        self._fv_entropy = tk.DoubleVar(value=d["entropy"])
        self._fv_min_size = tk.DoubleVar(value=d["min_size"])
        self._fv_grad_energy = tk.DoubleVar(value=d["grad_energy"])
        self._fv_aspect_ratio = tk.DoubleVar(value=d["aspect_ratio"])
        self._fv_kurtosis = tk.DoubleVar(value=d["kurtosis"])
        self._fv_top_n = tk.IntVar(value=DEFAULT_FILTER_TOP_N)

        # Slider definitions: (row, col, label, var, from_, to, resolution, fmt)
        slider_defs = [
            (0, 0, "perim_ratio \u2264", self._fv_perim_ratio, 1.0, 3.0, 0.05, "{:.2f}"),
            (0, 1, "cal_dist \u2264", self._fv_cal_dist, 0.0, 2.0, 0.05, "{:.2f}"),
            (0, 2, "min \u00b5m\u00b2 \u2265", self._fv_min_size, 0, 2000, 10, "{:.0f}"),
            (1, 0, "G min \u2265", self._fv_g_min, SLIDER_MIN_G, SLIDER_MAX_G, 0.1, "{:+.1f}"),
            (1, 1, "G max \u2264", self._fv_g_max, SLIDER_MIN_G, SLIDER_MAX_G, 0.1, "{:+.1f}"),
            (1, 2, "R max \u2264", self._fv_r_max, SLIDER_MIN_R, SLIDER_MAX_G, 0.1, "{:+.1f}"),
            (2, 0, "entropy \u2264", self._fv_entropy, 0.0, SLIDER_MAX_ENTROPY, 0.1, "{:.1f}"),
            (2, 1, "grad_energy \u2264", self._fv_grad_energy, 0.0, SLIDER_MAX_GRAD_ENERGY, 0.5, "{:.1f}"),
            (2, 2, "aspect_ratio \u2264", self._fv_aspect_ratio, 1.0, SLIDER_MAX_ASPECT_RATIO, 0.5, "{:.1f}"),
            (3, 0, "kurtosis \u2264", self._fv_kurtosis, -2.0, SLIDER_MAX_KURTOSIS, 1.0, "{:.0f}"),
        ]

        sliders_frame = ttk.Frame(filter_frame)
        sliders_frame.pack(fill="x")
        self._sliders_frame = sliders_frame
        for col in range(3):
            sliders_frame.columnconfigure(col, weight=1)

        self._filter_val_labels = []
        for row, col, label, var, from_, to, resolution, fmt in slider_defs:
            cell = ttk.Frame(sliders_frame)
            cell.grid(row=row, column=col, sticky="ew", padx=4, pady=1)
            ttk.Label(cell, text=label, width=13, anchor="e").pack(side="left")

            scale = tk.Scale(
                cell,
                from_=from_,
                to=to,
                resolution=resolution,
                orient="horizontal",
                variable=var,
                showvalue=False,
                length=130,
                command=self._on_any_slider_change,
            )
            scale.pack(side="left", padx=(2, 0))

            val_lbl = ttk.Label(cell, text=fmt.format(var.get()), width=7, anchor="w")
            val_lbl.pack(side="left", padx=(2, 0))
            self._filter_val_labels.append((val_lbl, var, fmt))

        # Bottom row: count + top N + reset
        bottom = ttk.Frame(filter_frame)
        bottom.pack(fill="x", pady=(4, 0))

        self._filter_count_var.set("")
        ttk.Label(bottom, textvariable=self._filter_count_var, font=("TkDefaultFont", 10, "bold")).pack(
            side="left", padx=4
        )

        ttk.Button(bottom, text="Reset", command=self._reset_filters).pack(side="right", padx=4)

        top_n_frame = ttk.Frame(bottom)
        top_n_frame.pack(side="right")
        ttk.Label(top_n_frame, text="Top N:").pack(side="left")
        top_n_spin = ttk.Spinbox(
            top_n_frame, from_=1, to=200, textvariable=self._fv_top_n, width=4, command=self._on_filter_change
        )
        top_n_spin.pack(side="left", padx=(2, 4))
        top_n_spin.bind("<Return>", lambda _: self._on_filter_change())
        top_n_spin.bind("<FocusOut>", lambda _: self._on_filter_change())

    def _on_any_slider_change(self, _value=None):
        """Called on every slider move — update value labels and debounce filter."""
        for lbl, var, fmt in self._filter_val_labels:
            lbl.configure(text=fmt.format(var.get()))
        self._on_filter_change()

    def _on_filter_change(self):
        """Debounced filter trigger."""
        if self._filter_debounce_id is not None:
            self.root.after_cancel(self._filter_debounce_id)
        self._filter_debounce_id = self.root.after(FILTER_DEBOUNCE_MS, self._apply_filters)

    def _apply_filters(self):
        """Kick off phase-1 worker: filter + dedup off the main thread."""
        self._filter_debounce_id = None
        if not self._all_detections:
            return

        # Bump generation to invalidate any in-flight workers
        self._filter_gen += 1
        gen = self._filter_gen

        # Snapshot all filter state on the main thread (Tk vars aren't thread-safe)
        visible_chips = self._get_visible_chips()
        active_tiers = self._active_tiers.copy() or None
        sliders_on = self._fv_sliders_enabled.get()
        pr_max = self._fv_perim_ratio.get()
        cd_max = self._fv_cal_dist.get()
        g_min = self._fv_g_min.get()
        g_max = self._fv_g_max.get()
        r_max = self._fv_r_max.get()
        ent_max = self._fv_entropy.get()
        min_size_um2 = self._fv_min_size.get()
        min_size_px = int(min_size_um2 / (self._um_per_px**2)) if min_size_um2 > 0 else 0
        ge_max = self._fv_grad_energy.get()
        ar_max = self._fv_aspect_ratio.get()
        kurt_max = self._fv_kurtosis.get()
        top_n = self._fv_top_n.get()
        all_dets = self._all_detections

        def phase1_worker():
            if visible_chips is not None:
                chip_filtered = [d for d in all_dets if d.get("chip_idx") in visible_chips]
            else:
                chip_filtered = all_dets
            if active_tiers is not None:
                chip_filtered = [d for d in chip_filtered if d.get("tier") in active_tiers]

            passing = []
            for d in chip_filtered:
                if sliders_on:
                    r, g = d["contrast_rgb"][0], d["contrast_rgb"][1]
                    if not (
                        d["perim_ratio"] <= pr_max
                        and d["cal_dist"] <= cd_max
                        and g >= g_min
                        and g <= g_max
                        and r <= r_max
                        and d.get("entropy", d.get("g_entropy", 0)) <= ent_max
                        and d["size_px"] >= min_size_px
                        and d["grad_energy"] <= ge_max
                        and d["aspect_ratio"] <= ar_max
                        and max(d["r_kurt"], d["g_kurt"], d["b_kurt"]) <= kurt_max
                    ):
                        continue
                passing.append(d)

            if self._filter_gen != gen:
                return

            passing.sort(key=lambda d: -d.get("score", 0))
            deduped = dedup_detections(passing)
            count_text = f"{len(deduped)} / {len(chip_filtered)} pass ({len(passing) - len(deduped)} dupes)"
            top = deduped[:top_n]

            if self._filter_gen != gen:
                return

            self.root.after(0, self._on_filter_phase1, gen, top, count_text)

        self._thumb_pool.submit(phase1_worker)

    def _on_filter_phase1(self, gen: int, top: list[Detection], count_text: str):
        """Main-thread: build grid skeleton with placeholders, fan out image loading."""
        if self._filter_gen != gen:
            return

        self._filter_count_var.set(count_text)
        self._last_top = top
        self._update_filtered_table(top)
        self._update_grid(top)

        # Fan out thumbnail loading to the pool
        for grid_idx, d in enumerate(top):
            frame_name = d.get("frame", "")
            chip_idx = d.get("chip_idx", 0)
            if not frame_name:
                continue
            self._thumb_pool.submit(self._thumb_worker, gen, grid_idx, chip_idx, frame_name, d)

    def _thumb_worker(self, gen: int, grid_idx: int, chip_idx: int, frame_name: str, d: Detection):
        """Pool worker: load frame, resize, draw bbox → post PIL image to main thread."""
        if self._filter_gen != gen:
            return

        frame_img = self._load_frame_cached(chip_idx, frame_name)
        if frame_img is None:
            return

        if self._filter_gen != gen:
            return

        fw, fh = frame_img.size
        scale = min(CROP_THUMB_SIZE / fw, CROP_THUMB_SIZE / fh)
        thumb = frame_img.resize((int(fw * scale), int(fh * scale)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        bx, by, bw, bh = d["bbox"]
        draw.rectangle(
            [
                int((bx - BBOX_PAD_PX) * scale),
                int((by - BBOX_PAD_PX) * scale),
                int((bx + bw + BBOX_PAD_PX) * scale),
                int((by + bh + BBOX_PAD_PX) * scale),
            ],
            outline="lime",
            width=2,
        )

        if self._filter_gen != gen:
            return

        self.root.after(0, self._on_thumb_ready, gen, grid_idx, thumb)

    def _on_thumb_ready(self, gen: int, grid_idx: int, pil_thumb: Image.Image):
        """Main-thread: insert a loaded thumbnail into its placeholder cell."""
        if self._filter_gen != gen:
            return
        if grid_idx >= len(self._thumb_cell_refs):
            return

        border_frame = self._thumb_cell_refs[grid_idx]
        if not border_frame.winfo_exists():
            return

        # Replace placeholder with image (frame stays fixed size)
        for w in border_frame.winfo_children():
            w.destroy()

        photo = ImageTk.PhotoImage(pil_thumb)
        self._crop_photos.append(photo)
        lbl = ttk.Label(border_frame, image=photo, cursor="hand2")
        lbl.place(relx=0.5, rely=0.5, anchor="center")
        lbl.bind("<Button-1>", lambda _e, idx=grid_idx: self._show_frame_popup_at(idx))

    def _reset_filters(self):
        """Reset all filter sliders to preset defaults and clear chip selection."""
        d = self._filter_defaults
        self._fv_perim_ratio.set(d["perim_ratio"])
        self._fv_cal_dist.set(d["cal_dist"])
        self._fv_g_min.set(d["g_min"])
        self._fv_g_max.set(d["g_max"])
        self._fv_r_max.set(d["r_max"])
        self._fv_entropy.set(d["entropy"])
        self._fv_min_size.set(d["min_size"])
        self._fv_grad_energy.set(d["grad_energy"])
        self._fv_aspect_ratio.set(d["aspect_ratio"])
        self._fv_kurtosis.set(d["kurtosis"])
        self._active_chips.clear()
        self._update_chip_button_visuals()
        self._active_tiers.clear()
        self._update_tier_button_visuals()
        for lbl, var, fmt in self._filter_val_labels:
            lbl.configure(text=fmt.format(var.get()))
        # Cancel pending debounce and apply immediately
        if self._filter_debounce_id is not None:
            self.root.after_cancel(self._filter_debounce_id)
            self._filter_debounce_id = None
        self._apply_filters()

    def _update_filtered_table(self, top: list[Detection]):
        """Repopulate the detection table rows (no widget destroy/rebuild)."""
        if self._filtered_table_frame is None:
            return

        # Create treeview once, reuse thereafter
        if self._det_table is None:
            self._create_detection_table(self._filtered_table_frame)

        table = self._det_table
        assert table is not None
        table.delete(*table.get_children())

        n_show = min(len(top), 50)
        for i, d in enumerate(top[:n_show]):
            r_val, g_val = d["contrast_rgb"][0], d["contrast_rgb"][1]
            table.insert(
                "",
                "end",
                values=(
                    i + 1,
                    d.get("chip_idx", "?"),
                    d.get("tier", "?"),
                    d.get("frame", "?"),
                    d["det_id"],
                    d["size_px"],
                    f"{r_val:+.3f}",
                    f"{g_val:+.3f}",
                    f"{d.get('score', 0):.3f}",
                    f"{d['cal_dist']:.3f}",
                    f"{d['grad_energy']:.1f}",
                    f"{d.get('entropy', d.get('g_entropy', 0)):.2f}",
                    f"{max(d['r_kurt'], d['g_kurt'], d['b_kurt']):.1f}",
                    f"{d['perim_ratio']:.2f}",
                    f"{d['aspect_ratio']:.1f}",
                ),
            )

    def _ensure_grid_cells(self, n: int):
        """Ensure at least *n* grid cells exist. Create the grid on first call."""
        if self._grid_frame is None:
            # Remove loading bar on first grid creation
            if self._loading_bar is not None:
                self._loading_bar.stop()
                self._loading_bar.destroy()
                self._loading_bar = None
            self._grid_frame = ttk.Frame(self._filtered_crops_frame)
            self._grid_frame.pack(fill="x")

        while len(self._grid_cells) < n:
            idx = len(self._grid_cells)
            cell = ttk.Frame(self._grid_frame, padding=2)

            border_frame = tk.Frame(cell, highlightthickness=0, width=CROP_THUMB_SIZE, height=CROP_THUMB_HEIGHT)
            border_frame.pack()
            border_frame.pack_propagate(False)

            # Placeholder
            placeholder = ttk.Label(border_frame, text="loading…", foreground="gray")
            placeholder.place(relx=0.5, rely=0.5, anchor="center")

            metric_label = ttk.Label(cell, text="", font=("TkDefaultFont", 7), justify="left")
            metric_label.pack(anchor="w")

            self._grid_cells.append(
                {
                    "cell": cell,
                    "border_frame": border_frame,
                    "metric_label": metric_label,
                    "grid_idx": idx,
                }
            )

    def _update_grid(self, top: list[Detection]):
        """Update the persistent grid cells with new detection data (main thread)."""
        if self._filtered_crops_frame is None:
            return

        self._crop_photos = []
        self._thumb_borders = {}
        self._thumb_cell_refs = []

        n = len(top)
        self._ensure_grid_cells(n)

        avail_w = self._detail_canvas.winfo_width() - 32
        cols = max(1, avail_w // (CROP_THUMB_SIZE + 16))

        # Update visible cells
        for grid_idx, d in enumerate(top):
            gc = self._grid_cells[grid_idx]
            cell = gc["cell"]
            border_frame = gc["border_frame"]
            metric_label = gc["metric_label"]

            # Position in grid
            row, col = divmod(grid_idx, cols)
            cell.grid(row=row * 2, column=col, sticky="nw")

            # Reset border frame: clear children, set annotation border
            for w in border_frame.winfo_children():
                w.destroy()
            ttk.Label(border_frame, text="loading…", foreground="gray").place(relx=0.5, rely=0.5, anchor="center")

            ann_key = self._detection_annotation_key(d)
            self._thumb_borders[ann_key] = border_frame
            self._thumb_cell_refs.append(border_frame)

            annotation = self._annotations.get(ann_key)
            if annotation == "good":
                border_frame.configure(highlightbackground="green", highlightthickness=3)
            elif annotation == "bad":
                border_frame.configure(highlightbackground="red", highlightthickness=3)
            else:
                border_frame.configure(highlightthickness=0)

            # Update metric label
            chip_idx = d.get("chip_idx", 0)
            r_val, g_val = d["contrast_rgb"][0], d["contrast_rgb"][1]
            size_um2 = d["size_px"] * self._um_per_px**2
            ent = d.get("entropy", d.get("g_entropy", 0))
            ge = d["grad_energy"]
            kurt = max(d["r_kurt"], d["g_kurt"], d["b_kurt"])
            line1 = f"#{grid_idx + 1} C{chip_idx} R={r_val:+.2f} G={g_val:+.2f} {size_um2:.0f}\u00b5m\u00b2"
            line2 = f"e={ent:.1f} g={ge:.1f} k={kurt:.0f}"
            metric_label.configure(text=f"{line1}\n{line2}")

        # Hide unused cells
        for grid_idx in range(n, len(self._grid_cells)):
            self._grid_cells[grid_idx]["cell"].grid_remove()

    def _load_frame_cached(self, chip_idx: int, frame_name: str) -> Image.Image | None:
        """Load a frame image with caching, keyed by (chip_idx, frame_name)."""
        key = (chip_idx, frame_name)
        if key in self._frame_cache:
            return self._frame_cache[key]

        scan_dir = self._chip_scan_dirs.get(chip_idx)
        if not scan_dir:
            return None

        frame_path = scan_dir / f"{frame_name}.jpg"
        if not frame_path.exists():
            return None

        try:
            img = Image.open(frame_path)
            img.load()  # Force full load so file handle is released
        except Exception:
            return None

        # LRU-ish eviction
        if len(self._frame_cache) >= FRAME_CACHE_MAX:
            oldest_key = next(iter(self._frame_cache))
            del self._frame_cache[oldest_key]

        self._frame_cache[key] = img
        return img

    def _load_frame_contour(self, chip_idx: int, frame_name: str, det_id: int) -> list[list[int]] | None:
        """Load contour points for a detection from per-frame seg JSON."""
        run_dir = self._selected_run.path if self._selected_run else None
        if not run_dir:
            return None
        frame_json_path = run_dir / f"chip_{chip_idx}" / "seg" / f"{frame_name}.json"
        if not frame_json_path.exists():
            return None
        try:
            with open(frame_json_path) as f:
                data = json.load(f)
            dets = data.get("detections", [])
            if det_id < len(dets):
                return dets[det_id].get("contour")
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _get_popup_data(self, grid_idx: int) -> tuple[Image.Image, Detection, str, list[list[int]] | None] | None:
        """Load image/detection/contour for a grid index. Returns None if invalid."""
        if not self._last_top or grid_idx < 0 or grid_idx >= len(self._last_top):
            return None
        d = self._last_top[grid_idx]
        chip_idx = d.get("chip_idx", 0)
        frame_name = d.get("frame", "")
        if not frame_name:
            return None
        frame_img = self._load_frame_cached(chip_idx, frame_name)
        if frame_img is None:
            return None
        det_id = d["det_id"]
        contour = self._load_frame_contour(chip_idx, frame_name, det_id)
        title = f"#{grid_idx + 1} C{chip_idx} {frame_name} d{det_id}"
        return frame_img, d, title, contour

    def _show_frame_popup_at(self, grid_idx: int):
        """Open a FlakeInspector for the detection at grid_idx in _last_top."""
        data = self._get_popup_data(grid_idx)
        if data is None:
            return
        frame_img, d, title, contour = data

        # Mutable cell so the closure tracks current index
        idx_cell = [grid_idx]

        def on_navigate(delta: int):
            new_idx = idx_cell[0] + delta
            result = self._get_popup_data(new_idx)
            if result is not None:
                idx_cell[0] = new_idx
            return result

        ctx = FlakeInspectorContext(
            overview_thumb=self._overview_stitch_thumb,
            overview_size_px=self._overview_stitch_size_px,
            stitch_meta=self._overview_stitch_meta,
            run_dir=self._selected_run.path if self._selected_run else Path(),
            annotations=self._annotations,
            on_annotation_change=self._on_annotation_change,
            um_per_px=self._um_per_px,
            total_count=len(self._last_top),
            revisit_lookup=self._revisit_lookup,
        )

        FlakeInspector(
            self.root,
            frame_img,
            d,
            title=title,
            contour=contour,
            on_navigate=on_navigate,
            context=ctx,
            grid_idx=grid_idx,
        )

    def _create_detection_table(self, parent: ttk.Widget):
        """Create the persistent treeview table widget (once per run)."""
        cols = (
            "rank",
            "chip",
            "tier",
            "frame",
            "det",
            "size",
            "R",
            "G",
            "score",
            "cal_d",
            "grad",
            "entr",
            "kurt",
            "pr",
            "ar",
        )
        table = ttk.Treeview(parent, columns=cols, show="headings", height=15, selectmode="none")

        col_spec = {
            "rank": ("#", 30),
            "chip": ("Chip", 35),
            "tier": ("T", 25),
            "frame": ("Frame", 110),
            "det": ("Det", 30),
            "size": ("Size", 55),
            "R": ("R", 55),
            "G": ("G", 55),
            "score": ("Score", 55),
            "cal_d": ("CalD", 55),
            "grad": ("Grad", 50),
            "entr": ("Entr", 45),
            "kurt": ("Kurt", 45),
            "pr": ("PR", 45),
            "ar": ("AR", 40),
        }
        sort_state: dict[str, bool] = {}

        def _sort_column(col_id: str):
            reverse = not sort_state.get(col_id, False)
            sort_state[col_id] = reverse

            items = [(table.set(iid, col_id), iid) for iid in table.get_children()]
            try:
                items.sort(key=lambda t: float(t[0]), reverse=reverse)
            except ValueError:
                items.sort(key=lambda t: t[0], reverse=reverse)

            for idx, (_val, iid) in enumerate(items):
                table.move(iid, "", idx)

            arrow = " \u25bc" if reverse else " \u25b2"
            for cid, (heading, _w) in col_spec.items():
                text = heading + arrow if cid == col_id else heading
                table.heading(cid, text=text)

        for col_id, (heading, width) in col_spec.items():
            table.heading(col_id, text=heading, command=lambda c=col_id: _sort_column(c))
            anchor = "w" if col_id == "frame" else "center"
            table.column(col_id, width=width, minwidth=width, anchor=anchor)

        scroll = ttk.Scrollbar(parent, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        table.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")

        self._det_table = table

    # ── Image helpers ────────────────────────────────────────────

    def _load_image_into_label(
        self, img_path: Path, parent: ttk.Widget, max_width: int, attr: str, popup_path: Path | None = None
    ):
        """Load an image, scale to max_width, display in a Label inside parent."""
        try:
            img = Image.open(img_path)
            w, h = img.size
            if w > max_width:
                scale = max_width / w
                img = img.resize((max_width, int(h * scale)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            setattr(self, attr, photo)
            lbl = ttk.Label(parent, image=photo, cursor="hand2")
            lbl.pack()
            click_path = popup_path if popup_path is not None else img_path
            lbl.bind("<Button-1>", lambda _e, p=click_path: ImagePopup(self.root, p))
        except Exception as e:
            ttk.Label(parent, text=f"Could not load image: {e}", foreground="red").pack()


# ── Entry point ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Browse completed find-flakes runs")
    parser.add_argument(
        "--scans-dir",
        type=Path,
        default=Path("scans"),
        help="Directory containing run_* folders (default: scans/)",
    )
    args = parser.parse_args()

    root = tk.Tk()

    # Better rendering on Linux (clam theme, proper fonts)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    # Platform-specific font: Windows renders fine with defaults,
    # Linux needs an explicit TrueType family for antialiasing
    if sys.platform == "win32":
        font_family = "Segoe UI"
    else:
        font_family = "DejaVu Sans"
    for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
        tkfont.nametofont(name).configure(family=font_family, size=10)

    RunViewerGUI(root, args.scans_dir)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
