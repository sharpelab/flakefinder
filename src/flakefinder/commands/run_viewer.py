"""Tkinter GUI for browsing completed find-flakes runs.

Provides a two-panel interface: run list on the left, run detail on the right.
Collaborators can browse runs, select chips, and view detection results without
CLI knowledge. Runs on the microscope PC (Windows 11).

Usage:
    uv run run-viewer
    uv run run-viewer --scans-dir path/to/scans
"""

from __future__ import annotations

import contextlib
import json
import sys
import tkinter as tk
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
    n_chips: int
    total_detections: int
    duration_s: float
    notes: str


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
    notes = cp.get("notes", "") or ""
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
        n_chips=n_chips,
        total_detections=total_detections,
        duration_s=duration_s,
        notes=notes,
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
            for idx, d in enumerate(dets):
                d.setdefault("frame", frame_name)
                d.setdefault("det_idx", idx)
                d["chip_idx"] = chip_idx
            all_dets.extend(dets)

    all_dets.sort(key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
    return all_dets, scan_dirs, um_per_px


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


# ── Image popup viewer ──────────────────────────────────────────────


class ImagePopup(tk.Toplevel):
    """Resizable image popup with contain scaling. Dismiss with Escape or click."""

    def __init__(self, parent: tk.Tk, img_path: Path):
        super().__init__(parent)
        self.withdraw()  # hide until fully rendered
        self.title(img_path.name)

        self._src_image = Image.open(img_path)
        self._photo: ImageTk.PhotoImage | None = None

        self._label = ttk.Label(self, anchor="center")
        self._label.pack(fill="both", expand=True)

        # Initial size: contain to 80% of screen
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = int(screen_w * 0.8)
        max_h = int(screen_h * 0.8)
        w, h = self._src_image.size
        scale = min(max_w / w, max_h / h)
        init_w, init_h = int(w * scale), int(h * scale)
        self.geometry(f"{init_w}x{init_h}")

        # Render initial image immediately (can't rely on Configure for first frame)
        self._render(init_w, init_h)

        # Resize on window change
        self._resize_pending = False
        self.bind("<Configure>", self._on_resize)

        # Dismiss on click, Escape
        self.bind("<Escape>", lambda _: self.destroy())
        self._label.bind("<Button-1>", lambda _: self.destroy())
        self.deiconify()
        self.focus_set()

    def _on_resize(self, event):
        # Only respond to top-level window resizes, debounce
        if event.widget is not self:
            return
        if not self._resize_pending:
            self._resize_pending = True
            self.after(50, self._do_resize)

    def _do_resize(self):
        self._resize_pending = False
        self._render(self._label.winfo_width(), self._label.winfo_height())

    def _render(self, win_w: int, win_h: int):
        if win_w < 2 or win_h < 2:
            return
        src_w, src_h = self._src_image.size
        scale = min(win_w / src_w, win_h / src_h)
        new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.NEAREST
        img = self._src_image.resize((new_w, new_h), resample)
        self._photo = ImageTk.PhotoImage(img)
        self._label.configure(image=self._photo)


class ZoomableFramePopup(tk.Toplevel):
    """Zoomable image popup centered on a detection. Scroll to zoom, drag to pan, Escape to close."""

    ZOOM_FACTOR = 1.3
    MIN_ZOOM = 0.05
    MAX_ZOOM = 5.0

    def __init__(
        self,
        parent: tk.Tk,
        src_image: Image.Image,
        det: Detection,
        title: str = "",
        contour: list[list[int]] | None = None,
        on_navigate: Callable[[int], tuple[Image.Image, Detection, str, list[list[int]] | None] | None] | None = None,
    ):
        super().__init__(parent)
        self.withdraw()  # hide until fully rendered
        self.title(title)

        self._src = src_image
        self._det = det
        self._contour = contour
        self._on_navigate = on_navigate
        self._photo: ImageTk.PhotoImage | None = None

        # Zoom state: center in source image coords
        bx, by, bw, bh = det["bbox"]
        self._cx = bx + bw / 2.0
        self._cy = by + bh / 2.0

        # Initial zoom: fit whole image, capped to reasonable window size
        sw, sh = src_image.size
        max_w, max_h = min(1200, int(self.winfo_screenwidth() * 0.5)), min(900, int(self.winfo_screenheight() * 0.7))
        self._zoom = min(max_w / sw, max_h / sh)
        self._win_w = max(400, int(sw * self._zoom))
        self._win_h = max(300, int(sh * self._zoom))
        self.geometry(f"{self._win_w}x{self._win_h}")

        self._canvas = tk.Canvas(self, highlightthickness=0, bg="black")
        self._canvas.pack(fill="both", expand=True)

        # Pan state
        self._drag_x = 0
        self._drag_y = 0

        # Bindings
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-4>", self._on_scroll_linux)
        self._canvas.bind("<Button-5>", self._on_scroll_linux)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self.bind("<Escape>", lambda _: self.destroy())
        self.bind("<Configure>", self._on_configure)
        if on_navigate:
            self.bind("<Left>", lambda _: self._navigate(-1))
            self.bind("<Right>", lambda _: self._navigate(1))

        # Render synchronously, then show
        self._render()
        self.deiconify()
        self.focus_set()

    def _navigate(self, delta: int):
        if not self._on_navigate:
            return
        result = self._on_navigate(delta)
        if result is None:
            return
        src_image, det, title, contour = result
        self._src = src_image
        self._det = det
        self._contour = contour
        self.title(title)
        # Re-center on new detection
        bx, by, bw, bh = det["bbox"]
        self._cx = bx + bw / 2.0
        self._cy = by + bh / 2.0
        self._render()

    def _on_configure(self, event):
        if event.widget is not self:
            return
        self._win_w = event.width
        self._win_h = event.height
        self._render()

    def _on_scroll(self, event):
        if event.delta > 0:
            self._zoom_at(event.x, event.y, self.ZOOM_FACTOR)
        else:
            self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_FACTOR)

    def _on_scroll_linux(self, event):
        if event.num == 4:
            self._zoom_at(event.x, event.y, self.ZOOM_FACTOR)
        else:
            self._zoom_at(event.x, event.y, 1.0 / self.ZOOM_FACTOR)

    def _zoom_at(self, mx: int, my: int, factor: float):
        """Zoom centered on mouse position."""
        # Convert mouse pos to source coords (use _win_w/h, reliable before mapping)
        cw, ch = self._win_w, self._win_h
        src_x = self._cx + (mx - cw / 2.0) / self._zoom
        src_y = self._cy + (my - ch / 2.0) / self._zoom

        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))

        # Adjust center so the point under the mouse stays fixed
        self._cx = src_x - (mx - cw / 2.0) / new_zoom
        self._cy = src_y - (my - ch / 2.0) / new_zoom
        self._zoom = new_zoom
        self._render()

    def _on_drag_start(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        dx = event.x - self._drag_x
        dy = event.y - self._drag_y
        self._drag_x = event.x
        self._drag_y = event.y
        self._cx -= dx / self._zoom
        self._cy -= dy / self._zoom
        self._render()

    def _render(self):
        cw, ch = self._win_w, self._win_h
        if cw < 2 or ch < 2:
            return
        sw, sh = self._src.size
        z = self._zoom

        # Source region visible in the canvas
        half_w = cw / (2.0 * z)
        half_h = ch / (2.0 * z)
        src_x0 = self._cx - half_w
        src_y0 = self._cy - half_h
        src_x1 = self._cx + half_w
        src_y1 = self._cy + half_h

        # Clamp to source bounds and compute canvas offsets for black borders
        crop_x0 = max(0.0, src_x0)
        crop_y0 = max(0.0, src_y0)
        crop_x1 = min(float(sw), src_x1)
        crop_y1 = min(float(sh), src_y1)

        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            return

        crop = self._src.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
        disp_w = max(1, int((crop_x1 - crop_x0) * z))
        disp_h = max(1, int((crop_y1 - crop_y0) * z))
        resample = Image.Resampling.LANCZOS if z < 1.0 else Image.Resampling.NEAREST
        display = crop.resize((disp_w, disp_h), resample)

        # Draw contour and bbox on display
        draw = ImageDraw.Draw(display)
        if self._contour and len(self._contour) >= 3:
            pts = [(int((x - crop_x0) * z), int((y - crop_y0) * z)) for x, y in self._contour]
            draw.polygon(pts, outline="cyan")
        bx, by, bw, bh = self._det["bbox"]
        pad = 5  # px padding around bbox
        rx0 = int((bx - pad - crop_x0) * z)
        ry0 = int((by - pad - crop_y0) * z)
        rx1 = int((bx + bw + pad - crop_x0) * z)
        ry1 = int((by + bh + pad - crop_y0) * z)
        draw.rectangle([rx0, ry0, rx1, ry1], outline="lime", width=2)

        self._photo = ImageTk.PhotoImage(display)
        # Position so the cropped region aligns correctly
        canvas_x = int((crop_x0 - src_x0) * z)
        canvas_y = int((crop_y0 - src_y0) * z)
        self._canvas.delete("all")
        self._canvas.create_image(canvas_x, canvas_y, anchor="nw", image=self._photo)


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

        # Chip toggle state
        self._active_chips: set[int] = set()  # empty = all shown
        self._chip_buttons: dict[int, tk.Button] = {}
        self._n_chips: int = 0
        self._last_top: list[Detection] = []
        self._last_thumb_cols: int = 0

        self._build_ui()
        self._refresh_runs()

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
        columns = ("date", "preset", "chips", "detections", "duration", "notes")
        self.tree = ttk.Treeview(self._left_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("date", text="Date/Time")
        self.tree.heading("preset", text="Preset")
        self.tree.heading("chips", text="Chips")
        self.tree.heading("detections", text="Det")
        self.tree.heading("duration", text="Dur")
        self.tree.heading("notes", text="Notes")

        self.tree.column("date", width=120, minwidth=90)
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
        if event.width != getattr(self, "_last_canvas_w", 0):
            self._last_canvas_w = event.width
            self._detail_canvas.itemconfig(self._detail_window, width=event.width)
            # Re-layout thumbnails if column count changed
            new_cols = max(1, (event.width - 32) // (CROP_THUMB_SIZE + 16))
            if new_cols != self._last_thumb_cols and self._last_top:
                self._last_thumb_cols = new_cols
                self._update_filtered_frames(self._last_top)

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
                "", "end", iid=str(i), values=(date_str, run.preset, run.n_chips, det_str, dur_str, notes_short)
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

    def _show_run_list(self):
        """Restore the left panel."""
        with contextlib.suppress(tk.TclError):
            self.paned.insert(0, self._left_frame, weight=1)

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
        self._last_top = []
        self._last_thumb_cols = 0

    def _show_run_detail(self, run: RunInfo):
        self._clear_detail()
        self._detail_canvas.yview_moveto(0)
        self._n_chips = run.n_chips

        # ── Phase 1: lightweight skeleton (renders immediately) ──

        # Back button
        ttk.Button(self._detail_frame, text="← Runs", command=self._show_run_list).pack(anchor="w", padx=8, pady=(4, 0))

        # Run header
        header = ttk.LabelFrame(self._detail_frame, text=run.name, padding=8)
        header.pack(fill="x", padx=8, pady=(8, 4))

        info_lines = []
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
        overview_path = self._find_overview_image(run.path)
        if overview_path:
            self._overview_container.pack(fill="x", padx=8, pady=4)
            ttk.Label(self._overview_container, text="Loading…", foreground="gray").pack()

        # Filter panel (cheap — just slider widgets)
        self._build_filter_panel(run.preset)

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
        self.root.after(1, self._load_overview, run, overview_path)

    def _load_overview(self, run: RunInfo, overview_path: Path | None):
        """Phase 2a: load overview image, then chain to detection loading."""
        if self._selected_run is not run:
            return

        if overview_path and self._overview_container.winfo_exists():
            for w in self._overview_container.winfo_children():
                w.destroy()
            self._load_image_into_label(overview_path, self._overview_container, OVERVIEW_MAX_WIDTH, "_overview_photo")

        self.root.after(1, self._load_detections, run)

    def _load_detections(self, run: RunInfo):
        """Phase 2b: load detection data and apply filters."""
        if self._selected_run is not run:
            return

        self._all_detections, self._chip_scan_dirs, self._um_per_px = _load_all_detections(run.path, run.n_chips)

        # Apply filters (replaces loading bar with thumbnails)
        if self._all_detections and self._filtered_crops_frame and self._filtered_table_frame:
            self._filtered_table_frame.pack(fill="x", padx=8, pady=4)
            self._apply_filters()
        elif self._filtered_crops_frame:
            # No detections — replace loading bar with message
            for w in self._filtered_crops_frame.winfo_children():
                w.destroy()
            ttk.Label(self._filtered_crops_frame, text="No detections", foreground="gray").pack(anchor="w")

    def _find_overview_image(self, run_dir: Path) -> Path | None:
        """Find the overview detection image."""
        for p in run_dir.glob("overview_*_stitch_chips_detected.png"):
            return p
        return None

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

        # Filter DoubleVars with preset defaults
        self._fv_perim_ratio = tk.DoubleVar(value=config.tier1_perim_ratio)
        self._fv_cal_dist = tk.DoubleVar(value=config.tier1_cal_dist)
        self._fv_g_min = tk.DoubleVar(value=max(config.tier1_g_min, -2.0))
        self._fv_g_max = tk.DoubleVar(value=min(config.tier1_g_max, 6.0))
        self._fv_r_max = tk.DoubleVar(value=max(config.tier1_r_max, -3.0))
        self._fv_entropy = tk.DoubleVar(value=min(config.tier1_entropy_max, 8.0))
        self._fv_min_size = tk.DoubleVar(value=0)
        self._fv_grad_energy = tk.DoubleVar(value=50.0)
        self._fv_aspect_ratio = tk.DoubleVar(value=6.0)
        self._fv_kurtosis = tk.DoubleVar(value=50.0)
        self._fv_top_n = tk.IntVar(value=DEFAULT_FILTER_TOP_N)

        # Slider definitions: (row, col, label, var, from_, to, resolution, fmt)
        slider_defs = [
            (0, 0, "perim_ratio \u2264", self._fv_perim_ratio, 1.0, 3.0, 0.05, "{:.2f}"),
            (0, 1, "cal_dist \u2264", self._fv_cal_dist, 0.0, 2.0, 0.05, "{:.2f}"),
            (0, 2, "min \u00b5m\u00b2 \u2265", self._fv_min_size, 0, 2000, 10, "{:.0f}"),
            (1, 0, "G min \u2265", self._fv_g_min, -2.0, 6.0, 0.1, "{:+.1f}"),
            (1, 1, "G max \u2264", self._fv_g_max, -2.0, 6.0, 0.1, "{:+.1f}"),
            (1, 2, "R max \u2264", self._fv_r_max, -3.0, 6.0, 0.1, "{:+.1f}"),
            (2, 0, "entropy \u2264", self._fv_entropy, 0.0, 8.0, 0.1, "{:.1f}"),
            (2, 1, "grad_energy \u2264", self._fv_grad_energy, 0.0, 50.0, 0.5, "{:.1f}"),
            (2, 2, "aspect_ratio \u2264", self._fv_aspect_ratio, 1.0, 6.0, 0.5, "{:.1f}"),
            (3, 0, "kurtosis \u2264", self._fv_kurtosis, -2.0, 50.0, 1.0, "{:.0f}"),
        ]

        sliders_frame = ttk.Frame(filter_frame)
        sliders_frame.pack(fill="x")
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
        """Filter all detections by chip selection + slider values and update results."""
        self._filter_debounce_id = None
        if not self._all_detections:
            return

        # Chip filter (empty active set = show all)
        visible_chips = self._get_visible_chips()

        # Read filter values once
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

        # Count detections visible (after chip filter, before slider filter)
        if visible_chips is not None:
            chip_filtered = [d for d in self._all_detections if d.get("chip_idx") in visible_chips]
        else:
            chip_filtered = self._all_detections

        passing = []
        for d in chip_filtered:
            rgb = d.get("contrast_rgb")
            if not rgb or len(rgb) < 2:
                continue
            r, g = rgb[0], rgb[1]
            if (
                d.get("perim_ratio", 0) <= pr_max
                and d.get("cal_dist", 0) <= cd_max
                and g >= g_min
                and g <= g_max
                and r <= r_max
                and d.get("entropy", d.get("g_entropy", 0)) <= ent_max
                and d.get("size_px", 0) >= min_size_px
                and d.get("grad_energy", 0) <= ge_max
                and d.get("aspect_ratio", 1.0) <= ar_max
                and max(d.get("r_kurt", 0), d.get("g_kurt", 0), d.get("b_kurt", 0)) <= kurt_max
            ):
                passing.append(d)

        passing.sort(key=lambda d: -d.get("score", 0))
        deduped = dedup_detections(passing)
        self._filter_count_var.set(f"{len(deduped)} / {len(chip_filtered)} pass ({len(passing) - len(deduped)} dupes)")

        top = deduped[:top_n]
        self._last_top = top
        self._update_filtered_table(top)
        self._update_filtered_frames(top)

    def _reset_filters(self):
        """Reset all filter sliders to preset defaults and clear chip selection."""
        config = self._filter_preset_config
        if config is None:
            return
        self._fv_perim_ratio.set(config.tier1_perim_ratio)
        self._fv_cal_dist.set(config.tier1_cal_dist)
        self._fv_g_min.set(max(config.tier1_g_min, -2.0))
        self._fv_g_max.set(min(config.tier1_g_max, 6.0))
        self._fv_r_max.set(max(config.tier1_r_max, -3.0))
        self._fv_entropy.set(min(config.tier1_entropy_max, 8.0))
        self._fv_min_size.set(0)
        self._fv_grad_energy.set(50.0)
        self._fv_aspect_ratio.set(6.0)
        self._fv_kurtosis.set(50.0)
        self._active_chips.clear()
        self._update_chip_button_visuals()
        for lbl, var, fmt in self._filter_val_labels:
            lbl.configure(text=fmt.format(var.get()))
        # Cancel pending debounce and apply immediately
        if self._filter_debounce_id is not None:
            self.root.after_cancel(self._filter_debounce_id)
            self._filter_debounce_id = None
        self._apply_filters()

    def _update_filtered_table(self, top: list[Detection]):
        """Rebuild the detection table with filtered results."""
        if self._filtered_table_frame is None:
            return
        for w in self._filtered_table_frame.winfo_children():
            w.destroy()
        if not top:
            ttk.Label(self._filtered_table_frame, text="No detections pass filters", foreground="gray").pack(anchor="w")
            return
        self._populate_detection_table(self._filtered_table_frame, top)

    def _update_filtered_frames(self, top: list[Detection]):
        """Show full-frame thumbnails with detection bbox for filtered top detections."""
        if self._filtered_crops_frame is None:
            return
        for w in self._filtered_crops_frame.winfo_children():
            w.destroy()
        self._crop_photos = []

        if not top:
            ttk.Label(self._filtered_crops_frame, text="No detections pass filters", foreground="gray").pack(anchor="w")
            return

        grid_frame = ttk.Frame(self._filtered_crops_frame)
        grid_frame.pack(fill="x")
        avail_w = self._detail_canvas.winfo_width() - 32  # padding
        cols = max(1, avail_w // (CROP_THUMB_SIZE + 16))

        for i, d in enumerate(top):
            frame_name = d.get("frame", "")
            chip_idx = d.get("chip_idx", 0)
            if not frame_name:
                continue

            row, col = divmod(i, cols)
            cell = ttk.Frame(grid_frame, padding=2)
            cell.grid(row=row * 2, column=col, sticky="nw")

            try:
                frame_img = self._load_frame_cached(chip_idx, frame_name)
                if frame_img is None:
                    ttk.Label(cell, text="[no frame]", foreground="gray").pack()
                    continue

                # Resize to thumbnail then draw bbox
                fw, fh = frame_img.size
                scale = min(CROP_THUMB_SIZE / fw, CROP_THUMB_SIZE / fh)
                thumb = frame_img.resize((int(fw * scale), int(fh * scale)), Image.Resampling.LANCZOS)
                draw = ImageDraw.Draw(thumb)
                bx, by, bw, bh = d["bbox"]
                pad = 5
                x0 = int((bx - pad) * scale)
                y0 = int((by - pad) * scale)
                x1 = int((bx + bw + pad) * scale)
                y1 = int((by + bh + pad) * scale)
                draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)

                photo = ImageTk.PhotoImage(thumb)
                self._crop_photos.append(photo)
                lbl = ttk.Label(cell, image=photo, cursor="hand2")
                lbl.pack()
                lbl.bind(
                    "<Button-1>",
                    lambda _e, idx=i: self._show_frame_popup_at(idx),
                )
            except Exception:
                ttk.Label(cell, text="[load error]", foreground="red").pack()

            rgb = d.get("contrast_rgb", (0, 0, 0))
            r_val, g_val = rgb[0], rgb[1]
            size_um2 = d.get("size_px", 0) * self._um_per_px**2
            ent = d.get("entropy", d.get("g_entropy", 0))
            ge = d.get("grad_energy", 0)
            kurt = max(d.get("r_kurt", 0), d.get("g_kurt", 0), d.get("b_kurt", 0))
            line1 = f"#{i + 1} C{chip_idx} R={r_val:+.2f} G={g_val:+.2f} {size_um2:.0f}\u00b5m\u00b2"
            line2 = f"e={ent:.1f} g={ge:.1f} k={kurt:.0f}"
            name_lbl = ttk.Label(cell, text=f"{line1}\n{line2}", font=("TkDefaultFont", 7), justify="left")
            name_lbl.pack(anchor="w")

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

    def _load_frame_contour(self, chip_idx: int, frame_name: str, det_idx: int) -> list[list[int]] | None:
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
            if det_idx < len(dets):
                return dets[det_idx].get("contour")
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
        det_idx = d.get("det_idx", -1)
        contour = self._load_frame_contour(chip_idx, frame_name, det_idx) if det_idx >= 0 else None
        title = f"#{grid_idx + 1} C{chip_idx} {frame_name} det#{det_idx}"
        return frame_img, d, title, contour

    def _show_frame_popup_at(self, grid_idx: int):
        """Open a ZoomableFramePopup for the detection at grid_idx in _last_top."""
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

        ZoomableFramePopup(self.root, frame_img, d, title=title, contour=contour, on_navigate=on_navigate)

    def _populate_detection_table(self, parent: ttk.Widget, ranked: list[Detection]):
        """Show a sortable treeview table of top detection scoring data. Click headers to sort."""
        n_show = min(len(ranked), 50)
        top = ranked[:n_show]

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
        table = ttk.Treeview(parent, columns=cols, show="headings", height=min(n_show, 15), selectmode="none")

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
        # Sort state per table: {col_id: reverse}
        sort_state: dict[str, bool] = {}

        def _sort_column(col_id: str):
            reverse = not sort_state.get(col_id, False)
            sort_state[col_id] = reverse

            items = [(table.set(iid, col_id), iid) for iid in table.get_children()]
            # Try numeric sort, fall back to string
            try:
                items.sort(key=lambda t: float(t[0]), reverse=reverse)
            except ValueError:
                items.sort(key=lambda t: t[0], reverse=reverse)

            for idx, (_val, iid) in enumerate(items):
                table.move(iid, "", idx)

            # Update header with sort indicator
            arrow = " \u25bc" if reverse else " \u25b2"
            for cid, (heading, _w) in col_spec.items():
                text = heading + arrow if cid == col_id else heading
                table.heading(cid, text=text)

        for col_id, (heading, width) in col_spec.items():
            table.heading(col_id, text=heading, command=lambda c=col_id: _sort_column(c))
            anchor = "w" if col_id == "frame" else "center"
            table.column(col_id, width=width, minwidth=width, anchor=anchor)

        for i, d in enumerate(top):
            r_val, g_val = 0.0, 0.0
            rgb = d.get("contrast_rgb")
            if rgb and len(rgb) >= 2:
                r_val, g_val = rgb[0], rgb[1]
            table.insert(
                "",
                "end",
                values=(
                    i + 1,
                    d.get("chip_idx", "?"),
                    d.get("tier", "?"),
                    d.get("frame", "?"),
                    d.get("det_idx", "?"),
                    d.get("size_px", 0),
                    f"{r_val:+.3f}",
                    f"{g_val:+.3f}",
                    f"{d.get('score', 0):.3f}",
                    f"{d.get('cal_dist', 0):.3f}",
                    f"{d.get('grad_energy', 0):.1f}",
                    f"{d.get('entropy', d.get('g_entropy', 0)):.2f}",
                    f"{max(d.get('r_kurt', 0), d.get('g_kurt', 0), d.get('b_kurt', 0)):.1f}",
                    f"{d.get('perim_ratio', 0):.2f}",
                    f"{d.get('aspect_ratio', 1.0):.1f}",
                ),
            )

        scroll = ttk.Scrollbar(parent, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        table.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")

    # ── Image helpers ────────────────────────────────────────────

    def _load_image_into_label(self, img_path: Path, parent: ttk.Widget, max_width: int, attr: str):
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
            lbl.bind("<Button-1>", lambda _e, p=img_path: ImagePopup(self.root, p))
        except Exception as e:
            ttk.Label(parent, text=f"Could not load image: {e}", foreground="red").pack()


# ── Entry point ──────────────────────────────────────────────────────


def main():
    import argparse

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
    import tkinter.font as tkfont

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
