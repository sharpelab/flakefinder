"""Tkinter GUI for browsing completed find-flakes runs.

Provides a two-panel interface: run list on the left, run detail on the right.
Collaborators can browse runs, select chips, and view detection results without
CLI knowledge. Runs on the microscope PC (Windows 11).

Usage:
    uv run run-viewer
    uv run run-viewer --scans-dir path/to/scans
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import NamedTuple

from PIL import Image, ImageTk

from flakefinder.segmentation import Detection

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


class ChipInfo(NamedTuple):
    """Per-chip data for the detail view."""

    idx: int
    has_seg: bool
    tier_1: int
    tier_2: int
    tier_3: int
    total_detections: int
    crop_images: list[Path]
    scatter_path: Path | None  # R-G scatter plot from rerank
    revisit_images: list[Path]
    scan_dir: Path | None  # for rerank --scan-dir
    ranked_detections: list[Detection]  # top detections sorted by tier/score


def _load_chip_info(run_dir: Path, chip_idx: int) -> ChipInfo:
    """Load chip detail data."""
    chip_dir = run_dir / f"chip_{chip_idx}"
    seg_dir = chip_dir / "seg"

    has_seg = False
    tier_1 = tier_2 = tier_3 = total_det = 0
    crop_images: list[Path] = []
    revisit_images: list[Path] = []
    ranked_detections: list[Detection] = []

    # Seg summary
    seg_summary = seg_dir / "summary.json"
    if seg_summary.exists():
        try:
            with open(seg_summary) as f:
                ss = json.load(f)
            stats = ss.get("stats", {})
            has_seg = True
            tier_1 = stats.get("tier_1", 0)
            tier_2 = stats.get("tier_2", 0)
            tier_3 = stats.get("tier_3", 0)
            total_det = stats.get("total_detections", 0)

            # Build ranked detection list from detections_by_frame
            all_dets = []
            for dets in ss.get("detections_by_frame", {}).values():
                all_dets.extend(dets)
            ranked_detections = sorted(all_dets, key=lambda d: (d.get("tier", 3), -d.get("score", 0)))
        except (json.JSONDecodeError, OSError):
            pass

    # Crop images from seg/crops/
    crops_dir = seg_dir / "crops"
    if crops_dir.is_dir():
        crop_images = sorted(crops_dir.glob("rank*.jpg"))

    # R-G scatter plot
    scatter_path = seg_dir / "rg_scatter.png"
    if not scatter_path.exists():
        scatter_path = None

    # Revisit images — look for revisit_* directories
    for revisit_dir in sorted(chip_dir.glob("revisit_*")):
        if revisit_dir.is_dir():
            for img in sorted(revisit_dir.glob("rank*_*.png")):
                revisit_images.append(img)

    # Find scan directory for rerank
    scan_dir = None
    for d in chip_dir.iterdir():
        if d.is_dir() and d.name.startswith("scan_"):
            scan_dir = d
            break

    return ChipInfo(
        idx=chip_idx,
        has_seg=has_seg,
        tier_1=tier_1,
        tier_2=tier_2,
        tier_3=tier_3,
        total_detections=total_det,
        crop_images=crop_images,
        scatter_path=scatter_path,
        revisit_images=revisit_images,
        scan_dir=scan_dir,
        ranked_detections=ranked_detections,
    )


# ── Formatting helpers ───────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


OVERVIEW_MAX_WIDTH = 600
CROP_THUMB_SIZE = 220
REVISIT_THUMB_SIZE = 200
DEFAULT_RERANK_TOP = 10


# ── Image popup viewer ──────────────────────────────────────────────


class ImagePopup(tk.Toplevel):
    """Resizable image popup with contain scaling. Dismiss with Escape or click."""

    def __init__(self, parent: tk.Tk, img_path: Path):
        super().__init__(parent)
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


# ── GUI ──────────────────────────────────────────────────────────────


class RunViewerGUI:
    def __init__(self, root: tk.Tk, scans_dir: Path):
        self.root = root
        self.scans_dir = scans_dir
        self.root.title("FlakeFinder Run Viewer")
        self.root.minsize(1000, 650)

        # Image references (prevent GC)
        self._overview_photo: ImageTk.PhotoImage | None = None
        self._scatter_photo: ImageTk.PhotoImage | None = None
        self._crop_photos: list[ImageTk.PhotoImage] = []
        self._revisit_photos: list[ImageTk.PhotoImage] = []

        # Current state
        self._runs: list[RunInfo] = []
        self._selected_run: RunInfo | None = None
        self._chip_selector_frame: ttk.LabelFrame | None = None

        self._build_ui()
        self._refresh_runs()

    def _build_ui(self):
        # Main horizontal paned window
        self.paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Left panel: Run list ─────────────────────────────────
        left_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, weight=1)

        # Refresh button
        btn_row = ttk.Frame(left_frame)
        btn_row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Button(btn_row, text="↻ Refresh", command=self._refresh_runs).pack(side="left")
        self._run_count_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self._run_count_var, foreground="gray").pack(side="right")

        # Treeview
        columns = ("date", "preset", "chips", "detections", "duration")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("date", text="Date/Time")
        self.tree.heading("preset", text="Preset")
        self.tree.heading("chips", text="Chips")
        self.tree.heading("detections", text="Detections")
        self.tree.heading("duration", text="Duration")

        self.tree.column("date", width=130, minwidth=100)
        self.tree.column("preset", width=70, minwidth=50)
        self.tree.column("chips", width=45, minwidth=35, anchor="center")
        self.tree.column("detections", width=75, minwidth=50, anchor="center")
        self.tree.column("duration", width=65, minwidth=50, anchor="center")

        tree_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
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

        # Mouse wheel scrolling
        self._detail_canvas.bind("<Enter>", self._bind_mousewheel)
        self._detail_canvas.bind("<Leave>", self._unbind_mousewheel)

        # Placeholder
        self._placeholder = ttk.Label(self._detail_frame, text="Select a run to view details", foreground="gray")
        self._placeholder.pack(pady=40)

    def _bind_mousewheel(self, _event):
        self._detail_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux
        self._detail_canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self._detail_canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event):
        self._detail_canvas.unbind_all("<MouseWheel>")
        self._detail_canvas.unbind_all("<Button-4>")
        self._detail_canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        self._detail_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self._detail_canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self._detail_canvas.yview_scroll(3, "units")

    def _on_detail_configure(self, _event):
        self._detail_canvas.configure(scrollregion=self._detail_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._detail_canvas.itemconfig(self._detail_window, width=event.width)

    # ── Run list ─────────────────────────────────────────────────

    def _refresh_runs(self):
        self._runs = _discover_runs(self.scans_dir)
        self.tree.delete(*self.tree.get_children())
        for i, run in enumerate(self._runs):
            date_str = run.timestamp.strftime("%Y-%m-%d %H:%M") if run.timestamp else run.name
            det_str = str(run.total_detections) if run.total_detections > 0 else "—"
            dur_str = _fmt_duration(run.duration_s) if run.duration_s > 0 else "—"
            self.tree.insert("", "end", iid=str(i), values=(date_str, run.preset, run.n_chips, det_str, dur_str))
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
        self._show_run_detail(run)

    # ── Run detail ───────────────────────────────────────────────

    def _clear_detail(self):
        for w in self._detail_frame.winfo_children():
            w.destroy()
        self._overview_photo = None
        self._scatter_photo = None
        self._crop_photos = []
        self._revisit_photos = []

    def _show_run_detail(self, run: RunInfo):
        self._clear_detail()
        self._detail_canvas.yview_moveto(0)

        # ── Run header ───────────────────────────────────────────
        header = ttk.LabelFrame(self._detail_frame, text=run.name, padding=8)
        header.pack(fill="x", padx=8, pady=(8, 4))

        # Load checkpoint for notes
        notes = ""
        cp_path = run.path / "checkpoint.json"
        if cp_path.exists():
            try:
                with open(cp_path) as f:
                    cp = json.load(f)
                notes = cp.get("notes", "")
            except (json.JSONDecodeError, OSError):
                pass

        info_lines = []
        if run.timestamp:
            info_lines.append(f"Date: {run.timestamp.strftime('%Y-%m-%d %H:%M')}")
        info_lines.append(f"Preset: {run.preset}    Chips: {run.n_chips}    Detections: {run.total_detections}")
        if run.duration_s > 0:
            info_lines.append(f"Duration: {_fmt_duration(run.duration_s)}")
        if notes:
            info_lines.append(f"Notes: {notes}")
        ttk.Label(header, text="\n".join(info_lines), justify="left").pack(anchor="w")

        # ── Overview image ───────────────────────────────────────
        overview_path = self._find_overview_image(run.path)
        if overview_path:
            overview_frame = ttk.LabelFrame(self._detail_frame, text="Overview", padding=4)
            overview_frame.pack(fill="x", padx=8, pady=4)
            self._load_image_into_label(overview_path, overview_frame, OVERVIEW_MAX_WIDTH, "_overview_photo")

        # ── Chip buttons ─────────────────────────────────────────
        if run.n_chips > 0:
            self._chip_selector_frame = ttk.LabelFrame(self._detail_frame, text="Chips", padding=4)
            self._chip_selector_frame.pack(fill="x", padx=8, pady=4)

            btn_row = ttk.Frame(self._chip_selector_frame)
            btn_row.pack(fill="x")
            for i in range(run.n_chips):
                ttk.Button(
                    btn_row,
                    text=f"Chip {i}",
                    command=lambda ci=i: self._show_chip_detail(run, ci),
                ).pack(side="left", padx=2, pady=2)

            # Chip detail area (filled when a chip button is clicked)
            self._chip_detail_frame = ttk.Frame(self._detail_frame)
            self._chip_detail_frame.pack(fill="x", padx=8, pady=4)

            # Auto-select chip 0
            self._show_chip_detail(run, 0)

    def _find_overview_image(self, run_dir: Path) -> Path | None:
        """Find the overview detection image."""
        for p in run_dir.glob("overview_*_stitch_chips_detected.png"):
            return p
        return None

    def _scroll_to_chip_selector(self):
        """Scroll the detail canvas so the chip selector is at the top."""
        if self._chip_selector_frame is None:
            return
        self._detail_frame.update_idletasks()
        self._detail_canvas.configure(scrollregion=self._detail_canvas.bbox("all"))
        # Get the chip selector's Y position within the detail frame
        y = self._chip_selector_frame.winfo_y()
        total_h = self._detail_frame.winfo_reqheight()
        if total_h > 0:
            self._detail_canvas.yview_moveto(y / total_h)

    def _show_chip_detail(self, run: RunInfo, chip_idx: int):
        """Populate chip detail area for the given chip."""
        # Clear previous chip detail
        for w in self._chip_detail_frame.winfo_children():
            w.destroy()
        self._scatter_photo = None
        self._crop_photos = []
        self._revisit_photos = []

        chip = _load_chip_info(run.path, chip_idx)

        # ── Seg stats + rerank row ───────────────────────────────
        stats_frame = ttk.LabelFrame(self._chip_detail_frame, text=f"Chip {chip_idx} — Detections", padding=8)
        stats_frame.pack(fill="x", pady=(0, 4))

        if chip.has_seg:
            top_row = ttk.Frame(stats_frame)
            top_row.pack(fill="x")

            stats_text = (
                f"Total: {chip.total_detections}    "
                f"Tier 1: {chip.tier_1}    "
                f"Tier 2: {chip.tier_2}    "
                f"Tier 3: {chip.tier_3}"
            )
            ttk.Label(top_row, text=stats_text).pack(side="left")

            # Rerank controls
            rerank_frame = ttk.Frame(top_row)
            rerank_frame.pack(side="right")
            ttk.Label(rerank_frame, text="Top N:").pack(side="left", padx=(8, 2))
            top_n_var = tk.StringVar(value=str(DEFAULT_RERANK_TOP))
            top_n_entry = ttk.Entry(rerank_frame, textvariable=top_n_var, width=4)
            top_n_entry.pack(side="left", padx=(0, 4))
            self._rerank_btn = ttk.Button(
                rerank_frame,
                text="Rerank",
                command=lambda: self._run_rerank(run, chip, top_n_var.get()),
            )
            self._rerank_btn.pack(side="left")
        else:
            ttk.Label(stats_frame, text="No segmentation results", foreground="gray").pack(anchor="w")

        # ── Crop images (individual detection crops) ─────────────
        if chip.crop_images:
            crops_frame = ttk.LabelFrame(self._chip_detail_frame, text=f"Chip {chip_idx} — Top Detections", padding=4)
            crops_frame.pack(fill="x", pady=4)
            self._populate_image_grid(crops_frame, chip.crop_images, self._crop_photos, CROP_THUMB_SIZE, cols=5)

        # ── Detection scoring table ──────────────────────────────
        if chip.ranked_detections:
            table_frame = ttk.LabelFrame(self._chip_detail_frame, text=f"Chip {chip_idx} — Scoring Data", padding=4)
            table_frame.pack(fill="x", pady=4)
            self._populate_detection_table(table_frame, chip.ranked_detections)

        # ── R-G scatter plot ──────────────────────────────────────
        if chip.scatter_path:
            scatter_frame = ttk.LabelFrame(self._chip_detail_frame, text=f"Chip {chip_idx} — R-G Scatter", padding=4)
            scatter_frame.pack(fill="x", pady=4)
            self._load_image_into_label(chip.scatter_path, scatter_frame, OVERVIEW_MAX_WIDTH, "_scatter_photo")

        # ── Revisit images ───────────────────────────────────────
        if chip.revisit_images:
            revisit_frame = ttk.LabelFrame(self._chip_detail_frame, text=f"Chip {chip_idx} — Revisit Images", padding=4)
            revisit_frame.pack(fill="x", pady=4)
            self._populate_image_grid(
                revisit_frame, chip.revisit_images, self._revisit_photos, REVISIT_THUMB_SIZE, cols=4
            )

        # Scroll to chip selector after layout settles
        self.root.after_idle(self._scroll_to_chip_selector)

    def _populate_image_grid(
        self,
        parent: ttk.Widget,
        images: list[Path],
        photo_refs: list[ImageTk.PhotoImage],
        thumb_size: int,
        cols: int,
    ):
        """Create a grid of clickable thumbnail images."""
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill="x")

        for i, img_path in enumerate(images):
            row, col = divmod(i, cols)
            cell = ttk.Frame(grid_frame, padding=2)
            cell.grid(row=row * 2, column=col, sticky="nw")

            try:
                img = Image.open(img_path)
                img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                photo_refs.append(photo)
                lbl = ttk.Label(cell, image=photo, cursor="hand2")
                lbl.pack()
                lbl.bind("<Button-1>", lambda _e, p=img_path: ImagePopup(self.root, p))
            except Exception:
                ttk.Label(cell, text="[load error]", foreground="red").pack()

            # Label with rank from filename
            name_lbl = ttk.Label(cell, text=img_path.stem, font=("TkDefaultFont", 7))
            name_lbl.pack()

    def _populate_detection_table(self, parent: ttk.Widget, ranked: list[Detection]):
        """Show a treeview table of top detection scoring data."""
        # Show top N matching the number of crops (or all if fewer)
        n_show = min(len(ranked), 50)
        top = ranked[:n_show]

        cols = ("rank", "tier", "frame", "det", "size", "R", "G", "score", "cal_d", "grad", "entr")
        table = ttk.Treeview(parent, columns=cols, show="headings", height=min(n_show, 15), selectmode="none")

        col_spec = {
            "rank": ("#", 30),
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
        }
        for col_id, (heading, width) in col_spec.items():
            table.heading(col_id, text=heading)
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
                ),
            )

        scroll = ttk.Scrollbar(parent, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        table.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")

    # ── Rerank ───────────────────────────────────────────────────

    def _run_rerank(self, run: RunInfo, chip: ChipInfo, top_n_str: str):
        """Run rerank_detections.py in a background thread."""
        try:
            top_n = int(top_n_str)
        except ValueError:
            top_n = DEFAULT_RERANK_TOP

        seg_dir = run.path / f"chip_{chip.idx}" / "seg"
        if not (seg_dir / "summary.json").exists():
            return

        cmd = ["uv", "run", "python", "scripts/rerank_detections.py", str(seg_dir), "--top", str(top_n)]
        if chip.scan_dir:
            cmd += ["--scan-dir", str(chip.scan_dir)]

        self._rerank_btn.configure(state="disabled", text="Reranking...")

        def _worker():
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.root.after(0, self._on_rerank_done, run, chip.idx, True, "")
            except subprocess.CalledProcessError as e:
                self.root.after(0, self._on_rerank_done, run, chip.idx, False, e.stderr[:200])
            except Exception as e:
                self.root.after(0, self._on_rerank_done, run, chip.idx, False, str(e)[:200])

        threading.Thread(target=_worker, daemon=True).start()

    def _on_rerank_done(self, run: RunInfo, chip_idx: int, success: bool, error: str):
        """Called on the main thread when rerank completes."""
        self._rerank_btn.configure(state="normal", text="Rerank")
        if success:
            self._show_chip_detail(run, chip_idx)
        else:
            from tkinter import messagebox

            messagebox.showerror("Rerank failed", f"Error:\n{error}")

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
