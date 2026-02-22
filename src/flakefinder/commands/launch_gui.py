"""Tkinter GUI launcher for the find-flakes pipeline.

Provides a simple form interface for collaborators to launch find-flakes
runs without knowing CLI syntax. Runs on the microscope PC (Windows 11).

Usage:
    uv run launch-gui
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from flakefinder.commands.find_flakes import PRESETS
from flakefinder.segmentation import DetectorConfig

# gui_state.json lives next to the script's working directory (repo root)
GUI_STATE_PATH = Path("gui_state.json")
MAX_RECENT_OPERATORS = 10

DEFAULT_PRESET = "2.5_10"
MATERIALS = DetectorConfig.material_names()
DEFAULT_INITIAL_Z = "24690"
DEFAULT_AREA_RECT = "8000,95000,0,78000"

# Sub-steps per chip for progress tracking: focus_map, scan (seg is pipelined/free)
SUBSTEPS_PER_CHIP = 2

# Overview image display width
OVERVIEW_MAX_WIDTH = 500


def _load_gui_state() -> dict:
    """Load GUI state from disk, or return empty defaults."""
    if GUI_STATE_PATH.exists():
        try:
            with open(GUI_STATE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"recent_operators": []}


def _save_gui_state(state: dict) -> None:
    """Persist GUI state to disk."""
    try:
        with open(GUI_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass  # non-critical


def _add_recent_operator(state: dict, name: str) -> None:
    """Add operator to recent list (most-recent first, deduped, capped)."""
    ops = state.setdefault("recent_operators", [])
    if name in ops:
        ops.remove(name)
    ops.insert(0, name)
    state["recent_operators"] = ops[:MAX_RECENT_OPERATORS]


def _fmt_duration(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class FindFlakesGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FlakeFinder")
        self.root.minsize(600, 500)
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.root.iconbitmap(str(icon_path))

        self.process: subprocess.Popen | None = None
        self.start_time: float | None = None
        self.timer_id: str | None = None
        self.total_chips: int | None = None
        self.substeps_done: int = 0
        self.chip_phase_start: float | None = None  # when chip processing began
        self.chip_pos: int = 0  # 1-based position of current chip
        self.run_dir: Path | None = None
        self.t1_count: int = 0
        self.t2_count: int = 0
        self.revisit_pts_done: int = 0
        self._overview_photo: ImageTk.PhotoImage | None = None  # prevent GC
        self._gui_state = _load_gui_state()

        self._build_ui()

    def _build_ui(self):
        # ── Progress section (top) ──────────────────────────────
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=8)
        progress_frame.pack(fill="x", padx=8, pady=(8, 4))

        status_row = ttk.Frame(progress_frame)
        status_row.pack(fill="x")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_row, textvariable=self.status_var).pack(side="left")
        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(status_row, textvariable=self.elapsed_var).pack(side="right")

        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill="x", pady=(4, 0))

        # ── Form section ────────────────────────────────────────
        form_frame = ttk.LabelFrame(self.root, text="Configuration", padding=8)
        form_frame.pack(fill="x", padx=8, pady=4)

        # Operator name
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        self.operator_label = ttk.Label(row, text="Operator name: *", width=18, anchor="w", foreground="red")
        self.operator_label.pack(side="left")
        self.operator_var = tk.StringVar()
        self.operator_var.trace_add("write", self._on_operator_changed)
        self.operator_combo = ttk.Combobox(row, textvariable=self.operator_var)
        self.operator_combo["values"] = self._gui_state.get("recent_operators", [])
        self.operator_combo.pack(side="left", fill="x", expand=True)
        # Show/hide indicator on startup
        self.root.after(10, self._on_operator_changed)

        # Notes
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Notes:", width=18, anchor="nw").pack(side="left", anchor="n")
        self.notes_text = tk.Text(row, height=3, width=40)
        self.notes_text.pack(side="left", fill="x", expand=True)

        # Preset — dropdown shows human-readable names, maps back to keys
        self._preset_key_by_name = {p["name"]: k for k, p in PRESETS.items()}
        preset_names = list(self._preset_key_by_name)
        default_name = PRESETS[DEFAULT_PRESET]["name"]
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Scan preset:", width=18, anchor="w").pack(side="left")
        self.preset_var = tk.StringVar(value=default_name)
        ttk.OptionMenu(row, self.preset_var, default_name, *preset_names).pack(side="left")

        # Material
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Material:", width=18, anchor="w").pack(side="left")
        self.material_var = tk.StringVar(value="hbn")
        ttk.OptionMenu(row, self.material_var, "hbn", *MATERIALS).pack(side="left")

        # Revisit magnifications
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Revisit mags:", width=18, anchor="w").pack(side="left")
        self.revisit_10x = tk.BooleanVar(value=False)
        self.revisit_20x = tk.BooleanVar(value=True)
        self.revisit_50x = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="10x", variable=self.revisit_10x).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(row, text="20x", variable=self.revisit_20x).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(row, text="50x", variable=self.revisit_50x).pack(side="left")

        # Upload after scan
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Upload:", width=18, anchor="w").pack(side="left")
        self.upload_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Upload after scan", variable=self.upload_var).pack(side="left", padx=(0, 12))
        self.substrate_var = tk.StringVar(value="90nm")
        ttk.Label(row, text="Substrate:").pack(side="left", padx=(0, 4))
        ttk.OptionMenu(row, self.substrate_var, "90nm", *["90nm", "285nm"]).pack(side="left")

        # ── Advanced options (collapsed) ────────────────────────
        self.advanced_visible = tk.BooleanVar(value=False)
        self.advanced_toggle = ttk.Button(self.root, text="▶ Advanced options", command=self._toggle_advanced)
        self.advanced_toggle.pack(fill="x", padx=8, pady=(4, 0))

        self.advanced_frame = ttk.LabelFrame(self.root, text="Advanced", padding=8)
        # Not packed yet — toggled by button

        # Initial Z
        row = ttk.Frame(self.advanced_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Initial Z (µm):", width=22, anchor="w").pack(side="left")
        self.initial_z_var = tk.StringVar(value=DEFAULT_INITIAL_Z)
        ttk.Entry(row, textvariable=self.initial_z_var, width=12).pack(side="left")

        # Scan speed override
        row = ttk.Frame(self.advanced_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Scan speed (mm/s):", width=22, anchor="w").pack(side="left")
        self.scan_speed_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.scan_speed_var, width=8).pack(side="left")
        ttk.Label(row, text="blank = preset default", foreground="gray").pack(side="left", padx=4)

        # Area rect override
        row = ttk.Frame(self.advanced_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Area rect:", width=22, anchor="w").pack(side="left")
        self.area_rect_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.area_rect_var, width=28).pack(side="left")
        ttk.Label(row, text="blank = default", foreground="gray").pack(side="left", padx=4)

        # Max revisit count (per chip)
        row = ttk.Frame(self.advanced_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Max revisit count:", width=22, anchor="w").pack(side="left")
        self.revisit_top_var = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.revisit_top_var, width=8).pack(side="left")
        ttk.Label(row, text="blank = all T1", foreground="gray").pack(side="left", padx=4)

        # ── Buttons ─────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=8, pady=6)
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left")

        # ── Overview image (hidden until chip detection) ────────
        self.overview_toggle = ttk.Button(self.root, text="▶ Overview", command=self._toggle_overview)
        # Not packed yet — shown when detection image is available
        self.overview_visible = tk.BooleanVar(value=False)
        self.overview_frame = ttk.LabelFrame(self.root, text="Overview", padding=4)
        self.overview_label = ttk.Label(self.overview_frame)
        self.overview_label.pack()
        self.detection_counts_var = tk.StringVar(value="")
        ttk.Label(self.overview_frame, textvariable=self.detection_counts_var).pack(pady=(4, 0))

        # ── Log area ────────────────────────────────────────────
        self.log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        self.log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.log_text = tk.Text(self.log_frame, wrap="word", state="disabled", height=12)
        scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        # Collect all form widgets for enable/disable
        self._form_widgets = [
            self.operator_combo,
            self.notes_text,
            self.start_btn,
        ]

    def _on_operator_changed(self, *_args):
        if self.operator_var.get().strip():
            self.operator_label.configure(text="Operator name:", foreground="")
        else:
            self.operator_label.configure(text="Operator name: *", foreground="red")

    def _selected_preset_key(self) -> str:
        """Get preset key from the human-readable dropdown selection."""
        return self._preset_key_by_name.get(self.preset_var.get(), DEFAULT_PRESET)

    def _toggle_advanced(self):
        if self.advanced_visible.get():
            self.advanced_frame.pack_forget()
            self.advanced_toggle.configure(text="▶ Advanced options")
            self.advanced_visible.set(False)
        else:
            self.advanced_frame.pack(fill="x", padx=8, pady=(0, 4), after=self.advanced_toggle)
            self.advanced_toggle.configure(text="▼ Advanced options")
            self.advanced_visible.set(True)

    def _toggle_overview(self):
        if self.overview_visible.get():
            self.overview_frame.pack_forget()
            self.overview_toggle.configure(text="▶ Overview")
            self.overview_visible.set(False)
        else:
            self.overview_frame.pack(fill="x", padx=8, pady=(0, 4), after=self.overview_toggle)
            self.overview_toggle.configure(text="▼ Overview")
            self.overview_visible.set(True)

    def _log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_form_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in self._form_widgets:
            if isinstance(w, tk.Text):
                w.configure(state=state)
            else:
                w.configure(state=state)
        self.start_btn.configure(state="normal" if enabled else "disabled")
        self.stop_btn.configure(state="disabled" if enabled else "normal")

    def _show_overview_image(self):
        """Load and display the chip detection overlay image."""
        if not self.run_dir:
            return
        preset_key = self._selected_preset_key()
        mag = PRESETS[preset_key]["overview_mag"]
        img_path = self.run_dir / f"overview_{mag}_stitch_chips_detected.png"
        if not img_path.exists():
            return

        try:
            img = Image.open(img_path)
            # Scale to fit panel width, preserving aspect ratio
            w, h = img.size
            scale = OVERVIEW_MAX_WIDTH / w
            new_w = OVERVIEW_MAX_WIDTH
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self._overview_photo = ImageTk.PhotoImage(img)
            self.overview_label.configure(image=self._overview_photo)
            # Show toggle button and auto-expand
            self.overview_toggle.pack(fill="x", padx=8, pady=(4, 0), before=self.log_frame)
            self.overview_visible.set(False)
            self._toggle_overview()  # expand it
        except Exception as e:
            self._log(f"[gui] Could not load overview image: {e}\n")

    def _build_command(self) -> list[str]:
        cmd = ["uv", "run", "find-flakes", "-q"]

        preset_key = self._selected_preset_key()
        cmd += ["--preset", preset_key]

        material = self.material_var.get()
        cmd += ["--material", material]

        # Combine operator name + notes into --notes
        operator = self.operator_var.get().strip()
        notes_body = self.notes_text.get("1.0", "end").strip()
        notes_parts = [f"Operator: {operator}"]
        if notes_body:
            notes_parts.append(notes_body)
        cmd += ["--notes", "\n".join(notes_parts)]

        # Advanced options
        initial_z = self.initial_z_var.get().strip()
        if initial_z and initial_z != DEFAULT_INITIAL_Z:
            cmd += ["--initial-z", initial_z]

        if self.revisit_10x.get():
            cmd += ["--revisit-mag", "10x"]
        if self.revisit_20x.get():
            cmd += ["--revisit-mag", "20x"]
        if self.revisit_50x.get():
            cmd += ["--revisit-mag", "50x"]

        revisit_top = self.revisit_top_var.get().strip()
        if revisit_top:
            cmd += ["--revisit-top", revisit_top]

        scan_speed = self.scan_speed_var.get().strip()
        if scan_speed:
            cmd += ["--scan-speed", scan_speed]

        area_rect = self.area_rect_var.get().strip()
        if area_rect:
            cmd += ["--area-rect", area_rect]

        if self.upload_var.get():
            cmd += ["--upload", "--substrate", self.substrate_var.get()]

        return cmd

    def _on_start(self):
        operator = self.operator_var.get().strip()
        if not operator:
            messagebox.showwarning("Missing field", "Operator name is required.")
            self.operator_combo.focus_set()
            return

        # Save operator to recent list
        _add_recent_operator(self._gui_state, operator)
        _save_gui_state(self._gui_state)
        self.operator_combo["values"] = self._gui_state["recent_operators"]

        cmd = self._build_command()
        self._log(f"$ {' '.join(cmd)}\n\n")

        self._set_form_enabled(False)
        self.total_chips = None
        self.substeps_done = 0
        self.chip_phase_start = None
        self.chip_pos = 0
        self.run_dir = None
        self.t1_count = 0
        self.t2_count = 0
        self.revisit_pts_done = 0
        self.detection_counts_var.set("")
        # Hide overview from previous run
        self.overview_frame.pack_forget()
        self.overview_toggle.pack_forget()
        self.overview_visible.set(False)
        self._overview_photo = None
        self.progress.configure(mode="indeterminate", maximum=100)
        self.progress.start(30)
        self.status_var.set("Starting pipeline...")
        self.start_time = time.monotonic()

        try:
            # CREATE_NEW_PROCESS_GROUP on Windows so CTRL_BREAK_EVENT reaches children
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **kwargs,
            )
        except FileNotFoundError:
            self._log("ERROR: Could not find 'uv' command. Is uv installed?\n")
            self._run_finished()
            return

        threading.Thread(target=self._read_output, daemon=True).start()
        self._tick_timer()

    def _on_stop(self):
        if self.process and self.process.poll() is None:
            self._log("\n--- Stopping process ---\n")
            if sys.platform == "win32":
                # Send Ctrl+C to process group so Python gets KeyboardInterrupt
                import signal

                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()

    def _read_output(self):
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.root.after(0, self._process_line, line)
        self.process.wait()
        self.root.after(0, self._run_finished)

    def _num_revisit_mags(self) -> int:
        """Count how many revisit magnifications are enabled."""
        return sum(v.get() for v in (self.revisit_10x, self.revisit_20x, self.revisit_50x))

    def _progress_total(self) -> int:
        """Compute dynamic progress total: scan substeps + revisit points."""
        if not self.total_chips:
            return 0
        scan = self.total_chips * SUBSTEPS_PER_CHIP
        revisit = self.t1_count * self._num_revisit_mags()
        return scan + revisit

    def _sync_progress_bar(self):
        """Update progress bar value and maximum from current state."""
        total = self._progress_total()
        if total > 0:
            done = self.substeps_done + self.revisit_pts_done
            self.progress.configure(maximum=total, value=min(done, total))

    def _advance_substep(self):
        """Increment sub-step counter and update progress bar."""
        self.substeps_done += 1
        self._sync_progress_bar()

    def _compute_eta(self) -> str:
        """Compute ETA string from chip-phase pace, or empty string.

        Only shows ETA after at least one full chip is complete to avoid
        misleading estimates during the first chip's long sub-steps.
        """
        if not self.chip_phase_start or not self.total_chips:
            return ""
        if self.substeps_done < SUBSTEPS_PER_CHIP:
            return ""
        total = self._progress_total()
        done = self.substeps_done + self.revisit_pts_done
        remaining = total - done
        if remaining <= 0:
            return ""
        elapsed_chip = time.monotonic() - self.chip_phase_start
        pace = elapsed_chip / done
        eta_s = pace * remaining
        return f"~{_fmt_duration(eta_s)} remaining"

    def _process_line(self, line: str):
        self._log(line)
        stripped = line.strip()

        # Parse run directory from [run] line
        if stripped.startswith("[run]"):
            m = re.match(r"\[run\]\s+(.+?)/?$", stripped)
            if m:
                self.run_dir = Path(m.group(1))
            self.status_var.set("Taking overview...")
        elif stripped.startswith("[overview]"):
            self.status_var.set("Stitching overview...")
        elif stripped.startswith("[stitch]"):
            self.status_var.set("Detecting chips...")
        elif stripped.startswith("[detect]"):
            m = re.search(r"(\d+)\s+chips?", stripped)
            if m:
                self.total_chips = int(m.group(1))
                self.substeps_done = 0
                self.chip_pos = 1
                self.chip_phase_start = time.monotonic()
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=self._progress_total(), value=0)
            n = self.total_chips or "?"
            self.status_var.set(f"Focusing chip 1/{n}...")
            self._show_overview_image()
        elif re.match(r"\[chip \d+\] focus_map", stripped):
            m = re.match(r"\[chip (\d+)\]", stripped)
            if m:
                self._advance_substep()
                n = self.total_chips or "?"
                self.status_var.set(f"Scanning chip {self.chip_pos}/{n}...")
        elif re.match(r"\[chip \d+\] scan", stripped):
            m = re.match(r"\[chip (\d+)\]", stripped)
            if m:
                self._advance_substep()
                n = self.total_chips or "?"
                if self.chip_pos < (self.total_chips or 0):
                    self.chip_pos += 1
                    self.status_var.set(f"Focusing chip {self.chip_pos}/{n}...")
                else:
                    self.status_var.set("Finishing scans...")
        elif stripped.startswith("[seg chip"):
            # Don't count as progress substep (seg is pipelined), but track detections
            # and extend progress bar to account for upcoming revisit work
            m_t1 = re.search(r"T1:(\d+)", stripped)
            m_t2 = re.search(r"T2:(\d+)", stripped)
            if m_t1:
                self.t1_count += int(m_t1.group(1))
            if m_t2:
                self.t2_count += int(m_t2.group(1))
            if m_t1 or m_t2:
                self.detection_counts_var.set(f"Detections: {self.t1_count} T1, {self.t2_count} T2")
            if m_t1 and self._num_revisit_mags() > 0:
                self._sync_progress_bar()
        elif stripped.startswith("[scans done]"):
            self.status_var.set("Finishing segmentation...")
        elif re.match(r"\[revisit\]", stripped):
            # [revisit] 20x: 8 chip(s)
            m = re.match(r"\[revisit\]\s+(\S+):", stripped)
            mag = m.group(1) if m else ""
            self.status_var.set(f"Revisiting {mag}...")
        elif re.match(r"\[revisit chip", stripped):
            # [revisit chip 0] 20x: 5 pts, 15s
            m = re.match(r"\[revisit chip (\d+)\]\s+(\S+):", stripped)
            if m:
                self.status_var.set(f"Revisiting chip {m.group(1)} at {m.group(2)}...")
            m_pts = re.search(r"(\d+)\s+pts", stripped)
            if m_pts:
                self.revisit_pts_done += int(m_pts.group(1))
                self._sync_progress_bar()
        elif stripped.startswith("[done]"):
            if self.upload_var.get():
                self.status_var.set("Preparing upload...")
            else:
                self.status_var.set(f"Complete! {stripped[6:].strip()}")
            total = self._progress_total()
            if total > 0:
                self.progress.configure(value=total)
        elif stripped.startswith("[upload] Packaging") or stripped.startswith("[upload] Uploading"):
            self.status_var.set("Uploading...")
        elif stripped.startswith("[upload] Complete"):
            self.status_var.set("Upload complete!")
        elif stripped.startswith("[upload] Dry-run"):
            self.status_var.set("Upload skipped (dry-run)")
        elif stripped.startswith("[upload] FAILED"):
            self.status_var.set("Upload failed")

    def _tick_timer(self):
        if self.start_time is not None and self.process is not None and self.process.poll() is None:
            elapsed = time.monotonic() - self.start_time
            eta = self._compute_eta()
            if eta:
                self.elapsed_var.set(f"{_fmt_duration(elapsed)} elapsed · {eta}")
            else:
                self.elapsed_var.set(f"{_fmt_duration(elapsed)} elapsed")
            self.timer_id = self.root.after(1000, self._tick_timer)

    def _run_finished(self):
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

        rc = self.process.returncode if self.process else -1
        self.process = None

        # Show final elapsed
        if self.start_time is not None:
            elapsed = time.monotonic() - self.start_time
            self.elapsed_var.set(f"{_fmt_duration(elapsed)} total")

        self.progress.stop()
        total = self._progress_total()
        if total > 0:
            self.progress.configure(mode="determinate", value=total, maximum=total)
        else:
            self.progress.configure(mode="determinate", value=0)

        if rc == 0:
            self._log("\n--- Pipeline completed successfully ---\n")
        elif rc is not None and rc < 0:
            # Negative return code = killed by signal (Unix)
            self._log("\n--- Pipeline stopped ---\n")
            self.status_var.set("Stopped")
        elif rc is not None and rc > 255:
            # Large positive return code = terminated (Windows)
            self._log("\n--- Pipeline stopped ---\n")
            self.status_var.set("Stopped")
        else:
            self._log(f"\n--- Pipeline failed (code {rc}) ---\n")
            self.status_var.set(f"Failed (code {rc})")

        self._set_form_enabled(True)


def main():
    root = tk.Tk()
    FindFlakesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
