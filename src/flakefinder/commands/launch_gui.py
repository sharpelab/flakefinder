"""Tkinter GUI launcher for the find-flakes pipeline.

Provides a simple form interface for collaborators to launch find-flakes
runs without knowing CLI syntax. Runs on the microscope PC (Windows 11).

Usage:
    uv run launch-gui
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

# Presets from find_flakes.py (keep in sync)
PRESETS = ("5_20", "2.5_10")
DEFAULT_PRESET = "2.5_10"
MATERIALS = ("hbn", "graphene")
DEFAULT_INITIAL_Z = "24690"
DEFAULT_AREA_RECT = "8000,95000,0,78000"
DEFAULT_REVISIT_TOP = "20"

# Sub-steps per chip for progress tracking: focus_map, scan, seg
SUBSTEPS_PER_CHIP = 3


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

        self.process: subprocess.Popen | None = None
        self.start_time: float | None = None
        self.timer_id: str | None = None
        self.total_chips: int | None = None
        self.substeps_done: int = 0
        self.chip_phase_start: float | None = None  # when chip processing began

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
        ttk.Label(row, text="Operator name:", width=18, anchor="w").pack(side="left")
        self.operator_var = tk.StringVar()
        self.operator_entry = ttk.Entry(row, textvariable=self.operator_var)
        self.operator_entry.pack(side="left", fill="x", expand=True)

        # Notes
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Notes:", width=18, anchor="nw").pack(side="left", anchor="n")
        self.notes_text = tk.Text(row, height=3, width=40)
        self.notes_text.pack(side="left", fill="x", expand=True)

        # Preset
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Scan preset:", width=18, anchor="w").pack(side="left")
        self.preset_var = tk.StringVar(value=DEFAULT_PRESET)
        ttk.OptionMenu(row, self.preset_var, DEFAULT_PRESET, *PRESETS).pack(side="left")

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

        # Revisit top N
        row = ttk.Frame(form_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Revisit top N:", width=18, anchor="w").pack(side="left")
        self.revisit_top_var = tk.StringVar(value=DEFAULT_REVISIT_TOP)
        ttk.Entry(row, textvariable=self.revisit_top_var, width=8).pack(side="left")

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

        # ── Buttons ─────────────────────────────────────────────
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=8, pady=6)
        self.start_btn = ttk.Button(btn_frame, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=(0, 4))
        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left")

        # ── Log area ────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.log_text = tk.Text(log_frame, wrap="word", state="disabled", height=12)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

        # Collect all form widgets for enable/disable
        self._form_widgets = [
            self.operator_entry,
            self.notes_text,
            self.start_btn,
        ]

    def _toggle_advanced(self):
        if self.advanced_visible.get():
            self.advanced_frame.pack_forget()
            self.advanced_toggle.configure(text="▶ Advanced options")
            self.advanced_visible.set(False)
        else:
            self.advanced_frame.pack(fill="x", padx=8, pady=(0, 4), after=self.advanced_toggle)
            self.advanced_toggle.configure(text="▼ Advanced options")
            self.advanced_visible.set(True)

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

    def _build_command(self) -> list[str]:
        cmd = ["uv", "run", "find-flakes", "-q"]

        preset = self.preset_var.get()
        cmd += ["--preset", preset]

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
        if revisit_top and revisit_top != DEFAULT_REVISIT_TOP:
            cmd += ["--revisit-top", revisit_top]

        scan_speed = self.scan_speed_var.get().strip()
        if scan_speed:
            cmd += ["--scan-speed", scan_speed]

        area_rect = self.area_rect_var.get().strip()
        if area_rect:
            cmd += ["--area-rect", area_rect]

        return cmd

    def _on_start(self):
        operator = self.operator_var.get().strip()
        if not operator:
            messagebox.showwarning("Missing field", "Operator name is required.")
            self.operator_entry.focus_set()
            return

        cmd = self._build_command()
        self._log(f"$ {' '.join(cmd)}\n\n")

        self._set_form_enabled(False)
        self.total_chips = None
        self.substeps_done = 0
        self.chip_phase_start = None
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

    def _advance_substep(self):
        """Increment sub-step counter and update progress bar."""
        self.substeps_done += 1
        if self.total_chips:
            total = self.total_chips * SUBSTEPS_PER_CHIP
            self.progress.configure(value=min(self.substeps_done, total))

    def _compute_eta(self) -> str:
        """Compute ETA string from chip-phase pace, or empty string."""
        if not self.chip_phase_start or not self.total_chips or self.substeps_done == 0:
            return ""
        total = self.total_chips * SUBSTEPS_PER_CHIP
        remaining = total - self.substeps_done
        if remaining <= 0:
            return ""
        elapsed_chip = time.monotonic() - self.chip_phase_start
        pace = elapsed_chip / self.substeps_done
        eta_s = pace * remaining
        return f"~{_fmt_duration(eta_s)} remaining"

    def _process_line(self, line: str):
        self._log(line)
        stripped = line.strip()

        # Parse progress from quiet-mode output
        if stripped.startswith("[run]"):
            self.status_var.set("Pipeline started")
        elif stripped.startswith("[overview]"):
            self.status_var.set("Overview complete")
        elif stripped.startswith("[stitch]"):
            self.status_var.set("Stitch complete")
        elif stripped.startswith("[detect]"):
            m = re.search(r"(\d+)\s+chips?", stripped)
            if m:
                self.total_chips = int(m.group(1))
                self.substeps_done = 0
                self.chip_phase_start = time.monotonic()
                self.progress.stop()
                total = self.total_chips * SUBSTEPS_PER_CHIP
                self.progress.configure(mode="determinate", maximum=total, value=0)
            self.status_var.set(f"Detected {self.total_chips} chips")
        elif re.match(r"\[chip \d+\] focus_map", stripped):
            m = re.match(r"\[chip (\d+)\]", stripped)
            if m:
                self._advance_substep()
                self.status_var.set(f"Chip {m.group(1)} — focus map done")
        elif re.match(r"\[chip \d+\] scan", stripped):
            m = re.match(r"\[chip (\d+)\]", stripped)
            if m:
                self._advance_substep()
                self.status_var.set(f"Chip {m.group(1)} — scan done")
        elif stripped.startswith("[seg chip"):
            m = re.match(r"\[seg chip (\d+)\]", stripped)
            if m:
                self._advance_substep()
                self.status_var.set(f"Segmentation chip {m.group(1)} done")
        elif stripped.startswith("[scans done]"):
            self.status_var.set("Scans complete, finishing segmentation...")
        elif stripped.startswith("[revisit"):
            self.status_var.set("Revisit captures...")
        elif stripped.startswith("[done]"):
            self.status_var.set(f"Complete! {stripped[6:].strip()}")
            if self.total_chips:
                total = self.total_chips * SUBSTEPS_PER_CHIP
                self.progress.configure(value=total)

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
        if self.total_chips:
            total = self.total_chips * SUBSTEPS_PER_CHIP
            self.progress.configure(mode="determinate", value=total, maximum=total)
        else:
            self.progress.configure(mode="determinate", value=0)

        if rc == 0:
            self._log("\n--- Pipeline completed successfully ---\n")
        else:
            self._log(f"\n--- Pipeline exited with code {rc} ---\n")
            self.status_var.set(f"Exited (code {rc})")

        self._set_form_enabled(True)


def main():
    root = tk.Tk()
    FindFlakesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
