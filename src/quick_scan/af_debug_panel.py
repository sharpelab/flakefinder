"""Autofocus debug panel — popout with sharpness curve plot and phase overrides."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from flakefinder.leica.autofocus import AF_DEFAULTS, AFDefaults, AutofocusResult


def _fmt(val: float | None) -> str:
    """Format a phase value for Python-code export (int when clean, else float)."""
    if val is None:
        return "None"
    if float(val).is_integer():
        return str(int(val))
    return str(val)


class _PhaseWidgets:
    """Widgets for a single AF phase (coarse, fine, or super-fine)."""

    def __init__(self, name: str, grid: QGridLayout, row: int):
        self.enable = QCheckBox(name)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1.0, 5000.0)
        self.range_spin.setDecimals(0)
        self.range_spin.setSuffix(" µm")
        self.range_spin.setFixedWidth(90)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 10000.0)
        self.speed_spin.setDecimals(0)
        self.speed_spin.setSuffix(" µm/s")
        self.speed_spin.setFixedWidth(100)
        grid.addWidget(self.enable, row, 0)
        grid.addWidget(QLabel("range"), row, 1)
        grid.addWidget(self.range_spin, row, 2)
        grid.addWidget(QLabel("speed"), row, 3)
        grid.addWidget(self.speed_spin, row, 4)

    def set_values(self, range_um: float | None, speed_um_s: float | None) -> None:
        """Populate from defaults. None = phase skipped (unchecked)."""
        enabled = range_um is not None and speed_um_s is not None
        self.enable.setChecked(enabled)
        if range_um is not None:
            self.range_spin.setValue(range_um)
        if speed_um_s is not None:
            self.speed_spin.setValue(speed_um_s)

    def get_values(self) -> tuple[float | None, float | None]:
        """Return (range, speed) or (None, None) if phase is unchecked."""
        if not self.enable.isChecked():
            return (None, None)
        return (self.range_spin.value(), self.speed_spin.value())

    def set_editable(self, editable: bool) -> None:
        self.enable.setEnabled(editable)
        self.range_spin.setEnabled(editable)
        self.speed_spin.setEnabled(editable)


class AFDebugPanel(QDialog):
    """Popout autofocus debugging dialog.

    Top: Run button + status. Middle: sharpness-curve plot with inset images.
    Bottom: override controls for coarse/fine/super-fine phase parameters.
    """

    run_requested = Signal(object)  # AFDefaults or None
    z_step_requested = Signal(float)  # signed delta in µm

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Autofocus Debug")
        self.resize(1000, 780)
        self._current_objective_pos: int | None = None
        self._save_dir: Path | None = None
        self._dirty = False
        self._updating = False

        layout = QVBoxLayout(self)

        # ── Top row: Run button + status ─────────────────────────
        top_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Autofocus")
        self._run_btn.setMinimumHeight(32)
        self._run_btn.clicked.connect(self._on_run_clicked)
        top_row.addWidget(self._run_btn)
        self._status_label = QLabel("No run yet")
        self._status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top_row.addWidget(self._status_label)
        layout.addLayout(top_row)

        # ── Plot ─────────────────────────────────────────────────
        self._figure = Figure(figsize=(10, 5))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas, stretch=1)
        self._draw_empty()

        # ── Z control row ────────────────────────────────────────
        # Up = -Z (retract, away from sample); Down = +Z (toward sample, crash risk).
        z_row = QHBoxLayout()
        self._z_label = QLabel("Z: — µm")
        self._z_label.setMinimumWidth(120)
        z_row.addWidget(self._z_label)
        self._z_up_btn = QPushButton("Up")
        self._z_up_btn.setToolTip("Retract (−Z, away from sample)")
        self._z_up_btn.clicked.connect(lambda: self.z_step_requested.emit(-self._z_step_spin.value()))
        z_row.addWidget(self._z_up_btn)
        z_row.addWidget(QLabel("step"))
        self._z_step_spin = QDoubleSpinBox()
        self._z_step_spin.setRange(0.1, 500.0)
        self._z_step_spin.setDecimals(1)
        self._z_step_spin.setSingleStep(1.0)
        self._z_step_spin.setSuffix(" µm")
        self._z_step_spin.setValue(10.0)
        self._z_step_spin.setFixedWidth(90)
        z_row.addWidget(self._z_step_spin)
        self._z_down_btn = QPushButton("Down")
        self._z_down_btn.setToolTip("Approach (+Z, toward sample)")
        self._z_down_btn.clicked.connect(lambda: self.z_step_requested.emit(self._z_step_spin.value()))
        z_row.addWidget(self._z_down_btn)
        z_row.addStretch(1)
        layout.addLayout(z_row)

        # ── Override group ───────────────────────────────────────
        override_group = QGroupBox("Overrides")
        og_layout = QVBoxLayout(override_group)

        self._override_check = QCheckBox("Override defaults")
        self._override_check.toggled.connect(self._on_override_toggled)
        og_layout.addWidget(self._override_check)

        phase_grid = QGridLayout()
        self._coarse = _PhaseWidgets("Coarse", phase_grid, 0)
        self._coarse.enable.setChecked(True)
        self._coarse.enable.setEnabled(False)  # coarse always required
        self._fine = _PhaseWidgets("Fine", phase_grid, 1)
        self._super_fine = _PhaseWidgets("Super-fine", phase_grid, 2)
        og_layout.addLayout(phase_grid)

        # Wire dirty-tracking on every edit
        for phase in (self._coarse, self._fine, self._super_fine):
            phase.enable.toggled.connect(self._mark_dirty)
            phase.range_spin.valueChanged.connect(self._mark_dirty)
            phase.speed_spin.valueChanged.connect(self._mark_dirty)

        # Bottom row: reset + open folder
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset to defaults")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        btn_row.addWidget(self._reset_btn)
        self._open_folder_btn = QPushButton("Open folder")
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        btn_row.addWidget(self._open_folder_btn)
        self._export_btn = QPushButton("Export to clipboard")
        self._export_btn.setToolTip("Copy current phase settings as an AFDefaults entry")
        self._export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self._export_btn)
        btn_row.addStretch(1)
        og_layout.addLayout(btn_row)

        layout.addWidget(override_group)

        # Start with override off → phases disabled
        self._set_phases_editable(False)

    # ── Public API (from app.py) ─────────────────────────────────

    def set_objective(self, position: int | None) -> None:
        """Inform panel of active objective. Updates default display when not dirty and not overriding."""
        self._current_objective_pos = position
        if not self._override_check.isChecked() and not self._dirty:
            self._load_defaults_for_current_objective()

    def set_busy(self, busy: bool) -> None:
        self._run_btn.setEnabled(not busy)
        self._z_up_btn.setEnabled(not busy)
        self._z_down_btn.setEnabled(not busy)
        self._z_step_spin.setEnabled(not busy)
        if busy:
            self._status_label.setText("Running autofocus…")

    def set_z(self, z_um: float) -> None:
        self._z_label.setText(f"Z: {z_um:.1f} µm")

    def show_result(self, result: AutofocusResult, save_dir: str) -> None:
        """Update plot and status from completed AF result."""
        self._save_dir = Path(save_dir)
        self._open_folder_btn.setEnabled(True)
        dz = result.selected_z_um - result.initial_z_um
        self._status_label.setText(
            f"Δz = {dz:+.2f} µm · best S = {result.selected_sharpness:.1f} · "
            f"{result.frame_count} frames · {result.scan_duration_s:.1f}s · "
            f"saved to {self._save_dir.name}/"
        )
        self._draw_result(result)

    # ── Internal ────────────────────────────────────────────────

    def _load_defaults_for_current_objective(self) -> None:
        """Populate phase fields from AF_DEFAULTS for the current objective."""
        self._updating = True
        try:
            af = AF_DEFAULTS.get(self._current_objective_pos) if self._current_objective_pos else None
            if af is None:
                # Generic fallback for objectives without tuned defaults (e.g. 150x)
                af = AFDefaults(
                    coarse_range_um=500.0,
                    coarse_speed_um_s=1000.0,
                    fine_range_um=50.0,
                    fine_speed_um_s=250.0,
                    super_fine_range_um=None,
                    super_fine_speed_um_s=None,
                )
            self._coarse.set_values(af.coarse_range_um, af.coarse_speed_um_s)
            self._fine.set_values(af.fine_range_um, af.fine_speed_um_s)
            self._super_fine.set_values(af.super_fine_range_um, af.super_fine_speed_um_s)
        finally:
            self._updating = False
        self._dirty = False

    def _set_phases_editable(self, editable: bool) -> None:
        self._coarse.range_spin.setEnabled(editable)
        self._coarse.speed_spin.setEnabled(editable)
        self._fine.set_editable(editable)
        self._super_fine.set_editable(editable)

    def _mark_dirty(self, *_args) -> None:
        if self._updating:
            return
        self._dirty = True

    def _build_af_defaults(self) -> AFDefaults | None:
        """Construct AFDefaults from UI. Returns None if override is off."""
        if not self._override_check.isChecked():
            return None
        c_range, c_speed = self._coarse.get_values()
        f_range, f_speed = self._fine.get_values()
        sf_range, sf_speed = self._super_fine.get_values()
        # Coarse is required — _coarse.enable is permanently checked
        assert c_range is not None and c_speed is not None
        return AFDefaults(
            coarse_range_um=c_range,
            coarse_speed_um_s=c_speed,
            fine_range_um=f_range,
            fine_speed_um_s=f_speed,
            super_fine_range_um=sf_range,
            super_fine_speed_um_s=sf_speed,
        )

    def _on_override_toggled(self, checked: bool) -> None:
        self._set_phases_editable(checked)
        if not checked and not self._dirty:
            self._load_defaults_for_current_objective()

    def _on_reset_clicked(self) -> None:
        self._load_defaults_for_current_objective()

    def _on_run_clicked(self) -> None:
        self.run_requested.emit(self._build_af_defaults())

    def _on_open_folder_clicked(self) -> None:
        if self._save_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._save_dir.resolve())))

    def _on_export_clicked(self) -> None:
        """Copy current phase settings to clipboard as an AFDefaults(...) block."""
        c_range, c_speed = self._coarse.get_values()
        f_range, f_speed = self._fine.get_values()
        sf_range, sf_speed = self._super_fine.get_values()
        pos = self._current_objective_pos if self._current_objective_pos is not None else "?"
        text = (
            f"{pos}: AFDefaults(\n"
            f"    coarse_range_um={_fmt(c_range)},\n"
            f"    coarse_speed_um_s={_fmt(c_speed)},\n"
            f"    fine_range_um={_fmt(f_range)},\n"
            f"    fine_speed_um_s={_fmt(f_speed)},\n"
            f"    super_fine_range_um={_fmt(sf_range)},\n"
            f"    super_fine_speed_um_s={_fmt(sf_speed)},\n"
            f"),"
        )
        QApplication.clipboard().setText(text)
        self._status_label.setText(f"Copied AFDefaults for pos {pos} to clipboard")

    # ── Plotting ────────────────────────────────────────────────

    def _draw_empty(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.text(0.5, 0.5, "Run autofocus to see sharpness curve", ha="center", va="center", color="gray")
        ax.set_axis_off()
        self._canvas.draw()

    def _draw_result(self, result: AutofocusResult) -> None:
        from flakefinder.commands.analyze_focus_map import plot_af_curves

        self._figure.clear()

        # Decide layout based on which insets are available
        insets = []
        if result.initial_image is not None:
            insets.append(("Initial", result.initial_image, result.initial_z_um, result.initial_sharpness))
        if result.best_frame is not None:
            insets.append(
                (
                    "Best",
                    result.best_frame.image,
                    result.best_frame.z_um,
                    result.best_frame.sharpness,
                )
            )
        if result.final_image is not None:
            insets.append(("Final", result.final_image, result.selected_z_um, result.final_sharpness))

        has_insets = len(insets) > 0
        if has_insets:
            ax_main = self._figure.add_axes((0.08, 0.12, 0.55, 0.80))
            inset_left = 0.66
            inset_width = 0.31
        else:
            ax_main = self._figure.add_axes((0.08, 0.12, 0.88, 0.80))

        plot_af_curves(
            ax_main,
            coarse_curve=cast(list[dict], list(result.sharpness_curve)),
            fine_curve=cast(list[dict], list(result.fine_sharpness_curve)),
            super_fine_curve=cast(list[dict], list(result.super_fine_sharpness_curve)),
            initial_z_um=result.initial_z_um,
            selected_z_um=result.selected_z_um,
            selected_sharpness=result.selected_sharpness,
            initial_label=f"Initial Z = {result.initial_z_um:.1f}",
            selected_label=f"Best Z = {result.selected_z_um:.1f}",
        )
        ax_main.set_xlabel("Z position (µm)")
        ax_main.set_ylabel("Sharpness")
        ax_main.legend(loc="best", fontsize=8)
        dz = result.selected_z_um - result.initial_z_um
        ax_main.set_title(f"Autofocus: Δz {dz:+.2f} µm")

        if has_insets:
            n = len(insets)
            inset_h = 0.80 / n - 0.02
            for i, (label, img, z, s) in enumerate(insets):
                y_pos = 0.12 + (n - 1 - i) * (inset_h + 0.02)
                ax_img = self._figure.add_axes((inset_left, y_pos, inset_width, inset_h))
                ax_img.imshow(np.asarray(img))
                ax_img.set_title(f"{label} (Z={z:.1f}, S={s:.1f})", fontsize=8)
                ax_img.axis("off")

        self._canvas.draw()
