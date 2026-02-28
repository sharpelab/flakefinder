"""Left dock control panel: exposure, white balance, light, camera preview, objectives."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Exposure log-slider mapping: 0.1 ms → 10 000 ms (5 decades)
_EXP_LOG_MIN = -1.0  # log10(0.1)
_EXP_LOG_MAX = 4.0  # log10(10000)
_EXP_SLIDER_STEPS = 1000

# White-balance slider range
_WB_MIN = 0.50
_WB_MAX = 4.00
_WB_SLIDER_STEPS = 350  # 0.01 resolution

# Objectives in display order
_OBJECTIVES = ["2.5x", "5x", "10x", "20x", "50x", "150x"]


def _exp_slider_to_ms(val: int) -> float:
    """Slider integer (0–1000) → exposure in ms (log scale)."""
    log_val = val / _EXP_SLIDER_STEPS * (_EXP_LOG_MAX - _EXP_LOG_MIN) + _EXP_LOG_MIN
    return 10.0**log_val


def _exp_ms_to_slider(ms: float) -> int:
    """Exposure in ms → slider integer (0–1000)."""
    ms = max(0.1, min(10000.0, ms))
    log_val = math.log10(ms)
    return round((log_val - _EXP_LOG_MIN) / (_EXP_LOG_MAX - _EXP_LOG_MIN) * _EXP_SLIDER_STEPS)


def _wb_slider_to_gain(val: int) -> float:
    return _WB_MIN + val / _WB_SLIDER_STEPS * (_WB_MAX - _WB_MIN)


def _wb_gain_to_slider(gain: float) -> int:
    gain = max(_WB_MIN, min(_WB_MAX, gain))
    return round((gain - _WB_MIN) / (_WB_MAX - _WB_MIN) * _WB_SLIDER_STEPS)


class ControlPanel(QWidget):
    """Dock widget contents: exposure, white balance, light, camera preview, objectives.

    Signals:
        exposure_changed(ms): user changed exposure
        wb_changed(r, g, b): user changed white balance
        shutter_toggled(is_open): user toggled shutter
        lamp_changed(intensity): user changed lamp intensity
        objective_clicked(mag_str): user clicked an objective button
        preview_move_requested(x_um, y_um): double-click on preview to move
    """

    exposure_changed = Signal(float)
    wb_changed = Signal(float, float, float)
    shutter_toggled = Signal(bool)
    lamp_changed = Signal(int)
    objective_clicked = Signal(str)
    autofocus_requested = Signal()
    scan_requested = Signal()
    cancel_requested = Signal()
    preview_move_requested = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        self.setMinimumWidth(400)

        # ── Exposure ─────────────────────────────────────────────
        self._exposure_group = self._build_exposure_group()
        layout.addWidget(self._exposure_group)

        # ── White Balance ────────────────────────────────────────
        self._wb_group = self._build_wb_group()
        layout.addWidget(self._wb_group)

        # ── Light ────────────────────────────────────────────────
        self._light_group = self._build_light_group()
        layout.addWidget(self._light_group)

        # ── Objectives ───────────────────────────────────────────
        self._obj_group = self._build_objective_group()
        layout.addWidget(self._obj_group)

        # ── Autofocus ─────────────────────────────────────────────
        self._af_button = QPushButton("Autofocus")
        self._af_button.setMinimumHeight(32)
        self._af_button.setEnabled(False)
        self._af_button.clicked.connect(self._on_autofocus_clicked)
        layout.addWidget(self._af_button)

        # ── Scan ──────────────────────────────────────────────────
        self._scan_button = QPushButton("Scan ROI")
        self._scan_button.setMinimumHeight(32)
        self._scan_button.setEnabled(False)
        self._scan_button.clicked.connect(self._on_scan_clicked)
        layout.addWidget(self._scan_button)

        # ── Cancel (hidden until busy) ────────────────────────────
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setMinimumHeight(32)
        self._cancel_button.setStyleSheet("background: #a33; color: white; font-weight: bold;")
        self._cancel_button.clicked.connect(self.cancel_requested.emit)
        self._cancel_button.hide()
        layout.addWidget(self._cancel_button)

        # Widgets to disable during busy state
        self._control_widgets = [
            self._exposure_group,
            self._wb_group,
            self._light_group,
            self._obj_group,
            self._af_button,
            self._scan_button,
        ]

        # ── Camera Preview ───────────────────────────────────────
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(120)
        self._preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._preview_label.setStyleSheet("background: #1e1e1e; border: 1px solid #444;")
        self._preview_label.setText("No camera feed")
        self._preview_label.installEventFilter(self)
        layout.addWidget(self._preview_label, stretch=1)

        # Overlay for busy states (AF, scan)
        self._preview_overlay = QLabel(self._preview_label)
        self._preview_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_overlay.setStyleSheet(
            "background: rgba(0, 0, 0, 160); color: white; font-size: 16px; font-weight: bold;"
        )
        self._preview_overlay.hide()

        # Viewport state for preview click-to-move
        self._preview_viewport: tuple[float, float, float, float] | None = None  # (x, y, fov_w, fov_h)
        self._preview_pixmap_rect: tuple[int, int, int, int] | None = None  # (x, y, w, h) within label

        # Block signals during programmatic updates
        self._updating = False

    # ── Build helpers ────────────────────────────────────────────

    def _build_exposure_group(self) -> QGroupBox:
        group = QGroupBox("Exposure")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        self._exp_slider = QSlider(Qt.Orientation.Horizontal)
        self._exp_slider.setRange(0, _EXP_SLIDER_STEPS)
        self._exp_slider.setValue(_exp_ms_to_slider(1.0))
        self._exp_slider.valueChanged.connect(self._on_exp_slider)
        layout.addWidget(self._exp_slider, stretch=1)

        self._exp_spin = QDoubleSpinBox()
        self._exp_spin.setRange(0.1, 10000.0)
        self._exp_spin.setDecimals(2)
        self._exp_spin.setSuffix(" ms")
        self._exp_spin.setValue(1.0)
        self._exp_spin.setFixedWidth(60)
        self._exp_spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self._exp_spin.editingFinished.connect(self._on_exp_spin)
        layout.addWidget(self._exp_spin)

        return group

    def _build_wb_group(self) -> QGroupBox:
        group = QGroupBox("White Balance")
        grid = QGridLayout(group)
        grid.setContentsMargins(4, 4, 4, 4)

        self._wb_sliders: dict[str, QSlider] = {}
        self._wb_spins: dict[str, QDoubleSpinBox] = {}

        for row, (channel, color) in enumerate([("R", "#e44"), ("G", "#4a4"), ("B", "#48f")]):
            label = QLabel(channel)
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            label.setFixedWidth(14)
            grid.addWidget(label, row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, _WB_SLIDER_STEPS)
            slider.setValue(_wb_gain_to_slider(1.0))
            slider.valueChanged.connect(self._on_wb_slider)
            grid.addWidget(slider, row, 1)
            self._wb_sliders[channel] = slider

            spin = QDoubleSpinBox()
            spin.setRange(_WB_MIN, _WB_MAX)
            spin.setDecimals(2)
            spin.setSingleStep(0.01)
            spin.setValue(1.0)
            spin.setFixedWidth(48)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.editingFinished.connect(self._on_wb_spin)
            grid.addWidget(spin, row, 2)
            self._wb_spins[channel] = spin

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedHeight(22)
        reset_btn.clicked.connect(self._on_wb_reset)
        grid.addWidget(reset_btn, 3, 0, 1, 3)

        return group

    def _build_light_group(self) -> QGroupBox:
        group = QGroupBox("Light")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)

        # Shutter toggle
        shutter_row = QHBoxLayout()
        self._shutter_check = QCheckBox()
        self._shutter_check.toggled.connect(self._on_shutter_toggle)
        shutter_row.addWidget(self._shutter_check)
        self._shutter_label = QLabel("Shutter Closed")
        shutter_row.addWidget(self._shutter_label, stretch=1)
        layout.addLayout(shutter_row)

        # Intensity
        row = QHBoxLayout()
        self._lamp_slider = QSlider(Qt.Orientation.Horizontal)
        self._lamp_slider.setRange(0, 255)
        self._lamp_slider.setValue(0)
        self._lamp_slider.valueChanged.connect(self._on_lamp_slider)
        row.addWidget(self._lamp_slider, stretch=1)

        self._lamp_spin = QSpinBox()
        self._lamp_spin.setRange(0, 255)
        self._lamp_spin.setValue(0)
        self._lamp_spin.setFixedWidth(55)
        self._lamp_spin.editingFinished.connect(self._on_lamp_spin)
        row.addWidget(self._lamp_spin)

        layout.addLayout(row)
        return group

    def _build_objective_group(self) -> QGroupBox:
        group = QGroupBox("Mag")
        grid = QGridLayout(group)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(3)

        self._obj_buttons: dict[str, QPushButton] = {}
        for i, mag in enumerate(_OBJECTIVES):
            btn = QPushButton(mag)
            btn.setCheckable(True)
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda checked, m=mag: self._on_obj_clicked(m))
            grid.addWidget(btn, i // 3, i % 3)
            self._obj_buttons[mag] = btn

        return group

    # ── Public update methods (called from app when hardware state arrives) ──

    def set_exposure_ms(self, ms: float) -> None:
        """Update exposure controls without emitting signals."""
        self._updating = True
        self._exp_slider.setValue(_exp_ms_to_slider(ms))
        self._exp_spin.setValue(ms)
        self._updating = False

    def set_gain_rgb(self, r: float, g: float, b: float) -> None:
        self._updating = True
        for ch, val in [("R", r), ("G", g), ("B", b)]:
            self._wb_sliders[ch].setValue(_wb_gain_to_slider(val))
            self._wb_spins[ch].setValue(val)
        self._updating = False

    def set_shutter(self, is_open: bool) -> None:
        self._updating = True
        self._shutter_check.setChecked(is_open)
        self._shutter_label.setText("Shutter Open" if is_open else "Shutter Closed")
        self._updating = False

    def set_lamp(self, intensity: int, max_intensity: int) -> None:
        self._updating = True
        self._lamp_slider.setRange(0, max_intensity)
        self._lamp_spin.setRange(0, max_intensity)
        self._lamp_slider.setValue(intensity)
        self._lamp_spin.setValue(intensity)
        self._updating = False

    def set_objective(self, mag: float | None) -> None:
        """Highlight the active objective button."""
        for mag_str, btn in self._obj_buttons.items():
            btn_mag = float(mag_str.replace("x", ""))
            btn.setChecked(mag is not None and btn_mag == mag)

    def set_preview_viewport(self, x_um: float, y_um: float, fov_w: float, fov_h: float) -> None:
        """Update viewport info used for preview click-to-move mapping."""
        self._preview_viewport = (x_um, y_um, fov_w, fov_h)

    @Slot(QImage)
    def update_preview(self, qimg: QImage) -> None:
        """Display a camera frame in the preview label."""
        pix = QPixmap.fromImage(qimg)
        label_size = self._preview_label.size()
        scaled = pix.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Track where the pixmap sits within the label (centered)
        px = (label_size.width() - scaled.width()) // 2
        py = (label_size.height() - scaled.height()) // 2
        self._preview_pixmap_rect = (px, py, scaled.width(), scaled.height())
        self._preview_label.setPixmap(scaled)

    # ── Event filter (preview double-click) ────────────────────────

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._preview_label:
            if event.type() == QEvent.Type.MouseButtonDblClick:
                assert isinstance(event, QMouseEvent)
                self._on_preview_double_click(event)
                return True
            if event.type() == QEvent.Type.Resize and self._preview_overlay.isVisible():
                self._preview_overlay.resize(self._preview_label.size())
        return super().eventFilter(obj, event)

    def _on_preview_double_click(self, event: QMouseEvent) -> None:
        if self._preview_viewport is None or self._preview_pixmap_rect is None:
            return
        vx, vy, fov_w, fov_h = self._preview_viewport
        px, py, pw, ph = self._preview_pixmap_rect
        # Click position relative to pixmap origin
        cx = event.position().x() - px
        cy = event.position().y() - py
        if pw <= 0 or ph <= 0 or cx < 0 or cy < 0 or cx > pw or cy > ph:
            return
        # Map to stage coords: (0,0) = top-left of FOV, (pw,ph) = bottom-right
        stage_x = vx - fov_w / 2 + (cx / pw) * fov_w
        stage_y = vy - fov_h / 2 + (cy / ph) * fov_h
        self.preview_move_requested.emit(stage_x, stage_y)

    # ── Internal signal handlers ─────────────────────────────────

    def _on_exp_slider(self, val: int) -> None:
        if self._updating:
            return
        ms = _exp_slider_to_ms(val)
        self._updating = True
        self._exp_spin.setValue(ms)
        self._updating = False
        self.exposure_changed.emit(ms)

    def _on_exp_spin(self) -> None:
        if self._updating:
            return
        ms = self._exp_spin.value()
        self._updating = True
        self._exp_slider.setValue(_exp_ms_to_slider(ms))
        self._updating = False
        self.exposure_changed.emit(ms)

    def _on_wb_slider(self, _val: int) -> None:
        if self._updating:
            return
        r = _wb_slider_to_gain(self._wb_sliders["R"].value())
        g = _wb_slider_to_gain(self._wb_sliders["G"].value())
        b = _wb_slider_to_gain(self._wb_sliders["B"].value())
        self._updating = True
        self._wb_spins["R"].setValue(r)
        self._wb_spins["G"].setValue(g)
        self._wb_spins["B"].setValue(b)
        self._updating = False
        self.wb_changed.emit(r, g, b)

    def _on_wb_spin(self) -> None:
        if self._updating:
            return
        r = self._wb_spins["R"].value()
        g = self._wb_spins["G"].value()
        b = self._wb_spins["B"].value()
        self._updating = True
        self._wb_sliders["R"].setValue(_wb_gain_to_slider(r))
        self._wb_sliders["G"].setValue(_wb_gain_to_slider(g))
        self._wb_sliders["B"].setValue(_wb_gain_to_slider(b))
        self._updating = False
        self.wb_changed.emit(r, g, b)

    def _on_wb_reset(self) -> None:
        self.set_gain_rgb(1.0, 1.0, 1.0)
        self.wb_changed.emit(1.0, 1.0, 1.0)

    def _on_shutter_toggle(self, checked: bool) -> None:
        if self._updating:
            return
        self._shutter_label.setText("Shutter Open" if checked else "Shutter Closed")
        self.shutter_toggled.emit(checked)

    def _on_lamp_slider(self, val: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._lamp_spin.setValue(val)
        self._updating = False
        self.lamp_changed.emit(val)

    def _on_lamp_spin(self) -> None:
        if self._updating:
            return
        val = self._lamp_spin.value()
        self._updating = True
        self._lamp_slider.setValue(val)
        self._updating = False
        self.lamp_changed.emit(val)

    def _on_obj_clicked(self, mag: str) -> None:
        if self._updating:
            return
        self.objective_clicked.emit(mag)

    def _on_autofocus_clicked(self) -> None:
        self.set_busy("Focusing…")
        self.autofocus_requested.emit()

    def _on_scan_clicked(self) -> None:
        self.set_busy("Scanning…")
        self.scan_requested.emit()

    def set_busy(self, label: str) -> None:
        """Disable all controls, show cancel button and preview overlay."""
        for w in self._control_widgets:
            w.setEnabled(False)
        self._cancel_button.show()
        self.set_preview_overlay(label)

    def set_idle(self) -> None:
        """Re-enable controls after a busy operation completes."""
        for w in self._control_widgets:
            w.setEnabled(True)
        self._cancel_button.hide()
        self.set_preview_overlay(None)
        # AF/scan buttons have their own enable logic — reset to default disabled
        # The app will re-enable them based on connection + ROI state
        self._af_button.setEnabled(False)
        self._scan_button.setEnabled(False)

    def set_autofocus_enabled(self, enabled: bool) -> None:
        self._af_button.setEnabled(enabled)

    def set_scan_enabled(self, enabled: bool) -> None:
        self._scan_button.setEnabled(enabled)

    def set_roi_info(self, w_um: float, h_um: float) -> None:
        """Update the scan button label with ROI dimensions."""
        if w_um >= 1000:
            self._scan_button.setText(f"Scan ROI ({w_um / 1000:.1f} × {h_um / 1000:.1f} mm)")
        else:
            self._scan_button.setText(f"Scan ROI ({w_um:.0f} × {h_um:.0f} µm)")

    def set_preview_overlay(self, text: str | None) -> None:
        """Show or hide a text overlay on the camera preview."""
        if text is None:
            self._preview_overlay.hide()
        else:
            self._preview_overlay.setText(text)
            self._preview_overlay.resize(self._preview_label.size())
            self._preview_overlay.show()
