"""Quick Scan GUI — interactive stage viewer for the Leica DM6M.

Usage:
    quick-scan              # connect to microscope on startup
    quick-scan --offline    # offline mode (canvas only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QIcon, QImage, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from quick_scan.control_panel import ControlPanel
from quick_scan.scope_manager import HardwareState, ScopeManager
from quick_scan.stage_canvas import StageCanvas, ViewportInfo


class QuickScanWindow(QMainWindow):
    """Main application window."""

    def __init__(self, connect: bool = False):
        super().__init__()
        self.setWindowTitle("Quick Scan")
        self.resize(1400, 900)

        icon_path = Path(__file__).parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # ── Central widget: stage canvas ─────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = StageCanvas()
        layout.addWidget(self._canvas)

        # ── Left dock: controls + camera preview ─────────────────
        self._controls = ControlPanel()
        dock = QDockWidget("Controls", self)
        dock.setWidget(self._controls)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # ── Toolbar ──────────────────────────────────────────────
        self._build_toolbar()

        # ── Status bar ───────────────────────────────────────────
        self._build_status_bar()

        # ── Scope manager ────────────────────────────────────────
        self._scope = ScopeManager(self)
        self._connected = False

        # Cached state for status bar
        self._scope_x = 0.0
        self._scope_y = 0.0
        self._scope_z = 0.0
        self._scope_mag: float | None = None

        # Wire scope signals
        self._scope.position_updated.connect(self._on_scope_position)
        self._scope.frame_ready.connect(self._on_frame)
        self._scope.hw_state_ready.connect(self._on_hw_state)
        self._scope.scope_connected.connect(self._on_connected)
        self._scope.scope_disconnected.connect(self._on_disconnected)
        self._scope.scope_error.connect(self._on_scope_error)

        # Wire canvas signals
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.move_requested.connect(self._on_move_requested)

        # Wire control signals
        self._controls.exposure_changed.connect(self._scope.set_exposure_ms)
        self._controls.wb_changed.connect(self._scope.set_gain_rgb)
        self._controls.shutter_toggled.connect(self._scope.set_shutter)
        self._controls.lamp_changed.connect(self._scope.set_lamp_intensity)
        self._controls.objective_clicked.connect(self._scope.switch_objective)
        self._controls.autofocus_requested.connect(self._scope.autofocus)
        self._controls.preview_move_requested.connect(self._on_move_requested)

        # AF completion → re-enable button + update status
        self._scope.autofocus_finished.connect(self._on_autofocus_finished)

        # Status bar refresh timer
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(200)

        if connect:
            self._scope.open()

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        overview_action = QAction("Go to Overview", self)
        overview_action.setShortcut(QKeySequence("Home"))
        overview_action.setToolTip("Fit full stage in view (Home)")
        overview_action.triggered.connect(self._canvas.go_to_overview)
        toolbar.addAction(overview_action)

        frame_action = QAction("Go to Frame", self)
        frame_action.setShortcut(QKeySequence("F"))
        frame_action.setToolTip("Center on current microscope viewport (F)")
        frame_action.triggered.connect(self._canvas.go_to_frame)
        toolbar.addAction(frame_action)

        toolbar.addSeparator()

        self._connect_action = QAction("Connect", self)
        self._connect_action.setCheckable(True)
        self._connect_action.setToolTip("Connect to microscope")
        self._connect_action.triggered.connect(self._on_connect_toggled)
        toolbar.addAction(self._connect_action)

    def _build_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)

        self._cursor_label = QLabel("Cursor: —")
        self._cursor_label.setMinimumWidth(220)
        status.addWidget(self._cursor_label)

        self._scope_label = QLabel("Stage: —")
        self._scope_label.setMinimumWidth(320)
        status.addWidget(self._scope_label)

        self._zoom_label = QLabel("Zoom: —")
        self._zoom_label.setMinimumWidth(120)
        status.addPermanentWidget(self._zoom_label)

    # ── Connection lifecycle ─────────────────────────────────────

    @Slot(bool)
    def _on_connect_toggled(self, checked: bool):
        if checked:
            self._scope_label.setText("Stage: connecting...")
            self._scope.open()
        else:
            self._scope.close()

    @Slot()
    def _on_connected(self):
        self._connected = True
        self._connect_action.setChecked(True)
        self._scope_label.setText("Stage: connected")

    @Slot()
    def _on_disconnected(self):
        self._connected = False
        self._connect_action.setChecked(False)
        self._canvas.hide_viewport()
        self._scope_label.setText("Stage: disconnected")

    @Slot(str)
    def _on_scope_error(self, msg: str):
        self._scope_label.setText(f"Stage: error — {msg}")
        self._connected = False
        self._connect_action.setChecked(False)
        self._canvas.hide_viewport()
        self._controls.set_autofocus_enabled(True)

    # ── Scope updates ────────────────────────────────────────────

    @Slot(float, float, float, object, int, float, float)
    def _on_scope_position(self, x, y, z, mag, obj_pos, fov_w, fov_h):
        self._scope_x = x
        self._scope_y = y
        self._scope_z = z
        self._scope_mag = mag

        if fov_w > 0 and fov_h > 0:
            self._canvas.set_viewport(ViewportInfo(x, y, fov_w, fov_h))
            self._controls.set_preview_viewport(x, y, fov_w, fov_h)

        mag_str = f"{mag}x" if mag else f"pos {obj_pos}"
        self._scope_label.setText(f"Stage: X={x:.0f}  Y={y:.0f}  Z={z:.0f} µm  [{mag_str}]")

        # Keep objective buttons in sync
        self._controls.set_objective(mag)

    @Slot(QImage)
    def _on_frame(self, qimg: QImage):
        # Update canvas overlay
        self._canvas.update_camera_frame(qimg)
        # Update dock preview
        self._controls.update_preview(qimg)

    @Slot(object)
    def _on_hw_state(self, state: HardwareState):
        self._controls.set_exposure_ms(state.exposure_ms)
        self._controls.set_gain_rgb(*state.gain_rgb)
        self._controls.set_shutter(state.shutter_open)
        self._controls.set_lamp(state.lamp_intensity, state.lamp_max)

    @Slot(float, float)
    def _on_autofocus_finished(self, best_z: float, best_sharpness: float):
        self._controls.set_autofocus_enabled(True)
        self._scope_label.setText(f"Stage: AF done — Z={best_z:.1f} µm (sharpness {best_sharpness:.0f})")

    # ── Canvas interactions ──────────────────────────────────────

    @Slot(float, float)
    def _on_cursor_moved(self, x_um: float, y_um: float):
        self._cursor_label.setText(f"Cursor: ({x_um:.0f}, {y_um:.0f}) µm")

    @Slot(float, float)
    def _on_move_requested(self, x_um: float, y_um: float):
        if self._connected:
            self._scope.move_to(x_um, y_um)

    # ── Status bar ───────────────────────────────────────────────

    @Slot()
    def _refresh_status(self):
        um_px = self._canvas.um_per_px()
        if um_px >= 1:
            self._zoom_label.setText(f"Zoom: {um_px:.0f} µm/px")
        else:
            self._zoom_label.setText(f"Zoom: {um_px:.2f} µm/px")

    # ── Cleanup ──────────────────────────────────────────────────

    def closeEvent(self, event):
        self._scope.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Quick Scan — interactive stage viewer")
    parser.add_argument("--offline", action="store_true", help="Start without connecting to microscope")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Quick Scan")

    window = QuickScanWindow(connect=not args.offline)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
