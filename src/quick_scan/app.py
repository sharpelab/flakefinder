"""Quick Scan GUI — interactive stage viewer for the Leica DM6M.

Usage:
    quick-scan              # offline mode (canvas only)
    quick-scan --connect    # connect to microscope for live viewport
"""

from __future__ import annotations

import argparse
import sys
import threading

from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from quick_scan.stage_canvas import StageCanvas, ViewportInfo


class MicroscopePoller(threading.Thread):
    """Background thread that polls microscope position.

    Posts results via a callback that must be scheduled onto the GUI thread.
    """

    def __init__(self, on_update, on_error, poll_interval_s: float = 0.2):
        super().__init__(daemon=True)
        self._on_update = on_update
        self._on_error = on_error
        self._poll_interval = poll_interval_s
        self._stop = threading.Event()

    def run(self):
        try:
            from flakefinder.data_utils import compute_frame_size_um, require_microscope_description
            from flakefinder.leica.microscope import Microscope

            desc = require_microscope_description()

            scope = Microscope()
            scope.__enter__()
            try:
                while not self._stop.is_set():
                    x, y = scope.stage.position_um
                    z = scope.z.position_um
                    mag = scope.nosepiece.magnification
                    obj_pos = scope.nosepiece.position

                    fov_w, fov_h = 0.0, 0.0
                    if mag is not None:
                        # 3x3 binning (index 2) is the default
                        size = compute_frame_size_um(desc.camera, mag, binning_idx=2)
                        if size is not None:
                            fov_w, fov_h = size

                    self._on_update(x, y, z, mag, obj_pos, fov_w, fov_h)
                    self._stop.wait(self._poll_interval)
            finally:
                scope.__exit__(None, None, None)
        except Exception as e:
            self._on_error(str(e))

    def stop(self):
        self._stop.set()


class QuickScanWindow(QMainWindow):
    """Main application window."""

    # Signals for cross-thread communication (thread → GUI)
    _scope_update = Signal(float, float, float, object, int, float, float)
    _scope_error = Signal(str)

    def __init__(self, connect: bool = False):
        super().__init__()
        self.setWindowTitle("Quick Scan")
        self.resize(1200, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Canvas
        self._canvas = StageCanvas()
        layout.addWidget(self._canvas)

        # Toolbar
        self._build_toolbar()

        # Status bar
        self._build_status_bar()

        # Connect canvas signals
        self._canvas.cursor_moved.connect(self._on_cursor_moved)

        # Microscope poller
        self._poller: MicroscopePoller | None = None
        self._scope_update.connect(self._on_scope_update)
        self._scope_error.connect(self._on_scope_error)

        # Cached scope state for status bar
        self._scope_x = 0.0
        self._scope_y = 0.0
        self._scope_z = 0.0
        self._scope_mag: float | None = None
        self._connected = False

        # Status bar refresh timer (updates zoom display)
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(200)

        if connect:
            self._start_polling()

    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Go to Overview
        overview_action = QAction("Go to Overview", self)
        overview_action.setShortcut(QKeySequence("Home"))
        overview_action.setToolTip("Fit full stage in view (Home)")
        overview_action.triggered.connect(self._canvas.go_to_overview)
        toolbar.addAction(overview_action)

        # Go to Frame
        frame_action = QAction("Go to Frame", self)
        frame_action.setShortcut(QKeySequence("F"))
        frame_action.setToolTip("Center on current microscope viewport (F)")
        frame_action.triggered.connect(self._canvas.go_to_frame)
        toolbar.addAction(frame_action)

        toolbar.addSeparator()

        # Connect toggle
        self._connect_action = QAction("Connect", self)
        self._connect_action.setCheckable(True)
        self._connect_action.setToolTip("Connect to microscope for live position tracking")
        self._connect_action.triggered.connect(self._on_connect_toggled)
        toolbar.addAction(self._connect_action)

    def _build_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)

        # Cursor position
        self._cursor_label = QLabel("Cursor: —")
        self._cursor_label.setMinimumWidth(220)
        status.addWidget(self._cursor_label)

        # Scope position
        self._scope_label = QLabel("Stage: —")
        self._scope_label.setMinimumWidth(320)
        status.addWidget(self._scope_label)

        # Zoom level
        self._zoom_label = QLabel("Zoom: —")
        self._zoom_label.setMinimumWidth(120)
        status.addPermanentWidget(self._zoom_label)

    # ── Microscope polling ──────────────────────────────────────

    def _start_polling(self):
        if self._poller is not None:
            return
        self._scope_label.setText("Stage: connecting...")
        self._connect_action.setChecked(True)
        self._poller = MicroscopePoller(
            on_update=lambda *args: self._scope_update.emit(*args),
            on_error=lambda msg: self._scope_error.emit(msg),
        )
        self._poller.start()

    def _stop_polling(self):
        if self._poller is not None:
            self._poller.stop()
            self._poller = None
        self._connected = False
        self._canvas.hide_viewport()
        self._scope_label.setText("Stage: disconnected")
        self._connect_action.setChecked(False)

    @Slot(bool)
    def _on_connect_toggled(self, checked: bool):
        if checked:
            self._start_polling()
        else:
            self._stop_polling()

    @Slot(float, float, float, object, int, float, float)
    def _on_scope_update(self, x, y, z, mag, obj_pos, fov_w, fov_h):
        self._scope_x = x
        self._scope_y = y
        self._scope_z = z
        self._scope_mag = mag
        self._connected = True

        # Update viewport rect
        if fov_w > 0 and fov_h > 0:
            self._canvas.set_viewport(ViewportInfo(x, y, fov_w, fov_h))

        # Update status
        mag_str = f"{mag}x" if mag else f"pos {obj_pos}"
        self._scope_label.setText(f"Stage: X={x:.0f}  Y={y:.0f}  Z={z:.0f} µm  [{mag_str}]")

    @Slot(str)
    def _on_scope_error(self, msg: str):
        self._scope_label.setText(f"Stage: error — {msg}")
        self._connected = False
        self._connect_action.setChecked(False)
        self._canvas.hide_viewport()
        self._poller = None

    # ── Status bar updates ──────────────────────────────────────

    @Slot(float, float)
    def _on_cursor_moved(self, x_um: float, y_um: float):
        self._cursor_label.setText(f"Cursor: ({x_um:.0f}, {y_um:.0f}) µm")

    @Slot()
    def _refresh_status(self):
        um_px = self._canvas.um_per_px()
        if um_px >= 1:
            self._zoom_label.setText(f"Zoom: {um_px:.0f} µm/px")
        else:
            self._zoom_label.setText(f"Zoom: {um_px:.2f} µm/px")

    # ── Cleanup ─────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_polling()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Quick Scan — interactive stage viewer")
    parser.add_argument("--connect", action="store_true", help="Connect to microscope for live position tracking")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Quick Scan")

    window = QuickScanWindow(connect=args.connect)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
