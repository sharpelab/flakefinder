"""QGraphicsView-based stage canvas with pan/zoom and coordinate grid.

The scene uses stage coordinates directly (µm). The view applies a scale
transform for zoom and handles pan via mouse drag. Grid lines are drawn
as scene items; labels use ItemIgnoresTransformations so they stay a
fixed screen size regardless of zoom.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

# Stage limits (µm) from microscope_description.json
STAGE_X_MIN = 0.0
STAGE_X_MAX = 95_172.0
STAGE_Y_MIN = 0.0
STAGE_Y_MAX = 85_103.0

# Colors
BG_COLOR = QColor(30, 30, 30)
GRID_COLOR = QColor(255, 255, 255, 40)
GRID_LABEL_COLOR = QColor(255, 255, 255, 100)
VIEWPORT_COLOR = QColor(255, 80, 80, 180)
VIEWPORT_FILL = QColor(255, 80, 80, 20)
STAGE_BORDER_COLOR = QColor(100, 100, 100, 80)

# Zoom limits (µm per pixel)
ZOOM_MIN = 0.01  # ~150x equivalent
ZOOM_MAX = 200.0  # full stage in a small window
ZOOM_FACTOR = 1.15  # per scroll tick

# Grid spacing tiers (µm) — pick the one that gives ~5-20 grid lines on screen
GRID_SPACINGS = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]

GRID_LABEL_FONT = QFont("sans-serif", 9)


class ViewportInfo(NamedTuple):
    """Current microscope viewport: center position + FOV size."""

    x_um: float
    y_um: float
    fov_w_um: float
    fov_h_um: float


class StageCanvas(QGraphicsView):
    """Pannable/zoomable stage view using QGraphicsScene in µm coordinates.

    Signals:
        cursor_moved(x_um, y_um): Emitted when cursor moves over canvas.
    """

    cursor_moved = Signal(float, float)
    move_requested = Signal(float, float)  # double-click → stage move

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        # Scene rect must be much larger than the stage so that Qt's internal
        # scrollbar range is always nonzero — otherwise pan-via-scrollbar has
        # no room to move.  Items only exist in the stage area so the extra
        # extent costs nothing.
        pad = 500_000.0
        self._scene.setSceneRect(-pad, -pad, 2 * pad, 2 * pad)
        self.setScene(self._scene)

        # Rendering
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(BG_COLOR))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        # Disable scroll bars — we handle pan manually
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Interaction
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Pan state
        self._panning = False
        self._pan_start = QPointF()

        # Stage border
        border_pen = QPen(STAGE_BORDER_COLOR, 0)
        self._scene.addRect(
            STAGE_X_MIN,
            STAGE_Y_MIN,
            STAGE_X_MAX - STAGE_X_MIN,
            STAGE_Y_MAX - STAGE_Y_MIN,
            pen=border_pen,
        )

        # Grid items (regenerated on zoom changes)
        self._grid_items: list = []
        self._last_grid_spacing = 0.0

        # Viewport rectangle (hidden until set_viewport called)
        self._viewport_rect = QGraphicsRectItem()
        self._viewport_rect.setPen(QPen(VIEWPORT_COLOR, 0))
        self._viewport_rect.setBrush(QBrush(VIEWPORT_FILL))
        self._viewport_rect.setZValue(100)
        self._viewport_rect.setVisible(False)
        self._scene.addItem(self._viewport_rect)
        self._viewport_info: ViewportInfo | None = None

        # Camera overlay (live frame rendered at viewport position)
        self._camera_pixmap = QGraphicsPixmapItem()
        self._camera_pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._camera_pixmap.setZValue(50)  # above grid, below viewport rect
        self._camera_pixmap.setVisible(False)
        self._scene.addItem(self._camera_pixmap)

        # Start at overview zoom
        self.go_to_overview()

    # ── Public API ──────────────────────────────────────────────

    def go_to_overview(self):
        """Fit full stage in view with padding."""
        pad = 3000.0
        stage_rect = QRectF(
            STAGE_X_MIN - pad,
            STAGE_Y_MIN - pad,
            (STAGE_X_MAX - STAGE_X_MIN) + 2 * pad,
            (STAGE_Y_MAX - STAGE_Y_MIN) + 2 * pad,
        )
        self.fitInView(stage_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_grid()

    def go_to_frame(self):
        """Center on current viewport position, zoom to ~5× FOV."""
        if self._viewport_info is None:
            return
        vi = self._viewport_info
        view_extent = max(vi.fov_w_um, vi.fov_h_um) * 5
        rect = QRectF(
            vi.x_um - view_extent / 2,
            vi.y_um - view_extent / 2,
            view_extent,
            view_extent,
        )
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_grid()

    def set_viewport(self, info: ViewportInfo):
        """Update the microscope viewport rectangle."""
        self._viewport_info = info
        self._viewport_rect.setRect(
            info.x_um - info.fov_w_um / 2,
            info.y_um - info.fov_h_um / 2,
            info.fov_w_um,
            info.fov_h_um,
        )
        self._viewport_rect.setVisible(True)

    def hide_viewport(self):
        """Hide the viewport rectangle and camera overlay."""
        self._viewport_rect.setVisible(False)
        self._camera_pixmap.setVisible(False)
        self._viewport_info = None

    def update_camera_frame(self, qimg: QImage) -> None:
        """Place a camera frame on the canvas at the current viewport position."""
        if self._viewport_info is None:
            return
        vi = self._viewport_info
        pix = QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        self._camera_pixmap.setPixmap(pix)
        # Scale so the pixmap fills the viewport FOV in scene (µm) coords
        scale = vi.fov_w_um / pix.width()
        self._camera_pixmap.setScale(scale)
        self._camera_pixmap.setPos(vi.x_um - vi.fov_w_um / 2, vi.y_um - vi.fov_h_um / 2)
        self._camera_pixmap.setVisible(True)

    def um_per_px(self) -> float:
        """Current scale: stage µm per screen pixel."""
        t = self.transform()
        if t.m11() == 0:
            return 1.0
        return 1.0 / abs(t.m11())

    # ── Zoom ────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        angle = event.angleDelta().y()
        if angle == 0:
            return

        factor = ZOOM_FACTOR if angle > 0 else 1.0 / ZOOM_FACTOR

        new_um_per_px = self.um_per_px() / factor
        if new_um_per_px < ZOOM_MIN or new_um_per_px > ZOOM_MAX:
            return

        # Anchor zoom on cursor position
        anchor = self.mapToScene(event.position().toPoint())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(factor, factor)
        # Keep anchor point stationary under cursor
        new_anchor = self.mapToScene(event.position().toPoint())
        delta = new_anchor - anchor
        self.translate(delta.x(), delta.y())

        self._update_grid()

    # ── Pan ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            # Pan via scrollbars (works reliably with the internal transform)
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.cursor_moved.emit(scene_pos.x(), scene_pos.y())
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.move_requested.emit(scene_pos.x(), scene_pos.y())
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    # ── Grid ────────────────────────────────────────────────────

    def _pick_grid_spacing(self) -> float:
        """Choose grid spacing so ~5-20 lines are visible."""
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        visible_extent = max(visible.width(), visible.height())
        if visible_extent <= 0:
            return 10_000.0

        ideal_spacing = visible_extent / 10
        best = GRID_SPACINGS[0]
        for s in GRID_SPACINGS:
            if abs(s - ideal_spacing) < abs(best - ideal_spacing):
                best = s
        return float(best)

    def _update_grid(self):
        """Regenerate grid lines and labels when spacing tier changes."""
        spacing = self._pick_grid_spacing()
        if spacing == self._last_grid_spacing:
            return
        self._last_grid_spacing = spacing

        # Remove old grid items
        for item in self._grid_items:
            self._scene.removeItem(item)
        self._grid_items.clear()

        pen = QPen(GRID_COLOR, 0)

        # Format: show decimals only for sub-mm grids
        def fmt(um: float) -> str:
            mm = um / 1000
            if spacing >= 1000:
                return f"{mm:.0f}"
            return f"{mm:.1f}"

        # Vertical lines + X labels along top edge
        x = math.ceil(STAGE_X_MIN / spacing) * spacing
        while x <= STAGE_X_MAX:
            line = self._scene.addLine(x, STAGE_Y_MIN, x, STAGE_Y_MAX, pen)
            line.setZValue(-10)
            self._grid_items.append(line)

            label = QGraphicsSimpleTextItem(fmt(x))
            label.setFont(GRID_LABEL_FONT)
            label.setBrush(QBrush(GRID_LABEL_COLOR))
            label.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(x, STAGE_Y_MIN)
            label.setZValue(-9)
            self._scene.addItem(label)
            self._grid_items.append(label)
            x += spacing

        # Horizontal lines + Y labels along left edge
        y = math.ceil(STAGE_Y_MIN / spacing) * spacing
        while y <= STAGE_Y_MAX:
            line = self._scene.addLine(STAGE_X_MIN, y, STAGE_X_MAX, y, pen)
            line.setZValue(-10)
            self._grid_items.append(line)

            label = QGraphicsSimpleTextItem(fmt(y))
            label.setFont(GRID_LABEL_FONT)
            label.setBrush(QBrush(GRID_LABEL_COLOR))
            label.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(STAGE_X_MIN, y)
            label.setZValue(-9)
            self._scene.addItem(label)
            self._grid_items.append(label)
            y += spacing

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_grid()
