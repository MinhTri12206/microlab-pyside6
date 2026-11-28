import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QScrollArea, QWidget


def cv_to_qimage(img: np.ndarray) -> QImage:
    """Chuyển mảng OpenCV (BGR hoặc Grayscale uint8) sang QImage (RGB/Gray)."""
    if img is None:
        return QImage()
    if img.ndim == 2:
        h, w = img.shape
        buf = np.ascontiguousarray(img)
        return QImage(buf.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, ch = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf = np.ascontiguousarray(rgb)
    return QImage(buf.data, w, h, ch * w, QImage.Format_RGB888).copy()


class ImageCanvas(QWidget):
    pixelHovered = Signal(int, int, str)
    roiSelected = Signal(int, int, int, int)
    rulerMeasured = Signal(int, int, int, int)
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.display_img: np.ndarray | None = None
        self.before_img: np.ndarray | None = None
        self._display_qimg = QImage()
        self._before_qimg = QImage()

        self.scale: float = 1.0
        self.tool_mode: str = "view"  # "view", "roi", "ruler"
        self.side_by_side: bool = False
        self.hold_before: bool = False

        # Trạng thái ROI
        self._roi_dragging = False
        self._roi_start: tuple[int, int] | None = None
        self._roi_end: tuple[int, int] | None = None

        # Trạng thái Thước đo (M1)
        self._ruler_pts: list[tuple[int, int]] = []

    def set_images(self, display_img: np.ndarray | None, before_img: np.ndarray | None = None):
        self.display_img = display_img
        self.before_img = before_img if before_img is not None else display_img
        self._display_qimg = cv_to_qimage(self.display_img)
        self._before_qimg = cv_to_qimage(self.before_img)
        self._update_canvas_size()
        self.update()

    def set_scale(self, scale: float):
        self.scale = max(0.1, min(10.0, float(scale)))
        self._update_canvas_size()
        self.zoomChanged.emit(self.scale)
        self.update()

    def set_tool_mode(self, mode: str):
        self.tool_mode = mode
        if mode != "roi":
            self._roi_dragging = False
        if mode != "ruler":
            self._ruler_pts = []
        cursor = Qt.CrossCursor if mode in ("roi", "ruler") else Qt.ArrowCursor
        self.setCursor(cursor)
        self.update()

    def set_side_by_side(self, enabled: bool):
        self.side_by_side = enabled
        self._update_canvas_size()
        self.update()

    def set_hold_before(self, active: bool):
        self.hold_before = active
        self.update()

    def clear_overlays(self):
        self._roi_start = None
        self._roi_end = None
        self._ruler_pts = []
        self.update()

    def get_roi_rect(self) -> tuple[int, int, int, int] | None:
        if self._roi_start and self._roi_end:
            x1, y1 = self._roi_start
            x2, y2 = self._roi_end
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return None

    def _update_canvas_size(self):
        if self.display_img is None:
            self.resize(400, 300)
            return
        h, w = self.display_img.shape[:2]
        sw, sh = int(w * self.scale), int(h * self.scale)
        if self.side_by_side and self.before_img is not None:
            self.resize(sw * 2 + 12, sh)
        else:
            self.resize(sw, sh)

    def _widget_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        if self.display_img is None:
            return None
        h, w = self.display_img.shape[:2]
        ix = int(pos.x() / self.scale)
        iy = int(pos.y() / self.scale)
        # Nếu đang ở chế độ song song và trỏ chuột bên ảnh phải
        if self.side_by_side and pos.x() > int(w * self.scale) + 12:
            ix = int((pos.x() - int(w * self.scale) - 12) / self.scale)
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(36, 36, 36))

        if self.display_img is None or self._display_qimg.isNull():
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.rect(), Qt.AlignCenter, "Mở ảnh hoặc video để bắt đầu")
            return

        h, w = self.display_img.shape[:2]
        sw, sh = int(w * self.scale), int(h * self.scale)

        if self.side_by_side and not self._before_qimg.isNull():
            left_rect = QRect(0, 0, sw, sh)
            right_rect = QRect(sw + 12, 0, sw, sh)
            painter.drawImage(left_rect, self._before_qimg)
            painter.drawImage(right_rect, self._display_qimg)
            painter.fillRect(sw, 0, 12, sh, QColor(65, 65, 65))
            self._draw_badge(painter, 8, 8, "Ảnh trước")
            self._draw_badge(painter, sw + 20, 8, "Ảnh sau")
        else:
            target = QRect(0, 0, sw, sh)
            qimg = self._before_qimg if (self.hold_before and not self._before_qimg.isNull()) else self._display_qimg
            painter.drawImage(target, qimg)
            if self.hold_before:
                self._draw_badge(painter, 8, 8, "Đang xem ảnh trước")

            # Vẽ khung chọn ROI
            roi = self.get_roi_rect()
            if roi is not None:
                x1, y1, x2, y2 = roi
                pen = QPen(QColor(0, 220, 255), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(
                    int(x1 * self.scale),
                    int(y1 * self.scale),
                    int((x2 - x1) * self.scale),
                    int((y2 - y1) * self.scale),
                )

            # Vẽ thước đo đoạn AB
            if self._ruler_pts:
                pen = QPen(QColor(255, 220, 0), 2)
                painter.setPen(pen)
                for idx, (px, py) in enumerate(self._ruler_pts):
                    sx, sy = int(px * self.scale), int(py * self.scale)
                    painter.drawEllipse(QPoint(sx, sy), 4, 4)
                    painter.drawText(sx + 6, sy - 6, "A" if idx == 0 else "B")
                if len(self._ruler_pts) == 2:
                    p1, p2 = self._ruler_pts
                    painter.drawLine(
                        int(p1[0] * self.scale), int(p1[1] * self.scale),
                        int(p2[0] * self.scale), int(p2[1] * self.scale),
                    )

    @staticmethod
    def _draw_badge(painter: QPainter, x: int, y: int, text: str):
        painter.save()
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        rect = QRect(x, y, 130, 22)
        painter.fillRect(rect, QColor(0, 0, 0, 165))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()

    def mouseMoveEvent(self, event):
        pt = self._widget_to_image(event.position().toPoint())
        if pt is not None and self.display_img is not None:
            ix, iy = pt
            val = self.display_img[iy, ix]
            if self.display_img.ndim == 2:
                val_str = f"Gray = {int(val)}"
            else:
                val_str = f"(B, G, R) = ({int(val[0])}, {int(val[1])}, {int(val[2])})"
            self.pixelHovered.emit(ix, iy, val_str)

            if self.tool_mode == "roi" and self._roi_dragging:
                self._roi_end = (ix, iy)
                self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pt = self._widget_to_image(event.position().toPoint())
        if pt is None:
            return

        if self.tool_mode == "roi":
            self._roi_dragging = True
            self._roi_start = pt
            self._roi_end = pt
            self.update()
        elif self.tool_mode == "ruler":
            if len(self._ruler_pts) >= 2:
                self._ruler_pts = []
            self._ruler_pts.append(pt)
            if len(self._ruler_pts) == 2:
                (x1, y1), (x2, y2) = self._ruler_pts
                self.rulerMeasured.emit(x1, y1, x2, y2)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.tool_mode == "roi" and self._roi_dragging:
            self._roi_dragging = False
            pt = self._widget_to_image(event.position().toPoint())
            if pt is not None:
                self._roi_end = pt
            roi = self.get_roi_rect()
            if roi is not None:
                self.roiSelected.emit(*roi)
            self.update()


class ImageViewer(QScrollArea):
    """Khung cuộn chứa ImageCanvas, hỗ trợ phóng to/thu nhỏ bằng con lăn chuột."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = ImageCanvas(self)
        self.setWidget(self.canvas)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QScrollArea { background-color: #242424; border: none; }")

    def wheelEvent(self, event):
        if self.canvas.display_img is None:
            return super().wheelEvent(event)
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.canvas.set_scale(self.canvas.scale * factor)
        event.accept()

    def zoom_in(self):
        self.canvas.set_scale(self.canvas.scale * 1.25)

    def zoom_out(self):
        self.canvas.set_scale(self.canvas.scale / 1.25)

    def zoom_reset(self):
        self.canvas.set_scale(1.0)