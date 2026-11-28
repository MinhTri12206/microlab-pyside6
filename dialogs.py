import csv

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


def _as_qimage(image):
    array = np.ascontiguousarray(image)
    if array.ndim == 2:
        h, w = array.shape
        return QImage(array.data, w, h, array.strides[0], QImage.Format_Grayscale8).copy()
    if array.ndim == 3 and array.shape[2] == 3:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
    raise ValueError("Ảnh cần có định dạng xám hoặc BGR 3 kênh.")


class ImageInfoDialog(QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Thông tin ảnh")
        layout = QFormLayout(self)
        for key, value in info.items():
            if key not in ("table_headers", "table_rows"):
                layout.addRow(str(key), QLabel(str(value)))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addRow(buttons)


class SplitChannelsDialog(QDialog):
    def __init__(self, blue, green, red, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tách kênh màu BGR")
        layout = QHBoxLayout(self)
        for name, image in (("B", blue), ("G", green), ("R", red)):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(QPixmap.fromImage(_as_qimage(image)).scaled(
                300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            label.setToolTip(f"Kênh {name}")
            layout.addWidget(label)


class ExportVideoDialog(QDialog):
    def __init__(self, video_source, operations, history_ops, parent=None):
        super().__init__(parent)
        self.video_source = video_source
        self.operations = operations
        self.history_ops = history_ops
        self.setWindowTitle("Xuất video")
        self.resize(420, 180)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.combo_operation = QComboBox()
        self.combo_operation.addItem("Không áp dụng thao tác", None)
        for op in operations:
            self.combo_operation.addItem(op.name, (op.func, {}))
        for label, func, params in history_ops:
            self.combo_operation.addItem(f"Lịch sử: {label}", (func, params))
        form.addRow("Xử lý mỗi khung hình:", self.combo_operation)

        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, max(0, video_source.total_frames - 1))
        self.spin_start.setValue(0)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, max(0, video_source.total_frames - 1))
        self.spin_end.setValue(max(0, video_source.total_frames - 1))
        form.addRow("Khung bắt đầu:", self.spin_start)
        form.addRow("Khung kết thúc:", self.spin_end)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.export_video)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def export_video(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu video", "ket_qua.mp4", "MP4 (*.mp4);;AVI (*.avi)"
        )
        if not path:
            return
        start, end = self.spin_start.value(), self.spin_end.value()
        if end < start:
            QMessageBox.warning(self, "Khoảng khung không hợp lệ", "Khung kết thúc phải lớn hơn hoặc bằng khung bắt đầu.")
            return

        selected = self.combo_operation.currentData()
        operation, params = selected if selected is not None else (None, {})
        first = self.video_source.read_frame(start)
        if first is None:
            QMessageBox.critical(self, "Lỗi video", "Không đọc được khung hình bắt đầu.")
            return
        height, width = first.shape[:2]
        extension = path.lower().rsplit(".", 1)[-1]
        codec = "XVID" if extension == "avi" else "mp4v"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), self.video_source.fps, (width, height), True)
        if not writer.isOpened():
            writer.release()
            QMessageBox.critical(self, "Lỗi xuất video", "Không thể tạo tệp video. Hãy thử lưu dạng AVI.")
            return

        try:
            for index in range(start, end + 1):
                frame = self.video_source.read_frame(index)
                if frame is None:
                    continue
                result = operation(frame, **params) if operation else frame
                if isinstance(result, tuple):
                    result = result[0]
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                elif result.shape[:2] != (height, width):
                    result = cv2.resize(result, (width, height))
                writer.write(np.ascontiguousarray(result, dtype=np.uint8))
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xuất video", str(exc))
            return
        finally:
            writer.release()
        QMessageBox.information(self, "Xuất video", f"Đã lưu video tại:\n{path}")
        self.accept()
