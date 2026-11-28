import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistogramPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(3.2, 2.0), dpi=95)
        self.figure.patch.set_facecolor("#2b2b2b")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.canvas)
        self.clear()

    def clear(self):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#cccccc", labelsize=7)
        self.ax.set_xlim(0, 255)
        self.ax.set_title("Histogram", color="#eeeeee", fontsize=9)
        self.figure.tight_layout(pad=0.6)
        self.canvas.draw_idle()

    def update_histogram(self, img: np.ndarray | None):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#cccccc", labelsize=7)

        if img is None:
            self.clear()
            return

        if img.ndim == 2:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
            self.ax.plot(hist, color="#e0e0e0", linewidth=1.2, label="Gray")
            self.ax.fill_between(range(256), hist, color="#aaaaaa", alpha=0.2)
        else:
            for idx, (col, lbl) in enumerate([("#3daee9", "B"), ("#2ecc71", "G"), ("#e74c3c", "R")]):
                hist = cv2.calcHist([img], [idx], None, [256], [0, 256]).ravel()
                self.ax.plot(hist, color=col, linewidth=1.1, label=lbl)
            self.ax.legend(loc="upper right", fontsize=7, facecolor="#2b2b2b", edgecolor="#555555", labelcolor="white")

        self.ax.set_xlim(0, 255)
        self.ax.set_xlabel("Mức xám (0-255)", color="#bbbbbb", fontsize=7)
        self.ax.set_ylabel("Số điểm ảnh", color="#bbbbbb", fontsize=7)
        self.figure.tight_layout(pad=0.6)
        self.canvas.draw_idle()