from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox


class VideoBar(QWidget):
    playToggled = Signal(bool)
    prevFrameRequested = Signal()
    nextFrameRequested = Signal()
    seekRequested = Signal(int)
    extractFrameRequested = Signal()
    modeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)

        self.btn_play = QPushButton("Phát")
        self.btn_prev = QPushButton("◀ Trước")
        self.btn_next = QPushButton("Sau ▶")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.lbl_info = QLabel("Khung 0 / 0 | 0.0 FPS")

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Xem bình thường", "normal")
        self.combo_mode.addItem("Phát hiện thay đổi (absdiff)", "diff")
        self.combo_mode.addItem("Theo vết chuyển động (LK)", "track")

        self.btn_extract = QPushButton("Trích khung → Chỉnh ảnh")

        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.combo_mode)
        layout.addWidget(self.btn_extract)

        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_prev.clicked.connect(self.prevFrameRequested.emit)
        self.btn_next.clicked.connect(self.nextFrameRequested.emit)
        self.slider.sliderMoved.connect(self.seekRequested.emit)
        self.btn_extract.clicked.connect(self.extractFrameRequested.emit)
        self.combo_mode.currentIndexChanged.connect(
            lambda: self.modeChanged.emit(self.combo_mode.currentData())
        )
        self.setVisible(False)

    def _toggle_play(self):
        self._playing = not self._playing
        self.btn_play.setText("Dừng" if self._playing else "Phát")
        self.playToggled.emit(self._playing)

    def set_playing(self, playing: bool):
        self._playing = playing
        self.btn_play.setText("Dừng" if self._playing else "Phát")

    def configure(self, total_frames: int, fps: float):
        self.slider.setRange(0, max(0, total_frames - 1))
        self.slider.setValue(0)
        self.update_status(0, total_frames, fps)
        self.setVisible(True)

    def update_status(self, current_idx: int, total_frames: int, fps: float):
        self.slider.blockSignals(True)
        self.slider.setValue(current_idx)
        self.slider.blockSignals(False)
        self.lbl_info.setText(f"Khung {current_idx + 1} / {total_frames} | {fps:.1f} FPS")