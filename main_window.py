import csv
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QToolBar,
    QFileDialog, QMessageBox, QLabel, QListWidget, QPushButton,
    QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget,
)
from core.registry import OperationRegistry, OperationSpec
from core.m1_pixel import (
    read_image_unicode, save_image_unicode, get_image_info,
    split_bgr_channels, convert_to_grayscale, crop_roi,
    measure_distances_tc, create_sample_blood_smear,
)
from core.m8_video import VideoSource, compute_frame_difference, MotionTrackerLK
from gui.image_viewer import ImageViewer
from gui.histogram_panel import HistogramPanel
from gui.param_panel import ParamPanel
from gui.video_bar import VideoBar
from gui.dialogs import ImageInfoDialog, SplitChannelsDialog, ExportVideoDialog


class MainWindow(QMainWindow):
    def __init__(self, registry: OperationRegistry):
        super().__init__()
        self.registry = registry
        self.setWindowTitle("MicroLab — Phần mềm xử lý ảnh hiển vi y sinh")
        self.resize(1380, 840)

        # Trạng thái ảnh & Lịch sử Undo/Redo (>= 10 bước, đặt tối đa 25 bước)
        self.current_file_path: str | None = None
        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.preview_image: np.ndarray | None = None
        self.active_op: OperationSpec | None = None

        # Mỗi phần tử lịch sử: (nhãn, bản sao ảnh, (OperationSpec, params) hoặc None)
        self.history: list[tuple[str, np.ndarray, tuple[OperationSpec, dict] | None]] = []
        self.history_index: int = -1
        self.max_history: int = 25

        # Dữ liệu xuất CSV ở Bảng kết quả (dành cho M6/SV3 dùng lại luôn)
        self._last_table_headers: list[str] = []
        self._last_table_rows: list[list] = []

        # Trạng thái Video (M8)
        self.video_source = VideoSource()
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._on_video_tick)
        self.video_mode = "normal"  # "normal", "diff", "track"
        self.video_prev_frame: np.ndarray | None = None
        self.video_curr_raw_frame: np.ndarray | None = None
        self.tracker_lk = MotionTrackerLK(max_corners=100)

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()

        # Nạp sẵn ảnh tiêu bản mẫu để phần mềm chạy thử được ngay khi bàn giao
        self._load_new_image(create_sample_blood_smear(), "Ảnh tiêu bản mẫu (BCCD giả lập)")

    # ------------------------------------------------------------------
    # DỰNG BỐ CỤC GIAO DIỆN (G1)
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Vùng giữa: Thanh so sánh nhanh + Vùng hiển thị ảnh + Thanh Video
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        compare_row = QHBoxLayout()
        self.btn_side_by_side = QPushButton("So sánh trước/sau cạnh nhau")
        self.btn_side_by_side.setCheckable(True)
        self.btn_hold_before = QPushButton("Giữ chuột để xem ảnh trước")
        self.btn_reset_orig = QPushButton("Khôi phục ảnh gốc")
        compare_row.addWidget(self.btn_side_by_side)
        compare_row.addWidget(self.btn_hold_before)
        compare_row.addStretch()
        compare_row.addWidget(self.btn_reset_orig)
        center_layout.addLayout(compare_row)

        self.viewer = ImageViewer()
        center_layout.addWidget(self.viewer, stretch=1)

        self.video_bar = VideoBar()
        center_layout.addWidget(self.video_bar)
        splitter.addWidget(center_widget)

        # Cột bên phải: Histogram, Tham số, Lịch sử, Bảng kết quả
        right_panel = QWidget()
        right_panel.setMinimumWidth(330)
        right_panel.setMaximumWidth(420)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(6)

        # 1. Histogram (G4)
        grp_hist = QGroupBox("Histogram")
        v_hist = QVBoxLayout(grp_hist)
        v_hist.setContentsMargins(4, 4, 4, 4)
        self.histogram_panel = HistogramPanel()
        self.histogram_panel.setFixedHeight(175)
        v_hist.addWidget(self.histogram_panel)
        right_layout.addWidget(grp_hist)

        # 2. Tham số (G2)
        grp_param = QGroupBox("Tham số & Xem trước")
        v_param = QVBoxLayout(grp_param)
        v_param.setContentsMargins(4, 4, 4, 4)
        self.param_panel = ParamPanel()
        v_param.addWidget(self.param_panel)
        right_layout.addWidget(grp_param, stretch=2)

        # 3. Lịch sử thao tác (G3)
        grp_hist_ops = QGroupBox("Lịch sử thao tác")
        v_h = QVBoxLayout(grp_hist_ops)
        v_h.setContentsMargins(4, 4, 4, 4)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(135)
        v_h.addWidget(self.history_list)
        h_btns = QHBoxLayout()
        self.btn_undo = QPushButton("Hoàn tác (Ctrl+Z)")
        self.btn_redo = QPushButton("Làm lại (Ctrl+Y)")
        h_btns.addWidget(self.btn_undo)
        h_btns.addWidget(self.btn_redo)
        v_h.addLayout(h_btns)
        right_layout.addWidget(grp_hist_ops)

        # 4. Bảng kết quả (M1, M4, M6, M7...)
        grp_res = QGroupBox("Bảng kết quả")
        v_res = QVBoxLayout(grp_res)
        v_res.setContentsMargins(4, 4, 4, 4)
        self.result_tabs = QTabWidget()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_table = QTableWidget()
        self.result_tabs.addTab(self.result_text, "Thông số")
        self.result_tabs.addTab(self.result_table, "Bảng số liệu")
        v_res.addWidget(self.result_tabs)
        self.btn_export_csv = QPushButton("Xuất CSV")
        self.btn_export_csv.setEnabled(False)
        v_res.addWidget(self.btn_export_csv)
        right_layout.addWidget(grp_res, stretch=2)

        splitter.addWidget(right_panel)
        splitter.setSizes([980, 380])

        # Thanh trạng thái (G1)
        self.lbl_coord = QLabel("x=—, y=— | Giá trị: —")
        self.lbl_img_meta = QLabel("Chưa có ảnh")
        self.lbl_zoom = QLabel("Phóng: 100%")
        self.statusBar().addWidget(self.lbl_coord, stretch=1)
        self.statusBar().addPermanentWidget(self.lbl_img_meta)
        self.statusBar().addPermanentWidget(self.lbl_zoom)

    def _build_menus(self):
        mb = self.menuBar()

        # 1. Tệp (G4)
        m_file = mb.addMenu("Tệp")
        self._add_action(m_file, "Mở ảnh...", self.open_image_dialog, QKeySequence.Open)
        self._add_action(m_file, "Nạp ảnh mẫu", lambda: self._load_new_image(create_sample_blood_smear(), "Ảnh mẫu"))
        m_file.addSeparator()
        self._add_action(m_file, "Lưu", self.save_image, QKeySequence.Save)
        self._add_action(m_file, "Lưu thành (PNG/JPG)...", self.save_image_as, QKeySequence.SaveAs)
        m_file.addSeparator()
        self._add_action(m_file, "Mở video...", self.open_video_dialog)
        self._add_action(m_file, "Mở chuỗi ảnh hiển vi (thư mục TIF)...", self.open_sequence_dialog)
        m_file.addSeparator()
        self._add_action(m_file, "Hoàn tác", self.undo, QKeySequence.Undo)
        self._add_action(m_file, "Làm lại", self.redo, QKeySequence.Redo)
        self._add_action(m_file, "Khôi phục ảnh gốc", self.restore_original)
        m_file.addSeparator()
        self._add_action(m_file, "Thoát", self.close, QKeySequence.Quit)

        # 2. Ảnh (M1)
        m_img = mb.addMenu("Ảnh")
        self._add_action(m_img, "Thông tin ảnh", self.show_image_info)
        self._add_action(m_img, "Tách kênh màu (B, G, R)", self.show_split_channels)
        self._add_action(m_img, "Chuyển ảnh xám", self.apply_grayscale)
        self._add_action(m_img, "Cắt theo vùng chọn (ROI)", self.apply_crop_roi)
        self._populate_registered_menu(m_img, "Ảnh")

        # 3–7. Các menu dành cho SV2 (M2–M5) & SV3 (M6–M7) tự động nạp từ Registry
        for menu_title in ["Điều chỉnh", "Bộ lọc", "Màu", "Hình thái", "Biên"]:
            menu = mb.addMenu(menu_title)
            self._populate_registered_menu(menu, menu_title)

        # 8. Video (M8)
        m_vid = mb.addMenu("Video")
        self._add_action(m_vid, "Mở chuỗi ảnh (thư mục D2)...", self.open_sequence_dialog)
        self._add_action(m_vid, "Mở tệp video (.mp4/.avi)...", self.open_video_dialog)
        m_vid.addSeparator()
        self._add_action(m_vid, "Phát / Dừng", lambda: self.video_bar.btn_play.click())
        self._add_action(m_vid, "Trích khung hiện tại để chỉnh ảnh", self.extract_current_frame)
        self._add_action(m_vid, "Áp lệnh cho toàn bộ video & Xuất...", self.open_export_video_dialog)
        m_vid.addSeparator()
        self._add_action(m_vid, "Chế độ: Xem bình thường", lambda: self._set_video_mode_ui("normal"))
        self._add_action(m_vid, "Chế độ: Phát hiện thay đổi (absdiff)", lambda: self._set_video_mode_ui("diff"))
        self._add_action(m_vid, "Chế độ: Theo vết chuyển động (Lucas-Kanade)", lambda: self._set_video_mode_ui("track"))

        # 9. AI (SV3)
        m_ai = mb.addMenu("AI")
        self._populate_registered_menu(m_ai, "AI")

        # 10. Trợ giúp
        m_help = mb.addMenu("Trợ giúp")
        self._add_action(m_help, "Giới thiệu MicroLab", self.show_about)

    def _populate_registered_menu(self, menu, menu_name: str):
        ops = self.registry.by_menu(menu_name)
        if not ops and menu_name != "Ảnh":
            placeholder = QAction("(Chưa gắn chức năng — xem hướng dẫn đăng ký)", self)
            placeholder.setEnabled(False)
            menu.addAction(placeholder)
            return
        for spec in ops:
            self._add_action(menu, spec.name, lambda _=False, s=spec: self.activate_operation(s))

    def _add_action(self, menu, text: str, slot, shortcut=None):
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _build_toolbar(self):
        tb = QToolBar("Hộp công cụ")
        tb.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, tb)

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)

        self.act_view = QAction("Xem", self, checkable=True)
        self.act_view.setChecked(True)
        self.act_roi = QAction("Chọn vùng", self, checkable=True)
        self.act_ruler = QAction("Thước đo", self, checkable=True)

        for act, mode in [(self.act_view, "view"), (self.act_roi, "roi"), (self.act_ruler, "ruler")]:
            tool_group.addAction(act)
            act.triggered.connect(lambda _=False, m=mode: self.viewer.canvas.set_tool_mode(m))
            tb.addAction(act)

        tb.addSeparator()
        tb.addAction(self._make_tb_action("Phóng +", self.viewer.zoom_in))
        tb.addAction(self._make_tb_action("Thu -", self.viewer.zoom_out))
        tb.addAction(self._make_tb_action("100%", self.viewer.zoom_reset))
        tb.addSeparator()
        tb.addAction(self._make_tb_action("Cắt ROI", self.apply_crop_roi))

    def _make_tb_action(self, label: str, slot):
        act = QAction(label, self)
        act.triggered.connect(slot)
        return act

    def _connect_signals(self):
        c = self.viewer.canvas
        c.pixelHovered.connect(self._on_pixel_hover)
        c.zoomChanged.connect(lambda s: self.lbl_zoom.setText(f"Phóng: {int(round(s * 100))}%"))
        c.rulerMeasured.connect(self._on_ruler_measured)
        c.roiSelected.connect(self._on_roi_selected)

        self.param_panel.previewRequested.connect(self._on_preview_operation)
        self.param_panel.applyRequested.connect(self._on_apply_operation)
        self.param_panel.cancelRequested.connect(self._on_cancel_operation)

        self.btn_side_by_side.toggled.connect(self.viewer.canvas.set_side_by_side)
        self.btn_hold_before.pressed.connect(lambda: self.viewer.canvas.set_hold_before(True))
        self.btn_hold_before.released.connect(lambda: self.viewer.canvas.set_hold_before(False))
        self.btn_reset_orig.clicked.connect(self.restore_original)

        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.history_list.currentRowChanged.connect(self._on_history_row_clicked)
        self.btn_export_csv.clicked.connect(self.export_results_csv)

        self.video_bar.playToggled.connect(self._on_video_play_toggled)
        self.video_bar.prevFrameRequested.connect(lambda: self._step_video(-1))
        self.video_bar.nextFrameRequested.connect(lambda: self._step_video(1))
        self.video_bar.seekRequested.connect(self._seek_video)
        self.video_bar.extractFrameRequested.connect(self.extract_current_frame)
        self.video_bar.modeChanged.connect(self._on_video_mode_changed)

    # ------------------------------------------------------------------
    # QUẢN LÝ ẢNH, LỊCH SỬ & UNDO/REDO (G3, G4)
    # ------------------------------------------------------------------
    def _load_new_image(self, img: np.ndarray, label: str, path: str | None = None):
        self._stop_video_if_playing()
        self.current_file_path = path
        self.original_image = img.copy()
        self.current_image = img.copy()
        self.preview_image = None
        self.active_op = None
        self.param_panel.clear_operation()
        self.viewer.canvas.clear_overlays()

        self.history = [(label, img.copy(), None)]
        self.history_index = 0
        self._refresh_history_ui()
        self._update_display(self.current_image, before_img=self.original_image)
        self.set_result_info({"Trạng thái": f"Đã nạp: {label}"})

    def _push_history(self, label: str, new_img: np.ndarray, op_record=None):
        # Xóa nhánh redo cũ nếu đang ở giữa lịch sử
        self.history = self.history[: self.history_index + 1]
        self.history.append((label, new_img.copy(), op_record))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
        self.current_image = new_img.copy()
        self.preview_image = None
        self._refresh_history_ui()

        before = self.history[max(0, self.history_index - 1)][1]
        self._update_display(self.current_image, before_img=before)

    def _refresh_history_ui(self):
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for idx, (lbl, _, _) in enumerate(self.history):
            self.history_list.addItem(f"{idx}. {lbl}")
        if 0 <= self.history_index < self.history_list.count():
            self.history_list.setCurrentRow(self.history_index)
        self.history_list.blockSignals(False)

        self.btn_undo.setEnabled(self.history_index > 0)
        self.btn_redo.setEnabled(self.history_index < len(self.history) - 1)

    def _on_history_row_clicked(self, row: int):
        if 0 <= row < len(self.history) and row != self.history_index:
            self.history_index = row
            self.current_image = self.history[row][1].copy()
            self.preview_image = None
            self.active_op = None
            self.param_panel.clear_operation()
            before = self.history[max(0, row - 1)][1]
            self._update_display(self.current_image, before_img=before)
            self._refresh_history_ui()

    def undo(self):
        if self.history_index > 0:
            self._on_history_row_clicked(self.history_index - 1)

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self._on_history_row_clicked(self.history_index + 1)

    def restore_original(self):
        if self.original_image is None:
            return
        self.active_op = None
        self.param_panel.clear_operation()
        self.viewer.canvas.clear_overlays()
        self._push_history("Khôi phục ảnh gốc", self.original_image.copy())

    def _update_display(self, img: np.ndarray, before_img: np.ndarray | None = None):
        self.viewer.canvas.set_images(img, before_img=before_img if before_img is not None else self.current_image)
        self.histogram_panel.update_histogram(img)
        info = get_image_info(img)
        self.lbl_img_meta.setText(f"{info['width']}×{info['height']}, {info['channels']} kênh ({info['dtype']})")

    def _on_pixel_hover(self, x: int, y: int, val_str: str):
        self.lbl_coord.setText(f"x={x}, y={y} | {val_str}")

    # ------------------------------------------------------------------
    # MỞ / LƯU TỆP (G4)
    # ------------------------------------------------------------------
    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh", "", "Ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Tất cả (*.*)"
        )
        if not path:
            return
        try:
            img = read_image_unicode(path)
            self._load_new_image(img, f"Mở ảnh ({path.split('/')[-1]})", path=path)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi mở ảnh", str(exc))

    def save_image(self):
        if self.current_image is None:
            return
        if not self.current_file_path:
            self.save_image_as()
            return
        try:
            save_image_unicode(self.current_file_path, self.current_image)
            self.statusBar().showMessage(f"Đã lưu: {self.current_file_path}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi lưu ảnh", str(exc))

    def save_image_as(self):
        if self.current_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu ảnh thành", "ket_qua.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        try:
            save_image_unicode(path, self.current_image)
            self.current_file_path = path
            self.statusBar().showMessage(f"Đã lưu: {path}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi lưu ảnh", str(exc))

    # ------------------------------------------------------------------
    # CƠ CHẾ XỬ LÝ THỐNG NHẤT (G2 — DÙNG CHUNG CHO SV1, SV2, SV3)
    # ------------------------------------------------------------------
    def activate_operation(self, spec: OperationSpec):
        if self.current_image is None:
            QMessageBox.warning(self, "Chưa có ảnh", "Hãy mở một ảnh trước khi xử lý.")
            return
        self.active_op = spec
        if not spec.params and not spec.extra_buttons:
            # Lệnh không có tham số: áp dụng ngay
            self._on_apply_operation({})
        else:
            self.param_panel.load_operation(spec)

    def _run_spec(self, spec: OperationSpec, params: dict) -> tuple[np.ndarray, dict]:
        res = spec.func(self.current_image.copy(), **params)
        if isinstance(res, tuple):
            return res[0], (res[1] if isinstance(res[1], dict) else {})
        return res, {}

    def _on_preview_operation(self, params: dict):
        if self.active_op is None or self.current_image is None:
            return
        try:
            out_img, info = self._run_spec(self.active_op, params)
            self.preview_image = out_img
            self._update_display(out_img, before_img=self.current_image)
            if info:
                self.set_result_info(info)
        except Exception as exc:
            self.statusBar().showMessage(f"Lỗi xem trước: {exc}", 4000)

    def _on_apply_operation(self, params: dict):
        if self.active_op is None or self.current_image is None:
            return
        try:
            out_img, info = self._run_spec(self.active_op, params)
            spec = self.active_op
            self.active_op = None
            self.param_panel.clear_operation()
            self._push_history(spec.name, out_img, op_record=(spec, dict(params)))
            if info:
                self.set_result_info(info)
        except Exception as exc:
            QMessageBox.critical(self, f"Lỗi khi áp dụng {self.active_op.name}", str(exc))

    def _on_cancel_operation(self):
        self.active_op = None
        self.preview_image = None
        self.param_panel.clear_operation()
        if self.current_image is not None:
            before = self.history[max(0, self.history_index - 1)][1]
            self._update_display(self.current_image, before_img=before)

    # ------------------------------------------------------------------
    # BẢNG KẾT QUẢ & XUẤT CSV (DÙNG CHUNG M1, M4, M6, M7)
    # ------------------------------------------------------------------
    def set_result_info(self, info: dict):
        lines = []
        for k, v in info.items():
            if k in ("table_headers", "table_rows"):
                continue
            lines.append(f"• {k}: {v}")
        if lines:
            self.result_text.setPlainText("\n".join(lines))

        if "table_headers" in info and "table_rows" in info:
            headers = info["table_headers"]
            rows = info["table_rows"]
            self._last_table_headers = headers
            self._last_table_rows = rows
            self.result_table.setColumnCount(len(headers))
            self.result_table.setHorizontalHeaderLabels([str(h) for h in headers])
            self.result_table.setRowCount(len(rows))
            for r_idx, row in enumerate(rows):
                for c_idx, cell in enumerate(row):
                    self.result_table.setItem(r_idx, c_idx, QTableWidgetItem(str(cell)))
            self.btn_export_csv.setEnabled(True)
            self.result_tabs.setCurrentIndex(1)
        else:
            self.result_tabs.setCurrentIndex(0)

    def export_results_csv(self):
        if not self._last_table_rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Xuất bảng kết quả CSV", "ket_qua.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(self._last_table_headers)
            writer.writerows(self._last_table_rows)
        self.statusBar().showMessage(f"Đã xuất CSV: {path}", 3000)

    # ------------------------------------------------------------------
    # CHỨC NĂNG M1 (MICROLAB)
    # ------------------------------------------------------------------
    def show_image_info(self):
        if self.current_image is None:
            return
        info = get_image_info(self.current_image)
        ImageInfoDialog(info, self).exec()

    def show_split_channels(self):
        if self.current_image is None:
            return
        try:
            b, g, r = split_bgr_channels(self.current_image)
            SplitChannelsDialog(b, g, r, self).exec()
        except Exception as exc:
            QMessageBox.warning(self, "Tách kênh màu", str(exc))

    def apply_grayscale(self):
        if self.current_image is None:
            return
        gray = convert_to_grayscale(self.current_image)
        self._push_history("Chuyển ảnh xám", gray)

    def _on_roi_selected(self, x1: int, y1: int, x2: int, y2: int):
        self.set_result_info({
            "Vùng chọn (ROI)": f"({x1}, {y1}) → ({x2}, {y2})",
            "Kích thước ROI": f"{x2 - x1} × {y2 - y1} px",
            "Gợi ý": "Bấm menu Ảnh > Cắt theo vùng chọn hoặc nút Cắt ROI",
        })

    def apply_crop_roi(self):
        if self.current_image is None:
            return
        roi = self.viewer.canvas.get_roi_rect()
        if roi is None:
            QMessageBox.information(self, "Chưa chọn vùng", "Hãy chọn công cụ 'Chọn vùng' ở hộp công cụ bên trái và kéo chuột trên ảnh.")
            return
        try:
            cropped = crop_roi(self.current_image, *roi)
            self.viewer.canvas.clear_overlays()
            self._push_history(f"Cắt ROI {roi}", cropped)
        except Exception as exc:
            QMessageBox.warning(self, "Lỗi cắt ảnh", str(exc))

    def _on_ruler_measured(self, x1: int, y1: int, x2: int, y2: int):
        res = measure_distances_tc(x1, y1, x2, y2)
        self.set_result_info({
            "Thước đo [TC]": f"A{res['A']} → B{res['B']}",
            "Khoảng cách Euclid": f"{res['Euclid']} px",
            "Khoảng cách D4 (khối phố)": f"{res['D4']} px",
            "Khoảng cách D8 (bàn cờ)": f"{res['D8']} px",
        })

    # ------------------------------------------------------------------
    # CHỨC NĂNG M8 (VIDEO & CHUỖI ẢNH HIỂN VI)
    # ------------------------------------------------------------------
    def open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở tệp video", "", "Video (*.mp4 *.avi *.mov *.mkv);;Tất cả (*.*)"
        )
        if not path:
            return
        try:
            self._stop_video_if_playing()
            self.video_source.open_video_file(path)
            self._init_video_ui()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi mở video", str(exc))

    def open_sequence_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chuỗi ảnh (ví dụ Fluo-N2DH-GOWT1/01)")
        if not folder:
            return
        try:
            self._stop_video_if_playing()
            self.video_source.open_image_sequence(folder, fps=10.0)
            self._init_video_ui()
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi mở chuỗi ảnh", str(exc))

    def _init_video_ui(self):
        self.video_prev_frame = None
        self.tracker_lk.reset()
        self.video_bar.configure(self.video_source.total_frames, self.video_source.fps)
        self._display_video_frame(0)

    def _display_video_frame(self, index: int):
        if not self.video_source.is_open():
            return
        frame = self.video_source.read_frame(index)
        if frame is None:
            return
        self.video_curr_raw_frame = frame.copy()

        if self.video_mode == "diff":
            if self.video_prev_frame is not None:
                shown = compute_frame_difference(self.video_prev_frame, frame, threshold=25)
            else:
                shown = np.zeros(frame.shape[:2], dtype=np.uint8)
        elif self.video_mode == "track":
            shown = self.tracker_lk.update(frame)
        else:
            shown = frame

        self.video_prev_frame = frame.copy()
        self._update_display(shown, before_img=frame)
        self.video_bar.update_status(self.video_source.current_index, self.video_source.total_frames, self.video_source.fps)

    def _on_video_play_toggled(self, playing: bool):
        if not self.video_source.is_open():
            self.video_bar.set_playing(False)
            return
        if playing:
            interval = max(15, int(1000 / max(1.0, self.video_source.fps)))
            self.video_timer.start(interval)
        else:
            self.video_timer.stop()

    def _stop_video_if_playing(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
        self.video_bar.set_playing(False)

    def _on_video_tick(self):
        nxt = self.video_source.current_index + 1
        if nxt >= self.video_source.total_frames:
            self._stop_video_if_playing()
            return
        self._display_video_frame(nxt)

    def _step_video(self, delta: int):
        if not self.video_source.is_open():
            return
        self._stop_video_if_playing()
        target = max(0, min(self.video_source.total_frames - 1, self.video_source.current_index + delta))
        if delta < 0:
            self.tracker_lk.reset()
        self._display_video_frame(target)

    def _seek_video(self, idx: int):
        self._stop_video_if_playing()
        self.video_prev_frame = None
        self.tracker_lk.reset()
        self._display_video_frame(idx)

    def _set_video_mode_ui(self, mode: str):
        idx = self.video_bar.combo_mode.findData(mode)
        if idx >= 0:
            self.video_bar.combo_mode.setCurrentIndex(idx)

    def _on_video_mode_changed(self, mode: str):
        self.video_mode = mode
        self.video_prev_frame = None
        self.tracker_lk.reset()
        if self.video_source.is_open():
            self._display_video_frame(max(0, self.video_source.current_index))

    def extract_current_frame(self):
        if self.video_curr_raw_frame is None:
            QMessageBox.information(self, "Chưa có khung hình", "Hãy mở video hoặc chuỗi ảnh trước.")
            return
        idx = self.video_source.current_index + 1
        self._load_new_image(self.video_curr_raw_frame.copy(), f"Trích khung #{idx}")

    def open_export_video_dialog(self):
        if not self.video_source.is_open():
            QMessageBox.warning(self, "Chưa mở video", "Hãy mở video hoặc chuỗi ảnh hiển vi trước khi xuất.")
            return
        self._stop_video_if_playing()
        history_ops = [
            (lbl, rec[0], rec[1]) for (lbl, _, rec) in self.history if rec is not None
        ]
        dlg = ExportVideoDialog(
            self.video_source,
            self.registry.video_compatible_ops(),
            history_ops,
            self,
        )
        dlg.exec()

    def show_about(self):
        QMessageBox.information(
            self,
            "MicroLab",
            "MicroLab — Phần mềm xử lý ảnh hiển vi y sinh\n"
            "Môn: Xử lý ảnh số và video số (ARIN03020)\n"
            "Khung giao diện PySide6, M1 & M8 — Sẵn sàng tích hợp M2–M7 và AI.",
        )

    def closeEvent(self, event):
        self._stop_video_if_playing()
        self.video_source.release()
        super().closeEvent(event)