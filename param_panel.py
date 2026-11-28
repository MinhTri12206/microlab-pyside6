from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ParamPanel(QWidget):
    """Build parameter controls for an operation registered in the app."""

    previewRequested = Signal(dict)
    applyRequested = Signal(dict)
    cancelRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spec = None
        self._widgets = {}

        layout = QVBoxLayout(self)
        self._form_host = QWidget(self)
        self._form = QFormLayout(self._form_host)
        layout.addWidget(self._form_host)

        buttons = QHBoxLayout()
        self.btn_apply = QPushButton("Áp dụng")
        self.btn_apply.clicked.connect(lambda: self.applyRequested.emit(self.values()))
        self.btn_preview = QPushButton("Xem trÆ°á»›c")
        self.btn_preview.clicked.connect(lambda: self.previewRequested.emit(self.values()))
        self.btn_cancel = QPushButton("Há»§y")
        self.btn_cancel.clicked.connect(self.cancelRequested.emit)
        self.btn_reset = QPushButton("Đặt lại")
        self.btn_reset.clicked.connect(self.reset_values)
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(self.btn_preview)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_reset)
        layout.addLayout(buttons)

        self._empty_label = QLabel("Chọn một thao tác để xem tham số.")
        self._empty_label.setWordWrap(True)
        self._form.addRow(self._empty_label)

    def load_operation(self, spec):
        self._clear_form()
        self._spec = spec
        self._widgets = {}
        if spec is None:
            self._empty_label = QLabel("Chọn một thao tác để xem tham số.")
            self._form.addRow(self._empty_label)
            self.btn_apply.setEnabled(False)
            self.btn_reset.setEnabled(False)
            return

        self.btn_apply.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_reset.setEnabled(True)
        for param in spec.params:
            kind = param.kind.lower()
            if kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(param.default))
                widget.toggled.connect(self._emit_params)
            elif kind == "choice":
                widget = QComboBox()
                for choice in param.choices:
                    widget.addItem(str(choice), choice)
                index = widget.findData(param.default)
                if index >= 0:
                    widget.setCurrentIndex(index)
                widget.currentIndexChanged.connect(self._emit_params)
            elif kind == "float":
                widget = QDoubleSpinBox()
                widget.setRange(float(param.min_val), float(param.max_val))
                widget.setSingleStep(float(param.step))
                widget.setValue(float(param.default))
                widget.valueChanged.connect(self._emit_params)
            else:
                widget = QSpinBox()
                widget.setRange(int(param.min_val), int(param.max_val))
                widget.setSingleStep(max(1, int(param.step)))
                widget.setValue(int(param.default))
                widget.valueChanged.connect(self._emit_params)
            self._widgets[param.name] = (param, widget)
            self._form.addRow(param.label, widget)

    def set_operation(self, spec):
        self.load_operation(spec)

    def clear_operation(self):
        self._clear_form()
        self._widgets = {}
        self._spec = None
        self._empty_label = QLabel("Chá»n má»™t thao tĂ¡c Ä‘á»ƒ xem tham sá»‘.")
        self._form.addRow(self._empty_label)
        self.btn_apply.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_cancel.setEnabled(False)

    def values(self):
        result = {}
        for name, (param, widget) in self._widgets.items():
            if param.kind.lower() == "choice":
                result[name] = widget.currentData()
            elif param.kind.lower() == "bool":
                result[name] = widget.isChecked()
            else:
                result[name] = widget.value()
        return result

    def reset_values(self):
        for param, widget in self._widgets.values():
            if param.kind.lower() == "choice":
                index = widget.findData(param.default)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif param.kind.lower() == "bool":
                widget.setChecked(bool(param.default))
            else:
                widget.setValue(param.default)
        self._emit_params()

    def _emit_params(self, *_):
        self.previewRequested.emit(self.values())

    def _clear_form(self):
        while self._form.rowCount():
            self._form.removeRow(0)

