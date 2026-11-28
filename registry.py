from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


@dataclass
class ParamSpec:
    """Mô tả một tham số để bảng Tham số tự động tạo widget tương ứng."""
    name: str
    label: str
    kind: str  # "int", "odd_int", "float", "choice", "bool"
    default: Any
    min_val: float = 0
    max_val: float = 255
    step: float = 1
    choices: List[Any] = field(default_factory=list)


@dataclass
class OperationSpec:
    """Mô tả một lệnh xử lý ảnh chuẩn hóa cho toàn bộ phần mềm.

    Quy ước hàm `func(img: np.ndarray, **params)`:
      - Đầu vào: ảnh NumPy uint8 (BGR 3 kênh hoặc xám 1 kênh) và các tham số.
      - Đầu ra:
          * `result_img` (np.ndarray), HOẶC
          * `(result_img, info_dict)` trong đó `info_dict` chứa dữ liệu đưa lên Bảng Kết quả
            (ví dụ: {"PSNR": "31.2 dB"}, {"count": 42, "table_headers": [...], "table_rows": [...]}).
    """
    id: str
    menu: str              # "Ảnh", "Điều chỉnh", "Bộ lọc", "Màu", "Hình thái", "Biên", "AI"
    name: str              # Tên hiển thị trên menu & lịch sử
    func: Callable
    params: List[ParamSpec] = field(default_factory=list)
    supports_video: bool = True
    extra_buttons: List[Tuple[str, Callable]] = field(default_factory=list)


class OperationRegistry:
    """Quản lý tập trung tất cả các lệnh xử lý ảnh."""

    def __init__(self):
        self._ops: dict[str, OperationSpec] = {}
        self._order: list[str] = []

    def register(self, spec: OperationSpec):
        self._ops[spec.id] = spec
        if spec.id not in self._order:
            self._order.append(spec.id)

    def get(self, op_id: str) -> Optional[OperationSpec]:
        return self._ops.get(op_id)

    def by_menu(self, menu_name: str) -> List[OperationSpec]:
        return [self._ops[oid] for oid in self._order if self._ops[oid].menu == menu_name]

    def all_ops(self) -> List[OperationSpec]:
        return [self._ops[oid] for oid in self._order]

    def video_compatible_ops(self) -> List[OperationSpec]:
        return [self._ops[oid] for oid in self._order if self._ops[oid].supports_video]