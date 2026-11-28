import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from core.registry import OperationRegistry, OperationSpec, ParamSpec
from gui.main_window import MainWindow


def register_demo_operations(registry: OperationRegistry):
    """Đăng ký 1 lệnh mẫu (CLAHE) để kiểm chứng bảng Tham số, Xem trước, Undo/Redo và Xuất Video.
    SV2 và SV3 sẽ đăng ký M2–M7 và AI theo đúng cấu trúc này.
    """
    def demo_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size)))
        if img.ndim == 2:
            return clahe.apply(img)
        b, g, r = cv2.split(img)
        return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])

    registry.register(
        OperationSpec(
            id="m2_clahe_demo",
            menu="Điều chỉnh",
            name="CLAHE (Mẫu kiểm thử)",
            func=demo_clahe,
            params=[
                ParamSpec("clip_limit", "Clip Limit", "float", default=2.0, min_val=1.0, max_val=5.0, step=0.1),
                ParamSpec("grid_size", "Kích thước lưới ô", "int", default=8, min_val=4, max_val=16, step=1),
            ],
        )
    )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    registry = OperationRegistry()
    register_demo_operations(registry)

    # Sau khi SV2 và SV3 xong, chỉ cần bỏ comment 2 dòng dưới:
    # from core.sv2_ops import register_sv2_operations
    # from core.sv3_ops import register_sv3_operations
    # register_sv2_operations(registry)
    # register_sv3_operations(registry)

    win = MainWindow(registry)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()