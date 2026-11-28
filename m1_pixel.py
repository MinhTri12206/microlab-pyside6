import os
import math
import cv2
import numpy as np


def read_image_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Đọc ảnh hỗ trợ đường dẫn tiếng Việt/Unicode trên Windows."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy tệp: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Tệp không hợp lệ hoặc không thể giải mã ảnh: {path}")
    return img


def save_image_unicode(path: str, img: np.ndarray) -> None:
    """Lưu ảnh PNG/JPG hỗ trợ đường dẫn Unicode."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        ext = ".png"
        path += ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Không thể mã hóa ảnh sang định dạng {ext}")
    buf.tofile(path)


def clip_to_uint8(img: np.ndarray) -> np.ndarray:
    """Quy ước môn học: tính toán ở float, cắt về [0, 255] và đổi sang uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def get_image_info(img: np.ndarray) -> dict:
    """M1: Lấy thông tin ảnh (kích thước, số kênh, kiểu dữ liệu, dung lượng chưa nén)."""
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    dtype_str = str(img.dtype)
    uncompressed_bytes = int(w) * int(h) * int(channels) * int(img.dtype.itemsize)
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "channel_desc": "Ảnh xám (1 kênh)" if channels == 1 else f"Ảnh màu ({channels} kênh BGR)",
        "dtype": dtype_str,
        "uncompressed_bytes": uncompressed_bytes,
    }


def split_bgr_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M1: Tách ảnh màu thành 3 kênh xám B, G, R."""
    if img.ndim == 2 or img.shape[2] < 3:
        raise ValueError("Ảnh hiện tại là ảnh xám, không thể tách 3 kênh màu B, G, R.")
    b, g, r = cv2.split(img)
    return b, g, r


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """M1: Chuyển ảnh màu sang ảnh xám."""
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """M1: Cắt ảnh theo hình chữ nhật (x1, y1) - (x2, y2)."""
    h, w = img.shape[:2]
    xa, xb = sorted((max(0, min(w, int(x1))), max(0, min(w, int(x2)))))
    ya, yb = sorted((max(0, min(h, int(y1))), max(0, min(h, int(y2)))))
    if xb - xa < 2 or yb - ya < 2:
        raise ValueError("Vùng chọn quá nhỏ. Hãy kéo chọn vùng chữ nhật lớn hơn 2x2 pixel.")
    return img[ya:yb, xa:xb].copy()


def measure_distances_tc(x1: int, y1: int, x2: int, y2: int) -> dict:
    """M1 [TC]: Tự cài đặt khoảng cách Euclid, D4 (khối phố), D8 (bàn cờ).
    Kiểm thử chuẩn theo đề: A(10, 20), B(13, 24) -> Euclid=5.0, D4=7, D8=4.
    """
    dx = abs(int(x2) - int(x1))
    dy = abs(int(y2) - int(y1))
    euclid = math.sqrt(dx * dx + dy * dy)
    d4 = dx + dy
    d8 = max(dx, dy)
    return {
        "A": (int(x1), int(y1)),
        "B": (int(x2), int(y2)),
        "Euclid": round(euclid, 2),
        "D4": d4,
        "D8": d8,
    }


def create_sample_blood_smear(width: int = 640, height: int = 480) -> np.ndarray:
    """Tạo ảnh tiêu bản phết máu giả lập (hồng cầu + nhân bạch cầu tím) để chạy thử ngay."""
    rng = np.random.default_rng(42)
    img = np.full((height, width, 3), (230, 225, 242), dtype=np.uint8)  # nền sáng hồng nhạt (BGR)

    # Vẽ các hồng cầu (RBC) dạng đĩa tròn màu hồng
    for _ in range(38):
        cx = int(rng.integers(35, width - 35))
        cy = int(rng.integers(35, height - 35))
        radius = int(rng.integers(18, 28))
        color = (int(rng.integers(155, 185)), int(rng.integers(140, 175)), int(rng.integers(205, 235)))
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius // 2, (205, 195, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, (120, 110, 175), 1, cv2.LINE_AA)

    # Vẽ 2 bạch cầu (WBC) với nhân màu tím đậm
    for cx, cy in [(210, 210), (455, 300)]:
        cv2.circle(img, (cx, cy), 38, (215, 185, 220), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx - 8, cy - 4), (20, 14), 25, 0, 360, (145, 45, 115), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx + 10, cy + 8), (16, 12), -30, 0, 360, (135, 35, 105), -1, cv2.LINE_AA)

    # Thêm nhiễu cảm biến nhẹ
    noise = rng.normal(0, 3.5, img.shape)
    return clip_to_uint8(img.astype(np.float32) + noise)