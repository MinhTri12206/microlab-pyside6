import os
import glob
import cv2
import numpy as np
from typing import Callable, List, Optional, Tuple
from core.m1_pixel import read_image_unicode


class VideoSource:
    """Lớp trừu tượng đọc thống nhất cả tệp video (.mp4/.avi) và chuỗi ảnh (.tif/.png/.jpg)."""

    def __init__(self):
        self.mode: Optional[str] = None  # "video" hoặc "sequence"
        self.cap: Optional[cv2.VideoCapture] = None
        self.files: List[str] = []
        self.total_frames: int = 0
        self.fps: float = 15.0
        self.current_index: int = -1
        self.source_path: str = ""

    def open_video_file(self, path: str) -> None:
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở tệp video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total <= 0:
            cap.release()
            raise ValueError("Video không có khung hình hợp lệ.")
        self.mode = "video"
        self.cap = cap
        self.total_frames = total
        self.fps = fps if fps > 0 else 25.0
        self.current_index = -1
        self.source_path = path

    def open_image_sequence(self, folder: str, fps: float = 10.0) -> None:
        """M8 MicroLab: Đọc chuỗi ảnh t000.tif, t001.tif... theo thứ tự tên."""
        self.release()
        exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        files = sorted(files)
        if not files:
            raise ValueError(f"Thư mục không chứa tệp ảnh hợp lệ (.tif/.png/.jpg): {folder}")
        self.mode = "sequence"
        self.files = files
        self.total_frames = len(files)
        self.fps = fps
        self.current_index = -1
        self.source_path = folder

    def is_open(self) -> bool:
        return self.mode is not None and self.total_frames > 0

    def read_frame(self, index: int) -> Optional[np.ndarray]:
        if not self.is_open():
            return None
        index = max(0, min(self.total_frames - 1, int(index)))

        if self.mode == "sequence":
            raw = read_image_unicode(self.files[index], cv2.IMREAD_UNCHANGED)
            # Chuẩn hóa ảnh 16-bit (như chuỗi ảnh hiển vi Fluo-N2DH-GOWT1) về 0-255 uint8
            if raw.dtype != np.uint8:
                frame = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                frame = raw
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self.current_index = index
            return frame

        if self.mode == "video" and self.cap is not None:
            if index != self.current_index + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
            self.current_index = index
            return frame
        return None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = None
        self.files = []
        self.total_frames = 0
        self.current_index = -1


def compute_frame_difference(prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: int = 25) -> np.ndarray:
    """M8: Phát hiện vùng thay đổi giữa 2 khung liên tiếp bằng cv2.absdiff -> ngưỡng."""
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    g2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame
    diff = cv2.absdiff(g2, g1)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    return mask


class MotionTrackerLK:
    """M8: Theo vết chuyển động tế bào bằng goodFeaturesToTrack + Lucas-Kanade."""

    def __init__(self, max_corners: int = 100):
        self.max_corners = max_corners
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.trail_mask: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.trail_mask = None

    def _init_points(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        self.prev_gray = gray
        self.prev_pts = pts
        h, w = gray.shape[:2]
        self.trail_mask = np.zeros((h, w, 3), dtype=np.uint8)
        rng = np.random.default_rng(123)
        self.colors = rng.integers(80, 255, size=(self.max_corners, 3), dtype=np.uint8)

    def update(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 5:
            self._init_points(vis)
            if self.prev_pts is not None:
                for p in self.prev_pts:
                    x, y = p.ravel()
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
            return vis

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        if next_pts is None or status is None:
            self._init_points(vis)
            return vis

        good_new = next_pts[status == 1]
        good_old = self.prev_pts[status == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = int(new[0]), int(new[1])
            c, d = int(old[0]), int(old[1])
            col = tuple(int(v) for v in self.colors[i % len(self.colors)])
            cv2.line(self.trail_mask, (c, d), (a, b), col, 2, cv2.LINE_AA)
            cv2.circle(vis, (a, b), 3, col, -1)

        out = cv2.add(vis, self.trail_mask)
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        return out


def export_processed_video(
    source: VideoSource,
    pipeline: List[Tuple[Callable, dict]],
    output_path: str,
    fps: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    """M8: Áp chuỗi lệnh đã chọn lên toàn bộ khung hình và xuất tệp .avi/.mp4."""
    if not source.is_open():
        raise ValueError("Chưa mở video hoặc chuỗi ảnh.")

    first = source.read_frame(0)
    if first is None:
        raise ValueError("Không đọc được khung hình đầu tiên.")

    out_fps = fps if fps and fps > 0 else source.fps
    ext = os.path.splitext(output_path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))

    # Chạy thử khung đầu để lấy kích thước đầu ra chính xác (phòng trường hợp có cắt/resize)
    sample_out, _ = _apply_pipeline(first, pipeline)
    h, w = sample_out.shape[:2]

    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Không thể tạo tệp xuất video: {output_path}")

    written = 0
    try:
        total = source.total_frames
        for idx in range(total):
            frame = source.read_frame(idx)
            if frame is None:
                break
            processed, info = _apply_pipeline(frame, pipeline)
            if processed.ndim == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if processed.shape[:2] != (h, w):
                processed = cv2.resize(processed, (w, h))

            # Nếu lệnh xử lý trả về số đếm đối tượng (ví dụ M6), vẽ lên góc trái khung hình
            count_val = info.get("count") or info.get("Số tế bào") or info.get("Tổng số")
            if count_val is not None:
                cv2.putText(
                    processed,
                    f"Count: {count_val}",
                    (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(processed)
            written += 1
            if progress_cb is not None:
                progress_cb(idx + 1, total)
    finally:
        writer.release()

    return written


def _apply_pipeline(frame: np.ndarray, pipeline: List[Tuple[Callable, dict]]) -> Tuple[np.ndarray, dict]:
    curr = frame.copy()
    merged_info: dict = {}
    for func, params in pipeline:
        res = func(curr, **params)
        if isinstance(res, tuple):
            curr, info = res
            if isinstance(info, dict):
                merged_info.update(info)
        else:
            curr = res
    return curr, merged_info