"""Pipeline Xử lý Hình thái học (Morphological Processing)."""

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

try:
    from .preprocessing import ImagePreprocessor, save_image
except ImportError:
    from preprocessing import ImagePreprocessor, save_image


def get_structuring_element(shape: str, size: int) -> np.ndarray:
    """Tạo phần tử cấu trúc (Structuring Element) với các hình dáng cơ bản."""
    if size < 3 or size % 2 == 0:
        raise ValueError("Kích thước Kernel phải là số lẻ và >= 3")
    
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    morph_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (size, size))


def apply_opening(binary_image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Phép Opening: Khử nhiễu nền bằng chuỗi Erosion theo sau bởi Dilation."""
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(binary_image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Phép Closing: Lấp lỗ bằng chuỗi Dilation theo sau bởi Erosion (Cascading)."""
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def run_morphological_pipeline(
    image_path: Path,
    output_dir: Path | None = None,
    median_kernel: int = 5,
    defect_mode: str = "both",
    k_std: float = 4.0,          # Trả lại k_std=4.0 giống Directional để tránh nhiễu vân vải
    apply_tophat: bool = True,
    tophat_kernel: int = 21,     # Đã giảm từ 51 xuống 21 để không bắt lầm vân vải dệt
    resize_width: int | None = None,
    morph_shape: str = "ellipse",
    open_size: int = 3,          # Kernel nhỏ cho Opening (Giữ lại lỗi mờ)
    close_size: int = 5,         # Kernel lớn cho Closing (Kết dính khối)
    iterations: int = 3,
) -> dict[str, np.ndarray | float]:
    """Chạy toàn bộ pipeline: Tiền xử lý -> Statistical Threshold -> Morphological."""
    
    preprocessor = ImagePreprocessor(
        median_kernel=median_kernel,
        resize_width=resize_width,
        defect_mode=defect_mode,
        k_std=k_std,
        apply_tophat=apply_tophat,
        tophat_kernel=tophat_kernel
    )
    preprocessing = preprocessor.process(image_path, output_dir)
    
    binary_mask = preprocessing["binary_mask"]
    
    # Tạo 2 bộ Kernel độc lập cho Mở và Đóng
    kernel_open = get_structuring_element(morph_shape, open_size)
    kernel_close = get_structuring_element(morph_shape, close_size)
    
    # Mở bằng Kernel nhỏ (xóa rác 1-2 pixel mà không làm mất vết dơ)
    opened_image = apply_opening(binary_mask, kernel_open, iterations=1)
    
    # Đóng bằng Kernel lớn (Nối các mảng dơ đứt đoạn thành một khối Stain hoàn chỉnh)
    closed_image = apply_closing(opened_image, kernel_close, iterations=iterations)

    results: dict[str, np.ndarray | float] = {
        **preprocessing,
        "morph_kernel_open": kernel_open,
        "morph_kernel_close": kernel_close,
        "morph_opening": opened_image,
        "morph_closing": closed_image,
    }

    if output_dir is not None:
        save_image(output_dir / "04_morph_opening.png", opened_image)
        save_image(output_dir / "05_morph_closing.png", closed_image)

    return results