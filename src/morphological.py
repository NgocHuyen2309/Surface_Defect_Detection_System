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
    defect_mode: str = "bright",
    k_std: float = 4.0,
    apply_tophat: bool = True,
    tophat_kernel: int = 51,
    resize_width: int | None = None,
    morph_shape: str = "rect",
    morph_size: int = 5,
    iterations: int = 1,
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
    kernel = get_structuring_element(morph_shape, morph_size)
    
    # Chạy Opening để khử nhiễu (Dùng iter = 1 để tránh làm rách thêm viền lỗi)
    opened_image = apply_opening(binary_mask, kernel, iterations=1)
    
    # Chạy Closing với số vòng lặp linh hoạt (Cascading) để lấp kín các khe hở lớn
    closed_image = apply_closing(opened_image, kernel, iterations=iterations)

    results: dict[str, np.ndarray | float] = {
        **preprocessing,
        "morph_kernel": kernel,
        "morph_opening": opened_image,
        "morph_closing": closed_image,
    }

    if output_dir is not None:
        save_image(output_dir / "04_morph_opening.png", opened_image)
        save_image(output_dir / "05_morph_closing.png", closed_image)

    return results