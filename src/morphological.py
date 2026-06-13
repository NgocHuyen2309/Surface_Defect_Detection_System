"""Pipeline xử lý hình thái học cho các lỗi dạng vùng/khối trên bề mặt vải."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .preprocessing import ImagePreprocessor, save_image
except ImportError:
    from preprocessing import ImagePreprocessor, save_image


def get_structuring_element(shape: str, size: int) -> np.ndarray:
    """Tạo phần tử cấu trúc cho các phép toán hình thái học."""
    if size < 3 or size % 2 == 0:
        raise ValueError("Kích thước Kernel phải là số lẻ và >= 3")

    # Cho phép đổi hình dạng kernel khi cần thử nghiệm trên từng loại vải.
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    morph_shape = shape_map.get(shape.lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph_shape, (size, size))


def apply_opening(binary_image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Phép Opening: loại bỏ nhiễu nhỏ bằng Erosion rồi Dilation."""
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(binary_image: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Phép Closing: nối và lấp các vùng lỗi bằng Dilation rồi Erosion."""
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def run_morphological_pipeline(
    image_path: Path,
    output_dir: Path | None = None,
    median_kernel: int = 5,
    defect_mode: str = "both",
    k_std: float = 3.5,
    apply_tophat: bool = True,
    tophat_kernel: int = 21,
    resize_width: int | None = None,
    morph_shape: str = "ellipse",
    open_size: int = 3,
    close_size: int = 5,
    iterations: int = 3,
) -> dict[str, np.ndarray | float]:
    """Chạy pipeline: tiền xử lý -> threshold -> Opening -> Closing."""
    # Tiền xử lý tạo ảnh xám, ảnh đã lọc nhiễu, ảnh cân bằng sáng và binary mask.
    preprocessor = ImagePreprocessor(
        median_kernel=median_kernel,
        resize_width=resize_width,
        defect_mode=defect_mode,
        k_std=k_std,
        apply_tophat=apply_tophat,
        tophat_kernel=tophat_kernel,
    )
    preprocessing = preprocessor.process(image_path, output_dir)
    binary_mask = preprocessing["binary_mask"]

    # Kernel nhỏ cho Opening để xóa nhiễu vân vải li ti.
    kernel_open = get_structuring_element(morph_shape, open_size)

    # Kernel lớn hơn cho Closing để gom vùng lỗi bị đứt đoạn thành khối rõ hơn.
    kernel_close = get_structuring_element(morph_shape, close_size)

    opened_image = apply_opening(binary_mask, kernel_open, iterations=1)
    closed_image = apply_closing(opened_image, kernel_close, iterations=iterations)

    results: dict[str, np.ndarray | float] = {
        **preprocessing,
        "morph_kernel_open": kernel_open,
        "morph_kernel_close": kernel_close,
        "morph_opening": opened_image,
        "morph_closing": closed_image,
    }

    # Lưu ảnh minh họa tác động của Opening và Closing cho báo cáo.
    if output_dir is not None:
        save_image(output_dir / "04_morph_opening.png", opened_image)
        save_image(output_dir / "05_morph_closing.png", closed_image)

    return results
