"""Pipeline phát hiện biên Canny cho bài toán phát hiện lỗi bề mặt."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .preprocessing import preprocess_image, save_image
except ImportError:
    from preprocessing import preprocess_image, save_image


def gaussian_smoothing(gray_image: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Làm mịn ảnh bằng bộ lọc Gaussian."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Kích thước kernel Gaussian phải là số lẻ và >= 3")
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)


def compute_gradient(smoothed_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tính độ lớn và hướng gradient bằng toán tử Sobel."""
    gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(gradient_x, gradient_y)
    max_value = float(magnitude.max())
    if max_value > 0:
        magnitude = magnitude / max_value * 255.0

    angle = np.rad2deg(np.arctan2(gradient_y, gradient_x))
    angle[angle < 0] += 180
    return magnitude.astype(np.float32), angle.astype(np.float32)


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Làm mảnh biên bằng cách giữ cực đại cục bộ theo hướng gradient."""
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.float32)

    # So sánh mỗi điểm ảnh với hai điểm lân cận theo hướng gradient
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            direction = angle[row, col]
            before = 255.0
            after = 255.0

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                before = magnitude[row, col - 1]
                after = magnitude[row, col + 1]
            elif 22.5 <= direction < 67.5:
                before = magnitude[row - 1, col + 1]
                after = magnitude[row + 1, col - 1]
            elif 67.5 <= direction < 112.5:
                before = magnitude[row - 1, col]
                after = magnitude[row + 1, col]
            elif 112.5 <= direction < 157.5:
                before = magnitude[row - 1, col - 1]
                after = magnitude[row + 1, col + 1]

            if magnitude[row, col] >= before and magnitude[row, col] >= after:
                suppressed[row, col] = magnitude[row, col]

    return suppressed


def double_threshold(
    image: np.ndarray,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
) -> tuple[np.ndarray, int, int]:
    """Phân loại điểm biên thành biên mạnh, biên yếu và nền."""
    if not 0 < low_ratio < high_ratio < 1:
        raise ValueError("Cần thỏa điều kiện 0 < low_ratio < high_ratio < 1")

    high_threshold = float(image.max()) * high_ratio
    low_threshold = high_threshold * low_ratio / high_ratio

    strong_value = 255
    weak_value = 75
    thresholded = np.zeros_like(image, dtype=np.uint8)

    # Gán nhãn biên mạnh và biên yếu theo hai ngưỡng
    strong_rows, strong_cols = np.where(image >= high_threshold)
    weak_rows, weak_cols = np.where((image >= low_threshold) & (image < high_threshold))
    thresholded[strong_rows, strong_cols] = strong_value
    thresholded[weak_rows, weak_cols] = weak_value

    return thresholded, weak_value, strong_value


def hysteresis(thresholded: np.ndarray, weak_value: int, strong_value: int) -> np.ndarray:
    """Giữ lại biên yếu nếu nó nối với vùng biên mạnh."""
    height, width = thresholded.shape
    output = thresholded.copy()

    # Kiểm tra vùng lân cận 3x3 quanh từng điểm biên yếu
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if output[row, col] == weak_value:
                local_patch = output[row - 1: row + 2, col - 1: col + 2]
                if np.any(local_patch == strong_value):
                    output[row, col] = strong_value
                else:
                    output[row, col] = 0

    output[output != strong_value] = 0
    return output.astype(np.uint8)


def run_canny_pipeline(
    image_path: Path,
    output_dir: Path | None = None,
    median_kernel: int = 5,
    gaussian_kernel: int = 5,
    sigma: float = 1.4,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
    invert_otsu: bool = False,
    resize_width: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Chạy toàn bộ pipeline tiền xử lý và phát hiện biên Canny."""
    preprocessing = preprocess_image(
        image_path=image_path,
        output_dir=output_dir,
        median_kernel=median_kernel,
        invert_otsu=invert_otsu,
        resize_width=resize_width,
    )

    # Thực hiện các bước chính của Canny
    median_image = preprocessing["median"]
    smoothed = gaussian_smoothing(median_image, gaussian_kernel, sigma)
    magnitude, angle = compute_gradient(smoothed)
    suppressed = non_maximum_suppression(magnitude, angle)
    thresholded, weak_value, strong_value = double_threshold(
        suppressed,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
    )
    edges = hysteresis(thresholded, weak_value, strong_value)

    results: dict[str, np.ndarray | float] = {
        **preprocessing,
        "gaussian": smoothed,
        "gradient_magnitude": magnitude,
        "gradient_angle": angle,
        "non_maximum_suppression": suppressed,
        "double_threshold": thresholded,
        "canny_edges": edges,
    }

    # Lưu chuỗi ảnh minh họa từng bước xử lý
    if output_dir is not None:
        save_image(output_dir / "04_gaussian.png", smoothed)
        save_image(output_dir / "05_gradient_magnitude.png", magnitude.astype(np.uint8))
        save_image(output_dir / "06_non_maximum_suppression.png", suppressed.astype(np.uint8))
        save_image(output_dir / "07_double_threshold.png", thresholded)
        save_image(output_dir / "08_canny_edges.png", edges)

    return results
