"""Pipeline phát hiện biên Canny cho bài toán phát hiện lỗi bề mặt."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .preprocessing import ImagePreprocessor, save_image
except ImportError:
    from preprocessing import ImagePreprocessor, save_image


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
    """Làm mảnh biên bằng cách giữ cực đại cục bộ (Tối ưu Numpy Vectorized)."""
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.float32)

    angle_q = np.zeros_like(angle, dtype=np.uint8)
    angle_q[(angle >= 22.5) & (angle < 67.5)] = 1
    angle_q[(angle >= 67.5) & (angle < 112.5)] = 2
    angle_q[(angle >= 112.5) & (angle < 157.5)] = 3

    mag_pad = np.pad(magnitude, 1, mode='constant', constant_values=0)
    mag_c = magnitude
    mag_r = mag_pad[1:-1, 2:]
    mag_l = mag_pad[1:-1, :-2]
    mag_ur = mag_pad[:-2, 2:]
    mag_dl = mag_pad[2:, :-2]
    mag_u = mag_pad[:-2, 1:-1]
    mag_d = mag_pad[2:, 1:-1]
    mag_ul = mag_pad[:-2, :-2]
    mag_dr = mag_pad[2:, 2:]

    q0 = angle_q == 0
    q1 = angle_q == 1
    q2 = angle_q == 2
    q3 = angle_q == 3

    keep = (q0 & (mag_c >= mag_r) & (mag_c >= mag_l)) | \
           (q1 & (mag_c >= mag_ur) & (mag_c >= mag_dl)) | \
           (q2 & (mag_c >= mag_u) & (mag_c >= mag_d)) | \
           (q3 & (mag_c >= mag_ul) & (mag_c >= mag_dr))

    suppressed[keep] = magnitude[keep]
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
    """Giữ lại biên yếu nếu nó nối với vùng biên mạnh (Tối ưu Dilation)."""
    strong_edges = (thresholded == strong_value).astype(np.uint8)
    weak_edges = (thresholded == weak_value).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    current_strong = strong_edges
    while True:
        dilated = cv2.dilate(current_strong, kernel)
        new_strong = np.bitwise_and(dilated, weak_edges)
        new_strong = np.bitwise_or(current_strong, new_strong)
        if np.array_equal(current_strong, new_strong):
            break
        current_strong = new_strong
        
    output = np.zeros_like(thresholded, dtype=np.uint8)
    output[current_strong > 0] = strong_value
    return output


def run_canny_pipeline(
    image_path: Path,
    output_dir: Path | None = None,
    median_kernel: int = 5,
    defect_mode: str = "bright",
    k_std: float = 4.0,
    apply_tophat: bool = False,
    tophat_kernel: int = 51,
    gaussian_kernel: int = 5,
    sigma: float = 1.4,
    low_ratio: float = 0.05,
    high_ratio: float = 0.15,
    resize_width: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Chạy toàn bộ pipeline tiền xử lý và phát hiện biên Canny."""
    
    preprocessor = ImagePreprocessor(
        median_kernel=median_kernel,
        resize_width=resize_width,
        defect_mode=defect_mode,
        k_std=k_std,
        apply_tophat=apply_tophat,
        tophat_kernel=tophat_kernel
    )
    preprocessing = preprocessor.process(image_path, output_dir)

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
    
    # -------------------------------------------------------------
    # BỔ SUNG LƯỢNG TỬ HÓA GÓC VÀ LỌC CÓ HƯỚNG (Directional Filters)
    # -------------------------------------------------------------
    horiz_defect_mask = np.zeros_like(edges)
    vert_defect_mask = np.zeros_like(edges)
    diag_defect_mask = np.zeros_like(edges)
    
    # Góc Gradient = 90 độ (Gradient dọc) -> Lỗi đường ngang (Horizontal line)
    is_horiz_defect = (angle >= 67.5) & (angle < 112.5)
    
    # Góc Gradient = 0 hoặc 180 độ (Gradient ngang) -> Lỗi đường dọc (Vertical line)
    is_vert_defect = (angle < 22.5) | (angle >= 157.5)
    
    # Góc chéo
    is_diag_defect = ((angle >= 22.5) & (angle < 67.5)) | ((angle >= 112.5) & (angle < 157.5))
    
    # Áp mask lên ảnh Canny Edges
    horiz_defect_mask[is_horiz_defect & (edges > 0)] = 255
    vert_defect_mask[is_vert_defect & (edges > 0)] = 255
    diag_defect_mask[is_diag_defect & (edges > 0)] = 255
    
    # Lọc hình thái học có hướng bằng Structuring Element (SE) dạng đoạn thẳng
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)) # SE Ngang
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)) # SE Dọc
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # SE Chéo
    
    filtered_horiz = cv2.morphologyEx(horiz_defect_mask, cv2.MORPH_OPEN, kernel_h)
    filtered_vert = cv2.morphologyEx(vert_defect_mask, cv2.MORPH_OPEN, kernel_v)
    filtered_diag = cv2.morphologyEx(diag_defect_mask, cv2.MORPH_OPEN, kernel_d)
    
    results["horiz_mask"] = filtered_horiz
    results["vert_mask"] = filtered_vert
    results["diag_mask"] = filtered_diag

    if output_dir is not None:
        save_image(output_dir / "04_gaussian.png", smoothed)
        save_image(output_dir / "05_magnitude.png", (magnitude / np.max(magnitude) * 255).astype(np.uint8))
        save_image(output_dir / "06_suppressed.png", suppressed.astype(np.uint8))
        save_image(output_dir / "07_canny_edges.png", edges)
        save_image(output_dir / "08a_horiz_lines.png", filtered_horiz)
        save_image(output_dir / "08b_vert_lines.png", filtered_vert)
        save_image(output_dir / "08c_diag_lines.png", filtered_diag)

    return results
