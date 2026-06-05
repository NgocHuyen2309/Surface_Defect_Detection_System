"""Pipeline phát hiện biên Canny được tùy biến thành Gradient Magnitude Pipeline cho lỗi bề mặt."""

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


def run_directional_gradient_pipeline(
    image_path: str | Path, 
    median_ksize: int = 7, 
    apply_tophat: bool = True, 
    tophat_ksize: int = 15,
    resize_width: int | None = None,
    output_dir: Path | None = None,
    k_std: float = 4.0
) -> dict[str, np.ndarray | float]:
    """
    Thực thi luồng trích xuất Gradient có hướng (Directional Gradient Magnitude).
    
    Args:
        image_path: Đường dẫn tới ảnh
        median_ksize: Kích thước kernel lọc trung vị
        apply_tophat: Có áp dụng TopHat/BlackHat không
        tophat_ksize: Kích thước kernel cho TopHat
        resize_width: Có resize ảnh không (để tăng tốc)
        output_dir: Thư mục lưu ảnh trung gian
        k_std: Hệ số nhân Std để tính ngưỡng thống kê
        
    Returns:
        dict chứa các ảnh trung gian và mặt nạ kết quả
    """
    
    preprocessor = ImagePreprocessor(
        median_kernel=median_ksize,
        resize_width=resize_width,
        defect_mode="bright",
        k_std=k_std,
        apply_tophat=apply_tophat,
        tophat_kernel=tophat_ksize
    )
    preprocessing = preprocessor.process(image_path, output_dir)

    # 1. Lấy ảnh đầu vào
    base_image = preprocessing["median"]
    
    # 2. Làm mờ Gaussian
    smoothed = gaussian_smoothing(base_image, 5, 1.4)
    
    # 3. Tính Gradient Magnitude và Angle
    magnitude, angle = compute_gradient(smoothed)
    
    # 4. Statistical Thresholding trực tiếp trên Magnitude (Thay thế Otsu)
    # Otsu thất bại trên nền vân vải (giữ lại quá nhiều nhiễu). 
    # Ta dùng Ngưỡng thống kê: Mean + k_std * Std để chỉ giữ lại các điểm Gradient đột biến
    mean_val = float(np.mean(magnitude))
    std_val = float(np.std(magnitude))
    
    # Đối với Gradient, ta giảm k_std một chút so với Intensity (ví dụ: -1.0) để không làm đứt nét vệt mờ
    adjusted_k = max(1.0, k_std - 1.0) 
    thresh_val = mean_val + adjusted_k * std_val
    
    _, thresholded = cv2.threshold(magnitude, thresh_val, 255, cv2.THRESH_BINARY)
    thresholded = thresholded.astype(np.uint8)
    
    # Xóa nhiễu hạt liti (1-2 pixel) ngay lập tức bằng MORPH_OPEN kernel 3x3 trước khi chia hướng
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel_clean)

    results: dict[str, np.ndarray | float] = {
        **preprocessing,
        "gaussian": smoothed,
        "gradient_magnitude": magnitude,
        "gradient_angle": angle,
        "otsu_thresholded": thresholded,
    }
    
    # -------------------------------------------------------------
    # BỔ SUNG LƯỢNG TỬ HÓA GÓC VÀ LỌC CÓ HƯỚNG (Directional Filters)
    # -------------------------------------------------------------
    horiz_defect_mask = np.zeros_like(thresholded)
    vert_defect_mask = np.zeros_like(thresholded)
    diag_defect_mask = np.zeros_like(thresholded)
    
    # Góc Gradient = 90 độ (Gradient dọc) -> Lỗi đường ngang (Horizontal line)
    is_horiz_defect = (angle >= 67.5) & (angle < 112.5)
    
    # Góc Gradient = 0 hoặc 180 độ (Gradient ngang) -> Lỗi đường dọc (Vertical line)
    is_vert_defect = (angle < 22.5) | (angle >= 157.5)
    
    # Góc chéo
    is_diag_defect = ((angle >= 22.5) & (angle < 67.5)) | ((angle >= 112.5) & (angle < 157.5))
    
    # Áp mask lên ảnh đã Threshold
    horiz_defect_mask[is_horiz_defect & (thresholded > 0)] = 255
    vert_defect_mask[is_vert_defect & (thresholded > 0)] = 255
    diag_defect_mask[is_diag_defect & (thresholded > 0)] = 255
    
    # Lọc hình thái học có hướng
    # Bước 1: MORPH_OPEN với kernel mảnh để xóa bỏ các đoạn vạch nhiễu của vân vải (Noise Removal)
    kernel_h_open = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kernel_v_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    kernel_d_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Bước 2: MORPH_CLOSE với kernel dày để nối các đoạn đứt gãy của vết xước thành 1 dải liền mạch (Gap Bridging)
    kernel_h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    kernel_v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21))
    kernel_d_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Áp dụng cho từng hướng
    filtered_horiz = cv2.morphologyEx(horiz_defect_mask, cv2.MORPH_OPEN, kernel_h_open)
    filtered_horiz = cv2.morphologyEx(filtered_horiz, cv2.MORPH_CLOSE, kernel_h_close)
    
    filtered_vert = cv2.morphologyEx(vert_defect_mask, cv2.MORPH_OPEN, kernel_v_open)
    filtered_vert = cv2.morphologyEx(filtered_vert, cv2.MORPH_CLOSE, kernel_v_close)
    
    filtered_diag = cv2.morphologyEx(diag_defect_mask, cv2.MORPH_OPEN, kernel_d_open)
    filtered_diag = cv2.morphologyEx(filtered_diag, cv2.MORPH_CLOSE, kernel_d_close)
    
    results["horiz_mask"] = filtered_horiz
    results["vert_mask"] = filtered_vert
    results["diag_mask"] = filtered_diag

    if output_dir is not None:
        save_image(output_dir / "04_gaussian.png", smoothed)
        save_image(output_dir / "05_magnitude.png", np.clip(magnitude, 0, 255).astype(np.uint8))
        save_image(output_dir / "06_otsu_thresholded.png", thresholded)
        save_image(output_dir / "08a_horiz_lines.png", filtered_horiz)
        save_image(output_dir / "08b_vert_lines.png", filtered_vert)
        save_image(output_dir / "08c_diag_lines.png", filtered_diag)

    return results
