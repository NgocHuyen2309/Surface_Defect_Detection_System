"""Các hàm tiền xử lý ảnh dùng chung cho pipeline xử lý ảnh."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Cấu hình định dạng ảnh được hỗ trợ
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def ensure_directory(directory: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    directory.mkdir(parents=True, exist_ok=True)


def find_images(input_dir: Path) -> Iterable[Path]:
    """Duyệt toàn bộ ảnh trong thư mục đầu vào."""
    for image_path in sorted(input_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield image_path


def read_image(image_path: Path) -> np.ndarray:
    """Đọc ảnh đầu vào theo định dạng BGR của OpenCV."""
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Chuyển ảnh BGR sang ảnh xám."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def resize_by_width(image: np.ndarray, width: int | None) -> np.ndarray:
    """Resize ảnh theo chiều rộng và giữ nguyên tỷ lệ."""
    if width is None or width <= 0:
        return image

    height, current_width = image.shape[:2]
    scale = width / float(current_width)
    new_height = int(round(height * scale))
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)


def apply_median_filter(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Khử nhiễu xung bằng bộ lọc trung vị."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Kích thước kernel Median phải là số lẻ và >= 3")
    return cv2.medianBlur(gray_image, kernel_size)


def apply_otsu_threshold(gray_image: np.ndarray, invert: bool = False) -> tuple[float, np.ndarray]:
    """Nhị phân hóa ảnh bằng ngưỡng tự động Otsu."""
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_value, binary_image = cv2.threshold(
        gray_image,
        0,
        255,
        threshold_type + cv2.THRESH_OTSU,
    )
    return threshold_value, binary_image


def save_image(output_path: Path, image: np.ndarray) -> None:
    """Lưu ảnh kết quả ra ổ đĩa."""
    ensure_directory(output_path.parent)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise OSError(f"Không thể ghi ảnh: {output_path}")


def preprocess_image(
    image_path: Path,
    output_dir: Path | None = None,
    median_kernel: int = 5,
    invert_otsu: bool = False,
    resize_width: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Chạy pipeline tiền xử lý gồm grayscale, Median Filter và Otsu."""
    image_bgr = read_image(image_path)

    # Chuẩn hóa ảnh đầu vào
    gray_image = to_grayscale(image_bgr)
    gray_image = resize_by_width(gray_image, resize_width)

    # Khử nhiễu và phân đoạn nhị phân
    median_image = apply_median_filter(gray_image, median_kernel)
    threshold_value, otsu_binary = apply_otsu_threshold(median_image, invert_otsu)

    results: dict[str, np.ndarray | float] = {
        "gray": gray_image,
        "median": median_image,
        "otsu_threshold": threshold_value,
        "otsu_binary": otsu_binary,
    }

    # Lưu ảnh trung gian phục vụ kiểm tra và báo cáo
    if output_dir is not None:
        save_image(output_dir / "01_grayscale.png", gray_image)
        save_image(output_dir / "02_median.png", median_image)
        save_image(output_dir / "03_otsu_binary.png", otsu_binary)

    return results
