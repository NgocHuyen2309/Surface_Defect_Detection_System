"""Các hàm tiền xử lý ảnh dùng chung cho pipeline xử lý ảnh (OOP)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def ensure_directory(directory: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    directory.mkdir(parents=True, exist_ok=True)


def find_images(input_dir: Path) -> Iterable[Path]:
    """Duyệt toàn bộ ảnh trong thư mục đầu vào."""
    for image_path in sorted(input_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield image_path


def save_image(output_path: Path, image: np.ndarray) -> None:
    """Lưu ảnh kết quả ra ổ đĩa."""
    ensure_directory(output_path.parent)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise OSError(f"Không thể ghi ảnh: {output_path}")


class ImagePreprocessor:
    """Class tiền xử lý ảnh theo chuẩn OOP."""
    
    def __init__(
        self, 
        median_kernel: int = 5, 
        resize_width: int | None = None, 
        defect_mode: str = "both", 
        k_std: float = 4.0,
        apply_tophat: bool = False,
        tophat_kernel: int = 51
    ):
        self.median_kernel = median_kernel
        self.resize_width = resize_width
        self.defect_mode = defect_mode.lower()
        self.k_std = k_std
        self.apply_tophat = apply_tophat
        self.tophat_kernel = tophat_kernel
        
    def read_image(self, image_path: Path) -> np.ndarray:
        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        return image
        
    def to_grayscale(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
    def resize_by_width(self, image: np.ndarray) -> np.ndarray:
        if self.resize_width is None or self.resize_width <= 0:
            return image
        height, current_width = image.shape[:2]
        scale = self.resize_width / float(current_width)
        new_height = int(round(height * scale))
        return cv2.resize(image, (self.resize_width, new_height), interpolation=cv2.INTER_AREA)
        
    def apply_median_filter(self, gray_image: np.ndarray) -> np.ndarray:
        if self.median_kernel < 3 or self.median_kernel % 2 == 0:
            raise ValueError("Kích thước kernel Median phải là số lẻ và >= 3")
        return cv2.medianBlur(gray_image, self.median_kernel)
        
    def apply_illumination_correction(self, gray_image: np.ndarray) -> np.ndarray:
        """Cân bằng ánh sáng nền bằng Top-Hat hoặc Black-Hat Transform."""
        if not self.apply_tophat:
            return gray_image
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tophat_kernel, self.tophat_kernel))
        if self.defect_mode == "both":
            tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
            corrected = cv2.add(tophat, blackhat)
        elif self.defect_mode == "bright":
            # Nổi bật đốm sáng (Hole) trên nền dệt
            corrected = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        else:
            # Nổi bật đốm tối (Stain, lines) trên nền dệt
            corrected = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
            
        return corrected
        
    def apply_statistical_threshold(self, gray_image: np.ndarray) -> tuple[float, np.ndarray]:
        """Cắt ngưỡng dựa trên Mean + k*Std."""
        mean, std = cv2.meanStdDev(gray_image)
        m = float(mean[0][0])
        s = float(std[0][0])
        
        # Nếu đã dùng Top-Hat/Black-Hat, ảnh đầu ra luôn có background tối (~0) và lỗi sáng (>0)
        if self.apply_tophat or self.defect_mode in ("bright", "both"):
            threshold_value = m + self.k_std * s
            threshold_type = cv2.THRESH_BINARY
        else:
            threshold_value = m - self.k_std * s
            threshold_type = cv2.THRESH_BINARY_INV
            
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, threshold_type)
        return threshold_value, binary_image
        
    def save_histogram(self, output_path: Path, gray_image: np.ndarray, threshold_value: float) -> None:
        """Vẽ histogram mức xám và đánh dấu ngưỡng Thống kê."""
        ensure_directory(output_path.parent)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(gray_image.ravel(), bins=256, range=(0, 255))
        axis.axvline(threshold_value, linestyle="--", color="red", linewidth=2, label=f"Threshold = {threshold_value:.2f}")
        axis.set_title("Histogram và Ngưỡng Statistical (Mean ± k*Std)")
        axis.set_xlabel("Giá trị điểm ảnh")
        axis.set_ylabel("Số lượng điểm ảnh")
        axis.set_xlim(0, 255)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        
    def process(self, image_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
        """Thực thi chuỗi tiền xử lý đầy đủ."""
        image_bgr = self.read_image(image_path)
        gray_image = self.to_grayscale(image_bgr)
        gray_image = self.resize_by_width(gray_image)
        
        median_image = self.apply_median_filter(gray_image)
        corrected_image = self.apply_illumination_correction(median_image)
        threshold_value, binary_mask = self.apply_statistical_threshold(corrected_image)
        
        results = {
            "gray": gray_image,
            "median": median_image,
            "corrected": corrected_image,
            "threshold_value": threshold_value,
            "binary_mask": binary_mask,
        }
        
        if output_dir is not None:
            save_image(output_dir / "01_grayscale.png", gray_image)
            save_image(output_dir / "02_median.png", median_image)
            if self.apply_tophat:
                save_image(output_dir / "02b_tophat_corrected.png", corrected_image)
            self.save_histogram(output_dir / "03a_histogram.png", corrected_image, threshold_value)
            save_image(output_dir / "03b_binary_mask.png", binary_mask)
            
        return results
