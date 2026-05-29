"""Trích xuất đặc trưng hình học (Region Moments) từ vùng lỗi."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RegionFeatureExtractor:
    """Class hỗ trợ rút trích đặc trưng bất biến từ ảnh nhị phân lỗi."""
    
    def __init__(self, original_image: np.ndarray, binary_mask: np.ndarray) -> None:
        """
        Khởi tạo với ảnh gốc để vẽ đè và mask để tính toán.
        
        Args:
            original_image: Ảnh BGR hoặc Grayscale gốc.
            binary_mask: Ảnh nhị phân phân đoạn vùng lỗi.
        """
        if len(original_image.shape) == 2:
            self.overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.overlay_image = original_image.copy()
            
        self.binary_mask = binary_mask.copy()
        self.contours: tuple[np.ndarray, ...] = ()
        self.features: list[dict[str, Any]] = []
        
    def extract_contours(self, min_area: float = 10.0) -> None:
        """
        Tìm viền của các vùng lỗi bằng cv2.findContours.
        
        Args:
            min_area: Diện tích tối thiểu để loại bỏ nhiễu.
        """
        contours, _ = cv2.findContours(
            self.binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = tuple(c for c in contours if cv2.contourArea(c) >= min_area)
        
    def compute_features(self, filename: str, branch: str) -> None:
        """
        Tính toán Area, Perimeter, Eccentricity cho từng vùng lỗi.
        
        Args:
            filename: Tên file đang xử lý.
            branch: Nhánh thuật toán (Canny hoặc Morphological).
        """
        for i, contour in enumerate(self.contours):
            moments = cv2.moments(contour)
            m00 = moments["m00"]
            
            if m00 == 0:
                continue
                
            area = m00
            perimeter = cv2.arcLength(contour, True)
            
            # Central moments theo slide 36
            mu20 = moments["mu20"]
            mu02 = moments["mu02"]
            mu11 = moments["mu11"]
            
            # Ma trận hiệp phương sai
            covariance_matrix = np.array([
                [mu20, mu11],
                [mu11, mu02]
            ])
            
            # Tính toán Eigenvalues
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            eigenvalues = sorted(eigenvalues, reverse=True)
            
            lambda1 = float(eigenvalues[0])
            lambda2 = float(eigenvalues[1])
            
            if lambda1 > 0:
                eccentricity = np.sqrt(1 - (lambda2 / lambda1))
            else:
                eccentricity = 0.0
                
            self.features.append({
                "filename": filename,
                "branch": branch,
                "region_id": i + 1,
                "area": area,
                "perimeter": perimeter,
                "eccentricity": eccentricity
            })
            
    def draw_overlay(self) -> np.ndarray:
        """
        Vẽ đè contours, bounding box và hiển thị thông số.
        
        Returns:
            Ảnh overlay BGR.
        """
        for i, contour in enumerate(self.contours):
            feature = next((f for f in self.features if f["region_id"] == i + 1), None)
            
            # Vẽ contour (Green)
            cv2.drawContours(self.overlay_image, [contour], -1, (0, 255, 0), 2)
            
            # Vẽ Bounding Box (Red)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.overlay_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            if feature:
                # Text: Area và Perimeter
                text = f"A:{feature['area']:.1f} P:{feature['perimeter']:.1f}"
                cv2.putText(
                    self.overlay_image, 
                    text, 
                    (x, max(y - 5, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
                
        return self.overlay_image

    def save_overlay(self, output_path: Path) -> None:
        """Lưu ảnh đã vẽ overlay."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), self.overlay_image)
        

class FeatureExporter:
    """Class hỗ trợ xuất dữ liệu ra file CSV."""
    
    @staticmethod
    def export_to_csv(features: list[dict[str, Any]], output_csv: Path) -> None:
        """
        Ghi nối (append) các feature vào file CSV.
        
        Args:
            features: Danh sách dictionary chứa thông số.
            output_csv: Đường dẫn file CSV đích.
        """
        if not features:
            return
            
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_csv.exists()
        
        fieldnames = ["filename", "branch", "region_id", "area", "perimeter", "eccentricity"]
        
        with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(features)
