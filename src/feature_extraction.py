"""Trích xuất đặc trưng hình học từ ảnh phân đoạn cho 2 luồng A/B Testing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BlobFeatureExtractor:
    """Class hỗ trợ rút trích đặc trưng khối (Blobs) từ nhánh Morphological."""
    
    def __init__(self, original_image: np.ndarray, binary_mask: np.ndarray) -> None:
        if len(original_image.shape) == 2:
            self.overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.overlay_image = original_image.copy()
            
        self.binary_mask = binary_mask.copy()
        self.contours: tuple[np.ndarray, ...] = ()
        self.feature_dict: dict[str, Any] = {}
        
    def extract_contours(self, min_area: float = 10.0) -> None:
        contours, _ = cv2.findContours(
            self.binary_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = tuple(c for c in contours if cv2.contourArea(c) >= min_area)
        
    def compute_features(self, filename: str, label: str) -> dict[str, Any]:
        total_area = 0.0
        max_area = 0.0
        max_perimeter = 0.0
        min_eccentricity = 1.0 # 1.0 is a line, 0.0 is a perfect circle
        
        for contour in self.contours:
            moments = cv2.moments(contour)
            m00 = moments["m00"]
            
            if m00 == 0:
                continue
                
            area = m00
            perimeter = cv2.arcLength(contour, True)
            
            total_area += area
            if area > max_area:
                max_area = area
                max_perimeter = perimeter
            
            # Central moments
            mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
            covariance_matrix = np.array([[mu20, mu11], [mu11, mu02]])
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            eigenvalues = sorted(eigenvalues, reverse=True)
            
            lambda1, lambda2 = float(eigenvalues[0]), float(eigenvalues[1])
            
            if lambda1 > 0:
                eccentricity = np.sqrt(1 - (lambda2 / lambda1))
            else:
                eccentricity = 0.0
                
            if eccentricity < min_eccentricity:
                min_eccentricity = eccentricity
                
        self.feature_dict = {
            "filename": filename,
            "label": label,
            "max_area": max_area,
            "total_area": total_area,
            "max_perimeter": max_perimeter,
            "min_eccentricity": min_eccentricity
        }
        return self.feature_dict
        
    def draw_overlay(self) -> np.ndarray:
        for contour in self.contours:
            cv2.drawContours(self.overlay_image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.overlay_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        if self.feature_dict:
            text = f"MaxA:{self.feature_dict['max_area']:.1f} Ecc:{self.feature_dict['min_eccentricity']:.2f}"
            cv2.putText(self.overlay_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        return self.overlay_image

    def save_overlay(self, output_path: Path) -> None:
        self.draw_overlay()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), self.overlay_image)
        

class LineFeatureExtractor:
    """Class hỗ trợ rút trích đặc trưng tuyến tính (Lines) từ nhánh Canny."""
    
    def __init__(self, original_image: np.ndarray, horiz_mask: np.ndarray, vert_mask: np.ndarray, diag_mask: np.ndarray) -> None:
        if len(original_image.shape) == 2:
            self.overlay_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            self.overlay_image = original_image.copy()
            
        self.horiz_mask = horiz_mask
        self.vert_mask = vert_mask
        self.diag_mask = diag_mask
        self.feature_dict: dict[str, Any] = {}
        
    def compute_features(self, filename: str, label: str) -> dict[str, Any]:
        horiz_length = np.count_nonzero(self.horiz_mask)
        vert_length = np.count_nonzero(self.vert_mask)
        diag_length = np.count_nonzero(self.diag_mask)
        
        self.feature_dict = {
            "filename": filename,
            "label": label,
            "horiz_length": float(horiz_length),
            "vert_length": float(vert_length),
            "diag_length": float(diag_length)
        }
        return self.feature_dict
        
    def draw_overlay(self) -> np.ndarray:
        # Màu BGR: Đỏ cho ngang, Xanh lam cho dọc, Xanh lá cho chéo
        self.overlay_image[self.horiz_mask > 0] = [0, 0, 255]
        self.overlay_image[self.vert_mask > 0] = [255, 0, 0]
        self.overlay_image[self.diag_mask > 0] = [0, 255, 0]
        
        if self.feature_dict:
            text = f"H:{self.feature_dict['horiz_length']} V:{self.feature_dict['vert_length']}"
            cv2.putText(self.overlay_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        return self.overlay_image

    def save_overlay(self, output_path: Path) -> None:
        self.draw_overlay()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), self.overlay_image)


class FeatureExporter:
    """Class hỗ trợ xuất dữ liệu ra file CSV cho A/B Testing."""
    
    @staticmethod
    def export_morph_csv(features: list[dict[str, Any]], output_csv: Path) -> None:
        if not features: return
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_csv.exists()
        fieldnames = ["filename", "label", "max_area", "total_area", "max_perimeter", "min_eccentricity"]
        
        with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(features)
            
    @staticmethod
    def export_canny_csv(features: list[dict[str, Any]], output_csv: Path) -> None:
        if not features: return
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = output_csv.exists()
        fieldnames = ["filename", "label", "horiz_length", "vert_length", "diag_length"]
        
        with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(features)
