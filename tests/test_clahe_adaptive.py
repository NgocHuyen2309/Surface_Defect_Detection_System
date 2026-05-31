import cv2
import numpy as np
from pathlib import Path

def test_mean_std(image_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    
    # Tính Mean và Std
    mean, std = cv2.meanStdDev(median)
    m = mean[0][0]
    s = std[0][0]
    
    # Đặt ngưỡng T = Mean + k * Std (vì lỗ thủng thường sáng hơn nền)
    k = 4.0
    threshold_value = m + k * s
    
    _, binary = cv2.threshold(median, threshold_value, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(str(output_dir / "04_mean_std_binary.png"), binary)
    print(f"Mean={m:.2f}, Std={s:.2f}, T={threshold_value:.2f}")

if __name__ == "__main__":
    img_path = Path(r"F:\ComputerVision_Final\ImagePA_Final_Project\data\raw\train\hole\1_8cb9eec6.jpg")
    out_dir = Path("data/processed/test_preprocessing")
    test_mean_std(img_path, out_dir)
