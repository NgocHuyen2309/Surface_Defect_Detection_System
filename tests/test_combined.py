import cv2
import numpy as np
from pathlib import Path

def test_combined():
    # Thử 1 ảnh stain (dark) và 1 ảnh hole (bright)
    stain_path = Path(r"F:\ComputerVision_Final\ImagePA_Final_Project\data\raw\train\stain\100_8e1833d2.jpg")
    hole_path = Path(r"F:\ComputerVision_Final\ImagePA_Final_Project\data\raw\train\hole\1_8cb9eec6.jpg")
    
    for img_path in [stain_path, hole_path]:
        print(f"Testing {img_path.name}...")
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(median, cv2.MORPH_BLACKHAT, kernel)
        
        combined = cv2.add(tophat, blackhat)
        
        # Statistical threshold on combined
        mean_val, std_val = cv2.meanStdDev(combined)
        threshold = mean_val[0][0] + 4.0 * std_val[0][0]
        _, binary = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)
        
        # Count non-zero
        area = np.count_nonzero(binary)
        print(f"  Tophat max: {tophat.max()}, Blackhat max: {blackhat.max()}, Combined area: {area}")
        
if __name__ == "__main__":
    test_combined()
