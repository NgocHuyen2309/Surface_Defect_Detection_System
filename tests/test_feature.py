import cv2
import numpy as np
from pathlib import Path
from src.feature_extraction import RegionFeatureExtractor, FeatureExporter

def test():
    # 1. Tạo một ảnh trắng và vẽ 1 hình chữ nhật và 1 hình tròn (giả lập 2 vùng lỗi)
    img_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
    img_bgr[:] = (200, 200, 200) # Nền xám nhạt
    
    mask = np.zeros((300, 300), dtype=np.uint8)
    
    # Lỗi 1: Hình chữ nhật
    cv2.rectangle(mask, (50, 50), (100, 150), 255, -1)
    
    # Lỗi 2: Hình tròn
    cv2.circle(mask, (200, 200), 30, 255, -1)
    
    # Lỗi 3: Đường line xiên mảnh (Eccentricity cao)
    cv2.line(mask, (20, 250), (120, 280), 255, 5)
    
    print("Khởi tạo Extractor...")
    extractor = RegionFeatureExtractor(img_bgr, mask)
    
    print("Trích xuất contours...")
    extractor.extract_contours(min_area=10.0)
    
    print("Tính toán features...")
    extractor.compute_features(filename="test_image.png", branch="morphological")
    
    for f in extractor.features:
        print(f"Lỗi {f['region_id']}: Area={f['area']}, Perimeter={f['perimeter']:.2f}, Eccentricity={f['eccentricity']:.4f}")
        
    print("Vẽ overlay...")
    overlay = extractor.draw_overlay()
    
    output_dir = Path("data/processed/features_overlay")
    extractor.save_overlay(output_dir / "test_image_overlay.png")
    print("Đã lưu ảnh overlay.")
    
    print("Xuất CSV...")
    csv_path = Path("data/processed/features.csv")
    FeatureExporter.export_to_csv(extractor.features, csv_path)
    print("Đã xuất CSV.")

if __name__ == "__main__":
    test()
