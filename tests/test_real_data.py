import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from morphological import run_morphological_pipeline
from canny_edge import run_canny_pipeline
from feature_extraction import RegionFeatureExtractor, FeatureExporter

def main():
    # Chọn ảnh lỗi từ dataset thật
    image_paths = list(Path("data/raw/train/hole").rglob("*.*"))
    if not image_paths:
        print("Không tìm thấy ảnh!")
        return
        
    image_path = image_paths[0]
    print(f"Đang kiểm thử trên ảnh: {image_path}")
    
    # --- NHÁNH 1: MORPHOLOGICAL ---
    print("\n--- TEST NHÁNH MORPHOLOGICAL ---")
    results_morph = run_morphological_pipeline(
        image_path=image_path,
        output_dir=Path("data/processed/morphological_test"),
        morph_shape="ellipse",
        morph_size=5,
        iterations=3
    )
    
    extractor_morph = RegionFeatureExtractor(results_morph["gray"], results_morph["morph_closing"])
    extractor_morph.extract_contours(min_area=50.0)
    extractor_morph.compute_features(filename=image_path.name, branch="morphological")
    
    print(f"Morphological - Tìm thấy {len(extractor_morph.features)} vùng lỗi:")
    for f in extractor_morph.features:
        print(f" - Lỗi {f['region_id']}: Area={f['area']}, Perimeter={f['perimeter']:.2f}, Eccentricity={f['eccentricity']:.4f}")
        
    extractor_morph.save_overlay(Path("data/processed/features_overlay") / f"{image_path.stem}_morph_overlay.png")

    # --- NHÁNH 2: CANNY EDGE ---
    print("\n--- TEST NHÁNH CANNY EDGE ---")
    results_canny = run_canny_pipeline(
        image_path=image_path,
        output_dir=Path("data/processed/canny_test"),
        median_kernel=5,
        gaussian_kernel=5,
        sigma=1.4,
        low_ratio=0.05,
        high_ratio=0.15,
        invert_otsu=False
    )
    
    extractor_canny = RegionFeatureExtractor(results_canny["gray"], results_canny["canny_edges"])
    extractor_canny.extract_contours(min_area=50.0)
    extractor_canny.compute_features(filename=image_path.name, branch="canny")
    
    print(f"Canny - Tìm thấy {len(extractor_canny.features)} vùng lỗi:")
    for f in extractor_canny.features:
        print(f" - Lỗi {f['region_id']}: Area={f['area']}, Perimeter={f['perimeter']:.2f}, Eccentricity={f['eccentricity']:.4f}")
        
    extractor_canny.save_overlay(Path("data/processed/features_overlay") / f"{image_path.stem}_canny_overlay.png")
    
    # --- XUẤT CSV ---
    csv_path = Path("data/processed/features.csv")
    FeatureExporter.export_to_csv(extractor_morph.features + extractor_canny.features, csv_path)
    print(f"\nĐã lưu CSV tại: {csv_path}")

if __name__ == "__main__":
    main()
