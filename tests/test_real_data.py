import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from morphological import run_morphological_pipeline
from directional_gradient import run_directional_gradient_pipeline
from feature_extraction import BlobFeatureExtractor, LineFeatureExtractor

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
        output_dir=Path("data/processed/morphological_test")
    )
    
    extractor_morph = BlobFeatureExtractor(results_morph["gray"], results_morph["morph_closing"])
    extractor_morph.extract_contours(min_area=50.0)
    extractor_morph.compute_features(filename=image_path.name, label="hole")
    
    print(f"Morphological - Kết quả: {extractor_morph.feature_dict}")
    extractor_morph.save_overlay(Path("data/processed/features_overlay") / f"{image_path.stem}_morph_overlay.png")

    # --- NHÁNH 2: DIRECTIONAL GRADIENT ---
    print("\n--- TEST NHÁNH DIRECTIONAL GRADIENT ---")
    results_directional = run_directional_gradient_pipeline(
        image_path=image_path,
        output_dir=Path("data/processed/directional_test")
    )
    
    extractor_directional = LineFeatureExtractor(
        results_directional["gray"], 
        results_directional["horiz_mask"],
        results_directional["vert_mask"],
        results_directional["diag_mask"]
    )
    extractor_directional.compute_features(filename=image_path.name, label="hole")
    
    print(f"Directional - Kết quả: {extractor_directional.feature_dict}")
    extractor_directional.save_overlay(Path("data/processed/features_overlay") / f"{image_path.stem}_directional_overlay.png")
    
    print(f"\nHoàn tất test script.")

if __name__ == "__main__":
    main()
