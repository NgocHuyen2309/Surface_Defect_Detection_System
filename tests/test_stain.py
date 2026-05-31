import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphological import run_morphological_pipeline
from preprocessing import find_images
from feature_extraction import BlobFeatureExtractor

def test_stains():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "train" / "stain"
    output_dir = PROJECT_ROOT / "data" / "processed" / "test_stain"
    images = list(find_images(raw_dir))[:10]
    
    for i, img_path in enumerate(images):
        print(f"Processing {img_path.name}...")
        
        # Thử với chế độ 'bright'
        out_bright = output_dir / "bright" / img_path.stem
        res_bright = run_morphological_pipeline(
            image_path=img_path,
            output_dir=out_bright,
            median_kernel=5,
            defect_mode="bright",
            k_std=4.0,
            resize_width=512,
            morph_shape="ellipse",
            morph_size=5,
            iterations=3
        )
        extractor = BlobFeatureExtractor(res_bright["gray"], res_bright["morph_closing"])
        extractor.extract_contours(min_area=10.0)
        feat_bright = extractor.compute_features(filename=img_path.name, label="stain")
        
        # Thử với chế độ 'dark'
        out_dark = output_dir / "dark" / img_path.stem
        res_dark = run_morphological_pipeline(
            image_path=img_path,
            output_dir=out_dark,
            median_kernel=5,
            defect_mode="dark",
            k_std=4.0,
            resize_width=512,
            morph_shape="ellipse",
            morph_size=5,
            iterations=3
        )
        extractor2 = BlobFeatureExtractor(res_dark["gray"], res_dark["morph_closing"])
        extractor2.extract_contours(min_area=10.0)
        feat_dark = extractor2.compute_features(filename=img_path.name, label="stain")
        
        print(f"   => Bright: max_area = {feat_bright.get('max_area', 0)}")
        print(f"   => Dark: max_area = {feat_dark.get('max_area', 0)}")

if __name__ == "__main__":
    test_stains()
