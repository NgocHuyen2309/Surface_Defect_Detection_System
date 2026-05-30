import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import ImagePreprocessor

def main():
    img_path = Path(r"F:\ComputerVision_Final\ImagePA_Final_Project\data\raw\train\hole\1_8cb9eec6.jpg")
    out_dir = Path("data/processed/test_preprocessing")
    
    preprocessor = ImagePreprocessor(
        median_kernel=5,
        defect_mode="bright",
        k_std=4.0,
        resize_width=512
    )
    preprocessor.process(img_path, out_dir)
    print("Xong!")

if __name__ == "__main__":
    main()
