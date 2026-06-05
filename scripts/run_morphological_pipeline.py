"""Chạy hàng loạt pipeline Morphological."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from morphological import run_morphological_pipeline
from preprocessing import find_images
from feature_extraction import BlobFeatureExtractor, FeatureExporter


def build_output_dir(image_path: Path, raw_dir: Path, output_dir: Path) -> Path:
    relative_parent = image_path.relative_to(raw_dir).parent
    return output_dir / relative_parent / image_path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy pipeline Hình thái học (Morphological)")
    parser.add_argument("--input", default="data/raw", help="Thư mục dataset đầu vào")
    parser.add_argument("--output", default="data/processed/morphological", help="Thư mục lưu kết quả")
    parser.add_argument("--median-kernel", type=int, default=7, help="Kích thước kernel Median")
    parser.add_argument("--tophat-kernel", type=int, default=21, help="Kích thước kernel TopHat")
    parser.add_argument("--defect-mode", choices=["bright", "dark", "both"], default="both", help="Loại khuyết tật")
    parser.add_argument("--k-std", type=float, default=4.0, help="Hệ số K cho Statistical Threshold")
    parser.add_argument("--morph-shape", choices=["rect", "cross", "ellipse"], default="ellipse", help="Hình dáng Kernel")
    parser.add_argument("--open-size", type=int, default=3, help="Kích thước Kernel cho Phép Mở")
    parser.add_argument("--close-size", type=int, default=5, help="Kích thước Kernel cho Phép Đóng")
    parser.add_argument("--iterations", type=int, default=3, help="Số vòng lặp Cascading cho phép Closing")
    parser.add_argument("--resize-width", type=int, default=512, help="Resize ảnh")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh chạy thử")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = (PROJECT_ROOT / args.input).resolve()
    output_dir = (PROJECT_ROOT / args.output).resolve()

    image_paths = list(find_images(raw_dir))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    all_features = []

    start_time = perf_counter()
    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu Morphological (Open {args.open_size}x{args.open_size}, Close {args.close_size}x{args.close_size}, {args.iterations} loops)...")

    for index, image_path in enumerate(image_paths, start=1):
        image_output_dir = build_output_dir(image_path, raw_dir, output_dir)
        results = run_morphological_pipeline(
            image_path=image_path,
            output_dir=image_output_dir,
            median_kernel=args.median_kernel,
            defect_mode=args.defect_mode,
            k_std=args.k_std,
            tophat_kernel=args.tophat_kernel,
            resize_width=args.resize_width,
            morph_shape=args.morph_shape,
            open_size=args.open_size,
            close_size=args.close_size,
            iterations=args.iterations,
        )
        
        # Gọi Feature Extractor với min_area=50.0 để loại bỏ rác li ti (Vì đã dùng k_std nhạy bén)
        label = image_path.parent.name
        extractor = BlobFeatureExtractor(results["gray"], results["morph_closing"])
        extractor.extract_contours(min_area=50.0)
        feature = extractor.compute_features(filename=image_path.name, label=label)
        all_features.append(feature)
        
        extractor.save_overlay(image_output_dir / "06_morph_overlay.png")
        
        if index % 100 == 0:
            print(f"[{index}/{len(image_paths)}] Đang xử lý...")

    csv_path = PROJECT_ROOT / "data" / "processed" / "morph_features.csv"
    FeatureExporter.export_morph_csv(all_features, csv_path)

    elapsed = perf_counter() - start_time
    print(f"Hoàn tất. Tổng thời gian: {elapsed:.3f}s | Trung bình: {elapsed / len(image_paths):.3f}s/ảnh")


if __name__ == "__main__":
    main()