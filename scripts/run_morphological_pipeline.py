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


def build_output_dir(image_path: Path, raw_dir: Path, output_dir: Path) -> Path:
    relative_parent = image_path.relative_to(raw_dir).parent
    return output_dir / relative_parent / image_path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy pipeline Hình thái học (Morphological)")
    parser.add_argument("--input", default="data/raw", help="Thư mục dataset đầu vào")
    parser.add_argument("--output", default="data/processed/morphological", help="Thư mục lưu kết quả")
    parser.add_argument("--median-kernel", type=int, default=5, help="Kích thước kernel Median")
    parser.add_argument("--invert-otsu", action="store_true", help="Đảo màu khi Otsu (cần để lỗi thành màu trắng)")
    parser.add_argument("--morph-shape", choices=["rect", "cross", "ellipse"], default="ellipse", help="Hình dáng Kernel")
    parser.add_argument("--morph-size", type=int, default=5, help="Kích thước Kernel")
    parser.add_argument("--iterations", type=int, default=1, help="Số vòng lặp Cascading cho phép Closing")
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

    start_time = perf_counter()
    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý Morphological ({args.morph_shape} {args.morph_size}x{args.morph_size}, {args.iterations} loops)...")

    for index, image_path in enumerate(image_paths, start=1):
        image_output_dir = build_output_dir(image_path, raw_dir, output_dir)
        run_morphological_pipeline(
            image_path=image_path,
            output_dir=image_output_dir,
            median_kernel=args.median_kernel,
            invert_otsu=args.invert_otsu,
            resize_width=args.resize_width,
            morph_shape=args.morph_shape,
            morph_size=args.morph_size,
            iterations=args.iterations,
        )
        print(f"[{index}/{len(image_paths)}] {image_path.name} -> Done")

    elapsed = perf_counter() - start_time
    print(f"Hoàn tất. Tổng thời gian: {elapsed:.3f}s | Trung bình: {elapsed / len(image_paths):.3f}s/ảnh")


if __name__ == "__main__":
    main()