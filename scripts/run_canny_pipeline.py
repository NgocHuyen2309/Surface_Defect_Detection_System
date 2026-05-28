"""Chạy hàng loạt pipeline tiền xử lý và phát hiện biên Canny."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from canny_edge import run_canny_pipeline
from preprocessing import find_images


def infer_label(image_path: Path, raw_dir: Path) -> str:
    """Lấy nhãn ảnh từ thư mục con đầu tiên trong data/raw."""
    relative_parts = image_path.relative_to(raw_dir).parts
    if len(relative_parts) >= 2:
        return relative_parts[0]
    return "unlabeled"


def build_parser() -> argparse.ArgumentParser:
    """Khai báo các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chạy pipeline tiền xử lý và Canny")
    parser.add_argument("--input", default="data/raw", help="Thư mục dataset đầu vào")
    parser.add_argument("--output", default="data/processed/canny", help="Thư mục lưu kết quả")
    parser.add_argument("--median-kernel", type=int, default=5, help="Kích thước kernel Median")
    parser.add_argument("--gaussian-kernel", type=int, default=5, help="Kích thước kernel Gaussian")
    parser.add_argument("--sigma", type=float, default=1.4, help="Độ lệch chuẩn Gaussian")
    parser.add_argument("--low-ratio", type=float, default=0.05, help="Tỷ lệ ngưỡng thấp")
    parser.add_argument("--high-ratio", type=float, default=0.15, help="Tỷ lệ ngưỡng cao")
    parser.add_argument("--invert-otsu", action="store_true", help="Đảo nền/vật thể khi dùng Otsu")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize theo chiều rộng, nhập 0 để bỏ qua")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh chạy thử, nhập 0 để chạy hết")
    return parser


def main() -> None:
    """Chạy pipeline cho toàn bộ ảnh trong thư mục đầu vào."""
    args = build_parser().parse_args()
    raw_dir = (PROJECT_ROOT / args.input).resolve()
    output_dir = (PROJECT_ROOT / args.output).resolve()
    resize_width = args.resize_width if args.resize_width > 0 else None

    image_paths = list(find_images(raw_dir))
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print(f"Không tìm thấy ảnh trong: {raw_dir}")
        print("Cấu trúc đề xuất: data/raw/<ten_lop>/<anh>")
        return

    start_time = perf_counter()
    print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")

    # Xử lý từng ảnh và lưu kết quả theo từng lớp
    for index, image_path in enumerate(image_paths, start=1):
        label = infer_label(image_path, raw_dir)
        image_output_dir = output_dir / label / image_path.stem
        run_canny_pipeline(
            image_path=image_path,
            output_dir=image_output_dir,
            median_kernel=args.median_kernel,
            gaussian_kernel=args.gaussian_kernel,
            sigma=args.sigma,
            low_ratio=args.low_ratio,
            high_ratio=args.high_ratio,
            invert_otsu=args.invert_otsu,
            resize_width=resize_width,
        )
        relative_path = image_path.relative_to(raw_dir)
        print(f"[{index}/{len(image_paths)}] {relative_path} -> {image_output_dir}")

    elapsed = perf_counter() - start_time
    average_time = elapsed / len(image_paths)
    print(f"Hoàn tất. Tổng thời gian: {elapsed:.3f}s | Trung bình: {average_time:.3f}s/ảnh")


if __name__ == "__main__":
    main()
