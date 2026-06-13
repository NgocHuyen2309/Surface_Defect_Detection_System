"""Tổng hợp thống kê dataset trong data/raw để đưa vào báo cáo."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "test", "val", "validation"}


def iter_images(raw_dir: Path) -> Iterable[Path]:
    """Duyệt toàn bộ ảnh trong thư mục data/raw."""
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def infer_split_label(path: Path, raw_dir: Path) -> tuple[str, str]:
    """Lấy tên split và nhãn từ cấu trúc thư mục dataset."""
    relative_parts = path.relative_to(raw_dir).parts

    # Cấu trúc chuẩn: data/raw/train/<label>/<image> hoặc data/raw/test/<label>/<image>.
    if len(relative_parts) >= 3 and relative_parts[0] in SPLIT_NAMES:
        return relative_parts[0], relative_parts[1]

    # Hỗ trợ trường hợp dataset chỉ có data/raw/<label>/<image>.
    if len(relative_parts) >= 2:
        return "all", relative_parts[0]

    return "all", "unlabeled"


def image_shape(path: Path) -> tuple[int, int] | None:
    """Đọc kích thước ảnh, trả về None nếu ảnh không đọc được."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape[:2]
    return height, width


def summarize(raw_dir: Path) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Tính thống kê theo split, lớp, định dạng file và kích thước ảnh."""
    split_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    extension_counts: dict[str, int] = defaultdict(int)
    resolution_counts: dict[str, int] = defaultdict(int)

    for path in iter_images(raw_dir):
        split_name, label = infer_split_label(path, raw_dir)
        split_counts[split_name] += 1
        class_counts[f"{split_name}/{label}"] += 1
        extension_counts[path.suffix.lower()] += 1

        # Kích thước ảnh giúp xác nhận dataset có cần resize trước khi chạy pipeline hay không.
        shape = image_shape(path)
        if shape is not None:
            height, width = shape
            resolution_counts[f"{width}x{height}"] += 1
        else:
            resolution_counts["decode_failed"] += 1

    return dict(split_counts), dict(class_counts), dict(extension_counts), dict(resolution_counts)


def write_csv(counts: dict[str, int], output_path: Path, key_name: str) -> None:
    """Ghi bảng thống kê ra file CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([key_name, "count"])
        for key, value in sorted(counts.items()):
            writer.writerow([key, value])


def write_markdown(
    split_counts: dict[str, int],
    class_counts: dict[str, int],
    extension_counts: dict[str, int],
    resolution_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Ghi thống kê dataset dạng Markdown để chèn vào báo cáo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(split_counts.values())

    lines = [
        "# Tóm tắt tập dữ liệu thực nghiệm",
        "",
        f"- Tổng số ảnh: **{total}**",
        "- Tỷ lệ chia mặc định: **train 80% / test 20%**",
        "- Thư mục đầu vào: `data/raw/`",
        "- Thư mục đầu ra nhánh Directional Gradient: `data/processed/directional/`",
        "",
        "## Phân bố train/test",
        "",
        "| Split | Số lượng |",
        "|---|---:|",
    ]
    for split_name, count in sorted(split_counts.items()):
        lines.append(f"| {split_name} | {count} |")

    lines.extend(["", "## Phân bố lớp theo split", "", "| Split/Lớp | Số lượng |", "|---|---:|"])
    for label, count in sorted(class_counts.items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "## Định dạng ảnh", "", "| Định dạng | Số lượng |", "|---|---:|"])
    for extension, count in sorted(extension_counts.items()):
        lines.append(f"| {extension} | {count} |")

    lines.extend(["", "## Kích thước ảnh", "", "| Kích thước | Số lượng |", "|---|---:|"])
    for resolution, count in sorted(resolution_counts.items()):
        lines.append(f"| {resolution} | {count} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Khai báo các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Thống kê dataset trong data/raw")
    parser.add_argument("--input", default="data/raw", help="Thư mục dataset đã chuẩn hóa")
    parser.add_argument("--output-dir", default="docs/dataset_stats", help="Thư mục lưu file thống kê")
    return parser


def main() -> None:
    """Chạy thống kê dataset và xuất CSV/Markdown."""
    args = build_parser().parse_args()
    raw_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Tạo các bảng nhỏ để dùng cho mục mô tả dataset và thực nghiệm.
    split_counts, class_counts, extension_counts, resolution_counts = summarize(raw_dir)
    write_csv(split_counts, output_dir / "split_distribution.csv", "split")
    write_csv(class_counts, output_dir / "class_distribution.csv", "split_class")
    write_csv(extension_counts, output_dir / "image_formats.csv", "extension")
    write_csv(resolution_counts, output_dir / "image_resolutions.csv", "resolution")
    write_markdown(
        split_counts,
        class_counts,
        extension_counts,
        resolution_counts,
        output_dir / "dataset_summary.md",
    )

    print(f"Đã ghi thống kê dataset vào: {output_dir}")


if __name__ == "__main__":
    main()
