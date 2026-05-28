"""Tổng hợp thống kê dataset trong data/raw để đưa vào báo cáo."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(raw_dir: Path) -> Iterable[Path]:
    """Duyệt toàn bộ ảnh trong thư mục data/raw."""
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def infer_label(path: Path, raw_dir: Path) -> str:
    """Lấy nhãn ảnh từ thư mục con đầu tiên."""
    relative_parts = path.relative_to(raw_dir).parts
    if len(relative_parts) >= 2:
        return relative_parts[0]
    return "unlabeled"


def image_shape(path: Path) -> tuple[int, int] | None:
    """Đọc kích thước ảnh, trả về None nếu ảnh lỗi."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    height, width = image.shape[:2]
    return height, width


def summarize(raw_dir: Path) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Tính thống kê theo lớp, định dạng file và kích thước ảnh."""
    class_counts: dict[str, int] = defaultdict(int)
    extension_counts: dict[str, int] = defaultdict(int)
    resolution_counts: dict[str, int] = defaultdict(int)

    # Đếm số lượng ảnh theo từng nhóm thông tin
    for path in iter_images(raw_dir):
        label = infer_label(path, raw_dir)
        class_counts[label] += 1
        extension_counts[path.suffix.lower()] += 1
        shape = image_shape(path)
        if shape is not None:
            height, width = shape
            resolution_counts[f"{width}x{height}"] += 1
        else:
            resolution_counts["decode_failed"] += 1

    return dict(class_counts), dict(extension_counts), dict(resolution_counts)


def write_csv(counts: dict[str, int], output_path: Path, key_name: str) -> None:
    """Ghi bảng thống kê ra file CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([key_name, "count"])
        for key, value in sorted(counts.items()):
            writer.writerow([key, value])


def write_markdown(
    class_counts: dict[str, int],
    extension_counts: dict[str, int],
    resolution_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Ghi thống kê dataset dạng Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(class_counts.values())

    lines = [
        "# Tóm tắt tập dữ liệu thực nghiệm",
        "",
        f"- Tổng số ảnh: **{total}**",
        f"- Số lớp: **{len(class_counts)}**",
        "- Thư mục đầu vào: `data/raw/`",
        "- Thư mục đầu ra nhánh Canny: `data/processed/canny/`",
        "",
        "## Phân bố lớp",
        "",
        "| Lớp | Số lượng |",
        "|---|---:|",
    ]
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
    """Chạy thống kê dataset."""
    args = build_parser().parse_args()
    raw_dir = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    class_counts, extension_counts, resolution_counts = summarize(raw_dir)
    write_csv(class_counts, output_dir / "class_distribution.csv", "class")
    write_csv(extension_counts, output_dir / "image_formats.csv", "extension")
    write_csv(resolution_counts, output_dir / "image_resolutions.csv", "resolution")
    write_markdown(class_counts, extension_counts, resolution_counts, output_dir / "dataset_summary.md")

    print(f"Đã ghi thống kê dataset vào: {output_dir}")


if __name__ == "__main__":
    main()
