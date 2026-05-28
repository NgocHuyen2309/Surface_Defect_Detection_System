"""Chuẩn hóa dataset lỗi bề mặt vải về cấu trúc data/raw/<ten_lop>/<anh>."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_KEYWORDS = {
    "mask",
    "masks",
    "annotation",
    "annotations",
    "groundtruth",
    "ground_truth",
    "label",
    "labels",
    "segmentation",
}

LABEL_RULES = [
    (
        "defect_free",
        (
            "defect_free",
            "defect-free",
            "defect free",
            "non_defect",
            "non-defect",
            "non defect",
            "no_defect",
            "good",
            "normal",
            "healthy",
        ),
    ),
    ("hole", ("hole", "holes")),
    ("horizontal", ("horizontal", "horiz")),
    ("vertical", ("vertical", "verticle", "vert")),
    ("lines", ("line", "lines", "scratch", "scratches")),
    ("stain", ("stain", "stains", "ink", "oil", "dirt")),
]


@dataclass(frozen=True)
class PreparedItem:
    """Thông tin của một ảnh sau khi chuẩn hóa."""

    source_path: Path
    destination_path: Path
    label: str


def normalize_text(value: str) -> str:
    """Chuẩn hóa chuỗi để dò nhãn ổn định hơn."""
    lowered = value.lower()
    for char in ["_", "-", ".", "(", ")", "[", "]"]:
        lowered = lowered.replace(char, " ")
    return " ".join(lowered.split())


def is_image_file(path: Path) -> bool:
    """Kiểm tra file có phải ảnh được hỗ trợ hay không."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def contains_mask_keyword(path: Path) -> bool:
    """Nhận diện file mask hoặc annotation để bỏ qua khi cần."""
    normalized_parts = {normalize_text(part) for part in path.parts}
    normalized_joined = normalize_text(" ".join(path.parts))
    return any(keyword in normalized_parts or keyword in normalized_joined for keyword in MASK_KEYWORDS)


def infer_label(path: Path) -> str:
    """Suy luận nhãn ảnh từ tên file và tên thư mục."""
    haystack = normalize_text(" ".join(path.parts))
    for label, keywords in LABEL_RULES:
        for keyword in keywords:
            if normalize_text(keyword) in haystack:
                return label
    return "unlabeled"


def stable_suffix(path: Path) -> str:
    """Tạo hậu tố ngắn để tránh trùng tên file."""
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return digest[:8]


def iter_source_images(source_dir: Path, skip_masks: bool = True) -> Iterable[Path]:
    """Duyệt ảnh từ thư mục dataset gốc."""
    for path in sorted(source_dir.rglob("*")):
        if not is_image_file(path):
            continue
        if skip_masks and contains_mask_keyword(path):
            continue
        yield path


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    copy_files: bool,
    skip_masks: bool,
    dry_run: bool,
    max_per_class: int | None,
) -> list[PreparedItem]:
    """Sao chép hoặc tạo liên kết ảnh vào thư mục data/raw."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {source_dir}")

    class_counts: dict[str, int] = {}
    prepared_items: list[PreparedItem] = []

    # Gom ảnh theo nhãn đã suy luận
    for source_path in iter_source_images(source_dir, skip_masks=skip_masks):
        label = infer_label(source_path)
        current_count = class_counts.get(label, 0)
        if max_per_class is not None and current_count >= max_per_class:
            continue

        destination_name = f"{source_path.stem}_{stable_suffix(source_path)}{source_path.suffix.lower()}"
        destination_path = output_dir / label / destination_name
        prepared_items.append(PreparedItem(source_path, destination_path, label))
        class_counts[label] = current_count + 1

        if dry_run:
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if copy_files:
            shutil.copy2(source_path, destination_path)
        else:
            if destination_path.exists() or destination_path.is_symlink():
                destination_path.unlink()
            destination_path.symlink_to(source_path.resolve())

    return prepared_items


def write_manifest(items: list[PreparedItem], output_dir: Path, dry_run: bool) -> None:
    """Ghi file manifest mô tả nguồn gốc từng ảnh."""
    if dry_run:
        return

    manifest_path = output_dir / "dataset_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["label", "destination_path", "source_path"])
        for item in items:
            writer.writerow([item.label, item.destination_path.as_posix(), item.source_path.as_posix()])


def print_summary(items: list[PreparedItem]) -> None:
    """In nhanh số lượng ảnh theo từng lớp."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.label] = counts.get(item.label, 0) + 1

    print("Tóm tắt dataset sau chuẩn hóa:")
    for label, count in sorted(counts.items()):
        print(f"  - {label}: {count}")
    print(f"Tổng cộng: {len(items)} ảnh")


def build_parser() -> argparse.ArgumentParser:
    """Khai báo các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chuẩn hóa dataset lỗi vải về data/raw")
    parser.add_argument("--source", required=True, help="Thư mục dataset đã giải nén")
    parser.add_argument("--output", default="data/raw", help="Thư mục đầu ra trong repo")
    parser.add_argument("--copy", action="store_true", help="Sao chép file thay vì tạo symlink")
    parser.add_argument("--include-masks", action="store_true", help="Giữ lại mask/annotation nếu có")
    parser.add_argument("--dry-run", action="store_true", help="Xem trước thao tác, không ghi file")
    parser.add_argument("--max-per-class", type=int, default=0, help="Giới hạn số ảnh mỗi lớp, nhập 0 để bỏ qua")
    return parser


def main() -> None:
    """Chạy quá trình chuẩn hóa dataset."""
    args = build_parser().parse_args()
    source_dir = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output).resolve()
    max_per_class = args.max_per_class if args.max_per_class > 0 else None

    items = prepare_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        copy_files=args.copy,
        skip_masks=not args.include_masks,
        dry_run=args.dry_run,
        max_per_class=max_per_class,
    )
    write_manifest(items, output_dir, dry_run=args.dry_run)
    print_summary(items)

    if any(item.label == "unlabeled" for item in items):
        print("\nCó ảnh nằm trong lớp unlabeled. Cần kiểm tra lại tên thư mục hoặc tên file.")


if __name__ == "__main__":
    main()
