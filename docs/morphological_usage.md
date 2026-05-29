# Hướng dẫn chạy pipeline Hình thái học (Morphological)

## 1. Yêu cầu tiên quyết (Quan trọng)
Trước khi chạy nhánh xử lý hình thái học, hệ thống cần có sẵn tập dữ liệu đã chia `train/test`. Bạn phải chạy script chuẩn hóa dataset của Trường An trước:
    
    python scripts/prepare_fabric_dataset.py --source "ĐƯỜNG_DẪN_TỚI_DATASET_GỐC" --output data/raw --copy

## 2. Mục đích
Thực thi Nhánh xử lý thứ 2 trong hệ thống (Morphological Processing) để bóc tách hình khối khuyết tật, làm đối trọng so sánh chéo với nhánh Canny.

## 3. Cách chạy script
**Lệnh tối ưu nhất (Khuyên dùng):** Sử dụng cờ `--invert-otsu` để lỗi sáng lên, Kernel hình Ellipse $5\times5$, kết hợp kỹ thuật Cascading (3 vòng lặp) để lấp kín các khe nứt rách lớn:

    python scripts/run_morphological_pipeline.py --input data/raw --output data/processed/morphological --resize-width 512 --invert-otsu --morph-shape ellipse --morph-size 5 --iterations 3

Thay đổi hình dáng và kích thước Structuring Element (ví dụ: hình chữ nhật 7x7, 1 vòng lặp):

    python scripts/run_morphological_pipeline.py --input data/raw --output data/processed/morphological --morph-shape rect --morph-size 7 --iterations 1

## 4. Cấu trúc đầu ra
Mỗi ảnh sẽ sinh ra các file trung gian nằm tại `data/processed/morphological/<split>/<ten_lop>/<ten_anh>/`:

* `01_grayscale.png`, `01b_histogram.png`, `02_median.png`, `03_otsu_binary.png` (Kế thừa từ Tiền xử lý chung)
* `04_morph_opening.png` (Ảnh đã khử nhiễu nền)
* `05_morph_closing.png` (Ảnh kết quả cuối cùng: đã lấp kín lỗ thủng)