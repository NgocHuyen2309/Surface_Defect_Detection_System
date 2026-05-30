# [Pull Request] Feat: Tối ưu hoá Tiền xử lý kép & Kiến trúc A/B Testing Machine Learning

## 📌 Liên kết Issue
- Closes #5 (Trích xuất đặc trưng bất biến Region Moments & A/B Testing Dataflow)

## 🚀 Tính năng & Thay đổi chính
Pull Request này là một bước tiến lớn, hoàn thiện toàn bộ khối Computer Vision truyền thống để chuẩn bị dữ liệu sạch nhất cho phòng thí nghiệm Machine Learning (Task 6).

- **Tối ưu Tiền xử lý (ImagePreprocessor):**
  - Triển khai thuật toán **Cân bằng ánh sáng kép (Absolute Contrast)**: Gộp `cv2.MORPH_TOPHAT` (bắt đốm sáng) và `cv2.MORPH_BLACKHAT` (bắt đốm tối) thông qua `cv2.add`. Giúp mô hình bắt trọn mọi loại khuyết tật (Holes, Stains) mà không cần lập trình cứng (hardcode) cờ `--defect-mode`.
  - Tự động điều chỉnh Cắt ngưỡng thống kê (Statistical Thresholding) dựa trên ma trận Combined.

- **Nâng cấp Canny Edge (Nhánh B):**
  - Lượng tử hóa góc Gradient (Angle Quantization) từ bộ lọc Sobel thành 3 hướng chính: Ngang, Dọc, Chéo.
  - Thiết kế **Directional Morphological Filters** (Lọc hình thái học có hướng) với các Structuring Element đặc thù (`1x15`, `15x1`) để giữ lại các đường xước và triệt tiêu hoàn toàn hạt nhiễu.

- **Đại tu Khối Trích xuất Đặc trưng (Feature Extraction):**
  - Xóa bỏ class cũ, thay bằng kiến trúc OOP 2 nhánh rõ rệt:
    - `BlobFeatureExtractor` (Dành cho Morphological): Bóc tách `max_area`, `total_area`, `max_perimeter`, `min_eccentricity`.
    - `LineFeatureExtractor` (Dành cho Canny): Tính toán tổng độ dài viền đứt gãy Ngang (`horiz_length`), Dọc (`vert_length`), Chéo (`diag_length`).
  - Ghi đè phương thức xuất dữ liệu CSV 1-chiều (mỗi ảnh 1 dòng) độc lập ra `morph_features.csv` và `canny_features.csv`.

## 🧪 Tài liệu kiểm thử & QA 
- Các script pipeline (`scripts/run_morphological_pipeline.py`, `scripts/run_canny_pipeline.py`) đã được chạy batch processing thành công trên toàn bộ ~2400 ảnh.
- Dữ liệu CSV thu được cực kỳ sạch và sẵn sàng đưa vào Training.
- Ảnh Overlays được render đầy đủ màu (Đỏ/Xanh lam/Xanh lục) tương ứng với các hướng gãy rách (Canny) và các khối Blobs (Morph).

## ✅ Checklist nghiệm thu
- [x] Code chạy ổn định 100%, không sinh ngoại lệ runtime khi quét toàn bộ Dataset.
- [x] Tối ưu PEP8, Docstrings đầy đủ.
- [x] CSV sinh ra chính xác dạng 1-chiều (1 ảnh = 1 dòng) phục vụ ML Classification.
- [x] Đã xóa toàn bộ file `.csv` rác của các lần test cũ.

---
*Vui lòng @TechLead review và approve để chuẩn bị tiến vào Task 6 (Training SVM & Random Forest)!*
