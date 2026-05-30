# [Pull Request] Feat: Xây dựng Module trích xuất đặc trưng hình học (Region Moments)

## 📌 Liên kết Issue
- Closes #5 (Trích xuất đặc trưng bất biến Region Moments)

## 🚀 Tính năng & Thay đổi chính
Pull Request này bổ sung luồng phân tích đặc trưng vùng lỗi (`Feature Extraction`), thu hẹp khoảng cách giữa luồng ảnh nhị phân đầu vào và module Machine Learning ở phía sau. 

- **Mã nguồn:** Thêm class `RegionFeatureExtractor` trong `src/feature_extraction.py` được lập trình chặt chẽ theo chuẩn OOP và PEP8.
- **Toán học Bất biến (Invariants):**
  - **Diện tích (Area):** Tính qua Raw Moment bậc không ($m_{00}$).
  - **Chu vi (Perimeter):** Đo viền với thuật toán `cv2.arcLength`.
  - **Độ lệch tâm (Eccentricity):** Được tính hoàn toàn bằng Toán học Đại số Tuyến tính thay vì Bounding Box truyền thống. Thiết lập ma trận Covariance từ các Central Moments ($\mu_{20}, \mu_{02}, \mu_{11}$), giải phương trình Eigenvalues ($\lambda_1, \lambda_2$) và áp dụng công thức $\epsilon = \sqrt{1 - \frac{\lambda_2}{\lambda_1}}$ theo đúng **Slide 36**.
- **Đầu ra Dữ liệu:** 
  - Khởi tạo `FeatureExporter` giúp gom (append) kết quả tính toán thành file `data/processed/features.csv`.
  - Tạo ảnh minh họa overlay trực quan: viền contour xanh, bounding box đỏ, text vàng chỉ định rõ `A` (Area) và `P` (Perimeter) cho từng lỗi.

## 🧪 Tài liệu kiểm thử & QA 
Đã bổ sung 2 tài liệu cực kỳ quan trọng cho Tech Lead tại thư mục `docs/`:
1. `docs/report_sections_feature_extraction.md`: Lý thuyết giải thích việc áp dụng ma trận Central Moments và sơ đồ Dataflow (có thể bê thẳng vào báo cáo Word).
2. `docs/testing_feature_extraction.md`: Hướng dẫn chạy Mock Test trên ảnh giả lập (hình tròn có Eccentricity = 0) và Integration Test trên dataset vải thật.

> **💡 Note for Reviewer:** Khi chạy test thực tế trên ảnh lỗi thật, bộ trích xuất trên luồng **Morphological** chạy hoàn hảo (tạo các khối solid blobs lớn). Tuy nhiên, trên luồng **Canny**, hệ thống báo tìm thấy 0 vùng lỗi do đặc tính của Canny chỉ tạo đường viền hở mảnh 1-pixel (Area xấp xỉ bằng 0 bị bộ lọc gạt bỏ). Điều này hoàn toàn đúng về mặt học thuật Computer Vision, chứng minh lý do dự án bắt buộc phải có nhánh Morphological để trợ lực cho Machine Learning!

## ✅ Checklist nghiệm thu
- [x] Code chạy không sinh ra ngoại lệ runtime.
- [x] Đã sử dụng Docstrings và Type Hints.
- [x] Không lưu dataset nặng (Raw data) vào lịch sử Commit.
- [x] Tệp `features.csv` được sinh ra đúng cấu trúc bảng.

---
*Vui lòng @TechLead review và approve để merge vào `main`!*
