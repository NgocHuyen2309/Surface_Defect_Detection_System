# Phân tích Thiết kế & Thực nghiệm: Trích xuất đặc trưng (Feature Extraction)

## 1. Cơ sở lý thuyết: Region Moments

Trong bài toán phát hiện khiếm khuyết bề mặt, việc trích xuất được một bộ số liệu đặc trưng (Features) có tính chất **Bất biến (Invariance)** đối với phép xoay, tịnh tiến và thu phóng là điều kiện tiên quyết để mô hình Học máy có thể phân loại tốt. 

Hệ thống được thiết kế theo kiến trúc OOP với 2 bộ Extractor độc lập:

### 1.1. BlobFeatureExtractor (Cho nhánh Morphological)
Trích xuất ba thuộc tính hình học khối:
1. **Diện tích (Area):** Dựa trên Raw Moment bậc 0 ($m_{00}$). Hệ thống tính toán `max_area` để nhận diện khối lỗi lớn nhất, và `total_area` để đo lường mức độ lan rộng của lỗi (như vết ố).
2. **Chu vi (Perimeter):** Đo lường chiều dài viền khép kín của đối tượng thông qua `cv2.arcLength`.
3. **Độ lệch tâm (Eccentricity - $\epsilon$):** Tính toán thông qua phương trình trị riêng (Eigenvalues) của Ma trận Hiệp phương sai từ các Central Moments ($\mu_{20}, \mu_{02}, \mu_{11}$). Giá trị $\epsilon \rightarrow 0$ đặc trưng cho lỗi tròn (hole), $\epsilon \rightarrow 1$ đặc trưng cho lỗi xước kéo dài.

### 1.2. LineFeatureExtractor (Cho nhánh Canny Edge)
Nhánh này tập trung đo lường cường độ đứt gãy tuyến tính. Sau khi Canny lượng tử hóa góc Gradient thành các mask Ngang, Dọc, Chéo, bộ trích xuất sẽ tính tổng số lượng pixel biên khuyết tật (Non-zero pixels) trên từng hướng, sinh ra bộ đặc trưng: `horiz_length`, `vert_length`, `diag_length`.

## 2. Phân tích thực nghiệm định lượng từ file CSV

Sau quá trình quét (Batch Processing) trên toàn bộ tập dữ liệu, hai file `morph_features.csv` và `canny_features.csv` đã bộc lộ rõ hiệu năng của từng phương pháp:

- **Morphological Pipeline (Thành công rực rỡ):** Bộ lọc Hình thái học kết hợp Cân bằng ánh sáng Top-Hat/Black-Hat đã triệt tiêu hoàn toàn nhiễu vân vải. Với các ảnh `defect_free` (không lỗi), giá trị `max_area` trả về xấp xỉ 0. Với các ảnh chứa `hole` hoặc `stain`, diện tích bắt được rất lớn và chuẩn xác. Đây là bộ dữ liệu vàng, cực kỳ nhiễu thấp (Low Noise) cho mô hình Machine Learning.
- **Canny Edge Pipeline (Bị áp đảo bởi kết cấu vải):** Mặc dù đã dùng các phép đóng có hướng (Directional Closing) để hàn gắn đường xước, thuật toán Canny vẫn bắt nhầm kết cấu đan chéo của sợi vải (Fabric Weave Texture) làm viền biên. Hậu quả là ở các ảnh vải lành lặn, độ dài đứt gãy vẫn cao, sinh ra lượng Báo động giả (False Positives) lớn. 

**Kết luận thực nghiệm:** Việc thiết lập 2 luồng A/B Testing đã mang lại minh chứng khoa học đắt giá. Trên nền vật liệu có kết cấu đan xen như vải dệt, thuật toán Hình thái học (Morphological) thể hiện sự ưu việt tuyệt đối so với các toán tử dò biên Gradient cục bộ (Canny).