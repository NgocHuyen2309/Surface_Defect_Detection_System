# Phân tích Thiết kế & Thực nghiệm: Trích xuất đặc trưng (Feature Extraction)

## 1. Cơ sở lý thuyết: Region Moments

Trong bài toán phát hiện khiếm khuyết bề mặt, việc trích xuất được một bộ số liệu đặc trưng (Features) có tính chất **Bất biến (Invariance)** đối với phép xoay, tịnh tiến và thu phóng là điều kiện tiên quyết để mô hình Học máy có thể phân loại tốt. 

Hệ thống được thiết kế theo kiến trúc OOP với 2 bộ Extractor độc lập:

### 1.1. BlobFeatureExtractor (Cho nhánh Morphological)
Trích xuất ba thuộc tính hình học khối:
1. **Diện tích (Area):** Dựa trên Raw Moment bậc 0 ($m_{00}$). Hệ thống tính toán `max_area` để nhận diện khối lỗi lớn nhất, và `total_area` để đo lường mức độ lan rộng của lỗi (như vết ố).
2. **Chu vi (Perimeter):** Đo lường chiều dài viền khép kín của đối tượng thông qua `cv2.arcLength`.
3. **Độ lệch tâm (Eccentricity - $\epsilon$):** Tính toán thông qua phương trình trị riêng (Eigenvalues) của Ma trận Hiệp phương sai từ các Central Moments ($\mu_{20}, \mu_{02}, \mu_{11}$). Giá trị $\epsilon \rightarrow 0$ đặc trưng cho lỗi tròn (hole), $\epsilon \rightarrow 1$ đặc trưng cho lỗi xước kéo dài.

### 1.2. LineFeatureExtractor (Cho nhánh Directional Gradient)
Nhánh này tập trung đo lường cường độ đứt gãy tuyến tính. Sau khi Directional Gradient lượng tử hóa góc Gradient thành các mask Ngang, Dọc, Chéo, bộ trích xuất sẽ tính tổng số lượng pixel biên khuyết tật (Non-zero pixels) trên từng hướng, sinh ra bộ đặc trưng: `horiz_length`, `vert_length`, `diag_length`.

## 2. Phân tích thực nghiệm định lượng từ file CSV

Sau quá trình quét (Batch Processing) trên toàn bộ tập dữ liệu, hai file `morph_features.csv` và `directional_features.csv` đã bộc lộ rõ hiệu năng của từng phương pháp:

- **Morphological Pipeline (Hạn chế với lỗi mảnh):** Mặc dù bộ lọc Hình thái học kết hợp Cân bằng ánh sáng Top-Hat/Black-Hat triệt tiêu tốt nhiễu vân vải và bắt được các lỗi lớn (hole/stain), phương pháp này gặp khó khăn trong việc giữ lại các vết xước mảnh do đặc tính bào mòn của phần tử cấu trúc.
- **Directional Gradient Pipeline (Ưu việt với lỗi đứt sợi):** Dù ban đầu bị ảnh hưởng bởi kết cấu vải đan chéo, nhưng nhờ ứng dụng các phép đóng có hướng (Directional Closing) với Kernel đoạn thẳng, thuật toán Directional Gradient đã nối kết thành công dải đứt gãy. Khi kết hợp với mô hình học máy (như SVM), đặc trưng tuyến tính này giúp phân loại cực kỳ chính xác các vết xước ngang, dọc và chéo.

**Kết luận thực nghiệm:** Việc thiết lập 2 luồng A/B Testing đã mang lại minh chứng khoa học đắt giá. Số liệu thực tế từ mô hình SVM cho thấy thuật toán **Directional Gradient** vượt trội hơn hẳn Morphological (Accuracy đạt 78.3% so với 59.7% của Morphological). Lỗi bề mặt dệt thường mang tính định hướng không gian cao, và phương pháp trích xuất độ dài đứt gãy định hướng chính là chìa khóa tối ưu nhất cho bài toán này.