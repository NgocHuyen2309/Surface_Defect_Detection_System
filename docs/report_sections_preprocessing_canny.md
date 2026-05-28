# Nội dung báo cáo: Dataset, tiền xử lý và Canny

## 1.1. Đặt vấn đề và mục tiêu nghiên cứu

Trong quá trình kiểm tra chất lượng sản phẩm dệt may, việc phát hiện lỗi bề mặt bằng mắt thường thường phụ thuộc nhiều vào kinh nghiệm của người kiểm tra. Cách làm thủ công dễ bị ảnh hưởng bởi ánh sáng, góc nhìn, độ mỏi mắt và tốc độ dây chuyền sản xuất. Một số lỗi như vết xước mảnh, lỗ nhỏ, vệt bẩn hoặc đường đứt sợi có kích thước nhỏ nên rất dễ bị bỏ sót nếu chỉ quan sát trực tiếp.

Vì vậy, hệ thống được xây dựng theo hướng tự động hóa một phần quá trình phát hiện lỗi bề mặt ảnh. Ảnh đầu vào được chuẩn hóa qua các bước tiền xử lý, sau đó đi qua các nhánh xử lý ảnh truyền thống để làm nổi bật vùng nghi ngờ lỗi. Kết quả xử lý được dùng để trích xuất đặc trưng hình học và phục vụ các mô hình phân loại ở giai đoạn sau.

Trong phạm vi nhánh Canny, mục tiêu chính là làm nổi bật ranh giới của các vùng bất thường trên bề mặt ảnh. Phương pháp này đặc biệt phù hợp với những lỗi có đặc trưng biên rõ như đường nứt, vết xước, lỗ thủng hoặc vùng có sự thay đổi cường độ sáng đột ngột.

## 3.1. Tầng tiền xử lý dữ liệu ảnh chung

Tầng tiền xử lý có nhiệm vụ đưa ảnh đầu vào về dạng ổn định hơn trước khi áp dụng các thuật toán phân đoạn. Pipeline sử dụng ba bước chính: chuyển ảnh sang grayscale, khử nhiễu bằng Median Filter và nhị phân hóa bằng Otsu.

Đầu tiên, ảnh màu được chuyển sang ảnh xám để giảm số chiều dữ liệu và tập trung vào thông tin cường độ sáng. Việc này giúp các thuật toán xử lý phía sau hoạt động đơn giản hơn, đồng thời giảm chi phí tính toán so với xử lý trực tiếp trên ba kênh màu.

Sau đó, Median Filter được dùng để giảm nhiễu xung và các điểm sáng/tối bất thường. Với mỗi điểm ảnh, bộ lọc lấy trung vị của các điểm trong vùng lân cận thay vì lấy trung bình. Nhờ đó, các nhiễu dạng muối tiêu được giảm bớt nhưng ranh giới của vùng lỗi vẫn được giữ tương đối tốt.

Cuối cùng, thuật toán Otsu được dùng để tìm ngưỡng nhị phân tự động. Phương pháp này chọn ngưỡng sao cho hai nhóm điểm ảnh nền và vật thể được tách biệt tốt nhất theo phân bố histogram. Kết quả của bước này là ảnh nhị phân, hỗ trợ việc quan sát nhanh vùng nghi ngờ lỗi và làm cơ sở so sánh với kết quả của các nhánh xử lý khác.

Các ảnh trung gian cần xuất cho mục này gồm:

| Bước | File kết quả |
|---|---|
| Grayscale | `01_grayscale.png` |
| Histogram + ngưỡng Otsu | `01b_histogram.png` |
| Median Filter | `02_median.png` |
| Otsu Threshold | `03_otsu_binary.png` |

## 3.2. Nhánh biên - thuật toán phát hiện biên Canny

Canny là thuật toán phát hiện biên nhiều bước, được thiết kế để tìm các đường biên rõ ràng và hạn chế nhiễu. Trong bài toán phát hiện lỗi bề mặt, Canny giúp làm nổi bật ranh giới của các vùng có thay đổi cường độ sáng mạnh, ví dụ vết xước, lỗ thủng hoặc đường lỗi trên vật liệu.

Quy trình Canny trong hệ thống gồm bốn bước chính.

Bước đầu tiên là làm mịn ảnh bằng Gaussian Filter. Ảnh sau Median Filter vẫn có thể còn nhiễu cục bộ, vì vậy Gaussian smoothing được áp dụng để giảm dao động cường độ nhỏ trước khi tính gradient.

Bước thứ hai là tính gradient bằng toán tử Sobel. Gradient cho biết mức độ thay đổi cường độ sáng theo hai hướng ngang và dọc. Từ đó, hệ thống tính được độ lớn gradient và hướng gradient tại từng điểm ảnh.

Bước thứ ba là Non-maximum Suppression. Ở bước này, mỗi điểm ảnh được so sánh với hai điểm lân cận theo hướng gradient. Chỉ những điểm có độ lớn gradient lớn nhất cục bộ mới được giữ lại. Kết quả là đường biên được làm mảnh hơn, tránh tình trạng biên bị dày và khó phân tích.

Bước cuối cùng là Double Threshold và Hysteresis. Các điểm biên được chia thành biên mạnh, biên yếu và nền. Biên yếu chỉ được giữ lại nếu nó liên thông với biên mạnh trong vùng lân cận. Cách làm này giúp loại bỏ nhiễu biên rời rạc nhưng vẫn giữ được các đoạn biên thật có cường độ không đồng đều.

Các ảnh minh họa cần xuất cho mục này gồm:

| Bước | File kết quả |
|---|---|
| Gaussian smoothing | `04_gaussian.png` |
| Gradient magnitude | `05_gradient_magnitude.png` |
| Non-maximum suppression | `06_non_maximum_suppression.png` |
| Double threshold | `07_double_threshold.png` |
| Canny edge map | `08_canny_edges.png` |

## 6.1. Thiết lập cấu hình hệ thống và tập dữ liệu thực nghiệm

Tập dữ liệu thực nghiệm được tổ chức theo cấu trúc train/test trong `data/raw`. Tập train chiếm 80% dữ liệu, dùng để phát triển pipeline và huấn luyện mô hình. Tập test chiếm 20% dữ liệu, được giữ riêng để đánh giá cuối cùng cho các tổ hợp Canny + SVM, Canny + Random Forest, Morphological + SVM và Morphological + Random Forest.

```text
data/raw/
  train/
    defect_free/
    hole/
    horizontal/
    vertical/
    lines/
    stain/
  test/
    defect_free/
    hole/
    horizontal/
    vertical/
    lines/
    stain/
```

Môi trường xử lý ảnh sử dụng Python, OpenCV, NumPy và Matplotlib. Ảnh đầu vào đi qua các bước grayscale, Median Filter, Otsu thresholding và Canny edge detection. Kết quả trung gian được lưu vào `data/processed/canny/` theo split, theo lớp và từng ảnh. Ngoài ảnh trung gian, pipeline còn xuất `01b_histogram.png` để quan sát phân bố mức xám và vị trí ngưỡng Otsu.

Bảng cấu hình mặc định của pipeline:

| Tham số | Giá trị mặc định |
|---|---:|
| Median kernel | 5 |
| Gaussian kernel | 5 |
| Sigma | 1.4 |
| Low threshold ratio | 0.05 |
| High threshold ratio | 0.15 |
| Resize width | 512 |

Sau khi chạy `scripts/summarize_dataset.py`, số liệu thống kê được lưu trong `docs/dataset_stats/`. Các bảng này có thể đưa trực tiếp vào mục 6.1 để mô tả tổng số ảnh, tỷ lệ train/test, số ảnh theo lớp, định dạng ảnh và kích thước ảnh.

## Gợi ý nhận xét cho 6.3. Error Analysis

Nhánh Canny thường cho kết quả tốt với các lỗi có biên sắc nét và độ tương phản rõ so với nền. Tuy nhiên, với các ảnh có nhiều vân vải, nếp nhăn hoặc nhiễu ánh sáng, thuật toán có thể tạo thêm biên giả. Vì vậy, khi đánh giá định tính cần chọn thêm một số ảnh khó như ảnh bị chói sáng, nền vải nhiều texture hoặc lỗi có biên mờ để so sánh với nhánh Morphological.

Nếu ảnh có nhiều biên giả, có thể tăng `sigma`, tăng `high_ratio` hoặc resize ảnh về kích thước nhỏ hơn để giảm nhiễu chi tiết. Nếu biên lỗi bị mất quá nhiều, có thể giảm `high_ratio` hoặc giữ ảnh ở độ phân giải lớn hơn.
