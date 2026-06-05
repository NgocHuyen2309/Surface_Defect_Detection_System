## 3.2. Nhánh Biên - Thuật toán Dò biên Directional Gradient (Directional Gradient Detection)

Là một đối trọng với bộ lọc Hình thái học, thuật toán Directional Gradient được triển khai từ đầu (from scratch) nhằm bóc tách các khuyết tật dạng tuyến tính (vết xước, sợi đứt) dựa trên sự biến thiên cường độ (Gradient). Quá trình thi hành gồm các giai đoạn cốt lõi:

1. **Gaussian Smoothing & Gradient (Toán tử Sobel):** Làm mịn ảnh bằng bộ lọc Gauss, sau đó tính toán đạo hàm theo hai hướng $x, y$ để thu thập Ma trận Độ lớn (Magnitude) và Hướng (Angle) của mép lỗi.
2. **Non-Maximum Suppression (Làm mảnh biên):** Quét qua các điểm cục bộ dọc theo hướng Gradient để triệt tiêu các điểm ảnh dư thừa, chuốt mảnh đường viền về kích thước chuẩn 1-pixel.
3. **Ngưỡng kép & Hysteresis (Double Thresholding):** Phân loại biên mạnh/yếu. Các biên yếu chỉ được giữ lại nếu chúng có tính liên thông không gian với một biên mạnh, giúp loại bỏ hoàn toàn các dải nhiễu rời rạc.
4. **Lọc Hình thái học có hướng (Directional Closing):** Lượng tử hóa góc Gradient thành 3 hướng (Ngang, Dọc, Chéo). Hệ thống áp dụng phép Đóng (Closing) với các Kernel dạng đoạn thẳng (ví dụ: $15 \times 1$ cho hướng dọc) để nối liền các viền xước đứt đoạn, tối ưu hóa việc đo đạc độ dài tuyến tính cho các loại lỗi đứt chỉ (broken yarn).