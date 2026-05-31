## 3.3. Nhánh Khối - Bộ lọc Toán học Hình thái học (Morphological Processing)

Bên cạnh nhánh Canny tập trung vào đường biên, hệ thống thiết kế nhánh rẽ thứ hai áp dụng các toán tử Hình thái học (Morphological) để giải quyết bài toán phát hiện lỗi dạng đốm/khối.

**Cân bằng Ánh sáng và Ngưỡng Thống kê (Illumination Correction & Statistical Thresholding):**
Bề mặt vải dệt thực tế thường có độ rọi sáng không đồng đều. Để khắc phục, hệ thống áp dụng kỹ thuật Cân bằng ánh sáng kép: Dùng `Top-Hat` để đẩy nổi các lỗi sáng (hole) và `Black-Hat` để đẩy nổi lỗi tối (stain), sau đó kết hợp lại. Nhờ đó, phông nền vải bị ép về giá trị tối tĩnh, cho phép sử dụng Ngưỡng Thống kê động ($T = \mu \pm k\sigma$) với $k = 2.5$ để bóc tách chính xác ngay cả những vết ố mờ nhất.

**Phân tách Kernel và Chuỗi lặp (Split Kernels & Cascading):**
Điểm đột phá của thuật toán nằm ở việc không sử dụng một kích thước phần tử cấu trúc (Structuring Element) cố định:
1. **Phép Mở (Opening - Kernel nhỏ 3x3):** Được áp dụng đầu tiên nhằm cạo sạch các điểm nhiễu lấm tấm (muối tiêu) mà không làm tổn hại đến các vết dơ mỏng manh.
2. **Phép Đóng (Closing - Kernel lớn 5x5 kết hợp Cascading):** Với các vết rách lớn hoặc đứt gãy, hệ thống áp dụng kỹ thuật lặp chuỗi `Cascading` (3 vòng lặp). Kernel lớn ép các cụm pixel đứt gãy giãn nở, kết dính chặt vào nhau, tạo thành một khối liên thông đặc ruột hoàn hảo, dọn đường cho việc đo đạc Area/Perimeter ở Module Trích xuất Đặc trưng.