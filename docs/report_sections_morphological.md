## 3.3. Nhánh Khối - Bộ lọc Toán học Hình thái học (Morphological Processing)

Bên cạnh nhánh Canny tập trung vào đường biên, hệ thống thiết kế nhánh rẽ thứ hai áp dụng các toán tử Hình thái học (Morphological) để giải quyết bài toán phát hiện lỗi. Phương pháp này dựa trên lý thuyết tập hợp (Set Theory), đặc biệt hiệu quả trong việc bóc tách các "khối" lỗi có diện tích lớn nhưng bị đứt gãy hoặc nhiễu cục bộ bên trong.

**Kỹ thuật Đảo cực phân đoạn (Polarity Inversion):**
Trong môi trường thực nghiệm, các khuyết tật (như lỗ thủng) thường hấp thụ ánh sáng và có cường độ pixel tối hơn nền vải. Thuật toán Otsu thông thường sẽ nhị phân hóa vật thể tối thành màu đen (Background) và nền sáng thành màu trắng (Foreground). Tuy nhiên, các toán tử Hình thái học yêu cầu đối tượng mục tiêu phải mang giá trị Foreground. Do đó, hệ thống tích hợp kỹ thuật đảo cực nhị phân (Invert Otsu) để đảm bảo khuyết tật luôn là khối màu trắng trên phông nền đen trước khi xử lý.

**Phần tử Cấu trúc và Chuỗi lặp (Cascading):**
Yếu tố cốt lõi của nhánh này là phần tử cấu trúc (Structuring Element - SE). Hệ thống hỗ trợ 3 dạng: Hình chữ nhật (Rect), Hình chữ thập (Cross), và Hình elip (Ellipse). Dạng Ellipse kích thước $5 \times 5$ thường được ưu tiên để duy trì độ cong tự nhiên của vết thủng trên vải dệt thay vì làm vuông vức chúng.

Quy trình xử lý thực thi tuần tự hai toán tử phi tuyến tính:
1. **Phép Mở (Opening - Khử nhiễu):** Là sự kết hợp giữa toán tử Co (Erosion) theo sau bởi toán tử Giãn (Dilation). Bản chất toán học của Opening là loại bỏ hoàn toàn các đối tượng có kích thước không chứa nổi phần tử cấu trúc. Các điểm nhiễu chấm trắng (do vân vải sờn, nhiễu sáng) nằm lảng vảng ngoài vùng khuyết tật sẽ bị triệt tiêu hoàn toàn mà không làm co hẹp kích thước tổng thể của vết xước chính.
2. **Phép Đóng (Closing - Lấp khoảng trống bằng Cascading):** Là sự kết hợp giữa Dilation theo sau bởi Erosion. Đặc tính của Closing là lấp đầy các khoảng trống tối màu nằm lọt thỏm bên trong đối tượng sáng. Với các vết rách lớn hoặc xơ xác rời rạc, hệ thống áp dụng kỹ thuật lặp **Cascading (Chuỗi Dilation và Erosion liên tiếp)** với số vòng lặp (iterations) được tinh chỉnh (thường là 3 vòng lặp). Kỹ thuật này ép các cụm pixel đứt gãy ở mép rách giãn nở và kết dính chặt vào nhau, tạo thành một khối liên thông đặc ruột hoàn hảo, đảm bảo tính nguyên vẹn cho thuật toán đo Diện tích (Area) và Chu vi (Perimeter) ở Phase tiếp theo.

Các ảnh minh họa so sánh tác động của kernel ảnh:
- `04_morph_opening.png` (Trạng thái đã khử sạch nhiễu nền)
- `05_morph_closing.png` (Trạng thái đã lấp kín lỗ thủng qua Cascading)