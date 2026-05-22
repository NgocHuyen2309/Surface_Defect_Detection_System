# 🔍 Surface Defect Detection: Morphological & Machine Learning Approach
**Đồ án môn học: Xử lý và Phân tích ảnh** | **Giảng viên hướng dẫn: TS. Đặng N.H. Thành**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org/)
[![ReactJS](https://img.shields.io/badge/React-18.x-61DAFB.svg)](https://reactjs.org/)

## 📖 Tổng quan dự án (Project Overview)
Dự án tập trung xây dựng hệ thống tự động phát hiện và phân loại khiếm khuyết trên bề mặt sản phẩm công nghiệp (PCB, vải dệt). Hệ thống kết hợp sức mạnh của **3 kỹ thuật xử lý ảnh truyền thống** để trích xuất vùng lỗi và **1 mô hình Machine Learning** để phân loại chính xác loại lỗi.

## 🛠 Quy trình xử lý (Pipeline 3+1) & Đặc trưng Phương pháp

Hệ thống được thiết kế theo mô hình xử lý dòng dữ liệu tuần tự khép kín, ứng dụng các phương pháp toán lý và phân tích thống kê cốt lõi sau:

### Các phương pháp chính (Core Methods)

#### 1. Tiền xử lý: Phân đoạn ảnh bằng thuật toán Otsu (Otsu's Thresholding)
- **Đặc trưng phương pháp:** Đây là một thuật toán tìm ngưỡng phân đoạn nhị phân tự động không giám sát (unsupervised). Bản chất toán học của phương pháp là duyệt qua toàn bộ dải giá trị histogram để tính toán toán tử tìm một ngưỡng $T$ tối ưu hóa, sao cho phương sai giữa hai lớp nền và đối tượng (inter-class variance) đạt giá trị cực đại, hoặc phương sai nội bộ lớp (within-class variance) đạt giá trị cực tiểu. Phương pháp này hoạt động hoàn toàn dựa trên phân bố thống kê tần suất của điểm ảnh mà không phụ thuộc vào tri thức tiên nghiệm về cấu trúc không gian hình học của ảnh.

#### 2. Xử lý chính: Bộ lọc Toán học Hình thái (Mathematical Morphology Filters)
- **Đặc trưng phương pháp:** Là một nhóm các toán tử phi tuyến tính được xây dựng vững chắc trên nền tảng của lý thuyết tập hợp (Set Theory). Đặc trưng cốt lõi của phương pháp là mức độ và cấu trúc tác động lên dữ liệu ma trận được quyết định hoàn toàn bởi hình dáng hình học và kích thước hình học của phần tử cấu trúc (Structuring Element / Kernel). Phép toán Mở (Opening) mang đặc tính toán học triệt tiêu hoàn toàn các cấu trúc đối tượng có kích thước nhỏ hơn phần tử cấu trúc mà không làm dịch chuyển ranh giới tổng thể, trong khi phép toán Đóng (Closing) mang đặc tính lấp đầy các khoảng trống cấu trúc bên trong một vùng liên thông.

#### 3. Trích xuất đặc trưng: Tính toán Moment Không gian (Spatial Moments Feature Extraction)
- **Đặc trưng phương pháp:** Phương pháp lượng hóa cấu trúc không gian hình học của thực thể thành các trị số vô hướng đặc trưng độc lập. Đặc tính quan trọng nhất của phương pháp này là tính bất biến đối với các phép biến đổi hình học cơ bản. Việc sử dụng toán tử các moment trung tâm (Central Moments) đảm bảo tính bất biến với phép tịnh tiến không gian (Translation Invariance), và tỷ số hình học như Aspect Ratio hay Độ lệch tâm (Eccentricity) đảm bảo tính bất biến với phép thu phóng ma trận (Scale Invariance), giúp tiến trình nhận dạng diễn ra độc lập với vị trí không gian vật lý của đối tượng lỗi trên ảnh.

#### 4. Phân loại thực thể: Bộ phân loại Học máy (Support Vector Machine / Random Forest)
- **Đặc trưng phương pháp:** - *Đối với SVM:* Phương pháp học máy có giám sát cấu trúc hóa việc tối đa hóa lề (margin maximization) giữa các siêu phẳng phân lớp dữ liệu bằng cách tìm kiếm một siêu phẳng (hyperplane) phân cách tối ưu hình học trong không gian đặc trưng đa chiều (Hyper-dimensional feature space).
  - *Đối với Random Forest:* Thuật toán sử dụng cơ chế học tập tập hợp (Ensemble Learning) dựa trên kỹ thuật lấy mẫu và kết hợp (Bagging). Đặc tính nổi bật là khả năng tổng hợp kết quả dự đoán của hàng loạt cây quyết định độc lập nhằm giảm thiểu tối đa phương sai (variance) của toàn hệ thống, ngăn chặn hiệu quả hiện tượng quá khớp (over-fitting) trên các tập dữ liệu nhiễu cao.

### Các phương pháp phụ trợ (Auxiliary Methods)

Toàn bộ đường ống kiến trúc hệ thống (Dataflow Pipeline) còn tích hợp ngầm định các thuật toán bổ trợ toán học sau nhằm đảm bảo sự luân chuyển dữ liệu không bị sai lệch:
- **Connected Component Labeling (CCL - Gắn nhãn thành phần liên thông):** Thuật toán duyệt đồ thị (Two-pass algorithm) để phân tách, nhóm các cụm điểm ảnh kề nhau thành từng đối tượng khuyết tật độc lập đưa vào bộ trích xuất đặc trưng.
- **Median Filtering (Lọc trung vị):** Phương pháp lọc không gian phi tuyến tính tích hợp ở đầu pipeline nhằm triệt tiêu các tín hiệu nhiễu xung (nhiễu muối tiêu) từ cảm biến thu nhận quang học phần cứng nhưng vẫn bảo toàn độ sắc nét của ranh giới đường biên lỗi.
- **Bicubic Interpolation (Nội suy ảnh cấp ba):** Kỹ thuật nội suy không gian được hệ thống sử dụng ở tầng trình diễn để tái lập và đồng nhất kích thước ma trận ảnh hiển thị trên giao diện người dùng mà không làm rạn nứt cấu trúc ma trận dữ liệu lỗi gốc.
---

## 📁 Cấu trúc thư mục (Repository Structure)

```text
📦 ImagePA_Final_Project
 ┣ 📂 app                  # Web Application Source Code
 ┃ ┣ 📂 backend            # FastAPI Server (Tích hợp API và Model)
 ┃ ┗ 📂 frontend           # Giao diện người dùng (ReactJS & Tailwind)
 ┣ 📂 data                 # Quản lý dữ liệu hình ảnh
 ┃ ┣ 📂 processed          # Ảnh sau khi qua các bước tiền xử lý/morphological
 ┃ ┗ 📂 raw                # Dataset ảnh lỗi gốc (PCB, dệt may)
 ┣ 📂 docs                 # Báo cáo mẫu AI & Slide thuyết trình
 ┣ 📂 models               # Lưu trữ file mô hình ML đã huấn luyện (.pkl)
 ┣ 📂 notebooks            # Jupyter Notebooks phục vụ thử nghiệm thuật toán
 ┣ 📂 src                  # Core Logic (Chia theo chuẩn 3+1)
 ┃ ┣ 📜 preprocessing.py   # Kỹ thuật 1: Tiền xử lý ảnh (Median, Otsu)
 ┃ ┣ 📜 morphological.py   # Kỹ thuật 2: Xử lý hình thái học
 ┃ ┣ 📜 feature_extraction.py # Kỹ thuật 3: Trích xuất đặc trưng vùng lỗi
 ┃ ┗ 📜 ml_classifier.py   # Mô hình ML: Phân loại khiếm khuyết
 ┣ 📜 .gitignore           # Cấu hình bỏ qua các tệp rác môi trường
 ┣ 📜 README.md            # Tài liệu hướng dẫn dự án
 ┗ 📜 requirements.txt     # Danh sách thư viện Python cần thiết
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy (Getting Started)

### 1. Cài đặt Môi trường
Clone repository về máy local:
```bash
git clone [https://github.com/NgocHuyen2309/ImagePA_Final_Project.git](https://github.com/NgocHuyen2309/ImagePA_Final_Project.git)
cd ImagePA_Final_Project
```

Thiết lập môi trường ảo và cài đặt thư viện:
```bash
python -m venv .venv

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate
# Kích hoạt môi trường (macOS/Linux)
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng

**Terminal 1 - Khởi chạy Backend:**
```bash
# Chạy từ thư mục gốc
uvicorn app.backend.main:app --reload
```

**Terminal 2 - Khởi chạy Frontend:**
```bash
cd app/frontend
npm install
npm start
```
*Giao diện sẽ hiển thị tại: `http://localhost:3000`*

---

## 👥 Thành viên nhóm & Phân công (Team Members)

| STT | Họ và Tên | Vai trò chính | Task kỹ thuật (3+1) |
| :---: | :--- | :--- | :--- |
| 1 | **Hứa Thị Ngọc Huyền** | Team Lead & Frontend | **Kỹ thuật 2:** Morphological Processing |
| 2 | **Phan Lê Trường An** | QA & Data Engine | **Kỹ thuật 1:** Image Pre-processing |
| 3 | **Huỳnh Minh Trí** | Backend Architecture | **Kỹ thuật 3:** Feature Extraction |
| 4 | **Phạm Duy Hoàng** | AI Engineer & Evaluation | **Mô hình ML:** Defect Classification |

---

## 📊 Báo cáo và Đánh giá (Report & Evaluation)
Tài liệu báo cáo chi tiết theo chuẩn format môn học được lưu trữ tại thư mục `docs/`. 

Phần thực nghiệm bao gồm:
- Phân tích và so sánh kết quả khi thay đổi kích thước Structuring Elements trong xử lý Morphological.
- Đánh giá mô hình ML thông qua Confusion Matrix, Accuracy và F1-Score trên tập dữ liệu thử nghiệm.
- So sánh thời gian thực thi của luồng xử lý trên nền tảng Web.

---

## 📚 Tài liệu tham khảo (References)
1. **Digital Image Processing (4th Edition)** - *Rafael C. Gonzalez, Richard E. Woods* [[Link Nhà Xuất Bản](https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003228/9780133356724) | [Trang Chủ Tác Giả](http://www.imageprocessingplace.com/)]
2. **Image analysis using mathematical morphology** - *Haralick, R. M., Sternberg, S. R., & Zhuang, X. (1987)* [[Đọc Bài Báo](https://haralick.org/journals/04767941.pdf)]
3. **Fabric defect detection systems and methods—A systematic literature review** - *Hanbay, K., Talu, M. F., & Özgüven, Ö. F. (2016)* [[Đọc Bài Báo](https://www.bingol.edu.tr/documents/Review(1).pdf)]
4. **Automatic fabric fault detection using morphological operations on bit plane** - *Tiwari, V., & Sharma, G. (2013)* [[Đọc Bài Báo](https://www.ijert.org/research/automatic-fabric-fault-detection-using-morphological-operations-on-bit-plane-IJERTV2IS100191.pdf)]
5. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** - *Aurélien Géron* [[GitHub Source](https://github.com/ageron/handson-ml3)]

---
*Mọi thay đổi mã nguồn và cập nhật tiến độ được quản lý nghiêm ngặt qua GitHub Issues và Project Board của nhóm.*