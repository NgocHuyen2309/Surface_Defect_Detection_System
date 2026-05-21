# 🔍 Surface Defect Detection: Morphological & Machine Learning Approach
**Đồ án môn học: Xử lý và Phân tích ảnh** | **Giảng viên hướng dẫn: TS. Đặng N.H. Thành**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org/)
[![ReactJS](https://img.shields.io/badge/React-18.x-61DAFB.svg)](https://reactjs.org/)

## 📖 Tổng quan dự án (Project Overview)
Dự án tập trung xây dựng hệ thống tự động phát hiện và phân loại khiếm khuyết trên bề mặt sản phẩm công nghiệp (PCB, vải dệt). Hệ thống kết hợp sức mạnh của **3 kỹ thuật xử lý ảnh truyền thống** để trích xuất vùng lỗi và **1 mô hình Machine Learning** để phân loại chính xác loại lỗi.

## 🛠 Quy trình xử lý (Pipeline 3+1) & Các phương pháp
Dự án được thiết kế theo luồng xử lý chuẩn công nghiệp, bám sát chương trình giảng dạy với **4 phương pháp chính** và **3 phương pháp phụ trợ**:

### Các phương pháp chính (Core Methods)
1.  **Tiền xử lý (Kỹ thuật 1):** Khử nhiễu không gian và nhị phân hóa bằng phương pháp Otsu (Otsu's Thresholding).
2.  **Xử lý hình thái học (Kỹ thuật 2):** Sử dụng các toán tử Erosion, Dilation, Opening, Closing (với các Structuring Elements đa dạng) để tách biệt và làm rõ vùng lỗi.
3.  **Trích xuất đặc trưng (Kỹ thuật 3):** Trích xuất thông qua Moment không gian, tính toán diện tích (Area), chu vi (Perimeter), độ lệch tâm (Eccentricity) đảm bảo tính bất biến hình học.
4.  **Phân loại Machine Learning (Mô hình AI):** Sử dụng mô hình Support Vector Machine (SVM) / Random Forest để phân loại lỗi dựa trên bộ đặc trưng đã trích xuất, ngăn chặn over-fitting.

### Các phương pháp phụ trợ (Auxiliary Methods)
- **Connected Component Labeling (CCL):** Thuật toán duyệt đồ thị phân tách các cụm pixel kề nhau thành từng đối tượng lỗi độc lập.
- **Median Filtering:** Lọc phi tuyến tính triệt tiêu nhiễu xung ở bước tiền xử lý.
- **Bicubic Interpolation:** Nội suy không gian tái lập kích thước ma trận ảnh hiển thị trên Web Browser.

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