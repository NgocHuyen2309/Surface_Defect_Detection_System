# Surface Defect Detection: Morphological vs Directional Gradient & ML Approach
**Đồ án môn học: Xử lý và Phân tích ảnh** | **Giảng viên hướng dẫn: TS. Đặng N.H. Thành**

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange.svg)](https://scikit-learn.org/)
[![ReactJS](https://img.shields.io/badge/React-18.x-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

## Tổng quan dự án (Project Overview)
Dự án tập trung xây dựng hệ thống tự động phát hiện và phân loại khiếm khuyết trên bề mặt sản phẩm công nghiệp (đặc biệt là vải dệt/fabric defect). 
Điểm nổi bật của phiên bản hiện tại là kiến trúc **Thử nghiệm A/B (A/B Testing Architecture)** kết hợp 2 nhánh xử lý ảnh truyền thống và 2 mô hình Học máy để tìm ra tổ hợp phân loại lỗi tối ưu nhất.

## Kiến trúc hệ thống (A/B Testing Pipeline)

Hệ thống được thiết kế theo mô hình xử lý luồng dữ liệu kép, bao gồm các bước:

### 1. Kỹ thuật Tiền xử lý & Trích xuất vùng lỗi (Computer Vision)
Hệ thống sử dụng song song 2 đường ống (Pipelines) để trích xuất các đặc trưng khiếm khuyết khác nhau:
- **Nhánh A (Morphological Processing):** Ứng dụng các toán tử lọc Toán học Hình thái (Mathematical Morphology) như Opening/Closing với phần tử cấu trúc (Structuring Element) kết hợp Thresholding (Otsu). Phương pháp này cực kỳ hiệu quả trong việc khử nhiễu nền dệt và gom cụm các vùng lỗi có cấu trúc khối (Hole, Stain).
- **Nhánh B (Directional Gradient):** Ứng dụng tích chập ma trận (Convolution) với các bộ lọc Gradient định hướng sắc nét (Sobel, Scharr, Custom Kernel) kết hợp với thuật toán Top-Hat. Phương pháp này xuất sắc trong việc bắt các khiếm khuyết dạng sợi mảnh, đường đứt nét có tính định hướng không gian rõ ràng dọc hoặc ngang.

### 2. Kỹ thuật Trích xuất Đặc trưng (Feature Extraction)
Sau khi vùng lỗi được khoanh vùng (Binarization & Connected Component Labeling), hệ thống trích xuất tập vector đặc trưng:
- **Morphological Features:** Diện tích (Area), Chu vi (Perimeter), Độ lệch tâm tối thiểu (Min Eccentricity).
- **Directional Features:** Độ dài theo chiều ngang (Horiz Length), Chiều dọc (Vert Length), và Đường chéo (Diag Length).

### 3. Phân loại thực thể: Học máy (Machine Learning Classifiers)
Tập đặc trưng trên được đưa vào 2 mô hình phân lớp hiện đại để đối chiếu chéo (Cross-validation):
- **Support Vector Machine (SVM):** Tìm kiếm siêu phẳng phân cách (hyperplane) tối ưu trong không gian đặc trưng đa chiều để phân biệt các nhóm lỗi.
- **Random Forest (RF):** Cơ chế học tập tập hợp (Ensemble Learning) gồm nhiều cây quyết định độc lập giúp giảm phương sai, chống over-fitting hiệu quả với dữ liệu nhiễu.

---

## Cấu trúc thư mục (Repository Structure)

```text
 ImagePA_Final_Project
 ┣  app                  # Hệ sinh thái Web Application
 ┃ ┣  backend            # Backend Services
 ┃ ┃ ┣  api_gateway      # Node.js API Gateway (Upload/Routing)
 ┃ ┃ ┗  ml_service       # Python FastAPI Server (Chạy AI/ML core)
 ┃ ┗  frontend           # Giao diện người dùng (ReactJS, Vite, Tailwind)
 ┣  data                 # Quản lý dữ liệu hình ảnh
 ┃ ┣  processed          # Ảnh sau khi qua các bước trích xuất CV
 ┃ ┗  raw                # Dataset ảnh lỗi gốc
 ┣  docs                 # Tài liệu Báo cáo, Kiến trúc & Hướng dẫn (pipeline_guide.md)
 ┃ ┗  evaluation         # Lưu các bảng biểu, biểu đồ Ma trận 2x2 tự động
 ┣  models               # Lưu trữ file mô hình ML đã huấn luyện (.pkl)
 ┣  reports              # Chứa các file CSV, đồ thị đánh giá (experiments)
 ┣  scripts              # Chứa script chạy tự động (huấn luyện, xuất báo cáo)
 ┣  src                  # Core Logic xử lý ảnh & học máy
 ┃ ┣  preprocessing.py   # Tiền xử lý ảnh (Median, Cân bằng sáng)
 ┃ ┣  morphological.py   # Thuật toán nhánh Morphological (Nhánh A)
 ┃ ┣  directional_gradient.py # Thuật toán nhánh Directional (Nhánh B)
 ┃ ┣  feature_extraction.py # Trích xuất vector đặc trưng
 ┃ ┣  rf_classifier.py   # Huấn luyện mô hình Random Forest
 ┃ ┗  svm_classifier.py  # Huấn luyện mô hình SVM
 ┣  README.md            # Tài liệu tổng quan dự án
 ┗  requirements.txt     # Danh sách thư viện Python cần thiết
```

---

## Hướng dẫn Cài đặt & Khởi chạy (Getting Started)

### 1. Cài đặt Môi trường
Hệ thống yêu cầu cả môi trường Python (Backend ML) và Node.js (Frontend, API Gateway).

Thiết lập môi trường ảo Python:
```bash
python -m venv .venv
# Kích hoạt (Windows)
.venv\Scripts\activate
# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Khởi chạy Hệ thống Web
Cần khởi động 3 dịch vụ đồng thời trên 3 cửa sổ terminal (cmd/powershell) khác nhau:

**Terminal 1 - API Gateway (Node.js)**
```bash
cd app/backend/api_gateway
npm install
node index.js
```
*(Chạy tại: `http://localhost:3000`)*

**Terminal 2 - Machine Learning Service (Python FastAPI)**
Lưu ý: Phải kích hoạt môi trường `.venv` trước.
```bash
cd app/backend/ml_service
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
*(Chạy tại: `http://127.0.0.1:8000`)*

**Terminal 3 - Frontend Dashboard (React/Vite)**
```bash
cd app/frontend
npm install
npm run dev
```
*(Truy cập giao diện Web tại: `http://localhost:5173`)*

---

## Thành viên nhóm & Phân công (Team Members)

| STT | Họ và Tên | Vai trò chính | Task kỹ thuật (3+1) |
| :---: | :--- | :--- | :--- |
| 1 | **Hứa Thị Ngọc Huyền** | Team Lead & Frontend | **Kỹ thuật 2:** Morphological Processing |
| 2 | **Phan Lê Trọng An** | QA & Data Engine | **Kỹ thuật 1:** Image Pre-processing & Directional |
| 3 | **Huỳnh Minh Trí** | Backend Architecture | **Kỹ thuật 3:** Feature Extraction |
| 4 | **Phạm Duy Hoàng** | AI Engineer & Evaluation | **Mô hình ML:** RF / SVM Classification |

---

## Báo cáo và Đánh giá tự động (Automated Evaluation)
Tài liệu báo cáo chi tiết và hướng dẫn sử dụng sâu hơn được lưu ở `docs/pipeline_guide.md`.
Dự án được tích hợp mã Script tự động sinh Báo cáo Ma trận 2x2. Sau khi chạy 2 file train ML, bạn chỉ cần gõ lệnh:
```bash
python scripts/generate_2x2_report.py
```
Script sẽ tự động tổng hợp File CSV, vẽ biểu đồ Bar Chart so sánh F1-Score/FPS và sinh ra file kết luận Markdown tại `docs/evaluation/report_2x2_evaluation.md`.

---
*Mọi thay đổi mã nguồn và cập nhật tiến độ được quản lý nghiêm ngặt qua GitHub.*
