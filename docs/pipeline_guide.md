# Hướng Dẫn Vận Hành Hệ Thống End-to-End Pipeline

Tài liệu này cung cấp các lệnh tuần tự để chạy toàn bộ hệ thống từ dữ liệu thô (Raw), trích xuất đặc trưng (Feature Extraction), cho đến khi huấn luyện và đánh giá mô hình Machine Learning.

## 1. Chuẩn hóa Tập dữ liệu (Train/Test Split)
Sắp xếp ảnh lỗi từ dataset gốc, lọc rác và chia tập Train (80%) / Test (20%).
```bash
python scripts/prepare_fabric_dataset.py --source "ĐƯỜNG_DẪN_TỚI_DATASET" --output data/raw --copy
```

## 2. Chạy Nhánh A: Morphological Pipeline (Lỗi Đốm/Khối)
Thực thi tiền xử lý TopHat/BlackHat, phân đoạn Hình thái học và trích xuất `morph_features.csv`. Cấu hình khuyên dùng để làm mờ vân vải:
```bash
python scripts/run_morphological_pipeline.py --input data/raw --output data/processed/morphological --median-kernel 7 --tophat-kernel 21 --k-std 3.5 --open-size 3 --close-size 5 --iterations 3 --resize-width 512
```

## 3. Chạy Nhánh B: Directional Gradient Pipeline (Lỗi Tuyến tính)
Thực thi phát hiện biên Directional Gradient, lượng tử hóa góc và Lọc có hướng (Directional Closing) để trích xuất `directional_features.csv`:
```bash
python scripts/run_directional_gradient.py --input data/raw --output data/processed/directional --median-kernel 7 --apply-tophat --tophat-kernel 21 --resize-width 512
```

## 4. Kiểm tra Đầu ra Xử lý ảnh (Output Verification)
Sau khi hoàn tất Phase Xử lý ảnh, hệ thống sẽ sinh ra:
- `data/processed/morph_features.csv`: Chứa các nhãn Area, Perimeter, Eccentricity.
- `data/processed/directional_features.csv`: Chứa các nhãn Horiz_Length, Vert_Length.
- Các ảnh Overlay được khoanh vùng (Bounding Box) nằm trong các thư mục con tương ứng. Sẵn sàng cho Module Machine Learning!

## 5. Huấn luyện Machine Learning: Random Forest
Thực thi luồng đánh giá A/B Testing trên 2 tập dữ liệu CSV đã trích xuất bằng Random Forest (Ensemble Learning).
```bash
python src/rf_classifier.py
```
*(Mẹo: Thêm cờ `--fast-grid` để tìm kiếm siêu tham số nhanh hơn).*

**Kết quả:**
- Tệp mô hình: `models/rf_model.pkl`.
- Báo cáo phân tích: `reports/experiments/random_forest/` (Confusion Matrix, Feature Importance).

## 6. Huấn luyện Machine Learning: Support Vector Machine (SVM)
Thuật toán phân loại tối đa hóa lề (Margin Maximization). Pipeline tích hợp sẵn `StandardScaler` và sử dụng `PCA` để nén chiều, trực quan hóa Siêu phẳng (Hyperplane) phân cách các loại lỗi.
```bash
python src/svm_classifier.py
```
*(Mẹo: Thêm cờ `--fast-grid` để bỏ qua grid rbf mở rộng giúp chạy nhanh).*

**Kết quả:**
- Tệp mô hình: `models/svm_model.pkl`.
- Báo cáo phân tích: `reports/experiments/svm/` (Confusion Matrix, Biểu đồ Hyperplane 2D Scatter).

## 7. Đóng gói Thực nghiệm Ma trận 2x2 (Evaluation & Reporting)
Thực thi script tổng hợp để so sánh chéo định lượng hiệu năng (Accuracy, F1-Score) và tốc độ xử lý thực tế (FPS) giữa cả 4 tổ hợp đường ống hệ thống (Morphological / Directional Gradient kết hợp với Random Forest / SVM).
```bash
python scripts/generate_2x2_report.py
```

**Kết quả đầu ra lưu trữ tại `docs/evaluation/`:**
- `2x2_matrix_comparison.csv`: Bảng số liệu định lượng thô phục vụ lưu trữ.
- `2x2_performance_chart.png`: Biểu đồ cột kép so sánh trực quan hiệu năng và tốc độ FPS của 4 tổ hợp.
- `report_2x2_evaluation.md`: Văn bản báo cáo phân tích thực nghiệm tự động đi kèm kết luận lựa chọn mô hình tối ưu cho hệ thống.

## 8. Khởi chạy Hệ thống Web (Web App Deployment)
Để trực quan hóa kết quả phân tích trong thời gian thực (Live Inspection) và xem báo cáo thống kê (System Analytics), bạn cần khởi động 3 dịch vụ đồng thời trên 3 cửa sổ terminal (cmd/powershell) khác nhau:

### 8.1. API Gateway (Node.js)
Đóng vai trò làm Router, quản lý file Upload, kết nối Database lưu lịch sử và phân luồng tới ML Backend.
```bash
cd app/backend/api_gateway
node index.js
```
*Dịch vụ sẽ chạy tại: `http://localhost:3000`*

### 8.2. Machine Learning Service (Python FastAPI)
Chịu trách nhiệm thực thi thuật toán Xử lý ảnh (Morphological, Directional Gradient) và suy luận mô hình AI (Random Forest, SVM). Đảm bảo rằng bạn đã kích hoạt môi trường ảo (`.venv`) trước khi chạy lệnh.
```bash
cd app/backend/ml_service
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
*Dịch vụ sẽ chạy tại: `http://127.0.0.1:8000`*

### 8.3. Frontend Dashboard (React/Vite)
Giao diện người dùng trực quan, hiển thị kết quả phân tích AI và biểu đồ thống kê.
```bash
cd app/frontend
npm run dev
```
*Truy cập giao diện Web tại: `http://localhost:5173`*