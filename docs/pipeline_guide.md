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
python scripts/run_morphological_pipeline.py --input data/raw --output data/processed/morphological --median-kernel 7 --tophat-kernel 21 --k-std 2.5 --open-size 3 --close-size 5 --iterations 3 --resize-width 512
```

## 3. Chạy Nhánh B: Canny Edge Pipeline (Lỗi Tuyến tính)
Thực thi phát hiện biên Canny, lượng tử hóa góc và Lọc có hướng (Directional Closing) để trích xuất `canny_features.csv`:
```bash
python scripts/run_canny_pipeline.py --input data/raw --output data/processed/canny --median-kernel 7 --apply-tophat --tophat-kernel 21 --resize-width 512
```

## 4. Kiểm tra Đầu ra Xử lý ảnh (Output Verification)
Sau khi hoàn tất Phase Xử lý ảnh, hệ thống sẽ sinh ra:
- `data/processed/morph_features.csv`: Chứa các nhãn Area, Perimeter, Eccentricity.
- `data/processed/canny_features.csv`: Chứa các nhãn Horiz_Length, Vert_Length.
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