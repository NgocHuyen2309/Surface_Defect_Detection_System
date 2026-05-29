# Hướng dẫn Kiểm thử (QA/Testing Guide): Feature Extraction

Tài liệu này dành cho Tech Lead / QA Engineer để thực hiện nghiệm thu (verify) tiến độ và chất lượng mã nguồn của module `Feature Extraction` dựa trên các tiêu chí Definition of Done (DoD) của đồ án.

---

## 1. Yêu cầu môi trường
Đảm bảo đã kích hoạt môi trường ảo chứa đủ các package OpenCV, NumPy, Pandas:
```bash
# Kích hoạt (Windows)
.venv\Scripts\activate

# Cài đặt nếu chưa có
pip install -r requirements.txt
```

---

## 2. Các kịch bản kiểm thử (Test Cases)

### Kịch bản 1: Kiểm định tính chính xác của Toán học (Mock Test)
**Mục đích:** Xác minh các công thức tính Invariants (Area, Perimeter, Eccentricity qua Eigenvalues) theo Slide bài giảng hoạt động chuẩn xác trong môi trường giả lập lý tưởng.

**Cách chạy:**
```bash
python test_feature.py
```

**Tiêu chí Pass (Expected Results):**
- Terminal in ra lỗi hình tròn (lỗi số 2) phải có `Eccentricity = 0.0000` (đúng tuyệt đối).
- Lỗi hình vuông/chữ nhật (lỗi số 1) và đường thẳng (lỗi số 3) có `Eccentricity > 0.8`.
- Ảnh đầu ra `data/processed/features_overlay/test_image_overlay.png` phải hiển thị 3 Bounding Box màu đỏ, viền Contour màu xanh lá, kèm Text A/P rõ ràng.
- `data/processed/features.csv` được tạo thành công và chứa thông số của `test_image.png`.

---

### Kịch bản 2: Kiểm định tích hợp trên Dữ liệu thật (Integration Test)
**Mục đích:** Đảm bảo hệ thống class OOP `RegionFeatureExtractor` phối hợp trơn tru với các hàm tiền xử lý truyền thống (Canny và Morphological) trên một bức ảnh lỗi vải thực tế.

**Yêu cầu:** Đã có ảnh trong `data/raw/train/hole/`. Nếu chưa có, chạy lệnh:
```bash
python scripts/prepare_fabric_dataset.py --source "ĐƯỜNG_DẪN_DATASET_GỐC" --output data/raw --copy --max-per-class 5
```

**Cách chạy:**
```bash
python test_real_data.py
```

**Tiêu chí Pass (Expected Results):**
- Console phân tách rõ 2 luồng: `Morphological` và `Canny Edge`.
- **Nhánh Morphological:** Tìm thấy khuyết tật (khối Solid Blobs), in ra tọa độ, kích thước diện tích chính xác (thường lớn hơn 50 pixels).
- **Nhánh Canny:** Sẽ báo `Tìm thấy 0 vùng lỗi`. **Lưu ý QA:** Đây là hành vi đúng bản chất học thuật do Canny sinh ra viền 1-pixel hở (không có area).
- Tệp `data/processed/features.csv` được append thêm dòng mới cho bức ảnh thực tế mà không đè mất kết quả của Kịch bản 1.
- Ảnh overlay (`*_morph_overlay.png`) đóng khung đúng các lỗ thủng trên bề mặt vải.

---

## 3. Checklist Bàn giao (Deliverables Review)
Tech Lead vui lòng rà soát thủ công các hạng mục sau:

- [ ] **Code Quality:** File `src/feature_extraction.py` được thiết kế OOP chuẩn mực, không có biến toàn cục, có type hint (`-> None`, `: np.ndarray`).
- [ ] **Documentations:** Chứa docstrings (PEP8) giải thích công năng hàm.
- [ ] **Data Pipeline:** File CSV được lưu với chuẩn `utf-8`, các headers: `filename`, `branch`, `region_id`, `area`, `perimeter`, `eccentricity` không bị lệch dòng.
- [ ] **Visualization:** Text in bằng `cv2.putText()` không bị đè ra ngoài ảnh và màu dễ đọc (vàng).

---
*Nếu tất cả các test case đều Pass, Tech Lead có thể Approve Pull Request cho nhánh `feature/feature-extraction-core` vào `main`.*
