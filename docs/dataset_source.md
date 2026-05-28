# Nguồn dataset đề xuất

## Dataset chính

**Fabric Defects Dataset - Mendeley Data**

- Link: https://data.mendeley.com/datasets/663j22s43c/3
- Giấy phép: CC BY 4.0
- Nội dung: ảnh bề mặt vải gồm ảnh không lỗi và nhiều nhóm ảnh lỗi.
- Lý do chọn: sát đề tài phát hiện lỗi bề mặt dệt may, số lượng ảnh đủ để chạy pipeline xử lý ảnh và phục vụ bước phân loại phía sau.

## Cách tải và tổ chức dữ liệu

Tải dataset về máy và giải nén ngoài repo, ví dụ:

```text
D:\ImagePA_Work\Fabric_Dataset_Source\
```

Sau đó chạy script chuẩn hóa. Script sẽ tự động chia ảnh vào `data/raw/train` và `data/raw/test` theo tỷ lệ 80/20:

```bash
python scripts/prepare_fabric_dataset.py ^
  --source "D:\ImagePA_Work\Fabric_Dataset_Source" ^
  --output data/raw ^
  --copy
```

Nếu chỉ muốn test nhanh:

```bash
python scripts/prepare_fabric_dataset.py ^
  --source "D:\ImagePA_Work\Fabric_Dataset_Source" ^
  --output data/raw ^
  --copy ^
  --max-per-class 20
```

Cấu trúc đầu ra:

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

## Lưu ý khi đưa lên GitHub

Không push toàn bộ dataset vào repo nếu dung lượng lớn. Repo chỉ cần chứa code, script chuẩn hóa, hướng dẫn nguồn dataset và thống kê nhỏ trong `docs/dataset_stats/`.

Các thư mục nên giữ local:

```text
data/raw/
data/processed/
```

Các file có thể push:

```text
docs/dataset_source.md
docs/dataset_stats/*.csv
docs/dataset_stats/*.md
```

## Nguồn thay thế

Có thể dùng **AITEX Fabric Image Database** nếu nhóm muốn dataset nhỏ hơn và có mask thủ công. Tuy nhiên số lượng ảnh ít hơn nên có thể không phù hợp bằng dataset chính khi cần huấn luyện và so sánh mô hình học máy.
