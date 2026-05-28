# Hướng dẫn chạy pipeline tiền xử lý và Canny

## 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 2. Chuẩn bị dataset

Dataset nên để ngoài repo, ví dụ:

```text
D:\ImagePA_Work\Fabric_Dataset_Source\
```

Sau đó chuẩn hóa vào `data/raw` và tự động chia train/test theo tỷ lệ 80/20:

```bash
python scripts/prepare_fabric_dataset.py ^
  --source "D:\ImagePA_Work\Fabric_Dataset_Source" ^
  --output data/raw ^
  --copy
```

Cấu trúc sau khi chuẩn hóa:

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

Có thể đổi tỷ lệ test nếu cần:

```bash
python scripts/prepare_fabric_dataset.py ^
  --source "D:\ImagePA_Work\Fabric_Dataset_Source" ^
  --output data/raw ^
  --copy ^
  --test-size 0.2 ^
  --random-state 42
```

## 3. Thống kê dataset

```bash
python scripts/summarize_dataset.py --input data/raw --output-dir docs/dataset_stats
```

Kết quả gồm:

```text
docs/dataset_stats/split_distribution.csv
docs/dataset_stats/class_distribution.csv
docs/dataset_stats/image_formats.csv
docs/dataset_stats/image_resolutions.csv
docs/dataset_stats/dataset_summary.md
```

## 4. Chạy thử vài ảnh

```bash
python scripts/run_canny_pipeline.py ^
  --input data/raw ^
  --output data/processed/canny ^
  --resize-width 512 ^
  --limit 10
```

## 5. Chạy toàn bộ dataset

```bash
python scripts/run_canny_pipeline.py ^
  --input data/raw ^
  --output data/processed/canny ^
  --resize-width 512
```

## 6. Kết quả đầu ra

Mỗi ảnh sẽ có một thư mục kết quả riêng:

```text
data/processed/canny/train/<ten_lop>/<ten_anh>/
  01_grayscale.png
  01b_histogram.png
  02_median.png
  03_otsu_binary.png
  04_gaussian.png
  05_gradient_magnitude.png
  06_non_maximum_suppression.png
  07_double_threshold.png
  08_canny_edges.png
```

`01b_histogram.png` là biểu đồ phân bố mức xám của ảnh sau Median Filter, có đường đánh dấu ngưỡng Otsu.

## 7. Commit đề xuất

```bash
git add src/preprocessing.py src/canny_edge.py ^
scripts/run_canny_pipeline.py scripts/prepare_fabric_dataset.py scripts/summarize_dataset.py ^
docs/report_sections_preprocessing_canny.md docs/canny_usage.md docs/dataset_source.md ^
requirements.txt

git commit -m "feat: implement preprocessing and canny edge pipeline closes #25"
git push origin feature/canny-edge-core
```

Không add `data/raw` và `data/processed` nếu dataset hoặc ảnh kết quả có dung lượng lớn.
