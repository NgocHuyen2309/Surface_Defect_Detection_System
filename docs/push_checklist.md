# Checklist trước khi push

## Kiểm tra nhanh

```bash
python scripts/prepare_fabric_dataset.py --source "D:\ImagePA_Work\Fabric_Dataset_Source" --output data/raw --copy --max-per-class 20
python scripts/run_canny_pipeline.py --input data/raw --output data/processed/canny --resize-width 512 --limit 10
python scripts/summarize_dataset.py --input data/raw --output-dir docs/dataset_stats
```

Sau khi chạy thử, kiểm tra có các file kết quả chính:

```text
data/raw/train/<ten_lop>/...
data/raw/test/<ten_lop>/...
data/processed/canny/train/<ten_lop>/<ten_anh>/01b_histogram.png
docs/dataset_stats/split_distribution.csv
```

## File nên add

```bash
git add src/preprocessing.py src/canny_edge.py ^
scripts/run_canny_pipeline.py scripts/prepare_fabric_dataset.py scripts/summarize_dataset.py ^
docs/report_sections_preprocessing_canny.md docs/canny_usage.md docs/dataset_source.md docs/push_checklist.md ^
requirements.txt
```

Nếu `docs/dataset_stats` chỉ gồm file nhỏ thì có thể add thêm:

```bash
git add docs/dataset_stats
```

## File không nên add

```text
data/raw/
data/processed/
```

## Commit và push

```bash
git commit -m "feat: implement preprocessing and canny edge pipeline closes #25"
git push origin feature/canny-edge-core
```
