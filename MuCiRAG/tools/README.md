# Tiện ích vận hành

`mlflow_ui.py` mở MLflow UI cho store cục bộ. Đây là tiện ích tùy chọn, không
thuộc quy trình chuẩn bị corpus hay tái lập benchmark.

Chạy từ `MuCiRAG/`:

```bash
uv run python tools/mlflow_ui.py --host 127.0.0.1 --port 5000
```

Script giữ cấu hình hostname `mlflow.quanghari.uk` của môi trường hiện tại.
Khi triển khai ở máy khác, điều chỉnh allowed hosts/CORS theo hostname của bạn.
