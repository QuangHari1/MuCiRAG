# MuCiRAG implementation

Xem [README chính](../README.md) để cài đặt, tải tài nguyên và chạy MuCiRAG.
Thư mục này giữ tên `MuCiRAG/` để tương thích đường dẫn các thí nghiệm cũ.

Từ thư mục này có thể tiếp tục dùng các lệnh trực tiếp:

```bash
uv sync --frozen
uv run python scripts/run_teleqna_benchmark.py --help
uv run python analysis/summarize_teleqna_runs.py --help
uv run python scripts/manage_assets.py validate
uv run python -m pytest tests -q
```

Các điểm bắt đầu đọc code:

- `src/rag/service.py`: `MuCiRAGService.run`, chuẩn bị query → seeds → citations → answer.
- `src/rag/corpus.py`: `EmbeddingCorpus`, truy xuất và giữ liên kết vector/chunk.
- `src/rag/ranking.py`: weighted reciprocal rank fusion.
- `src/rag/citation_expansion.py`: các chiến lược chọn citation.
- `src/settings.py`: đường dẫn tài nguyên tính từ gốc repository.
- `scripts/run_teleqna_benchmark.py`: checkpoint, luồng worker và báo cáo thí nghiệm.

[Kiến trúc](../docs/architecture.md) · [Tài nguyên](../docs/resources.md) ·
[Quy ước dataset](DATASET_LAYOUT.md)

Script tạo corpus và chạy benchmark ở `scripts/`; báo cáo, so sánh và Venn ở
[`analysis/`](analysis/README.md); tiện ích MLflow ở [`tools/`](tools/README.md).
Dataset đánh giá được lấy nguyên snapshot từ bundle, không tạo lại bằng script
lọc TeleQnA upstream.
