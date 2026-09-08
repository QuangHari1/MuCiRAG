# Chuẩn bị tài nguyên và chạy MuCiRAG

Các lệnh dưới đây chạy từ `MuCiRAG/`. Chọn một trong hai luồng.

## Dùng tài nguyên đã chuẩn bị

Tải bundle theo hướng dẫn ở [README chính](../../README.md), rồi:

```bash
uv run python scripts/manage_assets.py validate
uv run python scripts/run_teleqna_benchmark.py \
  --limit 1 --workers 1 --output results/teleqna/smoke.jsonl
```

Lệnh validate không gọi API. Benchmark gọi provider cho query embeddings và LLM.
Dataset TeleQnA được dùng nguyên snapshot trong bundle.

## Tạo lại tài nguyên từ nguồn

Cần có raw corpus, mapping/selection và các release summaries tương ứng trước
khi chạy. Không chạy lại trên profile đã dùng để báo cáo kết quả nếu đang thay
đổi chunking hoặc model; chuẩn bị profile riêng qua cấu hình.

| Thứ tự | Script | Kết quả |
| --- | --- | --- |
| 1 | `prepare_embedding_selection.py` | Chọn tài liệu |
| 2 | `extract_heading.py` | Metadata heading và reference |
| 3 | `chunking.py` | Chunks của các specification |
| 4 | `chunk_paper_release_summaries.py` | Chunks của summaries, với mode paper |
| 5 | `embed_chunks.py` | Vectors và ánh xạ row/chunk |
| 6 | `build_lexical_index.py` | Index BM25 |
| 7 | `embed_anchor_hierarchy.py` | Series/document anchors |

`run_offline_pipeline.py --mode paper` điều phối bước 1–4. Để dùng lại selection
đã có, thêm `--selection PATH`. Các bước còn lại chạy riêng:

```bash
uv run python scripts/run_offline_pipeline.py --mode paper
uv run python scripts/embed_chunks.py \
  --selection ../dataset/3gpp/embedding_selections/paper-baseline-gsma-rel18.json \
  --dry-run
```

Sau khi kiểm tra workload, bỏ `--dry-run` để tạo embeddings qua API tính phí.
Khi vectors hoàn thành, chạy `build_lexical_index.py`, rồi
`embed_anchor_hierarchy.py` (cũng gọi embedding API), và cuối cùng
`manage_assets.py validate`.

`build_release18_vocabulary.py` tạo catalog thuật ngữ từ Markdown, độc lập với
chuỗi trên. Chỉ chạy khi cần tái tạo catalog; tài nguyên đã có trong bundle có
thể dùng trực tiếp. `manage_assets.py stage` đóng gói artifacts sau khi kiểm tra.

Phân tích checkpoint và vẽ Venn nằm ở [`../analysis/`](../analysis/README.md).
Tiện ích vận hành nằm ở [`../tools/`](../tools/README.md).
