# Phân tích kết quả MuCiRAG

Các script ở đây đọc checkpoint đã có, không gọi LLM hoặc embedding provider.
Chạy các lệnh dưới đây từ `MuCiRAG/`:

```bash
# Accuracy tổng thể và theo nhóm câu hỏi.
uv run python analysis/summarize_teleqna_runs.py --run results/teleqna/run-a.jsonl

# So sánh nhiều lần chạy và xuất bảng/biểu đồ.
uv run python analysis/compare_teleqna_results.py \
  --run A=results/teleqna/run-a.jsonl \
  --run B=results/teleqna/run-b.jsonl \
  --output-dir results/analysis/comparison

# Venn chính xác cho ba lần chạy trên cùng tập câu hỏi.
uv run python analysis/plot_teleqna_correct_venn.py \
  --run A=results/teleqna/run-a.jsonl \
  --run B=results/teleqna/run-b.jsonl \
  --run C=results/teleqna/run-c.jsonl \
  --expected-count 1840 --output results/analysis/venn.png
```

Từ gốc repo, lệnh `python mucirag.py summarize ...` vẫn giữ nguyên.
Các đường dẫn script cũ dưới `scripts/` đã được thay bằng `analysis/`.

Phần đọc dataset/checkpoint dùng chung ở `src/evaluation/records.py`.
Summary và comparison từ chối ID trùng lặp. Venn lấy bản ghi cuối cùng của mỗi
ID và yêu cầu ba lần chạy phủ đúng cùng tập ID. Các chính sách này được giữ
nguyên khi refactor; không dùng Venn của các tập câu hỏi khác nhau để so sánh.

Summary báo riêng coverage, failed và unscored. Accuracy trên các câu đã chấm
không đồng nghĩa accuracy trên toàn dataset nếu run chưa đầy đủ.
