# Kiểm tra MuCiRAG runtime release — 2026-09-08

## Bản phát hành Hugging Face

- Repo private: https://huggingface.co/datasets/QuangHari/MuCiRAG-resources
- Commit: `9ecd1d1c552dd695d33b799e54a451c9657fbf2a`.
- Pin ở `MuCiRAG/resources.lock.json`.
- 71 file tài nguyên, 1.870.232.625 bytes; ngoài dataset card, bundle manifest
  và `.gitattributes` do Hub tạo.
- Bundle manifest SHA-256:
  `cda0808a45c7b664cc7c316b086e1e1b1a2a2d7ae2e5c5d0d7f81e7df2b4b02b`.

Đã upload rồi tải lại toàn bộ bằng `mucirag.py assets download` vào thư mục
sạch. Checksum của cả 71 file pass; README có sẵn tại đích không bị ghi đè.
Repo, private visibility và commit đã đối chiếu với Hub API.

## Runtime và dữ liệu

- 252.329 vectors, 15 nhóm, 1.024 chiều.
- TeleQnA đúng 1.840 câu.
- Vector/metadata dimensions, metadata checksum, từng source index/chunk ID,
  selected-chunk digest, finite values, anchor provenance và SQLite quick_check
  đều pass trên bản tải lại.
- Không có source Markdown/DOCX, source summary DOCX, heading Metadata hoặc
  Reference riêng trong staging và bản tải lại.
- MuCiRAG chạy retrieval và phân giải citation trên gói tối thiểu với fake
  client xác định. Trace của staging và bản tải từ Hub giống nhau hoàn toàn.
- Đây là kiểm tra offline. Không gọi provider thật, không chạy benchmark tính
  phí và chưa build lại Docker image.

## Code và Git

- 63 test pass; compile Python và `git diff --check` pass.
- CLI assets/benchmark/summarize chạy help được ngoài thư mục repo.
- `uv sync --project MuCiRAG --frozen --offline` kiểm tra thành công 113 packages.
- Đã sửa import, config, Docker và CLI theo thư mục `MuCiRAG/` hiện tại.
- Dataset, resources, results, model/vector/index files, cache, env và backup
  được ignore. `config.toml`, `pyproject.toml` và `resources.lock.json` vẫn được
  phép đưa lên Git.
- File `backup/paper-baseline-gsma-rel18.jsonl` đã bỏ khỏi Git index; bản local
  giữ nguyên. Kiểm tra index không còn dữ liệu/kết quả thuộc phạm vi này.

Tài nguyên nguồn bị loại khỏi bundle vẫn còn ở máy phát triển. Kết quả local
và MLflow không được upload. Thay đổi GitHub nằm trong working tree/index;
không tạo commit hay push code trong thao tác phát hành tài nguyên này.
