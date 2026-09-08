# Tài nguyên MuCiRAG

## Tải và chạy

Từ gốc repo, cài dependencies và tải bản phát hành đã pin:

```bash
uv sync --project MuCiRAG --frozen
uv run --project MuCiRAG python mucirag.py assets download
uv run --project MuCiRAG python mucirag.py assets validate
```

Repo private cần đăng nhập `hf auth login` trước. `MuCiRAG/resources.lock.json`
lưu dataset ID, commit SHA bất biến và checksum của bundle manifest. Download
chỉ lấy các đường dẫn runtime liệt kê trong manifest, kiểm tra SHA-256, giữ
README của code và không tải tài liệu nguồn.

Để tải vào thư mục riêng, thêm `--output /tmp/mucirag-download`. Để chọn bản
khác, truyền cùng lúc `--repo OWNER/REPO --revision COMMIT_SHA`.

## Phạm vi bundle

Gồm toàn bộ chunk JSON để giữ citation targets; vector arrays và row metadata;
embedding manifest; `lexical.sqlite3`; ba file `anchor_hierarchy.*`; selection
manifest; NN router và series descriptions; ba file vocabulary/definitions;
và `dataset/teleqna/TeleQnA.json` đúng 1.840 câu.

Không đóng gói source Markdown/DOCX, source summary DOCX, heading `Metadata/`,
`Reference/` riêng, enhanced experiments, cache, partial embeddings hoặc kết quả.
Vocabulary DOCX là tài nguyên runtime, không phải tài liệu specification nguồn.
Các tài nguyên loại khỏi bundle vẫn được giữ ở máy phát triển.

## Tạo bản phát hành mới

```bash
uv run --project MuCiRAG python mucirag.py assets stage --output /tmp/mucirag-assets-next
uv run --project MuCiRAG python mucirag.py assets verify --root /tmp/mucirag-assets-next
hf upload QuangHari/MuCiRAG-resources /tmp/mucirag-assets-next . --repo-type dataset
```

Kiểm tra dataset card và inventory trước upload. Sau upload, tải lại vào thư
mục sạch, kiểm tra checksum và khả năng chạy rồi cập nhật resources.lock.json
bằng commit SHA cuối cùng. Không dùng `main` để pin một kết quả paper.

`assets validate` kiểm tra số chiều/số dòng vectors, metadata checksum, mapping
row/chunk ID, selected-chunk digest, giá trị finite, anchor provenance và SQLite
integrity. Đây là kiểm tra offline, không gọi model.

## Rebuilding artifacts

The published checkout reproduces runs with the pinned prepared bundle.
Artifact-building scripts and raw documents are kept locally and excluded from
Git. Rebuilding from source requires those local scripts and the upstream
sources identified in the dataset card. See the
[repository README](../README.md) for the scope of this release.
