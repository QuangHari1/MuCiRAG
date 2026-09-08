# Kiến trúc MuCiRAG

## Online: từ câu hỏi đến câu trả lời

`MuCiRAGService` trong `MuCiRAG/src/rag/service.py` điều phối bốn bước:

1. `client.rephrase()` làm rõ câu hỏi; `vocabulary.enrich()` bổ sung thuật ngữ;
   `client.embed()` tạo vector cho câu đã bổ sung.
2. `_retrieve_seeds()` chọn một trong hai cách truy xuất: hierarchical trên tất
   cả chunks của manifest, hoặc router chọn series trước. `EmbeddingCorpus`
   thực hiện dense search và, khi cấu hình hybrid, hợp nhất với BM25 bằng RRF.
3. `_prepare_facets()` chỉ gọi model nếu dùng `gain`. `expand_citations()` theo
   những liên kết đã phân giải đến heading, chọn thêm chunks theo chiến lược
   citation đang bật và ghi lại đường đi/những quyết định loại ứng viên.
4. Seed contexts được đặt trước citation contexts. `client.answer()` nhận chúng
   và prompt câu hỏi; `RagResult` trả về đáp án cùng trace có cấu trúc.

Các hệ số, model và giới hạn đều đọc từ `config.toml`. Prompt ở `clients.py`.
Khi refactor, thứ tự lời gọi, nội dung prompt, query cho BM25, tie-breaking và
format context là một phần của hành vi thí nghiệm cần giữ ổn định.

## Retrieval

`corpus.py` chịu trách nhiệm ánh xạ embedding row → metadata → source chunk.
Không được giả định embedding row bằng chỉ số chunk trong file nguồn: selection
có thể chỉ chọn một phần tài liệu. `source_chunk_index` và `chunk_id` giữ ánh xạ.

Với hierarchical, điểm ngữ nghĩa kết hợp cosine similarity giữa query với
series anchor, document anchor và chunk vector. Top candidates được chọn trước
khi RRF với BM25. Với router, neural hoặc semantic router chọn series rồi mới
truy xuất. `ranking.py` chỉ làm việc với thứ hạng; không đọc file hay gọi model.

`citation_expansion.py` sở hữu duyệt đồ thị và các chiến lược:

| Chiến lược | Chọn đích citation |
| --- | --- |
| `rrf_bfs` | RRF dense/BM25 trong tập đích khả dụng ở mỗi độ sâu |
| `semantic_bfs` | Similarity của query và đích citation |
| `gain` | Mức tăng coverage của facets so với chunk cha |

Giới hạn `citation_max_chunks` áp dụng **mỗi độ sâu**. `citation_max_depth = 0`
hoặc `citation_max_chunks = 0` tắt mở rộng. Seed retrieval và mở rộng citation
là hai bước riêng; đổi citation strategy không đổi công thức xếp hạng seed.

## Offline: tạo artifacts (chỉ giữ local)

Các script tạo artifacts dưới đây chỉ giữ local và được ignore khỏi Git.
Bản checkout công bố dùng tài nguyên đã pin trên Hugging Face; chỉ
`manage_assets.py` và benchmark runner được giữ trong `scripts/`.

Các script local được chia theo nhiệm vụ:

| Nhóm | Script |
| --- | --- |
| Chọn corpus | `prepare_embedding_selection.py` |
| Heading và citation metadata | `extract_heading.py` |
| Chunking | `chunking.py`, `chunk_paper_release_summaries.py` |
| Vector | `embed_chunks.py`, `embed_anchor_hierarchy.py` |
| Lexical index | `build_lexical_index.py` |
| Vocabulary | `build_release18_vocabulary.py` |
| Chạy chuỗi preprocessing | `run_offline_pipeline.py` |
| Đóng gói, kiểm tra, tải artifacts | `manage_assets.py` |

Đổi corpus/chunking/embedding model có thể làm các artifacts không còn khớp.
Chuỗi xây dựng là selection → headings → chunks → embeddings → lexical index
→ hierarchy. Giữ fingerprint của toàn bộ profile khi công bố thí nghiệm.

## Evaluation

`evaluation/teleqna.py` parse câu hỏi và chấm multiple choice.
`scripts/run_teleqna_benchmark.py` sở hữu I/O checkpoint, worker, manifest và
debug artifacts. `evaluation/tracking.py` quản lý MLflow.
`evaluation/records.py` đọc dataset và các định dạng checkpoint dùng chung.
Các script summarize, compare và Venn ở `MuCiRAG/analysis/` đọc kết quả
để phân tích, không gọi model. Tiện ích mở MLflow UI ở `MuCiRAG/tools/`.

Khi đọc benchmark, phân biệt câu hỏi truy xuất với prompt chứa đáp án lựa chọn,
kết quả trả lời với debug context, và static tests với thực thi provider thật.
Các test offline dùng corpus nhỏ và fake clients để kiểm tra thứ tự gọi cùng
hành vi retrieval mà không phát sinh chi phí.
