# MuCiRAG

Research code for **MuCiRAG**, a retrieval-augmented generation method for
telecom question answering over 3GPP specifications. MuCiRAG combines
hierarchical semantic retrieval at the series, document and chunk levels with
BM25 rank fusion and citation-based context expansion.

## Setup

Requires Python 3.11 and `uv`. Run all commands from the repository root:

```bash
uv sync --project MuCiRAG --frozen
cp MuCiRAG/.env.example MuCiRAG/.env
```

Set `OPENAI_API_KEY` and `OPENROUTER_API_KEY` in `MuCiRAG/.env`.
The default configuration uses OpenAI `text-embedding-3-large` for query
embeddings and OpenRouter for rephrasing and answer generation.

## Resources

Download the prepared artifacts from
[QuangHari/MuCiRAG-resources](https://huggingface.co/datasets/QuangHari/MuCiRAG-resources).
The repository is private; `hf auth login` requires an account with access.

```bash
hf auth login
uv run --project MuCiRAG python mucirag.py assets download
uv run --project MuCiRAG python mucirag.py assets validate
```

The bundle contains chunk text and citation links, 252,329 precomputed
1,024-dimensional vectors, metadata, the BM25 index, hierarchical anchors,
router and vocabulary assets, and the 1,840-question TeleQnA snapshot.
The exact release and checksum are pinned in
[`MuCiRAG/resources.lock.json`](MuCiRAG/resources.lock.json).
Original source documents and re-embedding are not required to run the pipeline.

## Run

Ask a question:

```bash
uv run --project MuCiRAG python mucirag.py ask "What is network slicing?"
```

Evaluate one question before running the full benchmark:

```bash
uv run --project MuCiRAG python mucirag.py benchmark \
  --limit 1 --workers 1 --output results/teleqna/smoke.jsonl
```

Run and summarize the benchmark:

```bash
uv run --project MuCiRAG python mucirag.py benchmark \
  --workers 1 --output results/teleqna/mucirag.jsonl
uv run --project MuCiRAG python mucirag.py summarize \
  --run results/teleqna/mucirag.jsonl
```

Configure models and retrieval in [`MuCiRAG/config.toml`](MuCiRAG/config.toml).
Relative result paths resolve from `MuCiRAG/`. Repeat the same benchmark command
to resume; use a new result file when changing experiment settings. Query-time
embedding and LLM calls require provider credentials and may incur costs.
The supplied 1,840-question snapshot differs from the original filtered
1,810-question set; report the dataset version when comparing results.

## Code

- `MuCiRAG/src/rag/`: retrieval, hierarchical anchors, citation expansion and generation.
- `MuCiRAG/src/evaluation/`: scoring, checkpoint readers and experiment tracking.
- `MuCiRAG/scripts/`: resource management and benchmark execution.
- `MuCiRAG/analysis/`: result summaries, comparisons and Venn diagrams.
- `mucirag.py`: command-line entry point; append `--help` to any command.

## Acknowledgments

MuCiRAG uses [GSMA/3GPP](https://huggingface.co/datasets/GSMA/3GPP) corpus material
and [TeleQnA](https://huggingface.co/datasets/netop/TeleQnA), and retains
Telco-RAG document-selection, router and vocabulary resources. Source revisions
and attribution are recorded in the resource dataset card. Third-party
resources remain subject to their respective source terms.
