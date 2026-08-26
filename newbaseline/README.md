# Telco-RAG new baseline

This directory is a self-contained, one-round RAG benchmark for TeleQnA. It
does not import or require `Telco-RAG_api/` at runtime.

The normal use case is simple: you already have the prepared `dataset/`
directory and the TeleQnA question file, and want to reproduce or compare an
experiment. **Do not download, chunk, or embed anything again** in that case.

## What is fixed in the supplied experiment

`config.toml` is the experiment configuration. Its current defaults are:

| Component | Value |
| --- | --- |
| Retrieval corpus | Paper-compatible Release-18 selection: 553 documents |
| Embeddings | `text-embedding-3-large`, 1,024 dimensions |
| Rephrase and answer model | `gpt-4o-mini` |
| Temperature | `0.0` |
| Seed retrieval | Hybrid dense + BM25, fused with RRF; 8 chunks |
| Citation expansion | One hop; at most 4 extra chunks (12 total) |
| Tracking | MLflow local SQLite store and artifacts under `results/mlflow/` |

The precomputed corpus contains 252,329 embedded chunks. The full chunk source
contains 299,412 chunks so citation expansion can be enabled later without
rebuilding vectors.

## Required files

Run all local commands below **from `newbaseline/`**. The expected layout is:

```text
Telco-RAG/
├── newbaseline/
│   ├── config.toml
│   ├── pyproject.toml                    # UV dependency source of truth
│   ├── uv.lock                           # reproducible Python 3.11 lockfile
│   ├── resources/                         # router checkpoint and vocabulary
│   └── scripts/run_teleqna_benchmark.py
└── dataset/
    ├── teleqna/TeleQnA.json               # evaluation questions
    └── 3gpp/
        ├── Chunk/Rel-18/                  # ChunkSeries*.json
        ├── Embeddings/Rel-18/
        │   └── paper-baseline-gsma-rel18/ # manifest, vectors, metadata, lexical.sqlite3
        └── embedding_selections/
            └── paper-baseline-gsma-rel18.json
```

For an already embedded benchmark, `marked/Rel-18/` raw Markdown is not read
at runtime. Keep it only if you want to rebuild chunks or embeddings. Do not
move individual files out of the paths above: the manifest and configuration
refer to them.

## 1. Set the API key

For the default OpenAI configuration, create `newbaseline/.env`:

```env
OPENAI_API_KEY=sk-...
```

The key is used for query embeddings and the two LLM calls. It is never stored
in result files, manifests, or the Docker image.

To use a different compatible LLM, edit only `[llm]` and the two model names
under `[rag]` in `config.toml`. Keep the embedding model unchanged unless you
intend to regenerate the entire embedding corpus.

## 2. Install and verify locally

Install [uv](https://docs.astral.sh/uv/) and Python 3.11, then:

```bash
cd newbaseline
uv sync --all-groups
uv run scripts/build_lexical_index.py
uv run scripts/run_teleqna_benchmark.py --help
```

The lexical index command is an offline, one-time step. It reads the existing
chunks and writes `lexical.sqlite3`; it does not call an LLM or embedding API.

`pyproject.toml` and `uv.lock` are the dependency source of truth. `uv sync`
creates the ignored local `.venv` automatically; do not activate it manually.
The lockfile already pins the CPU-only PyTorch index.

Run one paid smoke-test question before a larger experiment:

```bash
uv run \
  scripts/run_teleqna_benchmark.py \
  --limit 1 --workers 1 --progress-every 1 \
  --output results/teleqna/smoke-test.jsonl
```

## 3. Run a benchmark

### Full dataset

Use a new output name for every distinct experiment. Results are appended one
question at a time, so rerunning the same command resumes safely.

```bash
uv run \
  scripts/run_teleqna_benchmark.py \
  --workers 4 --progress-every 10 \
  --output results/teleqna/repro-full.jsonl
```

`--workers 4` processes four independent questions concurrently. Reduce it if
the provider rate-limits the account. The output's sibling
`repro-full.manifest.json` records every non-secret run parameter and the
TeleQnA SHA-256.

### Last 200 questions

Reverse numeric question order first, then take 200 records. Comparison is off
by default:

```bash
uv run \
  scripts/run_teleqna_benchmark.py \
  --reverse --limit 200 --workers 4 --progress-every 10 \
  --output results/teleqna/hybrid-tail200.jsonl
```

To compare the same questions with an existing run, add:

```bash
--compare-to results/teleqna/paper-baseline-gsma-rel18.jsonl
```

The terminal then prints candidate accuracy, baseline accuracy, delta,
improved and regressed counts. The same data is saved next to the run as a
`.comparison.json` file.

Comparison is disabled by default. Add `--compare-to PATH` only when a paired
comparison is wanted.

### Resume or restart

- Run the same command again to resume only unfinished question IDs.
- Add `--overwrite` to restart the specified `--output` JSONL from zero.
- Never use the same output filename for two different configurations.

Questions with no expected option are saved but marked `unscored`; they do not
contribute to accuracy or comparisons.

## Results and error analysis

Each benchmark JSONL row stores the answer, expected/predicted option,
correctness, router decision, hybrid retrieval ranks, and citation paths.
The full retrieved text is intentionally not repeated in every row.

Generate error-analysis tables and a readable summary:

```bash
uv run \
  scripts/analyze_teleqna_errors.py \
  --results results/teleqna/my-change-tail200.jsonl \
  --output-dir results/analysis/my-change-tail200
```

Read `results/analysis/my-change-tail200/summary.md` first. The directory also
contains CSV/JSON breakdowns for wrong answers, semantic scores, router series,
and citation paths.

MLflow is local by default. Every benchmark records its config, progress/final
metrics, result JSONL, manifest, comparison, and available analysis files under
`results/mlflow/`; no account or cloud upload is needed. Set
`[experiment_tracking].mode = "disabled"` in `config.toml` to turn this off.

## Release-18 reference-link statistics

To analyse the **full GSMA Release-18 corpus** (all local `raw.md` files, not
the paper's 553-document selection), run:

```bash
uv run scripts/analyze_release18_references.py
```

This writes `results/release18-reference-stats/release18_reference_statistics.pdf`
and a JSON audit file next to it.  The report counts distinct 3GPP/ETSI TS/TR
targets found in each document's References section, excludes self-links, and
separately reports targets that are present in the full local GSMA Rel-18
corpus.  The JSON records every extracted source/target relation for audit.

To browse runs locally, keep this command running in a second terminal from
`newbaseline/`, then open <http://127.0.0.1:5000>:

```bash
uv run scripts/mlflow_ui.py
```

To import pre-existing JSONL checkpoints into the same UI once:

```bash
uv run \
python scripts/import_teleqna_results_to_mlflow.py
```

## Changing retrieval settings

For a clean ablation, edit `config.toml`, choose a fresh `--output` filename,
and rerun tail-200. The benchmark manifest captures these settings.

```toml
[rag]
retrieval_backend = "hybrid"
retrieval_top_k = 8          # final dense + BM25 seed chunks
citation_max_depth = 1       # follow citations one hop only
citation_total_chunks = 12   # 8 seeds + at most 4 cited chunks
citation_chunks_per_heading = 1
```

For each question, dense retrieval and BM25 each produce up to 32 candidates
(`4 * retrieval_top_k`). Reciprocal Rank Fusion with `k = 60` combines their
rankings and keeps the best 8, without comparing incompatible dense and BM25
scores or introducing a learned weight. Citation expansion then follows exact
resolved heading links from those seeds, globally ranks the targets against the
query, and adds at most 4. One target chunk per heading avoids near-duplicate
context; depth 1 avoids citation-chain drift.

Changing `rephrase_model`, `answer_model`, temperature, retrieval limits, or
citation settings does **not** require re-embedding. Changing `[embedding]`
does require new vectors; also set `router_backend = "semantic"` if the
embedding model is no longer the paper-compatible OpenAI model.

### Vocabulary preprocessing

`paper_legacy` reads the original paper DOCX and is the paper-equivalent
baseline. The default `release18_unambiguous` keeps the 569 copied paper term
definitions and merges spelling-only expansion aliases, such as hyphen and case
variants. It expands an acronym only when the cleaned Release-18 catalog has
exactly one meaning. Genuinely ambiguous acronyms such as `AMF` remain unchanged
and are handled by the embedding model and answer LLM from normal question and
retrieval context.

To reproduce the legacy vocabulary behavior, change only:

```toml
[vocabulary]
mode = "paper_legacy"
```

## Run with Docker

The published image already contains this baseline, its router assets, the
prepared dataset, chunk files, embeddings, and previous offline artifacts. It
does not contain API keys, `Telco-RAG_api/`, or a development virtualenv.

```bash
docker pull quanghari/telco-rag-newbaseline:latest

docker run --rm -it \
  -e OPENAI_API_KEY \
  -v telco-rag-results:/workspace/newbaseline/results \
  quanghari/telco-rag-newbaseline:latest \
  newbaseline/scripts/run_teleqna_benchmark.py \
  --reverse --limit 200 --workers 4 --progress-every 10
```

The named volume preserves new results after the container exits. Do not
bind-mount an empty host `results/` directory because it hides the baseline
JSONLs bundled in the image. To use a different prepared dataset, mount it
read-only at `/workspace/dataset`:

```bash
docker run --rm -it \
  -e OPENAI_API_KEY \
  -v /absolute/path/to/dataset:/workspace/dataset:ro \
  -v telco-rag-results:/workspace/newbaseline/results \
  quanghari/telco-rag-newbaseline:latest \
  newbaseline/scripts/run_teleqna_benchmark.py --limit 1
```

## Rebuild the corpus (optional)

Only do this when prepared artifacts are absent or you deliberately changed
the embedding model/corpus. It downloads data and embedding requires paid API
calls, so it is not part of normal benchmark reproduction.

From the repository root:

```bash
uv run --project newbaseline \
  newbaseline/scripts/run_offline_pipeline.py --mode paper

uv run --project newbaseline \
  newbaseline/scripts/run_offline_pipeline.py --mode paper --embed --dry-run
```

Run the final command again with `--embed` (without `--dry-run`) only when you
intend to create or resume paid embeddings. The paper selection covers 549
Release-18 specifications plus the four Rel-14--Rel-17 summary documents.
