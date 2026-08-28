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

Enhanced experiments live outside the baseline tree; see
[DATASET_LAYOUT.md](DATASET_LAYOUT.md) for the required isolated layout.

## 1. Set the API key

For the default OpenAI configuration, create `newbaseline/.env`:

```env
OPENAI_API_KEY=sk-...
```

The key is used for embeddings and the three LLM calls (rephrase, facets, and answer). It is never stored
in result files, manifests, or the Docker image.

To use a different compatible LLM, edit only `[llm]` and the two model names
under `[rag]` in `config.toml`. Keep the embedding model unchanged unless you
intend to regenerate the entire embedding corpus.

## 2. Install and verify locally

All host-side script commands below run from `newbaseline/`. Install
[uv](https://docs.astral.sh/uv/) and Python 3.11, then:

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

## Results

Each benchmark JSONL row stores the answer, expected/predicted option,
correctness, router decision, hybrid retrieval ranks, and citation paths.
The full retrieved text is intentionally not repeated in every row.

MLflow is local by default. Every benchmark records its config, progress/final
metrics, result JSONL, manifest, and comparison files under
`results/mlflow/`; no account or cloud upload is needed. Set
`[experiment_tracking].mode = "disabled"` in `config.toml` to turn this off.

## Changing retrieval settings

For a clean ablation, edit `config.toml`, choose a fresh `--output` filename,
and rerun tail-200. The benchmark manifest captures these settings.

```toml
[rag]
retrieval_backend = "hybrid"
retrieval_top_k = 8          # final dense + BM25 seed chunks
rrf_dense_weight = 0.5       # use 0.7 / 0.3 for dense-heavy fusion
rrf_bm25_weight = 0.5
anchor_strategy = "router"   # "router" (existing) or "hierarchical"
citation_strategy = "gain"   # "gain", "semantic_bfs", or "rrf_bfs"
citation_max_depth = 1       # follow citations one hop only
citation_min_gain = 0.01     # minimum marginal facet-coverage gain
citation_max_chunks = 4      # ceiling, not a forced citation count
citation_chunks_per_heading = 1
```

`anchor_strategy = "router"` preserves the existing `paper_nn` or semantic
series router. `anchor_strategy = "hierarchical"` skips that filter and ranks
every chunk in the active embedding manifest by `0.1 * series + 0.1 * document
+ 0.8 * chunk` semantic similarity. With hybrid retrieval this only replaces
the dense ranking; BM25 and RRF remain unchanged.

Before enabling hierarchical anchors, prepare its selection-specific artifact
once. This is intentionally a separate paid embedding step; benchmark runs only
read the resulting files and fail if their selection/model provenance differs.

```bash
uv run python scripts/embed_anchor_hierarchy.py
```

The script derives each document description from its title and `Scope` clause,
embeds document and series descriptions in batches, and writes the vectors,
descriptions, and provenance manifest next to the active corpus embeddings.
Use `--force` only when deliberately replacing an incompatible artifact after a
corpus or embedding-model change.

For each question, dense retrieval and BM25 each produce up to 32 candidates
(`4 * retrieval_top_k`). Reciprocal Rank Fusion with `k = 60` combines their
rankings and keeps the best 8, without comparing incompatible dense and BM25
scores or introducing a learned weight. After seed retrieval is complete, a
separate call creates a minimal list of answer-bearing facets. Facets never enter
the router, dense query, or BM25 query. Citation expansion follows exact resolved
heading links and measures how much a target increases facet coverage beyond the
exact parent chunk that cites it. If several seeds cite the same target, their
edges are scored separately and the selected target keeps the best parent path.
Only targets with marginal gain at least 0.01 are added, up to 4 chunks. One
target chunk per heading avoids near-duplicate context; depth 1 avoids
citation-chain drift.

`rrf_dense_weight` and `rrf_bm25_weight` apply to both hybrid seed retrieval
and `rrf_bfs` citation selection; they must be non-negative and sum to `1.0`.
For example, `0.7 / 0.3` favors semantic matches, while `0.3 / 0.7` favors
exact lexical terminology. The default `0.5 / 0.5` preserves equal-rank fusion.

Set `citation_strategy = "semantic_bfs"` to run the earlier citation baseline.
It follows the same precise citation graph breadth-first, ranks cited target
chunks by similarity to the enriched rephrased query, and keeps up to
`citation_max_chunks`. This strategy does not generate facets and ignores
`citation_min_gain`; the manifest and trace record `semantic_bfs` plus each
selected chunk's semantic score. Switch back to `gain` to use parent-local
facet evidence gain.

Set `citation_strategy = "rrf_bfs"` to keep the same breadth-first traversal
but fuse dense and BM25 ranks within the citation targets available at each
depth. It uses the enriched rephrased query for both ranks, RRF with `k = 60`,
and records dense rank, lexical rank, semantic score, and RRF score for each
selected citation. This requires `retrieval_backend = "hybrid"` and its lexical
index; it does not change the seed retrieval policy.

Seed chunks keep the same context format as the hybrid baseline. Selected
citations are appended after every seed and use the neutral label
`Referenced candidate`, together with the exact parent-to-target path. Facets
and numerical gain remain in the benchmark trace for analysis rather than
being asserted as evidence to the answer model.

Each TeleQnA JSONL row contains `trace.citation_debug`. Selected citations
include their full chunk text, parent-to-target path, max marginal gain, total
facet gain, per-facet gains, and facet coverage before/after selection. The
decision list also records candidate IDs, best candidate gain, and statuses such
as `below_min_gain`, `not_selected_by_gain`, or `duplicate_target`.

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

## Rebuild the corpus from a source-only dataset (optional)

Only do this when the image/release supplies raw source files but not prepared
chunks and vectors, or when deliberately changing the corpus or embedding
model. A source-only release must include the Release-18 `raw.md` tree, the
paper selection mapping, and the four release-summary DOCX files. Embedding
uses the configured paid API, so it is not part of normal benchmark
reproduction.

From `newbaseline/`, run the stages in this order:

```bash
cd newbaseline

# Skip when dataset/teleqna/TeleQnA.json was supplied with the image.
uv run python scripts/prepare_teleqna.py

# Selection -> headings -> chunks -> release-summary chunks.
uv run python scripts/run_offline_pipeline.py --mode paper

# Check the paid embedding workload without sending requests.
uv run python scripts/run_offline_pipeline.py --mode paper --embed --dry-run

# Create/resume vectors only after approving the embedding cost.
uv run python scripts/run_offline_pipeline.py --mode paper --embed

# Build the local BM25 index after vectors and chunks exist.
uv run python scripts/build_lexical_index.py
```

The paper selection covers 549 Release-18 specifications plus the four
Rel-14--Rel-17 summary documents. If using hierarchical retrieval, run
`uv run python scripts/embed_anchor_hierarchy.py` after the embedding stage.
For a prepared Docker dataset, none of these rebuild commands are needed:
only run `build_lexical_index.py` if `lexical.sqlite3` was intentionally left
out, then use `run_teleqna_benchmark.py`.
