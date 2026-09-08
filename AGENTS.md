# Repository Guidelines

## MuCiRAG

This repository contains the MuCiRAG research implementation. Source code and entry points live in `MuCiRAG/`. The old `Telco-RAG_api/` implementation has been removed.

- Python source: `MuCiRAG/src/`.
- Offline preparation and evaluation helpers: `MuCiRAG/scripts/`.
- Result reports and figures: `MuCiRAG/analysis/`.
- Shared checkpoint readers: `MuCiRAG/src/evaluation/records.py`.
- Optional operational helpers: `MuCiRAG/tools/`.
- Tests: `MuCiRAG/tests/`.
- Root command dispatcher: `mucirag.py`.
- User documentation: `README.md` and `docs/`.
- Data and resources: downloaded from a versioned Hugging Face bundle; never
  commit corpora, vectors, `.env`, virtual environments, or benchmark results.

## Validation

Use Python 3.11 and the existing `MuCiRAG/uv.lock`.

```bash
uv sync --project MuCiRAG --frozen
uv run --project MuCiRAG python -m pytest MuCiRAG/tests -q
python3 -m compileall -q MuCiRAG/src MuCiRAG/scripts mucirag.py
git diff --check
```

Offline resource validation:

```bash
uv run --project MuCiRAG python mucirag.py assets validate
```

Live LLM execution, paid embedding generation, offline tests, and static
validation are separate results. Never present one as evidence for another.

## Refactoring contracts

Use Python 4-space indentation, snake_case functions/modules and PascalCase
classes. Keep pipeline orchestration in `rag/service.py`, retrieval in
`rag/corpus.py`, pure ranking in `rag/ranking.py`, and citation traversal in
`rag/citation_expansion.py`. Prefer small named stages with clear input/output.

Preserve prompts, provider/model routing, original versus enriched query use,
ranking/tie-breaking, context order, result schema and checkpoint semantics
unless explicitly changing an experiment. Keep old results and config values
intact during readability refactors. Test behavior with small fake clients and
corpora. Use a fresh result path when experimental conditions change.
