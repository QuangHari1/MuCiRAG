---
language:
  - en
tags:
  - mucirag
  - telecommunications
  - 3gpp
  - retrieval-augmented-generation
  - embeddings
pretty_name: MuCiRAG runtime resources
---

# MuCiRAG runtime resources

Prepared artifacts for running MuCiRAG telecom question answering and its
1,840-question TeleQnA evaluation snapshot. Download these alongside the
[MuCiRAG source code](https://github.com/QuangHari1/MuCiRAG).

## Included files

| Path | Contents |
| --- | --- |
| `dataset/teleqna/TeleQnA.json` | Frozen local expanded evaluation set: 1,840 questions |
| `dataset/3gpp/Chunk/Rel-18/` | Complete chunk JSON, including text, headings, citation links and targets outside the embedded selection |
| `dataset/3gpp/Embeddings/Rel-18/paper-baseline-gsma-rel18/` | Chunk vectors, row metadata, embedding manifest, SQLite BM25 index, hierarchical anchor vectors/descriptions/manifest |
| `dataset/3gpp/embedding_selections/paper-baseline-gsma-rel18.json` | Selected document IDs and source provenance |
| `MuCiRAG/resources/` | NN router weights, series descriptions, vocabulary DOCX, Release-18 abbreviations and term definitions |
| `assets-manifest.json` | Inventory, exact byte sizes and SHA-256 checksums |

The embedding selection has **252,329 vectors**, **15 groups** (including
release summaries), and **1,024 dimensions**. The embedding model is
`text-embedding-3-large`. Its document selection contains 549 specifications
and 4 historical release summaries. The source-chunk collection is broader
than this selection so existing citation targets remain available.

Source specification Markdown/DOCX, source summary DOCX, separate heading
Metadata and Reference directories, partial embedding checkpoints, source code,
credentials, and experimental results are excluded. `3GPP_vocabulary.docx` is a
runtime vocabulary resource and is intentionally included. The benchmark hashes
it even when the JSON vocabulary mode is active.

## Download

From a current MuCiRAG checkout:

```bash
uv sync --project MuCiRAG --frozen
uv run --project MuCiRAG python mucirag.py assets download
uv run --project MuCiRAG python mucirag.py assets validate
```

The code's `MuCiRAG/resources.lock.json` pins this dataset's immutable commit
SHA and bundle manifest checksum. To select another release explicitly, supply
both `--repo QuangHari/MuCiRAG-resources` and `--revision COMMIT_SHA`.

This bundle supports running the prepared pipeline without the source documents.
Rebuilding chunks or document descriptions from scratch requires retrieving the
upstream source documents separately. Query-time embeddings and LLM calls still
use the providers configured in the source repository.

## Sources and attribution

- Release-18 Markdown used during preprocessing:
  [GSMA/3GPP](https://huggingface.co/datasets/GSMA/3GPP), revision
  `a056f6018a7e8e67052aa68a702e272d0ae95d75`.
- Reference selection and historical summaries: `netop/Embeddings3GPP-R18`,
  revision `b8d598e50cada8aaa4de641abbec77bef6b51839`, recorded in the code config.
- Evaluation originates from [netop/TeleQnA](https://huggingface.co/datasets/netop/TeleQnA),
  revision `0eba715a43f0ab7e4d9d7e09ceb258642a149391`, with the locally expanded
  snapshot of 1,840 questions. It differs from the original filtered 1,810-question
  set. Report the bundle/dataset checksum alongside benchmark results.
- Vocabulary and optional neural router retain their Telco-RAG lineage.

The underlying specifications remain subject to the terms of 3GPP and its
Organizational Partners, as described by the
[GSMA dataset card](https://huggingface.co/datasets/GSMA/3GPP#license-and-attribution).
The source repository's MIT license does not relicense third-party resources.
Cite the underlying specifications and datasets alongside MuCiRAG.

No author list, publication venue, DOI or benchmark accuracy is asserted here.
