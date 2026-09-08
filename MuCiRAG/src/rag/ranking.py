"""Pure ranking operations shared by corpus retrieval modes.

Scores from dense similarity and BM25 have different scales. RRF combines
rank positions instead; tie-breaking is deterministic for reproducibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import RetrievalHit

HYBRID_CANDIDATE_MULTIPLIER = 4
HYBRID_RRF_K = 60


def fuse_ranked_hits(
    dense_hits: list[RetrievalHit],
    lexical_hits: list[RetrievalHit],
    top_k: int,
    *,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> list[RetrievalHit]:
    """Fuse dense and BM25 rankings with weighted reciprocal rank fusion."""
    validate_rrf_weights(dense_weight, bm25_weight)
    ranked: dict[tuple[str, str], dict[str, Any]] = {}
    for method, weight, hits in (
        ("dense", dense_weight, dense_hits),
        ("lexical", bm25_weight, lexical_hits),
    ):
        for rank, hit in enumerate(hits, start=1):
            identity = hit.metadata.get("chunk_id", hit.metadata.get("source_chunk_index"))
            key = (hit.series, str(identity))
            row = ranked.setdefault(
                key,
                {"hit": hit, "score": 0.0, "dense_rank": None, "lexical_rank": None},
            )
            row["score"] += weight / (HYBRID_RRF_K + rank)
            row[f"{method}_rank"] = rank
            if method == "dense":
                row["hit"] = hit
    ordered = sorted(
        ranked.values(),
        key=lambda row: (
            -float(row["score"]),
            min(row["dense_rank"] or 10**9, row["lexical_rank"] or 10**9),
            str(row["hit"].metadata.get("chunk_id", "")),
        ),
    )[:top_k]
    return [
        RetrievalHit(
            score=float(row["score"]),
            series=row["hit"].series,
            text=row["hit"].text,
            metadata=dict(row["hit"].metadata),
            source_chunk_file=row["hit"].source_chunk_file,
            retrieval_method="hybrid",
            dense_rank=row["dense_rank"],
            lexical_rank=row["lexical_rank"],
            anchor_series_score=row["hit"].anchor_series_score,
            anchor_document_score=row["hit"].anchor_document_score,
            anchor_chunk_score=row["hit"].anchor_chunk_score,
            anchor_hierarchical_score=row["hit"].anchor_hierarchical_score,
        )
        for row in ordered
    ]

def validate_rrf_weights(dense_weight: float, bm25_weight: float) -> None:
    if dense_weight < 0 or bm25_weight < 0 or not np.isclose(dense_weight + bm25_weight, 1.0):
        raise ValueError("RRF dense and BM25 weights must be non-negative and sum to 1.")
