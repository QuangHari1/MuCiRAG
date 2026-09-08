"""Stable result types returned by the RAG runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    series: str
    text: str
    metadata: dict[str, Any]
    source_chunk_file: str | None = None
    origin: str = "semantic"
    citation_depth: int = 0
    parent_chunk_id: str | None = None
    citation: dict[str, Any] | None = None
    retrieval_method: str = "semantic"
    dense_rank: int | None = None
    lexical_rank: int | None = None
    citation_semantic_score: float | None = None
    citation_rrf_score: float | None = None
    citation_gain: float | None = None
    citation_total_gain: float | None = None
    citation_facets: tuple[str, ...] = ()
    citation_facet_gains: dict[str, float] = field(default_factory=dict)
    citation_coverage_before: dict[str, float] = field(default_factory=dict)
    citation_coverage_after: dict[str, float] = field(default_factory=dict)
    anchor_series_score: float | None = None
    anchor_document_score: float | None = None
    anchor_chunk_score: float | None = None
    anchor_hierarchical_score: float | None = None


@dataclass(frozen=True)
class CitationPath:
    parent_chunk_id: str
    depth: int
    reference: dict[str, Any]
    status: str
    target_chunk_ids: list[str]
    candidate_chunk_ids: list[str] = field(default_factory=list)
    best_candidate_semantic_score: float | None = None
    best_candidate_rrf_score: float | None = None
    best_candidate_gain: float | None = None


@dataclass(frozen=True)
class RagResult:
    question: str
    rephrased_query: str
    query_facets: list[str]
    enriched_query: str
    router_selected_series: list[str]
    empty_selected_series: list[str]
    searched_series: list[str]
    citation_min_gain: float
    citation_max_chunks: int
    retrievals: list[RetrievalHit]
    answer: str | None
    citation_paths: list[CitationPath]
    citation_strategy: str = "gain"
    anchor_strategy: str = "router"
    anchor_provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrievals"] = [
            {**asdict(hit), "metadata": dict(hit.metadata)} for hit in self.retrievals
        ]
        return payload
