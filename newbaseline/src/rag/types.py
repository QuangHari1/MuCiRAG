"""Stable result types returned by the RAG runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class CitationPath:
    parent_chunk_id: str
    depth: int
    reference: dict[str, Any]
    status: str
    target_chunk_ids: list[str]


@dataclass(frozen=True)
class RagResult:
    question: str
    rephrased_query: str
    enriched_query: str
    router_selected_series: list[str]
    empty_selected_series: list[str]
    searched_series: list[str]
    retrievals: list[RetrievalHit]
    answer: str | None
    citation_paths: list[CitationPath]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrievals"] = [
            {**asdict(hit), "metadata": dict(hit.metadata)} for hit in self.retrievals
        ]
        return payload
