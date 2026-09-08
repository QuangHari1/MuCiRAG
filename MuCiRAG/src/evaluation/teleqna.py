"""TeleQnA parsing, answer normalization, and compact benchmark traces."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from MuCiRAG.src.rag.corpus import HYBRID_RRF_K

OPTION_KEY_PATTERN = re.compile(r"^option\s+([1-9]\d*)$", re.IGNORECASE)
OPTION_PATTERN = re.compile(r"\boption\s*([1-9]\d*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class BenchmarkRecord:
    question_id: str
    question: str
    answer_prompt: str
    expected_option: str | None


def parse_record(question_id: str, payload: dict[str, Any]) -> BenchmarkRecord:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"TeleQnA record {question_id} has no question")
    options: list[tuple[int, str]] = []
    for key, value in payload.items():
        match = OPTION_KEY_PATTERN.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"TeleQnA record {question_id} has an empty {key}")
        options.append((int(match.group(1)), value))
    options.sort(key=lambda item: item[0])
    if len(options) < 2:
        raise ValueError(f"TeleQnA record {question_id} must have at least two options")
    answer = payload.get("answer")
    return BenchmarkRecord(
        question_id=question_id,
        question=question,
        answer_prompt=format_multiple_choice_question(question, options),
        expected_option=extract_option(answer) if isinstance(answer, str) else None,
    )


def format_multiple_choice_question(question: str, options: list[tuple[int, str]]) -> str:
    rendered_options = "\n".join(
        f"Option {number}: {option}" for number, option in options
    )
    return (
        f"{question}\n\n{rendered_options}\n\n"
        "Required response format: exactly `Option N`, with N replaced by one listed option number. "
        "Output only that label: no explanation, reasoning, punctuation, Markdown, code fence, option text, "
        "or second option. Valid response example: `Option 1`."
    )


def extract_option(answer: str | None) -> str | None:
    if not answer:
        return None
    match = OPTION_PATTERN.search(answer)
    return f"option {match.group(1)}" if match else None


def score_multiple_choice(expected_option: str | None, model_answer: str) -> tuple[str | None, bool | None]:
    predicted_option = extract_option(model_answer)
    if expected_option is None:
        return predicted_option, None
    return predicted_option, predicted_option == expected_option


def compact_retrieval_trace(result: Any) -> dict[str, Any]:
    """Persist compact seed provenance plus full selected-citation diagnostics."""
    citation_hits = [hit for hit in result.retrievals if hit.origin == "citation"]
    status_counts = Counter(path.status for path in result.citation_paths)
    citation_strategy = getattr(result, "citation_strategy", "gain")

    def rounded(values: dict[str, float]) -> dict[str, float]:
        return {key: round(value, 6) for key, value in values.items()}

    citation_policy = {
        "strategy": citation_strategy,
        "min_gain": (
            getattr(result, "citation_min_gain", None)
            if citation_strategy == "gain"
            else None
        ),
        "max_chunks": getattr(result, "citation_max_chunks", None),
        "gain_baseline": "parent_chunk" if citation_strategy == "gain" else None,
        "semantic_query": (
            "enriched_rephrased_query"
            if citation_strategy in {"semantic_bfs", "rrf_bfs"}
            else None
        ),
        "facets_used_for_seed_retrieval": False,
        "answer_context_policy": "baseline_seeds_append_neutral_citations",
    }
    if citation_strategy == "rrf_bfs":
        citation_policy["rrf_k"] = HYBRID_RRF_K

    return {
        "rephrased_query": getattr(result, "rephrased_query", None),
        "query_facets": getattr(result, "query_facets", []),
        "anchor_strategy": getattr(result, "anchor_strategy", "router"),
        "anchor_provenance": getattr(result, "anchor_provenance", None),
        "router_selected_series": result.router_selected_series,
        "empty_selected_series": result.empty_selected_series,
        "searched_series": result.searched_series,
        "retrievals": [
            {
                "score": hit.score,
                "series": hit.series,
                "chunk_id": hit.metadata.get("chunk_id"),
                "source_chunk_index": hit.metadata.get("source_chunk_index"),
                "document_key": hit.metadata.get("document_key"),
                "heading": hit.metadata.get("heading"),
                "origin": hit.origin,
                "retrieval_method": hit.retrieval_method,
                "dense_rank": hit.dense_rank,
                "lexical_rank": hit.lexical_rank,
                "anchor_series_score": hit.anchor_series_score,
                "anchor_document_score": hit.anchor_document_score,
                "anchor_chunk_score": hit.anchor_chunk_score,
                "anchor_hierarchical_score": hit.anchor_hierarchical_score,
                "citation_depth": hit.citation_depth,
                "parent_chunk_id": hit.parent_chunk_id,
                "citation_semantic_score": hit.citation_semantic_score,
                "citation_rrf_score": hit.citation_rrf_score,
                "citation_gain": hit.citation_gain,
                "citation_total_gain": hit.citation_total_gain,
                "citation_facets": list(hit.citation_facets),
            }
            for hit in result.retrievals
        ],
        "citation_paths": [
            {
                "parent_chunk_id": path.parent_chunk_id,
                "depth": path.depth,
                "status": path.status,
                "reference": path.reference,
                "target_chunk_ids": path.target_chunk_ids,
                "candidate_chunk_ids": path.candidate_chunk_ids,
                "best_candidate_semantic_score": path.best_candidate_semantic_score,
                "best_candidate_rrf_score": path.best_candidate_rrf_score,
                "best_candidate_gain": path.best_candidate_gain,
            }
            for path in result.citation_paths
        ],
        "citation_debug": {
            "policy": citation_policy,
            "selected_count": len(citation_hits),
            "selected_semantic_score_sum": (
                round(sum(hit.citation_semantic_score or 0.0 for hit in citation_hits), 6)
                if citation_strategy in {"semantic_bfs", "rrf_bfs"}
                else None
            ),
            "selected_rrf_score_sum": (
                round(sum(hit.citation_rrf_score or 0.0 for hit in citation_hits), 6)
                if citation_strategy == "rrf_bfs"
                else None
            ),
            "selected_max_gain_sum": (
                round(sum(hit.citation_gain or 0.0 for hit in citation_hits), 6)
                if citation_strategy == "gain"
                else None
            ),
            "selected_total_facet_gain_sum": (
                round(sum(hit.citation_total_gain or 0.0 for hit in citation_hits), 6)
                if citation_strategy == "gain"
                else None
            ),
            "decision_status_counts": dict(sorted(status_counts.items())),
            "selected": [
                {
                    "selection_rank": rank,
                    "chunk_id": hit.metadata.get("chunk_id"),
                    "parent_chunk_id": hit.parent_chunk_id,
                    "path": [hit.parent_chunk_id, hit.metadata.get("chunk_id")],
                    "depth": hit.citation_depth,
                    "series": hit.series,
                    "document_id": hit.metadata.get("document_id"),
                    "document_key": hit.metadata.get("document_key"),
                    "heading": hit.metadata.get("heading"),
                    "source_chunk_index": hit.metadata.get("source_chunk_index"),
                    "source_chunk_file": hit.source_chunk_file,
                    "semantic_score": (
                        round(hit.citation_semantic_score, 6)
                        if hit.citation_semantic_score is not None
                        else None
                    ),
                    "rrf_score": (
                        round(hit.citation_rrf_score, 6)
                        if hit.citation_rrf_score is not None
                        else None
                    ),
                    "max_gain": (
                        round(hit.citation_gain, 6)
                        if hit.citation_gain is not None
                        else None
                    ),
                    "total_facet_gain": (
                        round(hit.citation_total_gain, 6)
                        if hit.citation_total_gain is not None
                        else None
                    ),
                    "improved_facets": list(hit.citation_facets),
                    "facet_gains": rounded(hit.citation_facet_gains),
                    "coverage_before": rounded(hit.citation_coverage_before),
                    "coverage_after": rounded(hit.citation_coverage_after),
                    "reference": hit.citation,
                    "text": hit.text,
                }
                for rank, hit in enumerate(citation_hits, start=1)
            ],
        },
    }
