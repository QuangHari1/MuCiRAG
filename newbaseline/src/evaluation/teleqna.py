"""TeleQnA parsing, answer normalization, and compact benchmark traces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
        "Return exactly `Option N` and nothing else."
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
    """Persist enough provenance for audit without duplicating chunk text 1,810 times."""
    return {
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
                "citation_depth": hit.citation_depth,
                "parent_chunk_id": hit.parent_chunk_id,
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
            }
            for path in result.citation_paths
        ],
    }
