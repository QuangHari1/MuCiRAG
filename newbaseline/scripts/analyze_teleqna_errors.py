"""Create an offline retrieval and citation error-analysis report for TeleQnA."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


RELEASE_PATTERN = re.compile(r"\[3GPP Release\s+(\d+)\]", re.IGNORECASE)
OPTION_PATTERN = re.compile(r"^option\s+\d+$", re.IGNORECASE)


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load a benchmark checkpoint while rejecting duplicate question rows."""
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = row.get("question_id")
        if not isinstance(question_id, str):
            raise ValueError(f"Missing question_id at {path}:{line_number}")
        if question_id in rows:
            raise ValueError(f"Duplicate question_id {question_id!r} at {path}:{line_number}")
        rows[question_id] = row
    return rows


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def as_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def release_for(question: str) -> str:
    match = RELEASE_PATTERN.search(question)
    return match.group(1) if match else "unknown"


def option_count(payload: dict[str, Any]) -> int:
    return sum(OPTION_PATTERN.fullmatch(key) is not None for key in payload if isinstance(key, str))


def option_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and OPTION_PATTERN.fullmatch(key) and isinstance(value, str)
    }


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return mean(materialized) if materialized else None


def safe_median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return median(materialized) if materialized else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def round_or_blank(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def result_status(row: dict[str, Any]) -> str:
    if row.get("is_correct") is True:
        return "correct"
    if row.get("is_correct") is False:
        return "wrong"
    return "unscored"


def trace_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    retrievals = as_list(trace.get("retrievals"))
    semantic = [item for item in retrievals if item.get("origin", "semantic") == "semantic"]
    citations = [item for item in retrievals if item.get("origin") == "citation"]
    semantic_scores = [score for item in semantic if (score := as_float(item.get("score"))) is not None]
    citation_scores = [score for item in citations if (score := as_float(item.get("score"))) is not None]
    paths = as_list(trace.get("citation_paths"))
    path_statuses = Counter(
        str(path.get("status", "unknown")) for path in paths if isinstance(path, dict)
    )
    citation_depths = Counter(
        str(item.get("citation_depth", "unknown")) for item in citations if isinstance(item, dict)
    )
    reference_types = Counter(
        str(reference.get("type", "unknown"))
        for path in paths
        if isinstance(path, dict)
        for reference in [path.get("reference")]
        if isinstance(reference, dict)
    )
    return {
        "retrieval_count": len(retrievals),
        "semantic_count": len(semantic),
        "citation_count": len(citations),
        "semantic_top1_score": max(semantic_scores, default=None),
        "semantic_mean_score": safe_mean(semantic_scores),
        "semantic_score_spread": (
            max(semantic_scores) - min(semantic_scores) if len(semantic_scores) > 1 else None
        ),
        "citation_top1_score": max(citation_scores, default=None),
        "citation_mean_score": safe_mean(citation_scores),
        "citation_vs_semantic_top1": (
            max(citation_scores) - max(semantic_scores) if citation_scores and semantic_scores else None
        ),
        "citation_path_count": len(paths),
        "citation_path_statuses": dict(sorted(path_statuses.items())),
        "citation_depths": dict(sorted(citation_depths.items())),
        "citation_reference_types": dict(sorted(reference_types.items())),
        "router_selected_series": trace.get("router_selected_series", []),
        "empty_selected_series": trace.get("empty_selected_series", []),
        "searched_series": trace.get("searched_series", []),
    }


def build_rows(
    result_rows: dict[str, dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
    corpus_release: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id, result in result_rows.items():
        payload = dataset.get(question_id)
        if not isinstance(payload, dict):
            raise ValueError(f"Benchmark result {question_id!r} is absent from the supplied dataset.")
        question = payload.get("question")
        if not isinstance(question, str):
            raise ValueError(f"Dataset record {question_id!r} has no question.")
        trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
        row = {
            "question_id": question_id,
            "status": result_status(result),
            "is_correct": result.get("is_correct"),
            "expected_option": result.get("expected_option"),
            "predicted_option": result.get("predicted_option"),
            "answer": result.get("answer", ""),
            "question": question,
            "options": option_payload(payload),
            "reference_answer": payload.get("answer", ""),
            "reference_explanation": payload.get("explanation", ""),
            "question_release": release_for(question),
            "corpus_release": corpus_release,
            "release_mismatch_risk": release_for(question) not in {corpus_release, "unknown"},
            "category": payload.get("category", "unknown"),
            "option_count": option_count(payload),
            "question_characters": len(question),
        }
        row.update(trace_metrics(trace))
        rows.append(row)
    return rows


def add_diagnostic_flags(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    scores = [row["semantic_top1_score"] for row in rows if row["semantic_top1_score"] is not None]
    low_cutoff = percentile(scores, 0.25)
    high_cutoff = percentile(scores, 0.75)
    for row in rows:
        flags: list[str] = []
        if row["status"] == "unscored":
            flags.append("unscored")
        if row["status"] == "wrong" and row["predicted_option"] not in row["options"]:
            flags.append("format_failure")
        if row["release_mismatch_risk"]:
            flags.append("release_mismatch_risk")
        score = row["semantic_top1_score"]
        if row["status"] == "wrong" and score is not None and low_cutoff is not None and score <= low_cutoff:
            flags.append("weak_semantic_signal")
        if row["status"] == "wrong" and score is not None and high_cutoff is not None and score >= high_cutoff:
            flags.append("high_semantic_score_but_wrong")
        if row["citation_count"]:
            flags.append("citation_selected")
        statuses = row["citation_path_statuses"]
        if any(status in statuses for status in ("target_heading_not_found", "target_chunk_file_missing", "unresolved")):
            flags.append("citation_resolution_issue")
        if statuses.get("not_selected_by_query_score"):
            flags.append("citation_filtered_by_global_score")
        row["diagnostic_flags"] = flags
    return low_cutoff, high_cutoff


def group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    summary: list[dict[str, Any]] = []
    for value, group in groups.items():
        scored = [row for row in group if row["status"] != "unscored"]
        correct = sum(row["status"] == "correct" for row in scored)
        summary.append(
            {
                field: value,
                "completed": len(group),
                "scored": len(scored),
                "correct": correct,
                "wrong": sum(row["status"] == "wrong" for row in group),
                "accuracy": correct / len(scored) if scored else None,
                "median_semantic_top1": safe_median(
                    row["semantic_top1_score"] for row in group if row["semantic_top1_score"] is not None
                ),
                "mean_citation_count": safe_mean(float(row["citation_count"]) for row in group),
            }
        )
    return sorted(summary, key=lambda row: (-row["completed"], row[field]))


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._\n"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rendered = ["| " + " | ".join(headers) + " |", separator]
    for row in rows:
        rendered.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(rendered) + "\n"


def render_group_table(summary: list[dict[str, Any]], field: str) -> str:
    return markdown_table(
        [field, "completed", "scored", "correct", "wrong", "accuracy", "median seed top-1", "mean citations"],
        [
            [
                str(row[field]),
                str(row["completed"]),
                str(row["scored"]),
                str(row["correct"]),
                str(row["wrong"]),
                round_or_blank(row["accuracy"]),
                round_or_blank(row["median_semantic_top1"]),
                round_or_blank(row["mean_citation_count"]),
            ]
            for row in summary
        ],
    )


def render_diagnostic_queue(rows: list[dict[str, Any]], title: str, limit: int = 15) -> str:
    wrong = [row for row in rows if row["status"] == "wrong"]
    if title == "high":
        selected = sorted(wrong, key=lambda row: row["semantic_top1_score"] or float("-inf"), reverse=True)[:limit]
        heading = "Wrong answers with the highest semantic seed score"
    elif title == "low":
        selected = sorted(wrong, key=lambda row: row["semantic_top1_score"] or float("inf"))[:limit]
        heading = "Wrong answers with the weakest semantic seed score"
    else:
        selected = sorted(wrong, key=lambda row: (row["citation_count"], row["semantic_top1_score"] or 0), reverse=True)[:limit]
        heading = "Wrong answers with the most citation context"
    return f"### {heading}\n\n" + markdown_table(
        ["question", "release", "expected", "predicted", "seed top-1", "citations", "flags", "question excerpt"],
        [
            [
                row["question_id"],
                row["question_release"],
                str(row["expected_option"]),
                str(row["predicted_option"]),
                round_or_blank(row["semantic_top1_score"]),
                str(row["citation_count"]),
                ", ".join(row["diagnostic_flags"]),
                row["question"][:120] + ("…" if len(row["question"]) > 120 else ""),
            ]
            for row in selected
        ],
    )


def option_sort_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"option\s+(\d+)", value, re.IGNORECASE)
    return (int(match.group(1)), value) if match else (999, value)


def confusion_matrix(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[str]]]:
    scored = [row for row in rows if row["status"] != "unscored"]
    expected_labels = sorted({str(row.get("expected_option")) for row in scored}, key=option_sort_key)
    predicted_labels = sorted({str(row.get("predicted_option")) for row in scored}, key=option_sort_key)
    return predicted_labels, [
        [
            expected,
            *[
                str(
                    sum(
                        row.get("expected_option") == expected
                        and str(row.get("predicted_option")) == predicted
                        for row in scored
                    )
                )
                for predicted in predicted_labels
            ],
        ]
        for expected in expected_labels
    ]


def router_series_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        selected = row.get("router_selected_series")
        if not isinstance(selected, list):
            continue
        for series in {str(value) for value in selected}:
            groups[series].append(row)
    summary: list[dict[str, Any]] = []
    for series, group in groups.items():
        scored = [row for row in group if row["status"] != "unscored"]
        correct = sum(row["status"] == "correct" for row in scored)
        summary.append(
            {
                "series": series,
                "selected_questions": len(group),
                "wrong": sum(row["status"] == "wrong" for row in group),
                "accuracy": correct / len(scored) if scored else None,
            }
        )
    return sorted(summary, key=lambda row: (-row["selected_questions"], row["series"]))


def path_status_by_outcome(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for status, count in row["citation_path_statuses"].items():
            counts[status][row["status"]] += count
    return [
        {
            "path_status": status,
            "correct_paths": outcomes["correct"],
            "wrong_paths": outcomes["wrong"],
            "unscored_paths": outcomes["unscored"],
            "total": sum(outcomes.values()),
        }
        for status, outcomes in sorted(counts.items())
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field], ensure_ascii=False) if isinstance(row.get(field), (dict, list)) else row.get(field, "")
                    for field in fields
                }
            )


def retrieval_rows(rows: list[dict[str, Any]], result_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["question_id"]: row for row in rows}
    flattened: list[dict[str, Any]] = []
    for question_id, result in result_rows.items():
        trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
        origin_ranks: Counter[str] = Counter()
        for rank, retrieval in enumerate(as_list(trace.get("retrievals")), start=1):
            if not isinstance(retrieval, dict):
                continue
            origin = str(retrieval.get("origin", "semantic"))
            origin_ranks[origin] += 1
            row = by_id[question_id]
            flattened.append(
                {
                    "question_id": question_id,
                    "status": row["status"],
                    "question_release": row["question_release"],
                    "overall_rank": rank,
                    "origin_rank": origin_ranks[origin],
                    "origin": origin,
                    "retrieval_method": retrieval.get("retrieval_method", "semantic"),
                    "dense_rank": retrieval.get("dense_rank"),
                    "lexical_rank": retrieval.get("lexical_rank"),
                    "score": retrieval.get("score"),
                    "series": retrieval.get("series"),
                    "document_key": retrieval.get("document_key"),
                    "heading": retrieval.get("heading"),
                    "chunk_id": retrieval.get("chunk_id"),
                    "citation_depth": retrieval.get("citation_depth"),
                    "parent_chunk_id": retrieval.get("parent_chunk_id"),
                }
            )
    return flattened


def citation_path_rows(rows: list[dict[str, Any]], result_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["question_id"]: row for row in rows}
    flattened: list[dict[str, Any]] = []
    for question_id, result in result_rows.items():
        trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
        for path in as_list(trace.get("citation_paths")):
            if not isinstance(path, dict):
                continue
            reference = path.get("reference") if isinstance(path.get("reference"), dict) else {}
            row = by_id[question_id]
            flattened.append(
                {
                    "question_id": question_id,
                    "status": row["status"],
                    "question_release": row["question_release"],
                    "depth": path.get("depth"),
                    "path_status": path.get("status"),
                    "parent_chunk_id": path.get("parent_chunk_id"),
                    "reference_type": reference.get("type"),
                    "target_series": reference.get("target_series"),
                    "target_document_id": reference.get("document_id"),
                    "target_heading": reference.get("target_heading"),
                    "target_chunk_ids": path.get("target_chunk_ids", []),
                }
            )
    return flattened


def analyze(results_path: Path, dataset_path: Path, output_dir: Path, corpus_release: str) -> dict[str, Any]:
    """Write a complete offline report and return its machine-readable summary."""
    result_rows = load_jsonl(results_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("TeleQnA dataset must be a JSON object keyed by question_id.")
    typed_dataset = {key: value for key, value in dataset.items() if isinstance(key, str) and isinstance(value, dict)}
    rows = build_rows(result_rows, typed_dataset, corpus_release)
    low_cutoff, high_cutoff = add_diagnostic_flags(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = len(rows)
    dataset_total = len(typed_dataset)
    scored = [row for row in rows if row["status"] != "unscored"]
    correct = sum(row["status"] == "correct" for row in scored)
    wrong = [row for row in rows if row["status"] == "wrong"]
    citation_statuses = Counter(
        status
        for row in rows
        for status, count in row["citation_path_statuses"].items()
        for _ in range(count)
    )
    wrong_flags = Counter(flag for row in wrong for flag in row["diagnostic_flags"])
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "results_path": str(results_path),
        "dataset_path": str(dataset_path),
        "corpus_release": corpus_release,
        "dataset_total": dataset_total,
        "completed": completed,
        "missing": max(dataset_total - completed, 0),
        "scored": len(scored),
        "correct": correct,
        "wrong": len(wrong),
        "unscored": completed - len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "semantic_top1_low_quartile": low_cutoff,
        "semantic_top1_high_quartile": high_cutoff,
        "citation_path_statuses": dict(sorted(citation_statuses.items())),
        "wrong_diagnostic_flags": dict(sorted(wrong_flags.items())),
        "by_release": group_summary(rows, "question_release"),
        "by_category": group_summary(rows, "category"),
        "by_option_count": group_summary(rows, "option_count"),
        "by_expected_option": group_summary(rows, "expected_option"),
        "by_citation_count": group_summary(rows, "citation_count"),
        "router_series": router_series_summary(rows),
        "citation_path_status_by_outcome": path_status_by_outcome(rows),
    }

    question_fields = [
        "question_id", "status", "is_correct", "expected_option", "predicted_option", "answer", "reference_answer", "reference_explanation", "question_release",
        "corpus_release", "release_mismatch_risk", "category", "option_count", "question_characters", "question", "options",
        "retrieval_count", "semantic_count", "citation_count", "semantic_top1_score", "semantic_mean_score",
        "semantic_score_spread", "citation_top1_score", "citation_mean_score", "citation_vs_semantic_top1",
        "citation_path_count", "citation_path_statuses", "citation_depths", "citation_reference_types",
        "router_selected_series", "empty_selected_series", "searched_series", "diagnostic_flags",
    ]
    write_csv(output_dir / "questions.csv", rows, question_fields)
    write_csv(output_dir / "errors.csv", wrong, question_fields)
    write_csv(
        output_dir / "retrievals.csv",
        retrieval_rows(rows, result_rows),
        ["question_id", "status", "question_release", "overall_rank", "origin_rank", "origin", "retrieval_method", "dense_rank", "lexical_rank", "score", "series", "document_key", "heading", "chunk_id", "citation_depth", "parent_chunk_id"],
    )
    write_csv(
        output_dir / "citation_paths.csv",
        citation_path_rows(rows, result_rows),
        ["question_id", "status", "question_release", "depth", "path_status", "parent_chunk_id", "reference_type", "target_series", "target_document_id", "target_heading", "target_chunk_ids"],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    score_comparison = [
        [
            status,
            str(len([row for row in rows if row["status"] == status])),
            round_or_blank(safe_median(row["semantic_top1_score"] for row in rows if row["status"] == status and row["semantic_top1_score"] is not None)),
            round_or_blank(safe_median(row["semantic_score_spread"] for row in rows if row["status"] == status and row["semantic_score_spread"] is not None)),
            round_or_blank(safe_median(row["citation_top1_score"] for row in rows if row["status"] == status and row["citation_top1_score"] is not None)),
            round_or_blank(safe_mean(float(row["citation_count"]) for row in rows if row["status"] == status)),
        ]
        for status in ("correct", "wrong", "unscored")
    ]
    option_labels, option_matrix = confusion_matrix(rows)
    report = [
        "# TeleQnA error analysis",
        "",
        f"- Results: `{results_path}`",
        f"- Dataset: `{dataset_path}`",
        f"- Corpus release: {corpus_release}",
        f"- Completion: {completed}/{dataset_total} ({dataset_total - completed} questions absent from this checkpoint)",
        f"- Accuracy: {correct}/{len(scored)} = {correct / len(scored):.2%}" if scored else "- Accuracy: no scored rows",
        "",
        "## Outcome and retrieval signals",
        "",
        markdown_table(
            ["outcome", "count", "median seed top-1", "median seed spread", "median citation top-1", "mean citations"],
            score_comparison,
        ),
        "Semantic score is a query-to-chunk similarity signal, not proof that the chunk contains the gold evidence. Compare score distributions and inspect headings before claiming a retrieval failure.",
        "",
        "## Accuracy by question slice",
        "",
        "### Release",
        "",
        render_group_table(summary["by_release"], "question_release"),
        "### Category",
        "",
        render_group_table(summary["by_category"], "category"),
        "### Number of answer options",
        "",
        render_group_table(summary["by_option_count"], "option_count"),
        "### Expected correct option",
        "",
        render_group_table(summary["by_expected_option"], "expected_option"),
        "### Expected versus predicted option",
        "",
        markdown_table(["expected \\ predicted", *option_labels], option_matrix),
        "### Selected citation chunks",
        "",
        render_group_table(summary["by_citation_count"], "citation_count"),
        "## Citation graph diagnostics",
        "",
        markdown_table(
            ["path status", "count"],
            [[status, str(count)] for status, count in summary["citation_path_statuses"].items()],
        ),
        "### Citation paths split by answer outcome",
        "",
        markdown_table(
            ["path status", "correct paths", "wrong paths", "unscored paths", "total"],
            [
                [
                    row["path_status"],
                    str(row["correct_paths"]),
                    str(row["wrong_paths"]),
                    str(row["unscored_paths"]),
                    str(row["total"]),
                ]
                for row in summary["citation_path_status_by_outcome"]
            ],
        ),
        "### Router series selected most often",
        "",
        markdown_table(
            ["series", "selected questions", "wrong", "accuracy"],
            [
                [row["series"], str(row["selected_questions"]), str(row["wrong"]), round_or_blank(row["accuracy"])]
                for row in summary["router_series"][:15]
            ],
        ),
        "`citation_paths.csv` contains the parent chunk, target document/heading, reference type and selected target IDs. `retrievals.csv` contains every selected seed/citation chunk and its score.",
        "",
        "## Wrong-answer diagnostic flags",
        "",
        markdown_table(
            ["flag", "wrong questions"],
            [[flag, str(count)] for flag, count in summary["wrong_diagnostic_flags"].items()],
        ),
        "Flags are triage cues, not causal labels. In particular, `release_mismatch_risk` only says the question release differs from the Release-18 corpus; it does not prove missing evidence.",
        "",
        render_diagnostic_queue(rows, "high"),
        render_diagnostic_queue(rows, "low"),
        render_diagnostic_queue(rows, "citation"),
        "## How to use this report",
        "",
        "1. Start with release/category slices, then open `errors.csv` for the largest weak slice.",
        "2. For a high-score wrong answer, inspect its `retrievals.csv` headings: this separates a confidently irrelevant match from an LLM/distractor error despite likely evidence.",
        "3. For a low-score wrong answer, inspect router series and semantic seeds first; citation tuning is unlikely to repair a bad seed.",
        "4. For citation cases, inspect `citation_paths.csv`: resolution failures indicate parsing/coverage issues; `not_selected_by_query_score` indicates a valid link that lost the global rerank.",
        "5. This run logs scores only for selected citation chunks. It cannot retrospectively compare every rejected candidate's score or heading size; log candidate count/local rank/global rank in a future run if that comparison is needed.",
    ]
    (output_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=root / "newbaseline/results/teleqna/paper-baseline-gsma-rel18.jsonl",
        help="Benchmark JSONL checkpoint to inspect.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "dataset/teleqna/TeleQnA.json",
        help="TeleQnA JSON used by the benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "newbaseline/results/analysis/paper-baseline-gsma-rel18",
        help="Directory for Markdown, JSON, and CSV analysis artifacts.",
    )
    parser.add_argument("--corpus-release", default="18", help="Release represented by the retrieval corpus.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(args.results, args.dataset, args.output_dir, args.corpus_release)
    print(
        f"Analyzed {summary['completed']}/{summary['dataset_total']} questions | "
        f"accuracy={summary['accuracy']:.2%} | output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
