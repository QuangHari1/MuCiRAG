"""Report overall and category accuracy for one or more TeleQnA checkpoints.

This is a read-only reporting utility.  It supports the current JSONL output
and the small legacy JSON variants read by ``evaluation.records``.
Accuracy is calculated only from rows with a known question and a boolean
correctness value; incomplete and failed rows are reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NEWBASELINE_ROOT = WORKSPACE_ROOT / "MuCiRAG"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from MuCiRAG.src.evaluation.records import load_dataset, load_json_records, option_number


KNOWN_CATEGORIES = ("Standards specifications", "Standards overview")
UNKNOWN_CATEGORY = "Uncategorized"


def category_for(record: dict[str, Any]) -> str:
    category = record.get("category")
    return category.strip() if isinstance(category, str) and category.strip() else UNKNOWN_CATEGORY


def correctness_for(row: dict[str, Any]) -> bool | None:
    """Apply the benchmark's correctness semantics to a result row."""
    if row.get("status") in {"error", "failed"}:
        return None
    correct = row.get("is_correct")
    if isinstance(correct, bool):
        return correct
    expected = option_number(row.get("expected_option"))
    predicted = option_number(row.get("predicted_option"))
    return expected == predicted if expected is not None and predicted is not None else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load one checkpoint and reject duplicate/malformed question IDs."""
    rows = load_json_records(path)
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")
    question_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Missing question_id at {path}:{row_number}")
        if question_id in question_ids:
            raise ValueError(f"Duplicate question_id {question_id!r} in {path}")
        question_ids.add(question_id)
    return rows


def percentage(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def group_summary(rows: list[dict[str, Any]], dataset_questions: int) -> dict[str, int | float | None]:
    scored = [row for row in rows if isinstance(row["is_correct"], bool)]
    correct = sum(row["is_correct"] is True for row in scored)
    failed = sum(row["status"] in {"error", "failed"} for row in rows)
    return {
        "run_rows": len(rows),
        "dataset_questions": dataset_questions,
        "scored_questions": len(scored),
        "correct_answers": correct,
        "accuracy_percent": percentage(correct, len(scored)),
        "coverage_percent": percentage(len(scored), dataset_questions),
        "unscored_rows": len(rows) - len(scored),
        "failed_rows": failed,
    }


def summarize_run(path: Path, dataset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize a checkpoint without treating partial checkpoints as wrong."""
    known_rows: list[dict[str, Any]] = []
    unknown_question_ids: list[str] = []
    for row in load_rows(path):
        question_id = row["question_id"]
        record = dataset.get(question_id)
        if record is None:
            unknown_question_ids.append(question_id)
            continue
        known_rows.append(
            {
                "question_id": question_id,
                "category": category_for(record),
                "is_correct": correctness_for(row),
                "status": row.get("status"),
            }
        )

    dataset_category_counts: dict[str, int] = {}
    for record in dataset.values():
        category = category_for(record)
        dataset_category_counts[category] = dataset_category_counts.get(category, 0) + 1
    extra_categories = sorted(category for category in dataset_category_counts if category not in KNOWN_CATEGORIES)
    categories = [*KNOWN_CATEGORIES, *extra_categories]

    return {
        "path": str(path),
        "checkpoint_rows": len(known_rows) + len(unknown_question_ids),
        "unknown_question_ids": unknown_question_ids,
        "overall": group_summary(known_rows, len(dataset)),
        "by_category": [
            {
                "category": category,
                **group_summary(
                    [row for row in known_rows if row["category"] == category],
                    dataset_category_counts.get(category, 0),
                ),
            }
            for category in categories
        ],
    }


def format_percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def print_summary(summary: dict[str, Any]) -> None:
    for run in summary["runs"]:
        overall = run["overall"]
        print(f"Run: {run['path']}")
        print(
            "  Overall: "
            f"{overall['correct_answers']}/{overall['scored_questions']} correct "
            f"({format_percentage(overall['accuracy_percent'])}); "
            f"coverage {overall['scored_questions']}/{overall['dataset_questions']} "
            f"({format_percentage(overall['coverage_percent'])})"
        )
        for category in run["by_category"]:
            print(
                f"  {category['category']}: "
                f"{category['correct_answers']}/{category['scored_questions']} correct "
                f"({format_percentage(category['accuracy_percent'])}); "
                f"coverage {category['scored_questions']}/{category['dataset_questions']}"
            )
        print(
            f"  Unscored: {overall['unscored_rows']} | failed: {overall['failed_rows']} | "
            f"unknown question IDs: {len(run['unknown_question_ids'])}"
        )


def resolve_path(path: Path) -> Path:
    """Resolve relative CLI paths after ``main`` switches to MuCiRAG."""
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Run from MuCiRAG; relative --run and --output paths are resolved there.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="A JSONL/JSON checkpoint; repeat this option to summarize multiple runs.",
    )
    parser.add_argument("--dataset", type=Path, default=WORKSPACE_ROOT / "dataset/teleqna/TeleQnA.json")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; omit to print only to the terminal.")
    return parser.parse_args()


def main() -> None:
    # Keep all relative result paths in the owned baseline directory regardless
    # of whether the user invoked this script from the repository root or not.
    os.chdir(NEWBASELINE_ROOT)
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    dataset = load_dataset(dataset_path)
    summary = {
        "dataset": {"path": str(dataset_path), "questions": len(dataset)},
        "runs": [summarize_run(resolve_path(path), dataset) for path in args.run],
    }
    print_summary(summary)
    if args.output is not None:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {output_path}")


if __name__ == "__main__":
    main()
