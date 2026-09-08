"""Compare TeleQnA checkpoints by their number of correct answers.

Each run is grouped independently from its completed rows. The charts compare
correct-answer counts per 3GPP Release and TeleQnA category; they deliberately
do not classify whether another run got the same question wrong.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from MuCiRAG.src.evaluation.records import load_dataset, load_json_records, option_number

RELEASE_PATTERN = re.compile(r"\[\s*3GPP\s+Release\s+(\d+)\s*\]", re.IGNORECASE)
RELEASES = ("14", "15", "16", "17", "18", "19")
CATEGORIES = ("Standards overview", "Standards specifications")
LABEL_PATTERN = re.compile(r"[^a-z0-9]+")


class BenchmarkRun:
    """One named checkpoint after its answer fields have been normalized."""

    def __init__(self, label: str, path: Path, rows: dict[str, dict[str, Any]]) -> None:
        self.label = label
        self.path = path
        self.rows = rows


def load_run(label: str, path: Path) -> BenchmarkRun:
    rows: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(load_json_records(path), start=1):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Missing question_id at {path}:{row_number}")
        if question_id in rows:
            raise ValueError(f"Duplicate question_id {question_id!r} in {path}")
        correct = row.get("is_correct")
        if row.get("status") in {"error", "failed"}:
            correct = None
        elif not isinstance(correct, bool):
            expected = option_number(row.get("expected_option"))
            predicted = option_number(row.get("predicted_option"))
            correct = expected == predicted if expected is not None and predicted is not None else None
        rows[question_id] = {"is_correct": correct}
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")
    return BenchmarkRun(label, path, rows)


def release_for(record: dict[str, Any]) -> str | None:
    question = record.get("question")
    match = RELEASE_PATTERN.search(question) if isinstance(question, str) else None
    return match.group(1) if match else None


def category_for(record: dict[str, Any]) -> str | None:
    category = record.get("category")
    return category if isinstance(category, str) and category in CATEGORIES else None


def aggregate_correct_counts(
    runs: list[BenchmarkRun], dataset: dict[str, dict[str, Any]], group_name: str, group_values: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Count each run's own scored and correct rows for fixed dataset slices."""
    group_for = release_for if group_name == "release" else category_for
    rows: list[dict[str, Any]] = []
    for value in group_values:
        dataset_questions = sum(group_for(record) == value for record in dataset.values())
        for run in runs:
            matching = [
                run_row["is_correct"]
                for question_id, run_row in run.rows.items()
                if question_id in dataset
                and group_for(dataset[question_id]) == value
                and isinstance(run_row["is_correct"], bool)
            ]
            rows.append(
                {
                    group_name: value,
                    "run": run.label,
                    "dataset_questions": dataset_questions,
                    "scored_questions": len(matching),
                    "correct_answers": sum(matching),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_correct_count_plot(
    path: Path,
    rows: list[dict[str, Any]],
    group_name: str,
    group_values: tuple[str, ...],
    title: str,
) -> None:
    """Write a grouped bar chart whose height is the number of correct answers."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    run_labels = list(dict.fromkeys(row["run"] for row in rows))
    values = {(row[group_name], row["run"]): row["correct_answers"] for row in rows}
    positions = list(range(len(group_values)))
    width = 0.8 / len(run_labels)
    figure, axis = plt.subplots(figsize=(max(8, 1.4 * len(group_values)), 4.5))
    for index, run_label in enumerate(run_labels):
        offset = (index - (len(run_labels) - 1) / 2) * width
        bars = axis.bar(
            [position + offset for position in positions],
            [values[(value, run_label)] for value in group_values],
            width,
            label=run_label,
        )
        for bar in bars:
            axis.annotate(
                str(int(bar.get_height())),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )
    axis.set_title(title)
    axis.set_xlabel(group_name.replace("_", " ").title())
    axis.set_ylabel("Number of correct answers")
    axis.set_xticks(positions, group_values)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def slug(value: str) -> str:
    normalized = LABEL_PATTERN.sub("-", value.lower()).strip("-")
    return normalized or "run"


def correct_overlap(left: BenchmarkRun, right: BenchmarkRun) -> dict[str, Any]:
    """Count correct-answer overlap only where both checkpoints are scored."""
    shared = [
        question_id
        for question_id in left.rows.keys() & right.rows.keys()
        if isinstance(left.rows[question_id]["is_correct"], bool)
        and isinstance(right.rows[question_id]["is_correct"], bool)
    ]
    left_only = sum(
        left.rows[question_id]["is_correct"] is True and right.rows[question_id]["is_correct"] is False
        for question_id in shared
    )
    right_only = sum(
        left.rows[question_id]["is_correct"] is False and right.rows[question_id]["is_correct"] is True
        for question_id in shared
    )
    both_correct = sum(
        left.rows[question_id]["is_correct"] is True and right.rows[question_id]["is_correct"] is True
        for question_id in shared
    )
    return {
        "left": left.label,
        "right": right.label,
        "shared_scored_questions": len(shared),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": both_correct,
        "both_wrong": len(shared) - left_only - right_only - both_correct,
    }


def write_correct_overlap_venn(path: Path, overlap: dict[str, Any]) -> None:
    """Draw a dependency-free two-set Venn diagram for one pair of runs."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.add_patch(Circle((0.4, 0.5), 0.3, color="#4C78A8", alpha=0.5))
    axis.add_patch(Circle((0.6, 0.5), 0.3, color="#F58518", alpha=0.5))
    axis.text(0.27, 0.5, str(overlap["left_only_correct"]), ha="center", va="center", fontsize=16)
    axis.text(0.5, 0.5, str(overlap["both_correct"]), ha="center", va="center", fontsize=16)
    axis.text(0.73, 0.5, str(overlap["right_only_correct"]), ha="center", va="center", fontsize=16)
    axis.text(0.27, 0.12, f"{overlap['left']} only", ha="center", va="center")
    axis.text(0.73, 0.12, f"{overlap['right']} only", ha="center", va="center")
    axis.text(0.5, 0.86, "Both correct", ha="center", va="center")
    axis.set_title(
        f"Correct-answer overlap: {overlap['left']} vs {overlap['right']}\n"
        f"Shared scored questions: {overlap['shared_scored_questions']}"
    )
    axis.set_aspect("equal")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_runs(runs: list[BenchmarkRun], dataset: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("Provide at least two --run LABEL=PATH arguments.")
    if len({run.label for run in runs}) != len(runs):
        raise ValueError("Each --run label must be unique.")
    output_dir.mkdir(parents=True, exist_ok=True)
    by_release = aggregate_correct_counts(runs, dataset, "release", RELEASES)
    by_category = aggregate_correct_counts(runs, dataset, "category", CATEGORIES)
    write_csv(output_dir / "correct_answers_by_release.csv", by_release)
    write_csv(output_dir / "correct_answers_by_category.csv", by_category)
    write_correct_count_plot(
        output_dir / "correct_answers_by_release.png",
        by_release,
        "release",
        RELEASES,
        "Correct answers by 3GPP Release",
    )
    write_correct_count_plot(
        output_dir / "correct_answers_by_category.png",
        by_category,
        "category",
        CATEGORIES,
        "Correct answers by TeleQnA category",
    )
    overlaps: list[dict[str, Any]] = []
    for left, right in combinations(runs, 2):
        overlap = correct_overlap(left, right)
        if len(runs) == 2:
            venn_path = output_dir / "correct_overlap_venn.png"
        else:
            venn_path = output_dir / f"correct_overlap_venn_{slug(left.label)}_vs_{slug(right.label)}.png"
        write_correct_overlap_venn(venn_path, overlap)
        overlaps.append({**overlap, "venn_path": str(venn_path)})
    summary = {
        "runs": [
            {
                "label": run.label,
                "path": str(run.path),
                "sha256": file_sha256(run.path),
                "rows": len(run.rows),
                "scored_rows": sum(isinstance(row["is_correct"], bool) for row in run.rows.values()),
                "correct_answers": sum(row["is_correct"] is True for row in run.rows.values()),
            }
            for run in runs
        ],
        "by_release": by_release,
        "by_category": by_category,
        "correct_overlaps": overlaps,
        "interpretation": "Bar heights are each run's independent number of correct answers; missing or unscored rows are reported in the CSV files.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use --run LABEL=PATH, for example baseline=results/run.jsonl.")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both LABEL and PATH are required for --run.")
    path = Path(raw_path)
    return label.strip(), path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run, metavar="LABEL=PATH")
    parser.add_argument("--dataset", type=Path, default=WORKSPACE_ROOT / "dataset/teleqna/TeleQnA.json")
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "MuCiRAG/results/analysis/comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset if args.dataset.is_absolute() else (WORKSPACE_ROOT / args.dataset).resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (WORKSPACE_ROOT / args.output_dir).resolve()
    dataset = load_dataset(dataset_path)
    summary = compare_runs([load_run(label, path) for label, path in args.run], dataset, output_dir)
    summary["dataset"] = {"path": str(dataset_path), "sha256": file_sha256(dataset_path), "questions": len(dataset)}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Compared {len(summary['runs'])} runs | output={output_dir}")


if __name__ == "__main__":
    main()
