"""Run the complete TeleQnA multiple-choice benchmark with resumable JSONL output."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
NEWBASELINE_ROOT = Path(__file__).resolve().parents[1]

from newbaseline.src.evaluation.teleqna import (
    compact_retrieval_trace,
    parse_record,
    score_multiple_choice,
)
from newbaseline.src.evaluation.tracking import start_experiment_tracker
from newbaseline.src.rag.corpus import HYBRID_CANDIDATE_MULTIPLIER, HYBRID_RRF_K
from newbaseline.src.rag.service import PaperRagService
from newbaseline.src.settings import load_settings


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return {
        question_id: record
        for question_id, record in payload.items()
        if isinstance(question_id, str) and isinstance(record, dict)
    }


def completed_rows(output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = row.get("question_id")
        if not isinstance(question_id, str):
            raise ValueError(f"Missing question_id at {output_path}:{line_number}")
        if question_id in completed:
            raise ValueError(
                f"Duplicate question_id {question_id!r} in {output_path}; "
                "use a new --output path or rerun with --overwrite."
            )
        completed[question_id] = row
    return completed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vocabulary_manifest_config(settings: Any) -> dict[str, Any]:
    """Record the effective vocabulary mode and immutable asset fingerprints."""
    resources = settings.workspace_root / settings.get("rag", "resources_dir")
    paper_file = settings.get("rag", "vocabulary")
    definitions_file = settings.get("vocabulary", "definitions_file")
    abbreviations_file = settings.get("vocabulary", "abbreviations_file")
    assets = {
        "paper_vocabulary": paper_file,
        "definitions": definitions_file,
        "abbreviations": abbreviations_file,
    }
    config: dict[str, Any] = {"vocabulary_mode": settings.get("vocabulary", "mode")}
    for role, filename in assets.items():
        path = resources / filename
        config[f"vocabulary_{role}_file"] = filename
        config[f"vocabulary_{role}_sha256"] = file_sha256(path)
    return config


def benchmark_config(settings: Any, args: argparse.Namespace, dataset_path: Path) -> dict[str, Any]:
    """Persist every non-secret setting that can affect a benchmark result."""
    lexical_index_path = (
        settings.dataset_dir
        / "3gpp"
        / "Embeddings"
        / f"Rel-{settings.release}"
        / settings.get("rag", "selection_id")
        / settings.get("rag", "lexical_index_file")
    )
    config = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "corpus_release": settings.release,
        "embedding_backend": settings.get("embedding", "backend"),
        "embedding_model": settings.get("embedding", "model"),
        "embedding_dimensions": settings.get("embedding", "dimensions"),
        "llm_provider": settings.get("llm", "provider"),
        "llm_temperature": settings.get("llm", "temperature"),
        "selection_id": settings.get("rag", "selection_id"),
        "rephrase_model": settings.get("rag", "rephrase_model"),
        "answer_model": settings.get("rag", "answer_model"),
        "router_backend": settings.get("rag", "router_backend"),
        "router_top_k": settings.get("rag", "router_top_k"),
        "retrieval_backend": settings.get("rag", "retrieval_backend"),
        "retrieval_top_k": settings.get("rag", "retrieval_top_k"),
        "lexical_index_file": settings.get("rag", "lexical_index_file"),
        "hybrid_candidate_multiplier": HYBRID_CANDIDATE_MULTIPLIER,
        "hybrid_rrf_k": HYBRID_RRF_K,
        "citation_max_depth": settings.get("rag", "citation_max_depth"),
        "citation_total_chunks": settings.get("rag", "citation_total_chunks"),
        "citation_chunks_per_heading": settings.get("rag", "citation_chunks_per_heading"),
        "workers": args.workers,
        "limit": args.limit,
    }
    if config["retrieval_backend"] == "hybrid":
        config["lexical_index_sha256"] = file_sha256(lexical_index_path)
    config.update(vocabulary_manifest_config(settings))
    return config


def write_manifest(output_path: Path, config: dict[str, Any]) -> Path:
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def compare_completed_rows(
    candidate_rows: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    question_ids: list[str],
) -> dict[str, Any]:
    """Compare two runs only where both have a scored result for the same question."""
    shared = [
        question_id
        for question_id in question_ids
        if question_id in candidate_rows
        and question_id in baseline_rows
        and candidate_rows[question_id].get("is_correct") is not None
        and baseline_rows[question_id].get("is_correct") is not None
    ]
    candidate_correct = sum(candidate_rows[question_id].get("is_correct") is True for question_id in shared)
    baseline_correct = sum(baseline_rows[question_id].get("is_correct") is True for question_id in shared)
    improved = sum(
        candidate_rows[question_id].get("is_correct") is True
        and baseline_rows[question_id].get("is_correct") is False
        for question_id in shared
    )
    regressed = sum(
        candidate_rows[question_id].get("is_correct") is False
        and baseline_rows[question_id].get("is_correct") is True
        for question_id in shared
    )
    denominator = len(shared)
    return {
        "shared_scored_questions": denominator,
        "candidate_accuracy": candidate_correct / denominator if denominator else None,
        "baseline_accuracy": baseline_correct / denominator if denominator else None,
        "accuracy_delta": (candidate_correct - baseline_correct) / denominator if denominator else None,
        "improved": improved,
        "regressed": regressed,
        "unchanged": denominator - improved - regressed,
    }


def write_comparison(output_path: Path, comparison: dict[str, Any], baseline_path: Path) -> Path:
    comparison_path = output_path.with_suffix(".comparison.json")
    payload = {"candidate_path": str(output_path), "baseline_path": str(baseline_path), **comparison}
    comparison_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return comparison_path


def evaluate_record(service: PaperRagService, question_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one question with an isolated service instance and return its durable row."""
    record = parse_record(question_id, payload)
    result = service.run(
        record.question,
        answer_prompt=record.answer_prompt,
        strict_multiple_choice=True,
    )
    if result.answer is None:  # defensive; benchmark always requests an answer
        raise RuntimeError(f"No answer returned for {question_id}")
    predicted_option, is_correct = score_multiple_choice(record.expected_option, result.answer)
    return {
        "question_id": question_id,
        "expected_option": record.expected_option,
        "predicted_option": predicted_option,
        "is_correct": is_correct,
        "answer": result.answer,
        "trace": compact_retrieval_trace(result),
    }


def parse_args(settings: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=settings.dataset_dir / "teleqna" / settings.get("teleqna", "include_file"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Checkpoint JSONL. Defaults to a name derived from --reverse and --limit.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N records, for a paid smoke test.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse numeric question order before applying --limit (for example, the last 200 questions).",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Existing benchmark JSONL to compare against. Comparison is disabled unless this is supplied.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Discard an existing output checkpoint.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent questions. Each worker owns a separate RAG service (default: 1).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress and cumulative accuracy every N answered questions (default: 1).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    result_directory = Path(settings.get("evaluation", "teleqna_output_dir"))
    if not result_directory.is_absolute():
        result_directory = NEWBASELINE_ROOT / result_directory
    selection_id = settings.get("rag", "selection_id")
    if args.output is None:
        if args.limit is not None:
            run_suffix = f"-tail{args.limit}" if args.reverse else f"-head{args.limit}"
        elif args.reverse:
            run_suffix = "-reverse"
        else:
            run_suffix = ""
        args.output = result_directory / f"{selection_id}{run_suffix}.jsonl"
    if args.compare_to and not args.compare_to.is_file():
        parser.error(f"--compare-to does not exist: {args.compare_to}")
    return args


def main() -> None:
    settings = load_settings()
    args = parse_args(settings)
    records = load_records(args.dataset)
    ordered_ids = sorted(records, key=lambda value: int(value.split()[-1]) if value.split()[-1].isdigit() else value)
    if args.reverse:
        ordered_ids.reverse()
    if args.limit is not None:
        ordered_ids = ordered_ids[: args.limit]
    if args.compare_to and args.compare_to.resolve() == args.output.resolve():
        raise ValueError("--compare-to must differ from --output.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed = completed_rows(args.output)
    run_config = benchmark_config(settings, args, args.dataset)
    run_config["question_order"] = "descending" if args.reverse else "ascending"
    run_config["compare_to"] = str(args.compare_to) if args.compare_to else None
    manifest_path = write_manifest(args.output, run_config)
    tracker = start_experiment_tracker(
        mode=settings.get("experiment_tracking", "mode"),
        experiment_name=settings.get("experiment_tracking", "experiment_name"),
        tracking_directory=settings.workspace_root / settings.get("experiment_tracking", "tracking_directory"),
        output_path=args.output,
        config=run_config,
    )
    already_completed = [completed[question_id] for question_id in ordered_ids if question_id in completed]
    scored_rows = [row for row in already_completed if row.get("is_correct") is not None]
    correct = sum(row.get("is_correct") is True for row in scored_rows)
    evaluated = len(scored_rows)
    if completed:
        accuracy = correct / evaluated if evaluated else 0.0
        print(f"Resuming: {len(already_completed)}/{len(ordered_ids)} | accuracy={accuracy:.1%}", flush=True)
        tracker.log_progress(len(already_completed), correct, evaluated)
    pending_ids = [question_id for question_id in ordered_ids if question_id not in completed]
    pending_records = [(question_id, records[question_id]) for question_id in pending_ids]
    print(
        f"Running: {len(pending_records)} questions with {args.workers} worker(s)",
        flush=True,
    )

    def write_row(output_file, row: dict[str, Any]) -> None:
        nonlocal correct, evaluated
        question_id = row["question_id"]
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_file.flush()
        completed[question_id] = row
        evaluated += 1
        correct += row["is_correct"] is True
        answered = len(completed)
        if answered % args.progress_every == 0 or answered == len(ordered_ids):
            accuracy = correct / evaluated if evaluated else 0.0
            verdict = (
                "correct" if row["is_correct"] is True else "incorrect" if row["is_correct"] is False else "unscored"
            )
            print(
                f"{answered}/{len(ordered_ids)} | accuracy={accuracy:.1%} | "
                f"{question_id}: predicted={row['predicted_option']}, expected={row['expected_option']} ({verdict})",
                flush=True,
            )
            tracker.log_progress(answered, correct, evaluated)

    try:
        with args.output.open("a", encoding="utf-8") as output_file:
            if args.workers == 1:
                service = PaperRagService(settings=settings)
                for question_id, payload in pending_records:
                    write_row(output_file, evaluate_record(service, question_id, payload))
            else:
                worker_state = threading.local()

                def evaluate_in_worker(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
                    service = getattr(worker_state, "service", None)
                    if service is None:
                        service = PaperRagService(settings=settings)
                        worker_state.service = service
                    question_id, payload = item
                    return evaluate_record(service, question_id, payload)

                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    # executor.map preserves input order, while the workers overlap all provider calls.
                    for row in executor.map(evaluate_in_worker, pending_records):
                        write_row(output_file, row)
    except BaseException as error:
        tracker.finish(
            status="failed",
            completed=len(completed),
            correct=correct,
            evaluated=evaluated,
            results_path=args.output,
            manifest_path=manifest_path,
            analysis_directory=args.output.parent.parent / "analysis" / args.output.stem,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    accuracy = correct / evaluated if evaluated else 0.0
    comparison: dict[str, Any] | None = None
    comparison_path: Path | None = None
    if args.compare_to:
        comparison = compare_completed_rows(completed, completed_rows(args.compare_to), ordered_ids)
        comparison_path = write_comparison(args.output, comparison, args.compare_to)
        tracker.log_comparison(comparison)
        if comparison["shared_scored_questions"]:
            print(
                f"Comparison: shared={comparison['shared_scored_questions']} | "
                f"candidate={comparison['candidate_accuracy']:.1%} | "
                f"baseline={comparison['baseline_accuracy']:.1%} | "
                f"delta={comparison['accuracy_delta']:+.1%} | improved={comparison['improved']} | "
                f"regressed={comparison['regressed']} | report={comparison_path}",
                flush=True,
            )
        else:
            print(f"Comparison: shared=0 | report={comparison_path}", flush=True)
    tracker.finish(
        status="completed",
        completed=len(completed),
        correct=correct,
        evaluated=evaluated,
        results_path=args.output,
        manifest_path=manifest_path,
        analysis_directory=args.output.parent.parent / "analysis" / args.output.stem,
        comparison_path=comparison_path,
    )
    print(f"Done: {len(ordered_ids)}/{len(ordered_ids)} | accuracy={accuracy:.1%} | results={args.output}")


if __name__ == "__main__":
    main()
