from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from newbaseline.scripts import run_teleqna_benchmark
from newbaseline.scripts.run_teleqna_benchmark import evaluate_record
from newbaseline.src.corpus import discover_source_documents, load_selected_document_keys
from newbaseline.src.evaluation.teleqna import (
    compact_retrieval_trace,
    parse_record,
    score_multiple_choice,
)
from newbaseline.src.rag.clients import OpenAICompatibleRagClient
from newbaseline.src.rag.types import CitationPath, RetrievalHit


class SourceDiscoveryTests(unittest.TestCase):
    def test_primary_source_is_discovered_and_priority_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_dir = Path(temporary_directory) / "Rel-18"
            preferred_only = release_dir / "23_series" / "23501" / "raw.md"
            both_preferred = release_dir / "24_series" / "24501" / "raw.md"
            fallback = both_preferred.with_name("source.md")
            preferred_only.parent.mkdir(parents=True)
            both_preferred.parent.mkdir(parents=True)
            preferred_only.write_text("primary", encoding="utf-8")
            both_preferred.write_text("preferred", encoding="utf-8")
            fallback.write_text("fallback", encoding="utf-8")

            documents = discover_source_documents(release_dir, ("raw.md", "source.md"))

        self.assertEqual([item.document_key for item in documents], ["23_series/23501", "24_series/24501"])
        self.assertTrue(all(item.source_name == "raw.md" for item in documents))

    def test_selection_keys_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selection.json"
            path.write_text(json.dumps({"documents": [{"document_key": "23_series/23501"}]}), encoding="utf-8")
            self.assertEqual(load_selected_document_keys(path), {"23_series/23501"})

    def test_selection_count_and_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "document_count": 2,
                        "documents": [
                            {"document_key": "23_series/23501"},
                            {"document_key": "23_series/23501"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate document_key"):
                load_selected_document_keys(path)

            path.write_text(
                json.dumps({"document_count": 2, "documents": [{"document_key": "23_series/23501"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "document_count"):
                load_selected_document_keys(path)


class TeleQnATests(unittest.TestCase):
    def test_tail_run_does_not_compare_unless_requested(self) -> None:
        settings = run_teleqna_benchmark.load_settings()
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_directory = Path(temporary_directory) / "teleqna"
            result_directory.mkdir()
            (result_directory / "paper-baseline-gsma-rel18.jsonl").write_text("", encoding="utf-8")
            settings.values["evaluation"]["teleqna_output_dir"] = str(result_directory)
            with patch("sys.argv", ["run_teleqna_benchmark.py", "--reverse", "--limit", "200"]):
                args = run_teleqna_benchmark.parse_args(settings)

        self.assertEqual(
            args.output,
            result_directory / f"{settings.get('rag', 'selection_id')}-tail200.jsonl",
        )
        self.assertIsNone(args.compare_to)

    def test_multiple_choice_prompt_and_answer_scoring(self) -> None:
        record = parse_record(
            "question 0",
            {
                "question": "Which option is correct?",
                "option 1": "A",
                "option 2": "B",
                "option 3": "C",
                "option 4": "D",
                "option 5": "E",
                "answer": "option 5: E",
            },
        )
        predicted, correct = score_multiple_choice(record.expected_option, "Option 5: E is correct.")
        self.assertIn("Option 4: D", record.answer_prompt)
        self.assertIn("Option 5: E", record.answer_prompt)
        self.assertTrue(record.answer_prompt.endswith("Return exactly `Option N` and nothing else."))
        self.assertEqual(predicted, "option 5")
        self.assertTrue(correct)

    def test_strict_multiple_choice_client_limits_output_to_one_option(self) -> None:
        captured: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Option 5"))]
                )

        client = object.__new__(OpenAICompatibleRagClient)
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        client._answer_model = "test-model"
        client._llm_provider = "openai"
        client._thinking_mode = "disabled"
        client._temperature = 0.0

        answer = client.answer("Question", ["Context"], strict_multiple_choice=True)

        self.assertEqual(answer, "Option 5")
        self.assertEqual(captured["max_tokens"], 8)
        self.assertEqual(captured["temperature"], 0.0)
        self.assertIn("Return exactly `Option N` and nothing else.", captured["messages"][0]["content"])
        self.assertNotIn("CITED EVIDENCE", captured["messages"][0]["content"])
        self.assertNotIn("Adds evidence for", captured["messages"][0]["content"])

    def test_rephrase_uses_the_original_retrieval_only_prompt(self) -> None:
        captured: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Clear AMF question"))]
                )

        client = object.__new__(OpenAICompatibleRagClient)
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        client._rephrase_model = "test-model"
        client._llm_provider = "openai"
        client._thinking_mode = "disabled"
        client._temperature = 0.0

        rephrased = client.rephrase("When does AMF registration start?")

        self.assertEqual(rephrased, "Clear AMF question")
        self.assertNotIn("response_format", captured)
        self.assertEqual(
            captured["messages"][0]["content"],
            "Rephrase the question to be clear and concise:\n\nWhen does AMF registration start?",
        )

    def test_facets_are_separate_and_release_metadata_is_removed(self) -> None:
        captured: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=json.dumps(
                                    {
                                        "facets": [
                                            "the condition that triggers AMF registration",
                                            "reference to 3GPP Release 18",
                                        ]
                                    }
                                )
                            )
                        )
                    ]
                )

        client = object.__new__(OpenAICompatibleRagClient)
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        client._rephrase_model = "test-model"
        client._llm_provider = "openai"
        client._thinking_mode = "disabled"
        client._temperature = 0.0

        facets = client.extract_facets(
            "When does AMF registration start? [3GPP Release 18]"
        )

        self.assertEqual(facets, ["the condition that triggers AMF registration"])
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        prompt = captured["messages"][0]["content"]
        self.assertIn("Do not output isolated entities", prompt)
        self.assertIn("will not be used as a retrieval query", prompt)
        self.assertNotIn("Release 18", prompt)

    def test_relative_benchmark_paths_are_rooted_at_newbaseline(self) -> None:
        settings = run_teleqna_benchmark.load_settings()
        argv = [
            "run_teleqna_benchmark.py",
            "--output",
            "newbaseline/results/teleqna/run.jsonl",
        ]
        with patch("sys.argv", argv):
            args = run_teleqna_benchmark.parse_args(settings)

        self.assertEqual(
            args.output,
            run_teleqna_benchmark.NEWBASELINE_ROOT / "results/teleqna/run.jsonl",
        )

    def test_benchmark_row_uses_answer_and_keeps_compact_trace(self) -> None:
        class FakeService:
            def run(self, question: str, answer_prompt: str, strict_multiple_choice: bool):
                self.question = question
                self.answer_prompt = answer_prompt
                self.strict_multiple_choice = strict_multiple_choice
                return SimpleNamespace(
                    answer="Option 2: B",
                    router_selected_series=["21"],
                    empty_selected_series=[],
                    searched_series=["21"],
                    retrievals=[],
                    citation_paths=[],
                )

        service = FakeService()
        row = evaluate_record(
            service,
            "question 0",
            {
                "question": "Which option is correct?",
                "option 1": "A",
                "option 2": "B",
                "option 3": "C",
                "option 4": "D",
                "answer": "option 2: B",
            },
        )

        self.assertEqual(row["question_id"], "question 0")
        self.assertTrue(row["is_correct"])
        self.assertEqual(row["trace"]["searched_series"], ["21"])
        self.assertIn("Option 4: D", service.answer_prompt)
        self.assertTrue(service.strict_multiple_choice)

    def test_compact_trace_includes_selected_citation_gain_debug(self) -> None:
        citation = RetrievalHit(
            score=0.12,
            series="23",
            text="The cited clause text.",
            metadata={
                "chunk_id": "cite-1",
                "document_id": "23501",
                "heading": "4.2",
                "source_chunk_index": 7,
            },
            source_chunk_file="ChunkSeries23.json",
            origin="citation",
            citation_depth=1,
            parent_chunk_id="seed-1",
            citation={"type": "external", "target_heading": "4.2"},
            retrieval_method="citation_gain",
            citation_gain=0.12,
            citation_total_gain=0.15,
            citation_facets=("registration condition",),
            citation_facet_gains={"registration condition": 0.12, "failure action": 0.03},
            citation_coverage_before={"registration condition": 0.5, "failure action": 0.4},
            citation_coverage_after={"registration condition": 0.62, "failure action": 0.43},
        )
        path = CitationPath(
            parent_chunk_id="seed-1",
            depth=1,
            reference={"type": "external", "target_heading": "4.2"},
            status="resolved",
            target_chunk_ids=["cite-1"],
            candidate_chunk_ids=["cite-1", "cite-2"],
            best_candidate_gain=0.12,
        )
        result = SimpleNamespace(
            rephrased_query="When does registration occur?",
            query_facets=["registration condition", "failure action"],
            router_selected_series=["23"],
            empty_selected_series=[],
            searched_series=["23"],
            citation_min_gain=0.01,
            citation_max_chunks=4,
            citation_strategy="gain",
            retrievals=[citation],
            citation_paths=[path],
        )

        trace = compact_retrieval_trace(result)
        selected = trace["citation_debug"]["selected"][0]

        self.assertEqual(
            trace["citation_debug"]["policy"],
            {
                "strategy": "gain",
                "min_gain": 0.01,
                "max_chunks": 4,
                "gain_baseline": "parent_chunk",
                "semantic_query": None,
                "facets_used_for_seed_retrieval": False,
                "answer_context_policy": "baseline_seeds_append_neutral_citations",
            },
        )
        self.assertEqual(trace["citation_debug"]["selected_count"], 1)
        self.assertEqual(selected["path"], ["seed-1", "cite-1"])
        self.assertEqual(selected["max_gain"], 0.12)
        self.assertIsNone(selected["semantic_score"])
        self.assertEqual(selected["total_facet_gain"], 0.15)
        self.assertEqual(selected["text"], "The cited clause text.")
        self.assertEqual(trace["citation_paths"][0]["candidate_chunk_ids"], ["cite-1", "cite-2"])

    def test_compact_trace_distinguishes_semantic_bfs_scores_from_gain(self) -> None:
        citation = RetrievalHit(
            0.72,
            "23",
            "Semantic citation text.",
            {"chunk_id": "cite-1", "document_id": "23501", "heading": "4.2"},
            origin="citation",
            citation_depth=1,
            parent_chunk_id="seed-1",
            retrieval_method="citation_semantic_bfs",
            citation_semantic_score=0.72,
        )
        path = CitationPath(
            parent_chunk_id="seed-1",
            depth=1,
            reference={"type": "external", "target_heading": "4.2"},
            status="resolved",
            target_chunk_ids=["cite-1"],
            candidate_chunk_ids=["cite-1"],
            best_candidate_semantic_score=0.72,
        )
        result = SimpleNamespace(
            rephrased_query="question",
            query_facets=[],
            router_selected_series=["23"],
            empty_selected_series=[],
            searched_series=["23"],
            citation_min_gain=0.01,
            citation_max_chunks=4,
            citation_strategy="semantic_bfs",
            retrievals=[citation],
            citation_paths=[path],
        )

        trace = compact_retrieval_trace(result)
        policy = trace["citation_debug"]["policy"]
        selected = trace["citation_debug"]["selected"][0]

        self.assertEqual(policy["strategy"], "semantic_bfs")
        self.assertIsNone(policy["min_gain"])
        self.assertEqual(policy["semantic_query"], "enriched_rephrased_query")
        self.assertEqual(selected["semantic_score"], 0.72)
        self.assertIsNone(selected["max_gain"])
        self.assertEqual(
            trace["citation_paths"][0]["best_candidate_semantic_score"],
            0.72,
        )

    def test_benchmark_workers_write_one_checkpoint_row_per_question(self) -> None:
        class FakeService:
            def __init__(self, settings):
                self.settings = settings

            def run(self, question: str, answer_prompt: str, strict_multiple_choice: bool):
                self.strict_multiple_choice = strict_multiple_choice
                return SimpleNamespace(
                    answer="Option 1: A",
                    router_selected_series=["21"],
                    empty_selected_series=[],
                    searched_series=["21"],
                    retrievals=[],
                    citation_paths=[],
                )

        class FakeTracker:
            def log_progress(self, completed: int, correct: int, evaluated: int) -> None:
                pass

            def log_comparison(self, comparison: dict) -> None:
                pass

            def finish(self, **kwargs) -> None:
                pass

        record = {
            "question": "Which option is correct?",
            "option 1": "A",
            "option 2": "B",
            "option 3": "C",
            "option 4": "D",
            "answer": "option 1: A",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "teleqna.json"
            output_path = root / "result.jsonl"
            baseline_path = root / "baseline.jsonl"
            dataset_path.write_text(
                json.dumps({"question 0": record, "question 1": record, "question 2": record}),
                encoding="utf-8",
            )
            baseline_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"question_id": "question 1", "is_correct": True},
                        {"question_id": "question 2", "is_correct": False},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "run_teleqna_benchmark.py",
                "--dataset",
                str(dataset_path),
                "--output",
                str(output_path),
                "--workers",
                "2",
                "--reverse",
                "--limit",
                "2",
                "--compare-to",
                str(baseline_path),
            ]
            with (
                patch.object(run_teleqna_benchmark, "PaperRagService", FakeService),
                patch.object(run_teleqna_benchmark, "start_experiment_tracker", return_value=FakeTracker()),
                patch.object(
                    run_teleqna_benchmark,
                    "benchmark_config",
                    return_value={
                        "test": True,
                        "llm_temperature": 0.0,
                        "citation_strategy": "gain",
                        "citation_gain_baseline": "parent_chunk",
                        "facets_used_for_seed_retrieval": False,
                        "answer_context_policy": "baseline_seeds_append_neutral_citations",
                        "vocabulary_mode": "release18_unambiguous",
                        "vocabulary_definitions_sha256": "0" * 64,
                        "vocabulary_abbreviations_sha256": "0" * 64,
                    },
                ),
                patch("sys.argv", argv),
            ):
                run_teleqna_benchmark.main()

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            comparison = json.loads(output_path.with_suffix(".comparison.json").read_text(encoding="utf-8"))
            manifest = json.loads(output_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

        self.assertEqual([row["question_id"] for row in rows], ["question 2", "question 1"])
        self.assertTrue(all(row["is_correct"] is True for row in rows))
        self.assertEqual(comparison["shared_scored_questions"], 2)
        self.assertEqual(comparison["candidate_accuracy"], 1.0)
        self.assertEqual(comparison["baseline_accuracy"], 0.5)
        self.assertEqual(comparison["accuracy_delta"], 0.5)
        self.assertEqual(comparison["improved"], 1)
        self.assertEqual(manifest["llm_temperature"], 0.0)
        self.assertEqual(manifest["citation_strategy"], "gain")
        self.assertEqual(manifest["citation_gain_baseline"], "parent_chunk")
        self.assertFalse(manifest["facets_used_for_seed_retrieval"])
        self.assertEqual(
            manifest["answer_context_policy"],
            "baseline_seeds_append_neutral_citations",
        )
        self.assertIn("vocabulary_mode", manifest)
        self.assertEqual(len(manifest["vocabulary_definitions_sha256"]), 64)
        self.assertEqual(len(manifest["vocabulary_abbreviations_sha256"]), 64)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
