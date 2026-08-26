from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_teleqna_errors.py"
SPEC = importlib.util.spec_from_file_location("teleqna_error_analysis", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TeleQnAErrorAnalysisTests(unittest.TestCase):
    def test_writes_drill_down_artifacts_and_release_risk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "teleqna.json"
            results_path = root / "results.jsonl"
            output_dir = root / "analysis"
            dataset_path.write_text(
                json.dumps(
                    {
                        "question 1": {
                            "question": "Question one [3GPP Release 18]",
                            "option 1": "A",
                            "option 2": "B",
                            "answer": "option 1: A",
                            "category": "spec",
                        },
                        "question 2": {
                            "question": "Question two [3GPP Release 17]",
                            "option 1": "A",
                            "option 2": "B",
                            "option 3": "C",
                            "answer": "option 2: B",
                            "category": "spec",
                        },
                        "question 3": {
                            "question": "Not yet benchmarked [3GPP Release 18]",
                            "option 1": "A",
                            "option 2": "B",
                            "answer": "option 1: A",
                            "category": "spec",
                        },
                    }
                ),
                encoding="utf-8",
            )
            results_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "question_id": "question 1",
                            "expected_option": "option 1",
                            "predicted_option": "option 1",
                            "is_correct": True,
                            "answer": "Option 1",
                            "trace": {
                                "retrievals": [{"origin": "semantic", "score": 0.9, "chunk_id": "seed"}],
                                "citation_paths": [],
                            },
                        },
                        {
                            "question_id": "question 2",
                            "expected_option": "option 2",
                            "predicted_option": "option 4",
                            "is_correct": False,
                            "answer": "Option 4",
                            "trace": {
                                "retrievals": [
                                    {"origin": "semantic", "score": 0.2, "chunk_id": "seed"},
                                    {"origin": "citation", "score": 0.8, "chunk_id": "citation", "citation_depth": 1},
                                ],
                                "citation_paths": [
                                    {
                                        "status": "not_selected_by_query_score",
                                        "depth": 1,
                                        "parent_chunk_id": "seed",
                                        "reference": {"type": "external", "target_heading": "4"},
                                        "target_chunk_ids": [],
                                    }
                                ],
                            },
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            summary = MODULE.analyze(results_path, dataset_path, output_dir, "18")

            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["missing"], 1)
            self.assertEqual(summary["accuracy"], 0.5)
            for name in (
                "summary.md", "summary.json", "questions.csv", "errors.csv", "retrievals.csv",
                "citation_paths.csv",
            ):
                self.assertTrue((output_dir / name).is_file())
            errors = (output_dir / "errors.csv").read_text(encoding="utf-8")
            self.assertIn("release_mismatch_risk", errors)
            self.assertIn("citation_filtered_by_global_score", errors)
            self.assertIn("format_failure", errors)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
