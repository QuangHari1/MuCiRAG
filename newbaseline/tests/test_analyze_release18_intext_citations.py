import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/analyze_release18_intext_citations.py"
SPEC = importlib.util.spec_from_file_location("analyze_release18_intext_citations", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeRelease18IntextCitationsTest(unittest.TestCase):
    def test_analyze_excludes_reference_heading_and_maps_content_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_dir = root / "Reference"
            raw_dir = root / "raw"
            metadata_dir = root / "Metadata"
            reference_dir.mkdir()
            (reference_dir / "ReferenceSeries10.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "10001",
                                "bibliography": [
                                    {"reference_id": "1", "document_id": "20001"},
                                    {"reference_id": "2", "document_id": "10001"},
                                ],
                            },
                            {"document_id": "20001", "bibliography": []},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            source_raw = raw_dir / "10_series/10001/raw.md"
            target_raw = raw_dir / "20_series/20001/raw.md"
            for path, text in ((source_raw, "# title\nbody [1], [1], and [2]\nreferences [1]\n"), (target_raw, "# title\nno citations\n")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            for relative_path, document_id, headings in (
                ("10_series/10001", "10001", [{"title": "Body", "line": 1}, {"title": "References", "line": 3}]),
                ("20_series/20001", "20001", [{"title": "Body", "line": 1}]),
            ):
                metadata_path = metadata_dir / relative_path / "headings.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps({"document_id": document_id, "headings": headings}), encoding="utf-8")

            report = MODULE.analyze(reference_dir, raw_dir, metadata_dir)

        metrics = report["metrics"]
        self.assertEqual(metrics["total_intext_bracket_markers"], 3)
        self.assertEqual(metrics["total_resolved_other_document_citation_occurrences"], 2)
        self.assertEqual(metrics["average_distinct_other_documents_cited_per_source_file"], 0.5)
        self.assertEqual(metrics["average_distinct_source_files_citing_each_cited_document"], 1.0)
        self.assertEqual(metrics["average_intext_citation_occurrences_per_local_release18_canonical_document"], 1.0)


if __name__ == "__main__":
    unittest.main()
