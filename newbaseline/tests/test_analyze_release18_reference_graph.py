import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/analyze_release18_reference_graph.py"
SPEC = importlib.util.spec_from_file_location("analyze_release18_reference_graph", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeRelease18ReferenceGraphTest(unittest.TestCase):
    def test_analyze_excludes_bibliography_internal_and_self_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            reference_dir = Path(temporary_directory)
            (reference_dir / "ReferenceSeries10.json").write_text(json.dumps({"documents": [
                {"document_id": "10001", "links": [
                    {"type": "bibliography", "document_id": "20001"},
                    {"type": "internal", "document_id": "10001"},
                    {"type": "external", "document_id": "20001"},
                    {"type": "external", "document_id": "20001"},
                    {"type": "external", "document_id": "10001"},
                ]},
                {"document_id": "20001", "links": []},
            ]}), encoding="utf-8")
            report = MODULE.analyze(reference_dir)

        metrics = report["metrics"]
        self.assertEqual(metrics["total_external_citation_link_records"], 2)
        self.assertEqual(metrics["average_external_citation_link_records_per_source_file"], 1.0)
        self.assertEqual(metrics["average_distinct_other_documents_cited_per_source_file"], 0.5)
        self.assertEqual(metrics["total_bibliography_link_records"], 1)
        self.assertEqual(metrics["average_bibliography_link_records_per_source_file"], 0.5)
        self.assertEqual(metrics["average_distinct_source_files_citing_each_cited_document"], 1.0)
        summary = MODULE.compact_summary(report)
        self.assertNotIn("documents", summary)
        self.assertEqual(summary["averages_per_source_file"]["external_link_records"], 1.0)


if __name__ == "__main__":
    unittest.main()
