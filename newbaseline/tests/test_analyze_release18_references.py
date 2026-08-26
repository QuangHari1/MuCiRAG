import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/analyze_release18_references.py"
SPEC = importlib.util.spec_from_file_location("analyze_release18_references", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeRelease18ReferencesTest(unittest.TestCase):
    def test_extract_reference_targets_normalizes_etsi_and_ignores_body_mentions(self) -> None:
        markdown = """## --- 1 Scope
3GPP TS 99.999 is only an in-text mention.
## --- 2 Normative references
- [1] 3GPP TS 23.501: Example.
- [2] ETSI TS 123 502: Example.
- [3] 3GPP TR 23.501: Duplicate target.
## --- 3 Terms
3GPP TS 88.888 is outside the reference section.
"""
        targets, mentions = MODULE.extract_reference_targets(markdown)

        self.assertEqual(targets, {"23501", "23502"})
        self.assertEqual(mentions, 3)
        self.assertEqual(MODULE.count_bracketed_reference_entries(markdown), 3)

    def test_summarize_excludes_self_links_and_resolves_local_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            release_dir = Path(temporary_directory) / "Rel-18"
            documents = {
                "23_series/23501/raw.md": """## 2 References
3GPP TS 23.501\n3GPP TS 23.502\nETSI TS 123 999\n## 3 Definitions\n""",
                "23_series/23502/raw.md": """## --- References
3GPP TR 23.501\n## --- 3 Terms\n""",
                "24_series/24008/raw.md": "## 1 Scope\nNo reference section.\n",
            }
            for relative_path, content in documents.items():
                path = release_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            report = MODULE.summarize(release_dir, top_n=2)

        self.assertEqual(report["input"]["raw_markdown_documents"], 3)
        self.assertEqual(report["metrics"]["unique_other_3gpp_document_links"], 3)
        self.assertEqual(report["metrics"]["unique_local_release18_document_links"], 2)
        self.assertEqual(report["metrics"]["documents_with_any_other_3gpp_link"], 2)
        self.assertEqual(report["metrics"]["documents_linking_to_more_than_10_other_3gpp_documents"], 0)
        self.assertEqual(report["metrics"]["average_incoming_local_release18_links_per_raw_file"], 0.667)
        self.assertEqual(report["metrics"]["bracketed_reference_entries"], 0)
        first_row = report["documents"][0]
        self.assertEqual(first_row["other_3gpp_targets"], ["23502", "23999"])
        self.assertEqual(first_row["local_release18_targets"], ["23502"])

    def test_write_pdf_creates_a_valid_pdf_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "report.pdf"
            MODULE.write_pdf(pdf_path, ["Reference statistics", "line two"])
            content = pdf_path.read_bytes()

        self.assertTrue(content.startswith(b"%PDF-1.4"))
        self.assertIn(b"%%EOF", content)


if __name__ == "__main__":
    unittest.main()
