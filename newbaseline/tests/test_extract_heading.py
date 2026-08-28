import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "extract_heading.py"
SPEC = importlib.util.spec_from_file_location("extract_heading", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EXTRACT_HEADING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACT_HEADING)


class ExtractHeadingTest(unittest.TestCase):
    def test_converter_marker_is_not_used_as_citation_heading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.md"
            raw_path.write_text("## 1527 1 Scope\nScope text\n## 1544 2 References\n", encoding="utf-8")
            headings = EXTRACT_HEADING.extract_headings(raw_path)

        self.assertEqual(
            headings,
            [
                {"heading": "1", "title": "Scope", "level": 1, "line": 1},
                {"heading": "2", "title": "References", "level": 1, "line": 3},
            ],
        )

    def test_normal_four_digit_number_is_unchanged_without_embedded_clause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.md"
            raw_path.write_text("## 1234 ordinary title without a clause number\n", encoding="utf-8")
            headings = EXTRACT_HEADING.extract_headings(raw_path)

        self.assertEqual(headings[0]["heading"], "1234")
        self.assertEqual(headings[0]["title"], "ordinary title without a clause number")

    def test_clause_suffix_letter_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.md"
            raw_path.write_text("## 2341 6.1.3.10B Gateway Mobile Location Centre\n", encoding="utf-8")
            headings = EXTRACT_HEADING.extract_headings(raw_path)

        self.assertEqual(headings[0]["heading"], "6.1.3.10B")
        self.assertEqual(headings[0]["level"], 4)

    def test_bold_heading_number_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.md"
            raw_path.write_text("## **5.2 Interoperability**\n", encoding="utf-8")
            headings = EXTRACT_HEADING.extract_headings(raw_path)

        self.assertEqual(headings[0]["heading"], "5.2")
        self.assertEqual(headings[0]["title"], "Interoperability")


if __name__ == "__main__":
    unittest.main()
