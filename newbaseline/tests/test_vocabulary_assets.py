"""Tests for the opt-in two-asset vocabulary mode."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from newbaseline.src.rag.vocabulary import Vocabulary


class VocabularyAssetsTests(unittest.TestCase):
    def test_spelling_only_expansion_aliases_are_merged_before_ambiguity_check(self) -> None:
        candidates = Vocabulary._parse_candidates(
            [
                {
                    "expansion": "Long-Term Evolution",
                    "sources": [{"series": "23", "source_document_key": "23_series/23001"}],
                },
                {
                    "expansion": "long term evolution",
                    "sources": [{"series": "36", "source_document_key": "36_series/36001"}],
                },
            ]
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_count, 2)
        self.assertEqual(candidates[0].source_series, ("23", "36"))
        self.assertEqual(candidates[0].source_document_ids, ("23001", "36001"))

    def test_release18_assets_keep_ambiguous_acronyms_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            definitions = root / "definitions.json"
            abbreviations = root / "abbreviations.json"
            definitions.write_text(
                json.dumps({"terms": [{"term": "PDU Session", "definition": "A session definition"}]}),
                encoding="utf-8",
            )
            abbreviations.write_text(
                json.dumps(
                    {
                        "acronyms": {
                            "SMF": [{"expansion": "Session Management Function"}],
                            "AMF": [
                                {
                                    "expansion": "Access and Mobility Management Function",
                                    "sources": [{"series": "23", "source_document_key": "23_series/23003"}],
                                },
                                {
                                    "expansion": "Authentication Management Field",
                                    "sources": [{"series": "31", "source_document_key": "31_series/31102"}],
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            vocabulary = Vocabulary.from_release18_assets(definitions, abbreviations)

        enriched = vocabulary.enrich("Does AMF coordinate a PDU Session with SMF?")
        self.assertIn("PDU Session: A session definition", enriched)
        self.assertIn("SMF: Session Management Function", enriched)
        self.assertNotIn("AMF:", enriched)
        self.assertEqual(
            vocabulary.ambiguous_abbreviations["AMF"][0].source_document_ids,
            ("23003",),
        )

if __name__ == "__main__":
    unittest.main()
