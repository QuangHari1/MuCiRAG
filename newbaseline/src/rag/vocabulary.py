"""Paper-compatible 3GPP terminology and abbreviation enrichment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document


def canonical_document_id(value: str) -> str | None:
    """Normalize catalog paths and retrieval metadata to one document key."""
    normalized = value.strip().replace("\\", "/").rstrip("/")
    if not normalized:
        return None
    return normalized.rsplit("/", 1)[-1] or None


def canonical_expansion_key(value: str) -> str:
    """Collapse case, punctuation, and hyphen-only spelling variants."""
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


@dataclass(frozen=True)
class AmbiguousAbbreviationCandidate:
    """One Release-18 expansion plus the series in which it is defined."""

    expansion: str
    source_series: tuple[str, ...]
    source_count: int
    source_document_ids: tuple[str, ...] = ()


class Vocabulary:
    def __init__(
        self,
        terms: dict[str, str],
        abbreviations: dict[str, str],
        ambiguous_abbreviations: dict[str, list[AmbiguousAbbreviationCandidate]] | None = None,
    ):
        self.terms = terms
        self.abbreviations = abbreviations
        self.ambiguous_abbreviations = ambiguous_abbreviations or {}

    @classmethod
    def from_docx(cls, path: Path) -> "Vocabulary":
        terms: dict[str, str] = {}
        abbreviations: dict[str, str] = {}
        in_terms = False
        in_abbreviations = False
        reference_count = 0
        for paragraph in Document(path).paragraphs:
            text = paragraph.text.strip()
            if "References" in text:
                reference_count += 1
            if reference_count < 2:
                continue
            if "Terms and definitions" in text:
                in_terms, in_abbreviations = True, False
            elif "Abbreviations" in text:
                in_terms, in_abbreviations = False, True
            elif in_terms and ":" in text:
                term, definition = text.split(":", 1)
                terms[term.strip()] = definition.strip().rstrip(".")
            elif in_abbreviations and "\t" in text:
                abbreviation, definition = text.split("\t", 1)
                if len(abbreviation.strip()) > 1:
                    abbreviations[abbreviation.strip()] = definition.strip()
        return cls(terms, abbreviations)

    @classmethod
    def from_release18_assets(cls, definitions_path: Path, abbreviations_path: Path) -> "Vocabulary":
        """Load paper term definitions plus safe, unambiguous Release-18 abbreviations.

        Every candidate expansion remains in the catalog. This runtime mode
        injects an abbreviation only when it has exactly one distinct candidate;
        ambiguous acronyms stay in the question unchanged rather than receiving
        an arbitrary expansion.
        """
        definitions_payload = json.loads(definitions_path.read_text(encoding="utf-8"))
        definition_rows = definitions_payload.get("terms")
        if not isinstance(definition_rows, list):
            raise ValueError(f"Missing terms list in {definitions_path}")
        terms: dict[str, str] = {}
        for index, row in enumerate(definition_rows):
            if not isinstance(row, dict) or not isinstance(row.get("term"), str) or not isinstance(
                row.get("definition"), str
            ):
                raise ValueError(f"Invalid term at {definitions_path}:{index}")
            terms[row["term"]] = row["definition"]

        abbreviations_payload = json.loads(abbreviations_path.read_text(encoding="utf-8"))
        candidates_by_acronym = abbreviations_payload.get("acronyms")
        if not isinstance(candidates_by_acronym, dict):
            raise ValueError(f"Missing acronyms object in {abbreviations_path}")
        abbreviations: dict[str, str] = {}
        ambiguous_abbreviations: dict[str, list[AmbiguousAbbreviationCandidate]] = {}
        for acronym, candidates in candidates_by_acronym.items():
            if not isinstance(acronym, str) or not isinstance(candidates, list) or not candidates:
                continue
            parsed_candidates = cls._parse_candidates(candidates)
            if len(parsed_candidates) == 1:
                abbreviations[acronym] = parsed_candidates[0].expansion
            elif len(parsed_candidates) > 1:
                ambiguous_abbreviations[acronym] = parsed_candidates
        return cls(terms, abbreviations, ambiguous_abbreviations)

    @staticmethod
    def _parse_candidates(candidates: list[Any]) -> list[AmbiguousAbbreviationCandidate]:
        grouped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            expansion = candidate.get("expansion")
            sources = candidate.get("sources", [])
            if not isinstance(expansion, str) or not expansion.strip() or not isinstance(sources, list):
                continue
            cleaned_expansion = expansion.strip()
            key = canonical_expansion_key(cleaned_expansion)
            if not key:
                continue
            row = grouped.setdefault(
                key,
                {
                    "expansion": cleaned_expansion,
                    "representative_source_count": len(sources),
                    "source_series": set(),
                    "source_count": 0,
                    "source_document_ids": set(),
                },
            )
            if len(sources) > row["representative_source_count"]:
                row["expansion"] = cleaned_expansion
                row["representative_source_count"] = len(sources)
            row["source_count"] += len(sources)
            row["source_series"].update(
                source["series"]
                for source in sources
                if isinstance(source, dict) and isinstance(source.get("series"), str)
            )
            row["source_document_ids"].update(
                document_id
                for source in sources
                if isinstance(source, dict)
                and isinstance(source.get("source_document_key"), str)
                and (document_id := canonical_document_id(source["source_document_key"])) is not None
            )
        return [
            AmbiguousAbbreviationCandidate(
                expansion=str(row["expansion"]),
                source_series=tuple(sorted(row["source_series"])),
                source_count=int(row["source_count"]),
                source_document_ids=tuple(sorted(row["source_document_ids"])),
            )
            for row in grouped.values()
        ]

    def enrich(self, question: str) -> str:
        normalized = re.sub(r"[!()\-\[\]{};:'\"\\,<>./?@#$%^&*_~]", "", question.lower())
        matched_terms = [
            f"{term}: {definition}"
            for term, definition in self.terms.items()
            if re.sub(r"[!()\-\[\]{};:'\"\\,<>./?@#$%^&*_~]", "", term.lower()) in normalized
        ]
        words = self._words(question)
        matched_abbreviations = [
            f"{word}: {self.abbreviations[word]}" for word in words if word in self.abbreviations
        ]
        terms_text = "\n".join(matched_terms)
        abbreviations_text = "\n".join(matched_abbreviations)
        return (
            f"{question}\n\nTerms and Definitions:\n\n{terms_text}"
            f"\n\nAbbreviations:\n\n{abbreviations_text}\n"
        )

    @staticmethod
    def _words(question: str) -> list[str]:
        return re.sub(r"[!()\-\[\]{};:'\"\\,<>./?@#$%^&*_~]", "", question).split()
