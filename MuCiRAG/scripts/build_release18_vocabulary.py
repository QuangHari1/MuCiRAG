"""Build a provenance-preserving Release-18 abbreviation catalog from Markdown.

The paper vocabulary is a single Release-17 DOCX parsed into one definition per
acronym. This utility deliberately keeps every distinct expansion and its
source table so ambiguous acronyms can later be resolved (or safely abstained
from) at query time.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset/3gpp/marked/Rel-18"
DEFAULT_OUTPUT = PROJECT_ROOT / "MuCiRAG/resources/3GPP_vocabulary_release18.json"
HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
ACRONYM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .+/_-]{1,63}$")


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def display_path(path: Path, input_root: Path) -> str:
    """Prefer workspace-relative provenance, but keep custom data roots usable."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.relative_to(input_root).as_posix()


def parse_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def is_abbreviation_heading(title: str) -> bool:
    return "abbreviation" in title.lower()


def is_definition_heading(title: str) -> bool:
    return "definition" in title.lower()


def extract_abbreviations(path: Path, input_root: Path, release: str) -> list[dict[str, Any]]:
    """Extract two-column abbreviation tables beneath abbreviation headings."""
    active_heading: str | None = None
    active_heading_level: int | None = None
    entries: list[dict[str, Any]] = []
    document_key = path.parent.relative_to(input_root).as_posix()
    series = path.parent.parent.name.removesuffix("_series")

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        heading = HEADING_PATTERN.match(line)
        if heading:
            title = heading.group("title")
            level = len(heading.group("marks"))
            if is_abbreviation_heading(title):
                active_heading = title
                active_heading_level = level
            elif is_definition_heading(title):
                active_heading = None
                active_heading_level = None
            elif active_heading_level is not None and level <= active_heading_level:
                active_heading = None
                active_heading_level = None
            continue
        if active_heading is None or not TABLE_ROW_PATTERN.match(line):
            continue
        cells = parse_table_cells(line)
        if len(cells) != 2 or is_separator_row(cells):
            continue
        acronym, expansion = (normalize_whitespace(cell) for cell in cells)
        if not ACRONYM_PATTERN.fullmatch(acronym) or len(expansion) < 3:
            continue
        entries.append(
            {
                "acronym": acronym,
                "expansion": expansion,
                "source_document_key": document_key,
                "source_path": display_path(path, input_root),
                "source_release": release,
                "series": series,
                "heading": active_heading,
                "line": line_number,
            }
        )
    return entries


def build_catalog(input_root: Path, release: str) -> dict[str, Any]:
    """Aggregate equal acronym/expansion rows while retaining every source."""
    # Use raw.md deliberately: this catalog records standard-source wording,
    # independent of any optional image/table verbalization used for chunks.
    files = sorted(input_root.glob("*_series/*/raw.md"))
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    source_rows = 0
    source_files_with_entries = 0
    for path in files:
        entries = extract_abbreviations(path, input_root, release)
        if entries:
            source_files_with_entries += 1
        source_rows += len(entries)
        for entry in entries:
            acronym = entry.pop("acronym")
            expansion = entry.pop("expansion")
            expansion_key = expansion.casefold()
            candidate = grouped[acronym].get(expansion_key)
            if candidate is None:
                candidate = {"expansion": expansion, "sources": []}
                grouped[acronym][expansion_key] = candidate
            candidate["sources"].append(entry)

    acronyms = {
        acronym: sorted(candidates.values(), key=lambda candidate: candidate["expansion"].casefold())
        for acronym, candidates in sorted(grouped.items())
    }
    ambiguous = {acronym: candidates for acronym, candidates in acronyms.items() if len(candidates) > 1}
    return {
        "schema_version": 1,
        "release": release,
        "input_directory": display_path(input_root, input_root),
        "source_files_scanned": len(files),
        "source_files_with_abbreviations": source_files_with_entries,
        "source_rows": source_rows,
        "unique_acronyms": len(acronyms),
        "unique_candidates": sum(len(candidates) for candidates in acronyms.values()),
        "ambiguous_acronyms": len(ambiguous),
        "acronyms": acronyms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release", default="18")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = build_catalog(args.input_dir, args.release)
    print(
        "Scanned "
        f"{catalog['source_files_scanned']} files; extracted {catalog['source_rows']} rows; "
        f"catalog={catalog['unique_acronyms']} acronyms / {catalog['unique_candidates']} candidates; "
        f"ambiguous={catalog['ambiguous_acronyms']}",
        flush=True,
    )
    if args.dry_run:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
