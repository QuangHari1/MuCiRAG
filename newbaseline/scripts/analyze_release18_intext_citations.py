"""Count in-text ``[n]`` citations and their document targets in full GSMA Rel-18.

The ReferenceSeries JSON is used as the authoritative mapping from a source
document's bibliography number to its 3GPP target document.  Literal ``[n]``
markers are counted from raw Markdown only in normal document headings:
References, Normative References, and Informative References sections are
excluded.  This avoids confusing bibliography entries with actual in-text
citations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analyze_release18_references import write_pdf
from newbaseline.src.settings import load_settings


SETTINGS = load_settings()
DEFAULT_REFERENCE_DIR = SETTINGS.dataset_dir / "3gpp/Reference/Rel-18"
DEFAULT_RAW_DIR = SETTINGS.release_dir
DEFAULT_METADATA_DIR = SETTINGS.dataset_dir / "3gpp/Metadata/Rel-18"
DEFAULT_OUTPUT_DIR = SETTINGS.workspace_root / "newbaseline/results/release18-intext-citation-stats"
BRACKET_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
REFERENCE_TITLES = {
    "reference",
    "references",
    "normative reference",
    "normative references",
    "informative reference",
    "informative references",
}


def is_reference_heading(heading: dict[str, object]) -> bool:
    """Return whether this metadata heading is a bibliography section."""
    title = heading.get("title")
    return isinstance(title, str) and title.strip().lower() in REFERENCE_TITLES


def intext_sections(raw_markdown: str, headings: list[dict[str, object]]) -> str:
    """Return content from every non-reference heading in one raw document."""
    lines = raw_markdown.splitlines()
    ordered_headings = sorted(headings, key=lambda heading: int(heading["line"]))
    sections: list[str] = []
    for index, heading in enumerate(ordered_headings):
        line = int(heading["line"])
        next_line = int(ordered_headings[index + 1]["line"]) if index + 1 < len(ordered_headings) else len(lines) + 1
        if not is_reference_heading(heading):
            # Heading metadata is one-indexed; slicing from ``line`` omits the heading itself.
            sections.append("\n".join(lines[line:next_line - 1]))
    return "\n".join(sections)


def load_bibliographies(reference_dir: Path) -> dict[str, dict[str, str]]:
    """Load source-document ``[n] -> target document ID`` maps from ReferenceSeries."""
    paths = sorted(reference_dir.glob("ReferenceSeries*.json"))
    if not paths:
        raise FileNotFoundError(f"No ReferenceSeries JSON files found in {reference_dir}")

    bibliographies: dict[str, dict[str, str]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = payload.get("documents")
        if not isinstance(documents, list):
            raise ValueError(f"Malformed documents array in {path}")
        for document in documents:
            source_id = document.get("document_id")
            entries = document.get("bibliography")
            if not isinstance(source_id, str) or not isinstance(entries, list):
                raise ValueError(f"Malformed document entry in {path}: {document}")
            mapping: dict[str, str] = {}
            for entry in entries:
                reference_id = entry.get("reference_id")
                target_id = entry.get("document_id")
                if isinstance(reference_id, str) and isinstance(target_id, str):
                    mapping[reference_id] = target_id
            bibliographies[source_id] = mapping
    return bibliographies


def percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 3) if denominator else 0.0


def canonical_document_id(document_id: str) -> str:
    """Collapse a GSMA source version variant such as ``38521-3`` to ``38521``."""
    return document_id.split("-", maxsplit=1)[0]


def analyze(reference_dir: Path, raw_dir: Path, metadata_dir: Path) -> dict[str, object]:
    """Calculate literal in-text markers, outgoing target diversity, and incoming citations."""
    bibliographies = load_bibliographies(reference_dir)
    source_ids = set(bibliographies)
    local_canonical_ids = {canonical_document_id(source_id) for source_id in source_ids}
    rows: list[dict[str, object]] = []
    incoming_occurrences: Counter[str] = Counter()
    incoming_sources: defaultdict[str, set[str]] = defaultdict(set)
    local_incoming_occurrences: Counter[str] = Counter()
    local_incoming_sources: defaultdict[str, set[str]] = defaultdict(set)

    for raw_path in sorted(raw_dir.rglob("raw.md")):
        relative_document = raw_path.relative_to(raw_dir).parent
        metadata_path = metadata_dir / relative_document / "headings.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing heading metadata for {raw_path}: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_id = metadata.get("document_id")
        headings = metadata.get("headings")
        if not isinstance(source_id, str) or not isinstance(headings, list):
            raise ValueError(f"Malformed heading metadata in {metadata_path}")
        if source_id not in bibliographies:
            raise ValueError(f"No ReferenceSeries document found for {source_id}")

        content = intext_sections(raw_path.read_text(encoding="utf-8", errors="ignore"), headings)
        marker_ids = BRACKET_CITATION_PATTERN.findall(content)
        target_map = bibliographies[source_id]
        resolved_targets = [target_map[marker_id] for marker_id in marker_ids if marker_id in target_map and target_map[marker_id] != source_id]
        unique_targets = sorted(set(resolved_targets))
        for target_id in resolved_targets:
            incoming_occurrences[target_id] += 1
            incoming_sources[target_id].add(source_id)
            if target_id in local_canonical_ids:
                local_incoming_occurrences[target_id] += 1
                local_incoming_sources[target_id].add(source_id)
        rows.append(
            {
                "source_document_id": source_id,
                "source_path": str(raw_path),
                "intext_bracket_markers": len(marker_ids),
                "resolved_other_document_citation_occurrences": len(resolved_targets),
                "unresolved_or_self_bracket_markers": len(marker_ids) - len(resolved_targets),
                "unique_other_documents_cited": unique_targets,
            }
        )

    total_sources = len(rows)
    total_markers = sum(int(row["intext_bracket_markers"]) for row in rows)
    total_resolved = sum(int(row["resolved_other_document_citation_occurrences"]) for row in rows)
    total_unique_outgoing = sum(len(row["unique_other_documents_cited"]) for row in rows)
    cited_targets = sorted(incoming_occurrences)
    cited_target_count = len(cited_targets)
    top_n = 15

    return {
        "methodology": {
            "source_scope": "All raw.md documents represented by ReferenceSeries files",
            "marker_definition": "Literal [integer] markers in non-reference headings only",
            "target_mapping": "ReferenceSeries bibliography maps each source [n] to a 3GPP target document ID",
            "excluded_sections": sorted(REFERENCE_TITLES),
            "excluded_targets": "Self-citations are excluded from other-document counts; markers without a ReferenceSeries target remain in the literal-marker count but cannot be attributed",
        },
        "input": {
            "reference_dir": str(reference_dir),
            "raw_dir": str(raw_dir),
            "source_documents": total_sources,
        },
        "metrics": {
            "total_intext_bracket_markers": total_markers,
            "average_intext_bracket_markers_per_source_file": round(total_markers / total_sources, 3),
            "total_resolved_other_document_citation_occurrences": total_resolved,
            "percent_bracket_markers_resolved_to_other_documents": percent(total_resolved, total_markers),
            "average_distinct_other_documents_cited_per_source_file": round(total_unique_outgoing / total_sources, 3),
            "documents_with_at_least_one_resolved_other_document_citation": sum(
                bool(row["unique_other_documents_cited"]) for row in rows
            ),
            "cited_target_documents": cited_target_count,
            "average_intext_citation_occurrences_per_cited_document": round(
                total_resolved / cited_target_count, 3
            ) if cited_target_count else 0.0,
            "average_distinct_source_files_citing_each_cited_document": round(
                sum(len(sources) for sources in incoming_sources.values()) / cited_target_count, 3
            ) if cited_target_count else 0.0,
            "local_release18_canonical_target_documents": len(local_canonical_ids),
            "average_intext_citation_occurrences_per_local_release18_canonical_document": round(
                sum(local_incoming_occurrences.values()) / len(local_canonical_ids), 3
            ) if local_canonical_ids else 0.0,
            "average_distinct_source_files_citing_each_local_release18_canonical_document": round(
                sum(len(sources) for sources in local_incoming_sources.values()) / len(local_canonical_ids), 3
            ) if local_canonical_ids else 0.0,
        },
        "top_cited_documents_by_occurrences": [
            {
                "target_document_id": target_id,
                "intext_citation_occurrences": count,
                "distinct_source_files": len(incoming_sources[target_id]),
            }
            for target_id, count in incoming_occurrences.most_common(top_n)
        ],
        "documents": rows,
    }


def pdf_lines(report: dict[str, object]) -> list[str]:
    """Render the requested three measures without bibliography-list pollution."""
    metrics = report["metrics"]
    input_data = report["input"]
    lines = [
        "GSMA Release-18: in-text [n] citation statistics",
        "",
        f"Source files: {input_data['source_documents']}",
        "Only normal document headings were scanned. Reference sections were excluded.",
        "",
        f"Average literal [n] markers per source file: {metrics['average_intext_bracket_markers_per_source_file']}",
        f"Average distinct other documents cited per source file: {metrics['average_distinct_other_documents_cited_per_source_file']}",
        f"Average in-text citation occurrences received per cited document: {metrics['average_intext_citation_occurrences_per_cited_document']}",
        f"Average distinct source files citing each cited document: {metrics['average_distinct_source_files_citing_each_cited_document']}",
        f"Average occurrences received per local Rel-18 canonical document: {metrics['average_intext_citation_occurrences_per_local_release18_canonical_document']}",
        f"Average source files citing each local Rel-18 canonical document: {metrics['average_distinct_source_files_citing_each_local_release18_canonical_document']}",
        "",
        f"Total literal [n] markers: {metrics['total_intext_bracket_markers']}",
        f"Resolved markers pointing to another document: {metrics['total_resolved_other_document_citation_occurrences']} ({metrics['percent_bracket_markers_resolved_to_other_documents']}%)",
        f"Documents with at least one resolved other-document citation: {metrics['documents_with_at_least_one_resolved_other_document_citation']}",
        "",
        "Most cited target documents (content citations only)",
        "Target ID  Occurrences  Source files",
    ]
    for row in report["top_cited_documents_by_occurrences"]:
        lines.append(
            f"{row['target_document_id']:<9}  {row['intext_citation_occurrences']:>11}  {row['distinct_source_files']:>12}"
        )
    lines.extend(
        [
            "",
            "Mapping note: ReferenceSeries supplies [n] to target-ID mappings. A literal marker",
            "without a known 3GPP target is retained in the marker count but not in target metrics.",
        ]
    )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args.reference_dir, args.raw_dir, args.metadata_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "release18_intext_citation_statistics.json"
    pdf_path = args.output_dir / "release18_intext_citation_statistics.pdf"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_pdf(pdf_path, pdf_lines(report))
    metrics = report["metrics"]
    print(f"Analysed {report['input']['source_documents']} Release-18 source files.")
    print(f"Average in-text [n] markers/source file: {metrics['average_intext_bracket_markers_per_source_file']}")
    print(f"Average distinct cited documents/source file: {metrics['average_distinct_other_documents_cited_per_source_file']}")
    print(f"PDF: {pdf_path}")
    print(f"JSON audit: {json_path}")


if __name__ == "__main__":
    main()
