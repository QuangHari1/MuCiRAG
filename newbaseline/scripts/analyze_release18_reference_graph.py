"""Summarise in-text cross-document links stored in Release-18 ReferenceSeries.

Only ``links.type == 'external'`` is counted. ``bibliography`` is the
References section and ``internal`` points to a clause in the same document,
so both are deliberately excluded. A link row is a stored citation-graph edge
deduplicated by the chunking pipeline; it is not a raw character count of
``[n]`` in Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_OUTPUT_DIR = SETTINGS.workspace_root / "newbaseline/results/release18-reference-graph-stats"


def canonical_document_id(document_id: str) -> str:
    """Collapse a GSMA source variant such as ``38521-3`` to ``38521``."""
    return document_id.split("-", maxsplit=1)[0]


def load_documents(reference_dir: Path) -> list[dict[str, object]]:
    """Load every document entry from the stored ReferenceSeries files."""
    paths = sorted(reference_dir.glob("ReferenceSeries*.json"))
    if not paths:
        raise FileNotFoundError(f"No ReferenceSeries JSON files found in {reference_dir}")
    documents: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("documents")
        if not isinstance(entries, list):
            raise ValueError(f"Malformed documents array in {path}")
        for entry in entries:
            if not isinstance(entry.get("document_id"), str) or not isinstance(entry.get("links"), list):
                raise ValueError(f"Malformed document entry in {path}: {entry}")
            documents.append(entry)
    return documents


def analyze(reference_dir: Path) -> dict[str, object]:
    """Compute outgoing target diversity and incoming citation frequencies."""
    documents = load_documents(reference_dir)
    source_ids = {str(document["document_id"]) for document in documents}
    local_canonical_ids = {canonical_document_id(source_id) for source_id in source_ids}
    incoming_records: Counter[str] = Counter()
    incoming_sources: defaultdict[str, set[str]] = defaultdict(set)
    bibliography_incoming_records: Counter[str] = Counter()
    bibliography_incoming_sources: defaultdict[str, set[str]] = defaultdict(set)
    local_incoming_records: Counter[str] = Counter()
    local_incoming_sources: defaultdict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, object]] = []

    for document in documents:
        source_id = str(document["document_id"])
        external_links = [
            link
            for link in document["links"]
            if isinstance(link, dict)
            and link.get("type") == "external"
            and isinstance(link.get("document_id"), str)
            and link["document_id"] != source_id
        ]
        bibliography_links = [
            link
            for link in document["links"]
            if isinstance(link, dict)
            and link.get("type") == "bibliography"
            and isinstance(link.get("document_id"), str)
            and link["document_id"] != source_id
        ]
        external_targets = sorted({str(link["document_id"]) for link in external_links})
        bibliography_targets = sorted({str(link["document_id"]) for link in bibliography_links})
        for link in external_links:
            target_id = str(link["document_id"])
            incoming_records[target_id] += 1
            incoming_sources[target_id].add(source_id)
            if target_id in local_canonical_ids:
                local_incoming_records[target_id] += 1
                local_incoming_sources[target_id].add(source_id)
        for link in bibliography_links:
            target_id = str(link["document_id"])
            bibliography_incoming_records[target_id] += 1
            bibliography_incoming_sources[target_id].add(source_id)
        rows.append(
            {
                "source_document_id": source_id,
                "external_citation_link_records": len(external_links),
                "external_unique_other_document_count": len(external_targets),
                "external_unique_other_document_ids": external_targets,
                "bibliography_link_records": len(bibliography_links),
                "bibliography_unique_other_document_count": len(bibliography_targets),
                "bibliography_unique_other_document_ids": bibliography_targets,
            }
        )

    source_count = len(rows)
    total_links = sum(int(row["external_citation_link_records"]) for row in rows)
    total_unique_targets = sum(int(row["external_unique_other_document_count"]) for row in rows)
    total_bibliography_links = sum(int(row["bibliography_link_records"]) for row in rows)
    total_unique_bibliography_targets = sum(int(row["bibliography_unique_other_document_count"]) for row in rows)
    cited_target_count = len(incoming_records)
    return {
        "methodology": {
            "source_scope": "All documents in ReferenceSeries*.json",
            "included_link_type": "external",
            "excluded_link_types": ["bibliography", "internal"],
            "other_document_rule": "External rows linking to the source document itself are excluded",
            "link_record_unit": "Stored graph edge deduplicated per source chunk, target document, and target heading",
        },
        "input": {
            "reference_dir": str(reference_dir),
            "source_documents": source_count,
            "local_release18_canonical_documents": len(local_canonical_ids),
        },
        "metrics": {
            "total_external_citation_link_records": total_links,
            "average_external_citation_link_records_per_source_file": round(total_links / source_count, 3),
            "average_distinct_other_documents_cited_per_source_file": round(total_unique_targets / source_count, 3),
            "documents_with_at_least_one_external_citation": sum(
                int(row["external_unique_other_document_count"]) > 0 for row in rows
            ),
            "total_bibliography_link_records": total_bibliography_links,
            "average_bibliography_link_records_per_source_file": round(total_bibliography_links / source_count, 3),
            "average_distinct_bibliography_documents_per_source_file": round(
                total_unique_bibliography_targets / source_count, 3
            ),
            "documents_with_at_least_one_bibliography_link": sum(
                int(row["bibliography_unique_other_document_count"]) > 0 for row in rows
            ),
            "cited_target_documents": cited_target_count,
            "average_external_citation_link_records_per_cited_document": round(total_links / cited_target_count, 3) if cited_target_count else 0.0,
            "average_distinct_source_files_citing_each_cited_document": round(
                sum(len(sources) for sources in incoming_sources.values()) / cited_target_count, 3
            ) if cited_target_count else 0.0,
            "average_distinct_source_files_listing_each_bibliography_document": round(
                sum(len(sources) for sources in bibliography_incoming_sources.values())
                / len(bibliography_incoming_sources),
                3,
            ) if bibliography_incoming_sources else 0.0,
            "average_external_citation_link_records_per_local_release18_canonical_document": round(
                sum(local_incoming_records.values()) / len(local_canonical_ids), 3
            ) if local_canonical_ids else 0.0,
            "average_distinct_source_files_citing_each_local_release18_canonical_document": round(
                sum(len(sources) for sources in local_incoming_sources.values()) / len(local_canonical_ids), 3
            ) if local_canonical_ids else 0.0,
        },
        "top_cited_documents": [
            {
                "target_document_id": target_id,
                "external_citation_link_records": count,
                "distinct_source_files": len(incoming_sources[target_id]),
            }
            for target_id, count in incoming_records.most_common(15)
        ],
        "documents": rows,
    }


def pdf_lines(report: dict[str, object]) -> list[str]:
    metrics = report["metrics"]
    input_data = report["input"]
    lines = [
        "GSMA Release-18: ReferenceSeries external-link statistics", "",
        f"ReferenceSeries source documents: {input_data['source_documents']}",
        "Included: external links only. Excluded: bibliography and internal links.", "",
        f"Average external link records per source file: {metrics['average_external_citation_link_records_per_source_file']}",
        f"Average distinct other documents cited per source file: {metrics['average_distinct_other_documents_cited_per_source_file']}",
        f"Average bibliography link records per source file: {metrics['average_bibliography_link_records_per_source_file']}",
        f"Average distinct bibliography documents per source file: {metrics['average_distinct_bibliography_documents_per_source_file']}",
        f"Average external link records received per local Rel-18 canonical document: {metrics['average_external_citation_link_records_per_local_release18_canonical_document']}",
        f"Average distinct source files citing each local Rel-18 canonical document: {metrics['average_distinct_source_files_citing_each_local_release18_canonical_document']}", "",
        f"Total external link records: {metrics['total_external_citation_link_records']}",
        f"Total bibliography link records: {metrics['total_bibliography_link_records']}",
        "", "Most cited targets", "Target ID  Link records  Source files",
    ]
    for row in report["top_cited_documents"]:
        lines.append(f"{row['target_document_id']:<9}  {row['external_citation_link_records']:>12}  {row['distinct_source_files']:>12}")
    lines.extend(["", "A link record is a stored citation-graph edge, not a literal raw [n] character count."])
    return lines


def compact_summary(report: dict[str, object]) -> dict[str, object]:
    """Return a small aggregate-only JSON document, without per-file rows."""
    input_data = report["input"]
    metrics = report["metrics"]
    return {
        "scope": {
            "reference_dir": input_data["reference_dir"],
            "source_documents": input_data["source_documents"],
            "local_release18_canonical_documents": input_data["local_release18_canonical_documents"],
            "included_link_type": "external",
            "separate_reference_list_link_type": "bibliography",
            "excluded_link_type": "internal",
            "link_record_unit": report["methodology"]["link_record_unit"],
        },
        "averages_per_source_file": {
            "external_link_records": metrics["average_external_citation_link_records_per_source_file"],
            "distinct_other_documents_cited_by_external_links": metrics[
                "average_distinct_other_documents_cited_per_source_file"
            ],
            "bibliography_link_records": metrics["average_bibliography_link_records_per_source_file"],
            "distinct_bibliography_documents": metrics[
                "average_distinct_bibliography_documents_per_source_file"
            ],
        },
        "incoming_citation_averages": {
            "distinct_source_files_citing_each_cited_target": metrics[
                "average_distinct_source_files_citing_each_cited_document"
            ],
            "distinct_source_files_citing_each_local_release18_canonical_document": metrics[
                "average_distinct_source_files_citing_each_local_release18_canonical_document"
            ],
        },
        "global_totals": {
            "external_link_records": metrics["total_external_citation_link_records"],
            "bibliography_link_records": metrics["total_bibliography_link_records"],
            "source_files_with_external_citation": metrics[
                "documents_with_at_least_one_external_citation"
            ],
            "source_files_with_bibliography_link": metrics[
                "documents_with_at_least_one_bibliography_link"
            ],
            "cited_target_documents": metrics["cited_target_documents"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_document_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the per-document counts for direct spreadsheet inspection."""
    fieldnames = [
        "source_document_id",
        "external_citation_link_records",
        "external_unique_other_document_count",
        "bibliography_link_records",
        "bibliography_unique_other_document_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def main() -> None:
    args = parse_args()
    report = analyze(args.reference_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "release18_reference_graph_statistics.json"
    summary_path = args.output_dir / "release18_reference_graph_summary.json"
    pdf_path = args.output_dir / "release18_reference_graph_statistics.pdf"
    csv_path = args.output_dir / "release18_reference_graph_by_document.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(compact_summary(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_document_csv(csv_path, report["documents"])
    write_pdf(pdf_path, pdf_lines(report))
    metrics = report["metrics"]
    print(f"Analysed {report['input']['source_documents']} ReferenceSeries source documents.")
    print(f"Average external link records/source file: {metrics['average_external_citation_link_records_per_source_file']}")
    print(f"Average distinct targets/source file: {metrics['average_distinct_other_documents_cited_per_source_file']}")
    print(f"PDF: {pdf_path}")
    print(f"JSON audit: {json_path}")
    print(f"Compact summary JSON: {summary_path}")
    print(f"Document CSV: {csv_path}")


if __name__ == "__main__":
    main()
