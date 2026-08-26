"""Measure document-to-document 3GPP reference links in the full GSMA Rel-18 corpus.

The analysis unit is one ``raw.md`` file under ``marked/Rel-18``.  It reads
only a document's References / Normative references section, extracts 3GPP and
ETSI TS/TR identifiers, and counts distinct *other* specification IDs per
source document.  A target is considered local only when its canonical
five-digit ID exists somewhere in the supplied full GSMA Release-18 corpus.

The script writes a human-readable PDF and a JSON audit file.  It deliberately
uses only the standard library so it can run in the existing baseline virtual
environment without adding a plotting or PDF dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.settings import load_settings


SETTINGS = load_settings()
DEFAULT_RELEASE_DIR = SETTINGS.release_dir
DEFAULT_OUTPUT_DIR = SETTINGS.workspace_root / "newbaseline/results/release18-reference-stats"
RAW_FILENAME = "raw.md"
DOCUMENT_DIRECTORY_PATTERN = re.compile(r"(?P<spec_id>\d{5})(?:-.+)?$")
STANDARD_REFERENCE_PATTERN = re.compile(
    r"\b(?:3GPP|ETSI)\s+(?:TS|TR)\s+(?P<number>\d{2,3}(?:[.\s-]?\d{3}))\b",
    re.IGNORECASE,
)
BRACKETED_REFERENCE_ENTRY_PATTERN = re.compile(r"(?m)^\s*(?:[-*+]\s*)?\[\s*\d+\s*\]")


def canonical_specification_id(value: str) -> str | None:
    """Normalise ``23.501`` and ETSI's ``123 501`` to the 3GPP ID ``23501``."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 5:
        return digits
    if len(digits) == 6 and digits.startswith("1"):
        return digits[1:]
    return None


def source_specification_id(raw_path: Path) -> str:
    """Get the canonical ID encoded by a GSMA raw Markdown directory name."""
    match = DOCUMENT_DIRECTORY_PATTERN.fullmatch(raw_path.parent.name)
    if match is None:
        raise ValueError(f"Unexpected GSMA document directory: {raw_path.parent}")
    return match.group("spec_id")


def heading_text(line: str) -> str | None:
    """Return Markdown heading text, accepting the corpus' decorative markers."""
    match = re.match(r"^\s*#{1,6}\s+(?P<text>.+?)\s*$", line)
    if match is None:
        return None
    return re.sub(r"[*_`]", "", match.group("text")).strip(" -")


def is_reference_heading(text: str) -> bool:
    """Recognise headings such as ``2 References`` and ``Normative references``."""
    return bool(
        re.match(
            r"^(?:\d+(?:\.\d+)*\s+)?(?:(?:normative|informative)\s+)?references?\b",
            text,
            re.IGNORECASE,
        )
    )


def reference_section(markdown: str) -> str:
    """Extract the first level-two References section, excluding subsequent clauses."""
    lines = markdown.splitlines()
    collecting = False
    section: list[str] = []
    for line in lines:
        text = heading_text(line)
        if not collecting:
            if text is not None and is_reference_heading(text):
                collecting = True
            continue
        if text is not None and line.lstrip().startswith("## "):
            break
        section.append(line)
    return "\n".join(section)


def extract_reference_targets(markdown: str) -> tuple[set[str], int]:
    """Return unique 3GPP target IDs and raw TS/TR mention count in References."""
    targets: set[str] = set()
    mentions = 0
    for match in STANDARD_REFERENCE_PATTERN.finditer(reference_section(markdown)):
        target = canonical_specification_id(match.group("number"))
        if target is not None:
            targets.add(target)
            mentions += 1
    return targets, mentions


def count_bracketed_reference_entries(markdown: str) -> int:
    """Count ``[1]``, ``[2]``, and similar entries in the References section."""
    return len(BRACKETED_REFERENCE_ENTRY_PATTERN.findall(reference_section(markdown)))


def percentile(values: list[int], fraction: float) -> float:
    """Linearly interpolated percentile, well-defined even for a single document."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[int]) -> list[dict[str, object]]:
    """Create fixed, directly comparable out-degree bands."""
    bands = (("0", 0, 0), ("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10),
             ("11-20", 11, 20), ("21-50", 21, 50), ("51+", 51, None))
    values = list(values)
    total = len(values)
    rows: list[dict[str, object]] = []
    for label, minimum, maximum in bands:
        count = sum(value >= minimum and (maximum is None or value <= maximum) for value in values)
        rows.append(
            {
                "band": label,
                "documents": count,
                "percent_of_documents": round(100 * count / total, 3) if total else 0.0,
            }
        )
    return rows


def percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 3) if denominator else 0.0


def summarize(release_dir: Path, top_n: int = 15) -> dict[str, object]:
    """Read the complete corpus and return audit rows plus aggregate link statistics."""
    raw_paths = sorted(release_dir.rglob(RAW_FILENAME))
    if not raw_paths:
        raise FileNotFoundError(f"No {RAW_FILENAME} files found under {release_dir}")

    local_ids = {source_specification_id(path) for path in raw_paths}
    document_rows: list[dict[str, object]] = []
    local_indegree: Counter[str] = Counter()

    for raw_path in raw_paths:
        source_id = source_specification_id(raw_path)
        markdown = raw_path.read_text(encoding="utf-8")
        targets, mentions = extract_reference_targets(markdown)
        other_targets = targets - {source_id}
        local_targets = other_targets & local_ids
        unavailable_targets = other_targets - local_ids
        for target in local_targets:
            local_indegree[target] += 1
        document_rows.append(
            {
                "source_path": str(raw_path),
                "source_specification_id": source_id,
                "bracketed_reference_entries": count_bracketed_reference_entries(markdown),
                "reference_mentions": mentions,
                "unique_3gpp_targets_including_self": sorted(targets),
                "other_3gpp_targets": sorted(other_targets),
                "local_release18_targets": sorted(local_targets),
                "targets_not_in_local_release18": sorted(unavailable_targets),
            }
        )

    outdegrees = [len(row["other_3gpp_targets"]) for row in document_rows]
    local_outdegrees = [len(row["local_release18_targets"]) for row in document_rows]
    sources_with_other = sum(value > 0 for value in outdegrees)
    sources_with_local = sum(value > 0 for value in local_outdegrees)
    total_other_edges = sum(outdegrees)
    total_local_edges = sum(local_outdegrees)
    total = len(document_rows)
    maximum_possible_edges = total * max(total - 1, 0)

    top_outbound = sorted(
        document_rows,
        key=lambda row: (-len(row["other_3gpp_targets"]), row["source_path"]),
    )[:top_n]
    top_inbound = [
        {
            "target_specification_id": target,
            "source_documents_linking_to_target": count,
        }
        for target, count in sorted(local_indegree.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    ]

    return {
        "methodology": {
            "source_scope": "All GSMA raw.md files under the supplied Release-18 directory",
            "unit_of_analysis": "One raw.md file; version variants are separate source documents",
            "link_definition": "Distinct 3GPP/ETSI TS or TR specification IDs found only in a References section",
            "other_document_rule": "A reference to the source's own canonical five-digit ID is excluded",
            "local_target_rule": "A target is local when its canonical ID appears in any supplied GSMA Release-18 raw.md directory",
            "exclusions": "URLs, RFCs, non-3GPP standards, in-text citations, and unrecognised reference formats are not counted",
        },
        "input": {
            "release_dir": str(release_dir),
            "raw_markdown_documents": total,
            "unique_canonical_specification_ids": len(local_ids),
        },
        "metrics": {
            "documents_with_any_other_3gpp_link": sources_with_other,
            "percent_documents_with_any_other_3gpp_link": percentage(sources_with_other, total),
            "documents_with_any_local_release18_link": sources_with_local,
            "percent_documents_with_any_local_release18_link": percentage(sources_with_local, total),
            "documents_linking_to_more_than_10_other_3gpp_documents": sum(value > 10 for value in outdegrees),
            "percent_documents_linking_to_more_than_10_other_3gpp_documents": percentage(sum(value > 10 for value in outdegrees), total),
            "documents_linking_to_more_than_10_local_release18_documents": sum(value > 10 for value in local_outdegrees),
            "percent_documents_linking_to_more_than_10_local_release18_documents": percentage(sum(value > 10 for value in local_outdegrees), total),
            "distinct_3gpp_reference_mentions": sum(int(row["reference_mentions"]) for row in document_rows),
            "bracketed_reference_entries": sum(int(row["bracketed_reference_entries"]) for row in document_rows),
            "average_bracketed_reference_entries_per_source_file": round(
                sum(int(row["bracketed_reference_entries"]) for row in document_rows) / total, 3
            ),
            "unique_other_3gpp_document_links": total_other_edges,
            "unique_local_release18_document_links": total_local_edges,
            "percent_other_links_resolved_inside_local_release18": percentage(total_local_edges, total_other_edges),
            "directed_link_density_percent": round(100 * total_other_edges / maximum_possible_edges, 6) if maximum_possible_edges else 0.0,
            "average_other_3gpp_links_per_document": round(statistics.mean(outdegrees), 3),
            "average_other_3gpp_links_per_source_file": round(statistics.mean(outdegrees), 3),
            "median_other_3gpp_links_per_document": round(statistics.median(outdegrees), 3),
            "p90_other_3gpp_links_per_document": round(percentile(outdegrees, 0.9), 3),
            "maximum_other_3gpp_links_per_document": max(outdegrees),
            "average_local_release18_links_per_document": round(statistics.mean(local_outdegrees), 3),
            "average_incoming_local_release18_links_per_raw_file": round(total_local_edges / total, 3),
            "average_incoming_local_release18_links_per_canonical_specification": round(
                total_local_edges / len(local_ids), 3
            ),
        },
        "outdegree_distribution_other_3gpp_documents": distribution(outdegrees),
        "outdegree_distribution_local_release18_documents": distribution(local_outdegrees),
        "top_outbound_sources": [
            {
                "source_path": row["source_path"],
                "source_specification_id": row["source_specification_id"],
                "other_3gpp_documents": len(row["other_3gpp_targets"]),
                "local_release18_documents": len(row["local_release18_targets"]),
            }
            for row in top_outbound
        ],
        "top_local_release18_targets": top_inbound,
        "documents": document_rows,
    }


def pdf_safe(value: object) -> str:
    """Escape text for a Helvetica PDF content stream (Base-14 fonts are ASCII)."""
    text = str(value).encode("ascii", "replace").decode("ascii")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf_lines(report: dict[str, object]) -> list[str]:
    """Render the key result tables into plain ASCII lines for the PDF."""
    input_data = report["input"]
    metrics = report["metrics"]
    lines = [
        "Release 18 GSMA 3GPP reference-link statistics",
        "",
        f"Input directory: {input_data['release_dir']}",
        f"Source raw.md files: {input_data['raw_markdown_documents']}",
        f"Canonical specification IDs: {input_data['unique_canonical_specification_ids']}",
        "",
        "Headline metrics (links are distinct target 3GPP IDs; self-links excluded)",
        f"Average bracketed [n] reference entries per source: {metrics['average_bracketed_reference_entries_per_source_file']}",
        f"Documents with >=1 other 3GPP link: {metrics['documents_with_any_other_3gpp_link']} ({metrics['percent_documents_with_any_other_3gpp_link']}%)",
        f"Documents with >=1 target present in this GSMA Rel-18 corpus: {metrics['documents_with_any_local_release18_link']} ({metrics['percent_documents_with_any_local_release18_link']}%)",
        f"Average other-document links per source: {metrics['average_other_3gpp_links_per_document']}",
        f"Average incoming links per raw file (local Rel-18 only): {metrics['average_incoming_local_release18_links_per_raw_file']}",
        f"Median / P90 / maximum: {metrics['median_other_3gpp_links_per_document']} / {metrics['p90_other_3gpp_links_per_document']} / {metrics['maximum_other_3gpp_links_per_document']}",
        f"Sources with >10 other-document links: {metrics['documents_linking_to_more_than_10_other_3gpp_documents']} ({metrics['percent_documents_linking_to_more_than_10_other_3gpp_documents']}%)",
        f"Sources with >10 local Rel-18 links: {metrics['documents_linking_to_more_than_10_local_release18_documents']} ({metrics['percent_documents_linking_to_more_than_10_local_release18_documents']}%)",
        f"Distinct other-document links / local links: {metrics['unique_other_3gpp_document_links']} / {metrics['unique_local_release18_document_links']}",
        f"Local Rel-18 target resolution: {metrics['percent_other_links_resolved_inside_local_release18']}%",
        f"Directed link density: {metrics['directed_link_density_percent']}%",
        "",
        "Out-degree distribution: other 3GPP documents",
        "Band      Documents    Percent",
    ]
    for row in report["outdegree_distribution_other_3gpp_documents"]:
        lines.append(f"{row['band']:<8}  {row['documents']:>9}    {row['percent_of_documents']:>7.3f}%")
    lines.extend(["", "Top source documents by outgoing other-document links", "Source ID  Other  Local  Path"])
    for row in report["top_outbound_sources"]:
        lines.append(
            f"{row['source_specification_id']:<9}  {row['other_3gpp_documents']:>5}  {row['local_release18_documents']:>5}  {row['source_path']}"
        )
    lines.extend(["", "Most referenced local Release-18 targets", "Target ID  Source documents"])
    for row in report["top_local_release18_targets"]:
        lines.append(f"{row['target_specification_id']:<9}  {row['source_documents_linking_to_target']:>16}")
    lines.extend(
        [
            "",
            "Methodology and limits",
            "Only explicit 3GPP/ETSI TS/TR identifiers in the first References section are measured.",
            "ETSI identifiers such as 123 501 are normalized to 3GPP ID 23501.",
            "URLs, RFCs, non-3GPP standards, in-text mentions, and unrecognised formats are excluded.",
            "The JSON companion contains every source path and its extracted target IDs for audit.",
        ]
    )
    return lines


def write_pdf(path: Path, lines: list[str]) -> None:
    """Write a compact, dependency-free PDF containing wrapped report text."""
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=108, break_long_words=False) or [""])
    page_size = 48
    pages = [wrapped_lines[index:index + page_size] for index in range(0, len(wrapped_lines), page_size)] or [[""]]
    page_object_start = 3
    content_object_start = page_object_start + len(pages)
    font_object = content_object_start + len(pages)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{page_object_start + index} 0 R' for index in range(len(pages)))}] /Count {len(pages)} >>".encode(),
        font_object: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for index, page_lines in enumerate(pages):
        page_object = page_object_start + index
        content_object = content_object_start + index
        content_parts = ["BT", "/F1 9 Tf", "50 760 Td", "12 TL"]
        for line in page_lines:
            content_parts.append(f"({pdf_safe(line)}) Tj")
            content_parts.append("T*")
        content_parts.append("ET")
        stream = "\n".join(content_parts).encode("ascii")
        objects[page_object] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_object} 0 R >> >> /Contents {content_object} 0 R >>".encode()
        )
        objects[content_object] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"

    path.parent.mkdir(parents=True, exist_ok=True)
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number in range(1, font_object + 1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(objects[number])
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {font_object + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {font_object + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    report = summarize(args.release_dir, args.top_n)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "release18_reference_statistics.json"
    pdf_path = args.output_dir / "release18_reference_statistics.pdf"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_pdf(pdf_path, make_pdf_lines(report))
    metrics = report["metrics"]
    print(f"Analysed {report['input']['raw_markdown_documents']} GSMA Release-18 raw Markdown files.")
    print(
        "Documents with >=1 other 3GPP link: "
        f"{metrics['documents_with_any_other_3gpp_link']} "
        f"({metrics['percent_documents_with_any_other_3gpp_link']}%)."
    )
    print(f"PDF: {pdf_path}")
    print(f"JSON audit: {json_path}")


if __name__ == "__main__":
    main()
