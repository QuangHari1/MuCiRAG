from pathlib import Path
import json
import re
import sys
import argparse

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MuCiRAG.src.settings import load_settings
from MuCiRAG.src.corpus import SourceDocument, discover_source_documents, load_selected_document_keys

SETTINGS = load_settings()
WORKSPACE_ROOT = SETTINGS.workspace_root
INPUT_ROOT = SETTINGS.release_dir
OUTPUT_ROOT = SETTINGS.metadata_dir


SECTION_NUMBER_PATTERN = r"(?:\d+[A-Za-z]?|[A-Z])(?:\.(?:\d+[A-Za-z]?|[A-Z]))*"


SECTION_PATTERN = re.compile(
    rf"""
    ^\s*
    (?P<markdown>\#{{1,6}})
    \s*
    (?:---\s*)?
    # Markdown conversion sometimes bolds the complete heading or just its
    # number. Treat the delimiters as decoration, not as part of a clause.
    (?:(?:\*{{2}}|__)\s*)?
    (?P<number>{SECTION_NUMBER_PATTERN})
    (?:(?:\*{{2}}|__)\s*)?
    \s+
    (?P<title>.+?)
    \s*$
    """,
    re.VERBOSE,
)

# Some converted PDFs prefix a real clause heading with a four-digit page/line
# marker, for example ``## 1527 1 Scope``.  Citations must use ``1`` rather
# than the converter marker, while the original Markdown line remains the
# boundary used later by ``chunking.py``.
EMBEDDED_SECTION_PATTERN = re.compile(r"^(?P<number>" + SECTION_NUMBER_PATTERN + r")\s+(?P<title>.+)$")


def clean_title(title: str) -> str:
    """
    Clean markdown decoration from heading title.
    """

    title = title.strip()

    # Remove trailing ---
    title = re.sub(r"\s*---\s*$", "", title)

    # Remove markdown emphasis
    title = title.replace("**", "")
    title = title.replace("__", "")

    # Collapse whitespace
    title = re.sub(r"\s+", " ", title)

    return title.strip()


def get_heading_level(section_number: str) -> int:
    """
    Determine hierarchy using section number rather than markdown '#'.

    Examples:
        1       -> 1
        3.1     -> 2
        5.2.1   -> 3

        A       -> 1
        A.1     -> 2
        A.1.2   -> 3
    """

    return section_number.count(".") + 1


def normalize_section_heading(section_number: str, title: str) -> tuple[str, str]:
    """Discard a converter marker only when a full clause number follows it."""
    embedded = EMBEDDED_SECTION_PATTERN.match(title)
    if section_number.isdigit() and len(section_number) >= 4 and embedded:
        return embedded.group("number"), clean_title(embedded.group("title"))
    return section_number, title


def extract_headings(raw_md_path: Path) -> list[dict]:
    headings = []

    with raw_md_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            match = SECTION_PATTERN.match(line)

            if not match:
                continue

            section_number = match.group("number")
            title = clean_title(match.group("title"))
            section_number, title = normalize_section_heading(section_number, title)

            headings.append(
                {
                    "heading": section_number,
                    "title": title,
                    "level": get_heading_level(section_number),
                    "line": line_number,
                }
            )

    return headings


def extract_document_info(raw_md_path: Path) -> dict:
    """
    Extract basic document information from the first part
    of raw.md.

    Example:
        # 3GPP TS 21.201 V18.1.0 (2025-03) ---
    """

    pattern = re.compile(
        r"""
        3GPP\s+
        (?P<type>TS|TR)\s+
        (?P<number>\d+\.\d+)
        \s+
        V(?P<version>\d+\.\d+\.\d+)
        (?:\s+\((?P<date>[^)]+)\))?
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    with raw_md_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        # Document header should be near beginning
        for _ in range(50):

            line = f.readline()

            if not line:
                break

            match = pattern.search(line)

            if match:

                number = match.group("number")

                # 21.201 -> 21201
                document_id = number.replace(".", "")

                return {
                    "document_id": document_id,
                    "document_number": number,
                    "document_type": match.group("type").upper(),
                    "version": match.group("version"),
                    "date": match.group("date"),
                }

    return {}


def process_document(
    source_document: SourceDocument,
    input_root: Path,
    output_root: Path,
    release: str,
) -> None:
    """
    Process one raw.md.
    """

    raw_md_path = source_document.path
    relative = raw_md_path.relative_to(input_root)

    # Example:
    #
    # 21_series/21201/raw.md
    #
    document_dir = relative.parent

    output_dir = output_root / document_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_info = extract_document_info(raw_md_path)
    headings = extract_headings(raw_md_path)

    parts = document_dir.parts

    series = None

    if parts:
        match = re.match(
            r"(\d+)_series",
            parts[0],
        )

        if match:
            series = match.group(1)

    if not document_info.get("document_id"):
        document_info["document_id"] = document_dir.name

    result = {
        **document_info,
        "release": release,
        "series": series,
        "document_key": document_dir.as_posix(),
        "source": str(relative),
        "source_file_name": source_document.source_name,
        "heading_count": len(headings),
        "headings": headings,
    }

    output_path = output_dir / "headings.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[OK] {document_info['document_id']}: "
        f"{len(headings)} headings"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--selection", type=Path, help="Only process document keys in this selection JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_keys = load_selected_document_keys(args.selection)
    sources = discover_source_documents(
        args.release_dir,
        (SETTINGS.get("chunking", "preferred_source_name"), SETTINGS.get("chunking", "fallback_source_name")),
        selected_keys,
    )
    print(f"Found {len(sources)} selected source documents")
    failures: list[str] = []
    for source_document in sources:
        try:
            process_document(source_document, args.release_dir, args.output_dir, SETTINGS.release)
        except Exception as error:
            failures.append(str(source_document.path))
            print(f"[ERROR] {source_document.path}: {error}")
    if failures:
        # Do not let a pipeline continue from a silently incomplete metadata set.
        raise RuntimeError(f"Heading extraction failed for {len(failures)} document(s)")


if __name__ == "__main__":
    main()
