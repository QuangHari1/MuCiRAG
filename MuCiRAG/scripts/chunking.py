from pathlib import Path
import json
import re
import sys
import argparse
import hashlib

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MuCiRAG.src.settings import load_settings
from MuCiRAG.src.corpus import SourceDocument, discover_source_documents, load_selected_document_keys

# ============================================================
# CONFIG
# ============================================================

SETTINGS = load_settings()
WORKSPACE_ROOT = SETTINGS.workspace_root
RAW_ROOT = SETTINGS.release_dir
METADATA_ROOT = SETTINGS.metadata_dir

CHUNK_OUTPUT_ROOT = SETTINGS.chunk_dir
REFERENCE_OUTPUT_ROOT = SETTINGS.reference_dir

PREFERRED_SOURCE_NAME = SETTINGS.get("chunking", "preferred_source_name")
FALLBACK_SOURCE_NAME = SETTINGS.get("chunking", "fallback_source_name")

TARGET_TOKENS = SETTINGS.get("chunking", "target_tokens")
MAX_TOKENS = SETTINGS.get("chunking", "max_tokens")

CHARS_PER_TOKEN = SETTINGS.get("chunking", "characters_per_token")

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN


# ============================================================
# BASIC HELPERS
# ============================================================
def find_source_files(root: Path, selected_document_keys: set[str] | None = None) -> list[SourceDocument]:
    """
    Find exactly one source markdown file per document folder.

    The source filename is selected by ``[chunking]`` in ``config.toml``.
    """

    return discover_source_documents(
        root,
        (PREFERRED_SOURCE_NAME, FALLBACK_SOURCE_NAME),
        selected_document_keys,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

    
def estimate_tokens(text: str) -> int:

    text = text.strip()

    if not text:
        return 0

    return max(
        1,
        round(len(text) / CHARS_PER_TOKEN),
    )


def normalize_text(text: str) -> str:

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        lines.append(line)

    text = "\n".join(lines)

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def document_id_to_series(
    document_id: str,
) -> str | None:
    """
    21103 -> 21
    23501 -> 23
    33501 -> 33
    """

    if not document_id:
        return None

    digits = re.sub(
        r"\D",
        "",
        str(document_id),
    )

    if len(digits) < 2:
        return None

    return digits[:2]


def build_target_info(
    document_id: str,
) -> dict:
    """
    Metadata telling us where target chunks live.
    """

    series = document_id_to_series(
        document_id
    )

    result = {
        "document_id": document_id,
    }

    if series:

        result["target_series"] = series

        result["target_chunk_file"] = (
            f"ChunkSeries{series}.json"
        )

    return result


# ============================================================
# IMAGE MARKDOWN
# ============================================================

IMAGE_PATTERN = re.compile(
    r"!\[(.*?)\]\((.*?)\)",
    re.DOTALL,
)


def clean_markdown_images(
    text: str,
) -> str:

    return IMAGE_PATTERN.sub(
        "",
        text,
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(
    text: str,
) -> list[str]:

    text = normalize_text(
        text
    )

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    result = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9\[(])",
            paragraph,
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:
                result.append(sentence)

        result.append(
            "__PARAGRAPH_BREAK__"
        )

    if (
        result
        and result[-1]
        == "__PARAGRAPH_BREAK__"
    ):
        result.pop()

    return result


# ============================================================
# HARD SPLIT
# ============================================================

def hard_split_text(
    text: str,
    max_chars: int = MAX_CHARS,
) -> list[str]:

    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    pieces = []

    remaining = text

    while len(remaining) > max_chars:

        candidate = remaining[:max_chars]

        split_pos = -1

        for separator in [
            "; ",
            ", ",
            " ",
        ]:

            pos = candidate.rfind(
                separator
            )

            if pos > int(
                max_chars * 0.55
            ):

                split_pos = (
                    pos
                    + len(separator)
                )

                break

        if split_pos == -1:
            split_pos = max_chars

        piece = (
            remaining[:split_pos]
            .strip()
        )

        if piece:
            pieces.append(piece)

        remaining = (
            remaining[split_pos:]
            .strip()
        )

    if remaining:
        pieces.append(remaining)

    return pieces


# ============================================================
# CHUNK ONE HEADING
# ============================================================

def chunk_section(
    text: str,
) -> list[str]:

    text = clean_markdown_images(
        text
    )

    text = normalize_text(
        text
    )

    if not text:
        return []

    if len(text) <= TARGET_CHARS:
        return [text]

    units = split_sentences(
        text
    )

    chunks = []

    current_parts = []
    current_length = 0

    for unit in units:

        if (
            unit
            == "__PARAGRAPH_BREAK__"
        ):

            if current_parts:

                if (
                    current_parts[-1]
                    != "\n\n"
                ):
                    current_parts.append(
                        "\n\n"
                    )

            continue

        unit = unit.strip()

        if not unit:
            continue

        # Very long single sentence/table/etc.
        if len(unit) > MAX_CHARS:

            if current_parts:

                chunk = normalize_text(
                    "".join(
                        current_parts
                    )
                )

                if chunk:
                    chunks.append(chunk)

                current_parts = []
                current_length = 0

            chunks.extend(
                hard_split_text(unit)
            )

            continue

        separator = ""

        if current_parts:

            if (
                current_parts[-1]
                != "\n\n"
            ):
                separator = " "

        projected_length = (
            current_length
            + len(separator)
            + len(unit)
        )

        # Below target
        if projected_length <= TARGET_CHARS:

            if separator:
                current_parts.append(
                    separator
                )

            current_parts.append(
                unit
            )

            current_length = (
                projected_length
            )

            continue

        # Slightly over target but below max.
        # Finish sentence.
        if projected_length <= MAX_CHARS:

            if separator:
                current_parts.append(
                    separator
                )

            current_parts.append(
                unit
            )

            chunk = normalize_text(
                "".join(
                    current_parts
                )
            )

            if chunk:
                chunks.append(chunk)

            current_parts = []
            current_length = 0

            continue

        # Would exceed max.
        if current_parts:

            chunk = normalize_text(
                "".join(
                    current_parts
                )
            )

            if chunk:
                chunks.append(chunk)

        current_parts = [
            unit
        ]

        current_length = len(
            unit
        )

    if current_parts:

        chunk = normalize_text(
            "".join(
                current_parts
            )
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# ============================================================
# BIBLIOGRAPHY / REFERENCES SECTION
# ============================================================

REFERENCE_ENTRY_PATTERN = re.compile(
    r"""
    ^\s*
    [-*]?
    \s*
    \[
        (?P<ref_id>\d+)
    \]
    \s*
    (?:
        3GPP\s+
    )?
    (?P<doc_type>TS|TR)
    \s+
    (?P<doc_number>\d{2}\.\d{3})
    \b
    """,
    re.VERBOSE
    | re.IGNORECASE,
)


def parse_reference_dictionary(
    section_text: str,
) -> dict[str, dict]:

    references = {}

    for line in section_text.splitlines():

        match = (
            REFERENCE_ENTRY_PATTERN
            .match(line)
        )

        if not match:
            continue

        ref_id = (
            match.group("ref_id")
        )

        doc_type = (
            match.group("doc_type")
            .upper()
        )

        doc_number = (
            match.group(
                "doc_number"
            )
        )

        document_id = (
            doc_number.replace(
                ".",
                "",
            )
        )

        target_info = (
            build_target_info(
                document_id
            )
        )

        references[ref_id] = {
            "reference_id": ref_id,
            "document_type": doc_type,
            "document_number": doc_number,
            **target_info,
        }

    return references


def build_reverse_reference_dictionary(
    reference_dictionary: dict,
) -> dict[str, str]:
    """
    Bibliography:
        [3] TS 21.103

    becomes:
        21103 -> "3"

    Allows:
        TS 21.103

    in normal text to recover reference_id=3
    even if [3] isn't written beside it.
    """

    reverse = {}

    for (
        ref_id,
        reference,
    ) in reference_dictionary.items():

        document_id = (
            reference.get(
                "document_id"
            )
        )

        if document_id:

            # First matching bibliography entry wins.
            if document_id not in reverse:
                reverse[
                    document_id
                ] = ref_id

    return reverse


# ============================================================
# CITATION REGEX
# ============================================================

EXPLICIT_REFERENCE_PATTERN = re.compile(
    r"""
    (?:
        3GPP\s+
    )?
    (?P<doc_type>TS|TR)
    \s+
    (?P<doc_number>\d{2}\.\d{3})
    \s*
    (?:
        \[
            (?P<ref_id>\d+)
        \]
    )?
    """,
    re.VERBOSE
    | re.IGNORECASE,
)


BRACKET_REFERENCE_PATTERN = re.compile(
    r"\[(\d+)\]"
)


CLAUSE_DOC_PATTERN = re.compile(
    r"""
    clause\s+
    (?P<clause>
        \d+(?:\.\d+)+
    )
    \s+
    (?:
        of|
        in
    )
    \s+
    (?:
        3GPP\s+
    )?
    (?P<doc_type>TS|TR)
    \s+
    (?P<doc_number>
        \d{2}\.\d{3}
    )
    """,
    re.VERBOSE
    | re.IGNORECASE,
)


DOC_CLAUSE_PATTERN = re.compile(
    r"""
    (?:
        3GPP\s+
    )?
    (?P<doc_type>TS|TR)
    \s+
    (?P<doc_number>
        \d{2}\.\d{3}
    )
    [,\s]+
    clause\s+
    (?P<clause>
        \d+(?:\.\d+)+
    )
    """,
    re.VERBOSE
    | re.IGNORECASE,
)


INTERNAL_CLAUSE_PATTERN = re.compile(
    r"""
    \bclause\s+
    (?P<clause>
        \d+(?:\.\d+)+
    )
    \b
    """,
    re.VERBOSE
    | re.IGNORECASE,
)


# ============================================================
# BUILD REFERENCE OBJECT
# ============================================================

def create_reference_object(
    *,
    ref_type: str,
    document_id: str,
    reference_id: str | None = None,
    target_heading: str | None = None,
) -> dict:
    """
    IMPORTANT:
    Never writes null values.

    Instead of:

        "reference_id": null
        "target_heading": null

    the fields simply don't exist.
    """

    result = {
        "type": ref_type,
        **build_target_info(
            document_id
        ),
    }

    if reference_id is not None:

        result[
            "reference_id"
        ] = str(reference_id)

    if target_heading is not None:

        result[
            "target_heading"
        ] = target_heading

    return result


# ============================================================
# EXTRACT REFERENCES FROM NORMAL CHUNK
# ============================================================

def extract_chunk_references(
    text: str,
    reference_dictionary: dict,
    current_document_id: str,
) -> list[dict]:

    references = []

    seen = set()

    external_clause_spans = []

    reverse_reference_dictionary = (
        build_reverse_reference_dictionary(
            reference_dictionary
        )
    )

    # --------------------------------------------------------
    # 1. External clause references
    # --------------------------------------------------------

    for pattern in [
        CLAUSE_DOC_PATTERN,
        DOC_CLAUSE_PATTERN,
    ]:

        for match in pattern.finditer(
            text
        ):

            document_number = (
                match.group(
                    "doc_number"
                )
            )

            document_id = (
                document_number
                .replace(".", "")
            )

            clause = (
                match.group(
                    "clause"
                )
            )

            # Try to recover bibliography number.
            ref_id = (
                reverse_reference_dictionary
                .get(document_id)
            )

            key = (
                "external",
                document_id,
                clause,
            )

            if key not in seen:

                references.append(
                    create_reference_object(
                        ref_type="external",
                        document_id=(
                            document_id
                        ),
                        reference_id=(
                            ref_id
                        ),
                        target_heading=(
                            clause
                        ),
                    )
                )

                seen.add(key)

            external_clause_spans.append(
                match.span()
            )

    # --------------------------------------------------------
    # 2. Explicit TS/TR references
    # --------------------------------------------------------

    for match in (
        EXPLICIT_REFERENCE_PATTERN
        .finditer(text)
    ):

        document_number = (
            match.group(
                "doc_number"
            )
        )

        document_id = (
            document_number
            .replace(".", "")
        )

        ref_id = (
            match.group(
                "ref_id"
            )
        )

        # Important:
        #
        # TS 21.103
        #
        # may not have [x] right after it.
        #
        # Recover x from the document bibliography.
        if ref_id is None:

            ref_id = (
                reverse_reference_dictionary
                .get(document_id)
            )

        key = (
            "external",
            document_id,
            None,
        )

        if key in seen:
            continue

        references.append(
            create_reference_object(
                ref_type="external",
                document_id=(
                    document_id
                ),
                reference_id=(
                    ref_id
                ),
            )
        )

        seen.add(key)

    # --------------------------------------------------------
    # 3. Bare [x]
    # --------------------------------------------------------

    for ref_id in (
        BRACKET_REFERENCE_PATTERN
        .findall(text)
    ):

        reference = (
            reference_dictionary
            .get(ref_id)
        )

        if not reference:
            continue

        document_id = (
            reference[
                "document_id"
            ]
        )

        key = (
            "external",
            document_id,
            None,
        )

        if key in seen:
            continue

        references.append(
            create_reference_object(
                ref_type="external",
                document_id=(
                    document_id
                ),
                reference_id=(
                    ref_id
                ),
            )
        )

        seen.add(key)

    # --------------------------------------------------------
    # 4. Internal clause references
    # --------------------------------------------------------

    for match in (
        INTERNAL_CLAUSE_PATTERN
        .finditer(text)
    ):

        span = match.span()

        inside_external = False

        for external_span in (
            external_clause_spans
        ):

            if (
                span[0]
                >= external_span[0]
                and
                span[1]
                <= external_span[1]
            ):

                inside_external = True
                break

        if inside_external:
            continue

        clause = (
            match.group(
                "clause"
            )
        )

        key = (
            "internal",
            current_document_id,
            clause,
        )

        if key in seen:
            continue

        references.append(
            create_reference_object(
                ref_type="internal",
                document_id=(
                    current_document_id
                ),
                target_heading=(
                    clause
                ),
            )
        )

        seen.add(key)

    return references


# ============================================================
# BIBLIOGRAPHY CHUNK REFERENCES
# ============================================================

def extract_bibliography_references(
    text: str,
    reference_dictionary: dict,
) -> list[dict]:

    references = []

    seen = set()

    for (
        ref_id,
        ref_info,
    ) in reference_dictionary.items():

        pattern = re.compile(
            rf"\[\s*"
            rf"{re.escape(ref_id)}"
            rf"\s*\]"
        )

        if not pattern.search(text):
            continue

        document_id = (
            ref_info[
                "document_id"
            ]
        )

        key = (
            ref_id,
            document_id,
        )

        if key in seen:
            continue

        references.append(
            create_reference_object(
                ref_type="bibliography",
                document_id=(
                    document_id
                ),
                reference_id=(
                    ref_id
                ),
            )
        )

        seen.add(key)

    return references


# ============================================================
# LOAD HEADING METADATA
# ============================================================

def load_heading_metadata(
    metadata_path: Path,
) -> dict:

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# EXTRACT SECTIONS
# ============================================================

def extract_sections(
    raw_text: str,
    headings: list[dict],
) -> list[dict]:

    lines = (
        raw_text.splitlines()
    )

    sections = []

    for index, heading in enumerate(
        headings
    ):

        heading_line = (
            heading["line"]
        )

        content_start = (
            heading_line
        )

        if (
            index + 1
            < len(headings)
        ):

            next_heading_line = (
                headings[
                    index + 1
                ]["line"]
            )

            content_end = (
                next_heading_line
                - 1
            )

        else:

            content_end = (
                len(lines)
            )

        section_lines = lines[
            content_start:
            content_end
        ]

        section_text = normalize_text(
            "\n".join(
                section_lines
            )
        )

        sections.append(
            {
                **heading,
                "text": section_text,
            }
        )

    return sections


# ============================================================
# REFERENCE SECTION DETECTION
# ============================================================

def is_reference_section(
    section: dict,
) -> bool:

    title = (
        section
        .get("title", "")
        .strip()
        .lower()
    )

    return title in {
        "reference",
        "references",
        "normative reference",
        "normative references",
    }


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(source_document: SourceDocument):

    raw_md_path = source_document.path

    relative = (
        raw_md_path
        .relative_to(
            RAW_ROOT
        )
    )

    document_dir = (
        relative.parent
    )

    document_key = (
        document_dir.as_posix()
    )

    metadata_path = (
        METADATA_ROOT
        / document_dir
        / "headings.json"
    )

    if not metadata_path.exists():

        print(
            f"[SKIP] Missing metadata: "
            f"{metadata_path}"
        )

        return None

    metadata = (
        load_heading_metadata(
            metadata_path
        )
    )

    document_id = (
        metadata.get(
            "document_id",
            document_dir.name,
        )
    )

    document_number = (
        metadata.get(
            "document_number"
        )
    )

    series = (
        metadata.get(
            "series"
        )
    )

    if not series:

        series = (
            document_id_to_series(
                document_id
            )
        )

    headings = (
        metadata.get(
            "headings",
            [],
        )
    )

    if not headings:

        print(
            f"[SKIP] "
            f"{document_id}: "
            f"no headings"
        )

        return None

    raw_text = (
        raw_md_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )
    source_sha256 = sha256_file(raw_md_path)

    sections = extract_sections(
        raw_text,
        headings,
    )

    # ========================================================
    # REFERENCES DICTIONARY FOR DOCUMENT
    # ========================================================

    reference_dictionary = {}

    for section in sections:

        if not is_reference_section(
            section
        ):
            continue

        parsed = (
            parse_reference_dictionary(
                section["text"]
            )
        )

        reference_dictionary.update(
            parsed
        )

    # ========================================================
    # CREATE CHUNKS
    # ========================================================

    chunks = []

    document_links = []

    document_link_seen = set()

    global_chunk_index = 0

    for section in sections:

        heading = (
            section["heading"]
        )

        section_text = (
            section["text"]
        )

        ref_section = (
            is_reference_section(
                section
            )
        )

        section_chunks = (
            chunk_section(
                section_text
            )
        )

        for (
            heading_chunk_index,
            chunk_text,
        ) in enumerate(
            section_chunks
        ):

            chunk_id = (
                f"{document_key.replace('/', '__')}_"
                f"{global_chunk_index:06d}"
            )

            if ref_section:

                chunk_references = (
                    extract_bibliography_references(
                        text=chunk_text,
                        reference_dictionary=(
                            reference_dictionary
                        ),
                    )
                )

            else:

                chunk_references = (
                    extract_chunk_references(
                        text=chunk_text,
                        reference_dictionary=(
                            reference_dictionary
                        ),
                        current_document_id=(
                            document_id
                        ),
                    )
                )

            # --------------------------------------------
            # Add information to ReferenceSeriesXX
            # --------------------------------------------

            for reference in (
                chunk_references
            ):

                link_key = (
                    chunk_id,
                    reference.get("type"),
                    reference.get(
                        "document_id"
                    ),
                    reference.get(
                        "target_heading"
                    ),
                )

                if (
                    link_key
                    in document_link_seen
                ):
                    continue

                link = {
                    "source_chunk_id": (
                        chunk_id
                    ),
                    "source_heading": (
                        heading
                    ),
                    **reference,
                }

                document_links.append(
                    link
                )

                document_link_seen.add(
                    link_key
                )

            chunk = {
                "chunk_id": (
                    chunk_id
                ),
                "document_id": (
                    document_id
                ),
                "document_key": (
                    document_key
                ),
                "source_path": (
                    relative.as_posix()
                ),
                "source_file_name": source_document.source_name,
                "source_sha256": source_sha256,
                "document_number": (
                    document_number
                ),
                "release": SETTINGS.release,
                "series": (
                    series
                ),
                "heading": (
                    heading
                ),
                "chunk_index_in_heading": (
                    heading_chunk_index
                ),
                "token_count_estimate": (
                    estimate_tokens(
                        chunk_text
                    )
                ),
                "is_reference_section": (
                    ref_section
                ),
                "references": (
                    chunk_references
                ),
                "text": (
                    chunk_text
                ),
            }

            chunks.append(
                chunk
            )

            global_chunk_index += 1

    print(
        f"[OK] "
        f"{document_id}: "
        f"{len(chunks)} chunks | "
        f"{len(reference_dictionary)} bibliography refs | "
        f"{len(document_links)} links"
    )

    return {
        "series": series,
        "document_id": (
            document_id
        ),
        "document_number": (
            document_number
        ),
        "chunks": chunks,
        "bibliography": (
            reference_dictionary
        ),
        "links": (
            document_links
        ),
    }


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk selected 3GPP Markdown with heading and citation provenance.")
    parser.add_argument("--selection", type=Path, help="Only chunk document keys listed in this selection JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_files = find_source_files(RAW_ROOT, load_selected_document_keys(args.selection))

    print(
        f"Found "
        f"{len(raw_files)} "
        f"source documents"
    )

    series_chunks = {}

    series_references = {}

    success = 0
    failed = 0
    skipped = 0

    for source_document in raw_files:

        try:

            result = (
                process_document(source_document)
            )

            if result is None:

                skipped += 1
                continue

            series = (
                result["series"]
            )

            if not series:

                skipped += 1
                continue

            # ================================================
            # CHUNKS
            # ================================================

            if (
                series
                not in series_chunks
            ):

                series_chunks[
                    series
                ] = []

            series_chunks[
                series
            ].extend(
                result["chunks"]
            )

            # ================================================
            # REFERENCES
            # ================================================

            if (
                series
                not in series_references
            ):

                series_references[
                    series
                ] = []

            series_references[
                series
            ].append(
                {
                    "document_id": (
                        result[
                            "document_id"
                        ]
                    ),
                    "document_number": (
                        result[
                            "document_number"
                        ]
                    ),
                    "bibliography": list(
                        result[
                            "bibliography"
                        ].values()
                    ),
                    "links": (
                        result[
                            "links"
                        ]
                    ),
                }
            )

            success += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] "
                f"{source_document.path}: "
                f"{e}"
            )

    # A partial Chunk/Reference pair looks valid to later embedding and BFS
    # stages, but silently omits documents.  Refuse publication until every
    # selected document supplied usable heading metadata and chunked cleanly.
    if failed or skipped:
        raise RuntimeError(
            f"Refusing to publish partial corpus: success={success}, "
            f"failed={failed}, skipped={skipped}"
        )

    # ========================================================
    # CREATE OUTPUT FOLDERS
    # ========================================================

    CHUNK_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REFERENCE_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAVE CHUNK SERIES
    # ========================================================

    for series in sorted(
        series_chunks.keys(),
        key=lambda x: int(x),
    ):

        chunks = (
            series_chunks[
                series
            ]
        )

        chunks.sort(
            key=lambda x: (
                x["document_id"],
                x["chunk_id"],
            )
        )

        output_path = (
            CHUNK_OUTPUT_ROOT
            / f"ChunkSeries{series}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "release": SETTINGS.release,
                    "series": series,
                    "chunk_count": (
                        len(chunks)
                    ),
                    "chunks": chunks,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[SAVE CHUNK] "
            f"Series {series}: "
            f"{len(chunks)} chunks"
        )

    # ========================================================
    # SAVE REFERENCE SERIES
    # ========================================================

    for series in sorted(
        series_references.keys(),
        key=lambda x: int(x),
    ):

        documents = (
            series_references[
                series
            ]
        )

        documents.sort(
            key=lambda x: (
                x["document_id"]
            )
        )

        total_links = sum(
            len(
                document["links"]
            )
            for document
            in documents
        )

        output_path = (
            REFERENCE_OUTPUT_ROOT
            / f"ReferenceSeries{series}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "release": SETTINGS.release,
                    "series": series,
                    "document_count": (
                        len(documents)
                    ),
                    "link_count": (
                        total_links
                    ),
                    "documents": (
                        documents
                    ),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[SAVE REFERENCE] "
            f"Series {series}: "
            f"{total_links} links"
        )

    print()
    print(
        "================================="
    )
    print(
        f"Documents success : {success}"
    )
    print(
        f"Documents skipped : {skipped}"
    )
    print(
        f"Documents failed  : {failed}"
    )
    print(
        f"Chunk series      : "
        f"{len(series_chunks)}"
    )
    print(
        f"Reference series  : "
        f"{len(series_references)}"
    )
    print(
        "================================="
    )


if __name__ == "__main__":
    main()
