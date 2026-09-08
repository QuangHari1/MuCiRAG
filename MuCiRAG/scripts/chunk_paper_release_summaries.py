"""Chunk the four non-series Netop release-summary DOCX files.

These Rel-14 through Rel-17 documents are part of the paper repository but
have no GSMA Release-18 counterpart or numeric 3GPP series.  They are emitted
as one separately labelled chunk source so selection and embedding remain
traceable without pretending that they belong to a numbered series.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from xml.etree import ElementTree

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MuCiRAG.src.settings import load_settings


SETTINGS = load_settings()
WORKSPACE_ROOT = SETTINGS.workspace_root
SUMMARY_DIR = WORKSPACE_ROOT / SETTINGS.get("paper_release_summaries", "source_dir")
CHUNK_PATH = (
    SETTINGS.dataset_dir
    / "3gpp"
    / "Chunk"
    / f"Rel-{SETTINGS.release}"
    / SETTINGS.get("paper_release_summaries", "chunk_file")
)
TARGET_CHARS = SETTINGS.get("chunking", "target_tokens") * SETTINGS.get(
    "chunking", "characters_per_token"
)
MAX_CHARS = SETTINGS.get("chunking", "max_tokens") * SETTINGS.get(
    "chunking", "characters_per_token"
)
WORD_TEXT_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
PARAGRAPH_TAG = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"


def read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = [
        "".join(node.text or "" for node in paragraph.iter(WORD_TEXT_TAG)).strip()
        for paragraph in root.iter(PARAGRAPH_TAG)
    ]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(line.strip() for line in text.splitlines())).strip()


def split_text(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", normalize_text(text))
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > TARGET_CHARS:
            chunks.append(current)
            current = ""
        while len(sentence) > MAX_CHARS:
            chunks.append(sentence[:MAX_CHARS].strip())
            sentence = sentence[MAX_CHARS:].strip()
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def workspace_relative(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def build_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for release in SETTINGS.get("paper_release_summaries", "source_releases"):
        path = SUMMARY_DIR / f"rel_{release}.docx"
        if not path.exists():
            raise FileNotFoundError(f"Missing paper release summary: {path}")
        document_key = f"paper_release_summaries/rel_{release}"
        for index, text in enumerate(split_text(read_docx_text(path))):
            chunks.append(
                {
                    "chunk_id": f"{document_key.replace('/', '__')}_{index:06d}",
                    "document_id": f"rel_{release}",
                    "document_key": document_key,
                    "document_number": None,
                    "source_path": workspace_relative(path),
                    "source_release": release,
                    "heading": f"3GPP Release {release} summary",
                    "chunk_index_in_heading": index,
                    "token_count_estimate": max(
                        1, round(len(text) / SETTINGS.get("chunking", "characters_per_token"))
                    ),
                    "is_reference_section": False,
                    "references": [],
                    "text": text,
                }
            )
    return chunks


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


if __name__ == "__main__":
    chunks = build_chunks()
    write_json(
        CHUNK_PATH,
        {
            "release": SETTINGS.release,
            "series": "release-summaries",
            "chunk_count": len(chunks),
            "chunks": chunks,
        },
    )
    print(f"Wrote {len(chunks)} chunks from 4 paper release summaries to {CHUNK_PATH}")
