"""Offline anchor-hierarchy artifacts and their runtime validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from docx import Document


ARTIFACT_VERSION = 1
SUMMARY_SERIES = "release-summaries"
SUMMARY_DESCRIPTION = "3GPP Release 14 through Release 17 historical release summaries."
MAX_SCOPE_CHARS = 4_000
SCOPE_TITLE_PATTERN = re.compile(r"^(?:1\s+)?scope$", flags=re.IGNORECASE)
NEXT_SECTION_PATTERN = re.compile(r"^2\s+references$", flags=re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Hierarchy embedding must not be zero.")
    return vector / norm


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"[`#*_>|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_scope_title(text: str) -> bool:
    return bool(SCOPE_TITLE_PATTERN.fullmatch(_clean_markdown(text)))


def _scope_from_heading_metadata(headings: Any, lines: list[str]) -> str:
    if not isinstance(headings, list):
        return ""
    scope_index = next(
        (
            index
            for index, heading in enumerate(headings)
            if isinstance(heading, dict) and _is_scope_title(str(heading.get("title", "")))
        ),
        None,
    )
    if scope_index is None:
        return ""
    start = headings[scope_index].get("line")
    end = headings[scope_index + 1].get("line") if scope_index + 1 < len(headings) else len(lines) + 1
    if not isinstance(start, int) or not isinstance(end, int) or not 0 < start <= end:
        return ""
    return _clean_markdown("\n".join(lines[start : end - 1]))[:MAX_SCOPE_CHARS]


def _scope_from_markdown_headings(lines: list[str]) -> str:
    """Fallback for malformed extracted heading metadata such as ``1527 1 Scope``."""
    start: int | None = None
    for index, line in enumerate(lines):
        match = re.match(r"^\s*#{1,6}\s+(?:---\s*)?(.*)$", line)
        if not match:
            continue
        heading = _clean_markdown(match.group(1))
        if start is None and re.fullmatch(r"(?:\d+\s+)*1\s+scope", heading, flags=re.IGNORECASE):
            start = index
        elif start is not None and NEXT_SECTION_PATTERN.fullmatch(heading):
            return _clean_markdown("\n".join(lines[start:index]))[:MAX_SCOPE_CHARS]
    if start is None:
        return ""
    return _clean_markdown("\n".join(lines[start:]))[:MAX_SCOPE_CHARS]


def _document_description_from_docx(raw_path: Path) -> tuple[str, str]:
    paragraphs = [_clean_markdown(paragraph.text) for paragraph in Document(raw_path).paragraphs]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    title = next(
        (
            paragraph
            for paragraph in paragraphs[:20]
            if re.search(r"release\s+\d+", paragraph, flags=re.IGNORECASE)
            and re.search(r"(?:description|summary)", paragraph, flags=re.IGNORECASE)
        ),
        paragraphs[0] if paragraphs else "",
    )
    scope_start = next((index for index, paragraph in enumerate(paragraphs) if _is_scope_title(paragraph)), None)
    if scope_start is None:
        return title, ""
    scope_end = next(
        (
            index
            for index in range(scope_start + 1, len(paragraphs))
            if NEXT_SECTION_PATTERN.fullmatch(paragraphs[index])
        ),
        len(paragraphs),
    )
    return title, " ".join(paragraphs[scope_start + 1 : scope_end])[:MAX_SCOPE_CHARS]


def _document_fallback(metadata: dict[str, Any]) -> str:
    document_type = str(metadata.get("document_type") or "3GPP document")
    document_number = str(metadata.get("document_number") or metadata.get("document_id") or "unknown")
    series = str(metadata.get("series") or "unknown")
    return f"3GPP {document_type} {document_number}, series {series}."


def build_document_description(metadata: dict[str, Any], raw_path: Path | None) -> str:
    """Build a deterministic title-plus-Scope description for one document."""
    fallback = _document_fallback(metadata)
    if raw_path is None or not raw_path.is_file():
        return fallback
    if raw_path.suffix.casefold() == ".docx":
        title, scope = _document_description_from_docx(raw_path)
        parts = [fallback]
        if title:
            parts.append(f"Title: {title}.")
        if scope:
            parts.append(f"Scope: {scope}")
        return " ".join(parts)
    lines = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    title = ""
    ignored_titles = {"3gpp", "keywords", "postal address", "internet", "copyright notification"}
    for line in lines[:100]:
        if not re.match(r"^\s*#{1,6}\s+", line):
            continue
        candidate = _clean_markdown(re.sub(r"^\s*#{1,6}\s+(?:---\s*)?", "", line))
        if not candidate or candidate.casefold() in ignored_titles:
            continue
        if re.match(r"^3gpp\s+(?:ts|tr)\s+\d+\.\d+", candidate, flags=re.IGNORECASE):
            continue
        title = candidate
        break

    scope = _scope_from_heading_metadata(metadata.get("headings"), lines)
    if not scope:
        scope = _scope_from_markdown_headings(lines)

    parts = [fallback]
    if title:
        parts.append(f"Title: {title}.")
    if scope:
        parts.append(f"Scope: {scope}")
    return " ".join(parts)


def load_series_descriptions(path: Path, available_series: list[str]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    descriptions = {
        str(label).split()[0]: str(value["description"])
        for label, value in payload.items()
        if isinstance(value, dict) and isinstance(value.get("description"), str)
    }
    result: dict[str, str] = {}
    for series in available_series:
        if series == SUMMARY_SERIES:
            result[series] = SUMMARY_DESCRIPTION
        elif series in descriptions:
            result[series] = descriptions[series]
        else:
            raise ValueError(f"No series description for {series!r} in {path}")
    return result


def build_descriptions(
    corpus_manifest: dict[str, Any],
    embedding_root: Path,
    workspace_root: Path,
    metadata_root: Path,
    release_dir: Path,
    series_descriptions_path: Path,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Return ordered hierarchy rows and source hashes for one embedding selection."""
    series_records = corpus_manifest.get("series")
    if not isinstance(series_records, dict) or not series_records:
        raise ValueError("Embedding manifest has no series records.")
    series_names = list(series_records)
    rows = [
        {"kind": "series", "key": series, "text": text}
        for series, text in load_series_descriptions(series_descriptions_path, series_names).items()
    ]
    document_rows: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for series, record in series_records.items():
        if not isinstance(record, dict) or not isinstance(record.get("metadata_file"), str):
            raise ValueError(f"Embedding manifest has no metadata file for {series!r}")
        metadata_path = embedding_root / record["metadata_file"]
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            document_key = item.get("document_key")
            if not isinstance(document_key, str) or document_key in document_rows:
                continue
            document_rows[document_key] = {"series": series, **item}

    for document_key in sorted(document_rows):
        item = document_rows[document_key]
        headings_path = metadata_root / document_key / "headings.json"
        document_metadata: dict[str, Any] = dict(item)
        raw_path: Path | None = None
        if headings_path.is_file():
            heading_payload = json.loads(headings_path.read_text(encoding="utf-8"))
            if isinstance(heading_payload, dict):
                document_metadata = {**heading_payload, **item}
                source = heading_payload.get("source")
                if isinstance(source, str):
                    candidate = release_dir / source
                    if candidate.is_file():
                        raw_path = candidate
        if raw_path is None and isinstance(item.get("source_path"), str):
            candidate = workspace_root / str(item["source_path"])
            if candidate.is_file():
                raw_path = candidate
        if raw_path is not None:
            source_hashes[document_key] = sha256_file(raw_path)
        else:
            source_hashes[document_key] = "missing"
        rows.append(
            {
                "kind": "document",
                "key": document_key,
                "text": build_document_description(document_metadata, raw_path),
            }
        )
    if not any(row["kind"] == "document" for row in rows):
        raise ValueError("No document keys were found in embedding metadata.")
    return rows, source_hashes


def write_descriptions(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256_file(path)


@dataclass(frozen=True)
class AnchorHierarchy:
    """Loaded per-series and per-document semantic vectors."""

    series_vectors: dict[str, np.ndarray]
    document_vectors: dict[str, np.ndarray]
    provenance: dict[str, Any]

    @classmethod
    def load(
        cls,
        embedding_root: Path,
        *,
        manifest_file: str,
        vectors_file: str,
        corpus_manifest_sha256: str,
        embedding_backend: str,
        embedding_model: str,
        dimensions: int,
    ) -> "AnchorHierarchy":
        manifest_path = embedding_root / manifest_file
        vectors_path = embedding_root / vectors_file
        if not manifest_path.is_file() or not vectors_path.is_file():
            raise FileNotFoundError(
                "Hierarchical anchors require prepared artifacts. Run "
                "`cd newbaseline && uv run python scripts/embed_anchor_hierarchy.py` first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "artifact_version": ARTIFACT_VERSION,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "embedding_dimensions": dimensions,
            "vectors_file": vectors_file,
        }
        mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if mismatches:
            raise ValueError(f"Hierarchical anchor artifact is incompatible: {mismatches}")
        descriptions_file = manifest.get("descriptions_file")
        descriptions_sha256 = manifest.get("descriptions_sha256")
        descriptions_path = embedding_root / descriptions_file if isinstance(descriptions_file, str) else None
        if descriptions_path is None or not descriptions_path.is_file() or not isinstance(descriptions_sha256, str):
            raise ValueError("Hierarchical anchor artifact has no valid descriptions provenance.")
        if sha256_file(descriptions_path) != descriptions_sha256:
            raise ValueError("Hierarchical anchor descriptions do not match their manifest hash.")
        with np.load(vectors_path, allow_pickle=False) as payload:
            series_keys = payload["series_keys"].astype(str).tolist()
            document_keys = payload["document_keys"].astype(str).tolist()
            series_matrix = np.asarray(payload["series_vectors"], dtype=np.float32)
            document_matrix = np.asarray(payload["document_vectors"], dtype=np.float32)
        if (
            series_matrix.shape != (len(series_keys), dimensions)
            or document_matrix.shape != (len(document_keys), dimensions)
            or len(set(series_keys)) != len(series_keys)
            or len(set(document_keys)) != len(document_keys)
        ):
            raise ValueError("Hierarchical anchor vectors have invalid keys or dimensions.")
        return cls(
            series_vectors={key: normalize_vector(vector) for key, vector in zip(series_keys, series_matrix, strict=True)},
            document_vectors={key: normalize_vector(vector) for key, vector in zip(document_keys, document_matrix, strict=True)},
            provenance=manifest,
        )
