"""Embed full or selected Release 18 chunks with the baseline's embedding model.

Vectors are stored as float32 ``.npy`` arrays. Each embedding file has a
``.metadata.jsonl`` sidecar that maps its row to the original, unfiltered
``ChunkSeries*.json`` index and stable chunk metadata.
Interrupted runs resume from ``.partial/`` and publish a final array only after
the whole series succeeds. A selection manifest filters by version-aware
``document_key`` and writes to a separate output namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MuCiRAG.src.embeddings import EmbeddingProvider, create_embedding_provider
from MuCiRAG.src.settings import load_settings

SETTINGS = load_settings()
MODEL = SETTINGS.get("embedding", "model")
DIMENSIONS = SETTINGS.get("embedding", "dimensions")
DEFAULT_BATCH_SIZE = SETTINGS.get("embedding", "batch_size")

WORKSPACE_ROOT = SETTINGS.workspace_root
CHUNK_DIR = SETTINGS.chunk_dir
OUTPUT_DIR = SETTINGS.embedding_dir
PARTIAL_DIR = OUTPUT_DIR / ".partial"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SERIES_PATTERN = re.compile(r"^ChunkSeries(?P<series>\d+)\.json$")
SUMMARY_SOURCE_ID = "release-summaries"
SELECTION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FULL_SELECTION_ID = "full-gsma-rel18"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_sources(selected_series: set[str] | None) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for path in sorted(CHUNK_DIR.glob("ChunkSeries*.json")):
        match = SERIES_PATTERN.match(path.name)
        if not match:
            continue
        series = match.group("series")
        if selected_series is None or series in selected_series:
            sources.append((series, path))
    # The four paper summaries are not numeric 3GPP series, so keep them in a
    # named pseudo-series and include them only when the selection asks for it.
    summary_path = CHUNK_DIR / SETTINGS.get("paper_release_summaries", "chunk_file")
    if summary_path.exists() and (selected_series is None or SUMMARY_SOURCE_ID in selected_series):
        sources.append((SUMMARY_SOURCE_ID, summary_path))
    if not sources:
        raise FileNotFoundError(f"No chunk files found in {CHUNK_DIR}")
    return sources


def load_chunks(
    path: Path, selected_document_keys: set[str] | None = None
) -> list[dict[str, Any]]:
    chunks = read_json(path).get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"Missing chunks list in {path}")
    selected_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            raise ValueError(f"Chunk {index} in {path} has no text field")
        if not chunk["text"].strip():
            raise ValueError(f"Chunk {index} in {path} has empty text")
        if selected_document_keys is None:
            selected_chunks.append({**chunk, "_source_chunk_index": index})
            continue
        document_key = chunk.get("document_key")
        if not isinstance(document_key, str):
            raise ValueError(
                f"Chunk {index} in {path} has no document_key. "
                "Regenerate chunk metadata with MuCiRAG/scripts/extract_heading.py "
                "and MuCiRAG/scripts/chunking.py before using --selection."
            )
        if document_key in selected_document_keys:
            selected_chunks.append({**chunk, "_source_chunk_index": index})
    return selected_chunks


def metadata_path_for(series: str) -> Path:
    return OUTPUT_DIR / f"EmbeddingsSeries{series}.metadata.jsonl"


def write_row_metadata(path: Path, chunks: list[dict[str, Any]]) -> str:
    """Write a durable row-to-original-chunk mapping without duplicating chunk text.

    The sidecar retains heading, source-file, bibliography, and citation-link
    provenance.  This makes vectors portable without relying on row order or
    on a particular source filename (``raw.md`` versus verbalized Markdown).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        for embedding_row, chunk in enumerate(chunks):
            source_chunk_index = chunk.get("_source_chunk_index")
            if not isinstance(source_chunk_index, int):
                raise ValueError("Selected chunk is missing its original source index")
            row = {
                key: value
                for key, value in chunk.items()
                if key not in {"text", "_source_chunk_index"}
            }
            row.update({
                "embedding_row": embedding_row,
                "source_chunk_index": source_chunk_index,
            })
            temporary_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)
    return sha256_file(path)


def chunk_digest(chunks: list[dict[str, Any]]) -> str:
    """Hash embed-relevant source metadata, excluding the local row-mapping helper."""
    source_chunks = [
        {key: value for key, value in chunk.items() if key != "_source_chunk_index"}
        for chunk in chunks
    ]
    return hashlib.sha256(
        json.dumps(source_chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def configure_output(selection_id: str) -> None:
    """Keep full and filtered vector runs in separate resumable namespaces."""
    if not SELECTION_ID_PATTERN.fullmatch(selection_id):
        raise ValueError(f"Invalid selection_id: {selection_id}")
    global OUTPUT_DIR, PARTIAL_DIR, MANIFEST_PATH
    OUTPUT_DIR = SETTINGS.embedding_root(selection_id)
    PARTIAL_DIR = OUTPUT_DIR / ".partial"
    MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def load_selection(path: Path | None) -> tuple[str, set[str] | None, str | None]:
    """Load an optional document-key selection and identify its exact revision."""
    if path is None:
        return FULL_SELECTION_ID, None, None

    payload = read_json(path)
    selection_id = payload.get("selection_id")
    documents = payload.get("documents")
    if not isinstance(selection_id, str) or not isinstance(documents, list):
        raise ValueError(f"Invalid selection manifest: {path}")
    document_keys: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict) or not isinstance(document.get("document_key"), str):
            raise ValueError(f"Selection document {index} in {path} has no document_key")
        document_keys.add(document["document_key"])
    if not document_keys:
        raise ValueError(f"Selection manifest is empty: {path}")
    return selection_id, document_keys, sha256_file(path)


def read_manifest(selection_id: str, selection_sha256: str | None) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "source_chunk_directory": str(CHUNK_DIR.relative_to(WORKSPACE_ROOT)),
            "selection_id": selection_id,
            "selection_sha256": selection_sha256,
            "series": {},
        }
    return read_json(MANIFEST_PATH)


def has_completed_vectors(
    manifest: dict[str, Any], series: str, source_digest: str, chunk_count: int
) -> bool:
    record = manifest.get("series", {}).get(series)
    output_path = OUTPUT_DIR / f"EmbeddingsSeries{series}.npy"
    if not isinstance(record, dict) or not output_path.exists():
        return False
    if record.get("source_sha256") != source_digest:
        return False
    if record.get("chunk_count") != chunk_count:
        return False
    try:
        vectors = np.load(output_path, mmap_mode="r")
        return vectors.shape == (chunk_count, DIMENSIONS) and vectors.dtype == np.float32
    except (OSError, ValueError):
        return False


def read_resume_offset(state_path: Path, source_digest: str, chunk_count: int) -> int:
    if not state_path.exists():
        return 0
    state = read_json(state_path)
    if (
        state.get("source_sha256") != source_digest
        or state.get("chunk_count") != chunk_count
        or state.get("model") != MODEL
        or state.get("dimensions") != DIMENSIONS
    ):
        return 0
    offset = state.get("next_chunk_index", 0)
    return offset if isinstance(offset, int) and 0 <= offset <= chunk_count else 0


def save_resume_offset(
    state_path: Path, source_digest: str, chunk_count: int, next_chunk_index: int
) -> None:
    write_json_atomically(
        state_path,
        {
            "model": MODEL,
            "dimensions": DIMENSIONS,
            "source_sha256": source_digest,
            "chunk_count": chunk_count,
            "next_chunk_index": next_chunk_index,
        },
    )


def create_embeddings(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    return provider.embed(texts)


def embed_series(
    provider: EmbeddingProvider,
    series: str,
    source_path: Path,
    chunks: list[dict[str, Any]],
    batch_size: int,
    manifest: dict[str, Any],
) -> None:
    source_digest = chunk_digest(chunks)
    chunk_count = len(chunks)
    output_path = OUTPUT_DIR / f"EmbeddingsSeries{series}.npy"
    metadata_path = metadata_path_for(series)
    partial_path = PARTIAL_DIR / f"EmbeddingsSeries{series}.npy"
    state_path = PARTIAL_DIR / f"EmbeddingsSeries{series}.json"

    if has_completed_vectors(manifest, series, source_digest, chunk_count):
        record = manifest["series"][series]
        if not metadata_path.exists():
            record["metadata_sha256"] = write_row_metadata(metadata_path, chunks)
            record["metadata_file"] = metadata_path.name
            write_json_atomically(MANIFEST_PATH, manifest)
            print(f"Series {series}: wrote row metadata for existing vectors")
        print(f"Series {series}: already complete ({chunk_count} chunks)")
        return

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    offset = read_resume_offset(state_path, source_digest, chunk_count)
    if offset == 0:
        partial_path.unlink(missing_ok=True)
        matrix = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(chunk_count, DIMENSIONS),
        )
    else:
        matrix = np.lib.format.open_memmap(partial_path, mode="r+")

    print(f"Series {series}: {offset}/{chunk_count} chunks embedded")
    while offset < chunk_count:
        end = min(offset + batch_size, chunk_count)
        texts = [chunk["text"] for chunk in chunks[offset:end]]
        matrix[offset:end] = np.asarray(create_embeddings(provider, texts), dtype=np.float32)
        matrix.flush()
        offset = end
        save_resume_offset(state_path, source_digest, chunk_count, offset)
        print(f"Series {series}: {offset}/{chunk_count} chunks embedded", flush=True)

    del matrix
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(output_path)
    state_path.unlink(missing_ok=True)
    metadata_sha256 = write_row_metadata(metadata_path, chunks)
    manifest.setdefault("series", {})[series] = {
        "chunk_file": source_path.name,
        "vector_file": output_path.name,
        "metadata_file": metadata_path.name,
        "metadata_sha256": metadata_sha256,
        "chunk_count": chunk_count,
        "source_sha256": source_digest,
    }
    write_json_atomically(MANIFEST_PATH, manifest)


def print_dry_run(sources: list[tuple[str, Path, list[dict[str, Any]]]]) -> None:
    total_chunks = 0
    total_tokens = 0
    selected_documents: set[str] = set()
    for _, _, chunks in sources:
        total_chunks += len(chunks)
        total_tokens += sum(chunk.get("token_count_estimate", 0) for chunk in chunks)
        selected_documents.update(
            chunk.get("document_key", chunk.get("document_id", "")) for chunk in chunks
        )
    print(f"Chunk files: {len(sources)}")
    print(f"Documents: {len(selected_documents)}")
    print(f"Chunks: {total_chunks}")
    print(f"Estimated input tokens: {total_tokens}")
    print(f"Output directory: {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", help="Embed one series, e.g. --series 23")
    parser.add_argument(
        "--selection",
        type=Path,
        help="Selection JSON from prepare_embedding_selection.py; omit for the full GSMA corpus",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without calling OpenAI")
    parser.add_argument(
        "--allow-selection-extension",
        action="store_true",
        help="Permit an append-only selection update with independent new chunk sources.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    selected_series = set(args.series) if args.series else None
    selection_id, selected_document_keys, selection_sha256 = load_selection(args.selection)
    configure_output(selection_id)
    sources = []
    for series, source_path in chunk_sources(selected_series):
        chunks = load_chunks(source_path, selected_document_keys)
        if chunks:
            sources.append((series, source_path, chunks))
    if not sources:
        raise ValueError("No chunks matched the requested selection and series filters")
    if args.dry_run:
        print_dry_run(sources)
        return

    try:
        import numpy as numpy
    except ImportError as error:
        raise RuntimeError(
            "Install dependencies first: cd MuCiRAG && uv sync"
        ) from error

    global np
    np = numpy

    manifest = read_manifest(selection_id, selection_sha256)
    if (
        manifest.get("model") != MODEL
        or manifest.get("dimensions") != DIMENSIONS
        or manifest.get("selection_id") != selection_id
    ):
        raise RuntimeError(f"Existing manifest at {MANIFEST_PATH} uses a different embedding configuration")
    if manifest.get("selection_sha256") != selection_sha256:
        if not args.allow_selection_extension:
            raise RuntimeError(
                "Selection content changed. Re-run with --allow-selection-extension only when "
                "the change appends a new independent chunk source."
            )
        history = manifest.setdefault("selection_sha256_history", [])
        if manifest.get("selection_sha256") not in history:
            history.append(manifest.get("selection_sha256"))
        manifest["selection_sha256"] = selection_sha256
        write_json_atomically(MANIFEST_PATH, manifest)
    provider = create_embedding_provider(SETTINGS)
    for series, source_path, chunks in sources:
        embed_series(provider, series, source_path, chunks, args.batch_size, manifest)


if __name__ == "__main__":
    main()
