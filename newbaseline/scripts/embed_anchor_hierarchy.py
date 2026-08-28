"""Prepare reusable series/document embeddings for hierarchical anchor retrieval.

This is deliberately separate from benchmark execution: it is the only command
that embeds document descriptions, and it writes artifacts for one configured
embedding selection under ``dataset/3gpp/Embeddings``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from newbaseline.src.embeddings import create_embedding_provider
from newbaseline.src.rag.anchor_hierarchy import ARTIFACT_VERSION, build_descriptions, sha256_file
from newbaseline.src.settings import load_settings


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def write_descriptions_atomically(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        for row in rows:
            temporary.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)
    return sha256_file(path)


def write_vectors_atomically(
    path: Path,
    series_rows: list[dict[str, str]],
    document_rows: list[dict[str, str]],
    vectors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as temporary:
        np.savez_compressed(
            temporary,
            series_keys=np.asarray([row["key"] for row in series_rows]),
            series_vectors=vectors[: len(series_rows)],
            document_keys=np.asarray([row["key"] for row in document_rows]),
            document_vectors=vectors[len(series_rows) :],
        )
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def digest_source_hashes(source_hashes: dict[str, str]) -> str:
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embed_rows(provider: Any, rows: list[dict[str, str]], batch_size: int, dimensions: int) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = np.asarray(provider.embed([row["text"] for row in batch]), dtype=np.float32)
        if vectors.shape != (len(batch), dimensions):
            raise ValueError("Embedding provider returned invalid hierarchy vector dimensions.")
        batches.append(vectors)
        print(f"Embedded hierarchy descriptions: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(batches, axis=0)


def expected_manifest(
    settings: Any,
    corpus_manifest_sha256: str,
    descriptions_sha256: str,
    source_sha256: str,
    vectors_file: str,
) -> dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "selection_id": settings.get("rag", "selection_id"),
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "embedding_backend": settings.get("embedding", "backend"),
        "embedding_model": settings.get("embedding", "model"),
        "embedding_dimensions": settings.get("embedding", "dimensions"),
        "descriptions_sha256": descriptions_sha256,
        "description_source_sha256": source_sha256,
        "vectors_file": vectors_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an incompatible existing artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    embedding_root = settings.embedding_root(settings.get("rag", "selection_id"))
    corpus_manifest_path = embedding_root / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    resources = settings.workspace_root / settings.get("rag", "resources_dir")
    rows, source_hashes = build_descriptions(
        corpus_manifest,
        embedding_root,
        settings.workspace_root,
        settings.metadata_dir,
        settings.release_dir,
        resources / settings.get("rag", "series_descriptions"),
    )
    descriptions_path = embedding_root / settings.get("rag", "anchor_hierarchy_descriptions_file")
    vectors_path = embedding_root / settings.get("rag", "anchor_hierarchy_vectors_file")
    artifact_manifest_path = embedding_root / settings.get("rag", "anchor_hierarchy_manifest_file")
    descriptions_payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    descriptions_sha256 = hashlib.sha256(descriptions_payload.encode("utf-8")).hexdigest()
    expected = expected_manifest(
        settings,
        sha256_file(corpus_manifest_path),
        descriptions_sha256,
        digest_source_hashes(source_hashes),
        vectors_path.name,
    )
    if artifact_manifest_path.is_file() and vectors_path.is_file() and descriptions_path.is_file():
        existing = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            print(f"Hierarchy artifact is already compatible: {artifact_manifest_path}")
            return 0
        if not args.force:
            raise RuntimeError(
                "Existing hierarchy artifact is incompatible. Re-run with --force after confirming the new "
                "selection/model is intended."
            )

    series_rows = [row for row in rows if row["kind"] == "series"]
    document_rows = [row for row in rows if row["kind"] == "document"]
    provider = create_embedding_provider(settings)
    vectors = embed_rows(provider, rows, settings.get("embedding", "batch_size"), settings.get("embedding", "dimensions"))
    write_descriptions_atomically(descriptions_path, rows)
    write_vectors_atomically(vectors_path, series_rows, document_rows, vectors)
    write_json_atomically(
        artifact_manifest_path,
        {
            **expected,
            "descriptions_file": descriptions_path.name,
            "series_count": len(series_rows),
            "document_count": len(document_rows),
            "source_hash_count": len(source_hashes),
        },
    )
    print(f"Wrote hierarchy artifact: {artifact_manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
