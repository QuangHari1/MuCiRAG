"""Load and validate frozen series/document anchors for hierarchical retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1


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
                "`uv run --project MuCiRAG python mucirag.py assets download` from the repository root."
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
