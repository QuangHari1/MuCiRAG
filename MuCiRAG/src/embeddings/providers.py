"""Embedding adapters selected entirely through ``[embedding]`` configuration."""

from __future__ import annotations

import time
from typing import Any, Protocol

from MuCiRAG.src.settings import require_secret


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    def __init__(self, model: str, dimensions: int, max_retries: int) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=require_secret("OPENAI_API_KEY"))
        self._model = model
        self.dimensions = dimensions
        self._max_retries = max_retries

    def embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
                if len(vectors) != len(texts) or any(len(vector) != self.dimensions for vector in vectors):
                    raise RuntimeError("Embedding provider returned an unexpected vector shape")
                return vectors
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model: str, dimensions: int, normalize: bool) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install sentence-transformers before using [embedding].backend = 'sentence_transformers'."
            ) from exc
        self._model = SentenceTransformer(model)
        self.dimensions = dimensions
        self._normalize = normalize
        actual_dimensions = self._model.get_sentence_embedding_dimension()
        if actual_dimensions != dimensions:
            raise ValueError(f"Configured dimensions={dimensions}, but {model} produces {actual_dimensions}.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def create_embedding_provider(settings: Any) -> EmbeddingProvider:
    backend = settings.get("embedding", "backend")
    model = settings.get("embedding", "model")
    dimensions = settings.get("embedding", "dimensions")
    if backend == "openai":
        return OpenAIEmbeddingProvider(model, dimensions, settings.get("embedding", "max_retries"))
    if backend == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(
            model,
            dimensions,
            settings.get("embedding", "normalize"),
        )
    raise ValueError(f"Unsupported [embedding].backend: {backend}")
