"""The neural series router used by the Telco-oRAG paper baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

PAPER_SERIES: tuple[str, ...] = tuple(
    str(series) for series in (21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38)
)


class PaperNNRouter:
    """Load and run the paper checkpoint without changing its 18-label space."""

    def __init__(self, checkpoint_path: Path, descriptions_path: Path, similarity_scale: float = 10.0) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - installation error
            raise RuntimeError("Install torch to use the paper NN router.") from exc
        self._torch = torch
        self._similarity_scale = similarity_scale
        self._series, self._description_embeddings = self._load_descriptions(descriptions_path)
        self._model = self._build_model(nn, len(self._series), self._description_embeddings.shape[1])
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before weights_only
            state = torch.load(checkpoint_path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    @staticmethod
    def _build_model(nn: object, output_size: int, embedding_size: int):
        import torch
        import torch.nn.functional as functional

        class RouterNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1_1 = nn.Linear(embedding_size, 768)
                self.layer1_2 = nn.Linear(768, 512)
                self.layer1_3 = nn.Linear(512, 256)
                self.dropout1 = nn.Dropout(0.2)
                self.layer2_1 = nn.Linear(output_size, 128)
                self.layer2_2 = nn.Linear(128, 256)
                self.dropout2 = nn.Dropout(0.05)
                self.batchnorm1 = nn.BatchNorm1d(256)
                self.batchnorm2 = nn.BatchNorm1d(256)
                self.alfa = nn.Parameter(torch.ones(1), requires_grad=True)
                self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
                self.output_layer1 = nn.Linear(256, 128)
                self.output_layer2 = nn.Linear(128, output_size)
                self.leaky_relu = nn.LeakyReLU(0.01)

            def forward(self, embeddings, similarities):
                x1 = functional.relu(self.layer1_1(embeddings))
                x1 = self.dropout1(x1)
                x1 = functional.relu(self.layer1_2(x1))
                x1 = self.dropout1(x1)
                x1 = functional.relu(self.layer1_3(x1))
                x1 = self.batchnorm1(x1)
                x2 = functional.relu(self.layer2_1(similarities))
                x2 = self.dropout2(x2)
                x2 = functional.relu(self.layer2_2(x2))
                x2 = self.batchnorm2(x2)
                combined = self.alfa * x1 + self.beta * x2
                output = self.output_layer1(self.leaky_relu(combined))
                return self.output_layer2(self.leaky_relu(output))

        return RouterNetwork()

    @staticmethod
    def _load_descriptions(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
        data = json.loads(path.read_text(encoding="utf-8"))
        series = [label.split()[0] for label in data]
        vectors = [payload["embeddings"] for payload in data.values()]
        if tuple(series) != PAPER_SERIES:
            raise ValueError(f"Expected paper router labels {PAPER_SERIES}, got {tuple(series)}")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(PAPER_SERIES):
            raise ValueError(f"Invalid series description matrix in {path}")
        return tuple(series), matrix

    def route(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        """Return paper labels, including labels whose corpus group is empty."""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._description_embeddings.shape[1]:
            raise ValueError(f"Router expects {self._description_embeddings.shape[1]} dimensions, got {query.shape[1]}.")
        similarities = query @ self._description_embeddings.T
        with self._torch.no_grad():
            probabilities = self._torch.softmax(
                self._torch.from_numpy(similarities) * self._similarity_scale, dim=1
            )
            logits = self._model(self._torch.from_numpy(query), probabilities)
            _, indices = self._torch.topk(logits, k=min(top_k, len(self._series)), dim=1)
        return [self._series[index] for index in indices[0].tolist()]


class SemanticSeriesRouter:
    """Portable fallback router for a non-paper embedding model.

    It embeds the same 18 textual series descriptions with the selected model
    and ranks cosine similarity. It intentionally does not claim to reproduce
    the paper's trained NN router.
    """

    def __init__(self, descriptions_path: Path, embed_many: Callable[[list[str]], list[list[float]]]) -> None:
        data = json.loads(descriptions_path.read_text(encoding="utf-8"))
        self._series = tuple(label.split()[0] for label in data)
        if self._series != PAPER_SERIES:
            raise ValueError(f"Expected router labels {PAPER_SERIES}, got {self._series}")
        descriptions = [payload["description"] for payload in data.values()]
        self._description_embeddings = self._normalize(np.asarray(embed_many(descriptions), dtype=np.float32))

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def route(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._description_embeddings.shape[1]:
            raise ValueError("Query embedding dimensions do not match semantic router descriptions.")
        scores = self._normalize(query) @ self._description_embeddings.T
        indices = np.argsort(scores[0])[::-1][:top_k]
        return [self._series[index] for index in indices]
