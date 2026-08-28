"""Dense/hybrid retrieval with metadata-to-source chunk traceability."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .anchor_hierarchy import AnchorHierarchy, normalize_vector
from .citation_expansion import CitationExpander
from .lexical import SqliteBm25Index, build_bm25_index
from .types import CitationPath, RetrievalHit

SUMMARY_SERIES = "release-summaries"
HYBRID_CANDIDATE_MULTIPLIER = 4
HYBRID_RRF_K = 60


class PaperEmbeddingCorpus:
    """Read a selected embedding collection and preserve every original chunk link."""

    def __init__(
        self,
        embedding_root: Path,
        workspace_root: Path | None = None,
        retrieval_backend: str = "semantic",
        lexical_index_file: str = "lexical.sqlite3",
        rrf_dense_weight: float = 0.5,
        rrf_bm25_weight: float = 0.5,
    ) -> None:
        self.root = embedding_root
        self.manifest = json.loads((embedding_root / "manifest.json").read_text(encoding="utf-8"))
        self._series = self.manifest["series"]
        source_chunk_directory = Path(self.manifest["source_chunk_directory"])
        self._chunk_root = (
            source_chunk_directory
            if source_chunk_directory.is_absolute()
            else (workspace_root or Path.cwd()) / source_chunk_directory
        )
        self._metadata_cache: dict[str, list[dict[str, Any]]] = {}
        self._chunk_cache: dict[str, list[dict[str, Any]]] = {}
        self._heading_index: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
        if retrieval_backend not in {"semantic", "hybrid"}:
            raise ValueError(f"Unsupported retrieval backend: {retrieval_backend}")
        self._validate_rrf_weights(rrf_dense_weight, rrf_bm25_weight)
        self.retrieval_backend = retrieval_backend
        self.rrf_dense_weight = rrf_dense_weight
        self.rrf_bm25_weight = rrf_bm25_weight
        self._lexical_index = (
            SqliteBm25Index(embedding_root / lexical_index_file)
            if retrieval_backend == "hybrid"
            else None
        )
        self._citation_expander = CitationExpander(
            self._resolve_target,
            self._chunk_for_hit,
            self._lexical_index,
            rrf_dense_weight,
            rrf_bm25_weight,
        )

    @property
    def available_numeric_series(self) -> set[str]:
        return {name for name in self._series if name != SUMMARY_SERIES}

    @property
    def has_summaries(self) -> bool:
        return SUMMARY_SERIES in self._series

    @property
    def all_series(self) -> list[str]:
        """Every group in manifest order, including release summaries."""
        return list(self._series)

    def searched_series_for(self, selected_series: list[str]) -> tuple[list[str], list[str]]:
        searched = [series for series in selected_series if series in self.available_numeric_series]
        empty = [series for series in selected_series if series not in self.available_numeric_series]
        if self.has_summaries:
            searched.append(SUMMARY_SERIES)
        return searched, empty

    def search(
        self,
        selected_series: list[str],
        query_embedding: np.ndarray,
        top_k: int,
        query_text: str | None = None,
    ) -> tuple[list[RetrievalHit], list[str], list[str]]:
        """Search selected groups with semantic or rank-fused hybrid retrieval."""
        if self.retrieval_backend == "hybrid":
            if query_text is None:
                raise ValueError("Hybrid retrieval requires query_text.")
            candidate_k = max(top_k, top_k * HYBRID_CANDIDATE_MULTIPLIER)
            dense_by_query, searched, empty = self.search_many(
                selected_series,
                np.asarray(query_embedding, dtype=np.float32).reshape(1, -1),
                candidate_k,
            )
            lexical_hits = self._lexical_hits(query_text, searched, candidate_k)
            return self._fuse_ranked_hits(
                dense_by_query[0],
                lexical_hits,
                top_k,
                dense_weight=self.rrf_dense_weight,
                bm25_weight=self.rrf_bm25_weight,
            ), searched, empty
        hits_by_query, searched, empty = self.search_many(
            selected_series, np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), top_k
        )
        return hits_by_query[0], searched, empty

    def search_hierarchical(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        query_text: str | None,
        hierarchy: AnchorHierarchy,
        series_weight: float,
        document_weight: float,
        chunk_weight: float,
    ) -> tuple[list[RetrievalHit], list[str]]:
        """Rank all manifest chunks before applying the normal optional RRF step."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        weights = (series_weight, document_weight, chunk_weight)
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("Hierarchical anchor weights must be non-negative and sum to 1.")
        candidate_k = max(top_k, top_k * HYBRID_CANDIDATE_MULTIPLIER) if self.retrieval_backend == "hybrid" else top_k
        dense_hits = self._hierarchical_dense_hits(
            query_embedding,
            candidate_k,
            hierarchy,
            series_weight,
            document_weight,
            chunk_weight,
        )
        searched = self.all_series
        if self.retrieval_backend != "hybrid":
            return dense_hits[:top_k], searched
        if query_text is None:
            raise ValueError("Hybrid retrieval requires query_text.")
        lexical_hits = [
            self._with_hierarchical_scores(
                hit,
                query_embedding,
                hierarchy,
                series_weight,
                document_weight,
                chunk_weight,
            )
            for hit in self._lexical_hits(query_text, searched, candidate_k)
        ]
        return self._fuse_ranked_hits(
            dense_hits,
            lexical_hits,
            top_k,
            dense_weight=self.rrf_dense_weight,
            bm25_weight=self.rrf_bm25_weight,
        ), searched

    def _with_hierarchical_scores(
        self,
        hit: RetrievalHit,
        query_embedding: np.ndarray,
        hierarchy: AnchorHierarchy,
        series_weight: float,
        document_weight: float,
        chunk_weight: float,
    ) -> RetrievalHit:
        """Attach hierarchy diagnostics to a lexical-only candidate before RRF."""
        record = self._series[hit.series]
        metadata_rows = self._load_metadata(record["metadata_file"])
        chunk_id = hit.metadata.get("chunk_id")
        row_index = next(
            (
                index
                for index, metadata in enumerate(metadata_rows)
                if metadata.get("chunk_id") == chunk_id
            ),
            None,
        )
        if row_index is None:
            raise ValueError(f"Cannot locate lexical hierarchy candidate {chunk_id!r}.")
        query = normalize_vector(np.asarray(query_embedding, dtype=np.float32))
        vector = np.asarray(np.load(self.root / record["vector_file"], mmap_mode="r")[row_index], dtype=np.float32)
        chunk_score = float(query @ vector / max(float(np.linalg.norm(vector)), 1e-12))
        document_key = hit.metadata.get("document_key")
        document_vector = hierarchy.document_vectors.get(document_key) if isinstance(document_key, str) else None
        series_vector = hierarchy.series_vectors.get(hit.series)
        if document_vector is None or series_vector is None:
            raise ValueError(f"Hierarchy artifact is incomplete for lexical candidate {chunk_id!r}.")
        document_score = float(query @ document_vector)
        series_score = float(query @ series_vector)
        combined = series_weight * series_score + document_weight * document_score + chunk_weight * chunk_score
        return replace(
            hit,
            anchor_series_score=series_score,
            anchor_document_score=document_score,
            anchor_chunk_score=chunk_score,
            anchor_hierarchical_score=combined,
        )

    def _hierarchical_dense_hits(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        hierarchy: AnchorHierarchy,
        series_weight: float,
        document_weight: float,
        chunk_weight: float,
    ) -> list[RetrievalHit]:
        query = normalize_vector(np.asarray(query_embedding, dtype=np.float32))
        score_parts: list[np.ndarray] = []
        chunk_parts: list[np.ndarray] = []
        document_parts: list[np.ndarray] = []
        series_parts: list[np.ndarray] = []
        ranges: list[tuple[int, int, str]] = []
        offset = 0
        for series, record in self._series.items():
            series_vector = hierarchy.series_vectors.get(series)
            if series_vector is None:
                raise ValueError(f"Hierarchy artifact has no series vector for {series!r}.")
            vectors = np.asarray(np.load(self.root / record["vector_file"], mmap_mode="r"), dtype=np.float32)
            metadata = self._load_metadata(record["metadata_file"])
            if len(vectors) != len(metadata):
                raise ValueError(f"Embedding and metadata rows differ for {series}.")
            chunk_scores = (vectors @ query) / np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)
            document_scores = np.empty(len(metadata), dtype=np.float32)
            for index, item in enumerate(metadata):
                document_key = item.get("document_key")
                document_vector = hierarchy.document_vectors.get(document_key) if isinstance(document_key, str) else None
                if document_vector is None:
                    raise ValueError(
                        f"Hierarchy artifact has no document vector for {document_key!r} in series {series}."
                    )
                document_scores[index] = float(query @ document_vector)
            series_score = float(query @ series_vector)
            combined = (
                series_weight * series_score
                + document_weight * document_scores
                + chunk_weight * chunk_scores
            )
            score_parts.append(combined)
            chunk_parts.append(chunk_scores)
            document_parts.append(document_scores)
            series_parts.append(np.full(len(metadata), series_score, dtype=np.float32))
            ranges.append((offset, offset + len(metadata), series))
            offset += len(metadata)
        if not score_parts:
            return []
        scores = np.concatenate(score_parts)
        chunk_scores = np.concatenate(chunk_parts)
        document_scores = np.concatenate(document_parts)
        series_scores = np.concatenate(series_parts)
        limit = min(top_k, len(scores))
        positions = np.argpartition(-scores, limit - 1)[:limit]
        positions = sorted(positions.tolist(), key=lambda position: (-float(scores[position]), position))
        ends = np.asarray([end for _, end, _ in ranges])
        hits: list[RetrievalHit] = []
        for position in positions:
            series_index = int(np.searchsorted(ends, position, side="right"))
            start, _, series = ranges[series_index]
            local_index = position - start
            record = self._series[series]
            metadata = self._load_metadata(record["metadata_file"])[local_index]
            chunk = self._source_chunk(record["chunk_file"], metadata)
            hits.append(
                RetrievalHit(
                    score=float(scores[position]),
                    series=series,
                    text=chunk["text"],
                    metadata=metadata,
                    source_chunk_file=record["chunk_file"],
                    retrieval_method="hierarchical_semantic",
                    anchor_series_score=float(series_scores[position]),
                    anchor_document_score=float(document_scores[position]),
                    anchor_chunk_score=float(chunk_scores[position]),
                    anchor_hierarchical_score=float(scores[position]),
                )
            )
        return hits

    def _lexical_hits(
        self,
        query_text: str,
        searched_series: list[str],
        top_k: int,
    ) -> list[RetrievalHit]:
        if self._lexical_index is None:
            return []
        hits: list[RetrievalHit] = []
        for series, metadata_index, bm25_score in self._lexical_index.search(
            query_text, searched_series, top_k
        ):
            record = self._series[series]
            metadata = self._load_metadata(record["metadata_file"])[metadata_index]
            source_chunk_file = record["chunk_file"]
            chunk = self._source_chunk(source_chunk_file, metadata)
            hits.append(
                RetrievalHit(
                    score=bm25_score,
                    series=series,
                    text=chunk["text"],
                    metadata=metadata,
                    source_chunk_file=source_chunk_file,
                    retrieval_method="lexical",
                )
            )
        return hits

    @staticmethod
    def _fuse_ranked_hits(
        dense_hits: list[RetrievalHit],
        lexical_hits: list[RetrievalHit],
        top_k: int,
        *,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[RetrievalHit]:
        """Fuse dense and BM25 rankings with weighted reciprocal rank fusion."""
        PaperEmbeddingCorpus._validate_rrf_weights(dense_weight, bm25_weight)
        ranked: dict[tuple[str, str], dict[str, Any]] = {}
        for method, weight, hits in (
            ("dense", dense_weight, dense_hits),
            ("lexical", bm25_weight, lexical_hits),
        ):
            for rank, hit in enumerate(hits, start=1):
                identity = hit.metadata.get("chunk_id", hit.metadata.get("source_chunk_index"))
                key = (hit.series, str(identity))
                row = ranked.setdefault(
                    key,
                    {"hit": hit, "score": 0.0, "dense_rank": None, "lexical_rank": None},
                )
                row["score"] += weight / (HYBRID_RRF_K + rank)
                row[f"{method}_rank"] = rank
                if method == "dense":
                    row["hit"] = hit
        ordered = sorted(
            ranked.values(),
            key=lambda row: (
                -float(row["score"]),
                min(row["dense_rank"] or 10**9, row["lexical_rank"] or 10**9),
                str(row["hit"].metadata.get("chunk_id", "")),
            ),
        )[:top_k]
        return [
            RetrievalHit(
                score=float(row["score"]),
                series=row["hit"].series,
                text=row["hit"].text,
                metadata=dict(row["hit"].metadata),
                source_chunk_file=row["hit"].source_chunk_file,
                retrieval_method="hybrid",
                dense_rank=row["dense_rank"],
                lexical_rank=row["lexical_rank"],
                anchor_series_score=row["hit"].anchor_series_score,
                anchor_document_score=row["hit"].anchor_document_score,
                anchor_chunk_score=row["hit"].anchor_chunk_score,
                anchor_hierarchical_score=row["hit"].anchor_hierarchical_score,
            )
            for row in ordered
        ]

    @staticmethod
    def _validate_rrf_weights(dense_weight: float, bm25_weight: float) -> None:
        if dense_weight < 0 or bm25_weight < 0 or not np.isclose(dense_weight + bm25_weight, 1.0):
            raise ValueError("RRF dense and BM25 weights must be non-negative and sum to 1.")

    def build_lexical_index(self, output_path: Path) -> int:
        """Build the persistent BM25 index aligned to embedding metadata rows."""
        def rows():
            for series, record in self._series.items():
                metadata_rows = self._load_metadata(record["metadata_file"])
                for metadata_index, metadata in enumerate(metadata_rows):
                    chunk = self._source_chunk(record["chunk_file"], metadata)
                    chunk_id = metadata.get("chunk_id")
                    if not isinstance(chunk_id, str) or not isinstance(chunk.get("text"), str):
                        continue
                    heading = chunk.get("heading")
                    yield (
                        chunk_id,
                        series,
                        metadata_index,
                        heading if isinstance(heading, str) else "",
                        chunk["text"],
                    )

        return build_bm25_index(output_path, rows())

    def search_many(
        self, selected_series: list[str], query_embeddings: np.ndarray, top_k: int
    ) -> tuple[list[list[RetrievalHit]], list[str], list[str]]:
        """Search one index with many query vectors to avoid repeated FAISS setup."""
        searched, empty = self.searched_series_for(selected_series)
        if not searched:
            return [[] for _ in range(len(query_embeddings))], searched, empty
        matrices: list[np.ndarray] = []
        owners: list[str] = []
        for series in searched:
            record = self._series[series]
            vectors = np.load(self.root / record["vector_file"], mmap_mode="r")
            metadata = self._load_metadata(record["metadata_file"])
            if len(vectors) != len(metadata):
                raise ValueError(f"Embedding and metadata rows differ for {series}.")
            matrices.append(np.asarray(vectors, dtype=np.float32))
            owners.extend([series] * len(vectors))
        all_vectors = np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32)
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or not len(queries):
            raise ValueError("query_embeddings must be a non-empty two-dimensional matrix.")
        if all_vectors.shape[1] != queries.shape[1]:
            raise ValueError(f"Corpus has {all_vectors.shape[1]} dimensions, query has {queries.shape[1]}.")
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - installation error
            raise RuntimeError("Install faiss-cpu to run semantic retrieval.") from exc
        index = faiss.IndexFlatIP(all_vectors.shape[1])
        index.add(all_vectors)
        scores, positions = index.search(np.ascontiguousarray(queries), min(top_k, len(owners)))
        offsets: dict[str, int] = {}
        offset = 0
        for series, matrix in zip(searched, matrices, strict=True):
            offsets[series] = offset
            offset += len(matrix)
        hits_by_query: list[list[RetrievalHit]] = []
        for query_scores, query_positions in zip(scores, positions, strict=True):
            hits: list[RetrievalHit] = []
            for score, position in zip(query_scores, query_positions, strict=True):
                if position < 0:
                    continue
                series = owners[int(position)]
                local_index = int(position) - offsets[series]
                record = self._series[series]
                metadata = self._load_metadata(record["metadata_file"])[local_index]
                source_chunk_file = record["chunk_file"]
                chunk = self._source_chunk(source_chunk_file, metadata)
                hits.append(
                    RetrievalHit(
                        score=float(score),
                        series=series,
                        text=chunk["text"],
                        metadata=metadata,
                        source_chunk_file=source_chunk_file,
                    )
                )
            hits_by_query.append(hits)
        return hits_by_query, searched, empty

    def expand_citations(
        self,
        seed_hits: list[RetrievalHit],
        *,
        max_depth: int,
        max_citation_chunks: int,
        chunks_per_heading: int,
        facets: list[str],
        facet_embeddings: np.ndarray,
        min_gain: float,
        embed_many: Callable[[list[str]], list[list[float]]] | None = None,
        embedding_batch_size: int = 128,
        selection_strategy: str = "gain",
        query_embedding: np.ndarray | None = None,
        query_text: str | None = None,
    ) -> tuple[list[RetrievalHit], list[CitationPath]]:
        """Expand precise citations using facet gain or query-semantic BFS.

        Expansion deliberately reads raw chunk files rather than the active
        embedding selection. A cited target can therefore be in another local
        3GPP series, while the semantic seed retrieval remains paper-subset
        compatible. Citations lacking a target heading are ignored because
        they cannot be resolved to a defensible target clause.
        """
        return self._citation_expander.expand(
            seed_hits,
            max_depth=max_depth,
            max_citation_chunks=max_citation_chunks,
            chunks_per_heading=chunks_per_heading,
            facets=facets,
            facet_embeddings=facet_embeddings,
            min_gain=min_gain,
            embed_many=embed_many,
            embedding_batch_size=embedding_batch_size,
            selection_strategy=selection_strategy,
            query_embedding=query_embedding,
            query_text=query_text,
        )


    def _load_metadata(self, filename: str) -> list[dict[str, Any]]:
        if filename not in self._metadata_cache:
            path = self.root / filename
            self._metadata_cache[filename] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return self._metadata_cache[filename]

    def _chunk_for_hit(self, hit: RetrievalHit) -> dict[str, Any]:
        if not hit.source_chunk_file:
            raise ValueError(f"Retrieval hit {hit.metadata.get('chunk_id')} has no source chunk file")
        return self._source_chunk(hit.source_chunk_file, hit.metadata)

    def _source_chunk(self, chunk_file: str, metadata: dict[str, Any]) -> dict[str, Any]:
        chunks = self._load_chunk_file(chunk_file)
        source_index = metadata.get("source_chunk_index")
        if isinstance(source_index, int) and 0 <= source_index < len(chunks):
            chunk = chunks[source_index]
        else:
            expected_chunk_id = metadata.get("chunk_id")
            chunk = next((item for item in chunks if item.get("chunk_id") == expected_chunk_id), None)
            if chunk is None:
                raise ValueError(f"Cannot find source chunk {expected_chunk_id} in {chunk_file}.")
        expected_chunk_id = metadata.get("chunk_id")
        if expected_chunk_id and chunk.get("chunk_id") != expected_chunk_id:
            raise ValueError(f"Chunk identity mismatch for {chunk_file} row {source_index}.")
        return chunk

    def _resolve_target(self, reference: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        target_file = reference.get("target_chunk_file")
        document_id = reference.get("document_id")
        heading = reference.get("target_heading")
        if not isinstance(target_file, str):
            return [], "missing_target_chunk_file"
        if not isinstance(document_id, str) or not isinstance(heading, str):
            return [], "invalid_target_reference"
        try:
            index = self._index_headings(target_file)
        except FileNotFoundError:
            return [], "target_chunk_file_missing"
        targets = index.get((document_id, heading), [])
        if targets:
            return targets, "resolved"
        descendant_targets = [
            chunk
            for (candidate_document_id, candidate_heading), chunks in index.items()
            if candidate_document_id == document_id and candidate_heading.startswith(f"{heading}.")
            for chunk in chunks
        ]
        if descendant_targets:
            return descendant_targets, "resolved_descendant"
        return [], "target_heading_not_found"

    def _index_headings(self, chunk_file: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
        if chunk_file not in self._heading_index:
            index: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for chunk in self._load_chunk_file(chunk_file):
                document_id = chunk.get("document_id")
                heading = chunk.get("heading")
                if isinstance(document_id, str) and isinstance(heading, str):
                    index.setdefault((document_id, heading), []).append(chunk)
            for chunks in index.values():
                chunks.sort(key=lambda item: item.get("chunk_index_in_heading", 0))
            self._heading_index[chunk_file] = index
        return self._heading_index[chunk_file]

    def _load_chunk_file(self, chunk_file: str) -> list[dict[str, Any]]:
        if chunk_file not in self._chunk_cache:
            path = Path(chunk_file)
            if not path.is_absolute():
                path = self._chunk_root / path
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunks = payload["chunks"] if isinstance(payload, dict) else payload
            if not isinstance(chunks, list):
                raise ValueError(f"Chunk file {chunk_file} does not contain a chunk list.")
            for source_chunk_index, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    # Raw chunk files predate embedding metadata. Preserve an
                    # existing index, otherwise attach the stable file offset
                    # in memory so citation-expanded chunks remain traceable.
                    chunk.setdefault("source_chunk_index", source_chunk_index)
            self._chunk_cache[chunk_file] = chunks
        return self._chunk_cache[chunk_file]
