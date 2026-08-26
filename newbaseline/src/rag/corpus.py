"""Dense/hybrid retrieval with metadata-to-source chunk traceability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .lexical import SqliteBm25Index, build_bm25_index
from .types import CitationPath, RetrievalHit

SUMMARY_SERIES = "release-summaries"
RESOLVED_CITATION_STATUSES = {"resolved", "resolved_descendant"}
HYBRID_CANDIDATE_MULTIPLIER = 4
HYBRID_RRF_K = 60


@dataclass(frozen=True)
class _CitationCandidate:
    parent: RetrievalHit
    reference: dict[str, Any]
    target: dict[str, Any]
    target_file: str
    score: float | None
    order: int


class PaperEmbeddingCorpus:
    """Read a selected embedding collection and preserve every original chunk link."""

    def __init__(
        self,
        embedding_root: Path,
        workspace_root: Path | None = None,
        retrieval_backend: str = "semantic",
        lexical_index_file: str = "lexical.sqlite3",
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
        self.retrieval_backend = retrieval_backend
        self._lexical_index = (
            SqliteBm25Index(embedding_root / lexical_index_file)
            if retrieval_backend == "hybrid"
            else None
        )

    @property
    def available_numeric_series(self) -> set[str]:
        return {name for name in self._series if name != SUMMARY_SERIES}

    @property
    def has_summaries(self) -> bool:
        return SUMMARY_SERIES in self._series

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
            return self._fuse_ranked_hits(dense_by_query[0], lexical_hits, top_k), searched, empty
        hits_by_query, searched, empty = self.search_many(
            selected_series, np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), top_k
        )
        return hits_by_query[0], searched, empty

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
    ) -> list[RetrievalHit]:
        """Fuse dense and BM25 rankings with reciprocal rank fusion."""
        ranked: dict[tuple[str, str], dict[str, Any]] = {}
        for method, hits in (("dense", dense_hits), ("lexical", lexical_hits)):
            for rank, hit in enumerate(hits, start=1):
                identity = hit.metadata.get("chunk_id", hit.metadata.get("source_chunk_index"))
                key = (hit.series, str(identity))
                row = ranked.setdefault(
                    key,
                    {"hit": hit, "score": 0.0, "dense_rank": None, "lexical_rank": None},
                )
                row["score"] += 1.0 / (HYBRID_RRF_K + rank)
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
            )
            for row in ordered
        ]

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
        total_limit: int,
        chunks_per_heading: int,
        query_embedding: np.ndarray | None = None,
        embed_many: Callable[[list[str]], list[list[float]]] | None = None,
        embedding_batch_size: int = 128,
    ) -> tuple[list[RetrievalHit], list[CitationPath]]:
        """Breadth-first expansion through precise in-text clause citations.

        Expansion deliberately reads raw chunk files rather than the active
        embedding selection. A cited target can therefore be in another local
        3GPP series, while the semantic seed retrieval remains paper-subset
        compatible. Citations lacking a target heading are ignored because
        they cannot be resolved to a defensible target clause.
        """
        if (
            max_depth < 0
            or total_limit < len(seed_hits)
            or chunks_per_heading < 1
            or embedding_batch_size < 1
        ):
            raise ValueError("Invalid citation expansion limits")
        selected = list(seed_hits[:total_limit])
        selected_ids = {hit.metadata.get("chunk_id") for hit in selected if hit.metadata.get("chunk_id")}
        paths: list[CitationPath] = []
        frontier = list(selected)

        for depth in range(1, max_depth + 1):
            if len(selected) >= total_limit or not frontier:
                break
            if depth == 1:
                depth_one_hits, depth_one_paths = self._select_depth_one_citations(
                    frontier,
                    selected_ids,
                    total_limit - len(selected),
                    chunks_per_heading,
                    query_embedding,
                    embed_many,
                    embedding_batch_size,
                )
                selected.extend(depth_one_hits)
                selected_ids.update(hit.metadata["chunk_id"] for hit in depth_one_hits)
                paths.extend(depth_one_paths)
                frontier = depth_one_hits
                continue
            next_frontier: list[RetrievalHit] = []
            for parent in frontier:
                if len(selected) >= total_limit:
                    break
                parent_chunk_id = parent.metadata.get("chunk_id")
                if not isinstance(parent_chunk_id, str):
                    continue
                source_chunk = self._chunk_for_hit(parent)
                for reference in source_chunk.get("references", []):
                    if len(selected) >= total_limit:
                        break
                    if not isinstance(reference, dict):
                        continue
                    if reference.get("type") not in {"internal", "external"}:
                        continue
                    if not isinstance(reference.get("target_heading"), str):
                        continue
                    targets, status = self._resolve_target(reference)
                    added_ids: list[str] = []
                    for target, target_score in self._select_citation_targets(
                        targets,
                        query_embedding,
                        embed_many,
                        chunks_per_heading,
                    ):
                        chunk_id = target.get("chunk_id")
                        if not isinstance(chunk_id, str) or chunk_id in selected_ids:
                            continue
                        target_file = reference.get("target_chunk_file")
                        if not isinstance(target_file, str):
                            continue
                        metadata = {key: value for key, value in target.items() if key != "text"}
                        hit = RetrievalHit(
                            score=parent.score if target_score is None else target_score,
                            series=str(target.get("series", reference.get("target_series", "unknown"))),
                            text=target["text"],
                            metadata=metadata,
                            source_chunk_file=target_file,
                            origin="citation",
                            citation_depth=depth,
                            parent_chunk_id=parent_chunk_id,
                            citation=dict(reference),
                            retrieval_method="citation",
                        )
                        selected.append(hit)
                        next_frontier.append(hit)
                        selected_ids.add(chunk_id)
                        added_ids.append(chunk_id)
                        if len(selected) >= total_limit:
                            break
                    if targets and not added_ids and status in RESOLVED_CITATION_STATUSES:
                        status = "duplicate_target"
                    paths.append(
                        CitationPath(
                            parent_chunk_id=parent_chunk_id,
                            depth=depth,
                            reference=dict(reference),
                            status=status,
                            target_chunk_ids=added_ids,
                        )
                    )
            frontier = next_frontier
        return selected, paths

    def _select_depth_one_citations(
        self,
        seed_hits: list[RetrievalHit],
        selected_ids: set[str],
        available_slots: int,
        chunks_per_heading: int,
        query_embedding: np.ndarray | None,
        embed_many: Callable[[list[str]], list[list[float]]] | None,
        embedding_batch_size: int,
    ) -> tuple[list[RetrievalHit], list[CitationPath]]:
        """Choose depth-one citations globally by query relevance, not source order."""
        references: list[tuple[RetrievalHit, dict[str, Any], str, list[dict[str, Any]], str]] = []
        for parent in seed_hits:
            parent_chunk_id = parent.metadata.get("chunk_id")
            if not isinstance(parent_chunk_id, str):
                continue
            source_chunk = self._chunk_for_hit(parent)
            for reference in source_chunk.get("references", []):
                if not isinstance(reference, dict):
                    continue
                if reference.get("type") not in {"internal", "external"}:
                    continue
                if not isinstance(reference.get("target_heading"), str):
                    continue
                target_file = reference.get("target_chunk_file")
                if not isinstance(target_file, str):
                    continue
                targets, status = self._resolve_target(reference)
                references.append((parent, dict(reference), target_file, targets, status))

        scored_targets = self._score_targets(
            [
                target
                for _, _, _, targets, status in references
                if status in RESOLVED_CITATION_STATUSES
                for target in targets
            ],
            query_embedding,
            embed_many,
            embedding_batch_size,
        )
        candidate_groups: list[tuple[tuple[RetrievalHit, dict[str, Any], str, list[dict[str, Any]], str], list[_CitationCandidate]]] = []
        order = 0
        for reference_record in references:
            parent, reference, target_file, targets, status = reference_record
            candidates: list[_CitationCandidate] = []
            ranked_targets = sorted(
                targets,
                key=lambda target: (
                    -scored_targets.get(target.get("chunk_id"), parent.score),
                    target.get("chunk_index_in_heading", 0),
                ),
            )[:chunks_per_heading]
            for target in ranked_targets:
                chunk_id = target.get("chunk_id")
                if not isinstance(chunk_id, str) or chunk_id in selected_ids:
                    continue
                candidates.append(
                    _CitationCandidate(
                        parent=parent,
                        reference=reference,
                        target=target,
                        target_file=target_file,
                        score=scored_targets.get(chunk_id),
                        order=order,
                    )
                )
                order += 1
            candidate_groups.append((reference_record, candidates))

        best_candidates: dict[str, _CitationCandidate] = {}
        for _, candidates in candidate_groups:
            for candidate in candidates:
                chunk_id = candidate.target["chunk_id"]
                existing = best_candidates.get(chunk_id)
                candidate_score = candidate.score if candidate.score is not None else candidate.parent.score
                existing_score = existing.score if existing and existing.score is not None else (existing.parent.score if existing else None)
                if existing is None or candidate_score > existing_score or (
                    candidate_score == existing_score and candidate.order < existing.order
                ):
                    best_candidates[chunk_id] = candidate

        ranked_candidates = sorted(
            best_candidates.values(),
            key=lambda candidate: (
                -(candidate.score if candidate.score is not None else candidate.parent.score),
                candidate.order,
            ),
        )[:available_slots]
        selected_candidates = {candidate.target["chunk_id"]: candidate for candidate in ranked_candidates}
        hits = [
            RetrievalHit(
                score=candidate.parent.score if candidate.score is None else candidate.score,
                series=str(candidate.target.get("series", candidate.reference.get("target_series", "unknown"))),
                text=candidate.target["text"],
                metadata={key: value for key, value in candidate.target.items() if key != "text"},
                source_chunk_file=candidate.target_file,
                origin="citation",
                citation_depth=1,
                parent_chunk_id=candidate.parent.metadata["chunk_id"],
                citation=candidate.reference,
                retrieval_method="citation",
            )
            for candidate in ranked_candidates
        ]
        paths: list[CitationPath] = []
        for (parent, reference, _, targets, status), candidates in candidate_groups:
            added_ids = [
                candidate.target["chunk_id"]
                for candidate in candidates
                if selected_candidates.get(candidate.target["chunk_id"]) == candidate
            ]
            if status in RESOLVED_CITATION_STATUSES and not added_ids:
                status = "not_selected_by_query_score" if candidates else "duplicate_target"
            paths.append(
                CitationPath(
                    parent_chunk_id=parent.metadata["chunk_id"],
                    depth=1,
                    reference=reference,
                    status=status,
                    target_chunk_ids=added_ids,
                )
            )
        return hits, paths

    @staticmethod
    def _score_targets(
        targets: list[dict[str, Any]],
        query_embedding: np.ndarray | None,
        embed_many: Callable[[list[str]], list[list[float]]] | None,
        batch_size: int,
    ) -> dict[str, float]:
        if query_embedding is None or embed_many is None:
            return {}
        unique_targets = {
            target["chunk_id"]: target
            for target in targets
            if isinstance(target.get("chunk_id"), str) and isinstance(target.get("text"), str)
        }
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        scores: dict[str, float] = {}
        target_items = list(unique_targets.items())
        for start in range(0, len(target_items), batch_size):
            batch = target_items[start : start + batch_size]
            vectors = np.asarray(embed_many([target["text"] for _, target in batch]), dtype=np.float32)
            if vectors.shape != (len(batch), query.shape[0]):
                raise ValueError("Citation target embeddings do not match the query embedding shape.")
            scores.update({chunk_id: float(vector @ query) for (chunk_id, _), vector in zip(batch, vectors, strict=True)})
        return scores

    @staticmethod
    def _select_citation_targets(
        targets: list[dict[str, Any]],
        query_embedding: np.ndarray | None,
        embed_many: Callable[[list[str]], list[list[float]]] | None,
        limit: int,
    ) -> list[tuple[dict[str, Any], float | None]]:
        """Preserve small clauses, but rank oversized cited clauses against the query."""
        if len(targets) <= limit or query_embedding is None or embed_many is None:
            return [(target, None) for target in targets[:limit]]

        texts = [target["text"] for target in targets]
        vectors = np.asarray(embed_many(texts), dtype=np.float32)
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if vectors.shape != (len(targets), query.shape[0]):
            raise ValueError("Citation target embeddings do not match the query embedding shape.")
        scores = vectors @ query
        ranked_indices = sorted(range(len(targets)), key=lambda index: (-float(scores[index]), index))
        return [(targets[index], float(scores[index])) for index in ranked_indices[:limit]]

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
