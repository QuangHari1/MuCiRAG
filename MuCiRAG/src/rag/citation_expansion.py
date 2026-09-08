"""Citation-only expansion policies kept separate from corpus retrieval.

The expander knows how to score already-resolved citation targets.  The corpus
continues to own file access and reference resolution, which keeps selection
policy independent of the on-disk chunk layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .lexical import SqliteBm25Index
from .types import CitationPath, RetrievalHit


RESOLVED_STATUSES = frozenset({"resolved", "resolved_descendant"})
RRF_K = 60


@dataclass(frozen=True)
class _GainCandidate:
    parent: RetrievalHit
    reference: dict[str, Any]
    target: dict[str, Any]
    target_file: str
    vector: np.ndarray
    group_index: int
    order: int


@dataclass(frozen=True)
class _RankedCandidate:
    parent: RetrievalHit
    reference: dict[str, Any]
    target: dict[str, Any]
    target_file: str
    score: float
    semantic_score: float
    dense_rank: int | None
    lexical_rank: int | None
    group_index: int
    order: int


@dataclass
class _ReferenceGroup:
    parent: RetrievalHit
    reference: dict[str, Any]
    target_file: str | None
    targets: list[dict[str, Any]]
    status: str
    candidate_ids: list[str]
    initial_semantic_score: float | None = None
    initial_rrf_score: float | None = None
    initial_gain: float | None = None


class CitationExpander:
    """Select resolvable cited chunks without changing seed retrieval."""

    def __init__(
        self,
        resolve_target: Callable[[dict[str, Any]], tuple[list[dict[str, Any]], str]],
        load_parent_chunk: Callable[[RetrievalHit], dict[str, Any]],
        lexical_index: SqliteBm25Index | None,
        dense_weight: float,
        bm25_weight: float,
    ) -> None:
        self._resolve_target = resolve_target
        self._load_parent_chunk = load_parent_chunk
        self._lexical_index = lexical_index
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    def expand(
        self,
        seed_hits: list[RetrievalHit],
        *,
        max_depth: int,
        max_citation_chunks: int,
        chunks_per_heading: int,
        facets: list[str],
        facet_embeddings: np.ndarray,
        min_gain: float,
        embed_many: Callable[[list[str]], list[list[float]]] | None,
        embedding_batch_size: int,
        selection_strategy: str,
        query_embedding: np.ndarray | None,
        query_text: str | None,
    ) -> tuple[list[RetrievalHit], list[CitationPath]]:
        """Apply citation selection with ``max_citation_chunks`` available at every depth."""
        if (
            max_depth < 0
            or max_citation_chunks < 0
            or chunks_per_heading < 1
            or min_gain < 0
            or embedding_batch_size < 1
        ):
            raise ValueError("Invalid citation expansion limits")
        if selection_strategy not in {"gain", "semantic_bfs", "rrf_bfs"}:
            raise ValueError(f"Unsupported citation strategy: {selection_strategy}")
        if max_depth == 0 or max_citation_chunks == 0 or not seed_hits:
            return list(seed_hits), []
        if selection_strategy in {"semantic_bfs", "rrf_bfs"}:
            if query_embedding is None or embed_many is None:
                raise ValueError(f"{selection_strategy} citation selection requires query_embedding and embed_many.")
            if selection_strategy == "rrf_bfs" and (query_text is None or self._lexical_index is None):
                raise ValueError("rrf_bfs citation selection requires a hybrid retrieval backend and query_text.")
            return self._expand_ranked_bfs(
                seed_hits,
                max_depth=max_depth,
                max_citation_chunks=max_citation_chunks,
                chunks_per_heading=chunks_per_heading,
                query_embedding=query_embedding,
                embed_many=embed_many,
                embedding_batch_size=embedding_batch_size,
                selection_strategy=selection_strategy,
                query_text=query_text,
            )
        if embed_many is None:
            raise ValueError("Citation gain requires embed_many.")
        return self._expand_gain(
            seed_hits,
            max_depth=max_depth,
            max_citation_chunks=max_citation_chunks,
            chunks_per_heading=chunks_per_heading,
            facets=facets,
            facet_embeddings=facet_embeddings,
            min_gain=min_gain,
            embed_many=embed_many,
            embedding_batch_size=embedding_batch_size,
        )

    def _expand_gain(
        self,
        seed_hits: list[RetrievalHit],
        *,
        max_depth: int,
        max_citation_chunks: int,
        chunks_per_heading: int,
        facets: list[str],
        facet_embeddings: np.ndarray,
        min_gain: float,
        embed_many: Callable[[list[str]], list[list[float]]],
        embedding_batch_size: int,
    ) -> tuple[list[RetrievalHit], list[CitationPath]]:
        facet_matrix = self._normalize_rows(np.asarray(facet_embeddings, dtype=np.float32))
        if facet_matrix.ndim != 2 or len(facet_matrix) != len(facets) or not facets:
            raise ValueError("facets and facet_embeddings must be non-empty and aligned.")
        selected = list(seed_hits)
        selected_ids = {hit.metadata.get("chunk_id") for hit in selected if hit.metadata.get("chunk_id")}
        paths: list[CitationPath] = []
        frontier = list(selected)

        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            groups = self._reference_groups(frontier, selected_ids)
            targets = self._unselected_targets(groups, selected_ids)
            if not targets:
                paths.extend(self._gain_paths(groups, depth, {}, selected_ids, min_gain))
                break
            parent_by_id = {
                group.parent.metadata["chunk_id"]: group.parent
                for group in groups
                if group.status in RESOLVED_STATUSES and isinstance(group.parent.metadata.get("chunk_id"), str)
            }
            parent_items = list(parent_by_id.items())
            target_items = list(targets.items())
            vectors = self._embed_texts(
                [hit.text for _, hit in parent_items] + [target["text"] for _, target in target_items],
                embed_many,
                embedding_batch_size,
            )
            if vectors.shape[1] != facet_matrix.shape[1]:
                raise ValueError("Citation context embeddings do not match facet embeddings.")
            parent_vectors = self._normalize_rows(vectors[: len(parent_items)])
            target_vectors = self._normalize_rows(vectors[len(parent_items) :])
            coverage = {
                chunk_id: facet_matrix @ vector
                for (chunk_id, _), vector in zip(parent_items, parent_vectors, strict=True)
            }
            vectors_by_id = {
                chunk_id: vector for (chunk_id, _), vector in zip(target_items, target_vectors, strict=True)
            }
            candidates = self._gain_candidates(groups, coverage, facet_matrix, vectors_by_id, chunks_per_heading)
            added_by_group, next_frontier = self._select_gain_candidates(
                candidates,
                coverage,
                facet_matrix,
                facets,
                min_gain,
                max_citation_chunks,
                depth,
                selected,
                selected_ids,
            )
            paths.extend(self._gain_paths(groups, depth, added_by_group, selected_ids, min_gain))
            frontier = next_frontier
        return selected, paths

    @staticmethod
    def _unselected_targets(groups: list[_ReferenceGroup], selected_ids: set[Any]) -> dict[str, dict[str, Any]]:
        return {
            target["chunk_id"]: target
            for group in groups
            if group.status in RESOLVED_STATUSES
            for target in group.targets
            if isinstance(target.get("chunk_id"), str)
            and isinstance(target.get("text"), str)
            and target["chunk_id"] not in selected_ids
        }

    def _gain_candidates(
        self,
        groups: list[_ReferenceGroup],
        coverage: dict[str, np.ndarray],
        facets: np.ndarray,
        vectors_by_id: dict[str, np.ndarray],
        chunks_per_heading: int,
    ) -> list[_GainCandidate]:
        candidates: list[_GainCandidate] = []
        order = 0
        for group_index, group in enumerate(groups):
            if group.status not in RESOLVED_STATUSES or group.target_file is None:
                continue
            parent_id = group.parent.metadata["chunk_id"]
            ranked = []
            for target in group.targets:
                chunk_id = target.get("chunk_id")
                vector = vectors_by_id.get(chunk_id) if isinstance(chunk_id, str) else None
                if vector is None:
                    continue
                gain = np.maximum(0.0, facets @ vector - coverage[parent_id])
                ranked.append((float(np.max(gain)), int(target.get("chunk_index_in_heading", 0)), target, vector))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            group.initial_gain = max((item[0] for item in ranked), default=None)
            for _, _, target, vector in ranked[:chunks_per_heading]:
                chunk_id = target["chunk_id"]
                group.candidate_ids.append(chunk_id)
                candidates.append(_GainCandidate(group.parent, group.reference, target, group.target_file, vector, group_index, order))
                order += 1
        unique: dict[tuple[str, str], _GainCandidate] = {}
        for candidate in candidates:
            unique.setdefault((candidate.parent.metadata["chunk_id"], candidate.target["chunk_id"]), candidate)
        return list(unique.values())

    @staticmethod
    def _select_gain_candidates(
        candidates: list[_GainCandidate],
        coverage: dict[str, np.ndarray],
        facets: np.ndarray,
        facet_names: list[str],
        min_gain: float,
        available_slots: int,
        depth: int,
        selected: list[RetrievalHit],
        selected_ids: set[Any],
    ) -> tuple[dict[int, list[str]], list[RetrievalHit]]:
        added_by_group: dict[int, list[str]] = {}
        next_frontier: list[RetrievalHit] = []
        remaining = list(candidates)
        while remaining and len(next_frontier) < available_slots:
            ranked = []
            for candidate in remaining:
                parent_id = candidate.parent.metadata["chunk_id"]
                gains = np.maximum(0.0, facets @ candidate.vector - coverage[parent_id])
                ranked.append((float(np.max(gains)), float(np.sum(gains)), candidate.order, candidate, gains))
            ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
            max_gain, _, _, candidate, gains = ranked[0]
            if max_gain <= 0.0 or max_gain + 1e-12 < min_gain:
                break
            parent_id = candidate.parent.metadata["chunk_id"]
            before = coverage[parent_id].copy()
            after = np.maximum(before, facets @ candidate.vector)
            target = candidate.target
            hit = RetrievalHit(
                score=max_gain,
                series=str(target.get("series", candidate.reference.get("target_series", "unknown"))),
                text=target["text"],
                metadata={key: value for key, value in target.items() if key != "text"},
                source_chunk_file=candidate.target_file,
                origin="citation",
                citation_depth=depth,
                parent_chunk_id=parent_id,
                citation=candidate.reference,
                retrieval_method="citation_gain",
                citation_gain=max_gain,
                citation_total_gain=float(np.sum(gains)),
                citation_facets=tuple(name for name, gain in zip(facet_names, gains, strict=True) if float(gain) + 1e-12 >= min_gain),
                citation_facet_gains={name: float(gain) for name, gain in zip(facet_names, gains, strict=True)},
                citation_coverage_before={name: float(value) for name, value in zip(facet_names, before, strict=True)},
                citation_coverage_after={name: float(value) for name, value in zip(facet_names, after, strict=True)},
            )
            next_frontier.append(hit)
            selected.append(hit)
            selected_ids.add(target["chunk_id"])
            added_by_group.setdefault(candidate.group_index, []).append(target["chunk_id"])
            coverage[parent_id] = after
            remaining = [item for item in remaining if item.target["chunk_id"] != target["chunk_id"]]
        return added_by_group, next_frontier

    def _expand_ranked_bfs(
        self,
        seed_hits: list[RetrievalHit],
        *,
        max_depth: int,
        max_citation_chunks: int,
        chunks_per_heading: int,
        query_embedding: np.ndarray,
        embed_many: Callable[[list[str]], list[list[float]]],
        embedding_batch_size: int,
        selection_strategy: str,
        query_text: str | None,
    ) -> tuple[list[RetrievalHit], list[CitationPath]]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        selected = list(seed_hits)
        selected_ids = {hit.metadata.get("chunk_id") for hit in selected if hit.metadata.get("chunk_id")}
        paths: list[CitationPath] = []
        frontier = list(selected)
        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            groups = self._reference_groups(frontier, selected_ids)
            targets = self._unselected_targets(groups, selected_ids)
            semantic_scores = self._score_semantic_targets(list(targets.values()), query, embed_many, embedding_batch_size)
            if selection_strategy == "rrf_bfs":
                scores, dense_ranks, lexical_ranks = self._score_rrf_targets(semantic_scores, query_text or "")
            else:
                scores, dense_ranks, lexical_ranks = semantic_scores, {}, {}
            candidates = self._ranked_candidates(
                groups,
                scores,
                semantic_scores,
                dense_ranks,
                lexical_ranks,
                chunks_per_heading,
                selection_strategy,
            )
            best_by_target: dict[str, _RankedCandidate] = {}
            for candidate in candidates:
                current = best_by_target.get(candidate.target["chunk_id"])
                if current is None or candidate.score > current.score or (candidate.score == current.score and candidate.order < current.order):
                    best_by_target[candidate.target["chunk_id"]] = candidate
            chosen = sorted(best_by_target.values(), key=lambda item: (-item.score, item.order))[
                :max_citation_chunks
            ]
            added_by_group: dict[int, list[str]] = {}
            next_frontier: list[RetrievalHit] = []
            for candidate in chosen:
                target = candidate.target
                chunk_id = target["chunk_id"]
                hit = RetrievalHit(
                    score=candidate.score,
                    series=str(target.get("series", candidate.reference.get("target_series", "unknown"))),
                    text=target["text"],
                    metadata={key: value for key, value in target.items() if key != "text"},
                    source_chunk_file=candidate.target_file,
                    origin="citation",
                    citation_depth=depth,
                    parent_chunk_id=candidate.parent.metadata["chunk_id"],
                    citation=candidate.reference,
                    retrieval_method=f"citation_{selection_strategy}",
                    dense_rank=candidate.dense_rank,
                    lexical_rank=candidate.lexical_rank,
                    citation_semantic_score=candidate.semantic_score,
                    citation_rrf_score=candidate.score if selection_strategy == "rrf_bfs" else None,
                )
                selected.append(hit)
                next_frontier.append(hit)
                selected_ids.add(chunk_id)
                added_by_group.setdefault(candidate.group_index, []).append(chunk_id)
            paths.extend(self._ranked_paths(groups, depth, added_by_group, selected_ids, selection_strategy))
            frontier = next_frontier
        return selected, paths

    def _ranked_candidates(
        self,
        groups: list[_ReferenceGroup],
        scores: dict[str, float],
        semantic_scores: dict[str, float],
        dense_ranks: dict[str, int],
        lexical_ranks: dict[str, int],
        chunks_per_heading: int,
        selection_strategy: str,
    ) -> list[_RankedCandidate]:
        candidates: list[_RankedCandidate] = []
        order = 0
        for group_index, group in enumerate(groups):
            if group.status not in RESOLVED_STATUSES or group.target_file is None:
                continue
            ranked = sorted(group.targets, key=lambda target: (-scores.get(target.get("chunk_id"), float("-inf")), int(target.get("chunk_index_in_heading", 0))))
            group.initial_semantic_score = max((semantic_scores.get(target.get("chunk_id"), float("-inf")) for target in ranked), default=None)
            if selection_strategy == "rrf_bfs":
                group.initial_rrf_score = max(
                    (scores.get(target.get("chunk_id"), float("-inf")) for target in ranked),
                    default=None,
                )
            for target in ranked[:chunks_per_heading]:
                chunk_id = target.get("chunk_id")
                if not isinstance(chunk_id, str) or chunk_id not in scores:
                    continue
                group.candidate_ids.append(chunk_id)
                candidates.append(_RankedCandidate(group.parent, group.reference, target, group.target_file, scores[chunk_id], semantic_scores[chunk_id], dense_ranks.get(chunk_id), lexical_ranks.get(chunk_id), group_index, order))
                order += 1
        return candidates

    def _reference_groups(self, parents: list[RetrievalHit], selected_ids: set[Any]) -> list[_ReferenceGroup]:
        groups: list[_ReferenceGroup] = []
        for parent in parents:
            if not isinstance(parent.metadata.get("chunk_id"), str):
                continue
            for reference in self._load_parent_chunk(parent).get("references", []):
                if not isinstance(reference, dict) or reference.get("type") not in {"internal", "external"}:
                    continue
                if not isinstance(reference.get("target_heading"), str):
                    continue
                targets, status = self._resolve_target(reference)
                target_file = reference.get("target_chunk_file")
                groups.append(_ReferenceGroup(parent, dict(reference), target_file if isinstance(target_file, str) else None, [target for target in targets if isinstance(target.get("chunk_id"), str) and target["chunk_id"] not in selected_ids], status, []))
        return groups

    def _score_semantic_targets(self, targets: list[dict[str, Any]], query: np.ndarray, embed_many: Callable[[list[str]], list[list[float]]], batch_size: int) -> dict[str, float]:
        unique = {target["chunk_id"]: target for target in targets if isinstance(target.get("chunk_id"), str) and isinstance(target.get("text"), str)}
        if not unique:
            return {}
        items = list(unique.items())
        vectors = self._embed_texts([target["text"] for _, target in items], embed_many, batch_size)
        if vectors.shape[1] != query.shape[0]:
            raise ValueError("Citation target embeddings do not match the query embedding shape.")
        return {chunk_id: float(vector @ query) for (chunk_id, _), vector in zip(items, vectors, strict=True)}

    def _score_rrf_targets(self, semantic_scores: dict[str, float], query_text: str) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
        if self._lexical_index is None:  # pragma: no cover - caller validates this
            raise RuntimeError("rrf_bfs requires a lexical index.")
        dense_order = sorted(semantic_scores, key=lambda item: (-semantic_scores[item], item))
        dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_order, start=1)}
        lexical_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(self._lexical_index.search_chunk_ids(query_text, dense_order), start=1)}
        scores = {chunk_id: self._dense_weight / (RRF_K + dense_rank) + (self._bm25_weight / (RRF_K + lexical_ranks[chunk_id]) if chunk_id in lexical_ranks else 0.0) for chunk_id, dense_rank in dense_ranks.items()}
        return scores, dense_ranks, lexical_ranks

    @staticmethod
    def _embed_texts(texts: list[str], embed_many: Callable[[list[str]], list[list[float]]], batch_size: int) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = np.asarray(embed_many(batch), dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(batch):
                raise ValueError("Embedding provider returned an invalid citation matrix.")
            batches.append(vectors)
        return np.concatenate(batches, axis=0)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("Expected a non-empty two-dimensional embedding matrix.")
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _gain_paths(groups: list[_ReferenceGroup], depth: int, added: dict[int, list[str]], selected_ids: set[Any], min_gain: float) -> list[CitationPath]:
        paths = []
        for index, group in enumerate(groups):
            target_ids = added.get(index, [])
            status = group.status
            if status in RESOLVED_STATUSES and not target_ids:
                if not group.targets or (group.candidate_ids and all(item in selected_ids for item in group.candidate_ids)):
                    status = "duplicate_target"
                elif group.initial_gain is None or group.initial_gain + 1e-12 < min_gain:
                    status = "below_min_gain"
                else:
                    status = "not_selected_by_gain"
            paths.append(CitationPath(group.parent.metadata["chunk_id"], depth, group.reference, status, target_ids, list(group.candidate_ids), best_candidate_gain=group.initial_gain))
        return paths

    @staticmethod
    def _ranked_paths(groups: list[_ReferenceGroup], depth: int, added: dict[int, list[str]], selected_ids: set[Any], strategy: str) -> list[CitationPath]:
        paths = []
        for index, group in enumerate(groups):
            target_ids = added.get(index, [])
            status = group.status
            if status in RESOLVED_STATUSES and not target_ids:
                if not group.targets or (group.candidate_ids and all(item in selected_ids for item in group.candidate_ids)):
                    status = "duplicate_target"
                else:
                    status = "not_selected_by_semantic_score" if strategy == "semantic_bfs" else "not_selected_by_rrf_score"
            paths.append(CitationPath(group.parent.metadata["chunk_id"], depth, group.reference, status, target_ids, list(group.candidate_ids), best_candidate_semantic_score=group.initial_semantic_score, best_candidate_rrf_score=group.initial_rrf_score))
        return paths
