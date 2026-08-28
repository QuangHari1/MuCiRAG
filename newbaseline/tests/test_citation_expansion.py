from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newbaseline.src.rag.corpus import PaperEmbeddingCorpus
from newbaseline.src.rag.lexical import build_bm25_index
from newbaseline.src.rag.types import RetrievalHit


def chunk(
    chunk_id: str,
    document_id: str,
    heading: str,
    references: list[dict] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "series": document_id[:2],
        "heading": heading,
        "chunk_index_in_heading": int(chunk_id[-1]) if chunk_id[-1].isdigit() else 0,
        "references": references or [],
        "text": f"text for {chunk_id}",
    }


def build_corpus(
    root: Path,
    chunks: list[dict],
    *,
    retrieval_backend: str = "semantic",
    rrf_dense_weight: float = 0.5,
    rrf_bm25_weight: float = 0.5,
) -> PaperEmbeddingCorpus:
    chunk_root = root / "chunks"
    chunk_root.mkdir()
    by_series: dict[str, list[dict]] = {}
    for item in chunks:
        by_series.setdefault(item["series"], []).append(item)
    for series, rows in by_series.items():
        (chunk_root / f"ChunkSeries{series}.json").write_text(
            json.dumps({"chunks": rows}), encoding="utf-8"
        )
    (root / "manifest.json").write_text(
        json.dumps({"source_chunk_directory": str(chunk_root), "series": {"21": {}}}),
        encoding="utf-8",
    )
    if retrieval_backend == "hybrid":
        build_bm25_index(
            root / "lexical.sqlite3",
            [
                (item["chunk_id"], item["series"], index, item["heading"], item["text"])
                for index, item in enumerate(chunks)
            ],
        )
    return PaperEmbeddingCorpus(
        root,
        retrieval_backend=retrieval_backend,
        rrf_dense_weight=rrf_dense_weight,
        rrf_bm25_weight=rrf_bm25_weight,
    )


def expand(
    corpus: PaperEmbeddingCorpus,
    seed: dict,
    vectors: dict[str, list[float]],
    facets: list[str],
    facet_embeddings: list[list[float]],
    *,
    max_depth: int = 1,
    max_citation_chunks: int = 4,
    chunks_per_heading: int = 2,
    min_gain: float = 0.01,
):
    return corpus.expand_citations(
        [
            RetrievalHit(
                0.9,
                seed["series"],
                seed["text"],
                {"chunk_id": seed["chunk_id"]},
                f"ChunkSeries{seed['series']}.json",
            )
        ],
        max_depth=max_depth,
        max_citation_chunks=max_citation_chunks,
        chunks_per_heading=chunks_per_heading,
        facets=facets,
        facet_embeddings=np.asarray(facet_embeddings, dtype=np.float32),
        min_gain=min_gain,
        embed_many=lambda texts: [vectors[text] for text in texts],
    )


class CitationExpansionTests(unittest.TestCase):
    def test_expands_precise_cross_series_citations_breadth_first(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "external",
                    "document_id": "22001",
                    "target_heading": "2",
                    "target_series": "22",
                    "target_chunk_file": "ChunkSeries22.json",
                },
                {
                    "type": "external",
                    "document_id": "22999",
                    "target_heading": "9",
                    "target_chunk_file": "ChunkSeries22.json",
                },
            ],
        )
        target_one = chunk(
            "target0",
            "22001",
            "2",
            [
                {
                    "type": "external",
                    "document_id": "23001",
                    "target_heading": "3",
                    "target_series": "23",
                    "target_chunk_file": "ChunkSeries23.json",
                }
            ],
        )
        target_two = chunk("target1", "22001", "2")
        depth_two = [chunk("depth2a", "23001", "3"), chunk("depth2b", "23001", "3")]
        vectors = {
            item["text"]: [float(index == position) for index in range(5)]
            for position, item in enumerate([seed, target_one, target_two, *depth_two])
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [seed, target_one, target_two, *depth_two])
            expanded, paths = expand(
                corpus,
                seed,
                vectors,
                [f"facet {index}" for index in range(5)],
                np.eye(5).tolist(),
                max_depth=2,
            )

        self.assertEqual(
            [hit.metadata["chunk_id"] for hit in expanded],
            ["seed0", "target0", "target1", "depth2a", "depth2b"],
        )
        self.assertEqual([hit.citation_depth for hit in expanded], [0, 1, 1, 2, 2])
        self.assertTrue(all(hit.retrieval_method == "citation_gain" for hit in expanded[1:]))
        self.assertIn("target_heading_not_found", [path.status for path in paths])

    def test_max_citation_chunks_is_a_hard_ceiling(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "2",
                    "target_chunk_file": "ChunkSeries21.json",
                }
            ],
        )
        targets = [chunk("target0", "21001", "2"), chunk("target1", "21001", "2")]
        vectors = {
            seed["text"]: [1.0, 0.0, 0.0],
            targets[0]["text"]: [0.0, 1.0, 0.0],
            targets[1]["text"]: [0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [seed, *targets])
            expanded, _ = expand(
                corpus,
                seed,
                vectors,
                ["covered", "need one", "need two"],
                np.eye(3).tolist(),
                max_citation_chunks=1,
            )

        self.assertEqual(len(expanded), 2)

    def test_parent_heading_descendants_are_ranked_by_marginal_gain(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "2",
                    "target_chunk_file": "ChunkSeries21.json",
                }
            ],
        )
        children = [
            chunk("child0", "21001", "2.1"),
            chunk("child1", "21001", "2.2"),
            chunk("child2", "21001", "2.3"),
        ]
        vectors = {
            seed["text"]: [1.0, 0.0],
            children[0]["text"]: [0.9, 0.1],
            children[1]["text"]: [0.5, 0.5],
            children[2]["text"]: [0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [seed, *children])
            expanded, paths = expand(
                corpus,
                seed,
                vectors,
                ["missing property"],
                [[0.0, 1.0]],
                max_citation_chunks=1,
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "child2"])
        self.assertEqual(paths[0].status, "resolved_descendant")
        self.assertAlmostEqual(expanded[1].citation_gain or 0.0, 1.0)

    def test_stops_when_best_gain_is_below_threshold(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "2",
                    "target_chunk_file": "ChunkSeries21.json",
                }
            ],
        )
        target = chunk("target0", "21001", "2")
        vectors = {seed["text"]: [1.0, 0.0], target["text"]: [1.0, 0.005]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [seed, target])
            expanded, paths = expand(
                corpus,
                seed,
                vectors,
                ["missing property"],
                [[0.0, 1.0]],
                min_gain=0.01,
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0"])
        self.assertEqual(paths[0].status, "below_min_gain")

    def test_greedy_coverage_does_not_add_redundant_targets(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "2",
                    "target_chunk_file": "ChunkSeries21.json",
                },
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "3",
                    "target_chunk_file": "ChunkSeries21.json",
                },
            ],
        )
        first = chunk("first0", "21001", "2")
        duplicate = chunk("duplicate0", "21001", "3")
        vectors = {
            seed["text"]: [1.0, 0.0],
            first["text"]: [0.0, 1.0],
            duplicate["text"]: [0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [seed, first, duplicate])
            expanded, paths = expand(
                corpus,
                seed,
                vectors,
                ["missing property"],
                [[0.0, 1.0]],
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "first0"])
        self.assertEqual(paths[1].status, "not_selected_by_gain")
        self.assertEqual(expanded[1].citation_facets, ("missing property",))
        self.assertEqual(expanded[1].citation_facet_gains, {"missing property": 1.0})
        self.assertEqual(expanded[1].citation_coverage_before, {"missing property": 0.0})
        self.assertEqual(expanded[1].citation_coverage_after, {"missing property": 1.0})
        self.assertAlmostEqual(expanded[1].citation_total_gain or 0.0, 1.0)
        self.assertEqual(paths[1].candidate_chunk_ids, ["duplicate0"])
        self.assertAlmostEqual(paths[1].best_candidate_gain or 0.0, 1.0)

    def test_unrelated_seed_does_not_suppress_parent_local_gain(self) -> None:
        reference = {
            "type": "internal",
            "document_id": "21001",
            "target_heading": "3",
            "target_chunk_file": "ChunkSeries21.json",
        }
        parent = chunk("parent0", "21001", "1", [reference])
        unrelated = chunk("other0", "21001", "2")
        target = chunk("target0", "21001", "3")
        vectors = {
            parent["text"]: [0.0, 1.0],
            unrelated["text"]: [1.0, 0.0],
            target["text"]: [1.0, 0.0],
        }
        seeds = [
            RetrievalHit(
                0.9,
                item["series"],
                item["text"],
                {"chunk_id": item["chunk_id"]},
                "ChunkSeries21.json",
            )
            for item in (parent, unrelated)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [parent, unrelated, target])
            expanded, _ = corpus.expand_citations(
                seeds,
                max_depth=1,
                max_citation_chunks=4,
                chunks_per_heading=1,
                facets=["target information"],
                facet_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                min_gain=0.01,
                embed_many=lambda texts: [vectors[text] for text in texts],
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["parent0", "other0", "target0"])
        self.assertEqual(expanded[-1].parent_chunk_id, "parent0")
        self.assertAlmostEqual(expanded[-1].citation_gain or 0.0, 1.0)

    def test_duplicate_target_keeps_the_best_parent_edge(self) -> None:
        reference = {
            "type": "internal",
            "document_id": "21001",
            "target_heading": "3",
            "target_chunk_file": "ChunkSeries21.json",
        }
        covered_parent = chunk("parent0", "21001", "1", [reference])
        missing_parent = chunk("parent1", "21001", "2", [reference])
        target = chunk("target0", "21001", "3")
        vectors = {
            covered_parent["text"]: [1.0, 0.0],
            missing_parent["text"]: [0.0, 1.0],
            target["text"]: [1.0, 0.0],
        }
        seeds = [
            RetrievalHit(
                0.9,
                item["series"],
                item["text"],
                {"chunk_id": item["chunk_id"]},
                "ChunkSeries21.json",
            )
            for item in (covered_parent, missing_parent)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [covered_parent, missing_parent, target])
            expanded, paths = corpus.expand_citations(
                seeds,
                max_depth=1,
                max_citation_chunks=4,
                chunks_per_heading=1,
                facets=["target information"],
                facet_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                min_gain=0.01,
                embed_many=lambda texts: [vectors[text] for text in texts],
            )

        self.assertEqual(expanded[-1].metadata["chunk_id"], "target0")
        self.assertEqual(expanded[-1].parent_chunk_id, "parent1")
        self.assertEqual(paths[0].status, "duplicate_target")
        self.assertEqual(paths[1].target_chunk_ids, ["target0"])

    def test_semantic_bfs_ranks_cited_targets_by_query_similarity(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "2",
                    "target_chunk_file": "ChunkSeries21.json",
                },
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": "3",
                    "target_chunk_file": "ChunkSeries21.json",
                },
            ],
        )
        low_score = chunk("low0", "21001", "2")
        high_score = chunk("high0", "21001", "3")
        vectors = {
            low_score["text"]: [0.2, 0.8],
            high_score["text"]: [0.9, 0.1],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(Path(temporary_directory), [seed, low_score, high_score])
            expanded, paths = corpus.expand_citations(
                [
                    RetrievalHit(
                        0.9,
                        seed["series"],
                        seed["text"],
                        {"chunk_id": seed["chunk_id"]},
                        "ChunkSeries21.json",
                    )
                ],
                max_depth=1,
                max_citation_chunks=1,
                chunks_per_heading=1,
                facets=[],
                facet_embeddings=np.empty((0, 2), dtype=np.float32),
                min_gain=0.99,
                embed_many=lambda texts: [vectors[text] for text in texts],
                selection_strategy="semantic_bfs",
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "high0"])
        self.assertEqual(expanded[-1].retrieval_method, "citation_semantic_bfs")
        self.assertAlmostEqual(expanded[-1].citation_semantic_score or 0.0, 0.9)
        self.assertIsNone(expanded[-1].citation_gain)
        self.assertEqual(paths[0].status, "not_selected_by_semantic_score")
        self.assertEqual(paths[1].target_chunk_ids, ["high0"])
        self.assertAlmostEqual(paths[1].best_candidate_semantic_score or 0.0, 0.9)

    def test_rrf_bfs_fuses_dense_and_bm25_ranks_within_citation_targets(self) -> None:
        seed = chunk(
            "seed0",
            "21001",
            "1",
            [
                {
                    "type": "internal",
                    "document_id": "21001",
                    "target_heading": heading,
                    "target_chunk_file": "ChunkSeries21.json",
                }
                for heading in ("2", "3", "4")
            ],
        )
        semantic = chunk("semantic0", "21001", "2")
        middle = chunk("middle0", "21001", "3")
        lexical = chunk("lexical0", "21001", "4")
        lexical["text"] = "keyword-specific cited clause"
        vectors = {
            semantic["text"]: [0.9, 0.1],
            middle["text"]: [0.5, 0.5],
            lexical["text"]: [0.1, 0.9],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            corpus = build_corpus(
                Path(temporary_directory),
                [seed, semantic, middle, lexical],
                retrieval_backend="hybrid",
                rrf_dense_weight=0.7,
                rrf_bm25_weight=0.3,
            )
            expanded, paths = corpus.expand_citations(
                [
                    RetrievalHit(
                        0.9,
                        seed["series"],
                        seed["text"],
                        {"chunk_id": seed["chunk_id"]},
                        "ChunkSeries21.json",
                    )
                ],
                max_depth=1,
                max_citation_chunks=1,
                chunks_per_heading=1,
                facets=[],
                facet_embeddings=np.empty((0, 2), dtype=np.float32),
                min_gain=0.99,
                embed_many=lambda texts: [vectors[text] for text in texts],
                selection_strategy="rrf_bfs",
                query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                query_text="keyword",
            )

        self.assertEqual([hit.metadata["chunk_id"] for hit in expanded], ["seed0", "lexical0"])
        self.assertEqual(expanded[-1].retrieval_method, "citation_rrf_bfs")
        self.assertEqual(expanded[-1].dense_rank, 3)
        self.assertEqual(expanded[-1].lexical_rank, 1)
        self.assertAlmostEqual(expanded[-1].citation_semantic_score or 0.0, 0.1)
        self.assertIsNotNone(expanded[-1].citation_rrf_score)
        self.assertAlmostEqual(
            expanded[-1].citation_rrf_score or 0.0,
            0.7 / (60 + 3) + 0.3 / (60 + 1),
        )
        self.assertEqual(paths[0].status, "not_selected_by_rrf_score")
        self.assertAlmostEqual(paths[2].best_candidate_rrf_score or 0.0, expanded[-1].score)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
