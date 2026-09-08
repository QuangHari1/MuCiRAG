"""MuCiRAG orchestration: query preparation, retrieval, citations, and answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..settings import Settings, load_settings
from ..embeddings import EmbeddingProvider, create_embedding_provider
from .anchor_hierarchy import AnchorHierarchy, sha256_file
from .clients import OpenAICompatibleRagClient, RagClient
from .corpus import EmbeddingCorpus
from .router import PaperNNRouter, SemanticSeriesRouter
from .types import RagResult, RetrievalHit
from .vocabulary import Vocabulary


class MuCiRAGService:
    """Coordinate query preparation, seed retrieval, citations, and answering.

    Retrieval algorithms live in ``corpus`` and ``citation_expansion``; this
    class makes their execution order and provider calls explicit.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: RagClient | None = None,
        router: PaperNNRouter | SemanticSeriesRouter | None = None,
        corpus: EmbeddingCorpus | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        resources = self.settings.workspace_root / self.settings.get("rag", "resources_dir")
        embedding_root = self.settings.embedding_root(self.settings.get("rag", "selection_id"))
        embedding_provider = create_embedding_provider(self.settings) if client is None else None
        self.client = client or OpenAICompatibleRagClient(
            embedding_provider=embedding_provider,
            rephrase_model=self.settings.get("rag", "rephrase_model"),
            answer_model=self.settings.get("rag", "answer_model"),
            llm_provider=self.settings.get("llm", "provider"),
            api_key_env=self.settings.get("llm", "api_key_env"),
            base_url=self.settings.get("llm", "base_url"),
            thinking_mode=self.settings.get("llm", "thinking_mode"),
            temperature=self.settings.get("llm", "temperature"),
        )
        self.anchor_strategy = self.settings.get("rag", "anchor_strategy")
        if self.anchor_strategy not in {"router", "hierarchical"}:
            raise ValueError(f"Unsupported [rag].anchor_strategy: {self.anchor_strategy}")
        self.router = self._create_router(resources, embedding_provider, router)
        self.corpus = corpus or EmbeddingCorpus(
            embedding_root,
            self.settings.workspace_root,
            retrieval_backend=self.settings.get("rag", "retrieval_backend"),
            lexical_index_file=self.settings.get("rag", "lexical_index_file"),
            rrf_dense_weight=self.settings.get("rag", "rrf_dense_weight"),
            rrf_bm25_weight=self.settings.get("rag", "rrf_bm25_weight"),
        )
        self.anchor_hierarchy = self._load_anchor_hierarchy()
        self.vocabulary = vocabulary if vocabulary is not None else self._load_vocabulary(resources)

    def _create_router(
        self,
        resources: Path,
        embedding_provider: EmbeddingProvider | None,
        router: PaperNNRouter | SemanticSeriesRouter | None,
    ) -> PaperNNRouter | SemanticSeriesRouter | None:
        """Hierarchical retrieval bypasses routing; ablations can inject a router."""
        if self.anchor_strategy == "hierarchical":
            return None
        elif router is not None:
            return router
        elif self.settings.get("rag", "router_backend") == "paper_nn":
            if (
                self.settings.get("embedding", "backend") != "openai"
                or self.settings.get("embedding", "model") != "text-embedding-3-large"
                or self.settings.get("embedding", "dimensions") != 1024
            ):
                raise RuntimeError(
                    "rag.router_backend='paper_nn' requires OpenAI text-embedding-3-large at 1024 dimensions. "
                    "Use router_backend='semantic' for another embedding model."
                )
            return PaperNNRouter(
                resources / self.settings.get("rag", "router_checkpoint"),
                resources / self.settings.get("rag", "series_descriptions"),
                self.settings.get("rag", "router_similarity_scale"),
            )
        elif self.settings.get("rag", "router_backend") == "semantic":
            if embedding_provider is None:
                raise ValueError("Inject a router when constructing MuCiRAGService with a custom client.")
            return SemanticSeriesRouter(
                resources / self.settings.get("rag", "series_descriptions"), embedding_provider.embed
            )
        else:
            raise ValueError(f"Unsupported [rag].router_backend: {self.settings.get('rag', 'router_backend')}")

    def _load_anchor_hierarchy(self) -> AnchorHierarchy | None:
        """Load frozen anchors only when hierarchical retrieval needs them."""
        if self.anchor_strategy == "hierarchical":
            anchor_root_value = Path(self.settings.get("rag", "anchor_hierarchy_embedding_root"))
            anchor_root = (
                anchor_root_value
                if anchor_root_value.is_absolute()
                else self.settings.workspace_root / anchor_root_value
            )
            return AnchorHierarchy.load(
                anchor_root,
                manifest_file=self.settings.get("rag", "anchor_hierarchy_manifest_file"),
                vectors_file=self.settings.get("rag", "anchor_hierarchy_vectors_file"),
                corpus_manifest_sha256=sha256_file(anchor_root / "manifest.json"),
                embedding_backend=self.settings.get("embedding", "backend"),
                embedding_model=self.settings.get("embedding", "model"),
                dimensions=self.settings.get("embedding", "dimensions"),
            )
        return None

    def _load_vocabulary(self, resources: Path) -> Vocabulary:
        """Select the configured terminology assets independently of retrieval."""
        vocabulary_mode = self.settings.get("vocabulary", "mode")
        if vocabulary_mode == "paper_legacy":
            return Vocabulary.from_docx(resources / self.settings.get("rag", "vocabulary"))
        elif vocabulary_mode == "release18_unambiguous":
            return Vocabulary.from_release18_assets(
                resources / self.settings.get("vocabulary", "definitions_file"),
                resources / self.settings.get("vocabulary", "abbreviations_file"),
            )
        else:
            raise ValueError(f"Unsupported [vocabulary].mode: {vocabulary_mode}")

    def run(
        self,
        question: str,
        include_answer: bool = True,
        answer_prompt: str | None = None,
        strict_multiple_choice: bool = False,
    ) -> RagResult:
        """Retrieve for ``question`` and optionally answer a richer prompt.

        TeleQnA uses the plain question for routing/retrieval while supplying
        its answer choices only to the final answer model.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")
        citation_strategy = self.settings.get("rag", "citation_strategy")
        if citation_strategy not in {"gain", "semantic_bfs", "rrf_bfs"}:
            raise ValueError(f"Unsupported citation strategy: {citation_strategy}")
        rephrased = self.client.rephrase(question)
        enriched = self.vocabulary.enrich(rephrased)
        query_embedding = np.asarray(self.client.embed(enriched), dtype=np.float32)
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding has an invalid shape.")
        seed_retrievals, selected, searched, empty = self._retrieve_seeds(
            question, query_embedding
        )
        facets, facet_embeddings = self._prepare_facets(question, query_embedding, citation_strategy)
        citation_min_gain = self.settings.get("rag", "citation_min_gain")
        citation_max_chunks = self.settings.get("rag", "citation_max_chunks")
        retrievals, citation_paths = self.corpus.expand_citations(
            seed_retrievals,
            max_depth=self.settings.get("rag", "citation_max_depth"),
            max_citation_chunks=citation_max_chunks,
            chunks_per_heading=self.settings.get("rag", "citation_chunks_per_heading"),
            facets=facets,
            facet_embeddings=facet_embeddings,
            min_gain=citation_min_gain,
            embed_many=self.client.embed_many,
            selection_strategy=citation_strategy,
            query_embedding=query_embedding,
            query_text=enriched,
        )
        answer = None
        if include_answer:
            contexts = self._format_contexts(retrievals)
            answer = self.client.answer(
                answer_prompt or question,
                contexts,
                strict_multiple_choice=strict_multiple_choice,
            )
        return RagResult(
            question=question,
            rephrased_query=rephrased,
            query_facets=facets,
            enriched_query=enriched,
            router_selected_series=selected,
            empty_selected_series=empty,
            searched_series=searched,
            citation_min_gain=citation_min_gain,
            citation_max_chunks=citation_max_chunks,
            retrievals=retrievals,
            answer=answer,
            citation_paths=citation_paths,
            citation_strategy=citation_strategy,
            anchor_strategy=self.anchor_strategy,
            anchor_provenance=(self.anchor_hierarchy.provenance if self.anchor_hierarchy else None),
        )

    def _retrieve_seeds(
        self, question: str, query_embedding: np.ndarray
    ) -> tuple[list[RetrievalHit], list[str], list[str], list[str]]:
        """Return seeds, selected series, searched series, and empty series.

        Dense retrieval uses the enriched embedding. BM25 deliberately uses
        the original question, preserving the experiment's query policy.
        """
        if self.anchor_strategy == "hierarchical":
            selected: list[str] = []
            empty: list[str] = []
            if self.anchor_hierarchy is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("Hierarchical anchor artifact was not loaded.")
            seed_retrievals, searched = self.corpus.search_hierarchical(
                query_embedding,
                self.settings.get("rag", "retrieval_top_k"),
                query_text=question,
                hierarchy=self.anchor_hierarchy,
                series_weight=self.settings.get("rag", "anchor_series_weight"),
                document_weight=self.settings.get("rag", "anchor_document_weight"),
                chunk_weight=self.settings.get("rag", "anchor_chunk_weight"),
            )
        else:
            if self.router is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("Router was not initialized.")
            selected = self.router.route(query_embedding, self.settings.get("rag", "router_top_k"))
            seed_retrievals, searched, empty = self.corpus.search(
                selected,
                query_embedding,
                self.settings.get("rag", "retrieval_top_k"),
                query_text=question,
            )
        return seed_retrievals, selected, searched, empty

    def _prepare_facets(
        self, question: str, query_embedding: np.ndarray, strategy: str
    ) -> tuple[list[str], np.ndarray]:
        """Only the gain ablation needs additional LLM and embedding calls."""
        if strategy != "gain":
            return [], np.empty((0, query_embedding.shape[0]), dtype=np.float32)
        facets = self.client.extract_facets(question)
        embeddings = np.asarray(self.client.embed_many(facets), dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(facets):
            raise ValueError("Facet embeddings have an invalid shape.")
        return facets, embeddings

    @staticmethod
    def _format_contexts(retrievals: list[RetrievalHit]) -> list[str]:
        """Preserve baseline seed blocks and append neutral citation candidates."""
        seeds = [hit for hit in retrievals if hit.origin != "citation"]
        citations = [hit for hit in retrievals if hit.origin == "citation"]
        return [
            MuCiRAGService._format_context(hit)
            for hit in [*seeds, *citations]
        ]

    @staticmethod
    def _format_context(hit: RetrievalHit) -> str:
        """Keep seed formatting baseline-compatible and cite labels neutral."""
        metadata = hit.metadata
        chunk_id = metadata.get("chunk_id")
        if hit.origin == "citation":
            source = (
                "Referenced candidate; "
                f"series={hit.series}; document={metadata.get('document_id')}; "
                f"heading={metadata.get('heading')}; chunk_id={chunk_id}; "
                f"citation_depth={hit.citation_depth}; cited_by={hit.parent_chunk_id}; "
                f"path={hit.parent_chunk_id} -> {chunk_id}"
            )
            context = f"[{source}]\n{hit.text}"
        else:
            source = (
                f"series={hit.series}; document={metadata.get('document_id')}; "
                f"heading={metadata.get('heading')}; chunk_id={chunk_id}"
            )
            context = f"[{source}]\n{hit.text}"
        return context


# Compatibility for existing notebooks importing the pre-MuCiRAG name.
PaperRagService = MuCiRAGService
