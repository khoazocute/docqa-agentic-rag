"""Hybrid retrieval: dense vector search + BM25, merged with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore

from docqa.config import Settings

logger = logging.getLogger(__name__)

# Very frequent Vietnamese function syllables. Removing them from the BM25 corpus keeps
# scores focused on content words. Vietnamese has no stemming, so stemming is disabled.
VI_STOPWORDS = [
    "và",
    "của",
    "là",
    "có",
    "các",
    "những",
    "được",
    "cho",
    "với",
    "trong",
    "này",
    "khi",
    "thì",
    "để",
    "một",
    "từ",
    "đến",
    "về",
    "theo",
    "như",
    "nếu",
    "hoặc",
    "nào",
    "gì",
    "bị",
    "do",
    "đã",
    "sẽ",
    "đang",
    "ra",
    "vào",
    "tại",
    "lại",
    "cũng",
    "mà",
    "thế",
    "sao",
]


@dataclass
class RetrievedChunk:
    node: BaseNode
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None

    @property
    def text(self) -> str:
        return self.node.get_content()

    @property
    def source(self) -> str:
        meta = self.node.metadata or {}
        return meta.get("source") or meta.get("file_name") or "unknown"

    @property
    def page(self) -> str | None:
        page = (self.node.metadata or {}).get("page_label")
        return str(page) if page else None

    @property
    def section(self) -> str | None:
        return (self.node.metadata or {}).get("section") or None


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_vector_score: float | None = None


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[NodeWithScore]], k: int = 60
) -> list[tuple[str, float]]:
    """Classic RRF: score(d) = sum over lists of 1 / (k + rank_d). Ranks start at 1."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.node.node_id] = scores.get(hit.node.node_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def protect_leaders(
    ranking: list[tuple[str, float]],
    ranked_lists: Sequence[Sequence[NodeWithScore]],
    top_k: int,
) -> list[tuple[str, float]]:
    """Guarantee each retriever's #1 result a place in the fused top-k.

    RRF rewards agreement: a chunk found by both retrievers at mediocre ranks outscores a chunk
    ranked #1 by only one of them (1/61 vs ~2/62). That drops, for example, a definition
    paragraph that dense search ranks first but BM25 misses because it mentions the keyword
    only once. Leaders replace the lowest-ranked non-leader; the final order stays by RRF score.
    """
    leaders = list(dict.fromkeys(ranked[0].node.node_id for ranked in ranked_lists if ranked))
    selected = [node_id for node_id, _ in ranking[:top_k]]
    for leader in leaders:
        if leader in selected:
            continue
        for i in range(len(selected) - 1, -1, -1):
            if selected[i] not in leaders:
                selected[i] = leader
                break
    scores = dict(ranking)
    return sorted(((node_id, scores[node_id]) for node_id in selected), key=lambda x: -x[1])


def build_bm25(nodes: list[BaseNode], top_k: int) -> BaseRetriever | None:
    if not nodes:
        return None
    from llama_index.retrievers.bm25 import BM25Retriever

    logging.getLogger("bm25s").setLevel(logging.WARNING)  # bm25s forces DEBUG on import
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=min(top_k, len(nodes)),
        skip_stemming=True,
        language=VI_STOPWORDS,
    )


class HybridRetriever(BaseRetriever):
    """LlamaIndex-compatible retriever that can run in vector, bm25 or hybrid mode.

    ``search(source=...)`` restricts both retrievers to one document: the vector side through a
    metadata filter pushed down to the vector store, the BM25 side through a per-document BM25
    index built lazily and cached.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever | None,
        bm25_retriever: BaseRetriever | None,
        *,
        mode: str = "hybrid",
        top_k: int = 4,
        rrf_k: int = 60,
        embed_query: Callable[[str], list[float]] | None = None,
        vector_for_source: Callable[[str], BaseRetriever | None] | None = None,
        bm25_for_source: Callable[[str], BaseRetriever | None] | None = None,
    ) -> None:
        super().__init__()
        self._vector = vector_retriever
        self._bm25 = bm25_retriever
        self._embed_query = embed_query
        self._vector_for_source = vector_for_source
        self._bm25_for_source = bm25_for_source
        self.mode = mode
        self.top_k = top_k
        self.rrf_k = rrf_k

    @classmethod
    def from_index(
        cls,
        index: VectorStoreIndex,
        settings: Settings,
        embed_query: Callable[[str], list[float]] | None = None,
    ) -> HybridRetriever:
        nodes = list(index.docstore.docs.values())
        k = settings.candidate_top_k
        bm25_cache: dict[str, BaseRetriever | None] = {}

        def vector_for_source(source: str) -> BaseRetriever:
            from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

            filters = MetadataFilters(filters=[MetadataFilter(key="source", value=source)])
            return index.as_retriever(similarity_top_k=k, filters=filters)

        def bm25_for_source(source: str) -> BaseRetriever | None:
            if source not in bm25_cache:
                subset = [n for n in nodes if (n.metadata or {}).get("source") == source]
                bm25_cache[source] = build_bm25(subset, k)
            return bm25_cache[source]

        return cls(
            index.as_retriever(similarity_top_k=k) if nodes else None,
            build_bm25(nodes, k),
            mode=settings.retrieval_mode,
            top_k=settings.top_k,
            rrf_k=settings.rrf_k,
            embed_query=embed_query,
            vector_for_source=vector_for_source if nodes else None,
            bm25_for_source=bm25_for_source if nodes else None,
        )

    def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        source: str | None = None,
    ) -> RetrievalResult:
        mode = mode or self.mode
        top_k = top_k or self.top_k
        bundle = QueryBundle(query_str=query)

        vector, bm25 = self._vector, self._bm25
        if source:
            vector = self._vector_for_source(source) if self._vector_for_source else None
            bm25 = self._bm25_for_source(source) if self._bm25_for_source else None

        vector_hits: list[NodeWithScore] = []
        bm25_hits: list[NodeWithScore] = []
        if mode in ("vector", "hybrid") and vector is not None:
            if self._embed_query is not None:
                bundle.embedding = self._embed_query(query)  # cached; skips re-embedding
            vector_hits = vector.retrieve(bundle)
        if mode in ("bm25", "hybrid") and bm25 is not None:
            # BM25 returns zero-score padding when fewer documents match; drop it.
            bm25_hits = [hit for hit in bm25.retrieve(bundle) if (hit.score or 0) > 0]

        vector_scores = {h.node.node_id: h.score for h in vector_hits}
        bm25_scores = {h.node.node_id: h.score for h in bm25_hits}
        nodes = {h.node.node_id: h.node for h in [*vector_hits, *bm25_hits]}
        best_vector = max((s for s in vector_scores.values() if s is not None), default=None)

        if mode == "hybrid":
            ranking = reciprocal_rank_fusion([vector_hits, bm25_hits], k=self.rrf_k)
            ranking = protect_leaders(ranking, [vector_hits, bm25_hits], top_k)
        else:
            hits = vector_hits if mode == "vector" else bm25_hits
            ranking = [(h.node.node_id, float(h.score or 0.0)) for h in hits]

        chunks = [
            RetrievedChunk(
                node=nodes[node_id],
                score=round(score, 6),
                vector_score=vector_scores.get(node_id),
                bm25_score=bm25_scores.get(node_id),
            )
            for node_id, score in ranking[:top_k]
        ]
        return RetrievalResult(chunks=chunks, best_vector_score=best_vector)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = self.search(query_bundle.query_str)
        return [NodeWithScore(node=c.node, score=c.score) for c in result.chunks]
