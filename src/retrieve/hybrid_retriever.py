"""Hybrid retrieval combining dense and sparse (BM25) search."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from src.ingest.embedders import Embedder
    from src.store.base import BaseVectorStore


class HybridRetriever:
    """Combines dense (embedding) and sparse (BM25) retrieval.

    Dense search captures semantic similarity. BM25 catches exact
    keyword matches (acronyms, proper nouns, codes) that embeddings
    miss. Results are merged via reciprocal rank fusion (RRF) or
    weighted score fusion.

    Args:
        store: A vector store for dense retrieval.
        embedder: An embedder for query vectorization.
        corpus: List of document texts to build the BM25 index from.
    """

    def __init__(
        self,
        store: BaseVectorStore,
        embedder: Embedder,
        corpus: list[str],
    ):
        self._store = store
        self._embedder = embedder
        self._corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion: str = "rrf",
        dense_weight: float = 0.7,
    ) -> list[dict]:
        """Retrieve documents using hybrid dense + sparse search.

        Args:
            query: The search query string.
            top_k: Number of final results to return.
            fusion: Fusion method — "rrf" (reciprocal rank fusion) or "weighted".
            dense_weight: Weight for dense scores when fusion="weighted" (0-1).
                Sparse weight is 1 - dense_weight.

        Returns:
            List of dicts with keys "text", "score", sorted by score descending.

        Raises:
            ValueError: If fusion method is not recognized.
        """
        dense_results = self._dense_search(query, top_k=top_k)
        sparse_results = self._sparse_search(query, top_k=top_k)

        if fusion == "rrf":
            return self._rrf_fusion(dense_results, sparse_results, top_k)
        elif fusion == "weighted":
            return self._weighted_fusion(
                dense_results, sparse_results, top_k, dense_weight
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion!r}")

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        query_embedding = self._embedder.embed([query])[0]
        results = self._store.query(query_embedding=query_embedding, top_k=top_k)
        return [{"text": r["text"], "distance": r["distance"]} for r in results]

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"text": self._corpus[i], "bm25_score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    def _rrf_fusion(
        self, dense: list[dict], sparse: list[dict], top_k: int, k: int = 60
    ) -> list[dict]:
        """Reciprocal rank fusion: score = sum(1 / (k + rank)) across rankers."""
        scores: dict[str, float] = {}
        for rank, r in enumerate(dense):
            scores[r["text"]] = scores.get(r["text"], 0) + 1.0 / (k + rank + 1)
        for rank, r in enumerate(sparse):
            scores[r["text"]] = scores.get(r["text"], 0) + 1.0 / (k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"text": text, "score": score} for text, score in ranked[:top_k]]

    def _weighted_fusion(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
        dense_weight: float,
    ) -> list[dict]:
        """Weighted score fusion with min-max normalized scores."""
        scores: dict[str, float] = {}
        # Normalize dense distances to [0, 1] similarity (lower distance = higher sim)
        if dense:
            max_dist = max(r["distance"] for r in dense)
            min_dist = min(r["distance"] for r in dense)
            dist_range = max_dist - min_dist if max_dist != min_dist else 1.0
            for r in dense:
                sim = 1.0 - (r["distance"] - min_dist) / dist_range
                scores[r["text"]] = dense_weight * sim

        # Normalize BM25 scores to [0, 1]
        if sparse:
            max_bm25 = max(r["bm25_score"] for r in sparse)
            min_bm25 = min(r["bm25_score"] for r in sparse)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
            sparse_weight = 1.0 - dense_weight
            for r in sparse:
                norm = (r["bm25_score"] - min_bm25) / bm25_range
                scores[r["text"]] = scores.get(r["text"], 0) + sparse_weight * norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"text": text, "score": score} for text, score in ranked[:top_k]]
