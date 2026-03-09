"""Cross-encoder reranking for retrieval refinement."""

from __future__ import annotations

from sentence_transformers import CrossEncoder


class Reranker:
    """Reranks retrieval results using a cross-encoder model.

    Cross-encoders jointly encode the query and document together,
    producing more accurate relevance scores than bi-encoder similarity
    at the cost of higher latency.

    Args:
        model_name: HuggingFace cross-encoder model name.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank retrieval results by cross-encoder relevance score.

        Args:
            query: The search query string.
            results: List of dicts from a retriever, each with a "text" key.
            top_k: Number of top results to return after reranking.

        Returns:
            List of dicts (same format as input) reordered by relevance,
            with a "rerank_score" key added to each.
        """
        if not results:
            return []
        pairs = [[query, r["text"]] for r in results]
        scores = self._model.predict(pairs)
        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)
        ranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
        return ranked[:top_k]
