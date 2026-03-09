"""Dense retriever using vector similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingest.embedders import Embedder
    from src.store.base import BaseVectorStore


class Retriever:
    """Retrieves documents by embedding similarity.

    Embeds the query, then finds the closest vectors in the store.

    Args:
        store: A vector store instance to search.
        embedder: An embedder to convert queries to vectors.
    """

    def __init__(self, store: BaseVectorStore, embedder: Embedder):
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the top-k most similar documents to the query.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys "text", "metadata", "distance",
            sorted by distance ascending (most similar first).
        """
        query_embedding = self._embedder.embed([query])[0]
        return self._store.query(query_embedding=query_embedding, top_k=top_k)
