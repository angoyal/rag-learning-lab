"""Base vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVectorStore(ABC):
    """Abstract interface for vector stores.

    All vector store implementations must support adding texts with
    their pre-computed embeddings and querying by embedding vector.
    """

    @abstractmethod
    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add texts and their embeddings to the store.

        Args:
            texts: List of text strings to store.
            embeddings: NumPy array of shape (len(texts), dimension).
            metadatas: Optional list of metadata dicts, one per text.
        """

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Query the store for the most similar texts.

        Args:
            query_embedding: 1-D NumPy array of the query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys "text", "metadata", "distance",
            sorted by distance ascending (most similar first).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of items in the store."""
