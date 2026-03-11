"""ChromaDB vector store implementation."""

from __future__ import annotations

import uuid

import chromadb
import numpy as np

from src.store.base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Vector store backed by ChromaDB.

    Uses an in-memory ephemeral client by default. For persistence,
    pass a chromadb client configured with a persistent directory.

    Args:
        collection_name: Name of the ChromaDB collection.
        client: Optional pre-configured ChromaDB client.
    """

    def __init__(
        self,
        collection_name: str = "default",
        client: chromadb.ClientAPI | None = None,
        persist_directory: str | None = None,
    ):
        if client:
            self._client = client
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add texts and their embeddings to the ChromaDB collection.

        Args:
            texts: List of text strings to store.
            embeddings: NumPy array of shape (len(texts), dimension).
            metadatas: Optional list of metadata dicts, one per text.
        """
        ids = [str(uuid.uuid4()) for _ in texts]
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """Query ChromaDB for the most similar texts.

        Args:
            query_embedding: 1-D NumPy array of the query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys "text", "metadata", "distance",
            sorted by distance ascending.
        """
        if self._collection.count() == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        """Return the number of items in the collection."""
        return self._collection.count()

    def ingested_sources(self) -> set[str]:
        """Return the set of source filenames already stored in the collection."""
        if self._collection.count() == 0:
            return set()
        all_meta = self._collection.get(include=["metadatas"])
        return {m["source"] for m in all_meta["metadatas"] if "source" in m}
