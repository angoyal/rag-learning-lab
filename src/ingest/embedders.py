"""Embedding model wrapper around sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper around sentence-transformers for text embedding.

    Args:
        model_name: HuggingFace model name (e.g. "all-MiniLM-L6-v2").
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            NumPy array of shape (len(texts), dimension) with float32 embeddings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True)
