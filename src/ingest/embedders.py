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
        self._model = SentenceTransformer(
            model_name, trust_remote_code=True,
        )
        self.dimension: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed a list of texts into dense vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to encode per GPU batch. Larger values
                use more memory but improve throughput.

        Returns:
            NumPy array of shape (len(texts), dimension) with float32 embeddings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._model.encode(
            texts, convert_to_numpy=True, batch_size=batch_size,
        )
