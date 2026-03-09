"""Tests for embedding model wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from src.ingest.embedders import Embedder

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIM = 384


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Load embedder once for all tests in this module."""
    return Embedder(MODEL_NAME)


@pytest.mark.unit
def test_embedder_dimension(embedder: Embedder):
    assert embedder.dimension == EXPECTED_DIM


@pytest.mark.unit
def test_embed_single_text(embedder: Embedder):
    result = embedder.embed(["Hello world."])
    assert result.shape == (1, EXPECTED_DIM)
    assert result.dtype == np.float32


@pytest.mark.unit
def test_embed_batch(embedder: Embedder):
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    result = embedder.embed(texts)
    assert result.shape == (3, EXPECTED_DIM)


@pytest.mark.unit
def test_embed_empty_list(embedder: Embedder):
    result = embedder.embed([])
    assert result.shape == (0, EXPECTED_DIM)


@pytest.mark.unit
def test_embed_deterministic(embedder: Embedder):
    text = ["Determinism check."]
    a = embedder.embed(text)
    b = embedder.embed(text)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_similar_texts_closer(embedder: Embedder):
    texts = [
        "The cat sat on the mat.",
        "A cat was sitting on a mat.",
        "Quantum mechanics describes subatomic particles.",
    ]
    vecs = embedder.embed(texts)
    # Cosine similarity: similar sentences should be closer
    sim_close = _cosine_sim(vecs[0], vecs[1])
    sim_far = _cosine_sim(vecs[0], vecs[2])
    assert sim_close > sim_far


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
