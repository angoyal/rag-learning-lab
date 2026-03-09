"""Tests for retriever, reranker, and hybrid retriever."""

from __future__ import annotations

import uuid

import pytest
from src.ingest.embedders import Embedder
from src.retrieve.hybrid_retriever import HybridRetriever
from src.retrieve.reranker import Reranker
from src.retrieve.retriever import Retriever
from src.store.chroma_store import ChromaStore


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def reranker_model() -> Reranker:
    return Reranker("cross-encoder/ms-marco-MiniLM-L-2-v2")


CORPUS = [
    "Python is a high-level programming language.",
    "Java is widely used for enterprise applications.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Mount Everest is the tallest mountain on Earth.",
    "Neural networks learn patterns from training data.",
    "Bread dough needs time to rise before baking.",
]


def _build_store(embedder: Embedder, texts: list[str] | None = None) -> ChromaStore:
    texts = texts or CORPUS
    store = ChromaStore(collection_name=f"test_{uuid.uuid4().hex[:8]}")
    store.add(texts=texts, embeddings=embedder.embed(texts))
    return store


# -- Step 8: Retriever tests --


@pytest.mark.unit
def test_retriever_returns_top_k(embedder: Embedder):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("programming language", top_k=3)
    assert len(results) == 3


@pytest.mark.unit
def test_retriever_finds_relevant(embedder: Embedder):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("programming language", top_k=1)
    assert "Python" in results[0]["text"] or "Java" in results[0]["text"]


@pytest.mark.unit
def test_retriever_result_format(embedder: Embedder):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("mountains", top_k=1)
    assert "text" in results[0]
    assert "distance" in results[0]
    assert "metadata" in results[0]


@pytest.mark.unit
def test_retriever_sorted_by_distance(embedder: Embedder):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("coding", top_k=5)
    distances = [r["distance"] for r in results]
    assert distances == sorted(distances)


@pytest.mark.unit
def test_retriever_empty_store(embedder: Embedder):
    store = ChromaStore(collection_name=f"empty_{uuid.uuid4().hex[:8]}")
    retriever = Retriever(store, embedder)
    results = retriever.retrieve("anything", top_k=5)
    assert results == []


# -- Step 9: Reranker tests --


@pytest.mark.unit
def test_reranker_reorders(embedder: Embedder, reranker_model: Reranker):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    initial = retriever.retrieve("What programming language is easy to read?", top_k=6)
    reranked = reranker_model.rerank(
        "What programming language is easy to read?", initial, top_k=3
    )
    assert len(reranked) == 3
    # Reranked results should have rerank_score
    assert "rerank_score" in reranked[0]


@pytest.mark.unit
def test_reranker_scores_descending(embedder: Embedder, reranker_model: Reranker):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    initial = retriever.retrieve("tall mountains", top_k=6)
    reranked = reranker_model.rerank("tall mountains", initial, top_k=6)
    scores = [r["rerank_score"] for r in reranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_reranker_top_k_limits(embedder: Embedder, reranker_model: Reranker):
    results = [{"text": f"Doc {i}.", "distance": 0.1 * i, "metadata": {}} for i in range(10)]
    reranked = reranker_model.rerank("test query", results, top_k=3)
    assert len(reranked) == 3


@pytest.mark.unit
def test_reranker_empty_results(reranker_model: Reranker):
    reranked = reranker_model.rerank("query", [], top_k=5)
    assert reranked == []


@pytest.mark.unit
def test_reranker_improves_relevance(embedder: Embedder, reranker_model: Reranker):
    store = _build_store(embedder)
    retriever = Retriever(store, embedder)
    initial = retriever.retrieve("baking bread at home", top_k=6)
    reranked = reranker_model.rerank("baking bread at home", initial, top_k=1)
    assert "Bread" in reranked[0]["text"] or "baking" in reranked[0]["text"].lower()


# -- Step 10: Hybrid retriever tests --


@pytest.mark.unit
def test_hybrid_rrf_returns_results(embedder: Embedder):
    store = _build_store(embedder)
    hybrid = HybridRetriever(store, embedder, CORPUS)
    results = hybrid.retrieve("programming language", top_k=3, fusion="rrf")
    assert len(results) <= 3
    assert all("text" in r and "score" in r for r in results)


@pytest.mark.unit
def test_hybrid_weighted_returns_results(embedder: Embedder):
    store = _build_store(embedder)
    hybrid = HybridRetriever(store, embedder, CORPUS)
    results = hybrid.retrieve("programming", top_k=3, fusion="weighted")
    assert len(results) <= 3


@pytest.mark.unit
def test_hybrid_rrf_finds_relevant(embedder: Embedder):
    store = _build_store(embedder)
    hybrid = HybridRetriever(store, embedder, CORPUS)
    results = hybrid.retrieve("Python programming", top_k=2, fusion="rrf")
    texts = [r["text"] for r in results]
    assert any("Python" in t for t in texts)


@pytest.mark.unit
def test_hybrid_catches_keyword_match(embedder: Embedder):
    corpus = [
        "The XRF-7 protocol handles data transmission.",
        "Network protocols manage communication between devices.",
        "Cooking requires good ingredients.",
    ]
    store = ChromaStore(collection_name=f"kw_{uuid.uuid4().hex[:8]}")
    store.add(texts=corpus, embeddings=embedder.embed(corpus))
    hybrid = HybridRetriever(store, embedder, corpus)
    results = hybrid.retrieve("XRF-7", top_k=2, fusion="rrf")
    texts = [r["text"] for r in results]
    assert any("XRF-7" in t for t in texts)


@pytest.mark.unit
def test_hybrid_invalid_fusion(embedder: Embedder):
    store = _build_store(embedder)
    hybrid = HybridRetriever(store, embedder, CORPUS)
    with pytest.raises(ValueError, match="Unknown fusion"):
        hybrid.retrieve("test", fusion="invalid")


@pytest.mark.unit
def test_hybrid_scores_descending(embedder: Embedder):
    store = _build_store(embedder)
    hybrid = HybridRetriever(store, embedder, CORPUS)
    results = hybrid.retrieve("mountains", top_k=5, fusion="rrf")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
