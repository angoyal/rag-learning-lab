"""Integration tests for the vector store and ingest pipeline."""

from __future__ import annotations

import uuid

import pytest
from src.ingest.chunkers import fixed_chunker
from src.ingest.embedders import Embedder
from src.ingest.readers import read_text
from src.store.chroma_store import ChromaStore


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder("all-MiniLM-L6-v2")


@pytest.fixture
def store() -> ChromaStore:
    """Fresh in-memory ChromaStore with a unique collection per test."""
    return ChromaStore(collection_name=f"test_{uuid.uuid4().hex[:8]}")


# -- ChromaStore basic operations --


@pytest.mark.integration
def test_add_and_query(store: ChromaStore, embedder: Embedder):
    texts = ["The cat sat on the mat.", "Dogs love to play fetch."]
    embeddings = embedder.embed(texts)
    store.add(texts=texts, embeddings=embeddings)

    query_vec = embedder.embed(["cat"])
    results = store.query(query_embedding=query_vec[0], top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == "The cat sat on the mat."


@pytest.mark.integration
def test_query_top_k(store: ChromaStore, embedder: Embedder):
    texts = [f"Document number {i}." for i in range(10)]
    embeddings = embedder.embed(texts)
    store.add(texts=texts, embeddings=embeddings)

    query_vec = embedder.embed(["Document number 3."])
    results = store.query(query_embedding=query_vec[0], top_k=3)
    assert len(results) == 3


@pytest.mark.integration
def test_query_returns_distances(store: ChromaStore, embedder: Embedder):
    texts = ["Hello world.", "Goodbye world."]
    embeddings = embedder.embed(texts)
    store.add(texts=texts, embeddings=embeddings)

    query_vec = embedder.embed(["Hello"])
    results = store.query(query_embedding=query_vec[0], top_k=2)
    assert "distance" in results[0]
    # Results should be sorted by distance (ascending)
    assert results[0]["distance"] <= results[1]["distance"]


@pytest.mark.integration
def test_add_with_metadata(store: ChromaStore, embedder: Embedder):
    texts = ["First chunk.", "Second chunk."]
    embeddings = embedder.embed(texts)
    metadatas = [{"source": "doc1.txt", "index": 0}, {"source": "doc1.txt", "index": 1}]
    store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)

    query_vec = embedder.embed(["First"])
    results = store.query(query_embedding=query_vec[0], top_k=1)
    assert results[0]["metadata"]["source"] == "doc1.txt"
    assert results[0]["metadata"]["index"] == 0


@pytest.mark.integration
def test_add_incremental(store: ChromaStore, embedder: Embedder):
    batch1 = ["Cats are independent."]
    batch2 = ["Dogs are loyal."]
    store.add(texts=batch1, embeddings=embedder.embed(batch1))
    store.add(texts=batch2, embeddings=embedder.embed(batch2))

    query_vec = embedder.embed(["pets"])
    results = store.query(query_embedding=query_vec[0], top_k=10)
    assert len(results) == 2


@pytest.mark.integration
def test_query_empty_store(store: ChromaStore, embedder: Embedder):
    query_vec = embedder.embed(["anything"])
    results = store.query(query_embedding=query_vec[0], top_k=5)
    assert results == []


@pytest.mark.integration
def test_count(store: ChromaStore, embedder: Embedder):
    assert store.count() == 0
    texts = ["One.", "Two.", "Three."]
    store.add(texts=texts, embeddings=embedder.embed(texts))
    assert store.count() == 3


# -- Full ingest pipeline round-trip --


@pytest.mark.integration
def test_ingest_then_query_round_trip(tmp_path, embedder: Embedder):
    """End-to-end: read file -> chunk -> embed -> store -> query."""
    para1 = (
        "Python is a high-level programming language. "
        "It emphasizes code readability and simplicity. "
        "Python supports multiple programming paradigms."
    )
    para2 = (
        "The Great Wall of China is one of the seven wonders of the world. "
        "It stretches over 13,000 miles across northern China. "
        "Construction began in the 7th century BC."
    )
    f = tmp_path / "test.txt"
    f.write_text(para1 + "\n\n" + para2)

    doc = read_text(f)
    chunks = fixed_chunker(doc.text, chunk_size=250, chunk_overlap=0)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    metadatas = [{"source": doc.source, "chunk_index": c.index} for c in chunks]

    store = ChromaStore(collection_name=f"roundtrip_{uuid.uuid4().hex[:8]}")
    store.add(texts=texts, embeddings=embeddings, metadatas=metadatas)

    query_vec = embedder.embed(["What programming language emphasizes readability?"])
    results = store.query(query_embedding=query_vec[0], top_k=1)
    assert "Python" in results[0]["text"]

    query_vec = embedder.embed(["ancient construction in China"])
    results = store.query(query_embedding=query_vec[0], top_k=1)
    assert "China" in results[0]["text"]
