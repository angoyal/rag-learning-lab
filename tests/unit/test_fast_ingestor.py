"""Tests for the fast ingestor — producer-consumer pipeline, all chunking strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.ingest.chunkers import (
    chunk_text,
    fixed_chunker,
    recursive_chunker,
    semantic_chunker,
    sentence_chunker,
)
from src.ingest.embedders import Embedder
from src.ingest.fast_ingestor import FastIngestor, _read_doc
from src.pipeline import RAGPipeline
from src.store.chroma_store import ChromaStore

# ---------------------------------------------------------------------------
# Sample text used across all tests — long enough for meaningful chunking
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Machine learning is a subset of artificial intelligence. "
    "It allows systems to learn from data without being explicitly programmed. "
    "Supervised learning uses labeled training data to make predictions. "
    "Unsupervised learning finds hidden patterns in unlabeled data. "
    "Reinforcement learning trains agents through reward signals. "
    "Deep learning uses neural networks with many layers. "
    "Convolutional neural networks excel at image recognition tasks. "
    "Recurrent neural networks are designed for sequential data processing. "
    "Transformers have revolutionized natural language processing. "
    "Large language models are trained on vast amounts of text data. "
    "Transfer learning allows models to leverage pre-trained knowledge. "
    "Fine-tuning adapts pre-trained models to specific downstream tasks. "
    "Retrieval-augmented generation combines search with text generation. "
    "Vector databases store embeddings for fast similarity search. "
    "Embedding models convert text into dense numerical representations."
)


def _write_sample_files(tmp_path: Path, count: int = 3) -> list[Path]:
    """Create sample text files for testing."""
    files = []
    for i in range(count):
        f = tmp_path / f"doc_{i}.txt"
        f.write_text(f"Document {i}. {SAMPLE_TEXT}")
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# Chunking strategies on sample text
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkingStrategiesOnSampleText:
    """Verify all four chunking strategies produce valid chunks from sample text."""

    def test_fixed_chunker(self) -> None:
        chunks = fixed_chunker(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 200
            assert chunk.text == SAMPLE_TEXT[chunk.start:chunk.end]

    def test_recursive_chunker(self) -> None:
        chunks = recursive_chunker(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 200

    def test_sentence_chunker(self) -> None:
        chunks = sentence_chunker(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 200 + 50  # sentences may exceed slightly

    def test_semantic_chunker(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        chunks = semantic_chunker(SAMPLE_TEXT, embedder, threshold=0.5)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text in SAMPLE_TEXT

    def test_semantic_chunker_with_return_embeddings(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        result = semantic_chunker(
            SAMPLE_TEXT, embedder, threshold=0.5, return_embeddings=True,
        )
        chunks, embeddings = result
        assert len(chunks) >= 1
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == embedder.dimension

    def test_chunk_text_dispatches_fixed(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, "fixed", 200, 20)
        assert len(chunks) >= 2

    def test_chunk_text_dispatches_recursive(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, "recursive", 200, 20)
        assert len(chunks) >= 2

    def test_chunk_text_dispatches_sentence(self) -> None:
        chunks = chunk_text(SAMPLE_TEXT, "sentence", 200, 20)
        assert len(chunks) >= 2

    def test_chunk_text_dispatches_semantic(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        chunks = chunk_text(SAMPLE_TEXT, "semantic", 200, 20, embedder=embedder)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Semantic chunker edge cases with return_embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSemanticChunkerReturnEmbeddings:

    def test_empty_text(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        chunks, embeddings = semantic_chunker(
            "", embedder, return_embeddings=True,
        )
        assert chunks == []
        assert embeddings.shape[0] == 0

    def test_single_sentence(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        text = "Just one sentence here."
        chunks, embeddings = semantic_chunker(
            text, embedder, return_embeddings=True,
        )
        assert len(chunks) == 1
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == embedder.dimension

    def test_embeddings_are_normalized_means(self) -> None:
        """Chunk embeddings should be mean-pooled from sentence embeddings."""
        embedder = Embedder("all-MiniLM-L6-v2")
        chunks, embeddings = semantic_chunker(
            SAMPLE_TEXT, embedder, threshold=0.5, return_embeddings=True,
        )
        # Each embedding should be a valid vector (not NaN, not zero)
        for emb in embeddings:
            assert not np.any(np.isnan(emb))
            assert np.linalg.norm(emb) > 0


# ---------------------------------------------------------------------------
# _read_doc (producer function)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadDoc:

    def test_reads_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)
        result = _read_doc(f)
        assert result.error is None
        assert result.text == SAMPLE_TEXT

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        result = _read_doc(tmp_path / "missing.txt")
        assert result.error is not None

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = _read_doc(f)
        assert result.error is None
        assert result.text == ""


# ---------------------------------------------------------------------------
# _chunk_embed_store (consumer function)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkEmbedStore:

    def test_fixed_strategy(self) -> None:
        store = ChromaStore(collection_name="test_ces_fixed")
        embedder = Embedder("all-MiniLM-L6-v2")
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
        )
        result = ingestor._chunk_embed_store(SAMPLE_TEXT, "test.txt")
        assert result["num_chunks"] >= 2
        assert store.count() == result["num_chunks"]
        assert result["chunk_time_s"] >= 0
        assert result["embed_time_s"] >= 0
        assert result["store_time_s"] >= 0
        assert result["num_sentences"] >= 1

    def test_semantic_strategy(self) -> None:
        store = ChromaStore(collection_name="test_ces_semantic")
        embedder = Embedder("all-MiniLM-L6-v2")
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="semantic",
        )
        result = ingestor._chunk_embed_store(SAMPLE_TEXT, "test.txt")
        assert result["num_chunks"] >= 1
        assert store.count() == result["num_chunks"]

    def test_two_model_semantic(self) -> None:
        """Small model chunks, same model embeds (simulates two-model path)."""
        store = ChromaStore(collection_name="test_ces_two_model")
        embedder = Embedder("all-MiniLM-L6-v2")
        chunking_embedder = Embedder("all-MiniLM-L6-v2")
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="semantic",
            chunking_embedder=chunking_embedder,
        )
        result = ingestor._chunk_embed_store(SAMPLE_TEXT, "test.txt")
        assert result["num_chunks"] >= 1
        assert store.count() == result["num_chunks"]
        query_emb = embedder.embed(["What is deep learning?"])[0]
        results = store.query(query_emb, top_k=3)
        assert len(results) >= 1

    def test_empty_text_returns_zero(self) -> None:
        store = ChromaStore(collection_name="test_ces_empty")
        embedder = Embedder("all-MiniLM-L6-v2")
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
        )
        result = ingestor._chunk_embed_store("", "test.txt")
        assert result["num_chunks"] == 0


# ---------------------------------------------------------------------------
# FastIngestor — non-semantic strategies (parallel)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFastIngestorNonSemantic:

    @pytest.fixture()
    def embedder(self) -> Embedder:
        return Embedder("all-MiniLM-L6-v2")

    def test_fixed_chunking(self, tmp_path: Path, embedder: Embedder) -> None:
        store = ChromaStore(collection_name="test_fast_fixed")
        files = _write_sample_files(tmp_path, count=3)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
            batch_size=32, workers=2,
        )
        total = ingestor.ingest(files)
        assert total >= 3
        assert store.count() == total

    def test_recursive_chunking(
        self, tmp_path: Path, embedder: Embedder,
    ) -> None:
        store = ChromaStore(collection_name="test_fast_recursive")
        files = _write_sample_files(tmp_path, count=3)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="recursive", chunk_size=200, chunk_overlap=20,
            batch_size=32, workers=2,
        )
        total = ingestor.ingest(files)
        assert total >= 3
        assert store.count() == total

    def test_sentence_chunking(
        self, tmp_path: Path, embedder: Embedder,
    ) -> None:
        store = ChromaStore(collection_name="test_fast_sentence")
        files = _write_sample_files(tmp_path, count=3)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="sentence", chunk_size=200, chunk_overlap=20,
            batch_size=32, workers=2,
        )
        total = ingestor.ingest(files)
        assert total >= 3
        assert store.count() == total

    def test_skips_already_ingested(
        self, tmp_path: Path, embedder: Embedder,
    ) -> None:
        store = ChromaStore(collection_name="test_fast_skip")
        files = _write_sample_files(tmp_path, count=2)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
            batch_size=32, workers=2,
        )
        first_total = ingestor.ingest(files)
        second_total = ingestor.ingest(files)
        assert second_total == 0
        assert store.count() == first_total

    def test_handles_mixed_valid_invalid(
        self, tmp_path: Path, embedder: Embedder,
    ) -> None:
        store = ChromaStore(collection_name="test_fast_mixed")
        good = tmp_path / "good.txt"
        good.write_text(SAMPLE_TEXT)
        bad = tmp_path / "bad.xyz"
        bad.write_text("unsupported format")
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
            batch_size=32, workers=2,
        )
        total = ingestor.ingest([good, bad])
        assert total >= 1  # good file ingested
        assert store.count() >= 1


# ---------------------------------------------------------------------------
# FastIngestor — semantic strategy (parallel, embedding reuse)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFastIngestorSemantic:

    @pytest.fixture()
    def embedder(self) -> Embedder:
        return Embedder("all-MiniLM-L6-v2")

    def test_semantic_chunking(
        self, tmp_path: Path, embedder: Embedder,
    ) -> None:
        store = ChromaStore(collection_name="test_fast_sem_chunk")
        files = _write_sample_files(tmp_path, count=2)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="semantic",
            batch_size=32, workers=2,
        )
        total = ingestor.ingest(files)
        assert total >= 2
        assert store.count() == total

    def test_semantic_embeddings_are_queryable(
        self, tmp_path: Path, embedder: Embedder,
    ) -> None:
        store = ChromaStore(collection_name="test_fast_sem_query")
        """Verify that pre-computed semantic embeddings work for retrieval."""
        f = tmp_path / "ml.txt"
        f.write_text(SAMPLE_TEXT)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="semantic",
            batch_size=32, workers=2,
        )
        ingestor.ingest([f])
        # Query should find relevant chunks
        query_emb = embedder.embed(["What is deep learning?"])[0]
        results = store.query(query_emb, top_k=3)
        assert len(results) >= 1
        # At least one result should mention neural networks or deep learning
        texts = " ".join(r["text"] for r in results).lower()
        assert "neural" in texts or "deep" in texts or "learning" in texts


# ---------------------------------------------------------------------------
# FastIngestor — parallel execution verification
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFastIngestorParallel:
    """Verify that multiple worker threads process documents concurrently."""

    def test_parallel_with_many_docs(self, tmp_path: Path) -> None:
        """Ingest 8 docs with 4 workers — all should be stored correctly."""
        store = ChromaStore(collection_name="test_fast_parallel_many")
        embedder = Embedder("all-MiniLM-L6-v2")
        files = _write_sample_files(tmp_path, count=8)
        ingestor = FastIngestor(
            store=store, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
            batch_size=32, workers=4,
        )
        total = ingestor.ingest(files)
        assert total >= 8
        assert store.count() == total
        # Verify all 8 sources are present
        sources = store.ingested_sources()
        assert len(sources) == 8

    def test_single_worker_matches_multi_worker(
        self, tmp_path: Path,
    ) -> None:
        """Single-worker and multi-worker should store the same chunk count."""
        embedder = Embedder("all-MiniLM-L6-v2")
        files = _write_sample_files(tmp_path, count=4)

        store1 = ChromaStore(collection_name="test_fast_1worker")
        ingestor1 = FastIngestor(
            store=store1, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
            workers=1,
        )
        total1 = ingestor1.ingest(files)

        store4 = ChromaStore(collection_name="test_fast_4workers")
        ingestor4 = FastIngestor(
            store=store4, embedder=embedder,
            chunker_strategy="fixed", chunk_size=200, chunk_overlap=20,
            workers=4,
        )
        total4 = ingestor4.ingest(files)

        assert total1 == total4


# ---------------------------------------------------------------------------
# Pipeline routing — config selects ingestor
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPipelineIngestorRouting:

    def test_default_ingestor(self, tmp_path: Path) -> None:
        config = {
            "experiment_name": "test_default_route",
            "ingestion": {
                "chunker": "fixed",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "embedding_model": "all-MiniLM-L6-v2",
            },
            "retrieval": {"top_k": 3, "reranker": None},
            "generation": {
                "llm": "ollama/llama3.2",
                "temperature": 0.0,
                "prompt_template": "default_qa",
            },
        }
        p = RAGPipeline(config)
        assert p.ingestor_type == "default"
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)
        total = p.ingest([f])
        assert total >= 1

    def test_fast_ingestor_via_config(self, tmp_path: Path) -> None:
        config = {
            "experiment_name": "test_fast_route",
            "ingestion": {
                "ingestor": "fast",
                "chunker": "fixed",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "embedding_model": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "workers": 2,
            },
            "retrieval": {"top_k": 3, "reranker": None},
            "generation": {
                "llm": "ollama/llama3.2",
                "temperature": 0.0,
                "prompt_template": "default_qa",
            },
        }
        p = RAGPipeline(config)
        assert p.ingestor_type == "fast"
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)
        total = p.ingest([f])
        assert total >= 1

    def test_fast_semantic_via_config(self, tmp_path: Path) -> None:
        config = {
            "experiment_name": "test_fast_semantic_route",
            "ingestion": {
                "ingestor": "fast",
                "chunker": "semantic",
                "embedding_model": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "workers": 2,
            },
            "retrieval": {"top_k": 3, "reranker": None},
            "generation": {
                "llm": "ollama/llama3.2",
                "temperature": 0.0,
                "prompt_template": "default_qa",
            },
        }
        p = RAGPipeline(config)
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)
        total = p.ingest([f])
        assert total >= 1

    def test_two_model_semantic_via_config(self, tmp_path: Path) -> None:
        config = {
            "experiment_name": "test_two_model_route",
            "ingestion": {
                "ingestor": "fast",
                "chunker": "semantic",
                "chunking_model": "all-MiniLM-L6-v2",
                "embedding_model": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "workers": 2,
            },
            "retrieval": {"top_k": 3, "reranker": None},
            "generation": {
                "llm": "ollama/llama3.2",
                "temperature": 0.0,
                "prompt_template": "default_qa",
            },
        }
        p = RAGPipeline(config)
        assert p.chunking_embedder is not None
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT)
        total = p.ingest([f])
        assert total >= 1


# ---------------------------------------------------------------------------
# Embedder batch_size parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderBatchSize:

    def test_default_batch_size(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        texts = ["hello world"] * 10
        result = embedder.embed(texts)
        assert result.shape == (10, embedder.dimension)

    def test_custom_batch_size(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        texts = ["hello world"] * 100
        result = embedder.embed(texts, batch_size=16)
        assert result.shape == (100, embedder.dimension)

    def test_batch_size_larger_than_input(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        texts = ["hello"]
        result = embedder.embed(texts, batch_size=1000)
        assert result.shape == (1, embedder.dimension)

    def test_empty_input(self) -> None:
        embedder = Embedder("all-MiniLM-L6-v2")
        result = embedder.embed([], batch_size=32)
        assert result.shape == (0, embedder.dimension)
