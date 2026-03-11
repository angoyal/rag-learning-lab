"""Fast document ingestor with producer-consumer parallelism.

Architecture:
- Producer threads (configurable count): read PDFs from disk and extract
  text in parallel. This is the I/O bottleneck for large corpora.
- Main thread (consumer): receives extracted text, chunks it, embeds on
  GPU, and stores to ChromaDB. GPU and ChromaDB are both single-threaded
  resources, so running them on the main thread avoids locking entirely.

The speedup comes from overlapping: while the main thread embeds doc N
on the GPU, producer threads are already reading docs N+1 through N+8
from disk.
"""

from __future__ import annotations

import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ingest.chunkers import chunk_text, semantic_chunker
from src.ingest.embedders import Embedder
from src.ingest.metrics import IngestMetrics
from src.ingest.readers import read_document
from src.store.chroma_store import ChromaStore


@dataclass
class ReadResult:
    """Result of reading a document in a producer thread.

    Attributes:
        path: Original file path.
        text: Extracted text content (empty string if no text).
        file_size_bytes: Size of the file on disk.
        read_time_s: Time spent reading the file.
        error: Error message if reading failed.
    """

    path: Path
    text: str = ""
    file_size_bytes: int = 0
    read_time_s: float = 0.0
    error: str | None = None


def _read_doc(path: Path) -> ReadResult:
    """Read a document file in a producer thread, recording timing.

    Args:
        path: Path to the document file.

    Returns:
        ReadResult with extracted text, file size, read time, or error.
    """
    try:
        file_size = path.stat().st_size
        t0 = time.monotonic()
        doc = read_document(path)
        read_time = time.monotonic() - t0
        return ReadResult(
            path=path, text=doc.text,
            file_size_bytes=file_size, read_time_s=read_time,
        )
    except Exception as e:
        return ReadResult(path=path, error=str(e))


class FastIngestor:
    """High-throughput document ingestor with producer-consumer parallelism.

    Producer threads read PDFs from disk in parallel (I/O-bound).
    The main thread consumes extracted text: chunks, embeds on GPU,
    and stores to ChromaDB (all single-threaded resources).

    For semantic chunking with a large embedding model, a separate
    lightweight ``chunking_embedder`` can detect sentence boundaries
    quickly while the main ``embedder`` produces high-quality vectors
    for storage and retrieval.

    Args:
        store: ChromaDB vector store to write to.
        embedder: Embedder for computing storage/retrieval embeddings.
        chunker_strategy: Chunking strategy name.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        batch_size: Number of texts per embedding batch.
        workers: Number of parallel reader threads.
        chunking_embedder: Optional lightweight embedder for semantic
            split-point detection. Falls back to ``embedder`` if not set.
    """

    def __init__(
        self,
        store: ChromaStore,
        embedder: Embedder,
        chunker_strategy: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 256,
        workers: int = 4,
        chunking_embedder: Embedder | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.chunking_embedder = chunking_embedder
        self.chunker_strategy = chunker_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.workers = workers

    def ingest(self, paths: list[Path]) -> int:
        """Ingest documents into the vector store using producer-consumer pipeline.

        Producer threads read documents from disk in parallel. The main
        thread processes each result: chunks text, embeds on GPU, and
        stores to ChromaDB.

        Args:
            paths: List of file paths to ingest.

        Returns:
            Total number of chunks stored.
        """
        already_ingested = self.store.ingested_sources()
        to_process = [p for p in paths if str(p) not in already_ingested]
        skipped = len(paths) - len(to_process)

        if skipped:
            print(
                f"  Skipped {skipped} already-ingested document(s)",
                flush=True,
            )

        if not to_process:
            return 0

        return self._ingest_producer_consumer(to_process)

    def _chunk_embed_store(self, text: str, source: str) -> dict:
        """Chunk text, embed, and store to ChromaDB. Runs on main thread.

        For semantic chunking with a separate chunking_embedder:
        1. Use the small chunking_embedder to find split points (fast)
        2. Re-embed the final chunks with the main embedder (high quality)

        Args:
            text: Extracted document text.
            source: File path string for metadata.

        Returns:
            Dict with num_chunks, num_sentences, chunk_time_s,
            embed_time_s, and store_time_s.
        """
        # Count sentences for metrics
        sentences = re.split(r"(?<=[.!?])\s+", text)
        num_sentences = len([s for s in sentences if s.strip()])

        # --- Chunk ---
        t0 = time.monotonic()
        if self.chunker_strategy == "semantic":
            chunker_emb = self.chunking_embedder or self.embedder
            if self.chunking_embedder:
                chunks = semantic_chunker(
                    text, chunker_emb, batch_size=self.batch_size,
                )
            else:
                chunks, embeddings = semantic_chunker(
                    text, chunker_emb, return_embeddings=True,
                    batch_size=self.batch_size,
                )
        else:
            chunks = chunk_text(
                text,
                strategy=self.chunker_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        chunk_time = time.monotonic() - t0

        if not chunks:
            return {
                "num_chunks": 0, "num_sentences": num_sentences,
                "chunk_time_s": chunk_time,
                "embed_time_s": 0.0, "store_time_s": 0.0,
            }

        texts = [c.text for c in chunks]
        metadatas = [
            {"source": source, "chunk_index": c.index}
            for c in chunks
        ]

        # --- Embed ---
        t0 = time.monotonic()
        if self.chunker_strategy != "semantic" or self.chunking_embedder:
            embeddings = self.embedder.embed(
                texts, batch_size=self.batch_size,
            )
        embed_time = time.monotonic() - t0

        # --- Store ---
        t0 = time.monotonic()
        self.store.add(texts, np.array(embeddings), metadatas)
        store_time = time.monotonic() - t0

        return {
            "num_chunks": len(chunks),
            "num_sentences": num_sentences,
            "chunk_time_s": chunk_time,
            "embed_time_s": embed_time,
            "store_time_s": store_time,
        }

    def _ingest_producer_consumer(self, paths: list[Path]) -> int:
        """Read docs in parallel (producers), process on main thread (consumer).

        Collects per-document timing metrics and writes them to a JSONL
        log file in ``debug/logs/``.

        Args:
            paths: List of file paths to process.

        Returns:
            Total number of chunks stored.
        """
        n = len(paths)
        total = 0
        failed = 0
        empty = 0
        completed = 0

        print(
            f"  Ingesting {n} documents "
            f"({self.workers} reader threads)...",
            flush=True,
        )

        result_queue: queue.Queue[ReadResult] = queue.Queue(
            maxsize=self.workers * 2,
        )

        def _producer_worker(path: Path) -> None:
            result = _read_doc(path)
            result_queue.put(result)

        with (
            IngestMetrics() as metrics,
            ThreadPoolExecutor(max_workers=self.workers) as pool,
        ):
            futures = [
                pool.submit(_producer_worker, p) for p in paths
            ]

            for _ in range(n):
                read_result = result_queue.get()
                completed += 1
                doc_t0 = time.monotonic()

                if read_result.error:
                    failed += 1
                    metrics.record({
                        "source": read_result.path.name,
                        "file_size_bytes": read_result.file_size_bytes,
                        "read_time_s": read_result.read_time_s,
                        "num_sentences": 0,
                        "chunk_time_s": 0.0,
                        "num_chunks": 0,
                        "embed_time_s": 0.0,
                        "store_time_s": 0.0,
                        "total_time_s": 0.0,
                        "error": read_result.error,
                    })
                    print(
                        f"  [{completed}/{n}] FAILED "
                        f"{read_result.path.name}: {read_result.error}",
                        flush=True,
                    )
                    continue

                if not read_result.text.strip():
                    empty += 1
                    continue

                try:
                    stage_metrics = self._chunk_embed_store(
                        read_result.text, str(read_result.path),
                    )
                except Exception as e:
                    failed += 1
                    metrics.record({
                        "source": read_result.path.name,
                        "file_size_bytes": read_result.file_size_bytes,
                        "read_time_s": read_result.read_time_s,
                        "num_sentences": 0,
                        "chunk_time_s": 0.0,
                        "num_chunks": 0,
                        "embed_time_s": 0.0,
                        "store_time_s": 0.0,
                        "total_time_s": time.monotonic() - doc_t0,
                        "error": str(e),
                    })
                    print(
                        f"  [{completed}/{n}] FAILED "
                        f"{read_result.path.name}: {e}",
                        flush=True,
                    )
                    continue

                chunks_stored = stage_metrics["num_chunks"]
                total_time = (
                    read_result.read_time_s
                    + stage_metrics["chunk_time_s"]
                    + stage_metrics["embed_time_s"]
                    + stage_metrics["store_time_s"]
                )

                metrics.record({
                    "source": read_result.path.name,
                    "file_size_bytes": read_result.file_size_bytes,
                    "read_time_s": read_result.read_time_s,
                    "num_sentences": stage_metrics["num_sentences"],
                    "chunk_time_s": stage_metrics["chunk_time_s"],
                    "num_chunks": chunks_stored,
                    "embed_time_s": stage_metrics["embed_time_s"],
                    "store_time_s": stage_metrics["store_time_s"],
                    "total_time_s": total_time,
                    "error": None,
                })

                if chunks_stored == 0:
                    empty += 1
                else:
                    total += chunks_stored

                if completed % 10 == 0 or completed == n or completed == 1:
                    print(
                        f"  [{completed}/{n}] "
                        f"{read_result.path.name} -> "
                        f"{chunks_stored} chunks "
                        f"(total stored: {total})",
                        flush=True,
                    )

                # Flush metrics every 100 documents
                if completed % 100 == 0:
                    metrics.flush()

            for f in futures:
                f.result()

        if failed:
            print(f"  {failed} document(s) failed", flush=True)
        if empty:
            print(
                f"  {empty} document(s) had no extractable text",
                flush=True,
            )
        print(
            f"  Done: {total} chunks from "
            f"{n - failed - empty} documents",
            flush=True,
        )
        return total
