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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ingest.chunkers import chunk_text, semantic_chunker
from src.ingest.embedders import Embedder
from src.ingest.readers import read_document
from src.store.chroma_store import ChromaStore


@dataclass
class ReadResult:
    """Result of reading a document in a producer thread.

    Attributes:
        path: Original file path.
        text: Extracted text content (empty string if no text).
        error: Error message if reading failed.
    """

    path: Path
    text: str = ""
    error: str | None = None


@dataclass
class DocResult:
    """Result of processing a single document through the full pipeline.

    Attributes:
        source: File path string.
        chunks_stored: Number of chunks successfully stored.
        error: Error message if processing failed.
    """

    source: str
    chunks_stored: int = 0
    error: str | None = None


def _read_doc(path: Path) -> ReadResult:
    """Read a document file in a producer thread.

    Args:
        path: Path to the document file.

    Returns:
        ReadResult with extracted text or error.
    """
    try:
        doc = read_document(path)
        return ReadResult(path=path, text=doc.text)
    except Exception as e:
        return ReadResult(path=path, error=str(e))


class FastIngestor:
    """High-throughput document ingestor with producer-consumer parallelism.

    Producer threads read PDFs from disk in parallel (I/O-bound).
    The main thread consumes extracted text: chunks, embeds on GPU,
    and stores to ChromaDB (all single-threaded resources).

    Args:
        store: ChromaDB vector store to write to.
        embedder: Embedder for computing text embeddings.
        chunker_strategy: Chunking strategy name.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.
        batch_size: Number of texts per embedding batch.
        workers: Number of parallel reader threads.
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
    ):
        self.store = store
        self.embedder = embedder
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

    def _chunk_embed_store(self, text: str, source: str) -> int:
        """Chunk text, embed, and store to ChromaDB. Runs on main thread.

        Args:
            text: Extracted document text.
            source: File path string for metadata.

        Returns:
            Number of chunks stored.
        """
        if self.chunker_strategy == "semantic":
            chunks, embeddings = semantic_chunker(
                text, self.embedder, return_embeddings=True,
                batch_size=self.batch_size,
            )
        else:
            chunks = chunk_text(
                text,
                strategy=self.chunker_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if not chunks:
                return 0
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(
                texts, batch_size=self.batch_size,
            )

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [
            {"source": source, "chunk_index": c.index}
            for c in chunks
        ]
        self.store.add(
            texts,
            embeddings if isinstance(embeddings, np.ndarray)
            else np.array(embeddings),
            metadatas,
        )
        return len(chunks)

    def _ingest_producer_consumer(self, paths: list[Path]) -> int:
        """Read docs in parallel (producers), process on main thread (consumer).

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

        # Use a bounded queue so producers don't get too far ahead
        # of the consumer (avoids memory bloat from reading all PDFs
        # before any are processed)
        result_queue: queue.Queue[ReadResult] = queue.Queue(
            maxsize=self.workers * 2,
        )

        def _producer_worker(path: Path) -> None:
            result = _read_doc(path)
            result_queue.put(result)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # Submit all read tasks — they'll block on the bounded queue
            # if the consumer falls behind
            futures = [
                pool.submit(_producer_worker, p) for p in paths
            ]

            # Consumer loop on main thread
            for _ in range(n):
                read_result = result_queue.get()
                completed += 1

                if read_result.error:
                    failed += 1
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
                    chunks_stored = self._chunk_embed_store(
                        read_result.text, str(read_result.path),
                    )
                except Exception as e:
                    failed += 1
                    print(
                        f"  [{completed}/{n}] FAILED "
                        f"{read_result.path.name}: {e}",
                        flush=True,
                    )
                    continue

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

            # Wait for all producer threads to finish
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
