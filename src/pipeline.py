"""End-to-end RAG pipeline wiring all components from config."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.generate.llm_client import OllamaClient
from src.generate.prompt_templates import render_prompt, render_rewrite
from src.ingest.chunkers import chunk_text
from src.ingest.embedders import Embedder
from src.ingest.fast_ingestor import FastIngestor
from src.ingest.readers import read_document
from src.retrieve.reranker import Reranker
from src.retrieve.retriever import Retriever
from src.store.chroma_store import ChromaStore


def load_config(config_path: str) -> dict:
    """Load a YAML experiment config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """Config-driven RAG pipeline that wires together all components.

    Reads a config dict and builds: Reader -> Chunker -> Embedder -> Store ->
    Retriever -> (optional Reranker) -> Prompt Builder -> LLM.

    Args:
        config: Configuration dict with ingestion, retrieval, and generation sections.
    """

    def __init__(self, config: dict):
        ingestion = config.get("ingestion", {})
        retrieval = config.get("retrieval", {})
        generation = config.get("generation", {})

        # Ingestion settings
        self.ingestor_type = ingestion.get("ingestor", "default")
        self.chunker_strategy = ingestion.get("chunker", "fixed")
        self.chunk_size = ingestion.get("chunk_size", 512)
        self.chunk_overlap = ingestion.get("chunk_overlap", 50)
        self.batch_size = ingestion.get("batch_size", 256)
        self.workers = ingestion.get("workers", 4)

        # Build components
        self.embedder = Embedder(ingestion.get("embedding_model", "all-MiniLM-L6-v2"))
        store_config = config.get("store", {})
        self.store = ChromaStore(
            collection_name=config.get("experiment_name", "default"),
            persist_directory=store_config.get("persist_directory"),
        )
        self.retriever = Retriever(self.store, self.embedder)

        # Retrieval settings
        self.top_k = retrieval.get("top_k", 5)
        reranker_model = retrieval.get("reranker")
        self.reranker = Reranker(reranker_model) if reranker_model else None

        # Generation settings
        llm_spec = generation.get("llm", "ollama/llama3.2")
        model = llm_spec.split("/", 1)[1] if "/" in llm_spec else llm_spec
        self.llm = OllamaClient(
            model=model,
            temperature=generation.get("temperature", 0.1),
        )
        self.prompt_template = generation.get("prompt_template", "default_qa")

    def ingest(self, paths: list[Path]) -> int:
        """Ingest documents into the vector store.

        Routes to either the default sequential ingestor or the fast
        parallel ingestor based on the ``ingestion.ingestor`` config value.

        Args:
            paths: List of file paths to ingest.

        Returns:
            Total number of chunks stored.
        """
        if self.ingestor_type == "fast":
            return self._ingest_fast(paths)
        return self._ingest_default(paths)

    def _ingest_fast(self, paths: list[Path]) -> int:
        """Ingest using the fast parallel ingestor.

        Args:
            paths: List of file paths to ingest.

        Returns:
            Total number of chunks stored.
        """
        ingestor = FastIngestor(
            store=self.store,
            embedder=self.embedder,
            chunker_strategy=self.chunker_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            batch_size=self.batch_size,
            workers=self.workers,
        )
        return ingestor.ingest(paths)

    def _ingest_default(self, paths: list[Path]) -> int:
        """Ingest using the default sequential ingestor.

        Reads each file, chunks the text, computes embeddings, and stores them.
        Skips documents already in the store. Logs errors and continues on failure.

        Args:
            paths: List of file paths to ingest.

        Returns:
            Total number of chunks stored.
        """
        already_ingested = self.store.ingested_sources()
        total = 0
        failed = 0
        skipped = 0
        n = len(paths)

        for i, path in enumerate(paths, 1):
            path = Path(path)
            if str(path) in already_ingested:
                skipped += 1
                continue

            try:
                doc = read_document(path)
                chunks = chunk_text(
                    doc.text,
                    strategy=self.chunker_strategy,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    embedder=self.embedder if self.chunker_strategy == "semantic" else None,
                )
                if not chunks:
                    print(f"  [{i}/{n}] No text extracted: {path.name}")
                    continue
                texts = [c.text for c in chunks]
                embeddings = self.embedder.embed(texts)
                metadatas = [{"source": str(path), "chunk_index": c.index} for c in chunks]
                self.store.add(texts, embeddings, metadatas)
                total += len(chunks)
                print(f"  [{i}/{n}] {path.name} -> {len(chunks)} chunks")
            except Exception as e:
                failed += 1
                print(f"  [{i}/{n}] FAILED {path.name}: {e}")

        if skipped:
            print(f"  Skipped {skipped} already-ingested document(s)")
        if failed:
            print(f"  Failed on {failed} document(s)")
        return total

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question using conversation history.

        Uses the LLM to produce a self-contained question that can be
        understood without the conversation history, improving RAG retrieval.

        Args:
            question: The user's follow-up question.
            history: List of prior conversation turns with "role" and "content".

        Returns:
            The rewritten, self-contained question string.
        """
        prompt = render_rewrite(question, history)
        return self.llm.generate(prompt).strip()

    def query(
        self, question: str, history: list[dict] | None = None
    ) -> dict:
        """Run the full RAG query pipeline including LLM generation.

        When conversation history is provided, first rewrites the question
        to be self-contained, then retrieves and generates with history context.

        Args:
            question: The user's question.
            history: Optional list of prior conversation turns.

        Returns:
            Dict with "answer", "chunks", "prompt", "results", and
            "rewritten_query" keys.
        """
        rewritten = question
        if history:
            rewritten = self.rewrite_query(question, history)

        result = self.build_prompt(rewritten, history=history)
        answer = self.llm.generate(result["prompt"])
        return {
            "answer": answer,
            "chunks": result["chunks"],
            "prompt": result["prompt"],
            "results": result["results"],
            "rewritten_query": rewritten if history else None,
        }

    def build_prompt(
        self, question: str, history: list[dict] | None = None
    ) -> dict:
        """Run retrieval and build the prompt without calling the LLM.

        Useful for testing and debugging the retrieval + prompt pipeline
        without needing a running Ollama instance.

        Args:
            question: The user's question.
            history: Optional list of prior conversation turns.

        Returns:
            Dict with "chunks", "prompt", and "results" keys.
        """
        results = self.retriever.retrieve(question, top_k=self.top_k)
        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=self.top_k)
        chunks = [r["text"] for r in results]

        template = self.prompt_template
        if history:
            template = "conversational_qa"
        prompt = render_prompt(template, chunks, question, history=history)
        return {"chunks": chunks, "prompt": prompt, "results": results}
