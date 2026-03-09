"""End-to-end RAG pipeline wiring all components from config."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.generate.llm_client import OllamaClient
from src.generate.prompt_templates import render_prompt
from src.ingest.chunkers import chunk_text
from src.ingest.embedders import Embedder
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
        self.chunker_strategy = ingestion.get("chunker", "fixed")
        self.chunk_size = ingestion.get("chunk_size", 512)
        self.chunk_overlap = ingestion.get("chunk_overlap", 50)

        # Build components
        self.embedder = Embedder(ingestion.get("embedding_model", "all-MiniLM-L6-v2"))
        self.store = ChromaStore(
            collection_name=config.get("experiment_name", "default"),
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

        Reads each file, chunks the text, computes embeddings, and stores them.

        Args:
            paths: List of file paths to ingest.

        Returns:
            Total number of chunks stored.
        """
        total = 0
        for path in paths:
            doc = read_document(Path(path))
            chunks = chunk_text(
                doc.text,
                strategy=self.chunker_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                embedder=self.embedder if self.chunker_strategy == "semantic" else None,
            )
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed(texts)
            metadatas = [{"source": doc.source, "chunk_index": c.index} for c in chunks]
            self.store.add(texts, embeddings, metadatas)
            total += len(chunks)
        return total

    def query(self, question: str) -> dict:
        """Run the full RAG query pipeline including LLM generation.

        Args:
            question: The user's question.

        Returns:
            Dict with "answer", "chunks", and "prompt" keys.
        """
        result = self.build_prompt(question)
        answer = self.llm.generate(result["prompt"])
        return {"answer": answer, "chunks": result["chunks"], "prompt": result["prompt"]}

    def build_prompt(self, question: str) -> dict:
        """Run retrieval and build the prompt without calling the LLM.

        Useful for testing and debugging the retrieval + prompt pipeline
        without needing a running Ollama instance.

        Args:
            question: The user's question.

        Returns:
            Dict with "chunks", "prompt", and "results" keys.
        """
        results = self.retriever.retrieve(question, top_k=self.top_k)
        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=self.top_k)
        chunks = [r["text"] for r in results]
        prompt = render_prompt(self.prompt_template, chunks, question)
        return {"chunks": chunks, "prompt": prompt, "results": results}
