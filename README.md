# RAG Learning Lab

An open-source project for learning Retrieval-Augmented Generation (RAG) in public. Build intuition on what works by running progressive experiments — from a naive baseline to production-grade patterns — while tracking every decision with proper evaluation and experiment management.

## Two-Machine Workflow

```
Mac (dev)                              Linux (GPU testing)
  - Write code                           - Ollama + Llama 3.2
  - Unit tests, linting                  - Full pipeline runs
  - Notebooks (CPU)                      - Eval sweeps
  - MLflow UI (via SSH tunnel)           - NVIDIA GPU (16 GB VRAM)
```

## Project Structure

```
src/
├── ingest/          # Document loading, chunking, embedding
├── store/           # Vector store backends (Chroma, LanceDB)
├── retrieve/        # Retrieval strategies (dense, hybrid, reranking)
├── generate/        # LLM clients and prompt templates
├── evaluate/        # RAGAS, DeepEval, custom metrics, MLflow logging
├── pipeline.py      # End-to-end RAG pipeline
└── experiment_runner.py

configs/             # YAML experiment configs
scripts/             # CLI entrypoints for ingestion, experiments, sweeps
tests/               # Unit, integration, and evaluation tests
notebooks/           # Interactive exploration and visualization
data/                # Raw docs, processed chunks, eval datasets
deploy/              # Docker, AWS Terraform, GCP Terraform
docs/                # Learnings and experiment log
```

## Experiments

| # | Experiment | Focus |
|---|-----------|-------|
| 01 | Baseline | Naive RAG with default settings |
| 02 | Semantic Chunking | Compare chunking strategies |
| 03 | BGE Embeddings | Swap embedding models |
| 04 | With Reranker | Add cross-encoder reranking |
| 05 | Hybrid Retrieval | Combine dense + sparse search |
| 06 | Prompt Variations | Test prompt template impact |
| 07 | LLM Comparison | Compare generation models |
| 08 | Scale Test | Performance under load |
| 09 | From Scratch | Build the core loop without any framework |

## Getting Started

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone <repo-url>
cd rag-learning-lab
uv sync --all-extras

# Install Ollama and pull a model
brew install ollama        # macOS
ollama serve
ollama pull llama3.2

# Verify setup
uv run pytest tests/ -v
uv run ruff check src/ tests/
```

## Development

```bash
make lint              # Run ruff linter
make format            # Auto-format code
make test-unit         # Run unit tests
make test-integration  # Run integration tests (needs Ollama running)
make eval              # Run evaluation suite
make test              # Run all tests
```

## Tech Stack

- **Orchestration**: LlamaIndex (primary), Haystack (alt)
- **LLM Serving**: Ollama (local), vLLM (alt)
- **Vector Stores**: ChromaDB, LanceDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2, BGE-M3)
- **Evaluation**: RAGAS, DeepEval, custom metrics
- **Experiment Tracking**: MLflow
- **Serving**: FastAPI
- **Package Management**: uv
