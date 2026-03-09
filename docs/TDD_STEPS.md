# TDD Implementation Steps

Progress tracker for the RAG Learning Lab, following the plan's weekly learning order.

| Step | Component | Files | Status |
|---|---|---|---|
| 1 | Chunk dataclass + fixed/recursive/sentence chunkers | `src/ingest/chunkers.py`, `tests/unit/test_chunkers.py` | Done |
| 2 | Notebook: Understanding Chunking | `notebooks/01_understanding_chunking.ipynb` | Done |
| 3 | Document readers (PDF, DOCX, MD, HTML, TXT) | `src/ingest/readers.py`, `tests/unit/test_readers.py` | Done |
| 4 | Embedders (sentence-transformers wrapper) | `src/ingest/embedders.py`, `tests/unit/test_embedders.py` | Done |
| 5 | Semantic chunker (embedding-based) | `src/ingest/chunkers.py` (add), `tests/unit/test_chunkers.py` (add) | Done |
| 6 | Vector store interface + ChromaDB | `src/store/base.py`, `src/store/chroma_store.py`, `tests/integration/test_ingest_pipeline.py` | Done |
| 7 | Notebook: Embedding Space Viz | `notebooks/02_embedding_space_viz.ipynb` | Done |
| 8 | Retriever (top-k similarity) | `src/retrieve/retriever.py`, `tests/unit/test_retriever.py` | Done |
| 9 | Reranker (cross-encoder) | `src/retrieve/reranker.py`, `tests/unit/test_retriever.py` | Done |
| 10 | Hybrid retriever (BM25 + dense fusion) | `src/retrieve/hybrid_retriever.py`, `tests/unit/test_retriever.py` | Done |
| 11 | Notebook: Retrieval Deep Dive | `notebooks/03_retrieval_deep_dive.ipynb` | Done |
| 12 | LLM client (Ollama) | `src/generate/llm_client.py` | Done |
| 13 | Prompt templates (Jinja2) | `src/generate/prompt_templates.py`, `tests/unit/test_prompt_templates.py` | Done |
| 14 | Notebook: Prompt Engineering | `notebooks/04_prompt_engineering.ipynb` | Done |
| 15 | Pipeline (wire components from config) | `src/pipeline.py`, `tests/integration/test_query_pipeline.py` | Done |
| 16 | Experiment runner + MLflow | `src/experiment_runner.py`, `tests/integration/test_experiment_runner.py` | Done |
| 17 | Evaluation harness (RAGAS/DeepEval) | `src/evaluate/ragas_eval.py`, `src/evaluate/deepeval_eval.py`, `src/evaluate/custom_metrics.py`, `tests/unit/test_custom_metrics.py`, `tests/unit/test_eval_harness.py` | Done |
| 18 | Notebook: Evaluation Analysis | `notebooks/05_evaluation_analysis.ipynb` | Done |
| 19 | Security tests (implement stubs) | `tests/security/functional/`, `tests/security/deployment/` | Done |
| 20 | Deployment (implement scripts) | `scripts/deploy.py`, `scripts/post_deploy_checks.py`, `tests/unit/test_deploy.py` | Done |

## Advanced Steps (second pass)

| Step | Component | Files | Status |
|---|---|---|---|
| 3a | Streaming readers (yield per page/section for large files) | `src/ingest/readers.py` (add) | Pending |
| 3b | Encoding detection (chardet/charset-normalizer) | `src/ingest/readers.py` (add) | Pending |
| 3c | PDF table extraction (pdfplumber tables) | `src/ingest/readers.py` (add) | Pending |
| 3d | Markdown front matter (YAML → metadata) | `src/ingest/readers.py` (add) | Pending |
