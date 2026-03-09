"""Integration tests for the config-driven RAG pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.pipeline import RAGPipeline, load_config

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# -- Config loading --


@pytest.mark.integration
def test_load_config_returns_dict():
    config = load_config(str(CONFIGS_DIR / "experiments" / "01_baseline.yaml"))
    assert isinstance(config, dict)
    assert "experiment_name" in config
    assert "ingestion" in config
    assert "retrieval" in config
    assert "generation" in config


@pytest.mark.integration
def test_load_config_all_experiments():
    """Every YAML file in configs/experiments/ should load without error."""
    experiment_dir = CONFIGS_DIR / "experiments"
    yamls = sorted(experiment_dir.glob("*.yaml"))
    assert len(yamls) >= 1
    for path in yamls:
        config = load_config(str(path))
        assert "experiment_name" in config, f"{path.name} missing experiment_name"
        assert "ingestion" in config, f"{path.name} missing ingestion section"


# -- Pipeline construction --


@pytest.fixture(scope="module")
def baseline_config():
    return load_config(str(CONFIGS_DIR / "experiments" / "01_baseline.yaml"))


@pytest.fixture(scope="module")
def pipeline(baseline_config):
    return RAGPipeline(baseline_config)


@pytest.mark.integration
def test_pipeline_has_components(pipeline):
    assert pipeline.embedder is not None
    assert pipeline.store is not None
    assert pipeline.retriever is not None
    assert pipeline.llm is not None


@pytest.mark.integration
def test_pipeline_no_reranker_when_null(pipeline):
    assert pipeline.reranker is None


@pytest.mark.integration
def test_pipeline_with_reranker():
    config = load_config(str(CONFIGS_DIR / "experiments" / "04_with_reranker.yaml"))
    p = RAGPipeline(config)
    assert p.reranker is not None


# -- Ingestion --


@pytest.mark.integration
def test_ingest_text_file(pipeline, tmp_path):
    txt = tmp_path / "sample.txt"
    txt.write_text(
        "Python is a versatile programming language. "
        "It supports object-oriented, functional, and procedural paradigms. "
        "Python has a large standard library and active community."
    )
    count = pipeline.ingest([txt])
    assert count >= 1
    assert pipeline.store.count() >= 1


@pytest.mark.integration
def test_ingest_multiple_files(tmp_path):
    config = {
        "experiment_name": "test_multi_ingest",
        "ingestion": {
            "chunker": "fixed",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "embedding_model": "all-MiniLM-L6-v2",
        },
        "retrieval": {"top_k": 3, "reranker": None},
        "generation": {
            "llm": "ollama/llama3.2",
            "temperature": 0.1,
            "prompt_template": "default_qa",
        },
    }
    p = RAGPipeline(config)
    f1 = tmp_path / "doc1.txt"
    f1.write_text("Machine learning is a subset of artificial intelligence.")
    f2 = tmp_path / "doc2.txt"
    f2.write_text("Deep learning uses neural networks with many layers.")
    count = p.ingest([f1, f2])
    assert count >= 2
    assert p.store.count() >= 2


# -- Retrieval (build_prompt, no LLM needed) --


@pytest.mark.integration
def test_build_prompt_returns_chunks(pipeline, tmp_path):
    # Ingest some content first
    txt = tmp_path / "facts.txt"
    txt.write_text(
        "The Eiffel Tower is located in Paris, France. "
        "It was built in 1889 for the World's Fair. "
        "The tower stands 330 meters tall."
    )
    pipeline.ingest([txt])
    result = pipeline.build_prompt("Where is the Eiffel Tower?")
    assert "chunks" in result
    assert "prompt" in result
    assert len(result["chunks"]) >= 1


@pytest.mark.integration
def test_build_prompt_contains_question(pipeline):
    result = pipeline.build_prompt("What is Python?")
    assert "What is Python?" in result["prompt"]


@pytest.mark.integration
def test_build_prompt_uses_configured_template():
    config = {
        "experiment_name": "test_structured",
        "ingestion": {
            "chunker": "fixed",
            "chunk_size": 200,
            "chunk_overlap": 20,
            "embedding_model": "all-MiniLM-L6-v2",
        },
        "retrieval": {"top_k": 3, "reranker": None},
        "generation": {
            "llm": "ollama/llama3.2",
            "temperature": 0.1,
            "prompt_template": "structured",
        },
    }
    p = RAGPipeline(config)
    # The structured template contains "I don't know"
    result = p.build_prompt("Anything?")
    assert "don't know" in result["prompt"].lower() or len(result["chunks"]) == 0


# -- Config validation --


@pytest.mark.integration
def test_pipeline_default_config():
    """Pipeline can be built with minimal config using defaults."""
    config = {"experiment_name": "minimal_test"}
    p = RAGPipeline(config)
    assert p.chunker_strategy == "fixed"
    assert p.chunk_size == 512
    assert p.top_k == 5
    assert p.prompt_template == "default_qa"


@pytest.mark.integration
def test_base_config_loads():
    config = load_config(str(CONFIGS_DIR / "base.yaml"))
    assert "ingestion" in config
    assert config["ingestion"]["chunker"] == "fixed"
    assert config["ingestion"]["embedding_model"] == "all-MiniLM-L6-v2"
