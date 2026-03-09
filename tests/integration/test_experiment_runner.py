"""Integration tests for the experiment runner."""

from __future__ import annotations

import pytest
import yaml
from src.experiment_runner import _log_nested_params, load_eval_set, run_experiment


@pytest.mark.integration
def test_load_eval_set_list_format(tmp_path):
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        yaml.dump([
            {"question": "What is Python?", "reference_answer": "A programming language."},
            {"question": "What is RAG?", "reference_answer": "Retrieval-Augmented Generation."},
        ])
    )
    items = load_eval_set(str(eval_file))
    assert len(items) == 2
    assert items[0]["question"] == "What is Python?"


@pytest.mark.integration
def test_load_eval_set_dict_format(tmp_path):
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        yaml.dump({
            "questions": [
                {"question": "Q1?"},
                {"question": "Q2?"},
            ]
        })
    )
    items = load_eval_set(str(eval_file))
    assert len(items) == 2


@pytest.mark.integration
def test_run_experiment_no_data(tmp_path):
    """Experiment runs with no data paths and no eval set — logs params only."""
    config = {
        "experiment_name": "test_empty_run",
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
            "prompt_template": "default_qa",
        },
    }
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(yaml.dump(config))

    result = run_experiment(str(config_file))
    assert result["experiment_name"] == "test_empty_run"
    assert result["num_queries"] == 0
    assert result["avg_latency_s"] == 0.0
    assert result["results"] == []


@pytest.mark.integration
def test_log_nested_params():
    """Verify _log_nested_params flattens a nested dict without error."""
    config = {
        "ingestion": {"chunker": "fixed", "chunk_size": 512},
        "retrieval": {"top_k": 5, "reranker": None},
        "generation": {"llm": "ollama/llama3.2"},
        "evaluation": {"metrics": ["faithfulness", "relevancy"]},
    }
    # Should not raise — MLflow logs to local file store
    _log_nested_params(config)
