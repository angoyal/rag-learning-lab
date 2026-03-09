"""Experiment runner for systematic RAG evaluations.

Loads a YAML config, builds a RAG pipeline, runs queries from an eval set,
and logs all parameters and metrics to MLflow for comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import yaml

from src.pipeline import RAGPipeline


def load_eval_set(eval_path: str) -> list[dict]:
    """Load an evaluation dataset (list of question/answer dicts) from YAML.

    Args:
        eval_path: Path to a YAML file containing a list of dicts,
            each with at least a "question" key and optionally a
            "reference_answer" key.

    Returns:
        List of eval item dicts.
    """
    with open(eval_path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data


def run_experiment(config_path: str) -> dict:
    """Run a full experiment: build pipeline, ingest, query, and log to MLflow.

    Args:
        config_path: Path to the experiment YAML config file.

    Returns:
        Dict with "experiment_name", "num_queries", "avg_latency_s",
        "avg_chunks_per_query", and "results" (list of per-query outputs).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", Path(config_path).stem)

    # Build pipeline
    pipeline = RAGPipeline(config)

    # Ingest documents if data paths are configured
    data_paths = config.get("data", {}).get("documents", [])
    num_chunks = 0
    if data_paths:
        paths = [Path(p) for p in data_paths]
        num_chunks = pipeline.ingest(paths)

    # Load eval questions if configured
    eval_path = config.get("data", {}).get("eval_set")
    if eval_path:
        eval_items = load_eval_set(eval_path)
    else:
        eval_items = []

    # Run queries and collect metrics
    results = []
    latencies = []
    chunk_counts = []

    for item in eval_items:
        question = item["question"]
        start = time.perf_counter()
        output = pipeline.query(question)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        chunk_counts.append(len(output["chunks"]))
        results.append({
            "question": question,
            "answer": output["answer"],
            "chunks": output["chunks"],
            "latency_s": elapsed,
            "reference_answer": item.get("reference_answer"),
        })

    # Compute aggregate metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_chunks = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0.0

    # Log to MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=experiment_name):
        # Log all config params (flattened)
        _log_nested_params(config)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("num_chunks_ingested", num_chunks)

        # Log aggregate metrics
        mlflow.log_metric("num_queries", len(eval_items))
        mlflow.log_metric("avg_latency_s", avg_latency)
        mlflow.log_metric("avg_chunks_per_query", avg_chunks)

        # Log per-query latencies
        for i, lat in enumerate(latencies):
            mlflow.log_metric("query_latency_s", lat, step=i)

    return {
        "experiment_name": experiment_name,
        "num_queries": len(eval_items),
        "avg_latency_s": avg_latency,
        "avg_chunks_per_query": avg_chunks,
        "results": results,
    }


def _log_nested_params(d: dict, prefix: str = "") -> None:
    """Recursively log a nested dict as flat MLflow params."""
    for key, value in d.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            _log_nested_params(value, full_key)
        elif isinstance(value, list):
            mlflow.log_param(full_key, str(value))
        else:
            mlflow.log_param(full_key, value)
