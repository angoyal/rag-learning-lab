"""Run a single experiment from a config file.

Usage:
    uv run python scripts/run_experiment.py --config configs/experiments/01_baseline.yaml
"""

from __future__ import annotations

import argparse

from src.experiment_runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAG experiment")
    parser.add_argument(
        "--config",
        default="configs/experiments/01_baseline.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    print(f"Running experiment from {args.config}...")
    result = run_experiment(args.config)

    print(f"\nExperiment: {result['experiment_name']}")
    print(f"Queries run: {result['num_queries']}")
    print(f"Avg latency: {result['avg_latency_s']:.3f}s")
    print(f"Avg chunks/query: {result['avg_chunks_per_query']:.1f}")

    if result["results"]:
        print("\n--- Results ---")
        for i, r in enumerate(result["results"], 1):
            print(f"\nQ{i}: {r['question']}")
            print(f"A:  {r['answer'][:200]}...")
            print(f"    ({len(r['chunks'])} chunks, {r['latency_s']:.2f}s)")


if __name__ == "__main__":
    main()
