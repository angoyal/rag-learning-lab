"""Ingest documents into the vector store.

Usage:
    uv run python scripts/ingest_docs.py --config configs/experiments/01_baseline.yaml
    uv run python scripts/ingest_docs.py \\
        --config configs/experiments/01_baseline.yaml --data-dir data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import RAGPipeline, load_config

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


def find_documents(data_dir: str) -> list[Path]:
    """Find all supported document files in a directory."""
    root = Path(data_dir)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.glob(f"**/*{ext}"))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--config",
        default="configs/experiments/01_baseline.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing documents to ingest (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    # CLI --data-dir overrides config; otherwise use first path from config's data.documents
    if args.data_dir:
        data_dir = args.data_dir
    else:
        doc_dirs = config.get("data", {}).get("documents", ["data/raw"])
        data_dir = doc_dirs[0] if doc_dirs else "data/raw"

    files = find_documents(data_dir)
    if not files:
        print(f"No documents found in {data_dir}")
        return

    print(f"Found {len(files)} document(s) in {data_dir}")
    num_chunks = pipeline.ingest(files)
    collection = config.get("experiment_name", "default")
    print(f"\nIngested {num_chunks} chunks into collection '{collection}'")


if __name__ == "__main__":
    main()
