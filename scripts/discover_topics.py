"""Discover topic clusters from ingested paper embeddings in ChromaDB.

Pulls all embeddings, aggregates to document level, clusters with KMeans,
and prints representative paper titles per cluster.

Usage:
    uv run python scripts/discover_topics.py --config configs/experiments/arxiv_auto_scaling.yaml
    uv run python scripts/discover_topics.py \
        --config configs/experiments/01_baseline.yaml --n-clusters 15
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
import yaml
from sklearn.cluster import KMeans


def load_config(config_path: str) -> dict:
    """Load a YAML experiment config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_arxiv_metadata(data_dir: str) -> dict[str, dict]:
    """Load arxiv metadata YAML, keyed by source path for lookup.

    Args:
        data_dir: Directory containing arxiv_metadata.yaml.

    Returns:
        Dict mapping source path strings to metadata dicts.
    """
    meta_path = Path(data_dir) / "arxiv_metadata.yaml"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        data = yaml.safe_load(f)
    if not data or "papers" not in data:
        return {}
    lookup = {}
    for paper in data["papers"]:
        pdf_file = paper.get("pdf_file", "")
        lookup[pdf_file] = paper
        lookup[str(Path(data_dir) / pdf_file)] = paper
    return lookup


def main():
    """Discover and print topic clusters from ingested embeddings."""
    parser = argparse.ArgumentParser(description="Discover topic clusters from ingested papers")
    parser.add_argument(
        "--config",
        default="configs/experiments/01_baseline.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of topic clusters (default: 10)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of representative papers per cluster (default: 5)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    persist_dir = config.get("store", {}).get("persist_directory", "data/chroma")
    collection_name = config.get("experiment_name", "default")

    # Load arxiv metadata for title lookup
    doc_dirs = config.get("data", {}).get("documents", ["data/raw"])
    arxiv_meta = {}
    for d in doc_dirs:
        arxiv_meta.update(load_arxiv_metadata(d))

    print(f"Connecting to ChromaDB at {persist_dir}, collection '{collection_name}'")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)

    total_chunks = collection.count()
    print(f"Total chunks in collection: {total_chunks}")

    # Pull embeddings in batches, aggregating per-document on the fly
    # to avoid holding all chunk embeddings in memory at once.
    # We accumulate running sums and counts per source document.
    batch_size = 10000
    doc_sums: dict[str, np.ndarray] = {}
    doc_counts: dict[str, int] = defaultdict(int)
    total_loaded = 0

    while total_loaded < total_chunks:
        batch = collection.get(
            include=["embeddings", "metadatas"],
            limit=batch_size,
            offset=total_loaded,
        )
        for emb, meta in zip(batch["embeddings"], batch["metadatas"]):
            source = meta.get("source", "unknown")
            emb_arr = np.array(emb)
            if source in doc_sums:
                doc_sums[source] += emb_arr
            else:
                doc_sums[source] = emb_arr.copy()
            doc_counts[source] += 1
        total_loaded += len(batch["ids"])
        print(f"  Loaded {total_loaded}/{total_chunks} chunks")

    # Compute mean embeddings per document
    doc_sources = sorted(doc_sums.keys())
    doc_embeddings = np.array([
        doc_sums[source] / doc_counts[source] for source in doc_sources
    ])

    n_docs = len(doc_sources)
    print(f"\n{n_docs} documents, {total_loaded} total chunks")

    n_clusters = min(args.n_clusters, n_docs)
    print(f"Clustering into {n_clusters} topics...\n")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(doc_embeddings)
    centroids = kmeans.cluster_centers_

    # For each cluster, find the papers closest to the centroid
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = doc_embeddings[cluster_indices]

        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest = np.argsort(distances)[: args.top_n]

        print(f"{'=' * 80}")
        print(f"CLUSTER {cluster_id + 1}  ({len(cluster_indices)} papers)")
        print(f"{'=' * 80}")

        for rank, idx in enumerate(closest, 1):
            source = doc_sources[cluster_indices[idx]]
            filename = Path(source).name

            meta = arxiv_meta.get(filename) or arxiv_meta.get(source)
            if meta:
                title = meta.get("title", filename)
                categories = ", ".join(meta.get("categories", []))
                print(f"  {rank}. {title}")
                if categories:
                    print(f"     [{categories}]")
            else:
                print(f"  {rank}. {filename}")
        print()

    # Cluster size summary
    print(f"{'=' * 80}")
    print("CLUSTER SIZE SUMMARY")
    print(f"{'=' * 80}")
    for cluster_id in range(n_clusters):
        count = int(np.sum(labels == cluster_id))
        bar = "#" * (count // 2)
        print(f"  Cluster {cluster_id + 1:2d}: {count:4d} papers {bar}")


if __name__ == "__main__":
    main()
