"""Discover topic clusters using HDBSCAN density-based clustering.

Unlike KMeans, HDBSCAN discovers the number of clusters automatically
and labels outlier papers as noise (-1).

Usage:
    uv run python scripts/discover_topics_hdbscan.py \
        --config configs/experiments/arxiv_auto_scaling.yaml
    uv run python scripts/discover_topics_hdbscan.py \
        --config configs/experiments/01_baseline.yaml \
        --min-cluster-size 20 --min-samples 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import chromadb
import hdbscan
import numpy as np
import yaml

CACHE_DIR = Path("data/cache")


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


def _aggregate_from_chromadb(
    collection: chromadb.Collection, total_chunks: int
) -> tuple[list[str], np.ndarray]:
    """Load embeddings from ChromaDB and aggregate to document level.

    Uses streaming aggregation (running sum + count) to avoid holding
    all chunk embeddings in memory.

    Args:
        collection: ChromaDB collection to read from.
        total_chunks: Total number of chunks in the collection.

    Returns:
        Tuple of (sorted source list, document embedding matrix).
    """
    batch_size = 50000
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

    doc_sources = sorted(doc_sums.keys())
    doc_embeddings = np.array([
        doc_sums[s] / doc_counts[s] for s in doc_sources
    ])
    return doc_sources, doc_embeddings


def load_doc_embeddings(
    collection: chromadb.Collection,
    total_chunks: int,
    collection_name: str,
    force_reload: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Load document embeddings, using a disk cache for speed.

    On first run, aggregates chunk embeddings from ChromaDB and saves the
    result to data/cache/. Subsequent runs load directly from cache (~instant).
    Use --reload to force a fresh load from ChromaDB.

    Args:
        collection: ChromaDB collection to read from.
        total_chunks: Total number of chunks in the collection.
        collection_name: Used to name the cache files.
        force_reload: If True, skip cache and reload from ChromaDB.

    Returns:
        Tuple of (sorted source list, document embedding matrix).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_cache = CACHE_DIR / f"{collection_name}_doc_embeddings.npy"
    src_cache = CACHE_DIR / f"{collection_name}_doc_sources.json"

    if not force_reload and emb_cache.exists() and src_cache.exists():
        print("  Loading from cache...")
        doc_embeddings = np.load(emb_cache)
        with open(src_cache) as f:
            doc_sources = json.load(f)
        print(f"  Loaded {len(doc_sources)} documents from cache")
        return doc_sources, doc_embeddings

    print("  Loading from ChromaDB (first run, will cache)...")
    doc_sources, doc_embeddings = _aggregate_from_chromadb(
        collection, total_chunks
    )

    np.save(emb_cache, doc_embeddings)
    with open(src_cache, "w") as f:
        json.dump(doc_sources, f)
    print(f"  Cached to {emb_cache}")

    return doc_sources, doc_embeddings


def paper_title(source: str, arxiv_meta: dict[str, dict]) -> str:
    """Look up paper title from arxiv metadata, falling back to filename."""
    filename = Path(source).name
    meta = arxiv_meta.get(filename) or arxiv_meta.get(source)
    if meta:
        return meta.get("title", filename)
    return filename


def paper_categories(source: str, arxiv_meta: dict[str, dict]) -> str:
    """Look up paper categories from arxiv metadata."""
    filename = Path(source).name
    meta = arxiv_meta.get(filename) or arxiv_meta.get(source)
    if meta:
        return ", ".join(meta.get("categories", []))
    return ""


def main():
    """Discover and print topic clusters using HDBSCAN."""
    parser = argparse.ArgumentParser(
        description="Discover topic clusters using HDBSCAN"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/01_baseline.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=15,
        help="Minimum papers to form a cluster (default: 15)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Core point density threshold (default: 5)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Representative papers per cluster (default: 5)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force reload from ChromaDB, ignoring cache",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached embeddings and exit",
    )
    args = parser.parse_args()

    if args.clear_cache:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.iterdir():
                f.unlink()
            CACHE_DIR.rmdir()
            print("Cache cleared")
        else:
            print("No cache to clear")
        return

    config = load_config(args.config)
    persist_dir = config.get("store", {}).get("persist_directory", "data/chroma")
    collection_name = config.get("experiment_name", "default")

    doc_dirs = config.get("data", {}).get("documents", ["data/raw"])
    arxiv_meta = {}
    for d in doc_dirs:
        arxiv_meta.update(load_arxiv_metadata(d))

    print(
        f"Connecting to ChromaDB at {persist_dir}, "
        f"collection '{collection_name}'"
    )
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)

    total_chunks = collection.count()
    print(f"Total chunks in collection: {total_chunks}")

    doc_sources, doc_embeddings = load_doc_embeddings(
        collection, total_chunks, collection_name, args.reload
    )
    n_docs = len(doc_sources)
    print(f"\n{n_docs} documents")
    print(
        f"Clustering with HDBSCAN "
        f"(min_cluster_size={args.min_cluster_size}, "
        f"min_samples={args.min_samples})...\n"
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(doc_embeddings)

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"Found {n_clusters} clusters, {n_noise} noise papers\n")

    # Print each cluster with representative papers
    for cluster_id in sorted(set(labels) - {-1}):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = doc_embeddings[cluster_indices]

        # Centroid = mean of cluster members
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(
            cluster_embeddings - centroid, axis=1
        )
        closest = np.argsort(distances)[: args.top_n]

        print(f"{'=' * 80}")
        print(f"CLUSTER {cluster_id + 1}  ({len(cluster_indices)} papers)")
        print(f"{'=' * 80}")

        for rank, idx in enumerate(closest, 1):
            source = doc_sources[cluster_indices[idx]]
            title = paper_title(source, arxiv_meta)
            cats = paper_categories(source, arxiv_meta)
            print(f"  {rank}. {title}")
            if cats:
                print(f"     [{cats}]")
        print()

    # Print noise papers
    if n_noise > 0:
        noise_indices = np.where(labels == -1)[0]
        print(f"{'=' * 80}")
        print(f"NOISE  ({n_noise} papers that don't fit any cluster)")
        print(f"{'=' * 80}")
        for idx in noise_indices[:10]:
            title = paper_title(doc_sources[idx], arxiv_meta)
            print(f"  - {title}")
        if n_noise > 10:
            print(f"  ... and {n_noise - 10} more")
        print()

    # Summary
    print(f"{'=' * 80}")
    print("CLUSTER SIZE SUMMARY")
    print(f"{'=' * 80}")
    for cluster_id in sorted(set(labels) - {-1}):
        count = int(np.sum(labels == cluster_id))
        bar = "#" * (count // 2)
        print(f"  Cluster {cluster_id + 1:2d}: {count:4d} papers {bar}")
    if n_noise:
        bar = "#" * (n_noise // 2)
        print(f"  Noise      : {n_noise:4d} papers {bar}")


if __name__ == "__main__":
    main()
