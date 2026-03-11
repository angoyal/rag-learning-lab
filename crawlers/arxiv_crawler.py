"""arXiv paper crawler using the arxiv.py package.

Downloads research papers as PDFs and saves metadata to a YAML sidecar file.

Usage:
    uv run python crawlers/arxiv_crawler.py --query "retrieval augmented generation" --max-papers 10
    uv run python crawlers/arxiv_crawler.py --query "cat:cs.IR" --max-papers 5
    uv run python crawlers/arxiv_crawler.py \
        --query "retrieval augmented generation" \
        --query "dense passage retrieval" \
        --max-papers 5
    uv run python crawlers/arxiv_crawler.py --query "RAG" --max-papers 10 --sort-by date
"""

import argparse
import urllib.error
from pathlib import Path

import arxiv
import yaml

DATA_DIR = Path("data/raw")
METADATA_FILE = DATA_DIR / "arxiv_metadata.yaml"

SORT_CRITERIA = {
    "relevance": arxiv.SortCriterion.Relevance,
    "date": arxiv.SortCriterion.LastUpdatedDate,
    "submitted": arxiv.SortCriterion.SubmittedDate,
}


def short_id(result: arxiv.Result) -> str:
    """Extract short arXiv ID (e.g. '2312.10997v1') from a result's entry_id."""
    return result.entry_id.split("/abs/")[-1]


def load_metadata() -> dict:
    """Load existing metadata from the YAML sidecar file.

    Returns:
        Dict with a "papers" key containing a list of paper metadata dicts.
    """
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            data = yaml.safe_load(f)
            if data and "papers" in data:
                return data
    return {"papers": []}


def save_metadata(metadata: dict) -> None:
    """Save metadata to the YAML sidecar file.

    Args:
        metadata: Dict with a "papers" key containing paper metadata.
    """
    with open(METADATA_FILE, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main():
    """CLI entrypoint: parse args, search arXiv, download PDFs, save metadata."""
    parser = argparse.ArgumentParser(description="Download papers from arXiv")
    parser.add_argument(
        "--query", action="append", required=True, help="Search query (can be repeated)"
    )
    parser.add_argument(
        "--max-papers", type=int, default=5, help="Max papers per query (default: 5)"
    )
    parser.add_argument(
        "--sort-by",
        choices=["relevance", "date", "submitted"],
        default="relevance",
        help="Sort order (default: relevance)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    existing_ids = {p["arxiv_id"] for p in metadata["papers"]}

    client = arxiv.Client()

    for query in args.query:
        print(f"\nSearching: {query}")
        search = arxiv.Search(
            query=query,
            max_results=args.max_papers,
            sort_by=SORT_CRITERIA[args.sort_by],
        )

        results = list(client.results(search))
        print(f"  Found {len(results)} results")

        for result in results:
            paper_id = short_id(result)
            pdf_filename = f"{paper_id.replace('/', '_')}.pdf"

            if paper_id in existing_ids:
                print(f"  Skip (exists): {result.title[:80]}")
                continue

            pdf_path = DATA_DIR / pdf_filename
            if pdf_path.exists():
                print(f"  Skip (file exists): {pdf_filename}")
                existing_ids.add(paper_id)
                metadata["papers"].append({
                    "arxiv_id": paper_id,
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "pdf_file": pdf_filename,
                    "categories": result.categories,
                })
                continue

            print(f"  Downloading: {result.title[:80]}")
            try:
                result.download_pdf(dirpath=str(DATA_DIR), filename=pdf_filename)
            except urllib.error.HTTPError as e:
                print(f"  FAILED ({e.code} {e.reason}): {paper_id} — skipping")
                continue

            metadata["papers"].append({
                "arxiv_id": paper_id,
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_file": pdf_filename,
                "categories": result.categories,
            })
            existing_ids.add(paper_id)

    save_metadata(metadata)
    print(f"\nMetadata saved to {METADATA_FILE}")
    print(f"Total papers: {len(metadata['papers'])}")


if __name__ == "__main__":
    main()
