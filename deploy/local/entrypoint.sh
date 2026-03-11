#!/bin/bash
set -e

case "$1" in
    serve)
        exec uv run uvicorn scripts.ask:app --host 0.0.0.0 --port 8000
        ;;
    ingest)
        shift
        exec uv run python scripts/ingest_docs.py "$@"
        ;;
    crawl)
        shift
        exec uv run python crawlers/arxiv_crawler.py "$@"
        ;;
    *)
        echo "Unknown command: $1. Use: serve, ingest, or crawl" >&2
        exit 1
        ;;
esac
