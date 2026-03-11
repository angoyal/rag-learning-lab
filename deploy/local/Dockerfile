# Multi-purpose image for RAG Learning Lab.
# Use different entrypoints for query service vs batch jobs.
#
#   Query service:  docker run -p 8000:8000 rag-lab serve
#   Ingest job:     docker run rag-lab ingest --config configs/experiments/01_baseline.yaml
#   Crawl job:      docker run rag-lab crawl --query "RAG" --max-papers 10

# --- Builder stage: install deps with build tools ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# --- Runtime stage: clean image without build tools ---
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /usr/local/bin/uv

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY crawlers/ crawlers/
COPY configs/ configs/
COPY deploy/local/entrypoint.sh /entrypoint.sh

# Create data directories and set ownership
RUN mkdir -p data/raw data/chroma && \
    chown -R appuser:appuser /app && \
    chmod +x /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
