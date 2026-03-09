.PHONY: lint format test-unit test-integration eval test security-test clean sync \
       deploy-local deploy-aws deploy-gcp rollback sync-plan

sync:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

test-unit:
	uv run pytest tests/unit -v -m unit

test-integration:
	uv run pytest tests/integration -v -m integration

eval:
	uv run pytest tests/eval -v -m eval

test:
	uv run pytest tests/ -v

security-test:
	uv run pytest tests/security/ -v -m security

deploy-local:
	uv run python scripts/deploy.py --target local

deploy-aws:
	uv run python scripts/deploy.py --target aws

deploy-gcp:
	uv run python scripts/deploy.py --target gcp

rollback:
	uv run python scripts/deploy.py --rollback last-approved

sync-plan:
	@cp ../rag-learning-lab-plan.md docs/PLAN.md
	@if git diff --quiet docs/PLAN.md; then \
		echo "Plan is up to date"; \
	else \
		git add docs/PLAN.md && \
		git commit -m "Sync plan from master plan file" && \
		echo "Plan synced and committed"; \
	fi

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist/ build/ htmlcov/ .coverage
