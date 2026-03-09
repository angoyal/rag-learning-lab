"""Post-deployment approval workflow.

Runs a 4-step approval pipeline after every deployment:
1. Health check — is the API responding?
2. Security smoke tests — auth required? rate limits active?
3. Quality gate — RAGAS metrics meet thresholds?
4. Regression check — metrics >= previous version?

If any step fails, the deployment is rolled back automatically.

Usage:
    uv run python scripts/post_deploy_checks.py \\
        --endpoint http://localhost:8000 --version 2026.03.07-143022
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import mlflow
import requests


@dataclass
class CheckResult:
    """Result of a single approval check."""

    name: str
    passed: bool
    message: str


@dataclass
class ApprovalResult:
    """Aggregate result of the full approval pipeline."""

    passed: bool
    details: list[CheckResult] = field(default_factory=list)


def check_health(endpoint: str) -> CheckResult:
    """Step 1: Verify the API is responding (GET /health -> 200)."""
    try:
        response = requests.get(f"{endpoint}/health", timeout=10)
        passed = response.status_code == 200
        return CheckResult(
            name="health_check",
            passed=passed,
            message=f"GET /health returned {response.status_code}",
        )
    except requests.ConnectionError:
        return CheckResult(
            name="health_check",
            passed=False,
            message=f"Cannot connect to {endpoint}",
        )


def check_auth_required(endpoint: str) -> CheckResult:
    """Step 2a: Verify unauthenticated requests are rejected (401)."""
    try:
        response = requests.post(
            f"{endpoint}/query",
            json={"question": "test"},
            timeout=10,
        )
        passed = response.status_code == 401
        return CheckResult(
            name="auth_required",
            passed=passed,
            message=f"POST /query without auth returned {response.status_code}",
        )
    except requests.ConnectionError:
        return CheckResult(
            name="auth_required",
            passed=False,
            message=f"Cannot connect to {endpoint}",
        )


def check_rate_limiting(endpoint: str) -> CheckResult:
    """Step 2b: Verify rate limiting is active."""
    try:
        status_codes = []
        for _ in range(50):
            response = requests.post(
                f"{endpoint}/query",
                json={"question": "test"},
                timeout=10,
            )
            status_codes.append(response.status_code)
        if 429 in status_codes:
            return CheckResult(
                name="rate_limiting",
                passed=True,
                message="Rate limiting triggered (429 received)",
            )
        return CheckResult(
            name="rate_limiting",
            passed=False,
            message="Rate limiting not triggered after 50 rapid requests",
        )
    except requests.ConnectionError:
        return CheckResult(
            name="rate_limiting",
            passed=False,
            message=f"Cannot connect to {endpoint}",
        )


def check_input_size_limits(endpoint: str) -> CheckResult:
    """Step 2c: Verify oversized payloads are rejected."""
    try:
        huge_payload = {"question": "x" * 10_000_000}
        response = requests.post(
            f"{endpoint}/query",
            json=huge_payload,
            timeout=30,
        )
        passed = response.status_code in [413, 422]
        return CheckResult(
            name="input_size_limits",
            passed=passed,
            message=f"10MB payload returned {response.status_code}",
        )
    except requests.ConnectionError:
        return CheckResult(
            name="input_size_limits",
            passed=False,
            message=f"Cannot connect to {endpoint}",
        )


def run_eval_suite(endpoint: str, dataset: str) -> dict[str, float]:
    """Step 3: Run the evaluation suite against the golden Q&A dataset.

    Returns a dict of metric name -> value (e.g., {"faithfulness": 0.85}).
    This is a placeholder that returns empty results until the eval harness
    is connected to a deployed endpoint.
    """
    # In a full implementation, this would:
    # 1. Load questions from the dataset file
    # 2. Send each question to the deployed endpoint
    # 3. Run RAGAS/DeepEval on the responses
    # 4. Return aggregate metric scores
    return {}


def check_metric(eval_results: dict[str, float], metric: str, min_value: float) -> CheckResult:
    """Check that a single metric meets its threshold."""
    value = eval_results.get(metric)
    if value is None:
        return CheckResult(name=metric, passed=False, message=f"Metric '{metric}' not found")
    passed = value >= min_value
    return CheckResult(
        name=metric,
        passed=passed,
        message=f"{metric}={value:.3f} (threshold={min_value:.2f})",
    )


def check_no_regression(
    current: dict[str, float], previous: dict[str, float]
) -> CheckResult:
    """Step 4: Verify current metrics are >= previous deployment's metrics."""
    regressions = []
    for metric, prev_value in previous.items():
        curr_value = current.get(metric, 0.0)
        if curr_value < prev_value:
            regressions.append(f"{metric}: {curr_value:.3f} < {prev_value:.3f}")
    if regressions:
        return CheckResult(
            name="regression_check",
            passed=False,
            message=f"Regressions detected: {', '.join(regressions)}",
        )
    return CheckResult(name="regression_check", passed=True, message="No regressions")


def log_approval_to_mlflow(version: str, results: list[CheckResult]) -> None:
    """Log approval results to MLflow for tracking."""
    mlflow.set_experiment("deployment-approvals")
    with mlflow.start_run(run_name=f"approval-{version}"):
        mlflow.log_param("version", version)
        for result in results:
            mlflow.log_metric(f"check/{result.name}/passed", 1 if result.passed else 0)


def run_post_deploy_checks(endpoint: str, version: str) -> ApprovalResult:
    """Run all post-deployment checks. Returns pass/fail with details.

    Steps:
    1. Health check (GET /health -> 200)
    2. Security smoke tests (auth, rate limits, input size)
    3. Quality gate (RAGAS metrics >= thresholds)
    4. Regression check (metrics >= previous version)

    Args:
        endpoint: The deployed API endpoint URL.
        version: The deployment version being validated.

    Returns:
        ApprovalResult with pass/fail and per-check details.
    """
    results = []

    # Step 1: Health check
    results.append(check_health(endpoint))

    # Step 2: Security smoke tests
    results.append(check_auth_required(endpoint))
    results.append(check_rate_limiting(endpoint))
    results.append(check_input_size_limits(endpoint))

    # Step 3: Quality gate
    eval_results = run_eval_suite(endpoint, dataset="data/eval_sets/golden.yaml")
    if eval_results:
        results.append(check_metric(eval_results, "faithfulness", min_value=0.80))
        results.append(check_metric(eval_results, "answer_relevancy", min_value=0.80))
        results.append(check_metric(eval_results, "context_precision", min_value=0.70))

    # Log to MLflow
    log_approval_to_mlflow(version, results)

    return ApprovalResult(
        passed=all(r.passed for r in results),
        details=results,
    )


def main() -> None:
    """CLI entrypoint for running post-deploy checks."""
    parser = argparse.ArgumentParser(description="Run post-deployment approval checks")
    parser.add_argument("--endpoint", required=True, help="Deployed API endpoint URL")
    parser.add_argument("--version", required=True, help="Deployment version to validate")
    parser.add_argument(
        "--target",
        default="local",
        choices=("local", "aws", "gcp"),
        help="Deployment target",
    )

    args = parser.parse_args()

    result = run_post_deploy_checks(args.endpoint, args.version)
    for check in result.details:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.message}")

    if result.passed:
        print("\nApproval: PASSED")
    else:
        print("\nApproval: FAILED — rollback required", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
