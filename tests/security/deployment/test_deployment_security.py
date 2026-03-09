"""Tests for deployment security verification.

Post-deployment checks for authentication, rate limiting, input validation,
debug endpoint exposure, TLS enforcement, and CORS policy.

These tests require a deployed FastAPI application. Set the DEPLOY_URL
environment variable to run them:
    DEPLOY_URL=http://localhost:8000 uv run pytest tests/security/deployment/ -v
"""

import os

import pytest
import requests

DEPLOY_URL = os.environ.get("DEPLOY_URL")
SKIP_REASON = "Set DEPLOY_URL env var to run deployment security tests"


@pytest.mark.security
@pytest.mark.skipif(not DEPLOY_URL, reason=SKIP_REASON)
def test_api_auth_required():
    """Hitting /query and /ingest without auth must return 401."""
    for endpoint in ["/query", "/ingest"]:
        response = requests.post(
            f"{DEPLOY_URL}{endpoint}",
            json={"question": "test"},
            timeout=10,
        )
        assert response.status_code == 401, (
            f"{endpoint} returned {response.status_code} without auth, expected 401"
        )


@pytest.mark.security
@pytest.mark.skipif(not DEPLOY_URL, reason=SKIP_REASON)
def test_rate_limiting():
    """Sending 100 requests in rapid succession must trigger rate limiting (429)."""
    responses = [
        requests.post(
            f"{DEPLOY_URL}/query",
            json={"question": "test"},
            timeout=10,
        )
        for _ in range(100)
    ]
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes, "Rate limiting not triggered after 100 rapid requests"


@pytest.mark.security
@pytest.mark.skipif(not DEPLOY_URL, reason=SKIP_REASON)
def test_input_size_limits():
    """A 10MB query payload must be rejected (413 or 422), not crash the server."""
    huge_payload = {"question": "x" * 10_000_000}
    response = requests.post(
        f"{DEPLOY_URL}/query",
        json=huge_payload,
        timeout=30,
    )
    assert response.status_code in [413, 422], (
        f"Expected 413/422 for oversized payload, got {response.status_code}"
    )


@pytest.mark.security
@pytest.mark.skipif(not DEPLOY_URL, reason=SKIP_REASON)
def test_no_debug_endpoints():
    """Debug endpoints must return 404 or require auth in production."""
    for path in ["/docs", "/debug", "/metrics"]:
        response = requests.get(f"{DEPLOY_URL}{path}", timeout=10)
        assert response.status_code in [401, 403, 404], (
            f"{path} returned {response.status_code}, expected 401/403/404"
        )


@pytest.mark.security
@pytest.mark.skipif(not DEPLOY_URL, reason=SKIP_REASON)
def test_tls_enforcement():
    """HTTP requests on cloud deployments must redirect to HTTPS or be refused."""
    if not DEPLOY_URL or DEPLOY_URL.startswith("http://localhost"):
        pytest.skip("TLS test only applies to cloud deployments")
    http_url = DEPLOY_URL.replace("https://", "http://")
    response = requests.get(http_url, timeout=10, allow_redirects=False)
    assert response.status_code in [301, 302, 308], (
        f"HTTP request not redirected to HTTPS, got {response.status_code}"
    )


@pytest.mark.security
@pytest.mark.skipif(not DEPLOY_URL, reason=SKIP_REASON)
def test_cors_policy():
    """Cross-origin requests from unauthorized domains must be rejected."""
    response = requests.options(
        f"{DEPLOY_URL}/query",
        headers={
            "Origin": "https://evil-site.example.com",
            "Access-Control-Request-Method": "POST",
        },
        timeout=10,
    )
    # The response should not include the evil origin in allowed origins
    allowed = response.headers.get("Access-Control-Allow-Origin", "")
    assert "evil-site.example.com" not in allowed
