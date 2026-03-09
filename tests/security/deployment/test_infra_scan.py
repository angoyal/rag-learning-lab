"""Tests for infrastructure security scanning.

Wrappers around pip-audit and gitleaks to run as pytest tests in CI.
These tests verify that security scanning tools produce clean results.
"""

import shutil
import subprocess

import pytest


@pytest.mark.security
def test_no_dependency_cves():
    """All installed Python packages must be free of known CVEs."""
    if not shutil.which("pip-audit"):
        pytest.skip("pip-audit not installed (uv add --dev pip-audit)")
    result = subprocess.run(
        ["pip-audit", "--strict"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"pip-audit found vulnerabilities:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.security
def test_no_secrets_in_repo():
    """No API keys, tokens, or passwords committed to git."""
    if not shutil.which("gitleaks"):
        pytest.skip("gitleaks not installed (brew install gitleaks)")
    result = subprocess.run(
        ["gitleaks", "detect", "--source", ".", "--no-banner"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"gitleaks found secrets:\n{result.stdout}\n{result.stderr}"
    )
