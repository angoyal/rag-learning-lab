"""Tests for deployment scripts — pure function tests that don't need Docker."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
from deploy import (  # noqa: E402
    DeploymentRecord,
    generate_version,
    load_deployment_history,
    save_deployment_record,
    show_history,
)
from post_deploy_checks import (  # noqa: E402
    ApprovalResult,
    CheckResult,
    check_metric,
    check_no_regression,
)

# -- Deploy script tests --


@pytest.mark.unit
def test_generate_version_format():
    version = generate_version()
    # CalVer format: YYYY.MM.DD-HHMMSS
    parts = version.split("-")
    assert len(parts) == 2
    date_parts = parts[0].split(".")
    assert len(date_parts) == 3
    assert len(parts[1]) == 6  # HHMMSS


@pytest.mark.unit
def test_save_and_load_history(tmp_path, monkeypatch):
    history_path = tmp_path / "history.yaml"
    monkeypatch.setattr("deploy.HISTORY_PATH", history_path)

    record = DeploymentRecord(
        version="2026.03.08-120000",
        target="local",
        git_sha="abc1234",
        timestamp="2026-03-08T12:00:00Z",
        status="active",
        approval="passed",
        config="configs/experiments/01_baseline.yaml",
        notes="Test deployment",
    )
    save_deployment_record(record)
    loaded = load_deployment_history()
    assert len(loaded) == 1
    assert loaded[0].version == "2026.03.08-120000"
    assert loaded[0].target == "local"
    assert loaded[0].approval == "passed"


@pytest.mark.unit
def test_load_empty_history(tmp_path, monkeypatch):
    history_path = tmp_path / "history.yaml"
    monkeypatch.setattr("deploy.HISTORY_PATH", history_path)
    loaded = load_deployment_history()
    assert loaded == []


@pytest.mark.unit
def test_show_history_empty(tmp_path, monkeypatch, capsys):
    history_path = tmp_path / "history.yaml"
    monkeypatch.setattr("deploy.HISTORY_PATH", history_path)
    show_history()
    captured = capsys.readouterr()
    assert "No deployment history" in captured.out


@pytest.mark.unit
def test_multiple_records(tmp_path, monkeypatch):
    history_path = tmp_path / "history.yaml"
    monkeypatch.setattr("deploy.HISTORY_PATH", history_path)

    for i in range(3):
        record = DeploymentRecord(
            version=f"2026.03.0{i+1}-120000",
            target="local",
            git_sha=f"sha{i}",
            timestamp=f"2026-03-0{i+1}T12:00:00Z",
            status="active" if i == 2 else "superseded",
            approval="passed",
            config="config.yaml",
            notes=f"Deploy {i+1}",
        )
        save_deployment_record(record)
    loaded = load_deployment_history()
    assert len(loaded) == 3


# -- Post-deploy checks tests --


@pytest.mark.unit
def test_check_metric_passes():
    result = check_metric({"faithfulness": 0.85}, "faithfulness", min_value=0.80)
    assert result.passed is True
    assert "0.850" in result.message


@pytest.mark.unit
def test_check_metric_fails():
    result = check_metric({"faithfulness": 0.70}, "faithfulness", min_value=0.80)
    assert result.passed is False


@pytest.mark.unit
def test_check_metric_missing():
    result = check_metric({}, "faithfulness", min_value=0.80)
    assert result.passed is False
    assert "not found" in result.message


@pytest.mark.unit
def test_check_no_regression_passes():
    current = {"faithfulness": 0.85, "relevancy": 0.90}
    previous = {"faithfulness": 0.80, "relevancy": 0.85}
    result = check_no_regression(current, previous)
    assert result.passed is True


@pytest.mark.unit
def test_check_no_regression_fails():
    current = {"faithfulness": 0.75}
    previous = {"faithfulness": 0.80}
    result = check_no_regression(current, previous)
    assert result.passed is False
    assert "Regressions" in result.message


@pytest.mark.unit
def test_approval_result_all_pass():
    results = [
        CheckResult("check1", True, "ok"),
        CheckResult("check2", True, "ok"),
    ]
    approval = ApprovalResult(passed=all(r.passed for r in results), details=results)
    assert approval.passed is True


@pytest.mark.unit
def test_approval_result_one_fail():
    results = [
        CheckResult("check1", True, "ok"),
        CheckResult("check2", False, "failed"),
    ]
    approval = ApprovalResult(passed=all(r.passed for r in results), details=results)
    assert approval.passed is False
