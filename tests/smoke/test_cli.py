"""Smoke tests da CLI ``credit-sentinel``."""

from __future__ import annotations

import pytest

from credit_sentinel import __version__
from credit_sentinel.cli import main
from credit_sentinel.utils.env_check import CheckResult, run_checks, summarize

pytestmark = pytest.mark.smoke


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"credit-sentinel {__version__}"


def test_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    assert "env-check" in capsys.readouterr().out


def test_without_subcommand_prints_help_and_returns_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_unknown_subcommand_returns_nonzero() -> None:
    assert main(["nao-existe"]) != 0


def test_env_check_reports_python_and_uv(capsys: pytest.CaptureFixture[str]) -> None:
    main(["env-check"])
    out = capsys.readouterr().out
    assert "python" in out
    assert "uv" in out


def test_required_check_failure_turns_summary_into_error() -> None:
    results = [
        CheckResult(name="python", ok=False, detail="3.11", required=True),
        CheckResult(name="docker", ok=False, detail="ausente", required=False),
    ]
    assert summarize(results) == 1
    assert summarize(results[1:]) == 0


def test_run_checks_covers_python_uv_docker_and_port() -> None:
    names = [result.name for result in run_checks()]
    assert names == ["python", "uv", "docker", "porta 5001"]
