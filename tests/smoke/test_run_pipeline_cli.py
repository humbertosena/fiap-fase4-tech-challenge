"""Smoke do subcomando ``run-pipeline``: o lote ruim bloqueia e o bom passa."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.observability.logger import PipelineEvent, read_events
from credit_sentinel.observability.pipeline_runner import AuditReport

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke

ARGS = ["--quick", "--no-mlflow"]


def _events(settings: Settings, window: str) -> dict[str, PipelineEvent]:
    events = read_events(settings.logs_dir / "pipeline_events.jsonl")
    return {e.stage: e for e in events if e.window == window and e.status != "STARTED"}


def test_bad_batch_is_blocked_and_the_next_stages_are_skipped(
    project_clean: Settings,
) -> None:
    assert main(["run-pipeline", "--window", "bad_batch", *ARGS]) == 0
    stages = _events(project_clean, "bad_batch")
    assert stages["ingestion"].status == "SUCCESS"
    assert stages["contract_validation"].status == "BLOCKED"
    assert stages["contract_validation"].rejected_records > 0
    assert stages["inference"].status == "SKIPPED"
    assert stages["drift_detection"].status == "SKIPPED"
    assert stages["slo_evaluation"].status == "SUCCESS"
    alerts = stages["slo_evaluation"].details["alerts"]
    assert [a["severity"] for a in alerts] == ["P1"]


@pytest.mark.usefixtures("project_clean")
def test_strict_mode_turns_the_blocked_batch_into_exit_code_2() -> None:
    assert main(["run-pipeline", "--window", "bad_batch", "--strict", *ARGS]) == 2


def test_healthy_window_runs_every_stage_with_success(project_clean: Settings) -> None:
    assert main(["run-pipeline", "--window", "mes_1", "--strict", *ARGS]) == 0
    stages = _events(project_clean, "mes_1")
    for stage in ("ingestion", "contract_validation", "inference", "drift_detection"):
        assert stages[stage].status == "SUCCESS", stage
    assert stages["inference"].records_processed == 500
    assert stages["tracking"].status == "SKIPPED"
    alerts = stages["slo_evaluation"].details["alerts"]
    assert [a["severity"] for a in alerts] == ["P4"]


def test_quick_runs_write_the_audit_outside_the_versioned_reports(
    project_clean: Settings,
) -> None:
    assert main(["run-pipeline", "--window", "all", *ARGS]) == 0
    audit = project_clean.logs_dir / "last_run" / "slo_audit_report.json"
    report = AuditReport.model_validate_json(audit.read_text("utf-8"))
    assert [w.window for w in report.windows] == [
        "bad_batch",
        "mes_1",
        "mes_2",
        "mes_3_crise",
    ]
    assert not (project_clean.reports_dir / "slo_audit_report.json").exists()


def test_missing_model_points_to_train(
    project_clean: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    (project_clean.models_dir / "baseline" / "model.joblib").unlink()
    capsys.readouterr()
    assert main(["run-pipeline", "--window", "mes_1", *ARGS]) == 1
    assert "train" in capsys.readouterr().out
