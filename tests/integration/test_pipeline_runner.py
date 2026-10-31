"""Pipeline ponta a ponta: bloqueio, drift, hooks de tracking e privacidade dos logs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.contracts.privacy_sanitizer import scan_paths
from credit_sentinel.drift.detection import MissingInputError
from credit_sentinel.observability.logger import PipelineEvent, read_events
from credit_sentinel.observability.pipeline_runner import (
    WINDOW_ORDER,
    AuditReport,
    WindowRunResult,
    run_pipeline,
)
from credit_sentinel.utils.io import read_parquet, write_parquet

if TYPE_CHECKING:
    from pathlib import Path

    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.integration

CLIENT_ID = "CLI-0123456789"


class _ExplodingHook:
    def on_window_complete(
        self, result: WindowRunResult, artifacts: dict[str, Path]
    ) -> None:
        del result, artifacts
        msg = f"mlflow indisponível para {CLIENT_ID}"
        raise RuntimeError(msg)


class _RecordingHook:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, bool]]] = []

    def on_window_complete(
        self, result: WindowRunResult, artifacts: dict[str, Path]
    ) -> None:
        present = {name: path.exists() for name, path in artifacts.items()}
        self.calls.append((result.window, present))


def _run(
    settings: Settings, windows: tuple[str, ...] = WINDOW_ORDER, **options: object
) -> AuditReport:
    return run_pipeline(list(windows), settings=settings, quick=True, **options)  # type: ignore[arg-type]


def _by_window(report: AuditReport) -> dict[str, WindowRunResult]:
    return {w.window: w for w in report.windows}


def _events(settings: Settings) -> list[PipelineEvent]:
    return read_events(settings.logs_dir / "pipeline_events.jsonl")


def _severities(result: WindowRunResult) -> list[str]:
    return [alert.severity for alert in result.slo.alerts]


def test_bad_batch_is_blocked_with_p1_and_skipped_stages(
    project_clean: Settings,
) -> None:
    result = _by_window(_run(project_clean, ("bad_batch",)))["bad_batch"]
    assert result.decision == "BLOCKED"
    assert result.validation.decision == "BLOCKED"
    assert _severities(result) == ["P1"]
    assert result.stage_status == {
        "ingestion": "SUCCESS",
        "contract_validation": "BLOCKED",
        "inference": "SKIPPED",
        "drift_detection": "SKIPPED",
        "slo_evaluation": "SUCCESS",
        "tracking": "SKIPPED",
    }
    assert (result.drift, result.performance, result.predictions) == (None, None, None)
    assert result.slo.remaining_budget < 0


def test_stable_window_succeeds_with_only_the_informative_alert(
    project_clean: Settings,
) -> None:
    result = _by_window(_run(project_clean, ("mes_1",)))["mes_1"]
    assert result.decision == "SUCCESS"
    assert _severities(result) == ["P4"]
    assert result.slo.remaining_budget > 0.5
    assert result.performance is not None
    assert result.performance.auc_delta_vs_control is None


def test_crisis_window_raises_p2_with_the_retraining_action(
    project_clean: Settings,
) -> None:
    report = _run(project_clean, ("mes_1", "mes_3_crise"))
    crisis = _by_window(report)["mes_3_crise"]
    p2 = [a for a in crisis.slo.alerts if a.severity == "P2"]
    assert [a.slo_id for a in p2] == ["SLO2_estabilidade_psi"]
    assert "retreino" in p2[0].recommended_action.lower()
    assert crisis.drift is not None
    assert "utilizacao_credito" in crisis.drift.psi_critical_features
    assert crisis.performance is not None
    assert crisis.performance.auc_delta_vs_control is not None


def test_synthetic_drift_in_a_top_feature_also_raises_p2(
    project_clean: Settings,
) -> None:
    path = project_clean.production_dir / "mes_1.parquet"
    shifted = read_parquet(path)
    shifted["utilizacao_credito"] = 0.7 + 0.3 * shifted["utilizacao_credito"]
    write_parquet(shifted, path)
    result = _by_window(_run(project_clean, ("mes_1",)))["mes_1"]
    assert "P2" in _severities(result)
    assert result.slo.alerts[0].slo_id == "SLO2_estabilidade_psi"


def test_moderate_window_raises_p3_and_stays_without_p2(
    project_clean: Settings,
) -> None:
    result = _by_window(_run(project_clean, ("mes_1", "mes_2")))["mes_2"]
    assert "P3" in _severities(result)
    assert "P2" not in _severities(result)


def test_audit_report_is_consistent_and_round_trips_through_json(
    project_clean: Settings,
) -> None:
    report = _run(project_clean)
    assert AuditReport.model_validate_json(report.model_dump_json()) == report
    assert [w.window for w in report.windows] == list(WINDOW_ORDER)
    totals = report.totals
    assert (totals["windows"], totals["success"], totals["blocked"]) == (4, 3, 1)
    assert totals["alerts_by_severity"]["P1"] == 1
    assert "bad_batch" in totals["windows_over_budget"]
    assert [a.severity for a in report.alerts] == sorted(
        a.severity for a in report.alerts
    )
    assert report.reference_auc == pytest.approx(0.66, abs=0.15)
    assert all(w.run_id == report.run_id for w in report.windows)


def test_log_has_success_blocked_and_alerts_with_the_eight_required_fields(
    project_clean: Settings,
) -> None:
    report = _run(project_clean)
    events = _events(project_clean)
    assert {e.run_id for e in events} == {report.run_id}
    pairs = {(e.window, e.stage, e.status) for e in events}
    assert ("mes_1", "contract_validation", "SUCCESS") in pairs
    assert ("bad_batch", "contract_validation", "BLOCKED") in pairs
    assert ("bad_batch", "inference", "SKIPPED") in pairs
    final = {
        e.window: e
        for e in events
        if e.stage == "slo_evaluation" and e.status != "STARTED"
    }
    assert final["mes_3_crise"].details["alerts"][0]["severity"] == "P2"
    assert "P2" in (final["mes_3_crise"].message or "")


def test_logs_never_carry_a_client_id_or_other_pii(project_clean: Settings) -> None:
    _run(project_clean, hook_factory=lambda _run, _version: [_ExplodingHook()])
    raw = (project_clean.logs_dir / "pipeline_events.jsonl").read_text("utf-8")
    assert CLIENT_ID not in raw
    assert "mlflow indisponível para ***" in raw
    assert scan_paths([project_clean.logs_dir]) == []


def test_failing_hook_marks_tracking_failed_and_the_pipeline_keeps_going(
    project_clean: Settings,
) -> None:
    recorder = _RecordingHook()
    report = _run(
        project_clean,
        hook_factory=lambda _run, _version: [_ExplodingHook(), recorder],
    )
    assert len(report.windows) == 4
    for window in report.windows:
        assert window.stage_status["tracking"] == "FAILED"
        assert window.stage_status["slo_evaluation"] == "SUCCESS"
    assert [name for name, _ in recorder.calls] == list(WINDOW_ORDER)
    failed = [e for e in _events(project_clean) if e.status == "FAILED"]
    assert [e.stage for e in failed] == ["tracking"] * 4
    assert "ExplodingHook" in (failed[0].message or "")


def test_hooks_receive_the_window_artifacts(project_clean: Settings) -> None:
    recorder = _RecordingHook()
    _run(project_clean, ("mes_1",), hook_factory=lambda _r, _v: [recorder])
    ((window, present),) = recorder.calls
    assert window == "mes_1"
    assert present == {
        "validation_report": True,
        "slo_evaluation": True,
        "window_events": True,
    }


def test_successful_hook_leaves_tracking_success(project_clean: Settings) -> None:
    report = _run(
        project_clean, ("mes_1",), hook_factory=lambda _r, _v: [_RecordingHook()]
    )
    assert report.windows[0].stage_status["tracking"] == "SUCCESS"


def test_full_mode_runs_evidently_and_links_the_report(project_clean: Settings) -> None:
    report = run_pipeline(["mes_2"], settings=project_clean, quick=False)
    result = report.windows[0]
    assert result.drift_ref == "reports/evidently/mes_2_drift_report.html"
    assert (project_clean.project_root / result.drift_ref).exists()
    assert report.quick is False


def test_control_window_provides_the_auc_delta_versus_control(
    project_clean: Settings,
) -> None:
    report = _run(project_clean, ("mes_2",))
    performance = report.windows[0].performance
    assert performance is not None
    assert performance.auc_delta_vs_control is not None


def test_two_runs_have_different_run_ids_and_append_to_the_same_log(
    project_clean: Settings,
) -> None:
    first, second = _run(project_clean, ("mes_1",)), _run(project_clean, ("mes_1",))
    assert first.run_id != second.run_id
    assert {e.run_id for e in _events(project_clean)} == {first.run_id, second.run_id}


def test_source_date_epoch_anchors_the_audit_report(
    project_clean: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch = 1793400600  # 2026-10-30 19:50 -03:00
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(epoch))
    report = _run(project_clean, ("mes_1",))
    assert report.generated_at == datetime.fromtimestamp(epoch, UTC)
    assert all(
        abs(e.timestamp.timestamp() - epoch) < 60 for e in _events(project_clean)
    )


def test_unknown_window_is_rejected(project_clean: Settings) -> None:
    with pytest.raises(ValueError, match="mes_9"):
        _run(project_clean, ("mes_9",))


def test_missing_window_logs_the_failure_and_stops_the_run(
    project_clean: Settings,
) -> None:
    (project_clean.production_dir / "mes_2.parquet").unlink()
    with pytest.raises(MissingInputError, match="mes_2"):
        _run(project_clean, ("mes_2",))
    failed = [e for e in _events(project_clean) if e.status == "FAILED"]
    assert [(e.window, e.stage) for e in failed] == [("mes_2", "ingestion")]
