"""Logger JSON estruturado: campos obrigatórios, redação de PII e ``StageTimer``."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from credit_sentinel.contracts.validator import (
    CircuitBreakerOpenError,
    ContractViolationError,
    ValidationReport,
)
from credit_sentinel.observability.logger import (
    REQUIRED_FIELDS,
    PipelineEvent,
    PipelineLogger,
    StageTimer,
    get_pipeline_logger,
    read_events,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit

CLIENT_ID = "CLI-0123456789"
CPF = "123.456.789-09"
EMAIL = "maria.silva@banco.com.br"


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "pipeline_events.jsonl"


@pytest.fixture
def logger(log_path: Path) -> Iterator[PipelineLogger]:
    with get_pipeline_logger("run-1", "baseline-1.0.0", log_path) as pipeline_logger:
        yield pipeline_logger


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def _blocked_report(reason: str) -> ValidationReport:
    return ValidationReport(
        batch_name="bad_batch",
        run_id="val-1",
        validated_at=datetime(2026, 10, 22, tzinfo=UTC),
        schema_name="credit_scoring_inference",
        total_records=600,
        valid_records=438,
        rejected_records=162,
        rejected_ratio=0.27,
        decision="BLOCKED",
        blocked_reason=reason,
        violations=[],
        sample_failures=[],
    )


def test_every_line_is_json_with_the_eight_required_fields(
    logger: PipelineLogger, log_path: Path
) -> None:
    logger.emit(stage="ingestion", status="STARTED", window="mes_1")
    logger.emit(stage="ingestion", status="SUCCESS", records_processed=5000)
    for line in _lines(log_path):
        assert set(REQUIRED_FIELDS) <= set(line)
        event = PipelineEvent.model_validate(line)
        assert event.run_id == "run-1"
        assert event.model_version == "baseline-1.0.0"


def test_required_fields_are_exactly_the_contract_of_the_spec() -> None:
    assert REQUIRED_FIELDS == (
        "timestamp",
        "run_id",
        "stage",
        "status",
        "records_processed",
        "rejected_records",
        "latency_ms",
        "model_version",
    )


def test_event_rejects_unknown_stage_status_and_extra_fields() -> None:
    base = {
        "timestamp": "2026-10-22T20:15:00Z",
        "run_id": "r",
        "stage": "ingestion",
        "status": "SUCCESS",
        "records_processed": 1,
        "rejected_records": 0,
        "latency_ms": 3,
        "model_version": "v",
    }
    PipelineEvent.model_validate(base)
    for change in ({"stage": "deploy"}, {"status": "DONE"}, {"extra": 1}):
        with pytest.raises(ValidationError):
            PipelineEvent.model_validate({**base, **change})


def test_timestamp_is_utc_and_never_goes_backwards(
    logger: PipelineLogger, log_path: Path
) -> None:
    for _ in range(3):
        logger.emit(stage="inference", status="STARTED")
    stamps = [PipelineEvent.model_validate(x).timestamp for x in _lines(log_path)]
    assert all(s.utcoffset().total_seconds() == 0 for s in stamps)
    assert stamps == sorted(stamps)


def test_source_date_epoch_anchors_the_timestamps(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch = 1792542000  # 2026-10-20 21:00 -03:00
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(epoch))
    with get_pipeline_logger("r", "v", log_path) as pipeline_logger:
        pipeline_logger.emit(stage="ingestion", status="STARTED")
    first = PipelineEvent.model_validate(_lines(log_path)[0]).timestamp
    assert abs(first.timestamp() - epoch) < 5


def test_client_ids_cpf_and_emails_are_redacted_in_message_and_details(
    logger: PipelineLogger, log_path: Path
) -> None:
    logger.emit(
        stage="contract_validation",
        status="FAILED",
        message=f"falha no {CLIENT_ID} com cpf {CPF}",
        details={
            "contato": EMAIL,
            "amostras": [CLIENT_ID, {"obs": f"titular {CPF}"}],
            "id_cliente": CLIENT_ID,
            "cpf": CPF,
        },
    )
    raw = log_path.read_text("utf-8")
    for secret in (CLIENT_ID, CPF, EMAIL):
        assert secret not in raw
    event = PipelineEvent.model_validate(_lines(log_path)[0])
    assert event.message == "falha no *** com cpf ***"
    assert event.details["id_cliente"] == "***"
    assert event.details["cpf"] == "***"
    assert event.details["amostras"] == ["***", {"obs": "titular ***"}]


def test_numbers_and_booleans_in_details_are_preserved(
    logger: PipelineLogger, log_path: Path
) -> None:
    logger.emit(
        stage="drift_detection",
        status="SUCCESS",
        details={"psi": 0.66, "drift": True, "n": 3, "none": None},
    )
    event = PipelineEvent.model_validate(_lines(log_path)[0])
    assert event.details == {"psi": 0.66, "drift": True, "n": 3, "none": None}


def test_events_are_also_written_to_stderr(
    log_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with get_pipeline_logger("r", "v", log_path) as pipeline_logger:
        pipeline_logger.emit(stage="ingestion", status="STARTED", message=CLIENT_ID)
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    assert CLIENT_ID not in err[0]
    assert json.loads(err[0])["status"] == "STARTED"


def test_log_file_is_appended_across_runs(log_path: Path) -> None:
    for run_id in ("a", "b"):
        with get_pipeline_logger(run_id, "v", log_path) as pipeline_logger:
            pipeline_logger.emit(stage="ingestion", status="STARTED")
    assert [e.run_id for e in read_events(log_path)] == ["a", "b"]


def test_closing_twice_is_harmless(log_path: Path) -> None:
    pipeline_logger = get_pipeline_logger("r", "v", log_path)
    pipeline_logger.close()
    pipeline_logger.close()


def test_stage_timer_emits_started_and_success_with_latency(
    logger: PipelineLogger, log_path: Path
) -> None:
    with StageTimer(logger, "ingestion", "mes_1") as timer:
        timer.records_processed = 5000
        timer.details = {"arquivo": "mes_1.parquet"}
    started, success = read_events(log_path)
    assert (started.status, success.status) == ("STARTED", "SUCCESS")
    assert success.window == "mes_1"
    assert success.records_processed == 5000
    assert success.latency_ms >= 0
    assert timer.latency_ms == success.latency_ms
    assert timer.status == "SUCCESS"
    assert success.details == {"arquivo": "mes_1.parquet"}


def _fail_inside_stage(logger: PipelineLogger) -> None:
    with StageTimer(logger, "tracking", "mes_2"):
        msg = "disco cheio"
        raise RuntimeError(msg)


def test_stage_timer_maps_a_generic_exception_to_failed_and_reraises(
    logger: PipelineLogger, log_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="disco cheio"):
        _fail_inside_stage(logger)
    failed = read_events(log_path)[-1]
    assert failed.status == "FAILED"
    assert "RuntimeError" in (failed.message or "")
    assert "disco cheio" in (failed.message or "")


def _block_inside_stage(logger: PipelineLogger, error: type[Exception]) -> None:
    report = _blocked_report("violação estrutural (R01, R02): lote bloqueado")
    with StageTimer(logger, "contract_validation", "bad_batch") as timer:
        timer.rejected_records = 162
        raise error(report)


@pytest.mark.parametrize("error", [ContractViolationError, CircuitBreakerOpenError])
def test_stage_timer_maps_contract_errors_to_blocked(
    error: type[Exception], logger: PipelineLogger, log_path: Path
) -> None:
    with pytest.raises(error):
        _block_inside_stage(logger, error)
    blocked = read_events(log_path)[-1]
    assert blocked.status == "BLOCKED"
    assert blocked.rejected_records == 162
    assert "violação estrutural" in (blocked.message or "")


def test_stage_timer_skip_emits_skipped_without_started(
    logger: PipelineLogger, log_path: Path
) -> None:
    StageTimer(logger, "inference", "bad_batch").skip("lote bloqueado antes")
    (event,) = read_events(log_path)
    assert event.status == "SKIPPED"
    assert event.stage == "inference"
    assert event.message == "lote bloqueado antes"
    assert event.latency_ms == 0


def test_stage_timer_status_tracks_the_last_emitted_event(
    logger: PipelineLogger,
) -> None:
    timer = StageTimer(logger, "inference", "mes_2")
    assert timer.status is None
    timer.skip("sem insumo")
    assert timer.status == "SKIPPED"
