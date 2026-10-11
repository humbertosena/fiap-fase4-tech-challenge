"""Prova de bloqueio com os datasets versionados (referência e lote corrompido)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.contracts.privacy_sanitizer import scan_paths
from credit_sentinel.contracts.schema import (
    build_inference_schema,
    build_reference_schema,
)
from credit_sentinel.contracts.validator import (
    BatchContext,
    ContractViolationError,
    ValidationPolicy,
    ValidationReport,
    validate_batch,
)
from credit_sentinel.data.corruptor import ANOMALY_PLAN
from credit_sentinel.settings import Settings
from credit_sentinel.utils.io import read_json, read_parquet

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

SETTINGS = Settings(_env_file=None)
BAD_BATCH = SETTINGS.corrupted_dir / "bad_batch.parquet"
REFERENCE = SETTINGS.reference_dir / "credit_ref.parquet"
VALIDATION_REPORTS = SETTINGS.reports_dir / "validation"


@pytest.fixture(scope="module")
def bad_report(tmp_path_factory: pytest.TempPathFactory) -> ValidationReport:
    context = BatchContext(
        batch_name="bad_batch",
        run_id="prova",
        quarantine_dir=tmp_path_factory.mktemp("quarentena"),
    )
    with pytest.raises(ContractViolationError) as raised:
        validate_batch(
            read_parquet(BAD_BATCH),
            schema=build_inference_schema(),
            policy=ValidationPolicy(),
            context=context,
        )
    return raised.value.report


def _failures(report: ValidationReport, rule_id: str, column: str | None) -> int:
    return sum(
        v.n_failures
        for v in report.violations
        if v.rule_id == rule_id
        and v.column == column
        and not v.check.startswith("dtype")
    )


def test_bad_batch_is_blocked_by_structural_rules(bad_report: ValidationReport) -> None:
    assert bad_report.decision == "BLOCKED"
    assert "R01" in (bad_report.blocked_reason or "")
    assert "R02" in (bad_report.blocked_reason or "")


def test_inventory_covers_every_injected_anomaly(bad_report: ValidationReport) -> None:
    found = {v.rule_id for v in bad_report.violations}
    assert {spec.rule_id for spec in ANOMALY_PLAN} <= found


@pytest.mark.parametrize(
    ("rule_id", "column", "expected"),
    [
        ("R04", "idade", 30),
        ("R03", "renda_mensal", 24),
        ("R03", "score_bureau", 18),
        ("R08", "situacao_emprego", 18),
        ("R07", "taxa_endividamento", 12),
        ("R10", None, 12),
        ("R11", None, 6),
        ("R01", "cpf", 1),
    ],
)
def test_inventory_counts_match_injection(
    bad_report: ValidationReport, rule_id: str, column: str | None, expected: int
) -> None:
    assert _failures(bad_report, rule_id, column) == expected


def test_primary_key_failures_are_inventoried(bad_report: ValidationReport) -> None:
    checks = {
        v.check: v.n_failures for v in bad_report.violations if v.rule_id == "R02"
    }
    assert checks["not_nullable"] == 6
    assert checks["field_uniqueness"] == 36  # 18 duplicatas + 18 originais


def test_reference_passes_every_rule(tmp_path: Path) -> None:
    context = BatchContext(
        batch_name="credit_ref", run_id="prova", quarantine_dir=tmp_path
    )
    valid, report = validate_batch(
        read_parquet(REFERENCE),
        schema=build_reference_schema(),
        policy=ValidationPolicy(),
        context=context,
    )
    assert report.decision == "SUCCESS"
    assert report.rejected_records == 0
    assert len(valid) == 12_000


def test_versioned_reports_match_decisions_and_have_no_pii() -> None:
    bad = read_json(VALIDATION_REPORTS / "bad_batch_report.json")
    reference = read_json(VALIDATION_REPORTS / "credit_ref_report.json")
    assert bad["decision"] == "BLOCKED"
    assert reference["decision"] == "SUCCESS"
    assert reference["total_records"] == 12_000
    assert scan_paths([VALIDATION_REPORTS]) == []
