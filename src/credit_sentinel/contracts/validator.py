"""Validação de lotes com circuit breaker fail-fast em dois níveis (ADR 002).

1. Violação **estrutural** (R01 schema/tipos, R02 chave primária) bloqueia o lote.
2. Violações **de linha** (R03 a R12) vão para a quarentena enquanto a fração
   rejeitada não passar de ``max_rejected_ratio``; acima disso o breaker abre.

O inventário é sempre completo (``lazy=True``) e nunca há imputação.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, Field, SecretStr

from credit_sentinel.contracts.privacy_sanitizer import HASH_COLUMN, hash_identifier
from credit_sentinel.contracts.schema import RULES_BY_ID, rule_id_for
from credit_sentinel.data import dictionary as dd
from credit_sentinel.settings import get_settings
from credit_sentinel.utils.io import write_parquet

SAMPLE_LIMIT = 5
_COLUMN_PRESENCE_CHECKS = {"column_in_schema", "column_in_dataframe"}
_CASE_COLUMNS = ["schema_context", "column", "check", "failure_case", "index"]


class BatchContext(BaseModel):
    """Identificação do lote em validação.

    Attributes:
        batch_name: Nome lógico do lote (ex.: ``bad_batch``).
        run_id: Identificador da execução (nome do arquivo de quarentena).
        quarantine_dir: Diretório da quarentena; ``None`` desliga a gravação.
        pii_salt: Salt da pseudonimização das amostras do relatório.
    """

    batch_name: str
    run_id: str
    quarantine_dir: Path | None = None
    pii_salt: SecretStr = Field(default_factory=lambda: get_settings().pii_salt)


class ValidationPolicy(BaseModel):
    """Política do circuit breaker.

    Attributes:
        max_rejected_ratio: Fração máxima de linhas rejeitadas (SLO 1).
        quarantine: Se as linhas rejeitadas devem ser gravadas.
    """

    max_rejected_ratio: float = 0.01
    quarantine: bool = True


class RuleViolation(BaseModel):
    """Falhas agregadas de uma regra em uma coluna."""

    rule_id: str
    column: str | None
    check: str
    n_failures: int
    sample_index: list[int]


class ValidationReport(BaseModel):
    """Relatório completo de validação de um lote (serializável em JSON)."""

    batch_name: str
    run_id: str
    validated_at: datetime
    schema_name: str
    total_records: int
    valid_records: int
    rejected_records: int
    rejected_ratio: float
    decision: Literal["SUCCESS", "BLOCKED"]
    blocked_reason: str | None = None
    violations: list[RuleViolation]
    sample_failures: list[dict[str, Any]]


class _BlockedBatchError(Exception):
    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.blocked_reason)
        self.report = report


class ContractViolationError(_BlockedBatchError):
    """Lote bloqueado por violação estrutural (R01/R02)."""


class CircuitBreakerOpenError(_BlockedBatchError):
    """Lote bloqueado porque a fração de linhas inválidas passou do limite."""


@dataclass(frozen=True)
class _Outcome:
    cases: pd.DataFrame
    violations: list[RuleViolation]
    rejected: list[Any]


def _failure_cases(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return exc.failure_cases[_CASE_COLUMNS].copy()
    return pd.DataFrame(columns=_CASE_COLUMNS)


def _annotate(cases: pd.DataFrame) -> pd.DataFrame:
    pairs = zip(cases["check"], cases["column"], strict=True)
    cases["rule_id"] = [rule_id_for(check, column) for check, column in pairs]
    presence = cases["check"].isin(_COLUMN_PRESENCE_CHECKS)
    frame_level = (cases["schema_context"] == "DataFrameSchema") & ~presence
    cases["column"] = cases["column"].where(~presence, cases["failure_case"])
    cases.loc[frame_level, "column"] = None
    return cases


def _indices(group: pd.DataFrame) -> list[int]:
    return sorted({int(index) for index in group["index"].dropna()})


def _violations(cases: pd.DataFrame) -> list[RuleViolation]:
    keys = ["rule_id", "column", "check"]
    violations = []
    for (rule_id, column, check), group in cases.groupby(keys, dropna=False):
        rows = _indices(group)
        violations.append(
            RuleViolation(
                rule_id=str(rule_id),
                column=None if pd.isna(column) else str(column),
                check=str(check),
                n_failures=len(rows) or len(group),
                sample_index=rows[:SAMPLE_LIMIT],
            )
        )
    return violations


def _analyze(df: pd.DataFrame, schema: pa.DataFrameSchema) -> _Outcome:
    cases = _annotate(_failure_cases(df, schema))
    rejected = sorted({index for index in cases["index"].dropna() if index in df.index})
    return _Outcome(cases=cases, violations=_violations(cases), rejected=rejected)


def _hashed_id(df: pd.DataFrame, index: Any, salt: str) -> str | None:  # noqa: ANN401
    value = df.loc[index].get(dd.ID_COLUMN)
    return (
        None if value is None or pd.isna(value) else hash_identifier(str(value), salt)
    )


def _sample_failures(
    df: pd.DataFrame, outcome: _Outcome, salt: str
) -> list[dict[str, Any]]:
    samples = []
    for index in outcome.rejected[:SAMPLE_LIMIT]:
        rows = outcome.cases[outcome.cases["index"] == index]
        rules = sorted(set(rows["rule_id"]))
        involved = {c for rule in rules for c in RULES_BY_ID[rule].columns}
        columns = sorted((involved - {dd.ID_COLUMN}) & set(df.columns))
        values = json.loads(df.loc[[index], columns].to_json(orient="records"))[0]
        sample = {"index": int(index), HASH_COLUMN: _hashed_id(df, index, salt)}
        samples.append({**sample, "regras": rules, "valores": values})
    return samples


def _report(
    df: pd.DataFrame, outcome: _Outcome, context: BatchContext, schema_name: str
) -> ValidationReport:
    total, rejected = len(df), len(outcome.rejected)
    return ValidationReport(
        batch_name=context.batch_name,
        run_id=context.run_id,
        validated_at=datetime.now(UTC),
        schema_name=schema_name,
        total_records=total,
        valid_records=total - rejected,
        rejected_records=rejected,
        rejected_ratio=rejected / total if total else 0.0,
        decision="SUCCESS",
        violations=outcome.violations,
        sample_failures=_sample_failures(
            df, outcome, context.pii_salt.get_secret_value()
        ),
    )


def _blocked(report: ValidationReport, reason: str) -> ValidationReport:
    return report.model_copy(update={"decision": "BLOCKED", "blocked_reason": reason})


def _structural_rules(violations: list[RuleViolation]) -> list[str]:
    rules = {
        v.rule_id for v in violations if RULES_BY_ID[v.rule_id].severity == "estrutural"
    }
    return sorted(rules)


def _quarantine(df: pd.DataFrame, outcome: _Outcome, context: BatchContext) -> None:
    if outcome.rejected and context.quarantine_dir is not None:
        path = context.quarantine_dir / f"{context.run_id}.parquet"
        write_parquet(df.loc[outcome.rejected], path)


def validate_batch(
    df: pd.DataFrame,
    *,
    schema: pa.DataFrameSchema,
    policy: ValidationPolicy,
    context: BatchContext,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Valida um lote contra o contrato e aplica o circuit breaker.

    Args:
        df: Lote a validar.
        schema: Schema Pandera (referência ou inferência).
        policy: Política de rejeição e quarentena.
        context: Identificação do lote e diretório de quarentena.

    Returns:
        Linhas válidas e o relatório com decisão ``SUCCESS``.

    Raises:
        ContractViolationError: Violação estrutural (R01/R02).
        CircuitBreakerOpenError: Fração rejeitada acima do limite.
    """
    outcome = _analyze(df, schema)
    report = _report(df, outcome, context, schema.name or "")
    structural = _structural_rules(outcome.violations)
    if structural:
        reason = f"violação estrutural ({', '.join(structural)}): lote bloqueado"
        raise ContractViolationError(_blocked(report, reason))
    if report.rejected_ratio > policy.max_rejected_ratio:
        reason = (
            f"circuit breaker aberto: {report.rejected_ratio:.2%} das linhas"
            f" rejeitadas (limite {policy.max_rejected_ratio:.2%})"
        )
        raise CircuitBreakerOpenError(_blocked(report, reason))
    if policy.quarantine:
        _quarantine(df, outcome, context)
    return df.drop(index=outcome.rejected), report
