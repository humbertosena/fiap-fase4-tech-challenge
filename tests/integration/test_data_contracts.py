"""Testes do contrato de dados: cada regra R01 a R12 quebrada isoladamente."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pandera.pandas as pa
import pytest

from credit_sentinel.cli import main
from credit_sentinel.contracts.schema import (
    RULES,
    build_inference_schema,
    build_reference_schema,
    rule_id_for,
)
from credit_sentinel.contracts.validator import (
    BatchContext,
    CircuitBreakerOpenError,
    ContractViolationError,
    ValidationPolicy,
    validate_batch,
)
from credit_sentinel.utils.io import read_json

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from credit_sentinel.settings import Settings


def _failed_rules(df: pd.DataFrame, schema: pa.DataFrameSchema) -> set[str]:
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        pairs = zip(cases["check"], cases["column"], strict=True)
        return {rule_id_for(check, column) for check, column in pairs}
    return set()


def _first_row(df: pd.DataFrame, mask: pd.Series) -> int:
    return int(df.index[mask][0])


@pytest.mark.schema
def test_reference_sample_passes_every_rule(df_reference_small: pd.DataFrame) -> None:
    assert _failed_rules(df_reference_small, build_reference_schema()) == set()


@pytest.mark.schema
def test_inference_schema_accepts_batches_without_target(
    df_reference_small: pd.DataFrame,
) -> None:
    batch = df_reference_small.drop(columns=["inadimplente"])
    assert _failed_rules(batch, build_inference_schema()) == set()


@pytest.mark.schema
def test_reference_schema_requires_target(df_reference_small: pd.DataFrame) -> None:
    batch = df_reference_small.drop(columns=["inadimplente"])
    assert _failed_rules(batch, build_reference_schema()) == {"R01"}


def _extra_column(df: pd.DataFrame) -> None:
    df["cpf"] = "000.000.000-00"


def _wrong_dtype(df: pd.DataFrame) -> None:
    df["idade"] = df["idade"].astype(float)


def _duplicate_id(df: pd.DataFrame) -> None:
    df.loc[1, "id_cliente"] = df.loc[0, "id_cliente"]


def _malformed_id(df: pd.DataFrame) -> None:
    df.loc[0, "id_cliente"] = "cli-123"


def _null_income(df: pd.DataFrame) -> None:
    df.loc[0, "renda_mensal"] = float("nan")


def _set(column: str, value: object) -> Callable[[pd.DataFrame], None]:
    def mutate(df: pd.DataFrame) -> None:
        df.loc[0, column] = value

    return mutate


def _installment_above_income(df: pd.DataFrame) -> None:
    df.loc[0, "valor_parcela"] = df.loc[0, "renda_mensal"] * 1.2


def _unemployed_with_tenure(df: pd.DataFrame) -> None:
    row = _first_row(df, df["situacao_emprego"] != "desempregado")
    df.loc[row, ["situacao_emprego", "tempo_emprego_meses"]] = ["desempregado", 24]


def _tenure_above_age(df: pd.DataFrame) -> None:
    row = _first_row(df, df["idade"].between(18, 60))
    df.loc[row, "situacao_emprego"] = "CLT"
    df.loc[row, "tempo_emprego_meses"] = (df.loc[row, "idade"] - 16) * 12 + 1


CASES = [
    ("R01", _extra_column),
    ("R01", _wrong_dtype),
    ("R02", _duplicate_id),
    ("R02", _malformed_id),
    ("R03", _null_income),
    ("R04", _set("idade", 17)),
    ("R05", _set("renda_mensal", 2_000_000.0)),
    ("R06", _set("score_bureau", 1200)),
    ("R07", _set("utilizacao_credito", 1.5)),
    ("R08", _set("genero", "X")),
    ("R09", _set("qtd_parcelas_atraso_12m", 13)),
    ("R10", _installment_above_income),
    ("R11", _unemployed_with_tenure),
    ("R12", _tenure_above_age),
]


@pytest.mark.schema
@pytest.mark.parametrize(
    ("rule_id", "mutate"), CASES, ids=[c[1].__name__ for c in CASES]
)
def test_each_rule_is_detected_in_isolation(
    df_reference_small: pd.DataFrame,
    rule_id: str,
    mutate: Callable[[pd.DataFrame], None],
) -> None:
    broken = df_reference_small.copy()
    mutate(broken)
    assert _failed_rules(broken, build_reference_schema()) == {rule_id}


@pytest.mark.schema
@pytest.mark.parametrize("column", ["valor_parcela", "tempo_emprego_meses"])
def test_missing_column_is_structural_and_cross_checks_do_not_crash(
    df_reference_small: pd.DataFrame, column: str
) -> None:
    batch = df_reference_small.drop(columns=[column])
    assert _failed_rules(batch, build_reference_schema()) == {"R01"}


@pytest.mark.schema
def test_rules_metadata_covers_twelve_rules_with_severity() -> None:
    assert [rule.id for rule in RULES] == [f"R{n:02d}" for n in range(1, 13)]
    structural = {rule.id for rule in RULES if rule.severity == "estrutural"}
    assert structural == {"R01", "R02"}


@pytest.mark.schema
def test_rule_id_mapping_for_builtin_failures() -> None:
    assert rule_id_for("not_nullable", "id_cliente") == "R02"
    assert rule_id_for("not_nullable", "renda_mensal") == "R03"
    assert rule_id_for("field_uniqueness", "id_cliente") == "R02"
    assert rule_id_for("column_in_schema", None) == "R01"
    assert rule_id_for("dtype('int64')", "idade") == "R01"
    assert rule_id_for("R07 · utilização entre 0 e 1", "utilizacao_credito") == "R07"


def _context(tmp_path: Path, name: str = "lote_teste") -> BatchContext:
    return BatchContext(batch_name=name, run_id="run-teste", quarantine_dir=tmp_path)


def _break_ages(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    broken = df.copy()
    broken.loc[broken.index[:n_rows], "idade"] = 15
    return broken


@pytest.mark.integration
def test_clean_batch_succeeds_without_quarantine(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    valid, report = validate_batch(
        df_reference_small,
        schema=build_reference_schema(),
        policy=ValidationPolicy(),
        context=_context(tmp_path),
    )
    assert report.decision == "SUCCESS"
    assert report.rejected_records == 0
    assert len(valid) == len(df_reference_small)
    assert not list(tmp_path.iterdir())


@pytest.mark.integration
def test_small_share_of_bad_rows_goes_to_quarantine(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    batch = _break_ages(df_reference_small, 2)  # 0,4 % das linhas
    valid, report = validate_batch(
        batch,
        schema=build_reference_schema(),
        policy=ValidationPolicy(),
        context=_context(tmp_path),
    )
    assert report.decision == "SUCCESS"
    assert report.rejected_records == 2
    assert report.valid_records == len(batch) - 2
    assert len(valid) == len(batch) - 2
    quarantined = pd.read_parquet(tmp_path / "run-teste.parquet")
    assert len(quarantined) == 2


@pytest.mark.integration
def test_circuit_breaker_opens_above_one_percent(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    batch = _break_ages(df_reference_small, 10)  # 2 % das linhas
    with pytest.raises(CircuitBreakerOpenError) as raised:
        validate_batch(
            batch,
            schema=build_reference_schema(),
            policy=ValidationPolicy(),
            context=_context(tmp_path),
        )
    report = raised.value.report
    assert report.decision == "BLOCKED"
    assert report.rejected_ratio == pytest.approx(0.02)
    assert "circuit breaker" in (report.blocked_reason or "")


@pytest.mark.integration
def test_structural_violation_blocks_and_keeps_full_inventory(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    batch = _break_ages(df_reference_small, 1)
    batch["cpf"] = "000.000.000-00"
    batch.loc[3, "taxa_endividamento"] = 1.7
    with pytest.raises(ContractViolationError) as raised:
        validate_batch(
            batch,
            schema=build_reference_schema(),
            policy=ValidationPolicy(),
            context=_context(tmp_path),
        )
    report = raised.value.report
    assert report.decision == "BLOCKED"
    assert {v.rule_id for v in report.violations} == {"R01", "R04", "R07"}
    assert "R01" in (report.blocked_reason or "")
    assert not list(tmp_path.iterdir())


@pytest.mark.integration
def test_cross_field_violation_counts_rows_once(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    batch = df_reference_small.copy()
    batch.loc[0, "valor_parcela"] = batch.loc[0, "renda_mensal"] * 2
    _, report = validate_batch(
        batch,
        schema=build_reference_schema(),
        policy=ValidationPolicy(),
        context=_context(tmp_path),
    )
    (violation,) = report.violations
    assert violation.rule_id == "R10"
    assert violation.column is None
    assert violation.n_failures == 1
    assert violation.sample_index == [0]


@pytest.mark.integration
def test_validate_cli_exit_codes_and_report(project_tmp: Settings) -> None:
    assert main(["generate-data", "--rows", "300"]) == 0
    reference = project_tmp.reference_dir / "credit_ref.parquet"
    bad = project_tmp.corrupted_dir / "bad_batch.parquet"
    assert main(["validate", "--input", str(reference), "--schema", "reference"]) == 0
    assert main(["validate", "--input", str(bad)]) == 2
    report = read_json(project_tmp.reports_dir / "validation" / "bad_batch_report.json")
    assert report["decision"] == "BLOCKED"
    assert report["batch_name"] == "bad_batch"


@pytest.mark.integration
def test_sample_of_row_failing_only_on_id_has_empty_values(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    batch = df_reference_small.copy()
    batch.loc[0, "id_cliente"] = None
    with pytest.raises(ContractViolationError) as raised:
        validate_batch(
            batch,
            schema=build_reference_schema(),
            policy=ValidationPolicy(),
            context=_context(tmp_path),
        )
    (sample,) = raised.value.report.sample_failures
    assert sample["regras"] == ["R02"]
    assert sample["valores"] == {}
    assert sample["id_cliente_hash"] is None
