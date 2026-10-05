"""Testes do contrato de dados: cada regra R01 a R12 quebrada isoladamente."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera.pandas as pa
import pytest

from credit_sentinel.contracts.schema import (
    RULES,
    build_inference_schema,
    build_reference_schema,
    rule_id_for,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd


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
