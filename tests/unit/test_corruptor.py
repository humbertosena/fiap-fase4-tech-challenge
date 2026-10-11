"""Testes do corruptor que gera o lote com anomalias propositais."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.corruptor import ANOMALY_PLAN, corrupt_batch, is_valid_cpf
from credit_sentinel.data.generator import ScenarioParams, generate_portfolio
from credit_sentinel.utils.seeds import make_rng

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def bad(reference_params: ScenarioParams) -> pd.DataFrame:
    period = (date(2025, 7, 1), date(2026, 6, 30))
    clean = generate_portfolio(600, reference_params, make_rng(11), period)
    return corrupt_batch(clean, make_rng(7))


def _expected(key: str) -> int:
    return next(spec.n_rows for spec in ANOMALY_PLAN if spec.key == key)


def test_keeps_row_count_and_adds_cpf_column(bad: pd.DataFrame) -> None:
    assert len(bad) == 600
    assert list(bad.columns) == [*dd.ALL_COLUMNS, "cpf"]


def test_cpf_values_are_formatted_but_never_valid(bad: pd.DataFrame) -> None:
    assert bad["cpf"].str.fullmatch(r"\d{3}\.\d{3}\.\d{3}-\d{2}").all()
    assert not bad["cpf"].map(is_valid_cpf).any()


def test_is_valid_cpf_accepts_known_valid_number() -> None:
    assert is_valid_cpf("529.982.247-25")
    assert not is_valid_cpf("529.982.247-26")
    assert not is_valid_cpf("111.111.111-11")


def test_age_out_of_domain_count(bad: pd.DataFrame) -> None:
    invalid = ~bad["idade"].between(18, 100)
    assert invalid.sum() == _expected("idade_fora_dominio")
    assert (bad["idade"] < 0).any()
    assert (bad["idade"] > 100).any()


def test_null_counts_for_income_score_and_id(bad: pd.DataFrame) -> None:
    assert bad["renda_mensal"].isna().sum() == _expected("renda_nula")
    assert bad["score_bureau"].isna().sum() == _expected("score_nulo")
    assert bad[dd.ID_COLUMN].isna().sum() == _expected("id_nulo")


def test_duplicated_id_count(bad: pd.DataFrame) -> None:
    ids = bad[dd.ID_COLUMN].dropna()
    assert ids.duplicated(keep="first").sum() == _expected("id_duplicado")


def test_categorical_and_range_violations(bad: pd.DataFrame) -> None:
    off_enum = ~bad["situacao_emprego"].isin(dd.EMPLOYMENT)
    assert off_enum.sum() == _expected("emprego_fora_enum")
    assert (bad["taxa_endividamento"] > 1).sum() == _expected("endividamento_acima_1")


def test_cross_field_violations(bad: pd.DataFrame) -> None:
    installment_gt_income = bad["valor_parcela"] > bad["renda_mensal"]
    assert installment_gt_income.sum() == _expected("parcela_maior_renda")
    unemployed = bad["situacao_emprego"] == "desempregado"
    working_time = bad["tempo_emprego_meses"] > 0
    assert (unemployed & working_time).sum() == _expected("desempregado_com_tempo")


def test_corruption_is_deterministic(reference_params: ScenarioParams) -> None:
    period = (date(2025, 7, 1), date(2026, 6, 30))
    clean = generate_portfolio(400, reference_params, make_rng(3), period)
    pd.testing.assert_frame_equal(
        corrupt_batch(clean, make_rng(7)), corrupt_batch(clean, make_rng(7))
    )


def test_plan_lists_ten_anomalies_with_rule_ids() -> None:
    assert len(ANOMALY_PLAN) == 10
    assert all(spec.rule_id.startswith("R") for spec in ANOMALY_PLAN)
