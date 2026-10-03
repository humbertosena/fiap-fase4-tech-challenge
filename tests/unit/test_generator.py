"""Testes do gerador sintético de carteira de crédito (DGP)."""

from __future__ import annotations

import time
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import (
    ScenarioParams,
    calibrate_intercept,
    compute_logit,
    generate_portfolio,
    load_period,
    load_scenarios,
)
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

PERIOD = (date(2025, 7, 1), date(2026, 6, 30))


@pytest.fixture(scope="module")
def portfolio(reference_params: ScenarioParams) -> pd.DataFrame:
    return generate_portfolio(4000, reference_params, make_rng(42), PERIOD)


def test_shape_and_column_order(portfolio: pd.DataFrame) -> None:
    assert portfolio.shape == (4000, 18)
    assert list(portfolio.columns) == dd.ALL_COLUMNS


def test_no_nulls_and_unique_ids(portfolio: pd.DataFrame) -> None:
    assert not portfolio.isna().any().any()
    assert portfolio[dd.ID_COLUMN].is_unique
    assert portfolio[dd.ID_COLUMN].str.fullmatch(r"CLI-[0-9A-F]{10}").all()


def test_numeric_domains(portfolio: pd.DataFrame) -> None:
    assert portfolio["idade"].between(18, 100).all()
    assert portfolio["renda_mensal"].between(800, 60_000).all()
    assert portfolio["score_bureau"].between(0, 1000).all()
    assert portfolio["taxa_endividamento"].between(0, 1).all()
    assert portfolio["utilizacao_credito"].between(0, 1).all()
    assert portfolio["qtd_parcelas_atraso_12m"].between(0, 12).all()
    assert portfolio["tempo_emprego_meses"].between(0, 600).all()


def test_cross_field_rules_hold(portfolio: pd.DataFrame) -> None:
    assert (portfolio["valor_parcela"] <= portfolio["renda_mensal"]).all()
    unemployed = portfolio["situacao_emprego"] == "desempregado"
    assert (portfolio.loc[unemployed, "tempo_emprego_meses"] == 0).all()
    max_tenure = (portfolio["idade"] - 16) * 12
    assert (portfolio["tempo_emprego_meses"] <= max_tenure).all()


def test_categoricals_within_enums(portfolio: pd.DataFrame) -> None:
    for column, allowed in dd.ENUMS.items():
        assert set(portfolio[column].unique()) <= set(allowed), column
    assert set(portfolio["prazo_meses"].unique()) <= set(dd.TERM_OPTIONS)


def test_dates_within_period(portfolio: pd.DataFrame) -> None:
    dates = portfolio["data_referencia"]
    assert dates.min() >= pd.Timestamp(PERIOD[0])
    assert dates.max() <= pd.Timestamp(PERIOD[1])


def test_dtypes_are_stable(portfolio: pd.DataFrame) -> None:
    assert portfolio["idade"].dtype == np.int64
    assert portfolio["renda_mensal"].dtype == np.float64
    assert portfolio["possui_imovel"].dtype == bool
    assert portfolio["inadimplente"].dtype == np.int64
    assert str(portfolio["data_referencia"].dtype) == "datetime64[ns]"


def test_same_seed_is_deterministic(reference_params: ScenarioParams) -> None:
    first = generate_portfolio(300, reference_params, make_rng(1), PERIOD)
    second = generate_portfolio(300, reference_params, make_rng(1), PERIOD)
    pd.testing.assert_frame_equal(first, second)


def test_base_rate_is_calibrated_near_six_percent(portfolio: pd.DataFrame) -> None:
    assert portfolio["inadimplente"].mean() == pytest.approx(0.06, abs=0.01)


def test_stored_intercept_matches_calibration(
    reference_params: ScenarioParams,
) -> None:
    calibrated = calibrate_intercept(reference_params, 0.06, make_rng(2024))
    assert calibrated == pytest.approx(reference_params.beta0, abs=0.05)


def test_sensitive_attributes_have_zero_weight_in_logit(
    portfolio: pd.DataFrame, reference_params: ScenarioParams
) -> None:
    shuffled = portfolio.copy()
    shuffled["genero"] = np.where(portfolio["genero"] == "F", "M", "F")
    shuffled["regiao"] = "Norte"
    np.testing.assert_array_equal(
        compute_logit(portfolio, reference_params),
        compute_logit(shuffled, reference_params),
    )


def test_concept_gamma_raises_risk_only_in_score_band(
    portfolio: pd.DataFrame, reference_params: ScenarioParams
) -> None:
    shocked = reference_params.model_copy(update={"gamma_concept": 1.0})
    delta = compute_logit(portfolio, shocked) - compute_logit(
        portfolio, reference_params
    )
    in_band = portfolio["score_bureau"].between(550, 700)
    assert np.allclose(delta[in_band], 0.9)
    assert np.allclose(delta[~in_band], 0.0)


def test_generation_is_fast(reference_params: ScenarioParams) -> None:
    start = time.perf_counter()
    generate_portfolio(4000, reference_params, make_rng(3), PERIOD)
    assert time.perf_counter() - start < 2.0


def test_load_scenarios_and_period(scenarios_path: Path) -> None:
    scenarios = load_scenarios(scenarios_path)
    assert "reference" in scenarios
    assert load_period(scenarios_path, "reference") == PERIOD


def test_shares_must_sum_to_one(reference_params: ScenarioParams) -> None:
    data = reference_params.model_dump()
    data["region_shares"] = {"Norte": 0.5, "Sul": 0.2}
    with pytest.raises(ValueError, match="somar 1"):
        ScenarioParams.model_validate(data)
