"""Testes do motor de janelas temporais (covariate shift, T21)."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import generate_portfolio
from credit_sentinel.drift.simulator import (
    WINDOWS,
    ShiftDescription,
    WindowSpec,
    describe_shift,
    load_windows,
    simulate_all,
    simulate_window,
)
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from pathlib import Path

    from credit_sentinel.data.generator import ScenarioParams

pytestmark = pytest.mark.unit

REFERENCE_PERIOD = (date(2025, 7, 1), date(2026, 6, 30))
SMALL = 3000


@pytest.fixture(scope="module")
def specs(scenarios_path: Path) -> dict[str, WindowSpec]:
    return {spec.name: spec for spec in load_windows(scenarios_path)}


@pytest.fixture(scope="module")
def reference_frame(reference_params: ScenarioParams) -> pd.DataFrame:
    return generate_portfolio(8000, reference_params, make_rng(42), REFERENCE_PERIOD)


@pytest.fixture(scope="module")
def mes_1(specs: dict[str, WindowSpec]) -> pd.DataFrame:
    return simulate_window(specs["mes_1"].model_copy(update={"n_rows": SMALL}))


@pytest.fixture(scope="module")
def mes_2(specs: dict[str, WindowSpec]) -> pd.DataFrame:
    return simulate_window(specs["mes_2"].model_copy(update={"n_rows": SMALL}))


def test_windows_follow_declared_order(scenarios_path: Path) -> None:
    assert [spec.name for spec in load_windows(scenarios_path)] == list(WINDOWS)


def test_periods_are_consecutive_calendar_months(
    specs: dict[str, WindowSpec],
) -> None:
    assert specs["mes_1"].period == (date(2026, 7, 1), date(2026, 7, 31))
    assert specs["mes_2"].period == (date(2026, 8, 1), date(2026, 8, 31))


def test_each_window_has_five_thousand_rows(specs: dict[str, WindowSpec]) -> None:
    assert {spec.n_rows for spec in specs.values()} == {5000}


def test_seeds_are_base_plus_window_position(
    scenarios_path: Path, specs: dict[str, WindowSpec]
) -> None:
    assert [spec.seed for spec in specs.values()] == [43, 44]
    shifted = load_windows(scenarios_path, seed=100)
    assert [spec.seed for spec in shifted] == [101, 102]


def test_mes_1_keeps_the_reference_parameters(
    specs: dict[str, WindowSpec], reference_params: ScenarioParams
) -> None:
    assert specs["mes_1"].params == reference_params


def test_mes_2_overrides_the_covariate_parameters(
    specs: dict[str, WindowSpec], reference_params: ScenarioParams
) -> None:
    params = specs["mes_2"].params
    assert params.income_real_factor == pytest.approx(0.90)
    assert params.debt_logit_shift == pytest.approx(0.35)
    assert params.utilization_shift == pytest.approx(0.06)
    assert params.monthly_interest == pytest.approx(0.030)
    assert params.employment_shares["desempregado"] == pytest.approx(0.09)
    assert params.purpose_shares["consolidacao_dividas"] == pytest.approx(0.26)
    assert params.income_median == reference_params.income_median
    assert params.region_shares == reference_params.region_shares


def test_unknown_window_name_is_rejected(scenarios_path: Path) -> None:
    with pytest.raises(ValueError, match="mes_9"):
        simulate_all(scenarios_path, scenarios_path.parent, windows=["mes_9"])


def test_simulate_window_is_deterministic(specs: dict[str, WindowSpec]) -> None:
    spec = specs["mes_2"].model_copy(update={"n_rows": 400})
    pd.testing.assert_frame_equal(simulate_window(spec), simulate_window(spec))


def test_window_has_the_reference_schema_and_period(
    mes_1: pd.DataFrame, specs: dict[str, WindowSpec]
) -> None:
    start, end = specs["mes_1"].period
    assert list(mes_1.columns) == dd.ALL_COLUMNS
    assert len(mes_1) == SMALL
    assert mes_1[dd.DATE_COLUMN].min() >= pd.Timestamp(start)
    assert mes_1[dd.DATE_COLUMN].max() <= pd.Timestamp(end)
    assert mes_1[dd.DATE_COLUMN].min() > pd.Timestamp(REFERENCE_PERIOD[1])


def test_windows_draw_different_clients(
    mes_1: pd.DataFrame, mes_2: pd.DataFrame
) -> None:
    assert set(mes_1[dd.ID_COLUMN]).isdisjoint(mes_2[dd.ID_COLUMN])
    assert not mes_1["renda_mensal"].equals(mes_2["renda_mensal"])


def test_mes_1_is_a_stationary_control(
    mes_1: pd.DataFrame, reference_frame: pd.DataFrame
) -> None:
    for column in ("renda_mensal", "score_bureau", "valor_parcela", "idade"):
        ratio = mes_1[column].mean() / reference_frame[column].mean()
        assert ratio == pytest.approx(1.0, abs=0.05), column
    shares = mes_1["situacao_emprego"].value_counts(normalize=True)
    expected = reference_frame["situacao_emprego"].value_counts(normalize=True)
    assert (shares - expected).abs().max() < 0.03


def test_mes_2_shifts_the_covariates(
    mes_2: pd.DataFrame, reference_frame: pd.DataFrame
) -> None:
    income = mes_2["renda_mensal"].median() / reference_frame["renda_mensal"].median()
    assert 0.84 < income < 0.95
    assert (
        mes_2["taxa_endividamento"].mean()
        > reference_frame["taxa_endividamento"].mean() + 0.04
    )
    assert (
        mes_2["utilizacao_credito"].mean()
        > reference_frame["utilizacao_credito"].mean() + 0.04
    )
    jobless = (mes_2["situacao_emprego"] == "desempregado").mean()
    consolidation = (mes_2["finalidade_credito"] == "consolidacao_dividas").mean()
    assert jobless == pytest.approx(0.09, abs=0.02)
    assert consolidation == pytest.approx(0.26, abs=0.03)


def test_describe_shift_reports_means_and_shares(
    mes_2: pd.DataFrame, reference_frame: pd.DataFrame
) -> None:
    shift = describe_shift(reference_frame, mes_2)
    numeric = {item.feature: item for item in shift.numeric}
    categories = {(c.feature, c.category): c for c in shift.categorical}
    assert shift.reference_rows == len(reference_frame)
    assert shift.window_rows == len(mes_2)
    assert numeric["renda_mensal"].relative_change < -0.05
    assert (
        numeric["comprometimento_renda"].mean_cur
        > numeric["comprometimento_renda"].mean_ref
    )
    jobless = categories[("situacao_emprego", "desempregado")]
    assert jobless.delta_pp > 0.5
    assert jobless.share_cur == pytest.approx(0.09, abs=0.02)
    assert ShiftDescription.model_validate_json(shift.model_dump_json()) == shift


def test_describe_shift_of_identical_frames_is_zero(
    reference_frame: pd.DataFrame,
) -> None:
    shift = describe_shift(reference_frame, reference_frame)
    assert all(item.relative_change == 0 for item in shift.numeric)
    assert all(item.delta_pp == 0 for item in shift.categorical)
    assert shift.base_rate_cur == shift.base_rate_ref


def test_simulate_all_writes_one_parquet_per_window(
    scenarios_path: Path, tmp_path: Path
) -> None:
    paths = simulate_all(scenarios_path, tmp_path / "production")
    assert sorted(paths) == list(WINDOWS)
    assert all(path.suffix == ".parquet" and path.exists() for path in paths.values())
    stored = pd.read_parquet(paths["mes_1"])
    assert len(stored) == 5000
    assert list(stored.columns) == dd.ALL_COLUMNS


def test_simulate_all_can_select_a_subset(scenarios_path: Path, tmp_path: Path) -> None:
    paths = simulate_all(scenarios_path, tmp_path, windows=["mes_2"])
    assert list(paths) == ["mes_2"]
    assert [p.name for p in tmp_path.iterdir()] == ["mes_2.parquet"]
