"""Critérios de aceitação do SPEC-drift-simulation nos arquivos versionados."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import SCENARIOS_FILE, SCORE_BAND
from credit_sentinel.drift.simulator import WINDOWS, load_windows, simulate_window
from credit_sentinel.settings import Settings
from credit_sentinel.utils.io import read_parquet

pytestmark = pytest.mark.integration

SETTINGS = Settings(_env_file=None)
REFERENCE_END = pd.Timestamp(date(2026, 6, 30))
MONTHS = {
    "mes_1": (date(2026, 7, 1), date(2026, 7, 31)),
    "mes_2": (date(2026, 8, 1), date(2026, 8, 31)),
    "mes_3_crise": (date(2026, 9, 1), date(2026, 9, 30)),
}


def _band_rate(frame: pd.DataFrame) -> float:
    band = frame[frame["score_bureau"].between(*SCORE_BAND)]
    return float(band[dd.TARGET].mean())


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return read_parquet(SETTINGS.reference_dir / "credit_ref.parquet")


@pytest.fixture(scope="module")
def windows() -> dict[str, pd.DataFrame]:
    return {
        name: read_parquet(SETTINGS.production_dir / f"{name}.parquet")
        for name in WINDOWS
    }


def test_reference_and_production_live_in_different_folders() -> None:
    assert SETTINGS.reference_dir != SETTINGS.production_dir
    assert SETTINGS.production_dir.parent == SETTINGS.reference_dir.parent


def test_three_windows_share_the_reference_schema(
    reference: pd.DataFrame, windows: dict[str, pd.DataFrame]
) -> None:
    assert list(windows) == ["mes_1", "mes_2", "mes_3_crise"]
    for name, frame in windows.items():
        assert len(frame) == 5000, name
        assert list(frame.columns) == list(reference.columns), name
        assert frame.dtypes.equals(reference.dtypes), name


def test_periods_are_correct_and_after_the_reference(
    windows: dict[str, pd.DataFrame],
) -> None:
    for name, frame in windows.items():
        start, end = MONTHS[name]
        assert frame[dd.DATE_COLUMN].min() >= pd.Timestamp(start), name
        assert frame[dd.DATE_COLUMN].max() <= pd.Timestamp(end), name
        assert frame[dd.DATE_COLUMN].min() > REFERENCE_END, name


def test_no_client_appears_in_two_datasets(
    reference: pd.DataFrame, windows: dict[str, pd.DataFrame]
) -> None:
    frames = [reference, *windows.values()]
    ids = np.concatenate([frame[dd.ID_COLUMN].to_numpy() for frame in frames])
    assert len(set(ids)) == len(ids)


def test_mes_1_is_a_stationary_control(
    reference: pd.DataFrame, windows: dict[str, pd.DataFrame]
) -> None:
    control = windows["mes_1"]
    for column in dd.NUMERIC_COLUMNS:
        change = control[column].mean() / reference[column].mean() - 1
        assert abs(change) <= 0.03, column
    for column in (*dd.CATEGORICAL_COLUMNS, *dd.SENSITIVE_COLUMNS):
        shares = control[column].value_counts(normalize=True)
        expected = reference[column].value_counts(normalize=True)
        assert (shares - expected).abs().max() <= 0.015, column
    assert 0.05 <= control[dd.TARGET].mean() <= 0.07


def test_mes_2_sits_in_the_attention_zone(windows: dict[str, pd.DataFrame]) -> None:
    assert 0.08 <= windows["mes_2"][dd.TARGET].mean() <= 0.09


def test_mes_3_crise_has_covariate_and_prior_shift(
    reference: pd.DataFrame, windows: dict[str, pd.DataFrame]
) -> None:
    crisis = windows["mes_3_crise"]
    income = crisis["renda_mensal"].median() / reference["renda_mensal"].median()
    debt_gap = (
        crisis["taxa_endividamento"].mean() - reference["taxa_endividamento"].mean()
    )
    jobless = (crisis["situacao_emprego"] == "desempregado").mean()
    assert income <= 0.85
    assert debt_gap >= 0.10
    assert jobless >= 0.11
    assert 0.12 <= crisis[dd.TARGET].mean() <= 0.14


def test_mes_3_crise_has_concept_drift_in_the_mid_score_band(
    reference: pd.DataFrame, windows: dict[str, pd.DataFrame]
) -> None:
    assert _band_rate(windows["mes_3_crise"]) >= 2 * _band_rate(reference)


def test_default_rate_rises_window_after_window(
    windows: dict[str, pd.DataFrame],
) -> None:
    rates = [windows[name][dd.TARGET].mean() for name in WINDOWS]
    assert rates == sorted(rates)


def test_versioned_windows_are_reproducible(
    windows: dict[str, pd.DataFrame],
) -> None:
    config = SETTINGS.config_dir / SCENARIOS_FILE
    for spec in load_windows(config):
        regenerated = simulate_window(spec)
        pd.testing.assert_frame_equal(windows[spec.name], regenerated)
