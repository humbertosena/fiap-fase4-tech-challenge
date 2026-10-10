"""Teste de desempenho do treino completo (fora da CI)."""

from __future__ import annotations

import time
from datetime import date

import pytest

from credit_sentinel.data.generator import ScenarioParams, generate_portfolio
from credit_sentinel.ml.train_baseline import train_and_analyze
from credit_sentinel.settings import Settings
from credit_sentinel.utils.seeds import make_rng

pytestmark = pytest.mark.performance


def test_full_training_with_analysis_takes_less_than_60_seconds(
    reference_params: ScenarioParams,
) -> None:
    period = (date(2025, 7, 1), date(2026, 6, 30))
    reference = generate_portfolio(12_000, reference_params, make_rng(42), period)
    start = time.perf_counter()
    bundle = train_and_analyze(reference, settings=Settings(_env_file=None))
    assert time.perf_counter() - start < 60
    assert bundle.metrics.roc_auc >= 0.80
