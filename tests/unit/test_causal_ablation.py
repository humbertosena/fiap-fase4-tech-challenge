"""Ablação causal: definição dos cenários, deltas pareados e relatório."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from credit_sentinel.data.generator import SCENARIOS_FILE, load_scenarios
from credit_sentinel.governance.causal_ablation import (
    CATALOGUE,
    CONCEPT,
    COVARIATE,
    CausalAblation,
    ScenarioResult,
    ScenarioSpec,
    params_for,
    render_ablation_markdown,
    summarize,
)
from credit_sentinel.settings import Settings

if TYPE_CHECKING:
    from credit_sentinel.data.generator import ScenarioParams

pytestmark = pytest.mark.unit

SCENARIOS = load_scenarios(Settings(_env_file=None).config_dir / SCENARIOS_FILE)
REFERENCE, CRISIS = SCENARIOS["reference"], SCENARIOS["mes_3_crise"]


def _spec(key: str) -> ScenarioSpec:
    return next(spec for spec in CATALOGUE if spec.key == key)


def _fields_where_equal(a: ScenarioParams, b: ScenarioParams) -> set[str]:
    return {
        name for name in type(a).model_fields if getattr(a, name) == getattr(b, name)
    }


def test_catalogue_has_unique_keys_and_starts_with_the_control() -> None:
    keys = [spec.key for spec in CATALOGUE]
    assert len(keys) == len(set(keys)) == 11
    assert keys[:2] == ["controle", "crise"]


def test_every_field_is_a_real_scenario_parameter() -> None:
    names = set(type(REFERENCE).model_fields)
    for spec in CATALOGUE:
        assert set(spec.fields) <= names, spec.key


def test_control_is_the_reference_and_crisis_is_the_window() -> None:
    assert params_for(_spec("controle"), REFERENCE, CRISIS) == REFERENCE
    assert params_for(_spec("crise"), REFERENCE, CRISIS) == CRISIS


def test_only_turns_on_just_the_listed_shocks() -> None:
    params = params_for(_spec("so_covariate"), REFERENCE, CRISIS)
    for name in COVARIATE:
        assert getattr(params, name) == getattr(CRISIS, name)
    untouched = set(type(REFERENCE).model_fields) - set(COVARIATE)
    assert all(getattr(params, n) == getattr(REFERENCE, n) for n in untouched)


def test_without_turns_off_just_the_listed_shocks() -> None:
    params = params_for(_spec("crise_sem_concept"), REFERENCE, CRISIS)
    for name in CONCEPT:
        assert getattr(params, name) == getattr(REFERENCE, name)
    untouched = set(type(REFERENCE).model_fields) - set(CONCEPT)
    assert all(getattr(params, n) == getattr(CRISIS, n) for n in untouched)


def test_the_crisis_really_differs_from_the_reference_on_every_catalogued_field() -> (
    None
):
    same = _fields_where_equal(REFERENCE, CRISIS)
    for spec in CATALOGUE:
        assert not set(spec.fields) & same, spec.key


def test_summarize_uses_paired_deltas_against_the_control() -> None:
    control = np.array([0.80, 0.82, 0.78])
    aucs = np.array([0.76, 0.77, 0.75])
    result = summarize(_spec("crise"), np.array([0.1, 0.12, 0.11]), aucs, control)
    deltas = aucs - control
    assert result.delta_auc == pytest.approx(deltas.mean())
    assert result.delta_auc_std == pytest.approx(deltas.std(ddof=1))
    assert result.auc == pytest.approx(aucs.mean())
    assert result.base_rate == pytest.approx(0.11)


def test_the_control_has_zero_delta() -> None:
    control = np.array([0.80, 0.82, 0.78])
    result = summarize(_spec("controle"), np.array([0.06] * 3), control, control)
    assert result.delta_auc == 0.0
    assert result.delta_auc_std == 0.0


def _ablation() -> CausalAblation:
    scenarios = [
        ScenarioResult(
            key="controle",
            label="Controle",
            mode="only",
            fields=[],
            base_rate=0.0599,
            auc=0.7953,
            delta_auc=0.0,
            delta_auc_std=0.0,
        ),
        ScenarioResult(
            key="crise",
            label="Crise completa",
            mode="without",
            fields=[],
            base_rate=0.1222,
            auc=0.7574,
            delta_auc=-0.0379,
            delta_auc_std=0.0124,
        ),
    ]
    return CausalAblation(
        generated_at=datetime(2026, 11, 6, 14, 0, tzinfo=UTC),
        model_version="baseline-1.0.0",
        window="mes_3_crise",
        rows=5000,
        seeds=[45, 46],
        scenarios=scenarios,
    )


def test_ablation_round_trips_through_json() -> None:
    ablation = _ablation()
    assert CausalAblation.model_validate_json(ablation.model_dump_json()) == ablation


def test_markdown_is_in_portuguese_with_decimal_commas() -> None:
    text = render_ablation_markdown(_ablation())
    assert text.startswith("# Ablação causal")
    assert "| Crise completa |" in text
    assert "12,22 %" in text
    assert "0,757" in text
    assert "\N{MINUS SIGN} 0,038 ± 0,012" in text
    assert "baseline-1.0.0" in text
