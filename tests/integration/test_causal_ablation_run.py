"""Ablação causal executada sobre o projeto de teste (baseline pequeno, 500 linhas)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.drift.detection import MissingInputError
from credit_sentinel.governance.causal_ablation import (
    CATALOGUE,
    CausalAblation,
    run_ablation,
)

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.integration

ROWS = 500
SEEDS = 2


@pytest.fixture(scope="module")
def ablation(pipeline_project: Settings) -> CausalAblation:
    return run_ablation(settings=pipeline_project, n_seeds=SEEDS, rows=ROWS)


def test_every_catalogued_scenario_is_measured_once(ablation: CausalAblation) -> None:
    assert [s.key for s in ablation.scenarios] == [spec.key for spec in CATALOGUE]
    assert ablation.rows == ROWS
    assert ablation.window == "mes_3_crise"
    assert ablation.seeds == [45, 46]
    assert ablation.model_version == "baseline-1.0.0"


def test_the_control_is_the_reference_for_the_paired_deltas(
    ablation: CausalAblation,
) -> None:
    control = ablation.scenarios[0]
    assert (control.delta_auc, control.delta_auc_std) == (0.0, 0.0)


def test_the_crisis_has_a_higher_default_rate_than_the_control(
    ablation: CausalAblation,
) -> None:
    rates = {s.key: s.base_rate for s in ablation.scenarios}
    assert rates["crise"] > rates["controle"]
    assert rates["so_covariate"] > rates["controle"]
    assert rates["crise_sem_aperto"] > rates["crise"]


def test_the_report_survives_a_json_round_trip(ablation: CausalAblation) -> None:
    restored = CausalAblation.model_validate_json(ablation.model_dump_json())
    assert restored == ablation


def test_a_missing_baseline_points_to_train(project_tmp: Settings) -> None:
    with pytest.raises(MissingInputError, match="train"):
        run_ablation(settings=project_tmp, n_seeds=1, rows=ROWS)
