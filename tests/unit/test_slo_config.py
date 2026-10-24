"""``slo_config.yaml``: carga, validação e coerência com o modelo e o runbook."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from credit_sentinel.observability.slo_monitor import SloConfig, load_slo_config
from credit_sentinel.settings import Settings
from credit_sentinel.utils.io import read_json, read_yaml

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit

SETTINGS = Settings(_env_file=None)
CONFIG_PATH = SETTINGS.config_dir / "slo_config.yaml"
SLO_IDS = [
    "SLO1_integridade",
    "SLO2_estabilidade_psi",
    "SLO3_consistencia_ks",
    "SLO4_performance",
    "SLO5_latencia",
]


@pytest.fixture(scope="module")
def raw() -> dict[str, Any]:
    return read_yaml(CONFIG_PATH)


def test_versioned_yaml_loads_with_the_five_slos_in_order() -> None:
    config = load_slo_config(CONFIG_PATH)
    assert [s.id for s in config.slos] == SLO_IDS
    assert config.window_batches == 4
    assert (config.warn_at, config.breach_at) == (0.70, 1.0)


def test_objectives_match_the_spec() -> None:
    config = load_slo_config(CONFIG_PATH)
    objectives = {s.id: (s.objective.op, s.objective.value) for s in config.slos}
    assert objectives == {
        "SLO1_integridade": ("<=", 0.01),
        "SLO2_estabilidade_psi": ("<=", 0.25),
        "SLO3_consistencia_ks": ("<=", 0.34),
        "SLO4_performance": (">=", 0.75),
        "SLO5_latencia": ("<=", 60000),
    }


def test_slo2_features_are_inside_the_top3_of_the_model() -> None:
    top3 = read_json(SETTINGS.models_dir / "baseline" / "feature_importance.json")[
        "top3"
    ]
    slo2 = load_slo_config(CONFIG_PATH).slo("SLO2_estabilidade_psi")
    assert slo2.features
    assert set(slo2.features) <= set(top3)


def test_slo3_watches_the_five_main_numeric_features() -> None:
    slo3 = load_slo_config(CONFIG_PATH).slo("SLO3_consistencia_ks")
    assert slo3.features == [
        "renda_mensal",
        "score_bureau",
        "taxa_endividamento",
        "utilizacao_credito",
        "comprometimento_renda",
    ]


def test_there_is_one_alert_rule_per_severity_in_priority_order() -> None:
    config = load_slo_config(CONFIG_PATH)
    assert [a.severity for a in config.alerts] == ["P1", "P2", "P3", "P4"]
    assert all(a.action and a.runbook_anchor for a in config.alerts)


def test_unknown_slo_id_raises_a_clear_error() -> None:
    with pytest.raises(KeyError, match="SLO9"):
        load_slo_config(CONFIG_PATH).slo("SLO9")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["slos"].append(copy.deepcopy(d["slos"][0])), "duplicado"),
        (lambda d: d["alerts"][1].update(slos=["SLO9"]), "SLO9"),
        (lambda d: d["alerts"][2].pop("value"), "value"),
        (lambda d: d["slos"][0]["objective"].update(value=0), "maior que zero"),
        (lambda d: d["slos"][0]["objective"].update(op="=="), "op"),
        (lambda d: d["slos"][1].update(features=[]), "features"),
        (lambda d: d.update(warn_at=1.2), "warn_at"),
    ],
)
def test_invalid_configs_are_rejected(
    raw: dict[str, Any], mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    broken = copy.deepcopy(raw)
    mutate(broken)
    with pytest.raises((ValidationError, ValueError), match=message):
        SloConfig.model_validate(broken)
