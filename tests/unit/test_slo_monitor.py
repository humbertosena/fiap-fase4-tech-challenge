"""Error budget, estados dos SLOs e classificação de alertas P1 a P4."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from credit_sentinel.contracts.validator import ValidationReport
from credit_sentinel.drift.predictions import PerformanceSummary
from credit_sentinel.drift.statistical_tests import (
    DriftSummary,
    NumericDriftResult,
)
from credit_sentinel.observability.slo_monitor import (
    SloEvaluation,
    SloObservation,
    SloResult,
    SloSpec,
    classify_alerts,
    consumption,
    evaluate_slos,
    load_slo_config,
    slo_status,
)
from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings(_env_file=None)
CONFIG = load_slo_config(SETTINGS.config_dir / "slo_config.yaml")
REFERENCE_AUC = 0.86
PSI_FEATURES = ["score_bureau", "qtd_parcelas_atraso_12m", "utilizacao_credito"]
KS_FEATURES = [
    "renda_mensal",
    "score_bureau",
    "taxa_endividamento",
    "utilizacao_credito",
    "comprometimento_renda",
]


def _validation(
    name: str = "mes_1", *, ratio: float = 0.0, blocked: bool = False
) -> ValidationReport:
    total = 5000
    rejected = round(total * ratio)
    return ValidationReport(
        batch_name=name,
        run_id="val-1",
        validated_at=datetime(2026, 10, 24, tzinfo=UTC),
        schema_name="credit_scoring_inference",
        total_records=total,
        valid_records=total - rejected,
        rejected_records=rejected,
        rejected_ratio=ratio,
        decision="BLOCKED" if blocked else "SUCCESS",
        blocked_reason="violação estrutural (R01, R02): lote bloqueado"
        if blocked
        else None,
        violations=[],
        sample_failures=[],
    )


def _numeric(feature: str, *, psi: float, ks_fail: bool) -> NumericDriftResult:
    return NumericDriftResult(
        feature=feature,
        ks_statistic=0.1,
        ks_p_value=0.001 if ks_fail else 0.5,
        ks_drift=ks_fail,
        ks_drift_bonferroni=ks_fail,
        psi=psi,
        psi_level="critico" if psi >= 0.25 else "estavel",
        wasserstein_norm=0.1,
        mean_ref=1.0,
        mean_cur=1.1,
    )


def _drift(
    psi: dict[str, float] | None = None, failing: set[str] | None = None
) -> DriftSummary:
    psi, failing = psi or {}, failing or set()
    features = dict.fromkeys([*PSI_FEATURES, *KS_FEATURES], 0.01) | psi
    return DriftSummary(
        window="mes_x",
        reference_rows=12000,
        current_rows=5000,
        alpha=0.05,
        alpha_bonferroni=0.05 / 12,
        numeric=[
            _numeric(name, psi=value, ks_fail=name in failing)
            for name, value in features.items()
        ],
        categorical=[],
        n_drifted_features=len(failing),
        drifted_features=sorted(failing),
        psi_critical_features=[],
    )


def _performance(auc: float) -> PerformanceSummary:
    return PerformanceSummary(
        window="mes_x",
        roc_auc=auc,
        f1=0.3,
        precision=0.3,
        recall=0.5,
        brier=0.05,
        base_rate=0.06,
        auc_delta_vs_reference=auc - REFERENCE_AUC,
    )


def _observation(**changes: object) -> SloObservation:
    base = {
        "validation": _validation(),
        "drift": _drift(),
        "performance": _performance(0.83),
        "latency_ms": 1200,
        "reference_auc": REFERENCE_AUC,
        "run_id": "run-1",
    }
    return SloObservation(**(base | changes))


def _result(slo_id: str, value: float) -> SloResult:
    return SloResult(
        slo_id=slo_id,
        sli="rejected_ratio",
        observed=value,
        threshold=1.0,
        op="<=",
        consumption=value,
        status=slo_status(value, CONFIG),
    )


def _evaluation(
    consumptions: dict[str, float], *, blocked: bool = False
) -> SloEvaluation:
    results = [_result(slo, value) for slo, value in consumptions.items()]
    return SloEvaluation(
        window="mes_x",
        run_id="run-1",
        results=results,
        alerts=[],
        remaining_budget=0.0,
        decision="BLOCKED" if blocked else "SUCCESS",
        blocked_reason="circuito aberto" if blocked else None,
    )


@pytest.mark.parametrize(
    ("op", "observed", "threshold", "expected"),
    [
        ("<=", 0.005, 0.01, 0.5),
        ("<=", 0.0, 0.01, 0.0),
        ("<=", 0.01, 0.01, 1.0),
        ("<=", 0.27, 0.01, 27.0),
        (">=", 0.86, 0.75, 0.0),
        (">=", 0.75, 0.75, 1.0),
        (">=", 0.805, 0.75, 0.5),
        (">=", 0.90, 0.75, 0.0),
        (">=", 0.70, 0.75, pytest.approx(1.4545454)),
    ],
)
def test_consumption_for_both_directions(
    op: str, observed: float, threshold: float, expected: float
) -> None:
    result = consumption(op, observed, threshold, reference=REFERENCE_AUC)  # type: ignore[arg-type]
    assert result == pytest.approx(expected)


def test_ge_objective_needs_a_reference_above_the_limit() -> None:
    with pytest.raises(ValueError, match="referência"):
        consumption(">=", 0.8, 0.75, reference=0.75)
    with pytest.raises(ValueError, match="referência"):
        consumption(">=", 0.8, 0.75, reference=None)


@pytest.mark.parametrize(
    ("value", "status"),
    [
        (0.0, "OK"),
        (0.6999, "OK"),
        (0.70, "WARN"),
        (0.9999, "WARN"),
        (1.0, "BREACHED"),
        (27.0, "BREACHED"),
    ],
)
def test_status_boundaries_at_070_and_100(value: float, status: str) -> None:
    assert slo_status(value, CONFIG) == status


def test_healthy_window_is_all_ok_with_only_the_informative_alert() -> None:
    evaluation = evaluate_slos(CONFIG, _observation())
    assert [r.status for r in evaluation.results] == ["OK"] * 5
    assert [a.severity for a in evaluation.alerts] == ["P4"]
    assert evaluation.decision == "SUCCESS"
    assert 0.5 < evaluation.remaining_budget <= 1.0
    assert evaluation.window == "mes_1"
    assert evaluation.run_id == "run-1"


def test_crisis_window_breaches_psi_and_ks_and_overspends_the_budget() -> None:
    observation = _observation(
        validation=_validation("mes_3_crise"),
        drift=_drift(
            psi={"utilizacao_credito": 0.66},
            failing={"renda_mensal", "taxa_endividamento", "utilizacao_credito"},
        ),
        performance=_performance(0.764),
    )
    evaluation = evaluate_slos(CONFIG, observation)
    status = {r.slo_id: r.status for r in evaluation.results}
    assert status["SLO2_estabilidade_psi"] == "BREACHED"
    assert status["SLO3_consistencia_ks"] == "BREACHED"
    assert status["SLO4_performance"] == "WARN"
    severities = {a.slo_id: a.severity for a in evaluation.alerts}
    assert severities["SLO2_estabilidade_psi"] == "P2"
    assert severities["SLO3_consistencia_ks"] == "P3"
    assert severities["SLO4_performance"] == "P3"
    psi = next(r for r in evaluation.results if r.slo_id == "SLO2_estabilidade_psi")
    assert psi.observed == pytest.approx(0.66)
    assert psi.consumption == pytest.approx(0.66 / 0.25)
    assert evaluation.remaining_budget < 0


def test_psi_exactly_at_the_critical_limit_is_a_p2() -> None:
    observation = _observation(drift=_drift(psi={"score_bureau": 0.25}))
    alerts = evaluate_slos(CONFIG, observation).alerts
    assert [(a.severity, a.slo_id) for a in alerts] == [("P2", "SLO2_estabilidade_psi")]


def test_one_failing_ks_feature_in_five_is_tolerated() -> None:
    observation = _observation(drift=_drift(failing={"renda_mensal"}))
    evaluation = evaluate_slos(CONFIG, observation)
    ks = next(r for r in evaluation.results if r.slo_id == "SLO3_consistencia_ks")
    assert ks.observed == pytest.approx(0.2)
    assert ks.consumption == pytest.approx(0.2 / 0.34)
    assert ks.status == "OK"
    assert [a.severity for a in evaluation.alerts] == ["P4"]


def test_two_failing_ks_features_in_five_breach_the_slo() -> None:
    observation = _observation(drift=_drift(failing={"renda_mensal", "score_bureau"}))
    evaluation = evaluate_slos(CONFIG, observation)
    ks = next(r for r in evaluation.results if r.slo_id == "SLO3_consistencia_ks")
    assert (ks.observed, ks.status) == (pytest.approx(0.4), "BREACHED")
    assert [(a.severity, a.slo_id) for a in evaluation.alerts] == [
        ("P3", "SLO3_consistencia_ks")
    ]


def test_blocked_window_only_measures_what_exists_and_raises_a_single_p1() -> None:
    observation = _observation(
        validation=_validation("bad_batch", ratio=0.27, blocked=True),
        drift=None,
        performance=None,
        latency_ms=800,
    )
    evaluation = evaluate_slos(CONFIG, observation)
    assert [r.slo_id for r in evaluation.results] == [
        "SLO1_integridade",
        "SLO5_latencia",
    ]
    assert evaluation.decision == "BLOCKED"
    assert evaluation.remaining_budget < -10
    (alert,) = evaluation.alerts
    assert alert.severity == "P1"
    assert alert.slo_id is None
    assert alert.window == "bad_batch"
    assert "violação estrutural" in alert.message
    assert alert.runbook_anchor == "p1-circuit-breaker"


def test_missing_performance_skips_slo4_without_counting_it_as_zero() -> None:
    evaluation = evaluate_slos(CONFIG, _observation(performance=None))
    ids = [r.slo_id for r in evaluation.results]
    assert "SLO4_performance" not in ids
    assert len(ids) == 4


def test_auc_consumption_uses_the_reference_auc() -> None:
    evaluation = evaluate_slos(CONFIG, _observation(performance=_performance(0.805)))
    auc = next(r for r in evaluation.results if r.slo_id == "SLO4_performance")
    assert auc.consumption == pytest.approx((0.86 - 0.805) / (0.86 - 0.75))
    assert auc.op == ">="
    assert auc.reference == REFERENCE_AUC


def test_slow_batch_breaches_the_latency_slo() -> None:
    evaluation = evaluate_slos(CONFIG, _observation(latency_ms=90000))
    latency = next(r for r in evaluation.results if r.slo_id == "SLO5_latencia")
    assert (latency.status, latency.consumption) == ("BREACHED", 1.5)
    assert [a.severity for a in evaluation.alerts] == ["P3"]


def test_unknown_feature_in_a_slo_is_an_error() -> None:
    spec = SloSpec(
        id="SLO2_estabilidade_psi",
        sli="psi_top_features",
        objective={"op": "<=", "value": 0.25},
        features=["feature_que_nao_existe"],
    )
    config = CONFIG.model_copy(update={"slos": [spec]})
    with pytest.raises(ValueError, match="feature_que_nao_existe"):
        evaluate_slos(config, _observation())


def test_classify_alerts_does_not_duplicate_a_breach_as_p2_and_p3() -> None:
    evaluation = _evaluation(
        {"SLO2_estabilidade_psi": 2.64, "SLO3_consistencia_ks": 2.3}
    )
    alerts = classify_alerts(evaluation, CONFIG)
    assert [(a.severity, a.slo_id) for a in alerts] == [
        ("P2", "SLO2_estabilidade_psi"),
        ("P3", "SLO3_consistencia_ks"),
    ]


def test_p2_recommends_retraining_and_every_alert_has_action_and_anchor() -> None:
    evaluation = _evaluation({"SLO4_performance": 1.2, "SLO1_integridade": 0.8})
    alerts = classify_alerts(evaluation, CONFIG)
    p2 = alerts[0]
    assert p2.severity == "P2"
    assert "retreino" in p2.recommended_action.lower()
    assert all(a.recommended_action and a.runbook_anchor and a.window for a in alerts)


def test_alerts_are_sorted_from_the_most_severe() -> None:
    evaluation = _evaluation({"SLO3_consistencia_ks": 1.5, "SLO4_performance": 1.1})
    severities = [a.severity for a in classify_alerts(evaluation, CONFIG)]
    assert severities == sorted(severities)


def test_every_runbook_anchor_exists_in_the_runbook() -> None:
    text = (SETTINGS.docs_dir / "runbook_incidentes.md").read_text(encoding="utf-8")
    for rule in CONFIG.alerts:
        assert f'id="{rule.runbook_anchor}"' in text


@pytest.mark.parametrize(
    "anchor", ["p1-circuit-breaker", "p2-drift-critico", "p3-atencao"]
)
def test_runbook_sections_follow_the_five_incident_phases(anchor: str) -> None:
    text = (SETTINGS.docs_dir / "runbook_incidentes.md").read_text(encoding="utf-8")
    start = text.index(f'id="{anchor}"')
    nxt = re.search(r'<a id="p\d-', text[start + 10 :])
    section = text[start : start + 10 + nxt.start()] if nxt else text[start:]
    phases = ["Detecção", "Triagem", "Mitigação", "Resolução", "Pós-mortem"]
    positions = [section.index(phase) for phase in phases]
    assert positions == sorted(positions)
