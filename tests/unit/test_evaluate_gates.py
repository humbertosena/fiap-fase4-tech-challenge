"""Guardrails de promoção do challenger e auditoria mínima de fairness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.drift.predictions import decision_threshold
from credit_sentinel.governance.fairness_audit import (
    AuditSubject,
    GroupImpact,
    audit_subject,
    to_group_impacts,
)
from credit_sentinel.ml.evaluate_gates import (
    GATE_IDS,
    GatesReport,
    evaluate_promotion_gates,
    gate_auc,
    gate_brier,
    gate_fairness,
    gate_input_contract,
    gate_latency,
)
from credit_sentinel.ml.train_baseline import predict_proba, train_and_analyze

if TYPE_CHECKING:
    import pandas as pd

    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit


def _impact(attribute: str, group: str, ratio: float) -> GroupImpact:
    return GroupImpact(
        attribute=attribute,
        group=group,
        n=100,
        approval_rate=0.8 * ratio,
        reference_group="ref",
        ratio=ratio,
    )


@pytest.mark.parametrize(
    ("champion", "challenger", "passed"),
    [
        (0.760, 0.780, True),
        (0.760, 0.7801, True),
        (0.760, 0.7799, False),
        (0.760, 0.760, False),
        (0.800, 0.790, False),
    ],
)
def test_auc_gate_needs_two_points_over_the_champion(
    champion: float, challenger: float, *, passed: bool
) -> None:
    result = gate_auc(champion, challenger)
    assert result.gate_id == "G1_auc_vs_champion"
    assert result.passed is passed
    assert result.threshold == pytest.approx(champion + 0.02)
    assert result.observed == challenger


@pytest.mark.parametrize(
    ("champion", "challenger", "passed"),
    [(0.10, 0.09, True), (0.10, 0.10, True), (0.10, 0.1001, False)],
)
def test_brier_gate_accepts_equal_or_better_calibration(
    champion: float, challenger: float, *, passed: bool
) -> None:
    assert gate_brier(champion, challenger).passed is passed


def test_input_contract_gate_compares_names_and_types() -> None:
    base = {
        "inputs": [
            {"name": "idade", "type": "long"},
            {"name": "renda", "type": "double"},
        ]
    }
    same = {"inputs": [dict(i) for i in base["inputs"]]}
    assert gate_input_contract(base, same).passed
    renamed = {
        "inputs": [
            {"name": "idade", "type": "long"},
            {"name": "salario", "type": "double"},
        ]
    }
    retyped = {
        "inputs": [
            {"name": "idade", "type": "double"},
            {"name": "renda", "type": "double"},
        ]
    }
    fewer = {"inputs": base["inputs"][:1]}
    for other in (renamed, retyped, fewer):
        result = gate_input_contract(base, other)
        assert not result.passed
        assert result.observed >= 1


@pytest.mark.parametrize(
    ("ratio", "passed"),
    [(0.80, True), (1.25, True), (1.0, True), (0.79, False), (1.26, False)],
)
def test_fairness_gate_band_is_inclusive(ratio: float, *, passed: bool) -> None:
    impacts = [_impact("genero", "F", ratio), _impact("genero", "M", 1.0)]
    result = gate_fairness(impacts)
    assert result.passed is passed
    assert (result.threshold, result.threshold_upper) == (0.80, 1.25)


def test_fairness_gate_reports_the_worst_group() -> None:
    impacts = [_impact("regiao", "Norte", 0.9), _impact("regiao", "Sul", 0.7)]
    result = gate_fairness(impacts)
    assert result.observed == pytest.approx(0.7)
    assert "Sul" in result.detail


@pytest.mark.parametrize(
    ("seconds", "passed"), [(0.2, True), (5.0, True), (5.1, False)]
)
def test_latency_gate(seconds: float, *, passed: bool) -> None:
    result = gate_latency(seconds)
    assert result.passed is passed
    assert result.threshold == 5.0


def test_report_passes_only_when_every_gate_passes(
    df_reference_small: pd.DataFrame, settings_tmp: Settings
) -> None:
    model = train_and_analyze(df_reference_small, settings=settings_tmp)
    report = evaluate_promotion_gates(
        model, model, df_reference_small, settings=settings_tmp
    )
    assert isinstance(report, GatesReport)
    assert [r.gate_id for r in report.results] == list(GATE_IDS)
    by_id = {r.gate_id: r for r in report.results}
    assert not by_id["G1_auc_vs_champion"].passed
    assert by_id["G2_brier"].passed
    assert by_id["G3_contrato_de_entrada"].passed
    assert by_id["G5_latencia"].passed
    assert report.passed is False
    assert report.champion.roc_auc == report.challenger.roc_auc
    assert report.holdout_rows == len(df_reference_small)


def test_g4_is_computed_by_the_full_fairness_audit(
    df_reference_small: pd.DataFrame, settings_tmp: Settings
) -> None:
    model = train_and_analyze(df_reference_small, settings=settings_tmp)
    report = evaluate_promotion_gates(
        model, model, df_reference_small, settings=settings_tmp
    )
    subject = AuditSubject(
        model_version=model.version,
        role="challenger",
        dataset="holdout",
        threshold=decision_threshold(model),
    )
    scores = predict_proba(model, df_reference_small)
    audited = audit_subject(df_reference_small, scores, subject)
    assert report.fairness == to_group_impacts(audited)
    by_id = {r.gate_id: r for r in report.results}
    assert by_id["G4_fairness_aprovacao"].passed is all(r.passed for r in audited)
