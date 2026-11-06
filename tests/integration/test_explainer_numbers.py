"""Os números da página da Etapa 1 batem com os artefatos versionados."""

from __future__ import annotations

import pytest

from credit_sentinel.explainers.payloads import (
    Etapa1Payload,
    Etapa2Payload,
    build_payload,
)
from credit_sentinel.explainers.payloads_etapa3 import Etapa3Payload
from credit_sentinel.explainers.payloads_etapa4 import Etapa4Payload
from credit_sentinel.settings import Settings
from credit_sentinel.utils.io import read_json

pytestmark = pytest.mark.integration

SETTINGS = Settings(_env_file=None)


@pytest.fixture(scope="module")
def payload() -> Etapa1Payload:
    built = build_payload(1, settings=SETTINGS)
    assert isinstance(built, Etapa1Payload)
    return built


def test_validation_numbers_match_reports(payload: Etapa1Payload) -> None:
    folder = SETTINGS.reports_dir / "validation"
    bad = read_json(folder / "bad_batch_report.json")
    reference = read_json(folder / "credit_ref_report.json")
    assert payload.bad_batch.rejected_records == bad["rejected_records"]
    assert payload.bad_batch.decision == bad["decision"]
    assert [v.n_failures for v in payload.bad_batch.violations] == [
        v["n_failures"] for v in bad["violations"]
    ]
    assert payload.reference_report.valid_records == reference["valid_records"]


def test_model_numbers_match_metrics(payload: Etapa1Payload) -> None:
    metrics = read_json(SETTINGS.models_dir / "baseline" / "metrics.json")
    threshold = read_json(SETTINGS.models_dir / "baseline" / "threshold.json")
    assert payload.metrics.roc_auc == metrics["roc_auc"]
    assert payload.metrics.brier == metrics["brier"]
    assert payload.thresholds.cost_threshold == threshold["cost_threshold"]


def test_dataset_summary_matches_reference(payload: Etapa1Payload) -> None:
    assert payload.dataset.rows == 12_000
    assert payload.dataset.base_rate == pytest.approx(0.0645, abs=1e-4)


@pytest.fixture(scope="module")
def etapa2() -> Etapa2Payload:
    built = build_payload(2, settings=SETTINGS)
    assert isinstance(built, Etapa2Payload)
    return built


def test_etapa2_report_is_the_versioned_summary(etapa2: Etapa2Payload) -> None:
    summary = read_json(SETTINGS.reports_dir / "evidently_drift_summary.json")
    assert list(etapa2.report.windows) == list(summary["windows"])
    for name, window in summary["windows"].items():
        mine = etapa2.report.windows[name]
        assert mine.drift.n_drifted_features == window["drift"]["n_drifted_features"]
        assert mine.performance.roc_auc == window["performance"]["roc_auc"]
        assert mine.predictions.p90 == window["predictions"]["p90"]
        assert mine.evidently.flags == window["evidently"]["flags"]


def test_etapa2_window_cards_match_the_parquets(etapa2: Etapa2Payload) -> None:
    cards = {card.key: card for card in etapa2.windows}
    assert list(cards) == ["reference", "mes_1", "mes_2", "mes_3_crise"]
    assert cards["reference"].rows == 12_000
    assert {cards[k].rows for k in ("mes_1", "mes_2", "mes_3_crise")} == {5000}
    assert cards["mes_3_crise"].base_rate == pytest.approx(0.1296, abs=1e-4)


def test_etapa2_histograms_cover_every_dataset(etapa2: Etapa2Payload) -> None:
    for hist in etapa2.histograms.values():
        assert set(hist.counts) == {"reference", "mes_1", "mes_2", "mes_3_crise"}
        assert sum(hist.counts["reference"]) == 12_000
        assert sum(hist.counts["mes_2"]) == 5000


def test_etapa2_scenarios_come_from_the_yaml(etapa2: Etapa2Payload) -> None:
    rows = {row.parameter: row for row in etapa2.scenarios}
    assert rows["income_real_factor"].values["mes_3_crise"] == pytest.approx(0.80)
    assert rows["gamma_concept"].values["mes_3_crise"] == pytest.approx(1.5)
    assert rows["utilization_shift"].values["reference"] == 0


def test_etapa2_acceptance_is_derived_from_the_report(etapa2: Etapa2Payload) -> None:
    items = {item.criterion: item for item in etapa2.acceptance}
    assert sum(not item.ok for item in items.values()) == 1
    failing = next(item for item in items.values() if not item.ok)
    assert "renda_mensal" in failing.criterion
    assert "0,029" in failing.evidence


@pytest.fixture(scope="module")
def etapa3() -> Etapa3Payload:
    built = build_payload(3, settings=SETTINGS)
    assert isinstance(built, Etapa3Payload)
    return built


def test_etapa3_audit_and_gates_are_the_versioned_reports(
    etapa3: Etapa3Payload,
) -> None:
    audit = read_json(SETTINGS.reports_dir / "slo_audit_report.json")
    gates = read_json(SETTINGS.reports_dir / "gates_report.json")
    assert etapa3.audit.run_id == audit["run_id"]
    assert [w.window for w in etapa3.audit.windows] == [
        w["window"] for w in audit["windows"]
    ]
    for mine, theirs in zip(etapa3.audit.windows, audit["windows"], strict=True):
        assert mine.slo.remaining_budget == theirs["slo"]["remaining_budget"]
        assert [a.severity for a in mine.slo.alerts] == [
            a["severity"] for a in theirs["slo"]["alerts"]
        ]
    assert etapa3.gates.challenger.roc_auc == gates["challenger"]["roc_auc"]
    assert [g.passed for g in etapa3.gates.results] == [
        g["passed"] for g in gates["results"]
    ]


def test_etapa3_events_are_the_redacted_sample(etapa3: Etapa3Payload) -> None:
    lines = (
        (SETTINGS.docs_dir / "evidencias" / "etapa3" / "pipeline_events_sample.jsonl")
        .read_text("utf-8")
        .splitlines()
    )
    assert len(etapa3.events) == len(lines) == 40
    assert {e.run_id for e in etapa3.events} == {etapa3.audit.run_id}


def test_etapa3_slo_config_is_the_versioned_one(etapa3: Etapa3Payload) -> None:
    assert [s.id for s in etapa3.slo.slos] == [
        "SLO1_integridade",
        "SLO2_estabilidade_psi",
        "SLO3_consistencia_ks",
        "SLO4_performance",
        "SLO5_latencia",
    ]


def test_etapa3_acceptance_items_all_pass(etapa3: Etapa3Payload) -> None:
    failed = [i.criterion for i in etapa3.acceptance if not i.ok]
    assert failed == []
    assert len(etapa3.acceptance) >= 8


def test_etapa3_page_has_at_most_three_screenshots_under_the_limit(
    etapa3: Etapa3Payload,
) -> None:
    assert 1 <= len(etapa3.screenshots) <= 3
    assert all(s.size_bytes <= 400_000 for s in etapa3.screenshots)


@pytest.fixture(scope="module")
def etapa4() -> Etapa4Payload:
    built = build_payload(4, settings=SETTINGS)
    assert isinstance(built, Etapa4Payload)
    return built


def test_etapa4_fairness_and_causal_are_the_versioned_reports(
    etapa4: Etapa4Payload,
) -> None:
    audit = read_json(SETTINGS.reports_dir / "fairness_audit.json")
    ablation = read_json(SETTINGS.reports_dir / "causal_ablation.json")
    assert etapa4.fairness.passed is audit["passed"]
    assert len(etapa4.fairness.reports) == len(audit["reports"]) == 12
    first = etapa4.fairness.reports[0].groups[0]
    assert (
        first.disparate_impact == audit["reports"][0]["groups"][0]["disparate_impact"]
    )
    assert [s.key for s in etapa4.causal.scenarios] == [
        s["key"] for s in ablation["scenarios"]
    ]
    assert etapa4.causal.scenarios[1].delta_auc == ablation["scenarios"][1]["delta_auc"]


def test_etapa4_dag_evidence_matches_the_drift_report(etapa4: Etapa4Payload) -> None:
    summary = read_json(SETTINGS.reports_dir / "evidently_drift_summary.json")
    windows = summary["windows"]
    nodes = {node.id: node for node in etapa4.dag.nodes}
    crisis_psi = next(
        f["psi"]
        for f in windows["mes_3_crise"]["drift"]["numeric"]
        if f["feature"] == "taxa_endividamento"
    )
    debt = {e.label: e.value for e in nodes["DIV"].by_window["mes_3_crise"]}
    assert debt["PSI de taxa_endividamento"] == crisis_psi
    auc = {e.label: e.value for e in nodes["AUC"].by_window["mes_3_crise"]}
    assert (
        auc["ROC-AUC do champion"] == windows["mes_3_crise"]["performance"]["roc_auc"]
    )


def test_etapa4_false_approvals_in_the_band_match_the_readme(
    etapa4: Etapa4Payload,
) -> None:
    champion, challenger = etapa4.band
    assert (champion.role, challenger.role) == ("champion", "challenger")
    assert (champion.clients, champion.defaulters_approved) == (1062, 120)
    assert challenger.defaulters_approved == 104
    assert round(champion.expected_defaulters) == 44
    assert champion.defaulters_approved > challenger.defaulters_approved


def test_etapa4_star_script_is_the_versioned_roteiro(etapa4: Etapa4Payload) -> None:
    assert etapa4.star.total_seconds == 300
    assert [b.name.split(" ")[0] for b in etapa4.star.blocks][:2] == ["S", "T"]
    assert all(block.speech for block in etapa4.star.blocks)


def test_etapa4_acceptance_items_all_pass(etapa4: Etapa4Payload) -> None:
    failed = [i.criterion for i in etapa4.acceptance if not i.ok]
    assert failed == []
    assert len(etapa4.acceptance) >= 8
