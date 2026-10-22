"""Os números da página da Etapa 1 batem com os artefatos versionados."""

from __future__ import annotations

import pytest

from credit_sentinel.explainers.payloads import (
    Etapa1Payload,
    Etapa2Payload,
    build_payload,
)
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
