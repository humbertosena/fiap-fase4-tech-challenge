"""Os números da página da Etapa 1 batem com os artefatos versionados."""

from __future__ import annotations

import pytest

from credit_sentinel.explainers.payloads import Etapa1Payload, build_payload
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
