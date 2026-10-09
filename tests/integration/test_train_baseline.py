"""Testes de integração do treino do modelo baseline."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from sklearn.metrics import roc_auc_score

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import ScenarioParams, generate_portfolio
from credit_sentinel.ml.features import build_features
from credit_sentinel.ml.train_baseline import (
    MODEL_VERSION,
    ModelBundle,
    load_model_bundle,
    predict_proba,
    save_model_bundle,
    split_reference,
    train_baseline,
)
from credit_sentinel.settings import Settings
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def portfolio(reference_params: ScenarioParams) -> pd.DataFrame:
    period = (date(2025, 7, 1), date(2026, 6, 30))
    return generate_portfolio(2000, reference_params, make_rng(5), period)


@pytest.fixture(scope="module")
def trained(portfolio: pd.DataFrame) -> ModelBundle:
    pipeline, metrics = train_baseline(portfolio, settings=Settings(_env_file=None))
    return ModelBundle(pipeline=pipeline, metrics=metrics)


def test_baseline_learns_signal_on_small_sample(trained: ModelBundle) -> None:
    assert trained.metrics.roc_auc > 0.7
    assert trained.metrics.n_train == 1600
    assert trained.metrics.n_valid == 400
    assert trained.metrics.model_version == MODEL_VERSION


def test_sensitive_attributes_never_reach_the_model(trained: ModelBundle) -> None:
    seen = set(trained.pipeline.named_steps["features"].feature_names_in_)
    assert seen.isdisjoint(dd.SENSITIVE_COLUMNS)


def test_saved_bundle_reproduces_validation_metrics(
    trained: ModelBundle, portfolio: pd.DataFrame, tmp_path: Path
) -> None:
    save_model_bundle(trained, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} >= {
        "model.joblib",
        "metrics.json",
        "signature.json",
    }
    loaded = load_model_bundle(tmp_path)
    _, valid = split_reference(portfolio, seed=42)
    auc = roc_auc_score(valid[dd.TARGET], predict_proba(loaded, valid))
    assert auc == pytest.approx(trained.metrics.roc_auc, abs=1e-6)


def test_signature_lists_fourteen_inputs_and_double_output(
    trained: ModelBundle, tmp_path: Path
) -> None:
    save_model_bundle(trained, tmp_path)
    signature = load_model_bundle(tmp_path).signature
    assert len(signature["inputs"]) == 14
    assert signature["outputs"] == [{"type": "double"}]


def test_predict_proba_accepts_raw_batches(
    trained: ModelBundle, portfolio: pd.DataFrame
) -> None:
    scores = predict_proba(trained, portfolio.head(10))
    assert scores.shape == (10,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert build_features(portfolio.head(10)).shape == (10, 14)
