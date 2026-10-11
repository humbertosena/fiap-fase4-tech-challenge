"""Testes de integração do treino do modelo baseline."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from sklearn.metrics import roc_auc_score

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import ScenarioParams, generate_portfolio
from credit_sentinel.ml.features import build_features
from credit_sentinel.ml.mlflow_baseline import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL,
    log_baseline_run,
)
from credit_sentinel.ml.train_baseline import (
    MODEL_VERSION,
    ModelBundle,
    load_model_bundle,
    predict_proba,
    save_model_bundle,
    split_reference,
    train_and_analyze,
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


@pytest.fixture(scope="module")
def analyzed(portfolio: pd.DataFrame) -> ModelBundle:
    return train_and_analyze(portfolio, settings=Settings(_env_file=None))


def test_full_metrics_and_thresholds(analyzed: ModelBundle) -> None:
    metrics = analyzed.metrics
    assert 0 < metrics.brier < 0.25
    assert 0 < metrics.ks_statistic <= 1
    assert sum(map(sum, metrics.confusion_matrix)) == metrics.n_valid
    assert analyzed.threshold is not None
    assert metrics.threshold == pytest.approx(analyzed.threshold.f1_threshold)
    assert analyzed.threshold.cost_threshold <= analyzed.threshold.f1_threshold


def test_importance_covers_every_feature_in_descending_order(
    analyzed: ModelBundle,
) -> None:
    means = [item.importance_mean for item in analyzed.importance]
    assert len(means) == 14
    assert means == sorted(means, reverse=True)


def test_bundle_round_trip_keeps_threshold_and_importance(
    analyzed: ModelBundle, tmp_path: Path
) -> None:
    save_model_bundle(analyzed, tmp_path)
    assert (tmp_path / "threshold.json").exists()
    assert (tmp_path / "feature_importance.json").exists()
    loaded = load_model_bundle(tmp_path)
    assert loaded.threshold == analyzed.threshold
    assert loaded.importance == analyzed.importance


def test_mlflow_run_logs_metrics_and_registers_model(
    analyzed: ModelBundle, portfolio: pd.DataFrame, tmp_path: Path
) -> None:
    import mlflow
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(_env_file=None, mlflow_tracking_uri=uri, project_root=tmp_path)
    run_id = log_baseline_run(analyzed, build_features(portfolio.head(5)), settings)
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.data.metrics["roc_auc"] == pytest.approx(analyzed.metrics.roc_auc)
    assert client.get_experiment(run.info.experiment_id).name == EXPERIMENT_NAME
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    assert [v.run_id for v in versions] == [run_id]
    assert str(tmp_path) in run.info.artifact_uri
    mlflow.set_tracking_uri("")
